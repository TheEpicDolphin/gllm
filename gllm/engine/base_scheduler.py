import asyncio
import uuid

from abc import ABC, abstractmethod
from concurrent.futures import Future as ConcurrentFuture
from dataclasses import dataclass, fields

import torch

from gllm.config.generator_config import GeneratorConfig
from gllm.engine.batch_inputs import BatchInputs
from gllm.engine.llm_engine_base import GenerationRequest, GenerationResult
from gllm.engine.logprobs import TokenLogProbs
from gllm.engine.utils import complete_future_threadsafe
from gllm.model.kv_cache.paged_kv_cache import PagedKVCache
from gllm.model.model import Model
from gllm.sample.sampling_metadata import SamplingMetadata


MAX_NUM_STOP_TOKENS = 4
MAX_NUM_TOP_LOGPROBS = 20


@dataclass
class RequestStates:
    # [B_max]
    prompt_lens: torch.Tensor
    # [B_max]
    max_new_tokens: torch.Tensor
    # [B_max, max_num_stop_tokens]
    stop_token_ids: torch.Tensor
    # [B_max, T_max]
    token_ids: torch.Tensor
    # [B_max]
    seq_lens: torch.Tensor
    # [B_max]
    num_blocks: torch.Tensor
    # [B_max, max_num_blocks]
    block_table: torch.Tensor
    # [B_max]
    temperature: torch.Tensor
    # [B_max]
    top_k: torch.Tensor
    # [B_max]
    top_p: torch.Tensor
    # [B_max]
    num_top_logprobs: torch.Tensor
    # [B_max, T_max, max_num_logprobs]
    top_logprobs: torch.Tensor
    # [B_max, T_max, max_num_logprobs]
    top_logprobs_token_ids: torch.Tensor


    def select(self, req_idxs):
        return RequestStates(
            prompt_lens=self.prompt_lens[req_idxs],
            max_new_tokens=self.max_new_tokens[req_idxs],
            stop_token_ids=self.stop_token_ids[req_idxs],
            token_ids=self.token_ids[req_idxs],
            seq_lens=self.seq_lens[req_idxs],
            num_blocks=self.num_blocks[req_idxs],
            block_table=self.block_table[req_idxs],
            temperature=self.temperature[req_idxs],
            top_k=self.top_k[req_idxs],
            top_p=self.top_p[req_idxs],
            num_top_logprobs=self.num_top_logprobs[req_idxs],
            top_logprobs=self.top_logprobs[req_idxs],
            top_logprobs_token_ids=self.top_logprobs_token_ids[req_idxs],
        )


@dataclass
class ScheduledRequest:
    id: int
    gen_req: GenerationRequest
    gen_logprobs: list[TokenLogProbs]
    future: asyncio.Future | ConcurrentFuture


class BaseScheduler(ABC):
    def __init__(
        self,
        model: Model,
        gen_config: GeneratorConfig,
        device: str,
    ):
        self.model = model
        self.gen_config = gen_config
        self.device = device

        self.paged_kv_cache = PagedKVCache(
            model_config=self.model.config,
            gen_config=gen_config,
            device=device,
        )
        
        self.max_batch_size = gen_config.max_batch_size
        self.max_seq_len = gen_config.max_seq_len
        max_num_blocks = self.paged_kv_cache.num_required_blocks(self.max_seq_len)

        self.batch_size: int = 0
        self.reqs: list[ScheduledRequest] = [None] * self.max_batch_size
        
        # [max(B_max, T_max)]
        self.arange = torch.arange(max(self.max_batch_size, self.max_seq_len), device=device)
        # [block_size]
        self.block_offsets = torch.arange(self.paged_kv_cache.block_size, device=device)
        
        self.req_states = RequestStates(
            prompt_lens=torch.empty(self.max_batch_size, dtype=torch.int32, device=device),
            max_new_tokens=torch.empty(self.max_batch_size, dtype=torch.int32, device=device),
            stop_token_ids=torch.empty((self.max_batch_size, MAX_NUM_STOP_TOKENS), dtype=torch.long, device=device),
            token_ids=torch.empty((self.max_batch_size, self.max_seq_len), dtype=torch.long, device=device),
            seq_lens=torch.empty(self.max_batch_size, dtype=torch.int32, device=device),
            num_blocks=torch.empty(self.max_batch_size, dtype=torch.int32, device=device),
            block_table=torch.zeros((self.max_batch_size, max_num_blocks), dtype=torch.int32, device=device),
            temperature=torch.empty(self.max_batch_size, dtype=torch.float32, device=device),
            top_k=torch.empty(self.max_batch_size, dtype=torch.int32, device=device),
            top_p=torch.empty(self.max_batch_size, dtype=torch.float32, device=device),
            num_top_logprobs=torch.empty(self.max_batch_size, dtype=torch.int32, device=device),
            top_logprobs=torch.empty((self.max_batch_size, self.max_seq_len, MAX_NUM_TOP_LOGPROBS), dtype=torch.float32, device=device),
            top_logprobs_token_ids=torch.empty((self.max_batch_size, self.max_seq_len, MAX_NUM_TOP_LOGPROBS), dtype=torch.long, device=device),
        )
        self.req_state_keys = [f.name for f in fields(self.req_states)]


    @property
    def num_active_requests(self) -> int:
        return self.batch_size
    

    def _gen_unique_id(self):
        return uuid.uuid4().int
    

    def prepare_prefill_batch(
        self,
        reqs: list[GenerationRequest],
        futures: list[asyncio.Future],
    ) -> tuple[BatchInputs | None, int]:
        rs = self.req_states
        prefill_start_idx = self.batch_size
        for req, future in zip(reqs, futures):
            try:
                # Combine user and model stop token ids.
                user_stop_tokens = "".join(req.stop_tokens)
                user_stop_token_ids = self.model.tokenize(user_stop_tokens)
                stop_token_ids = list(set(self.model.eos_token_ids + user_stop_token_ids))
                if len(stop_token_ids) > MAX_NUM_STOP_TOKENS:
                    raise ValueError(
                        f"Too many stop tokens in request: {len(stop_token_ids)} > {MAX_NUM_STOP_TOKENS}"
                    )
                stop_token_ids += [stop_token_ids[-1]] * max(MAX_NUM_STOP_TOKENS - len(stop_token_ids), 0)

                prompt_token_ids = self.model.tokenize(req.prompt)
                allowed_num_new_tokens = self.max_seq_len - len(prompt_token_ids)
                if allowed_num_new_tokens <= 0:
                    raise ValueError(
                        f"Prompt has {len(prompt_token_ids)} tokens, but the engine only supports {self.max_seq_len}."
                    )
                
                if req.num_top_logprobs > MAX_NUM_TOP_LOGPROBS:
                    raise ValueError(
                        f"num_top_logprobs must be <= {MAX_NUM_TOP_LOGPROBS}."
                    )
                
                if req.max_new_tokens <= 0:
                    raise ValueError(
                        f"max_new_tokens must be >= 0."
                    )
                    
                if self.batch_size == self.max_batch_size:
                    raise RuntimeError(
                        f"Failed to enqueue request with error: Batch size limit ({self.max_batch_size}) has been reached."
                    )

                uid = self._gen_unique_id()
                idx = self.batch_size
                self.reqs[idx] = ScheduledRequest(
                    id=uid,
                    gen_req=req,
                    gen_logprobs=[],
                    future=future,
                )
                print(f"[LLMEngine] received request '{uid}'.")

                # Process new request.
                prompt_token_ids_tensor = torch.tensor(prompt_token_ids, device=self.device)
                stop_token_ids_tensor = torch.tensor(stop_token_ids, device=self.device)
                prompt_len = len(prompt_token_ids)
                num_prompt_blocks = self.paged_kv_cache.num_required_blocks(prompt_len)
                prompt_block_ids = self.paged_kv_cache.reserve_blocks(num_prompt_blocks)
                prompt_block_ids_tensor = torch.tensor(prompt_block_ids, device=self.device)
                
                # Clear row with pad token ids.
                rs.token_ids[idx, :] = self.model.pad_token_id
                # Clear row with dummy block id.
                rs.block_table[idx, :] = 0

                # Set tensors.
                rs.prompt_lens[idx] = prompt_len
                rs.stop_token_ids[idx] = stop_token_ids_tensor
                rs.max_new_tokens[idx] = req.max_new_tokens
                rs.num_top_logprobs[idx] = req.num_top_logprobs
                rs.token_ids[idx, :prompt_len] = prompt_token_ids_tensor
                rs.seq_lens[idx] = prompt_len
                rs.block_table[idx][:num_prompt_blocks] = prompt_block_ids_tensor
                rs.num_blocks[idx] = num_prompt_blocks
                rs.temperature[idx] = req.temperature
                rs.top_k[idx] = req.top_k
                rs.top_p[idx] = req.top_p
                self.batch_size += 1
            except Exception as e:
                complete_future_threadsafe(
                    future,
                    exception=e
                )

        prefill_end_idx = self.batch_size
        B = prefill_end_idx - prefill_start_idx
        if B == 0:
            # No new requests were enqueued.
            return None, prefill_start_idx

        seq_lens = rs.seq_lens[prefill_start_idx:prefill_end_idx]
        max_seq_len = seq_lens.max()
        num_blocks = rs.num_blocks[prefill_start_idx:prefill_end_idx]
        block_table = rs.block_table[prefill_start_idx:prefill_end_idx, :num_blocks.max()]

        # Create KV cache slot mapping.
        # [B, max_num_blocks * block_size]
        slot_mapping = (self.paged_kv_cache.block_size * block_table.unsqueeze(2) + self.block_offsets).view(B, -1)
        # [B, T]
        slot_mapping = slot_mapping[:, :max_seq_len].contiguous()

        # Update sampling metadata.
        sampling_metadata = SamplingMetadata(
            temperature=rs.temperature[prefill_start_idx:prefill_end_idx],
            top_k=rs.top_k[prefill_start_idx:prefill_end_idx],
            top_p=rs.top_p[prefill_start_idx:prefill_end_idx],
            num_top_logprobs=rs.num_top_logprobs[prefill_start_idx:prefill_end_idx],
        )

        batch = BatchInputs(
            max_seq_len=max_seq_len,
            seq_lens=seq_lens,
            token_ids=rs.token_ids[prefill_start_idx:prefill_end_idx, :max_seq_len],
            token_positions=self.arange[:max_seq_len].unsqueeze(0).expand(B, -1),
            slot_mapping=slot_mapping,
            sampling_metadata=sampling_metadata,
            paged_kv_cache=self.paged_kv_cache,
        )
        return batch, prefill_start_idx
        

    def prepare_decode_batch(self) -> BatchInputs:
        rs = self.req_states
        B = self.batch_size
        for idx in range(B):
            # Process ongoing request.
            cur_num_blocks = rs.num_blocks[idx].item()
            num_required_blocks = self.paged_kv_cache.num_required_blocks(rs.seq_lens[idx].item())
            # Allocate new blocks to hold the current sequence, if needed.
            num_new_blocks = num_required_blocks - cur_num_blocks
            if num_new_blocks > 0:
                new_block_ids = self.paged_kv_cache.reserve_blocks(num_new_blocks)
                new_block_ids_tensor = torch.tensor(new_block_ids, device=self.device)
                rs.block_table[idx][cur_num_blocks:num_required_blocks] = new_block_ids_tensor
            rs.num_blocks[idx] = num_required_blocks

        seq_lens = rs.seq_lens[:B]
        max_seq_len = seq_lens.max()
        num_blocks = rs.num_blocks[:B]
        block_table = rs.block_table[:B, :num_blocks.max()]

        # Create KV cache slot mapping.
        # [B, max_num_blocks * block_size]
        slot_mapping = (self.paged_kv_cache.block_size * block_table.unsqueeze(2) + self.block_offsets).view(B, -1)
        # [B, T_q]
        slot_mapping = slot_mapping[:, :max_seq_len].contiguous()

        # Update sampling metadata.
        sampling_metadata = SamplingMetadata(
            temperature=rs.temperature[:B],
            top_k=rs.top_k[:B],
            top_p=rs.top_p[:B],
            num_top_logprobs=rs.num_top_logprobs[:B],
        )

        return BatchInputs(
            max_seq_len=max_seq_len,
            seq_lens=seq_lens,
            token_ids=rs.token_ids[:B, :max_seq_len],
            token_positions=self.arange[:max_seq_len].unsqueeze(0).expand(B, -1),
            slot_mapping=slot_mapping,
            sampling_metadata=sampling_metadata,
            paged_kv_cache=self.paged_kv_cache,
        )
    

    def _process_finished_requests(
        self,
        finished_reqs: list[ScheduledRequest],
        finished_req_states: RequestStates,
    ):
        token_ids = finished_req_states.token_ids
        prompt_lens = finished_req_states.prompt_lens
        seq_lens = finished_req_states.seq_lens
        block_table = finished_req_states.block_table
        num_blocks = finished_req_states.num_blocks
        top_logprobs = finished_req_states.top_logprobs
        top_logprobs_token_ids = finished_req_states.top_logprobs_token_ids
        for idx, req in enumerate(finished_reqs):
            num_top_logprobs = req.gen_req.num_top_logprobs
            prompt_len = prompt_lens[idx]
            seq_len = seq_lens[idx]
            gen_token_ids = token_ids[idx, prompt_len:seq_len].tolist()
            generated_text = self.model.detokenize(gen_token_ids)
            # [T, num_top_logprobs]
            gen_top_logprobs = top_logprobs[idx, prompt_len:seq_len, :num_top_logprobs].tolist()
            # [T, num_top_logprobs]
            gen_top_logprobs_token_ids = top_logprobs_token_ids[idx, prompt_len:seq_len, :num_top_logprobs].tolist()
            gen_top_logprobs = [TokenLogProbs(logprobs, token_ids) for logprobs, token_ids in zip(gen_top_logprobs, gen_top_logprobs_token_ids)]
            complete_future_threadsafe(
                req.future,
                GenerationResult(
                    text=generated_text,
                    token_ids=gen_token_ids,
                    top_logprobs=gen_top_logprobs,
                )
            )
            print(f"[LLMEngine] completed request '{req.id}'.")
            # Release KV cache blocks.
            block_ids = block_table[idx, :num_blocks[idx]].tolist()
            self.paged_kv_cache.release_blocks(block_ids)
    

    def update(
        self,
        # [B, T_q]
        sampled_token_ids: torch.Tensor,
        # [B, T_q, top_logprobs]
        sampled_top_logprobs: torch.Tensor,
        # [B, T_q, top_logprobs]
        sampled_top_logprobs_token_ids: torch.Tensor,
        req_offset: int = 0,
    ) -> None:
        """
        sampled_token_ids: [B, T_q] tensor of sampled token ids for the current decode step,
            for each request in the batch. May be padded with -1 to indicate tokens that should
            be ignored (e.g. for rejected token ids during speculative decoding).
        sampled_logprobs: list of length B, with per-token logprobs.
        req_offset: Offset into the request batch. This is needed during the prefill step, when
            new requests are added to the end of the batch.
        """
        B, T_q = sampled_token_ids.shape
        seq_lens = self.req_states.seq_lens[req_offset:self.batch_size]
        prompt_lens = self.req_states.prompt_lens[req_offset:self.batch_size]
        token_ids = self.req_states.token_ids[req_offset:self.batch_size]
        stop_token_ids = self.req_states.stop_token_ids[req_offset:self.batch_size]
        max_new_tokens = self.req_states.max_new_tokens[req_offset:self.batch_size]
        top_logprobs = self.req_states.top_logprobs[req_offset:self.batch_size]
        top_logprobs_token_ids = self.req_states.top_logprobs_token_ids[req_offset:self.batch_size]

        # [B, T_q]
        dummy_token_mask = sampled_token_ids == -1
        # Mark requests as finished if stop token is detected.
        # [B, T_q]
        stop_token_mask = (sampled_token_ids.unsqueeze(-1) == stop_token_ids.unsqueeze(1)).any(dim=-1)
        # [B]
        finished_mask = stop_token_mask.any(dim=-1)
        
        # Set sampled token ids before the first dummy or stop token, whichever
        # comes first.
        mask = torch.cumsum((dummy_token_mask | stop_token_mask).long(), dim=-1) == 0
        positions = seq_lens.long().unsqueeze(-1) + self.arange[:T_q]
        batch_idxs = self.arange[:B].unsqueeze(-1).expand_as(positions)
        masked_batch_idxs = batch_idxs[mask]
        masked_positions = positions[mask]
        token_ids[masked_batch_idxs, masked_positions] = sampled_token_ids[mask]
        seq_lens += mask.sum(dim=-1)
        
        # Get number of generated tokens per request.
        gen_lens = seq_lens - prompt_lens
        # Mark requests as finished if max generation length is met.
        finished_mask |= (gen_lens >= max_new_tokens)

        # Set sampled top logprobs.
        num_top_logprobs = sampled_top_logprobs.shape[-1]
        top_logprobs[masked_batch_idxs, masked_positions, :num_top_logprobs] = sampled_top_logprobs[mask]
        top_logprobs_token_ids[masked_batch_idxs, masked_positions, :num_top_logprobs] = sampled_top_logprobs_token_ids[mask]

        # Compact the batch by moving over right-most ongoing requests to fill
        # empty slots of left-most finished requests, in that order, until there
        # are no more ongoing requests to the right of finished request slots.
        #
        # Example:
        #   Before Shift
        #       idx:   0  1  2  3  4  5  6
        #       req:   A  X  C  X  E  X  G
        #   After Shift
        #       idx:   0  1  2  3  4  5  6
        #       req:   A  G  C  E  -  -  -
        #
        # NOTE: With this approach, requests are never moved to a slot that is
        # being copied from, so there's no risk of overwriting data before it
        # is copied.

        # Get indices of finished requests.
        # [1, 3, 5]
        finished_req_idxs = req_offset + finished_mask.nonzero().flatten()

        # Decrement batch size by number of finished requests.
        self.batch_size -= finished_req_idxs.shape[0]

        # Process the finished requests.
        self._process_finished_requests(
            [self.reqs[idx] for idx in finished_req_idxs],
            self.req_states.select(finished_req_idxs)
        )

        # Get indices of ongoing requests in reverse order.
        # [6, 4, 2, 0]
        ongoing_req_idxs_reversed = req_offset + torch.flip((~finished_mask).nonzero().flatten(), dims=[0])
        # Clamp tensors to the minimum size of the two.
        min_size = min(finished_req_idxs.shape[0], ongoing_req_idxs_reversed.shape[0])
        # [1, 3, 5]
        finished_req_idxs = finished_req_idxs[:min_size]
        # [6, 4, 2]
        ongoing_req_idxs_reversed = ongoing_req_idxs_reversed[:min_size]
        # Calculate the number of requests that need to be moved.
        # 2
        num_moves = (ongoing_req_idxs_reversed > finished_req_idxs).sum()
        # [1, 3]
        to_req_idxs = finished_req_idxs[:num_moves]
        # [6, 4]
        from_req_idxs = ongoing_req_idxs_reversed[:num_moves]
        # Iterate state tensors.
        states_dict = self.req_states.__dict__
        for key in self.req_state_keys:
            states_dict[key][to_req_idxs] = states_dict[key][from_req_idxs]
        # Iterate reqs list.
        for from_idx, to_idx in zip(from_req_idxs, to_req_idxs):
            self.reqs[to_idx] = self.reqs[from_idx]

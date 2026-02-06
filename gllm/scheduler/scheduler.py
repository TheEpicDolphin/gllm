import asyncio
from dataclasses import dataclass
import uuid

import torch

from gllm.config.generator_config import GeneratorConfig
from gllm.engine.llm_engine_base import GenerationRequest, GenerationResult, TokenLogProbs
from gllm.model.kv_cache.paged_kv_cache import PagedKVCache
from gllm.model.model import Model
from gllm.sample.logprobs import TokenLogProbs
from gllm.sample.sampling_metadata import SamplingMetadata
from gllm.scheduler.batch_inputs import BatchInputs


@dataclass
class ScheduledRequest:
    id: int
    prompt_token_ids: list[int]
    generated_token_ids: list[int]
    generated_logprobs: list[TokenLogProbs]
    max_new_tokens: int
    stop_token_ids: set[int]
    temperature: float
    top_k: int
    top_p: float
    max_num_logprobs: int
    future: asyncio.Future


class Scheduler:
    def __init__(
        self,
        model: Model,
        paged_kv_cache: PagedKVCache,
        gen_config: GeneratorConfig,
        device: str,
    ):
        super().__init__()
        self.model = model
        self.paged_kv_cache = paged_kv_cache
        self.gen_config = gen_config
        self.device = device
        
        self.max_batch_size = gen_config.max_batch_size
        self.max_seq_len = gen_config.max_seq_len
        max_num_blocks = paged_kv_cache.num_required_blocks(self.max_seq_len)

        self.batch_size: int = 0
        self.reqs: list[ScheduledRequest] = [None] * self.max_batch_size
        
        # [max(B_max, T_max)]
        self.arange = torch.arange(max(self.max_batch_size, self.max_seq_len), device=device)
        # [block_size]
        self.block_offsets = torch.arange(paged_kv_cache.block_size, device=device)
        
        # [B_max, T_max]
        self.token_ids = torch.empty(
            (self.max_batch_size, self.max_seq_len),
            dtype=torch.int64,
            device=device,
        )

        # Attention metadata buffers.
        # [B_max]
        self.seq_lens = torch.empty(
            self.max_batch_size,
            dtype=torch.int32,
            device=device,
        )
        # [B_max]
        self.num_blocks = torch.empty(
            self.max_batch_size,
            dtype=torch.int32,
            device=device,
        )
        # [B_max, max_num_blocks]
        self.block_table = torch.zeros(
            (self.max_batch_size, max_num_blocks),
            dtype=torch.int32,
            device=device,
        )
        
        # Sampling metadata buffers.
        # [B_max]
        self.temperature = torch.empty(
            self.max_batch_size,
            dtype=torch.float32,
            device=device,
        )
        # [B_max]
        self.top_k = torch.empty(
            self.max_batch_size,
            dtype=torch.int32,
            device=device,
        )
        # [B_max]
        self.top_p = torch.empty(
            self.max_batch_size,
            dtype=torch.float32,
            device=device,
        )
        # [B_max]
        self.max_num_logprobs = torch.empty(
            self.max_batch_size,
            dtype=torch.int32,
            device=device,
        )


    @property    
    def has_active_requests(self) -> bool:
        return self.batch_size > 0
    

    def _gen_unique_id(self):
        return uuid.uuid4().int
    
    
    def _create_result(
        self,
        req: ScheduledRequest,
    ):
        generated_text = self.model.detokenize(req.generated_token_ids)
        return GenerationResult(
            token_ids=req.generated_token_ids,
            logprobs=req.generated_logprobs,
            text=generated_text,
        )
    

    def prepare_prefill_batch(
        self,
        reqs: list[GenerationRequest],
        futures: list[asyncio.Future],
    ) -> tuple[BatchInputs, int]:
        prefill_start_idx = self.batch_size
        for req, future in zip(reqs, futures):
            try:
                # Combine user and model stop token ids.
                user_stop_tokens = "".join(req.stop_tokens)
                user_stop_token_ids = self.model.tokenize(user_stop_tokens)
                stop_token_ids = set(self.model.eos_token_ids + user_stop_token_ids)
                prompt_token_ids = self.model.tokenize(req.prompt)
                allowed_num_new_tokens = self.max_seq_len - len(prompt_token_ids)
                if allowed_num_new_tokens <= 0:
                    raise ValueError(
                        f"Prompt has {len(prompt_token_ids)} tokens, but the engine only supports {self.max_seq_len}."
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
                    prompt_token_ids=prompt_token_ids,
                    generated_token_ids=[],
                    generated_logprobs=[],
                    max_new_tokens=min(allowed_num_new_tokens, req.max_new_tokens),
                    stop_token_ids=stop_token_ids,
                    temperature=req.temperature,
                    top_k=req.top_k,
                    top_p=req.top_p,
                    max_num_logprobs=req.max_num_logprobs,
                    future=future,
                )
                self.batch_size += 1

                # Process new request.
                prompt_token_ids_tensor = torch.tensor(prompt_token_ids, device=self.device)
                prompt_len = len(prompt_token_ids)
                num_prompt_blocks = self.paged_kv_cache.num_required_blocks(prompt_len)
                prompt_block_ids = self.paged_kv_cache.reserve_blocks(num_prompt_blocks)
                prompt_block_ids_tensor = torch.tensor(prompt_block_ids, device=self.device)
                
                # Clear row with pad token ids.
                self.token_ids[idx, :] = self.model.pad_token_id
                # Clear row with dummy block id.
                self.block_table[idx, :] = 0

                # Set tensors.
                self.token_ids[idx, :prompt_len] = prompt_token_ids_tensor
                self.seq_lens[idx] = prompt_len
                self.block_table[idx][:num_prompt_blocks] = prompt_block_ids_tensor
                self.num_blocks[idx] = num_prompt_blocks
                self.temperature[idx] = req.temperature
                self.top_k[idx] = req.top_k
                self.top_p[idx] = req.top_p
                self.max_num_logprobs[idx] = req.max_num_logprobs
            except Exception as e:
                future.set_exception(e)

        prefill_end_idx = self.batch_size
        B = prefill_end_idx - prefill_start_idx
        seq_lens = self.seq_lens[prefill_start_idx:prefill_end_idx]
        max_seq_len = seq_lens.max()
        num_blocks = self.num_blocks[prefill_start_idx:prefill_end_idx]
        block_table = self.block_table[prefill_start_idx:prefill_end_idx, :num_blocks.max()]

        # Create KV cache slot mapping.
        # [B, max_num_blocks, block_size]
        slot_mapping = self.paged_kv_cache.block_size * block_table.unsqueeze(2) + self.block_offsets

        # Update sampling metadata.
        sampling_metadata = SamplingMetadata(
            temperature=self.temperature[prefill_start_idx:prefill_end_idx],
            top_k=self.top_k[prefill_start_idx:prefill_end_idx],
            top_p=self.top_p[prefill_start_idx:prefill_end_idx],
            max_num_logprobs=self.max_num_logprobs[prefill_start_idx:prefill_end_idx],
        )

        batch = BatchInputs(
            max_seq_len=max_seq_len,
            seq_lens=seq_lens,
            # During prefill, the entire sequence is the query.
            max_query_len=max_seq_len,
            query_lens=seq_lens,
            token_ids=self.token_ids[prefill_start_idx:prefill_end_idx],
            token_positions=self.arange[:max_seq_len].unsqueeze(0).expand(B, -1),
            slot_mapping=slot_mapping.view(B, -1),
            sampling_metadata=sampling_metadata,
        )
        return batch, prefill_start_idx
        

    def prepare_decode_batch(self) -> BatchInputs:
        B = self.batch_size
        for idx in range(B):
            # Process ongoing request.
            cur_num_blocks = self.num_blocks[idx].item()
            num_required_blocks = self.paged_kv_cache.num_required_blocks(self.seq_lens[idx].item())
            # Allocate new blocks to hold the current sequence, if needed.
            num_new_blocks = num_required_blocks - cur_num_blocks
            if num_new_blocks > 0:
                new_block_ids = self.paged_kv_cache.reserve_blocks(num_new_blocks)
                new_block_ids_tensor = torch.tensor(new_block_ids, device=self.device)
                self.block_table[idx][cur_num_blocks:num_required_blocks] = new_block_ids_tensor
            self.num_blocks[idx] = num_required_blocks

        seq_lens = self.seq_lens[:B]
        max_seq_len = seq_lens.max()
        num_blocks = self.num_blocks[:B]
        block_table = self.block_table[:B, :num_blocks.max()]

        # Create KV cache slot mapping.
        # [B, max_num_blocks, block_size]
        slot_mapping = self.paged_kv_cache.block_size * block_table.unsqueeze(2) + self.block_offsets

        # Update sampling metadata.
        sampling_metadata = SamplingMetadata(
            temperature=self.temperature[:B],
            top_k=self.top_k[:B],
            top_p=self.top_p[:B],
            max_num_logprobs=self.max_num_logprobs[:B],
        )
        
        return BatchInputs(
            max_seq_len=max_seq_len,
            seq_lens=seq_lens,
            # For now, assume query length is always 1 during decoding.
            # This will change when spec decoding is implemented, which
            # allows for variable length queries during decoding.
            max_query_len=1,
            query_lens=torch.ones_like(seq_lens),  
            token_ids=self.token_ids[:B],
            token_positions=self.arange[:max_seq_len].unsqueeze(0).expand(B, -1),
            slot_mapping=slot_mapping.view(B, -1),
            sampling_metadata=sampling_metadata,
        )
    

    def update(
        self,
        sampled_token_ids: torch.Tensor,
        logprobs: list[TokenLogProbs],
        req_offset: int = 0,
    ) -> None:
        B = self.batch_size
        # Update token ids and sequence lengths.
        batch_idxs = self.arange[req_offset:B]
        seq_lens = self.seq_lens[req_offset:B]
        self.token_ids[batch_idxs, seq_lens] = sampled_token_ids
        seq_lens += 1

        # Find and remove finished requests.
        to_req_idxs = []
        for idx in range(req_offset, B):
            req = self.reqs[idx]
            sampled_token_id = sampled_token_ids[idx - req_offset].item()
            req.generated_token_ids.append(sampled_token_id)
            req.generated_logprobs.append(logprobs[idx - req_offset])
            if sampled_token_id in req.stop_token_ids \
                or len(req.generated_token_ids) >= req.max_new_tokens:
                # Mark request as finished.
                self.reqs[idx] = None
                result = self._create_result(req)
                req.future.set_result(result)
                print(f"[LLMEngine] completed request '{req.id}'.")
                # Release KV cache blocks.
                block_ids = self.block_table[idx, :self.num_blocks[idx]].tolist()
                self.paged_kv_cache.release_blocks(block_ids)
                to_req_idxs.append(idx)
        self.batch_size -= len(to_req_idxs)

        # Shift remaining request states to fill empty slots.
        for from_idx, to_idx in zip(range(self.batch_size, B), to_req_idxs):
            if from_idx == to_idx:
                continue
            self.reqs[to_idx] = self.reqs[from_idx]
            self.token_ids[to_idx] = self.token_ids[from_idx]
            self.seq_lens[to_idx] = self.seq_lens[from_idx]
            self.num_blocks[to_idx] = self.num_blocks[from_idx]
            self.block_table[to_idx] = self.block_table[from_idx]
            self.temperature[to_idx] = self.temperature[from_idx]
            self.top_k[to_idx] = self.top_k[from_idx]
            self.top_p[to_idx] = self.top_p[from_idx]
            self.max_num_logprobs[to_idx] = self.max_num_logprobs[from_idx]

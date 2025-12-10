import asyncio
import uuid
from dataclasses import dataclass

import torch
from tokenizers import Tokenizer

from gllm.config.generator_params import GeneratorParams
from gllm.model.layers.attention import AttentionMetadata
from gllm.model.model import HuggingFaceModel, Model
from gllm.sample.logprobs import TokenLogProbs
from gllm.sample.sampler import Sampler
from gllm.sample.sampling_metadata import SamplingMetadata
from gllm.scheduler.input_batch import InputBatch
from gllm.scheduler.request_state import RequestState


class LLM:
    def __init__(
        self,
        hf_model: HuggingFaceModel,
        gen_params: GeneratorParams,
        device: str,
        local_cache_dir: str | None = None,
    ):
        super().__init__()
        
        self.model = Model(
            hf_model=hf_model,
            gen_params=gen_params,
            device=device,
            local_cache_dir=local_cache_dir,
        )
        self.sampler = Sampler(
            max_batch_size=gen_params.max_batch_size,
            device=device,
        )
        self.gen_params = gen_params
        self.device = device
        
        max_batch_size = gen_params.max_batch_size
        max_seq_len = gen_params.max_seq_len
        max_num_blocks=self.model.paged_kv_cache.num_required_blocks(max_seq_len)
        
        # [T_max]
        self.arange = torch.arange(max_seq_len, device=device)
        # [block_size]
        self.block_offsets = torch.arange(self.gen_params.block_size, device=self.device)
        
        # [B_max, T_max]
        self.token_ids = torch.empty(
            (max_batch_size, max_seq_len),
            dtype=torch.int64,
            device=device,
        )

        # Attention metadata buffers.
        # [B_max]
        self.query_lens = torch.empty(
            max_batch_size,
            dtype=torch.int32,
            device=device,
        )
        # [B_max]
        self.seq_lens = torch.empty(
            max_batch_size,
            dtype=torch.int32,
            device=device,
        )
        # [B_max]
        self.num_blocks = torch.empty(
            max_batch_size,
            dtype=torch.int32,
            device=device,
        )
        # [B_max, max_num_blocks]
        self.block_table = torch.zeros(
            (max_batch_size, max_num_blocks),
            dtype=torch.int32,
            device=device,
        )
        
        # Sampling metadata buffers.
        # [B_max]
        self.temperature = torch.empty(
            max_batch_size,
            dtype=torch.float32,
            device=device,
        )
        # [B_max]
        self.top_k = torch.empty(
            max_batch_size,
            dtype=torch.int32,
            device=device,
        )
        # [B_max]
        self.top_p = torch.empty(
            max_batch_size,
            dtype=torch.float32,
            device=device,
        )
        # [B_max]
        self.max_num_logprobs = torch.empty(
            max_batch_size,
            dtype=torch.int32,
            device=device,
        )
        
        self.req_idx_map: dict[int, int] = {}
        
    
    @property
    def eos_token_ids(self) -> list[int]:
        return self.model.eos_token_ids
    
    
    @property
    def max_batch_size(self) -> int:
        return self.gen_params.max_batch_size
    
    
    @property
    def max_seq_len(self) -> int:
        return self.gen_params.max_seq_len
        
        
    def _split_request_indices(
        self,
        reqs: list[RequestState]
    ) -> tuple[list[int], list[int], list[tuple[int, int]]]:
        ongoing_req_idxs = []
        new_req_idxs = []
        req_idx_map = self.req_idx_map.copy()
        for idx, req in enumerate(reqs):
            if req.id in self.req_idx_map:
                ongoing_req_idxs.append(idx)
                req_idx_map.pop(req.id)
            else:
                new_req_idxs.append(idx)
        finished_req_idxs = list(req_idx_map.items())
        return ongoing_req_idxs, new_req_idxs, finished_req_idxs
        

    def prepare_batch(
        self,
        reqs: list[RequestState],
    ) -> InputBatch:
        token_ids = self.token_ids
        query_lens = self.query_lens
        seq_lens = self.seq_lens
        block_table = self.block_table
        num_blocks = self.num_blocks
        temperature = self.temperature
        top_k = self.top_k
        top_p = self.top_p
        max_num_logprobs = self.max_num_logprobs
        paged_kv_cache = self.model.paged_kv_cache
        block_size = self.gen_params.block_size
        
        ongoing_req_idxs, new_req_idxs, finished_req_idxs = self._split_request_indices(reqs)
            
        # Cleanup finished requests.
        for id, idx in finished_req_idxs:
            num_alloc_blocks = num_blocks[idx]
            block_ids = block_table[idx][:num_alloc_blocks].tolist()
            paged_kv_cache.release_blocks(block_ids)
            self.req_idx_map.pop(id)
        
        # Update existing requests.
        for idx in ongoing_req_idxs:
            req = reqs[idx]
            prev_idx = self.req_idx_map[req.id]
            # Process ongoing request.
            seq_len = len(req.prompt_token_ids) + len(req.generated_token_ids)
            cur_num_blocks = num_blocks[idx].item()
            num_required_blocks = paged_kv_cache.num_required_blocks(seq_len)
            
            # Update tensors.
            prev_idx = self.req_idx_map[req.id]
            token_ids[idx][seq_len - 1] = req.generated_token_ids[-1]
            query_lens[idx] = 1
            seq_lens[idx] = seq_len
            num_blocks[idx] = num_required_blocks
            if prev_idx != idx:
                token_ids[idx][:seq_len] = token_ids[prev_idx][:seq_len]
                block_table[idx][:cur_num_blocks] = block_table[prev_idx][:cur_num_blocks]
                temperature[idx] = temperature[prev_idx]
                top_k[idx] = top_k[prev_idx]
                top_p[idx] = top_p[prev_idx]
                max_num_logprobs[idx] = max_num_logprobs[prev_idx]
            
            # Allocate new blocks to hold the current sequence, if needed.
            num_new_blocks = num_required_blocks - cur_num_blocks
            if num_new_blocks > 0:
                new_block_ids = paged_kv_cache.reserve_blocks(num_new_blocks)
                new_block_ids_tensor = torch.tensor(new_block_ids, device=self.device)
                block_table[idx][cur_num_blocks:num_required_blocks] = new_block_ids_tensor

            # Update id => idx mapping.
            self.req_idx_map[req.id] = idx
            
        # Process new requests.
        for idx in new_req_idxs:
            req = reqs[idx]
            # Process new request.
            prompt_token_ids = req.prompt_token_ids
            prompt_token_ids_tensor = torch.tensor(prompt_token_ids, device=self.device)
            prompt_len = len(prompt_token_ids)
            num_prompt_blocks = paged_kv_cache.num_required_blocks(prompt_len)
            prompt_block_ids = paged_kv_cache.reserve_blocks(num_prompt_blocks)
            prompt_block_ids_tensor = torch.tensor(prompt_block_ids, device=self.device)
            
            # Clear row with pad token ids.
            token_ids[idx, :] = self.model.pad_token_id
            # Clear row with dummy block id.
            block_table[idx, :] = 0

            # Set tensors.
            token_ids[idx, :prompt_len] = prompt_token_ids_tensor
            query_lens[idx] = prompt_len
            seq_lens[idx] = prompt_len
            block_table[idx][:num_prompt_blocks] = prompt_block_ids_tensor
            num_blocks[idx] = num_prompt_blocks
            temperature[idx] = req.temperature
            top_k[idx] = req.top_k
            top_p[idx] = req.top_p
            max_num_logprobs[idx] = req.max_num_logprobs
            
            # Update id => idx mapping.
            self.req_idx_map[req.id] = idx

        bs = len(reqs)
        query_lens = query_lens[:bs]
        max_query_len = query_lens.max()
        seq_lens = seq_lens[:bs]
        max_seq_len = seq_lens.max()
        max_num_blocks = num_blocks[:bs].max()
        block_table = block_table[:bs, :max_num_blocks]
        context_lens = seq_lens - query_lens
        
        # Create KV cache slot mapping.
        # [B, max_num_blocks, block_size]
        slot_mapping = block_size * block_table.unsqueeze(2) + self.block_offsets
        # [B, max_num_blocks * block_size]
        slot_mapping = slot_mapping.view(bs, -1)
        
        # Apply padding to slot mapping and context. Padding is needed
        # to align variable length sequences during attention.
        max_context_len = context_lens.max()
        padded_slot_mapping = torch.zeros(
            (bs, max_context_len + max_query_len),
            dtype=slot_mapping.dtype,
            device=self.device,
        )
        context_bias = torch.zeros(
            (bs, max_query_len, max_context_len),
            dtype=self.model.dtype,
            device=self.device,
        )
        for i in range(bs):
            padding = max_context_len - context_lens[i]
            seq_len = seq_lens[i]
            padded_slot_mapping[i, padding:padding + seq_len] = slot_mapping[i, :seq_len]
            context_bias[i, :, :padding] = float("-inf")
        
        # Create query slot mapping.
        query_slot_mapping = padded_slot_mapping[:, -max_query_len:].contiguous()
        
        # Create causal attention bias for query.
        # [T_q, T_q]
        query_bias = torch.full(
            (max_query_len, max_query_len),
            float("-inf"),
            dtype=self.model.dtype,
            device=self.device,
        )
        query_bias.triu_(diagonal=1)
        # [B, T_q, T_q]
        query_bias = query_bias.expand(bs, -1, -1).contiguous()
        
        # Update query token ids for next decode step.
         # [B, T_q]
        query_token_positions = context_lens.unsqueeze(1) + self.arange[:max_query_len]
        query_token_ids = token_ids.gather(dim=-1, index=query_token_positions)
        
        # Update attention metadata.
        attention_metadata = AttentionMetadata(
            query_lens=query_lens,
            seq_lens=seq_lens,
            max_seq_len=max_seq_len,
            block_table=block_table,
            slot_mapping=padded_slot_mapping,
            query_slot_mapping=query_slot_mapping,
            context_bias=context_bias,
            query_bias=query_bias,
        )

        # Update sampling metadata.
        sampling_metadata = SamplingMetadata(
            temperature=temperature[:bs],
            top_k=top_k[:bs],
            top_p=top_p[:bs],
            max_num_logprobs=max_num_logprobs[:bs],
        )
        
        return InputBatch(
            query_token_ids=query_token_ids,
            positions=query_token_positions,
            attention_metadata=attention_metadata,
            sampling_metadata=sampling_metadata,
        )
        

    def decode_step(
        self,
        input_batch: InputBatch
    ) -> tuple[list[int], list[TokenLogProbs]]:
        # [B, T_q]
        input_token_ids = input_batch.query_token_ids
        positions = input_batch.positions
        attn_metadata = input_batch.attention_metadata
        # [B, T_q, hidden_size]
        token_embeddings = self.model.get_token_embeddings(input_token_ids)
        assert not torch.isnan(token_embeddings).any()
        # [B, T_q, hidden_size]
        output_hidden_states = self.model.forward(
            token_embeddings,
            positions,
            attn_metadata
        )
        assert not torch.isnan(output_hidden_states).any()
        # [B, T_q, vocab_size]
        logits = self.model.compute_logits(output_hidden_states)
        assert not torch.isnan(logits).any()
        query_lens = attn_metadata.query_lens
        # [B, vocab_size]
        final_logits = logits[self.arange[:query_lens.size(0)], query_lens - 1]
        # [B], [B]
        sampled_token_ids, logprobs = self.sampler.forward(
            final_logits,
            input_batch.sampling_metadata
        )
        return sampled_token_ids.tolist(), logprobs
        
        
    def tokenize(self, tokens: str) -> list[int]:
        return self.model.tokenizer.encode(tokens).ids


    def detokenize(self, token_ids: list[int]) -> str:
        return self.model.tokenizer.decode(token_ids)

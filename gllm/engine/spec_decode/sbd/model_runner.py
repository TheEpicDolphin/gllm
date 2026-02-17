import torch
from torch.profiler import record_function

from gllm.engine.config import EngineConfig
from gllm.engine.base_model_runner import BaseModelRunner
from gllm.engine.batch_inputs import BatchInputs
from gllm.engine.decode_output import DecodeOutput
from gllm.engine.prefill_output import PrefillOutput
from gllm.sample.spec_decode.sbd.sampler import Sampler
from gllm.model.kv_cache.paged_kv_cache import PagedKVCache
from gllm.model.layers.attention import AttentionMetadata
from gllm.model.model import Model


class ModelRunner(BaseModelRunner):
    def __init__(
        self,
        model: Model,
        engine_config: EngineConfig,
        device: str,
    ):
        super().__init__()
        self.sbd_block_size = engine_config.sbd_config.block_size
        self.model = model
        self.sampler = Sampler(
            mask_token_id=model.mask_token_id
        )
        # [max(B_max, T_max)]
        self.arange = torch.arange(max(engine_config.max_batch_size, engine_config.max_seq_len), device=device)

    
    def prefill_step(
        self,
        batch: BatchInputs,
    ) -> PrefillOutput:
        B = batch.seq_lens.shape[0]
        max_seq_len = batch.max_seq_len
        seq_lens = batch.seq_lens
        batch_idxs = self.arange[:B]
        seq_idxs = self.arange[:max_seq_len]
        # [B, T]
        token_positions = seq_idxs.unsqueeze(0).expand(B, -1)
        
        # Create causal attention bias.
        # [B, T, T]
        bias = torch.full(
            (B, max_seq_len, max_seq_len),
            fill_value=float("-inf"),
            dtype=self.model.dtype,
            device=self.model.device,
        )
        bias.triu_(diagonal=1)

        # Create attention metadata.
        attention_metadata = AttentionMetadata(
            positions=token_positions,
            seq_lens=seq_lens,
            query_lens=seq_lens,
            context_slot_mapping=None,
            query_slot_mapping=batch.slot_mapping,
            bias=bias,
        )

        with record_function("model.forward"):
            # [B, T, hidden_size]
            logits = self.model.forward(
                batch.token_ids,
                attention_metadata,
                batch.paged_kv_cache,
            )
        assert not torch.isnan(logits).any()
        
        with record_function("sample"):
            sampler_output = self.sampler.forward(
                # [B, 1, vocab_size]
                logits[batch_idxs, seq_lens - 1].unsqueeze(1),
                batch.sampling_metadata
            )
        return PrefillOutput(*sampler_output)


    def decode_step(
        self,
        batch: BatchInputs,
        paged_kv_cache: PagedKVCache,
    ) -> DecodeOutput:
        B = batch.seq_lens.shape[0]
        max_seq_len = batch.max_seq_len
        seq_lens = batch.seq_lens
        query_len = self.sbd_config.block_size
        context_lens = seq_lens - query_len
        max_context_len = context_lens.max()
        context_padding = max_context_len - context_lens
        # [B, T_q]
        query_token_positions = context_lens.unsqueeze(1) + self.arange[:query_len]
        # [B, T_q]
        query_token_ids = batch.token_ids.gather(dim=-1, index=query_token_positions)
        
        # SBD requires attention bias of all zeros. The query tokens
        # have bidirectional attention with other query tokens and
        # causal attention with context tokens.
        # [B, T_q, T]
        bias = torch.zeros(
            (B, query_len, max_seq_len),
            dtype=self.model.dtype,
            device=self.model.device,
        )
        # Apply padding to slot mapping and attention bias. Padding is
        # needed to align variable length sequences during attention.
        padded_slot_mapping = torch.zeros(
            (B, max_seq_len),
            dtype=batch.slot_mapping.dtype,
            device=batch.slot_mapping.device,
        )
        for i in range(B):
            padded_slot_mapping[i, context_padding[i]:] = batch.slot_mapping[i, :seq_lens[i]]
            bias[i, :, :context_padding[i]] = float("-inf")
        
        # Create query slot mapping.
        context_slot_mapping = padded_slot_mapping[:, :-query_len].contiguous()
        query_slot_mapping = padded_slot_mapping[:, -query_len:].contiguous()

        # Create attention metadata.
        attention_metadata = AttentionMetadata(
            positions=query_token_positions,
            query_lens=torch.full_like(context_lens, query_len),
            seq_lens=seq_lens,
            context_slot_mapping=context_slot_mapping,
            query_slot_mapping=query_slot_mapping,
            bias=bias,
        )

        with record_function("model.forward"):
            # [B, T_q, vocab_size]
            logits = self.model.forward(
                query_token_ids,
                attention_metadata,
                paged_kv_cache,
            )
        assert not torch.isnan(logits).any()
        
        with record_function("sample"):
            sampler_output = self.sampler.forward(
                # [B, T_q, vocab_size]
                logits[:, -query_len:],
                batch.sampling_metadata
            )

        return DecodeOutput(
            sampler_output.sampled_token_ids,
            None,
            None
        )

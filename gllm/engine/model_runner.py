import torch
from torch.profiler import record_function

from gllm.engine.config import EngineConfig
from gllm.engine.base_model_runner import BaseModelRunner
from gllm.engine.batch_inputs import BatchInputs
from gllm.engine.decode_output import DecodeOutput
from gllm.engine.prefill_output import PrefillOutput
from gllm.model.layers.attention import AttentionMetadata
from gllm.model.model import Model
from gllm.sample.sampler import Sampler


class ModelRunner(BaseModelRunner):
    def __init__(
        self,
        model: Model,
        engine_config: EngineConfig,
        device: str,
    ):
        super().__init__()
        self.model = model
        self.sampler = Sampler(
            max_batch_size=engine_config.max_batch_size,
            device=device,
        )
        self.engine_config = engine_config
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
    ) -> DecodeOutput:
        B = batch.seq_lens.shape[0]
        max_seq_len = batch.max_seq_len
        seq_lens = batch.seq_lens
        context_lens = seq_lens - 1
        max_context_len = context_lens.max()
        context_padding = max_context_len - context_lens
        # [B, 1]
        query_token_positions = context_lens.unsqueeze(1)
        # [B, 1]
        query_token_ids = batch.token_ids.gather(dim=-1, index=query_token_positions)
        
        # Initialize attention bias to all zeros.
        # [B, 1, T]
        bias = torch.zeros(
            (B, 1, max_seq_len),
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
        context_slot_mapping = padded_slot_mapping[:, :-1].contiguous()
        query_slot_mapping = padded_slot_mapping[:, -1:].contiguous()

        # Create attention metadata.
        attention_metadata = AttentionMetadata(
            positions=query_token_positions,
            seq_lens=seq_lens,
            query_lens=torch.ones_like(seq_lens),
            context_slot_mapping=context_slot_mapping,
            query_slot_mapping=query_slot_mapping,
            bias=bias,
        )

        with record_function("model.forward"):
            # [B, 1, hidden_size]
            logits = self.model.forward(
                query_token_ids,
                attention_metadata,
                batch.paged_kv_cache,
            )
        assert not torch.isnan(logits).any()
        
        with record_function("sample"):
            sampler_output = self.sampler.forward(
                logits,
                batch.sampling_metadata
            )
        return DecodeOutput(*sampler_output)

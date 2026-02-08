import torch
from torch.profiler import record_function

from gllm.config.generator_config import GeneratorConfig
from gllm.engine.base_model_runner import BaseModelRunner
from gllm.engine.batch_inputs import BatchInputs
from gllm.model.layers.attention import AttentionMetadata
from gllm.model.model import Model
from gllm.sample.logprobs import TokenLogProbs
from gllm.sample.sampler import Sampler


class ModelRunner(BaseModelRunner):
    def __init__(
        self,
        model_path: str,
        gen_config: GeneratorConfig,
        device: str,
    ):
        super().__init__()
        
        self.model = Model(
            model_path=model_path,
            max_seq_len=gen_config.max_seq_len,
            device=device,
            dtype=gen_config.model_dtype,
            kv_dtype=gen_config.kv_dtype,
            cpu_offloading=gen_config.cpu_offloading,
        )
        self.sampler = Sampler(
            max_batch_size=gen_config.max_batch_size,
            device=device,
        )
        self.gen_config = gen_config
        self.device = device
        # [max(B_max, T_max)]
        self.arange = torch.arange(max(gen_config.max_batch_size, gen_config.max_seq_len), device=device)
    

    def prefill_step(
        self,
        batch: BatchInputs,
    ) -> tuple[torch.Tensor, list[TokenLogProbs]]:
        B = batch.seq_lens.shape[0]
        seq_lens = batch.seq_lens
        max_seq_len = batch.max_seq_len
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
            device=self.device,
        )
        bias.triu_(diagonal=1)

        # Create attention metadata.
        attention_metadata = AttentionMetadata(
            positions=token_positions,
            seq_lens=seq_lens,
            query_lens=seq_lens,
            slot_mapping=batch.slot_mapping,
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
            # [B], [B]
            sampled_token_ids, logprobs = self.sampler.forward(
                # [B, vocab_size]
                logits[batch_idxs, seq_lens - 1],
                batch.sampling_metadata
            )
        return sampled_token_ids, logprobs
    

    def decode_step(
        self,
        batch: BatchInputs,
    ) -> tuple[torch.Tensor, list[TokenLogProbs]]:
        B = batch.seq_lens.shape[0]
        seq_lens = batch.seq_lens
        max_seq_len = batch.max_seq_len
        context_padding = max_seq_len - seq_lens
        # [B, 1]
        query_token_positions = (seq_lens - 1).unsqueeze(1)
        # [B, 1]
        query_token_ids = batch.token_ids.gather(dim=-1, index=query_token_positions)
        
        # Initialize attention bias to all zeros.
        # [B, 1, T]
        bias = torch.zeros(
            (B, 1, max_seq_len),
            dtype=self.model.dtype,
            device=self.device,
        )

        # Apply padding to slot mapping and attention bias. Padding is
        # needed to align variable length sequences during attention.
        padded_slot_mapping = torch.zeros(
            (B, max_seq_len),
            dtype=batch.slot_mapping.dtype,
            device=self.device,
        )
        for i in range(B):
            padded_slot_mapping[i, context_padding[i]:] = batch.slot_mapping[i, :seq_lens[i]]
            bias[i, :, :context_padding[i]] = float("-inf")
        
        # Create query slot mapping.
        query_slot_mapping = padded_slot_mapping[:, -1:].contiguous()

        # Create attention metadata.
        attention_metadata = AttentionMetadata(
            positions=query_token_positions,
            query_lens=torch.ones_like(seq_lens),
            seq_lens=seq_lens,
            slot_mapping=padded_slot_mapping,
            query_slot_mapping=query_slot_mapping,
            bias=bias,
        )

        with record_function("model.forward"):
            # [B, T_q, hidden_size]
            logits = self.model.forward(
                query_token_ids,
                attention_metadata,
                batch.paged_kv_cache,
            )
        assert not torch.isnan(logits).any()
        
        with record_function("sample"):
            # [B], [B]
            sampled_token_ids, logprobs = self.sampler.forward(
                # [B, vocab_size]
                logits[:, -1],
                batch.sampling_metadata
            )
        return sampled_token_ids, logprobs

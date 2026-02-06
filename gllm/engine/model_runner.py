import torch
from torch.profiler import record_function

from gllm.config.generator_config import GeneratorConfig
from gllm.model.kv_cache.paged_kv_cache import PagedKVCache
from gllm.model.layers.attention import AttentionMetadata
from gllm.model.model import Model
from gllm.sample.logprobs import TokenLogProbs
from gllm.sample.sampler import Sampler
from gllm.scheduler.batch_inputs import BatchInputs


class ModelRunner:
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
        # [T_max]
        self.arange = torch.arange(gen_config.max_seq_len, device=device)
        

    def step(
        self,
        batch: BatchInputs,
        paged_kv_cache: PagedKVCache,
    ) -> tuple[torch.Tensor, list[TokenLogProbs]]:
        B = batch.seq_lens.shape[0]
        max_query_len = batch.max_query_len
        seq_lens = batch.seq_lens
        query_lens = batch.query_lens
        context_lens = seq_lens - query_lens
        max_context_len = context_lens.max()

        # Update query token positions and ids for next decode step.
        # [B, T_q]
        query_token_positions = context_lens.unsqueeze(1) + self.arange[:max_query_len]
        
        # Initialize attention bias to all zeros.
        # [B, T_q, T]
        bias = torch.zeros(
            (B, max_query_len, max_context_len + max_query_len),
            dtype=self.model.dtype,
            device=self.device,
        )
        # Apply padding to slot mapping and attention bias. Padding is
        # needed to align variable length sequences during attention.
        padded_slot_mapping = torch.zeros(
            (B, max_context_len + max_query_len),
            dtype=batch.slot_mapping.dtype,
            device=self.device,
        )
        for i in range(B):
            padding = max_context_len - context_lens[i]
            seq_len = seq_lens[i]
            padded_slot_mapping[i, padding:padding + seq_len] = batch.slot_mapping[i, :seq_len]
            bias[i, :, :padding] = float("-inf")
        # Set causal attention bias for query.
        # [B, T_q, T_q]
        query_bias = bias[:, :, max_context_len:]
        query_bias.fill_(float("-inf"))
        query_bias.triu_(diagonal=1)
        
        # Create query slot mapping.
        query_slot_mapping = padded_slot_mapping[:, -max_query_len:].contiguous()

        # Create attention metadata.
        attention_metadata = AttentionMetadata(
            positions=query_token_positions,
            query_lens=query_lens,
            seq_lens=seq_lens,
            slot_mapping=padded_slot_mapping,
            query_slot_mapping=query_slot_mapping,
            bias=bias,
        )

        # [B, T_q]
        query_token_ids = batch.token_ids.gather(dim=-1, index=query_token_positions)
        with record_function("model.forward"):
            # [B, T_q, hidden_size]
            logits = self.model.forward(
                query_token_ids,
                attention_metadata,
                paged_kv_cache,
            )
        assert not torch.isnan(logits).any()
        
        # [B, vocab_size]
        final_logits = logits[self.arange[:query_lens.size(0)], query_lens - 1]
        
        with record_function("sample"):
            # [B], [B]
            sampled_token_ids, logprobs = self.sampler.forward(
                final_logits,
                batch.sampling_metadata
            )
        return sampled_token_ids, logprobs

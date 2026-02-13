from dataclasses import dataclass

import torch

from gllm.model.kv_cache.paged_kv_cache import PagedKVCache
from gllm.sample.sampling_metadata import SamplingMetadata


@dataclass
class BatchInputs:
    max_seq_len: int
    # [B]
    seq_lens: torch.Tensor
    # [B, T]
    token_ids: torch.Tensor
    # [B, T]
    slot_mapping: torch.Tensor
    # Sampling metadata.
    sampling_metadata: SamplingMetadata
    # KV cache.
    paged_kv_cache: PagedKVCache
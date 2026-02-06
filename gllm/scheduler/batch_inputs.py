from dataclasses import dataclass

import torch

from gllm.sample.sampling_metadata import SamplingMetadata


@dataclass
class BatchInputs:
    max_seq_len: int
    max_query_len: int
    # [B]
    seq_lens: torch.Tensor
    # [B]
    query_lens: torch.Tensor
    # [B, T_max]
    token_ids: torch.Tensor
    # [B, T_max]
    token_positions: torch.Tensor
    # [B, max_num_blocks * block_size]
    slot_mapping: torch.Tensor
    # Sampling metadata.
    sampling_metadata: SamplingMetadata
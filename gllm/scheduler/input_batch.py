import asyncio
from dataclasses import dataclass
from typing import Callable

import torch

from gllm.model.layers.attention import AttentionMetadata
from gllm.sample.sampling_metadata import SamplingMetadata


@dataclass
class InputBatch:
    # [B, T_q]
    query_token_ids: torch.Tensor
    # [B, T_q]
    positions: torch.Tensor
    # Attention metadata & buffers.
    attention_metadata: AttentionMetadata
    # Sampling metadata & buffers.
    sampling_metadata: SamplingMetadata

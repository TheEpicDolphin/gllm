from dataclasses import dataclass
from enum import StrEnum

import torch


class ActivationFunction(StrEnum):
    SILU = "silu"
    RELU = "relu"
    GELU = "gelu"
    SWIGLU = "swiglu"


@dataclass
class ModelConfig:
    dtype: torch.dtype
    hidden_size: int
    head_dim: int
    intermediate_size: int
    act_func: ActivationFunction
    num_layers: int
    num_attn_heads: int
    num_kv_heads: int
    rms_norm_eps: float
    eos_token_ids: list[int]
    kv_dtype: torch.dtype
    rope_theta: float
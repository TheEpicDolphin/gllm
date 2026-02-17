from dataclasses import dataclass

import torch


@dataclass
class SetBlockDecoderConfig:
    block_size: int


@dataclass
class EngineConfig:
    block_size: int
    max_batch_size: int
    max_seq_len: int
    max_queue_size: int = 0
    model_dtype: torch.dtype | None = None
    kv_dtype: torch.dtype | None = None
    # Enables offloading model weights to a specific device (e.g. CPU).
    offload_device: str | None = None
    spec_decode_config: SetBlockDecoderConfig | None = None
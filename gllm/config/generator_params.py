from dataclasses import dataclass


@dataclass
class GeneratorParams:
    block_size: int
    max_batch_size: int
    max_seq_len: int
    model_dtype: str | None = None
    kv_dtype: str | None = None
    # Enables CPU offloading of model weights.
    cpu_offloading: bool = False
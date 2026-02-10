from dataclasses import dataclass


@dataclass
class GeneratorConfig:
    block_size: int
    max_batch_size: int
    max_seq_len: int
    max_queue_size: int = 0
    model_dtype: str | None = None
    kv_dtype: str | None = None
    # Enables CPU offloading of model weights.
    cpu_offloading: bool = False
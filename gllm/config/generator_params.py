from dataclasses import dataclass


@dataclass
class GeneratorParams:
    block_size: int
    max_batch_size: int
    max_chunked_prefill: int
    max_seq_len: int
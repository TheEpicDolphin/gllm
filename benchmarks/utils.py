import argparse
import time
import torch
from dataclasses import dataclass
from typing import Type

from gllm.config.generator_params import GeneratorParams
from gllm.engine.llm_engine_base import GenerationRequest, LLMEngineBase


@dataclass
class BenchResult:
    batch_size: int
    context_len: int
    gen_len: int
    duration_ms: float
    peak_mem_mb: float


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def now():
    return time.perf_counter()
    
    
def make_requests(batch_size, context_len, gen_len):
    prompt = "Hello " * context_len
    return [
        GenerationRequest(
            prompt=prompt,
            max_new_tokens=gen_len,
            stop_tokens=[],
            temperature=1.0,
            top_k=50,
            top_p=0.95,
        )
        for _ in range(batch_size)
    ]


def measure_engine_generation(
    engine: LLMEngineBase,
    requests: list[GenerationRequest],
):
    torch.cuda.reset_peak_memory_stats()
    sync()

    t0 = now()
    results = engine.generate(requests)
    sync()
    t1 = now()

    peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
    return (t1 - t0) * 1000, peak_mem, results


def benchmark(
    engine_cls: Type,
    model: str,
    batch_size: int,
    context_len: int,
    gen_len: int,
    device="cuda",
) -> BenchResult:
    gen_params = GeneratorParams(
        block_size=32,
        max_batch_size=batch_size,
        max_seq_len=16384,
    )
    
    engine = engine_cls(
        hf_model=model,
        gen_params=gen_params,
        device=device,
    )

    requests = make_requests(
        batch_size,
        context_len,
        gen_len,
    )

    duration, peak_mem, _ = measure_engine_generation(
        engine,
        requests,
    )

    return BenchResult(
        batch_size=batch_size,
        context_len=context_len,
        gen_len=gen_len,
        duration_ms=duration,
        peak_mem_mb=peak_mem,
    )

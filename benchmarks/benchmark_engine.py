import argparse
import time
import torch
from dataclasses import dataclass

from gllm.engine.llm_engine import LLMEngine
from gllm.config.generator_params import GeneratorParams
from gllm.engine.llm_engine_base import GenerationRequest
from gllm.engine.hf_llm_engine import HuggingFaceLLMEngine


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def now():
    return time.perf_counter()


@dataclass
class BenchResult:
    engine: str
    batch_size: int
    context_len: int
    gen_len: int
    prefill_ms: float
    decode_ms_per_tok: float
    tokens_per_sec: float
    peak_mem_mb: float


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


def measure_engine(engine_name, engine_fn):
    torch.cuda.reset_peak_memory_stats()
    sync()

    t0 = now()
    result = engine_fn()
    sync()
    t1 = now()

    peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
    return (t1 - t0) * 1000, peak_mem, result


def benchmark_gllm(
    model,
    gen_params,
    device,
    requests,
):
    llm_engine = LLMEngine(
        hf_model=model,
        gen_params=gen_params,
        device=device,
    )

    sync()
    t0 = now()
    results = llm_engine.generate(requests)
    sync()
    t1 = now()

    del llm_engine
    return (t1 - t0), results


def benchmark_hf(
    model,
    device,
    requests,
):
    hf_engine = HuggingFaceLLMEngine(model, device)

    sync()
    t0 = now()
    results = hf_engine.generate(requests)
    sync()
    t1 = now()

    del hf_engine
    return (t1 - t0), results


def run_benchmarks(
    model,
    device="cuda",
):
    results = []
    for batch_size in [1, 2, 4, 8]:
        for context_len in [32, 128, 512, 1024]:
            for gen_len in [1, 4]:
                gen_params = GeneratorParams(
                    block_size=32,
                    max_batch_size=batch_size,
                    max_seq_len=2048,
                )

                requests = make_requests(
                    batch_size,
                    context_len,
                    gen_len,
                )

                # ---------------- gLLM ----------------
                gllm_time, gllm_mem, _ = measure_engine(
                    "gllm",
                    lambda: benchmark_gllm(
                        model, gen_params, device, requests
                    ),
                )

                total_tokens = batch_size * gen_len
                gllm_decode_ms_per_tok = (gllm_time * 1000) / total_tokens
                gllm_tok_per_sec = total_tokens / gllm_time

                results.append(
                    BenchResult(
                        engine="gllm",
                        batch_size=batch_size,
                        context_len=context_len,
                        gen_len=gen_len,
                        prefill_ms=gllm_time * 1000,
                        decode_ms_per_tok=gllm_decode_ms_per_tok,
                        tokens_per_sec=gllm_tok_per_sec,
                        peak_mem_mb=gllm_mem,
                    )
                )

                # ---------------- HF ----------------
                hf_time, hf_mem, _ = measure_engine(
                    "hf",
                    lambda: benchmark_hf(
                        model, device, requests
                    ),
                )

                hf_decode_ms_per_tok = (hf_time * 1000) / total_tokens
                hf_tok_per_sec = total_tokens / hf_time

                results.append(
                    BenchResult(
                        engine="hf",
                        batch_size=batch_size,
                        context_len=context_len,
                        gen_len=gen_len,
                        prefill_ms=hf_time * 1000,
                        decode_ms_per_tok=hf_decode_ms_per_tok,
                        tokens_per_sec=hf_tok_per_sec,
                        peak_mem_mb=hf_mem,
                    )
                )

                print(
                    f"[B={batch_size} C={context_len} G={gen_len}] "
                    f"gLLM {gllm_tok_per_sec:.6f} tok/s | "
                    f"HF {hf_tok_per_sec:.6f} tok/s"
                )
    return results
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    run_benchmarks(
        model=args.model,
        device=args.device,
    )


if __name__ == "__main__":
    main()

import argparse

from benchmarks.utils import benchmark
from gllm.engine.llm_engine import LLMEngine
from gllm.engine.hf_llm_engine import HuggingFaceLLMEngine


def benchmark_engine_vs_hf_generation(
    model,
    device="cuda",
):
    for batch_size in [1, 2, 4, 8]:
        for context_len in [32, 128, 512, 1024]:
            for gen_len in [1, 4]:
                # gllm
                gllm_bench_result = benchmark(
                    engine_cls=LLMEngine,
                    model=model,
                    batch_size=batch_size,
                    context_len=context_len,
                    gen_len=gen_len,
                    device=device,
                )

                # HuggingFace
                hf_bench_result = benchmark(
                    engine_cls=HuggingFaceLLMEngine,
                    model=model,
                    batch_size=batch_size,
                    context_len=context_len,
                    gen_len=gen_len,
                    device=device,
                )

                print(
                    f"[B={batch_size} C={context_len} G={gen_len}]\n"
                    f"gllm\t duration: {gllm_bench_result.duration_ms:.6f} ms\n"
                    f"hf\t duration: {hf_bench_result.duration_ms:.6f} ms"
                )
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    benchmark_engine_vs_hf_generation(
        model=args.model,
        device=args.device,
    )


if __name__ == "__main__":
    main()

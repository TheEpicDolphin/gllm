import argparse

from benchmarks.utils import benchmark
from gllm.engine.llm_engine import LLMEngine


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--batch-size", required=True, type=int)
    parser.add_argument("--context-len", required=True, type=int)
    parser.add_argument("--gen-len", required=True, type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cpu-offloading", action="store_true", help="Enable CPU offloading of model weights")
    parser.add_argument("--trace-file", default=None)
    args = parser.parse_args()

    result = benchmark(
        engine_cls=LLMEngine,
        model=args.model,
        batch_size=args.batch_size,
        context_len=args.context_len,
        gen_len=args.gen_len,
        device=args.device,
        cpu_offloading=args.cpu_offloading,
        trace_file=args.trace_file,
    )
    
    print(
        f"[B={args.batch_size} C={args.context_len} G={args.gen_len}] duration: {result.duration_ms:.6f} ms"
    )


if __name__ == "__main__":
    main()

import argparse

from gllm.engine.llm_engine import LLMEngine


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--batch-size", required=True)
    parser.add_argument("--context-len", required=True)
    parser.add_argument("--gen-len", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    benchmark(
        engine_cls=typeof(LLMEngine),
        model=args.model,
        batch_size=args.batch_size,
        context_len=args.context_len,
        gen_len=args.gen_len,
        device=args.device,
    )


if __name__ == "__main__":
    main()

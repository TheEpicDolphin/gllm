# gLLM — A Minimal LLM Inference Engine optimized for gaming GPUs

gLLM is a from-scratch **LLM inference engine** implemented in PyTorch and optimized for running locally on consumer gaming GPUs. This project deliberately avoids high-level inference frameworks in order to implement the techniques used by production systems such as vLLM, ExLlama, and optimized HuggingFace inference stacks. This project serves both as a learning exercise for me and as a tool for training and running AI in games, enabling unique gameplay experiences. I hope that one day others can benefit from this tool as well.

## Features

### Core Inference
- Autoregressive token-by-token decoding
- Batched inference with variable-length prompts and generations
- Top-k and nucleus (top-p) sampling
- Per-token log-probability tracking

### Attention & KV Cache
- Multi-headed self-attention with Rotary Position Embeddings (RoPE)
- Paged KV caching with block-based allocation and reuse
- Efficient separation of context and query attention
- Attention bias construction for padded and causal sequences

### Engine & Scheduling
- Processing multiple generation requests concurrently
- Continuous batching of incoming requests
- Request lifecycle management (enqueue → generate → finish)

Only Llama 3.1 models are currently supported. I plan to support more in the future!

## Architecture Overview

User Requests -> LLMEngine (Scheduler) -> Batch Construction -> Attention + Paged KV Cache -> Logits -> Sampler (top-k / top-p) -> Next Token

## Sampling Details
The sampler supports:
- Temperature scaling
- Top-k filtering
- Top-p (nucleus) filtering
- Optional log-probability extraction

Sampling flow:
1. Reduce logits to a per-request top-k candidate set
2. Apply top-p masking on sorted top-k logits
3. Sample via multinomial draw
4. Compute logprobs from the original unfiltered distribution

## Benchmarks
Below is my comparison of engine performance for gLLM vs HF's transformers library.
| Batch Size | Context Length | Generation Length | gLLM tok/s | HF tok/s |
|---|------|---|--------------|--------------|
| 1 | 32   | 1 | 0.000173     | 0.000101     |
| 1 | 32   | 4 | 0.000227     | 0.000438     |
| 1 | 128  | 1 | 0.000193     | 0.000111     |
| 1 | 128  | 4 | 0.000228     | 0.000442     |
| 1 | 512  | 1 | 0.000191     | 0.000108     |
| 1 | 512  | 4 | 0.000224     | 0.000445     |
| 1 | 1024 | 1 | 0.000184     | 0.000106     |
| 1 | 1024 | 4 | 0.000222     | 0.000420     |
| 2 | 32   | 1 | 0.000386     | 0.000227     |
| 2 | 32   | 4 | 0.000452     | 0.000864     |
| 2 | 128  | 1 | 0.000379     | 0.000221     |
| 2 | 128  | 4 | 0.000448     | 0.000885     |
| 2 | 512  | 1 | 0.000364     | 0.000208     |
| 2 | 512  | 4 | 0.000438     | 0.000838     |
| 2 | 1024 | 1 | 0.000343     | 0.000192     |
| 2 | 1024 | 4 | 0.000431     | 0.000787     |
| 4 | 32   | 1 | 0.000766     | 0.000444     |
| 4 | 32   | 4 | 0.000900     | 0.001774     |
| 4 | 128  | 1 | 0.000751     | 0.000413     |
| 4 | 128  | 4 | 0.000894     | 0.001726     |
| 4 | 512  | 1 | 0.000695     | 0.000406     |
| 4 | 512  | 4 | 0.000868     | 0.001622     |
| 4 | 1024 | 1 | 0.000611     | 0.000368     |
| 4 | 1024 | 4 | 0.000830     | 0.001467     |
| 8 | 32   | 1 | 0.001526     | 0.000882     |
| 8 | 32   | 4 | 0.001790     | 0.003485     |
| 8 | 128  | 1 | 0.001447     | 0.000843     |
| 8 | 128  | 4 | 0.001768     | 0.003354     |
| 8 | 512  | 1 | 0.001229     | 0.000740     |
| 8 | 512  | 4 | 0.001668     | 0.002890     |
| 8 | 1024 | 1 | 0.000983     | 0.000537     |
| 8 | 1024 | 4 | 0.000983     | 0.001680     |

gLLM's prefill beats that of HF, across all batch sizes. However, gLLM's decode needs is currently lagging and needs improvement. Will investigate further.

## Testing
Basic correctness tests can be run using pytest. They compare outputs with huggingface transformers, which is used as a baseline for this engine.

## Project Structure
```
gllm/
├── engine/      # Request scheduling & engine loop
├── llm/         # High-level LLM interface
├── model/       # Transformer layers & attention
├── scheduler/   # Request state tracking
├── sample/      # Token sampling & logprobs
├── config/      # Model and generation configuration
└── utils/       # Helper utilities
tests/           # Unit & integration tests
```

## Example Usage
```python
from gllm.engine import GeneratorParams, GenerationRequest, LLMEngine

engine = LLMEngine(
    hf_model=HuggingFaceModel.LLAMA_3_2_1B_INSTUCT,
    gen_params=GeneratorParams(
        block_size=16,
        max_batch_size=8,
        max_seq_len=256,
    ),
    device="cpu",
)

req = GenerationRequest(
    prompt="Once upon a time",
    max_new_tokens=64,
    temperature=0.8,
    top_k=50,
    top_p=0.95,
)
result = await engine.enqueue_request(req)

print(result.text)
```

## Future Extensions
- Speculative decoding
- Prefix caching
- Post-training with LoRa
- FP8 Quantization
- FlashAttention / Triton kernels
- Mixture of Experts
- Other models
- ... and most importantly, a framework for training game AI for general tasks

## Author
Giancarlo Delfin - AI Systems Engineer at Meta interested in LLM inference, game development, and optimizing systems

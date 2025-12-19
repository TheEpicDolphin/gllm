# gLLM — A Minimal LLM Inference Engine Optimized for Gaming GPUs

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

# gLLM — A Minimal LLM Inference Engine from Scratch

gLLM is a from-scratch **LLM inference engine** implemented in PyTorch. The project deliberately avoids high-level inference frameworks in order to explicitly implement the mechanics used by production systems such as vLLM, ExLlama, and optimized HuggingFace inference stacks.

## Features

### Core Inference
- Autoregressive token-by-token decoding
- Batched inference with variable-length prompts and generations
- Top-k and nucleus (top-p) sampling
- Per-token log-probability tracking

### Attention & KV Cache
- Multi-head self-attention with Rotary Position Embeddings (RoPE)
- Paged KV cache with block-based allocation and reuse
- Efficient separation of context and query attention
- Attention bias construction for padded and causal sequences

### Engine & Scheduling
- Request scheduling engine supporting multiple concurrent generation requests
- Dynamic request batching across decode steps
- Request lifecycle management (enqueue → generate → finish)
- Async-friendly interface using asyncio.Future

### Modular Design
- Clean separation of concerns:
  - LLM: model execution and attention metadata
  - Sampler: token sampling and logprob computation
  - LLMEngine: request scheduling and batching
  - Attention: attention kernels and KV cache integration
- Designed for extensibility (speculative decoding, quantization, multi-GPU)

## Architecture Overview

User Requests -> LLMEngine (Scheduler) -> Batch Construction -> Attention + Paged KV Cache -> Logits -> Sampler (top-k / top-p) -> Next Token

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

llm_engine = LLMEngine(
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
future = engine.enqueue_request(req)
result = await future

print(result.text)
```

## Sampling Details
The sampler supports:
- Temperature scaling
- Top-k filtering
- Top-p (nucleus) filtering
- Optional log-probability extraction

Sampling flow:
1. Reduce logits to a per-request top-k candidate set
2. Apply top-p masking on sorted logits
3. Sample via multinomial draw
4. Compute logprobs from the original unfiltered distribution

## Testing
Basic correctness tests can be run using pytest. They compare outputs with huggingface transformers, which is used as a baseline for this engine.

## Future Extensions
- Post-training with LoRa
- Speculative decoding
- FP8 Quantization
- FlashAttention / Triton kernels

## Motivations
- Learn how LLMs work under the hood.
- I love optimizing stuff.
- I want to apply this technology to gaming. For example: chess AI, realistic NPC dialogue, etc.

## Disclaimer
This project is intended for educational and experimental purposes.
It is not a drop-in replacement for production inference engines. Thought, hopefully one day it will be.

## Author
Giancarlo Delfin - AI Systems Engineer at Meta focused on LLM inference, systems performance, and model serving.

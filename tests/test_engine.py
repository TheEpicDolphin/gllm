import asyncio
import math
import pytest

from gllm.config.generator_params import GeneratorParams
from gllm.engine.hf_llm_engine import HuggingFaceLLMEngine
from gllm.engine.llm_engine import LLMEngine
from gllm.engine.llm_engine_base import GenerationRequest
from gllm.llm.llm import LLM
from gllm.model.model import HuggingFaceModel


@pytest.mark.asyncio
@pytest.mark.parametrize("prompt", [
    # Instruction
    "Explain the theory of relativity in simple terms.",
    "Write a short story about a dragon and a wizard.",
    "List three healthy breakfast options.",
    "Summarize the following paragraph: 'Artificial intelligence is rapidly changing the world...'",
    # Question & answer
    "What is the capital of France?",
    "Solve this math problem: 12 * 8 + 5 = ?",
    "Who wrote 'Pride and Prejudice'?",
    # Reasoning
    "Write a Python function to compute factorial of n.",
    "Explain step by step how to solve 45 + 37 - 12.",
    "Convert the list [1,2,3] into a dictionary where keys are the numbers and values are their squares.",
    # Short deterministic
    "Repeat the word 'hello' 3 times.",
    "2+2=",
    "Return the first letter of 'Python'.",
    # Translation
    "Translate this sentence to Spanish: 'The cat is on the roof.'",
    "Translate to French: 'I like to read books.'",
    # Creative
    # TODO: Investigate why these fail.
    # "Write a haiku about winter.",
    # "Compose a friendly email to remind someone of a meeting.",
    # "Describe a futuristic city in 3 sentences.",
])
async def test_individual_generation_correctness(prompt: str):
    model = HuggingFaceModel.LLAMA_3_2_1B_INSTUCT
    device = "cpu"
    gen_params = GeneratorParams(
        block_size=16,
        max_batch_size=8,
        max_seq_len=256,
    )
    
    request = GenerationRequest(
        prompt=prompt,
        max_new_tokens=16,
        stop_tokens=[],
        temperature=0.1,
        top_k=1,
        top_p=1.0,
    )
    
    # Generate using gllm engine.
    llm_engine = LLMEngine(
        LLM(
            hf_model=model,
            gen_params=gen_params,
            device=device,
        )
    )
    actual = llm_engine.generate([request])[0]
    del llm_engine
    
    # Generate using huggingface transformers engine.
    hf_llm_engine = HuggingFaceLLMEngine(model, device)
    expected = hf_llm_engine.generate([request])[0]
    del hf_llm_engine
    
    # Compare results.
    assert actual.text == expected.text


@pytest.mark.asyncio
@pytest.mark.parametrize("prompts", [
    [
        "Explain the theory of relativity in simple terms.",
        "Write a Python function to compute factorial of n.",
    ],
    [
        "Summarize the following paragraph: 'Artificial intelligence is rapidly changing the world...'",
        "2+2=",
        "Repeat the word 'hello' 3 times.",
        "Translate this sentence to Spanish: 'The cat is on the roof.'",
    ],
])
async def test_batched_generation_correctness(prompts: list[str]):  
    model = HuggingFaceModel.LLAMA_3_2_1B_INSTUCT
    device = "cpu"
    gen_params = GeneratorParams(
        block_size=16,
        max_batch_size=8,
        max_seq_len=256,
        model_dtype="float32",
        kv_dtype="float32",
    )
    
    requests = []
    for prompt in prompts:
        requests.append(
            GenerationRequest(
                prompt=prompt,
                max_new_tokens=16,
                stop_tokens=[],
                temperature=0.1,
                top_k=1,
                top_p=1.0,
                max_num_logprobs=1,
            )
        )
    
    # Create LLM engine.
    llm_engine = LLMEngine(
        LLM(
            hf_model=model,
            gen_params=gen_params,
            device=device,
        )
    )
    # Generate in batch.
    batched_results = llm_engine.generate(requests)
    # Generate individually.
    individual_results = [llm_engine.generate([req])[0] for req in requests]
    del llm_engine
    
    # Compare results.
    for batched_result, individual_result in zip(batched_results, individual_results):
        for breq_logprobs, ireq_logprobs in zip(batched_result.logprobs, individual_result.logprobs):
            # Compare logprobs for approximate equality.
            assert math.isclose(breq_logprobs[0], ireq_logprobs[0], rel_tol=1e-5, abs_tol=1e-5)
            # Compare tokens for exact equality.
            assert breq_logprobs[1] == ireq_logprobs[1]
        assert batched_result.text == individual_result.text
import asyncio
import pytest

from gllm.config.generator_params import GeneratorParams
from gllm.engine.hf_llm_engine import HuggingFaceLLMEngine
from gllm.engine.llm_engine import LLMEngine
from gllm.engine.llm_engine_base import GenerationRequest
from gllm.llm.llm import LLM
from gllm.model.model import HuggingFaceModel


INSTRUCTION_PROMPTS = [
    "Explain the theory of relativity in simple terms.",
    "Write a short story about a dragon and a wizard.",
    "List three healthy breakfast options.",
    "Summarize the following paragraph: 'Artificial intelligence is rapidly changing the world...'",
]

QUESTION_ANSWER_PROMPTS = [
    "What is the capital of France?",
    "Solve this math problem: 12 * 8 + 5 = ?",
    "Who wrote 'Pride and Prejudice'?",
]

TRANSLATION_PROMPTS = [
    "Translate this sentence to Spanish: 'The cat is on the roof.'",
    "Translate to French: 'I love programming.'",
]

REASONING_PROMPTS = [
    "Write a Python function to compute factorial of n.",
    "Explain step by step how to solve 45 + 37 - 12.",
    "Convert the list [1,2,3] into a dictionary where keys are the numbers and values are their squares.",
]

CREATIVE_PROMPTS = [
    "Write a haiku about winter.",
    "Compose a friendly email to remind someone of a meeting.",
    "Describe a futuristic city in 3 sentences.",
]

SHORT_DETERMINISTIC_PROMPTS = [
    "Repeat the word 'hello' 3 times.",
    "2+2=",
    "Return the first letter of 'Python'.",
]


@pytest.mark.asyncio
@pytest.mark.parametrize("prompts", [
    [
        INSTRUCTION_PROMPTS[0]
    ],
    [
        INSTRUCTION_PROMPTS[3],
        REASONING_PROMPTS[0],
        CREATIVE_PROMPTS[1],
        SHORT_DETERMINISTIC_PROMPTS[1],
    ],
])
async def test_generation_correctness(prompts: list[str]):
    model = HuggingFaceModel.LLAMA_3_2_1B_INSTUCT
    device = "cpu"
    gen_params = GeneratorParams(
        block_size=16,
        max_batch_size=8,
        max_chunked_prefill=128,
        max_seq_len=256,
    )
    
    requests = []
    for prompt in prompts:
        requests.append(
            GenerationRequest(
                prompt=prompt,
                max_new_tokens=32,
                stop_tokens=[],
                temperature=0.1,
                top_k=1,
                top_p=1.0,
            )
        )
    
    # Generate using gllm engine.
    llm_engine = LLMEngine(
        LLM(
            hf_model=model,
            gen_params=gen_params,
            device=device,
        )
    )
    results = llm_engine.generate(requests)
    del llm_engine
    
    # Generate using huggingface transformers engine.
    hf_llm_engine = HuggingFaceLLMEngine(model, device)
    hf_results = hf_llm_engine.generate(requests)
    del hf_llm_engine
    
    # Compare results.
    for actual, expected in zip(results, hf_results):
        print("ACTUAL: ", actual.text)
        print("EXPECTED: ", expected.text)
        assert actual.text == expected.text
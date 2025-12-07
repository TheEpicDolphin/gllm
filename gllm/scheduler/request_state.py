import asyncio
from dataclasses import dataclass

from gllm.sample.logprobs import TokenLogProbs


@dataclass
class RequestState:
    id: int
    prompt_token_ids: list[int]
    generated_token_ids: list[int]
    generated_logprobs: list[TokenLogProbs]
    is_finished: bool
    
    max_new_tokens: int
    stop_token_ids: set[int]
    temperature: float
    top_k: int
    top_p: float
    max_num_logprobs: int

    future: asyncio.Future
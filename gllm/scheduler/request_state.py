import asyncio
from dataclasses import dataclass

@dataclass
class RequestState:
    id: int
    prompt_token_ids: list[int]
    generated_token_ids: list[int]
    is_finished: bool
    
    max_new_tokens: int
    stop_token_ids: set[int]
    temperature: float
    top_k: int
    top_p: float

    future: asyncio.Future
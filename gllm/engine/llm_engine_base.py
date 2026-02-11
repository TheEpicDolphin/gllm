from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import NamedTuple

from gllm.engine.logprobs import TokenLogProbs


@dataclass
class GenerationRequest:
    prompt: str
    max_new_tokens: int
    stop_tokens: list[str]
    temperature: float = 0.7
    top_k: int = 40 # If 0, top_k is just set to vocab_size.
    top_p: float = 0.9
    num_top_logprobs: int = 0 # If 0, no per-token logprobs are returned.


@dataclass
class GenerationResult:
    token_ids: list[int]
    top_logprobs: list[TokenLogProbs] | None
    text: str


class LLMEngineBase(ABC):

    @abstractmethod
    def generate(self, reqs: list[GenerationRequest]) -> list[GenerationResult]:
        pass
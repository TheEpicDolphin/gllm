from dataclasses import dataclass
from typing import NamedTuple


@dataclass
class TokenLogProbs(NamedTuple):
    candidate_token_ids: list[int]
    logprobs: list[float]
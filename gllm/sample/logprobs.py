from dataclasses import dataclass
from typing import NamedTuple


@dataclass
class TokenLogProbs(NamedTuple):
    logprobs: list[float]
    candidate_token_ids: list[int]
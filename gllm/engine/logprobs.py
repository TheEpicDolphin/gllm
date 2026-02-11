from typing import NamedTuple


class TokenLogProbs(NamedTuple):
    logprobs: list[float]
    token_ids: list[int]
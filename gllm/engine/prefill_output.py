from typing import NamedTuple

import torch


class PrefillOutput(NamedTuple):
    # [B, 1]
    token_ids: torch.Tensor
    # [B, 1, top_logprobs]
    top_logprobs: torch.Tensor
    # [B, 1, top_logprobs]
    top_logprobs_token_ids: torch.Tensor
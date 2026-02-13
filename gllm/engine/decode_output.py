from typing import NamedTuple

import torch


class DecodeOutput(NamedTuple):
    # [B, T_q]
    token_ids: torch.Tensor
    # [B, T_q, top_logprobs]
    top_logprobs: torch.Tensor
    # [B, T_q, top_logprobs]
    top_logprobs_token_ids: torch.Tensor

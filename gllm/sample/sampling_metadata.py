from dataclasses import dataclass

import torch


@dataclass
class SamplingMetadata:
    # [B]
    temperature: torch.Tensor
    # [B]
    top_k: torch.Tensor
    # [B]
    top_p: torch.Tensor
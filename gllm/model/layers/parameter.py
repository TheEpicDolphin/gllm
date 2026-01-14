from dataclasses import dataclass
import torch


@dataclass
class Parameter:
    weights: torch.Tensor
    grad: torch.Tensor | None = None
    requires_grad: bool = True
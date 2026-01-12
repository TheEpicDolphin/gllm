import torch
import torch.nn.functional as F

from gllm.model.layers.base_module import BaseModule

class Linear(BaseModule):
    def __init__(
        self,
        weights: torch.Tensor,
    ):
        super().__init__(weights)


    def _forward_impl(
        self,
        x: torch.Tensor,
        weights: torch.Tensor | None,
    ) -> torch.Tensor:
        # out = W @ x
        return F.linear(x, weights)
        
        
    def _backward_impl(
        self,
        dL_dy: torch.Tensor,
        weights: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # dL/dx = W^T @ dL/dy
        dL_dx = weights.transpose(-1, -2) @ dL_dy
        return dL_dx, None
import torch

from gllm.model.layers.base_module import BaseModule
from gllm.model.params import Parameter


class Embedding(BaseModule):
    def __init__(
        self,
        parameter: Parameter,
    ):
        super().__init__(parameter)


    def _forward_impl(
        self,
        # [B, T]
        x: torch.Tensor,
        weights: torch.Tensor | None,
    ) -> torch.Tensor:
        self.cache_for_backward(x)
        # out = W_e[x]
        return weights[x]
        
        
    def _backward_impl(
        self,
        dL_dy: torch.Tensor,
        weights: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self._cache
        dL_dw = torch.zeros_like(weights)
        dL_dw[x] += dL_dy
        return 0, dL_dw
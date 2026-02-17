import torch
import torch.nn.functional as F

from gllm.model.layers.base_module import BaseModule
from gllm.model.params import Parameter


class Linear(BaseModule):
    def __init__(
        self,
        parameter: Parameter,
    ):
        super().__init__(parameter)


    def _forward_impl(
        self,
        x: torch.Tensor,
        weights: torch.Tensor | None,
    ) -> torch.Tensor:
        # Cache input for training backward pass.
        self.cache_for_backward(x)
        # y = x @ W^T
        return F.linear(x, weights)
        
        
    def _backward_impl(
        self,
        dL_dy: torch.Tensor,
        weights: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x, = self._cache
        
        # Theory:
        # dL/dx_ij = dL/dy_i1 * dy_i1/dx_ij + dL/dy_i2 * dy_i2/dx_ij + ... + dL/dy_iN * dy_iN/dx_ij
        #          = dL/dy_i1 * w_1j + dL/dy_i2 * w_2j + ... + dL/dy_iN * w_Nj
        # Which is equivalent to:
        # dL/dx = dL/dy @ W
        dL_dx = dL_dy @ weights
        
        # Theory:
        # dL/dw_ij = dL/dy_1j * dy_1j/dw_ij + dL/dy_2j * dy_2j/dw_ij + ... + dL/dy_Mj * dy_Mj/dw_ij
        #          = dL/dy_1j * x_1j + dL/dy_2j * x_2j + ... + dL/dy_Nj * x_Mj
        # Which is equivalent to:
        # dL/dW = (dL/dy)^T @ x
        dL_dw = dL_dy.transpose(-1, -2) @ x
        return dL_dx, dL_dw
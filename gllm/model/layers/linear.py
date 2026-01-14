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
        # Cache input for training backward pass.
        self.cache_for_backward(x)
        # out = W @ x
        return F.linear(x, weights)
        
        
    def _backward_impl(
        self,
        dL_dy: torch.Tensor,
        weights: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self._cache
        
        # Theory:
        # dL/dx_ij = dL/dy_1j * dy_1j/dx_ij + dL/dy_2j * dy_2j/dx_ij + ... + dL/dy_Nj * dy_Nj/dx_ij
        #          = dL/dy_1j * w_1j + dL/dy_2j * w_2j + ... + dL/dy_Nj * w_Nj
        # Which is equivalent to:
        # dL/dx = W^T @ dL/dy
        dL_dx = weights.transpose(-1, -2) @ dL_dy
        
        # Theory:
        # dL/dw_ij = dL/dy_i1 * dy_i1/dw_ij + dL/dy_i2 * dy_i2/dw_ij + ... + dL/dy_iN * dy_iN/dw_ij
        #          = dL/dy_i1 * x_j1 + dL/dy_i2 * x_j2 + ... + dL/dy_iN * x_jN
        # Which is equivalent to:
        # dL/dW = dL/dy @ x^T
        dL_dw = dL_dy @ x.transpose(-1, -2)
        return dL_dx, dL_dw
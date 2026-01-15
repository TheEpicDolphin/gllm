import torch

from gllm.model.layers.base_module import BaseModule

class RMSNorm(BaseModule):
    def __init__(
        self,
        weights: torch.Tensor,
        eps: float,
    ):
        super().__init__(weights)
        self.eps = eps
    
    
    def rms(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        # RMS(x) = sqrt((1/N) * (x_1^2 + x_2^2 + ... + x_N^2))
        x_sqr_mean = torch.mean(x * x, -1, keepdim=True)
        return torch.sqrt(x_sqr_mean + self.eps)
    
        
    def _forward_impl(
        self,
        x: torch.Tensor,
        weights: torch.Tensor | None,
    ) -> torch.Tensor:
        # Cache input for training backward pass.
        self.cache_for_backward(x)
        
        w = weights
        rms = self.rms(x)
        return (x / rms) * w

    
    def _backward_impl(
        self,
        dL_dy: torch.Tensor,
        weights: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x, = self._cache
        w = weights
        N = x.shape[-1]
        inv_rms = 1 / self.rms(x)
        
        # Theory:
        # dL/dx_i = dL/dy_1 * dy_1/dx_i + dL/dy_2 * dy_2/dx_i + ... + dL/dy_N * dy_N/dx_i
        # If j != i
        #       dy_j/dx_i = -x_i * x_j * w_i / (N * RMS(x)^3)
        # Else j == i
        #       dy_i/dx_i = w_i / RMS(x) - x_i * x_j * w_i / (N * RMS(x)^3)
        #
        # dy/dx = [ dy_1/dx_1   dy_2/dx_1   ... dy_N/dx_1 ]
        #         [ dy_1/dx_2   dy_2/dx_2   ... dy_N/dx_2 ]
        #         [                . . .                  ]
        #         [ dy_1/dx_N   dy_2/dx_N   ... dy_N/dx_N ]
        # dy/dx @ dL/dy = [ dL/dy_1 * dy_1/dx_1 + dL/dy_2 * dy_2/dx_1 + ... + dL/dy_N * dy_N/dx_1 ]
        #                 [ dL/dy_1 * dy_1/dx_2 + dL/dy_2 * dy_2/dx_2 + ... + dL/dy_N * dy_N/dx_2 ]
        #                 [                                 . . .                                 ]
        #                 [ dL/dy_1 * dy_1/dx_N + dL/dy_2 * dy_2/dx_N + ... + dL/dy_N * dy_N/dx_N ]
        #
        # Below code produces the same result as above, but without materializing
        # the NxN dy/dx jacobian matrix.
        dot = torch.sum(dL_dy * x * w, dim=-1, keepdim=True)
        dL_dx = dL_dy * w * inv_rms - x * dot * (1/N) * (inv_rms**3)
        
        # Theory:
        # dL/dw_i = dL/dy_i * dy_i/w_i = dL/dy_i * (x_i / RMS(x))
        dL_dw = dL_dy * x * inv_rms
        return dL_dx, dL_dw
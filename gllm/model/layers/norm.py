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
        x_sqr_mean = torch.mean(x * x, -1, keepdim=True)
        return torch.sqrt(x_sqr_mean + self.eps)
    
        
    def _forward_impl(
        self,
        x: torch.Tensor,
        weights: torch.Tensor | None,
    ) -> torch.Tensor:
        # Cache activations for training backward pass.
        self._cache_activations(x)
        
        w = weights
        rms = self.rms(x)
        return (x / rms) * w

    
    def _backward_impl(
        self,
        dL_dy: torch.Tensor,
        weights: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        w = weights
        x = self._cache
        N = x.shape[-1]
        inv_rms = 1 / self.rms(x)
        
        # Theory:
        #
        # dy_i/dx_j = w_i * [(i == j) / RMS(x) - x_i * x_j / (N * RMS(x)^3)]
        # dy/dx = [ dy_1/dx_1, dy_2/dx_1, ..., dy_N/dx_1 ]
        #         [ dy_1/dx_2, dy_2/dx_2, ..., dy_N/dx_2 ]
        #         [                . . .                 ]
        #         [ dy_1/dx_N, dy_2/dx_N, ..., dy_N/dx_N ]
        # dy/dx @ dL/dy = [ dL/dy_1 * dy_1/dx_1 + dL/dy_2 * dy_2/dx_1 + ... + dL/dy_N * dy_N/dx_1 ]
        #                 [ dL/dy_1 * dy_1/dx_2 + dL/dy_2 * dy_2/dx_2 + ... + dL/dy_N * dy_N/dx_2 ]
        #                 [                                 . . .                                 ]
        #                 [ dL/dy_1 * dy_1/dx_N + dL/dy_2 * dy_2/dx_N + ... + dL/dy_N * dy_N/dx_N ]
        #
        # Below code produces the same result as above, but without materializing
        # the NxN dy/dx jacobian matrix.
        dot = torch.sum(dL_dy, x * w, dim=-1, keepdim=True)
        dL_dx = dL_dy * w * inv_rms - x * dot * (inv_rms**3) / N
        return dL_dx, None
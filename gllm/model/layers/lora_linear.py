import torch
import torch.nn.functional as F

from gllm.model.layers.base_module import BaseModule


class LoRaLinear(BaseModule):
    def __init__(
        self,
        weights: torch.Tensor,
        r: int,
        alpha: float,
    ):
        super().__init__(None)
        
        # Initialize low-rank matrices.
        d_out, d_in = weights.shape
        # B is initialized to all zeros.
        B = torch.zeros((d_out, r), dtype=weights.dtype, device=weights.device)
        # A is initialized using normal distribution.
        A = 0.01 * torch.randn((r, d_in), dtype=weights.dtype, device=weights.device)
        
        self.scale = alpha / r
        self.linear_W = Linear(weights)
        self.linear_B = Linear(B)
        self.linear_A = Linear(A)
        
        # Backbone weights are frozen, no gradient tracking.
        self.linear_w.requires_grad = False
        
        self.child_modules = [
            self.linear_W,
            self.linear_B,
            self.linear_A,
        ]


    def _forward_impl(
        self,
        x: torch.Tensor,
        weights: torch.Tensor | None,
    ) -> torch.Tensor:
        # Cache input for training backward pass.
        self.cache_for_backward(x)
        # out = (W + (alpha / r) * B @ A) @ x
        #     = W @ x + (alpha / r) * B @ (A @ x)
        # We deliberately avoid materializing the B @ A matrix.
        h_w = self.linear_W.forward(x)
        h_a = self.linear_A.forward(x)
        h_b = self.linear_B.forward(h_a)
        return h_w + self.scale * h_b
        
        
    def _backward_impl(
        self,
        dL_dy: torch.Tensor,
        weights: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self._cache

        # y = h_w + h_b
        # dL/dh_w = dL/dy * dy/dh_w = dL/dy
        # dL/dh_b = dL/dy * dy/dh_b = dL/dy * (alpha / r)
        dL_dh_w = dL_dy
        dL_dh_b = dL_dy * self.scale
        
        # h_b = B @ h_a
        dL_dh_a = self.linear_B.backward(dL_dh_b)
        
        # h_w = W @ x
        dL_dx1 = self.linear_W.backward(dL_dh_w)
        # h_a = A @ x
        dL_dx2 = self.linear_A.backward(dL_dh_a)
        dL_dx = dL_dx1 + dL_dx2
        return dL_dx, None
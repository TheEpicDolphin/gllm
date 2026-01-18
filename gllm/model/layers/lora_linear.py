import torch

from gllm.model.layers.linear import Linear


class LoRALinear(Linear):
    def __init__(
        self,
        id: str,
        weights: torch.Tensor,
        r: int,
        alpha: float,
    ):
        super().__init__(id, weights)
        
        # Initialize low-rank matrices.
        d_out, d_in = weights.shape
        # B is initialized to all zeros.
        B = torch.zeros((d_out, r), dtype=weights.dtype, device=weights.device)
        # A is initialized using normal distribution.
        A = 0.01 * torch.randn((r, d_in), dtype=weights.dtype, device=weights.device)
        
        self.scale = alpha / r
        self.linear_B = Linear(f"{id}.lora_up", B)
        self.linear_A = Linear(f"{id}.lora_down", A)
        
        # Backbone weights are frozen, no gradient tracking.
        self._parameter.requires_grad = False
        
        self.child_modules = [
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
        # y = x @ (W + s(B @ A))^T = x(W^T + s(B @ A)^T) = xW^T + s((x @ A^T) @ B^T)
        # We deliberately avoid materializing the BA matrix.
        h_w = super().forward(x)
        h_a = self.linear_A.forward(x)
        h_b = self.linear_B.forward(h_a)
        return h_w + self.scale * h_b
        
        
    def _backward_impl(
        self,
        dL_dy: torch.Tensor,
        weights: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x, = self._cache

        # y = h_w + s * h_b
        # dL/dh_w = dL/dy * dy/dh_w = dL/dy
        # dL/dh_b = dL/dy * dy/dh_b = dL/dy * s
        dL_dh_w = dL_dy
        dL_dh_b = dL_dy * self.scale
        
        # h_b = h_a @ B^T
        dL_dh_a = self.linear_B.backward(dL_dh_b)
        
        # h_w = x @ W^T
        dL_dx1 = super().backward(dL_dh_w)
        # h_a = x @ A^T
        dL_dx2 = self.linear_A.backward(dL_dh_a)
        dL_dx = dL_dx1 + dL_dx2
        return dL_dx, None
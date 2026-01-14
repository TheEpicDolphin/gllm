import torch
import torch.nn.functional as F

from gllm.config.model_config import ActivationFunction, ModelConfig
from gllm.model.layers.base_module import BaseModule
from gllm.model.layers.linear import Linear

class FFN(BaseModule):
    def __init__(
        self,
        layer_idx: int,
        model_config: ModelConfig,
        safetensors
    ):
        super().__init__(None)
        
        ffn_prefix = f"model.layers.{layer_idx}.mlp"
        dtype = model_config.dtype
        
        # [intermediate_size, hidden_size]
        W_up = safetensors[f"{ffn_prefix}.up_proj.weight"].to(dtype)
        # [intermediate_size, hidden_size]
        W_gate = safetensors[f"{ffn_prefix}.gate_proj.weight"].to(dtype)
        # [hidden_size, intermediate_size]
        W_down = safetensors[f"{ffn_prefix}.down_proj.weight"].to(dtype)

        
        self.linear_up = Linear(W_up)
        self.linear_gate = Linear(W_gate)
        self.linear_down = Linear(W_down)
        
        self.child_modules = [
            self.linear_up,
            self.linear_gate,
            self.linear_down,
        ]

        if model_config.act_func == ActivationFunction.SILU:
            self.act_forward = F.silu
            self.act_backward = self.silu_backward
        else:
            raise NotImplementedError(f"The '{model_config.act_func}' activation function is not yet implemented.")


    def silu_backward(
        self,
        dL_dy: torch.Tensor,
        x: torch.Tensor,
    ) -> torch.Tensor:
        # silu(x) = x / (1 + e^-x)
        # d/dx[silu(x)] = 1 / (1 + e^-x) + x * e^-x / (1 + e^-x)^2
        #               = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
        sigma_x = torch.sigmoid(x)
        dy_dx = (sigma_x * (1 + x * (1 - sigma_x)))
        return dL_dy * dy_dx
        
    
    def gated_activation(
        self,
        # [B, T, hidden_size]
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        g = self.linear_gate.forward(x)
        u = self.linear_up.forward(x)
        a = self.act_forward(g)
        return g, u, a


    def _forward_impl(
        self,
        # [B, T, hidden_size]
        x: torch.Tensor,
        weights: torch.Tensor | None,
    ) -> torch.Tensor:
        # Cache activations for training backward pass.
        self.cache_for_backward(x)
        
        _, u, a = self.gated_activation(x)
        h = a * u
        return self.linear_down.forward(h)
        
    
    def _backward_impl(
        self,
        dL_dy: torch.Tensor,
        weights: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self._cache
        g, u, a = self.gated_activation(x)
        
        # Backpropagation logic:
        # dL/dx = dL/dy * dy/dh * [dh/da * da/dg * dg/dx + dh/du * du/dx]
    
        # Down projection.
        dL_dh = self.linear_down.backward(dL_dy)
        
        # h = a * u
        dL_da = dL_dh * u
        dL_du = dL_dh * a
        
        # a = silu(g)
        dL_dg = self.act_backward(dL_da, g)
        
        # Gate and up projections.
        dL_dx1 = self.linear_gate.backward(dL_dg)
        dL_dx2 = a * self.linear_up.backward(dL_du)
        dL_dx = dL_dx1 + dL_dx2
        return dL_dx, None
        
        
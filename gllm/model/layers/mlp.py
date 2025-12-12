import torch
import torch.nn.functional as F

from gllm.config.model_config import ActivationFunction, ModelConfig
from gllm.model.layers.linear import Linear

class MLP:
    def __init__(
        self,
        layer_idx: int,
        model_config: ModelConfig,
        safetensors
    ):
        mlp_prefix = f"model.layers.{layer_idx}.mlp"
        dtype = model_config.dtype
        W_down = safetensors[f"{mlp_prefix}.down_proj.weight"].to(dtype)
        W_gate = safetensors[f"{mlp_prefix}.gate_proj.weight"].to(dtype)
        W_up = safetensors[f"{mlp_prefix}.up_proj.weight"].to(dtype)
        
        self.linear_down = Linear(W_down)
        self.linear_gate = Linear(W_gate)
        self.linear_up = Linear(W_up)

        if model_config.act_func == ActivationFunction.SILU:
            self.act_func = self.silu
        else:
            raise NotImplementedError(f"The '{model_config.act_func}' activation function is not yet implemented.")


    def silu(self, x: torch.Tensor) -> torch.Tensor:
        G = self.linear_gate.forward(x)
        U = self.linear_up.forward(x)
        return self.linear_down.forward(F.silu(G) * U)


    def forward(
        self,
        # [B, T, hidden_size]
        x: torch.Tensor
    ) -> torch.Tensor:
        return self.act_func(x)

import torch
import torch.nn.functional as F

from gllm.config.model_config import ActivationFunction, ModelConfig

class MLP:
    def __init__(
        self,
        layer_idx: int,
        model_config: ModelConfig,
        safetensors
    ):
        mlp_prefix = f"model.layers.{layer_idx}.mlp"
        self.W_down = safetensors[f"{mlp_prefix}.down_proj.weight"]
        self.W_gate = safetensors[f"{mlp_prefix}.gate_proj.weight"]
        self.W_up = safetensors[f"{mlp_prefix}.up_proj.weight"]

        if model_config.act_func == ActivationFunction.SILU:
            self.act_func = self.silu
        else:
            raise NotImplementedError(f"The '{model_config.act_func}' activation function is not yet implemented.")


    def silu(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        W_down = self.W_down.to(device)
        W_gate = self.W_gate.to(device)
        W_up = self.W_up.to(device)
        
        G = F.linear(x, W_gate)
        U = F.linear(x, W_up)
        return F.linear(F.silu(G) * U, W_down)


    def forward(
        self,
        # [B, T, hidden_size]
        x: torch.Tensor
    ) -> torch.Tensor:
        return self.act_func(x)

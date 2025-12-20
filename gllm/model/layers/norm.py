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
    
        
    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        weights = self.get_weights(x.device)
        x_sqr_mean = torch.mean(x * x, -1, keepdim=True)
        rms = torch.sqrt(x_sqr_mean + self.eps)
        return (x / rms) * weights

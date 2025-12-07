import torch

class RMSNorm:
    def __init__(
        self,
        weights: torch.Tensor,
        eps: float,
    ):
        self.weights = weights
        self.eps = eps

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        weights_gpu = self.weights.to(x.device)
        
        x_sqr_mean = torch.mean(x * x, -1, keepdim=True)
        rms = torch.sqrt(x_sqr_mean + self.eps)
        return (x / rms) * weights_gpu

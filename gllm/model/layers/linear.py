import torch
import torch.nn.functional as F

class Linear:
    def __init__(
        self,
        W: torch.Tensor,
    ):
        self.W = W

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        # Load weights onto same device as input.
        W = self.W.to(x.device)
        # out = W @ x
        return F.linear(x, W)

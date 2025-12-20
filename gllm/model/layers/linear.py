import torch
import torch.nn.functional as F

from gllm.model.layers.base_module import BaseModule

class Linear(BaseModule):
    def __init__(
        self,
        weights: torch.Tensor,
    ):
        super().__init__(weights)


    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        weights = self.get_weights(x.device)
        # out = W @ x
        return F.linear(x, weights)

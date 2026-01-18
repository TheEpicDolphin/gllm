from typing import cast, Iterable

import torch

from gllm.model.layers.parameter import Parameter

StagingBuffers = torch.Tensor | list["StagingBuffers"]

class BaseModule:
    def __init__(
        self,
        id: str,
        weights: torch.Tensor | None,
    ):
        self._id = id
        self._parameter: Parameter | None = Parameter(weights) if weights is not None else None
        self._preloaded_weights = None
        self.child_modules: list[BaseModule] = []
        
        self._training: bool = False
        self._cache = None
    
    
    @property
    def training(self):
        return self._training
        
        
    @training.setter
    def training(self, value):
        self._training = value
        for module in self.child_modules:
            module.training = value
    
    
    @property
    def requires_grad(self):
        return self._parameter is not None and self._parameter.requires_grad
    
    
    @requires_grad.setter
    def requires_grad(self, value):
        self._parameter.requires_grad = value
        for module in self.child_modules:
            module.requires_grad = value
    
    
    def parameters(self) -> Iterable[Parameter]:
        if self._parameter is not None:
            yield self._parameter
        for module in self.child_modules:
            yield from module.parameters()
    
    
    def save_tensors(
        self,
        weights_dict: dict[str, torch.Tensor]
    ) -> None:
        if self._parameter is not None:
            weights_dict[f"{self._id}.weight"] = self._parameter.weights
        for module in self.child_modules:
            module.save_tensors(weights_dict)
    
        
    def _get_weights(self, device) -> torch.Tensor:
        if self._preloaded_weights is not None:
            return self._preloaded_weights
        elif self._parameter is not None:
            # Synchronously load to device if not already preloaded.
            return self._parameter.weights.to(device)
        else:
            # This module owns no weights.
            return None
    
    
    def allocate_staging_buffers(self) -> list[torch.Tensor]:
        if self._parameter is not None:
            return self._parameter.weights.cpu().pin_memory()
        else:
            return [module.allocate_staging_buffers() for module in self.child_modules]
        
            
    def preload_weights(
        self,
        device,
        staging_buffers: StagingBuffers,
    ):
        if (self._parameter is not None
            and device != self._parameter.weights.device):
            staging_buffer = cast(torch.Tensor, staging_buffers)
            self._preloaded_weights = torch.empty_like(self._parameter.weights, device=device)
            # Copy to pinned CPU staging buffer.
            staging_buffer.copy_(self._parameter.weights)
            # Copy from pinned CPU staging buffer to device, asynchronously.
            self._preloaded_weights.copy_(staging_buffer, non_blocking=True)
        else:
            for module, staging_buffer in zip(self.child_modules, staging_buffers):
                module.preload_weights(device, staging_buffer)

    
    def unload_weights(self):
        self._preloaded_weights = None
        for module in self.child_modules:
            module.unload_weights()
            
    
    def cache_for_backward(self, *args):
        if self._training:
            self._cache = (args)
        
    
    def _forward_impl(
        self,
        x: torch.Tensor,
        weights: torch.Tensor | None,
        *args,
        **kwargs
    ) -> torch.Tensor:
        raise NotImplementedError
    
    
    def forward(
        self,
        x: torch.Tensor,
        *args,
        **kwargs
    ) -> torch.Tensor:
        weights = self._get_weights(x.device)
        return self._forward_impl(x, weights, *args, **kwargs)
    
    
    def _backward_impl(
        self,
        x: torch.Tensor,
        weights: torch.Tensor | None,
        *args,
        **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError
    
        
    def backward(
        self,
        dL_dy: torch.Tensor,
        *args,
        **kwargs
    ) -> torch.Tensor:
        weights = self._get_weights(dL_dy.device)
        dL_dx, dL_dW = self._backward_impl(dL_dy, weights, *args, **kwargs)
        
        # Track gradients.
        if self.requires_grad:
            if self._parameter.grad is None:
                self._parameter.grad = torch.zeros_like(dL_dW)
            self._parameter.grad += dL_dW
        
        return dL_dx
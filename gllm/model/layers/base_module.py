from typing import cast, Iterable

import torch

from gllm.model.layers.parameter import Parameter

StagingBuffers = torch.Tensor | list["StagingBuffers"]

class BaseModule:
    def __init__(
        self,
        weights: torch.Tensor | None,
    ):
        self._parameter: Parameter | None = Parameter(weights) if weights is not None else None
        self._preloaded_weights = None
        self.child_modules: list[BaseModule] = []
        
        self.training: bool = False
        self._cache = None
        self._grad = None
        
        
    def set_training_mode(self, training):
        self.training = training
        for module in self.child_modules:
            module.set_training_mode(training)
    
    
    def parameters(self) -> Iterable[Parameter]:
        if self._parameter is not None:
            yield _parameter
        for module in self.child_modules:
            yield from module.parameters()
    
        
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
            and device != self.self._parameter.weights.device):
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
            
    
    def cache_for_backward(self, *args, **kwargs):
        if self.training:
            self._cache = (args, kwargs)
        
    
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
        dL_dx, grad = self._backward_impl(dL_dy, weights, *args, **kwargs)
        
        # Track gradients.
        if grad is not None:
            if self._parameter.grad is None:
                self._parameter.grad = torch.zeros_like(grad)
            self._parameter.grad += grad
        
        return dL_dx
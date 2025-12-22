from typing import cast

import torch

StagingBuffers = torch.Tensor | list["StagingBuffers"]

class BaseModule:
    def __init__(
        self,
        weights: torch.Tensor | None,
    ):
        self._weights = weights
        self._preloaded_weights = None
        self.child_modules: list[BaseModule] = []
        
        
    def get_weights(self, device) -> torch.Tensor:
        if self._preloaded_weights is None:
            # Synchronously load to device if not already preloaded.
            return self._weights.to(device)
        else:
            return self._preloaded_weights
    
    
    def allocate_staging_buffers(self) -> list[torch.Tensor]:
        if self._weights is not None:
            return self._weights.cpu().pin_memory()
        else:
            return [module.allocate_staging_buffers() for module in self.child_modules]
        
            
    def preload_weights(
        self,
        device,
        staging_buffers: StagingBuffers,
    ):
        if (self._weights is not None
            and device != self._weights.device):
            staging_buffer = cast(torch.Tensor, staging_buffers)
            self._preloaded_weights = torch.empty_like(self._weights, device=device)
            # Copy to pinned CPU staging buffer.
            staging_buffer.copy_(self._weights)
            # Copy from pinned CPU staging buffer to device, asynchronously.
            self._preloaded_weights.copy_(staging_buffer, non_blocking=True)
        else:
            for module, staging_buffer in zip(self.child_modules, staging_buffers):
                module.preload_weights(device, staging_buffer)

    
    def unload_weights(self):
        self._preloaded_weights = None
        for module in self.child_modules:
            module.unload_weights()
from collections import OrderedDict
from dataclasses import dataclass

import torch

from gllm.config.generator_params import GeneratorParams
from gllm.config.model_config import ModelConfig

@dataclass
class GPUBlock:
    id: int
    ref_count: int = 0
    

class PagedKVCache:
    def __init__(
        self,
        model_config: ModelConfig,
        gen_params: GeneratorParams,
        device: str,
    ):
        num_layers = model_config.num_layers
        num_q_heads = model_config.num_attn_heads
        num_kv_heads = model_config.num_kv_heads
        kv_dtype = model_config.kv_dtype
        max_batch_size = gen_params.max_batch_size
        max_seq_len = gen_params.max_seq_len
        self.block_size = gen_params.block_size
        head_dim = model_config.hidden_size // num_q_heads
        max_num_blocks = self.num_required_blocks(max_batch_size * max_seq_len)
        # [2, num_layers, max_num_blocks, block_size, num_kv_heads, head_dim]
        self.physical_kv_cache: torch.Tensor = torch.zeros(
            (
                2,  # K/V
                num_layers,
                max_num_blocks,
                self.block_size,
                num_kv_heads,
                head_dim,
            ),
            dtype=kv_dtype,
            device=device,
        )
        # Zero out the 0th block. It will be used as a placeholder for no-op attention.
        self.physical_kv_cache[:, :, 0, :, :, :] = 0
        
        self.flattened_kv_cache = self.physical_kv_cache.view(2, num_layers, -1, num_kv_heads, head_dim)
        
        # Not yet referenced by any requests.
        self.free_blocks = []
        # 0th block is reserved as a dummy block.
        for i in range(1, max_num_blocks):
            self.free_blocks.append(GPUBlock(i))
        
        # Currently referenced by one or more requests..
        self.active_blocks : dict[int, GPUBlock] = {}
        # Blocks not currently referenced by any requests.
        self.cached_blocks : OrderedDict[int, GPUBlock] = {}
        # Maps hash id to GPU block id.
        self.hash_id_map: dict[int, int] = {}
        
    
    def get_layer_kv_cache(self, layer: int) -> torch.Tensor:
        # [2, max_num_blocks * block_size, num_kv_heads, head_dim]
        return self.flattened_kv_cache[:, layer]
    
    
    def num_required_blocks(self, num_tokens: int) -> int:
        return 1 + (num_tokens - 1) // self.block_size
    
     
    def prefetch_blocks(
        self,
        token_ids: list[int],
    ) -> tuple[list[int], int]:
        # TODO: Hash each block of token_ids, and check in order:
        # 1. match in active_blocks
        # 2. match in cached_blocks
        # 3. any in free_blocks
        # 4. any in cached_blocks
        return ([], 0)
    
        
    def reserve_blocks(
        self,
        num_blocks: int,
    ) -> list[int]:
        block_ids = []
        for i in range(num_blocks):
            if len(self.free_blocks) > 0:
                block = self.free_blocks.pop()
            else:
                block = self.cached_blocks.pop()
                # Set block to all zeros.
                self.physical_kv_cache[:, :, block.id, :, :, :] = 0
            
            block_ids.append(block.id)
            self.active_blocks[block.id] = block
            block.ref_count += 1
        return block_ids
    
    
    def release_blocks(
        self,
        block_ids: list[int],
    ):
        for id in block_ids:
            block = self.active_blocks[id]
            block.ref_count -= 1
            
            if block.ref_count == 0:
                self.active_blocks.pop(id)
                self.cached_blocks[id] = block
    
    
    def get_kv_cache_tensors(
        self,
        block_ids: torch.Tensor,
    ) -> torch.Tensor:
        return self.physical_kv_cache[:, :, block_ids, :, :, :]
    
    
    def set_kv_cache_tensors(
        self,
        block_ids: torch.Tensor,
        tensors: torch.Tensor,
    ) -> torch.Tensor:
        self.physical_kv_cache[:, :, block_ids, :, :, :] = tensors 
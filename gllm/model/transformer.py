import torch
import torch.nn.functional as F

from gllm.config.model_config import ModelConfig
from gllm.model.layers.attention import Attention, AttentionMetadata
from gllm.model.layers.base_module import BaseModule, StagingBuffers
from gllm.model.layers.norm import RMSNorm
from gllm.model.layers.mlp import MLP

class Transformer(BaseModule):
    def __init__(
        self,
        layer_idx: int,
        model_config: ModelConfig,
        safetensors,
    ):
        super().__init__(None)
        
        dtype = model_config.dtype
        # Initialize layer input norm.
        input_layernorm_weights = safetensors[f"model.layers.{layer_idx}.input_layernorm.weight"].to(dtype)
        self.input_norm = RMSNorm(
            weights=input_layernorm_weights,
            eps=model_config.rms_norm_eps
        )
        # Initialize attention.
        self.attention = Attention(layer_idx, model_config, safetensors)
        # Initialize MLP.
        self.mlp = MLP(layer_idx, model_config, safetensors)
        # Initialize post-attention norm.
        post_attn_norm_weights = safetensors[f"model.layers.{layer_idx}.post_attention_layernorm.weight"].to(dtype)
        self.post_attn_norm = RMSNorm(
            weights=post_attn_norm_weights,
            eps=model_config.rms_norm_eps
        )
        
        # Stream for preloading weights to device.
        self.transfer_stream = torch.cuda.Stream()
        
        self.child_modules = [
            self.input_norm,
            self.attention,
            self.mlp,
            self.post_attn_norm,
        ]
    
    
    def preload_weights(
        self,
        device,
        staging_buffers: StagingBuffers,
    ):
        with torch.cuda.stream(self.transfer_stream):
            super().preload_weights(device, staging_buffers)
        

    def forward(
        self,
        # [B, T, hidden_size]
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        # [B, T_q, head_dim // 2]
        cos_pos: torch.Tensor,
        # [B, T_q, head_dim // 2]
        sin_pos: torch.Tensor,
        # [2, max_num_blocks * block_size, num_kv_heads, head_dim]
        kv_cache: torch.Tensor,
        attention_metadata: AttentionMetadata,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is not None:
            hidden_states += residual
            
        # Wait for weights transfer to finish.
        torch.cuda.current_stream().wait_stream(self.transfer_stream)
            
        # input layernorm
        residual = hidden_states
        hidden_states = self.input_norm.forward(hidden_states)
        # Self attention
        hidden_states = self.attention.forward(
            hidden_states,
            cos_pos,
            sin_pos,
            kv_cache,
            attention_metadata
        )
        hidden_states += residual
        # Post attention layernorm
        residual = hidden_states
        hidden_states = self.post_attn_norm.forward(hidden_states)
        # MLP
        hidden_states = self.mlp.forward(hidden_states)
        return hidden_states, residual
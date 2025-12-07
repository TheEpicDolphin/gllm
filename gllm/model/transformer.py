import torch
import torch.nn.functional as F

from gllm.config.model_config import ModelConfig
from gllm.model.layers.attention import Attention, AttentionMetadata
from gllm.model.layers.norm import RMSNorm
from gllm.model.layers.mlp import MLP

class Transformer:
    def __init__(
        self,
        layer_idx: int,
        model_config: ModelConfig,
        safetensors,
    ):
        # Initialize layer input norm.
        input_layernorm_weights = safetensors[f"model.layers.{layer_idx}.input_layernorm.weight"]
        self.input_norm = RMSNorm(
            weights=input_layernorm_weights,
            eps=model_config.rms_norm_eps
        )
        # Initialize post-attention norm.
        post_attn_norm_weights = safetensors[f"model.layers.{layer_idx}.post_attention_layernorm.weight"]
        self.post_attn_norm = RMSNorm(
            weights=post_attn_norm_weights,
            eps=model_config.rms_norm_eps
        )
        # Initialize attention.
        self.attention = Attention(layer_idx, model_config, safetensors)
        # Initialize MLP.
        self.mlp = MLP(layer_idx, model_config, safetensors)
        

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
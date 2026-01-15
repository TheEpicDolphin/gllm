import torch
import torch.nn.functional as F

from gllm.config.model_config import ModelConfig
from gllm.model.layers.attention import Attention, AttentionMetadata
from gllm.model.layers.base_module import BaseModule, StagingBuffers
from gllm.model.layers.norm import RMSNorm
from gllm.model.layers.ffn import FFN

class TransformerLayer(BaseModule):
    def __init__(
        self,
        layer_idx: int,
        model_config: ModelConfig,
        safetensors,
    ):
        super().__init__(None)
        
        dtype = model_config.dtype
        # Initialize layer input norm.
        input_layernorm_weights = safetensors[f"model.layers.{layer_idx}.input_layernorm.weight"].to(dtype=dtype)
        self.input_norm = RMSNorm(
            weights=input_layernorm_weights,
            eps=model_config.rms_norm_eps
        )
        # Initialize attention.
        self.attention = Attention(layer_idx, model_config, safetensors)
        # Initialize FFN.
        self.ffn = FFN(layer_idx, model_config, safetensors)
        # Initialize post-attention norm.
        post_attn_norm_weights = safetensors[f"model.layers.{layer_idx}.post_attention_layernorm.weight"].to(dtype=dtype)
        self.post_attn_norm = RMSNorm(
            weights=post_attn_norm_weights,
            eps=model_config.rms_norm_eps
        )
        
        self.child_modules = [
            self.input_norm,
            self.attention,
            self.ffn,
            self.post_attn_norm,
        ]
        

    def _forward_impl(
        self,
        # [B, T_q, hidden_size]
        x: torch.Tensor,
        weights: torch.Tensor | None,
        # [B, T_q, head_dim // 2]
        cos_pos: torch.Tensor,
        # [B, T_q, head_dim // 2]
        sin_pos: torch.Tensor,
        # [2, max_num_blocks * block_size, num_kv_heads, head_dim]
        kv_cache: torch.Tensor | None,
        attention_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        # input layernorm
        residual = x
        h_ln1 = self.input_norm.forward(x)
        # Self attention
        h_attn = self.attention.forward(
            h_ln1,
            cos_pos,
            sin_pos,
            kv_cache,
            attention_metadata
        )
        # h_attn_r = h_attn + x
        h_attn += residual
        # Post attention layernorm
        residual = h_attn
        h_ln2 = self.post_attn_norm.forward(h_attn)
        # Feed-forward network
        h_ffn = self.ffn.forward(h_ln2)
        # h_out = h_ffn + h_attn_resid
        h_ffn += residual
        return h_ffn
        
    
    def _backward_impl(
        self,
        dL_dy: torch.Tensor,
        weights: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Backpropagation logic:
        # dL/dx = dL/dy * [dy/dh_ffn * dh_ffn/dh_ln2 * dh_ln2/dh_attn_r * [dh_attn_r/dh_attn * dh_attn/dh_ln1 * dh_ln1/dx + 1])
        #               + dy/dh_attn_r * [dh_attn_r/dh_attn * dh_attn/dh_ln1 * dh_ln1/dx + 1]]
        
        # y = h_ffn + h_attn_r
        dL_dh_ffn = dL_dy
        dL_dh_attn_r_2 = dL_dy
        
        # h_ffn = ffn(h_ln2)
        dL_dh_ln2 = self.ffn.backward(dL_dh_ffn)
        
        # h_ln2 = RMSNorm(h_attn)
        dL_dh_attn_r_1 = self.post_attn_norm.backward(dL_dh_ln2)
        
        # h_attn_r = h_attn + x
        dL_dh_attn_1 = dL_dh_attn_r_1
        dL_dx1 = dL_dh_attn_r_1
        dL_dh_attn_2 = dL_dh_attn_r_2
        dL_dx2 = dL_dh_attn_r_2
        
        # h_attn = attention(h_ln1)
        dL_dh_ln1 = self.attention.backward(dL_dh_attn_1 + dL_dh_attn_2)
        
        # h_ln1 = RMSNorm(x)
        dL_dx3 = self.input_norm.backward(dL_dh_ln1)
        dL_dx = dL_dx1 + dL_dx2 + dL_dx3
        return dL_dx, None
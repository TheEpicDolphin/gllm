import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from gllm.config.model_config import ModelConfig
from gllm.model.layers.base_module import BaseModule
from gllm.model.layers.linear import Linear
from gllm.ops.attention.reference_attention import reference_attention

@dataclass
class AttentionMetadata:
    # [B]
    query_lens: torch.Tensor
    # [B]
    seq_lens: torch.Tensor
    max_seq_len: int
    # [B, max_num_blocks]
    block_table: torch.Tensor
    # [B, T]
    slot_mapping: torch.Tensor
    # [B, T_q]
    query_slot_mapping: torch.Tensor
    # [B, T_q, T - T_q]
    context_bias: torch.Tensor
    # [B, T_q, T_q]
    query_bias: torch.Tensor

class Attention(BaseModule):
    def __init__(
        self,
        layer_idx: int,
        model_config: ModelConfig,
        safetensors,
    ):
        super().__init__(None)
        
        self.layer_idx = layer_idx
        self.num_q_heads = model_config.num_attn_heads
        self.num_kv_heads = model_config.num_kv_heads
        self.hidden_size = model_config.hidden_size
        self.head_dim = model_config.head_dim
        
        assert self.num_q_heads % self.num_kv_heads == 0
        self.num_groups = self.num_q_heads // self.num_kv_heads
        
        # Sanity check.
        # hidden_size = num_q_heads * head_dim
        assert self.hidden_size == self.num_q_heads * self.head_dim
        
        attn_prefix = f"model.layers.{layer_idx}.self_attn"
        dtype = model_config.dtype
        
        # [hidden_size, hidden_size]
        W_q = safetensors[f"{attn_prefix}.q_proj.weight"].to(dtype=dtype)
        # [hidden_size, num_kv_heads * head_dim]
        W_k = safetensors[f"{attn_prefix}.k_proj.weight"].to(dtype=dtype)
        # [hidden_size, num_kv_heads * head_dim]
        W_v = safetensors[f"{attn_prefix}.v_proj.weight"].to(dtype=dtype)
        # [hidden_size, hidden_size]
        W_o = safetensors[f"{attn_prefix}.o_proj.weight"].to(dtype=dtype)
        
        self.linear_q = Linear(W_q)
        self.linear_k = Linear(W_k)
        self.linear_v = Linear(W_v)
        self.linear_o = Linear(W_o)
        
        self.child_modules = [
            self.linear_q,
            self.linear_k,
            self.linear_v,
            self.linear_o,
        ]
    
    
    def apply_rope(
        self,
        # [B, T_q, num_q_heads, head_dim]
        q: torch.Tensor,
        # [B, T_q, num_kv_heads, head_dim]
        k: torch.Tensor,
        # [B, T_q, head_dim // 2]
        cos_pos: torch.Tensor,
        # [B, T_q, head_dim // 2]
        sin_pos: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:       
        B, T_q, num_q_heads, head_dim = q.shape
        _, _, num_kv_heads, _ = k.shape
        
        # [B, T_q, num_q_heads, 2, head_dim // 2]
        q = q.view(B, T_q, num_q_heads, 2, -1)
        # [B, T_q, num_kv_heads, 2, head_dim // 2]
        k = k.view(B, T_q, num_kv_heads, 2, -1)
        
        # [B, T_q, num_q_heads, head_dim // 2]
        q_even, q_odd = q.unbind(dim=3)
        # [B, T_q, num_kv_heads, head_dim // 2]
        k_even, k_odd = k.unbind(dim=3)
        
        # [B, T_q, 1, head_dim // 2]
        cos_pos = cos_pos.unsqueeze(2)
        sin_pos = sin_pos.unsqueeze(2)
        
        # Apply rotations.
        q_r = torch.stack(
            [
                q_even * cos_pos - q_odd * sin_pos,
                q_even * sin_pos + q_odd * cos_pos,
            ],
            dim=3,
        ).view(B, T_q, num_q_heads, -1)
        k_r = torch.stack(
            [
                k_even * cos_pos - k_odd * sin_pos,
                k_even * sin_pos + k_odd * cos_pos,
            ],
            dim=3,
        ).view(B, T_q, num_kv_heads, -1)
        return q_r, k_r
        
    
    def forward(
        self,
        # [B, T_q, hidden_size]
        x: torch.Tensor,
        # [B, T_q, head_dim // 2]
        cos_pos: torch.Tensor,
        # [B, T_q, head_dim // 2]
        sin_pos: torch.Tensor,
        # [2, max_num_blocks * block_size, num_kv_heads, head_dim]
        kv_cache: torch.Tensor,
        attention_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        B, T_q, hidden_size = x.shape
        assert hidden_size == self.hidden_size
        
        # Transform using q, k, v weight matrices.
        # [B, T_q, num_q_heads * head_dim]
        q = self.linear_q.forward(x)
        # [B, T_q, num_kv_heads * head_dim]
        k = self.linear_k.forward(x)
        v = self.linear_v.forward(x)
        
        # [B, T_q, num_q_heads, head_dim]
        q = q.view(B, T_q, self.num_q_heads, self.head_dim)
        # [B, T_q, num_kv_heads, head_dim]
        k = k.view(B, T_q, self.num_kv_heads, self.head_dim)
        v = v.view(B, T_q, self.num_kv_heads, self.head_dim)
        
        # Apply RoPE rotation matrix to q and k.
        q, k = self.apply_rope(q, k, cos_pos, sin_pos)
        
        # Cache query token K/Vs.
        # TODO: Remove dummy query slots to reduce copying.
        # [B, T_q]
        query_slot_mapping = attention_metadata.query_slot_mapping
        # [B * T_q]
        query_slot_mapping = query_slot_mapping.view(-1)
        kv_dtype = kv_cache.dtype
        kv_cache[0, query_slot_mapping, :, :] = k.view(-1, self.num_kv_heads, self.head_dim).to(kv_dtype)
        kv_cache[1, query_slot_mapping, :, :] = v.view(-1, self.num_kv_heads, self.head_dim).to(kv_dtype)
        
        # Get sequence K/Vs.
        # [B, T]
        slot_mapping = attention_metadata.slot_mapping
        # [B * T]
        slot_mapping = slot_mapping.view(-1)
        # [B, T, num_kv_heads, head_dim]
        k = kv_cache[0, slot_mapping, :, :].view(B, -1, self.num_kv_heads, self.head_dim).to(q.dtype)
        v = kv_cache[1, slot_mapping, :, :].view(B, -1, self.num_kv_heads, self.head_dim).to(q.dtype)
        
        # Compute attention.
        attn_out = reference_attention(
            q,
            k,
            v,
            attention_metadata.query_bias,
            attention_metadata.context_bias
        )
        
        # [B, T_q, hidden_size]
        attn_out = attn_out.view(B, T_q, -1)
        return self.linear_o.forward(attn_out)
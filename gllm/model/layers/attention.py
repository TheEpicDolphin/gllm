import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from gllm.config.model_config import ModelConfig

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
    # [B, T_q, T]
    bias: torch.Tensor

class Attention:
    def __init__(
        self,
        layer_idx: int,
        model_config: ModelConfig,
        safetensors,
    ):
        self.layer_idx = layer_idx
        self.num_q_heads = model_config.num_attn_heads
        self.num_kv_heads = model_config.num_kv_heads
        self.hidden_size = model_config.hidden_size
        self.head_dim = model_config.head_dim
        
        assert self.num_q_heads % self.num_kv_heads == 0
        self.num_groups = self.num_q_heads // self.num_kv_heads
        
        # Sanity check.
        assert self.head_dim == self.hidden_size // self.num_q_heads
        
        attn_prefix = f"model.layers.{layer_idx}.self_attn"
        # [hidden_size, num_q_heads * head_dim]
        self.W_q = safetensors[f"{attn_prefix}.q_proj.weight"]
        # [hidden_size, num_kv_heads * head_dim]
        self.W_k = safetensors[f"{attn_prefix}.k_proj.weight"]
        # [hidden_size, num_kv_heads * head_dim]
        self.W_v = safetensors[f"{attn_prefix}.v_proj.weight"]
        # [num_q_heads * head_dim, hidden_size]
        self.W_o = safetensors[f"{attn_prefix}.o_proj.weight"]
    
    
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
        

    def naiive_attention(
        self,
        # [B, T_q, num_q_heads, head_dim]
        q: torch.Tensor,
        # [B, T_q, num_kv_heads, head_dim]
        k: torch.Tensor,
        # [B, T_q, num_kv_heads, head_dim]
        v: torch.Tensor,
    ) -> torch.Tensor:
        assert q.shape == k.shape
        assert k.shape == v.shape
        B, T_q, num_q_heads, head_dim = q.shape
        
        if self.num_groups > 1:
            # Multi-query attention.
            # [B, T_q, num_kv_heads, 1, head_dim]
            k = k.unsqueeze(-2)
            v = v.unsqueeze(-2)
            # [B, T_q, num_kv_heads, num_groups, head_dim]
            k = k.expand(-1, -1, -1, self.num_groups, -1)
            v = v.expand(-1, -1, -1, self.num_groups, -1)
            # [B, T_q, num_kv_heads * num_groups, head_dim]
            k = k.reshape(B, T_q, -1, self.head_dim)
            v = v.reshape(B, T_q, -1, self.head_dim)
            
        # [B, num_heads, T_q, head_dim]
        q = q.transpose(1, 2)
        # [B, num_heads, T_q, head_dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Compute softmax(Q @ K^T / sqrt(d_k)).
        # [B, num_heads, T_q, T_q]
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # [B, num_heads, T_q, T_q]
        attn_probs = F.softmax(attn_scores, dim=-1)

        # Scale values by scores.
        # [B, num_heads, T_q, head_dim]
        attn_out = torch.matmul(attn_probs, v)
        # [B, T_q, num_heads, head_dim]
        attn_out = attn_out.transpose(1, 2).contiguous()
        return attn_out
    
    def paged_attention(
        self,
        # [B, T_q, num_q_heads, head_dim]
        q: torch.Tensor,
        # [B, T_q, num_kv_heads, head_dim]
        k: torch.Tensor,
        # [B, T_q, num_kv_heads, head_dim]
        v: torch.Tensor,
        # [2, max_num_blocks * block_size, num_kv_heads, head_dim]
        kv_cache: torch.Tensor,
        attention_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        assert k.shape == v.shape
        B, T_q, num_q_heads, head_dim = q.shape
        _, _, num_kv_heads, _ = kv_cache.shape
        
        # Cache query token K/Vs.
        # TODO: Remove dummy query slots to reduce copying.
        # [B, T_q]
        query_slot_mapping = attention_metadata.query_slot_mapping
        query_slot_mapping = query_slot_mapping.view(-1)
        kv_cache[0, query_slot_mapping, :, :] = k.view(-1, num_kv_heads, head_dim)
        kv_cache[1, query_slot_mapping, :, :] = v.view(-1, num_kv_heads, head_dim)
        
        # Get sequence K/Vs.
        # [B, T]
        slot_mapping = attention_metadata.slot_mapping
        # [B * T]
        slot_mapping = slot_mapping.view(-1)
        # [B, T, num_kv_heads, head_dim]
        k = kv_cache[0, slot_mapping, :, :].view(B, -1, num_kv_heads, head_dim)
        v = kv_cache[1, slot_mapping, :, :].view(B, -1, num_kv_heads, head_dim)
        _, T, _, _ = k.shape
        
        if self.num_groups > 1:
            # Multi-query attention.
            # [B, T, num_kv_heads, 1, head_dim]
            k = k.unsqueeze(-2)
            v = v.unsqueeze(-2)
            # [B, T, num_kv_heads, num_groups, head_dim]
            k = k.expand(-1, -1, -1, self.num_groups, -1)
            v = v.expand(-1, -1, -1, self.num_groups, -1)
            # [B, T, num_kv_heads * num_groups, head_dim]
            k = k.reshape(B, T, -1, self.head_dim)
            v = v.reshape(B, T, -1, self.head_dim)
            
        # [B, num_heads, T_q, head_dim]
        q = q.transpose(1, 2)
        # [B, num_heads, T, head_dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Compute attention scores: Q @ K^T / sqrt(d_k).
        # [B, num_heads, T_q, T]
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)        
        # Apply attention bias.
        # [B, T_q, T]
        attn_bias = attention_metadata.bias
        attn_scores += attn_bias.unsqueeze(1)
        # Compute softmax of scores.
        # [B, num_heads, T_q, T]
        attn_probs = F.softmax(attn_scores, dim=-1)
        # Scale values by scores.
        # [B, num_heads, T_q, head_dim]
        attn_out = torch.matmul(attn_probs, v)
        # [B, T_q, num_heads, head_dim]
        attn_out = attn_out.transpose(1, 2).contiguous()
        return attn_out
        
    
    def forward(
        self,
        # [B, T_q, hidden_size]
        x: torch.Tensor,
        # [B, T_q, head_dim // 2]
        cos_pos: torch.Tensor,
        # [B, T_q, head_dim // 2]
        sin_pos: torch.Tensor,
        # [2, max_num_blocks * block_size, num_kv_heads, head_dim]
        kv_cache: torch.Tensor | None,
        attention_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        B, T_q, hidden_size = x.shape
        assert hidden_size == self.hidden_size
        
        # Load attention weights on GPU.
        device = x.device
        W_q = self.W_q.to(device)
        W_k = self.W_k.to(device)
        W_v = self.W_v.to(device)
        W_o = self.W_o.to(device)

        # Transform using q, k, v weight matrices.
        # [B, T_q, num_q_heads * head_dim]
        q = F.linear(x, W_q)
        # [B, T_q, num_kv_heads * head_dim]
        k = F.linear(x, W_k)
        v = F.linear(x, W_v)
        
        # [B, T_q, num_q_heads, head_dim]
        q = q.view(B, T_q, self.num_q_heads, self.head_dim)
        # [B, T_q, num_kv_heads, head_dim]
        k = k.view(B, T_q, self.num_kv_heads, self.head_dim)
        v = v.view(B, T_q, self.num_kv_heads, self.head_dim)
        
        # Apply RoPE rotation matrix to q and k.
        q, k = self.apply_rope(q, k, cos_pos, sin_pos)

        # Compute attention.
        if kv_cache is None:
            attn_out = self.naiive_attention(q, k, v)
        else:
            attn_out = self.paged_attention(q, k, v, kv_cache, attention_metadata)
        
        # [B, T_q, hidden_size]
        attn_out = attn_out.view(B, T_q, -1)
        return F.linear(attn_out, W_o)
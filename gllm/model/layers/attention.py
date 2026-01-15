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
    # [B, T_q]
    positions: torch.Tensor
    # [B]
    query_lens: torch.Tensor
    # [B]
    seq_lens: torch.Tensor
    # [B, max_num_blocks]
    block_table: torch.Tensor
    # [B, T]
    slot_mapping: torch.Tensor
    # [B, T_q]
    query_slot_mapping: torch.Tensor
    # [B, T_q, T]
    bias: torch.Tensor


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
    
    
    def rope_forward(
        self,
        # [B, T_q, num_heads, head_dim]
        x: torch.Tensor,
        # [B, T_q, head_dim // 2]
        cos_pos: torch.Tensor,
        # [B, T_q, head_dim // 2]
        sin_pos: torch.Tensor,
    ) -> torch.Tensor:
        B, T_q, num_heads, _ = x.shape
        
        # Theory:
        # y = R @ x
        # [  x'_i  ] = [ cos(theta), -sin(theta) ] @ [  x_i  ]
        # [ x'_i+1 ]   [ sin(theta),  cos(theta) ]   [ x_i+1 ]
        
        # [B, T_q, num_heads, 2, head_dim // 2]
        x = x.view(B, T_q, num_heads, 2, -1)
        
        # [B, T_q, num_heads, head_dim // 2]
        x_even, x_odd = x.unbind(dim=3)
        
        # [B, T_q, 1, head_dim // 2]
        cos_pos = cos_pos.unsqueeze(2)
        sin_pos = sin_pos.unsqueeze(2)
        
        # Apply rotations.
        x_r = torch.stack(
            [
                x_even * cos_pos - x_odd * sin_pos,
                x_even * sin_pos + x_odd * cos_pos,
            ],
            dim=3,
        ).view(B, T_q, num_heads, -1)
        return x_r
        
        
    def rope_backward(
        self,
        # [B, T_q, num_heads, head_dim]
        dL_dy: torch.Tensor,
        # [B, T_q, head_dim // 2]
        cos_pos: torch.Tensor,
        # [B, T_q, head_dim // 2]
        sin_pos: torch.Tensor,
    ):
        # y = R @ x
        # dL_dx = R^T @ dL_dy
        # R^T is just R, but with sin(theta) negated.
        dL_dx = self.rope_forward(dL_dy, cos_pos, -sin_pos) @ dL_dy
        return dL_dx
        
    
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
        q = self.rope_forward(q, cos_pos, sin_pos)
        k = self.rope_forward(k, cos_pos, sin_pos)
        
        if kv_cache is not None:
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
        attn_out, p = reference_attention(
            q,
            k,
            v,
            attention_metadata.bias,
            return_probs=True,
        )
        
        # Cache activations for training backward pass.
        self.cache_for_backward(q, k, v, p, cos_pos, sin_pos)
        
        # [B, T_q, hidden_size]
        attn_out = attn_out.view(B, T_q, -1)
        return self.linear_o.forward(attn_out)
    
    
    def _backward_impl(
        self,
        dL_dy: torch.Tensor,
        weights: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q, k, v, p, cos_pos, sin_pos = self._cache
        
        # Backpropagation logic:
        # dL/dx = dL/dy * dy/dO * [dO/dP * [dP/dQ * dQ/dx + dP/dK * dK/dx] + dO/dV * dV/dx]
    
        # y = O @ W_o^T
        dL_do = self.linear_o.backward(dL_dy)
                
        # O = P @ V
        print("dL_do: ", dL_do.shape)
        print("v: ", v.shape)
        dL_dp = dL_do @ v.transpose(-1, -2)
        dL_dv = p.transpose(-1, -2) @ dL_do
        
        # P = softmax(S)
        #
        # Theory:
        # dp_i/ds_j = (i == j) * softmax(s) - e^s_i * e^s_j / sum(e^s)^2
        # dp/ds = [ dy_1/dx_1   dy_2/dx_1   ... dy_N/dx_1 ]
        #         [ dy_1/dx_2   dy_2/dx_2   ... dy_N/dx_2 ]
        #         [                 . . .                 ]
        #         [ dy_1/dx_N   dy_2/dx_N   ... dy_N/dx_N ]
        # dp/ds @ dL/dp = [ dL/dp_1 * dp_1/ds_1 + dL/dp_2 * dp_2/ds_1 + ... + dL/dp_N * dp_N/ds_1 ]
        #                 [ dL/dp_1 * dp_1/ds_2 + dL/dp_2 * dp_2/ds_2 + ... + dL/dp_N * dp_N/ds_2 ]
        #                 [                                 . . .                                 ]
        #                 [ dL/dp_1 * dp_1/ds_N + dL/dp_2 * dp_2/ds_N + ... + dL/dp_N * dp_N/ds_N ]
        #
        # Below code produces the same result as above, but without materializing
        # the NxN dp/ds jacobian matrix.
        dot = torch.sum(dL_dp * p, dim=-1, keepdim=True)
        dL_ds = p * (dL_dp - dot)
        
        # S = Q @ K^T / sqrt(d)
        scale = 1.0 / d**0.5
        ds_dqr = k * scale
        dL_dqr = dL_ds @ ds_dqr
        ds_dkr = q * scale
        dL_dkr = dL_ds.transpose(-1, -2) @ ds_dkr
        
        # RoPE
        dL_dq = self.rope_backward(dL_dqr, cos_pos, sin_pos)
        dL_dk = self.rope_backward(dL_dkr, cos_pos, sin_pos)
        
        # Q, K, and V linear projections.
        dL_dx_q = self.linear_q.backward(dL_dq)
        dL_dx_k = self.linear_k.backward(dL_dk)
        dL_dx_v = self.linear_v.backward(dL_dv)
        dL_dx = dL_dx_q + dL_dx_k + dL_dx_v
        return dL_dx, None
        
        
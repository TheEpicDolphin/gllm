import torch
import torch.nn.functional as F


def softmax(x: torch.Tensor) -> torch.Tensor:
    x_stable = x - x.max(dim=-1, keepdim=True).values
    exp_x = torch.exp(x_stable)
    return exp_x / exp_x.sum(dim=-1, keepdim=True)


def reference_attention(
    # [B, T_q, num_q_heads, head_dim]
    q: torch.Tensor,
    # [B, T, num_kv_heads, head_dim]
    k: torch.Tensor,
    # [B, T, num_kv_heads, head_dim]
    v: torch.Tensor,
    # [B, T_q, T]
    bias: torch.Tensor,
    return_probs: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    assert k.shape == v.shape
    B, T_q, num_q_heads, head_dim = q.shape
    _, T, num_kv_heads, _ = k.shape
    
    num_groups = num_q_heads // num_kv_heads
    if num_groups > 1:
        # Multi-query attention.
        # [B, T, num_kv_heads, 1, head_dim]
        k = k.unsqueeze(-2)
        v = v.unsqueeze(-2)
        # [B, T, num_kv_heads, num_groups, head_dim]
        k = k.expand(-1, -1, -1, num_groups, -1)
        v = v.expand(-1, -1, -1, num_groups, -1)
        # [B, T, num_kv_heads * num_groups, head_dim]
        k = k.reshape(B, T, -1, head_dim)
        v = v.reshape(B, T, -1, head_dim)
        
    # [B, num_heads, T_q, head_dim]
    q = q.transpose(1, 2)
    # [B, num_heads, T, head_dim]
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    # Compute attention scores: Q @ K^T / sqrt(d_k).
    # [B, num_heads, T_q, T]
    S = torch.matmul(q, k.transpose(-2, -1)) / head_dim**0.5
    # Apply attention bias.
    S += bias.unsqueeze(1)
    # Compute probabilities.
    # [B, num_heads, T_q, T]
    P = F.softmax(S, dim=-1)
    # Scale values by probabilities.
    # [B, num_heads, T_q, head_dim]
    attn_out = torch.matmul(P, v)
    # [B, T_q, num_heads, head_dim]
    attn_out = attn_out.transpose(1, 2).contiguous()
    return (attn_out, P) if return_probs else attn_out

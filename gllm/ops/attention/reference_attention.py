import torch
import torch.nn.functional as F


def softmax(x: torch.Tensor) -> torch.Tensor:
    x_stable = x - x.max(dim=-1, keepdim=True).values
    exp_x = torch.exp(x_stable)
    return exp_x / exp_x.sum(dim=-1, keepdim=True)


def reference_attention_fwd(
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
        # Multi-query attention. Broadcast number of kv heads to
        # number of q heads.
        k = torch.repeat_interleave(k, num_groups, dim=-2)
        v = torch.repeat_interleave(v, num_groups, dim=-2)
    
    # Swap sequence and heads ordering for forward attention.
    # [B, num_heads, T_q, head_dim]
    q = q.transpose(1, 2)
    # [B, num_heads, T, head_dim]
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    # Compute attention scores: Q @ K^T / sqrt(d).
    # [B, num_heads, T_q, T]
    scale = 1.0 / head_dim**0.5
    S = torch.matmul(q, k.transpose(-2, -1)) * scale
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


def reference_attention_bwd(
    dL_dy: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    p: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, T_q, num_q_heads, head_dim = q.shape
    _, T, num_kv_heads, _ = k.shape
    
    num_groups = num_q_heads // num_kv_heads
    if num_groups > 1:
        # Multi-query attention. Broadcast number of kv heads to
        # number of q heads.
        k = torch.repeat_interleave(k, num_groups, dim=-2)
        v = torch.repeat_interleave(v, num_groups, dim=-2)
    
    # Swap sequence and head dimensions for backward attention.
    dL_dy = dL_dy.transpose(1, 2)
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    
    # Y = P @ V
    dL_dp = dL_dy @ v.transpose(-1, -2)
    dL_dv = p.transpose(-1, -2) @ dL_dy
    
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
    scale = 1.0 / head_dim**0.5
    ds_dqr = k * scale
    dL_dqr = dL_ds @ ds_dqr
    ds_dkr = q * scale
    dL_dkr = dL_ds.transpose(-1, -2) @ ds_dkr
    
    # Swap sequence and heads dimensions back.
    dL_dqr = dL_dqr.transpose(1, 2)
    dL_dkr = dL_dkr.transpose(1, 2)
    dL_dv = dL_dv.transpose(1, 2)
    
    if num_groups > 1:
        # Undo broadcasting.
        dL_dkr = dL_dkr[:, :, ::num_groups, :]
        dL_dv = dL_dv[:, :, ::num_groups, :]
    return dL_dqr, dL_dkr, dL_dv
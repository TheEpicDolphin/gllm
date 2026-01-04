import torch
import triton

import triton.language as tl


@triton.jit
def flash_attention_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    bias_ptr,
    out_ptr,
    stride_Q_B, stride_Q_T, stride_Q_H, stride_Q_D,
    stride_K_B, stride_K_T, stride_K_H, stride_K_D,
    stride_V_B, stride_V_T, stride_V_H, stride_V_D,
    stride_bias_B, stride_bias_T_q, stride_bias_T,
    stride_out_B, stride_out_T, stride_out_H, stride_out_D,
    T_q: tl.constexpr, T: tl.constexpr, G: tl.constexpr, D: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
) -> None:
    #                            K^T
    #              | - - - - - -  T  - - - - - - |
    #               __ __ __ __ __ __ __ __ __ __
    #            D |  |  |  |  |  |  |  |  |  |  |
    #              |__|__|__|__|__|__|__|__|__|__|
    #                     N
    #                     |
    #        Q            |                            V              out
    # _ _   ___           |                           ___             ___
    #  |   |___|          |                          |___|           |___|
    #  |   |___|          |                          |___|           |___|
    #      |___|          |             - - - - -  N |___| - -       |___|
    # T_q  |___|          |            |             |___|    |      |___|
    #      |___|          |            |             |___|    |      |___|
    #  | M |___| - - - - |__| - - - - -              |___|     - - M |___|
    # _|_  |___|         M x N                       |___|           |___|
    #        D                                       |___|             D
    #                                                |___|
    #                                                |___|
    #                                                  D

    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_h_kv = pid_h // G
    pid_i = tl.program_id(2)
    offs_i = pid_i * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    scale: tl.constexpr = 1.0 / float(D)**0.5
    # [BLOCK_M, D]
    Q_i = tl.load(
        q_ptr
        + pid_b * stride_Q_B
        + offs_i[:, None] * stride_Q_T
        + pid_h * stride_Q_H
        + offs_d[None, :] * stride_Q_D,
        mask=(offs_i[:, None] < T_q),
        other=0.0
    )
    # [BLOCK_M, D]
    out = tl.load(
        out_ptr
        + pid_b * stride_out_B
        + offs_i[:, None] * stride_out_T
        + pid_h * stride_out_H
        + offs_d[None, :] * stride_out_D,
        mask=(offs_i[:, None] < T_q),
        other=0.0
    )
    # [BLOCK_M]
    l = tl.zeros((BLOCK_M,), Q_i.dtype)
    # [BLOCK_M]
    m = tl.full((BLOCK_M,), -float("inf"), Q_i.dtype)
    for j in range(0, T, BLOCK_N):
        offs_j = j + tl.arange(0, BLOCK_N)
        # [D, BLOCK_N]
        K_j_T = tl.load(
            k_ptr
            + pid_b * stride_K_B
            + offs_j[None, :] * stride_K_T
            + pid_h_kv * stride_K_H
            + offs_d[:, None] * stride_K_D,
            mask=(offs_j[None, :] < T),
            other=0.0
        )
        # [BLOCK_N, D]
        V_j = tl.load(
            v_ptr
            + pid_b * stride_V_B
            + offs_j[:, None] * stride_V_T
            + pid_h_kv * stride_V_H
            + offs_d[None, :] * stride_V_D,
            mask=(offs_j[:, None] < T),
            other=0.0
        )

        # Compute the raw attention scores.
        # [BLOCK_M, BLOCK_N]
        S_ij = tl.dot(Q_i, K_j_T) * scale
        
        # Load and apply bias to attention scores.
        # [BLOCK_M, BLOCK_N]
        bias_ij = tl.load(
            bias_ptr
            + pid_b * stride_bias_B
            + offs_i[:, None] * stride_bias_T_q
            + offs_j[None, :] * stride_bias_T,
            mask=(offs_i[:, None] < T_q) & (offs_j[None, :] < T),
            other=0.0
        )
        S_ij += bias_ij

        # The current normalizer value is:
        #   l = rowsum(P_i0) + rowsum(P_i1) + ... + rowsum(P_ij-1)
        # Destabilize by removing the previous max score.
        l *= tl.exp(m)
        # The current output value is:
        #   (P_i0 x V_0 + P_i1 x V_1 + ... + P_ij-1 x V_j-1) / l.
        # Denormalize and destabilize.
        out *= l[:, None]

        # Update the max score.
        m = tl.maximum(m, tl.max(S_ij, axis=1))
        # Compute the unnormalized, probability. The max score is
        # subtracted for stability.
        P_ij = tl.exp(S_ij - m[:, None])

        inv_m_exp = tl.exp(-m)
        # Add the latest unnormalized probability.
        # l = e^(-m') * e^m * l + rowwsum(P_ij)
        l *= inv_m_exp
        l += tl.sum(P_ij, axis=1)
        # Add the latest block product.
        # out = (e^(-m') * e^m * l * out + P_ij x V_j) / l
        out *= inv_m_exp[:, None]
        out += tl.dot(P_ij, V_j)
        out /= l[:, None]

    tl.store(
        out_ptr
        + pid_b * stride_out_B
        + offs_i[:, None] * stride_out_T
        + pid_h * stride_out_H
        + offs_d[None, :] * stride_out_D,
        out,
        mask=(offs_i[:, None] < T_q),
    )
    
    
def flash_attention(
    # [B, T_q, num_q_heads, head_dim]
    q: torch.Tensor,
    # [B, T, num_kv_heads, head_dim]
    k: torch.Tensor,
    # [B, T, num_kv_heads, head_dim]
    v: torch.Tensor,
    # [B, T_q, T]
    bias: torch.Tensor,
) -> torch.Tensor:
    assert k.shape == v.shape
    B, T_q, num_q_heads, head_dim = q.shape
    _, T, num_kv_heads, _ = k.shape
    num_groups = num_q_heads // num_kv_heads    
    
    BLOCK_M = 16
    BLOCK_N = 16
    grid = (
        # One program per request in batch.
        B,
        # One program per head.
        num_q_heads,
        # One program per tile along query dimension.
        (T_q + BLOCK_M - 1) // BLOCK_M,
    )
    
    out = torch.zeros_like(q)
    flash_attention_kernel[grid](
        q, k, v, bias, out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        bias.stride(0), bias.stride(1), bias.stride(2),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        T_q, T, num_groups, head_dim,
        BLOCK_M, BLOCK_N,
    )
    return out
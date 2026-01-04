import asyncio
import math
import pytest

import torch

from gllm.ops.attention.flash_attention import flash_attention
from gllm.ops.attention.reference_attention import reference_attention


@pytest.mark.asyncio
@pytest.mark.parametrize("B, T_q, T, H_q, H_kv, D", [
    (
        1, 8, 8, 1, 1, 128
    ),
    (
        1, 16, 16, 1, 1, 128
    ),
    (
        1, 40, 40, 1, 1, 128
    ),
    (
        1, 256, 256, 1, 1, 128
    ),
    (
        1, 32, 256, 1, 1, 128
    ),
    (
        1, 32, 256, 2, 1, 128
    ),
    (
        8, 32, 256, 1, 1, 128
    ),
])
async def test_flash_attention_correctness(
    B: int,
    T_q: int,
    T: int,
    H_q: int,
    H_kv: int,
    D: int,
):
    q_shape = (B, T_q, H_q, D)
    kv_shape = (B, T, H_kv, D)

    dtype = torch.float32
    device = "cuda"
    
    # (B, T_q, H, D)
    q = torch.randn(*q_shape, dtype=dtype, device=device)
    # (B, T, H_kv, D)
    k = torch.randn(*kv_shape, dtype=dtype, device=device)
    v = torch.randn(*kv_shape, dtype=dtype, device=device)
    
    # [B, T_q, T]
    bias = torch.zeros((B, T_q, T), dtype=dtype, device=device)
    # Set causal attention bias for query.
    # [B, T_q, T_q]
    query_bias = bias[:, :, (T - T_q):]
    query_bias.fill_(float("-inf"))
    query_bias.triu_(diagonal=1)
    
    fa_out = flash_attention(q, k, v, bias)
    ref_out = reference_attention(q, k, v, bias)
    assert torch.allclose(fa_out, ref_out, rtol=1e-3, atol=1e-3), f"mean diff: {(fa_out - ref_out).abs().mean()}, max diff: {(fa_out - ref_out).abs().max()}"
import asyncio
import math
import pytest

import torch

from gllm.config.generator_config import GeneratorConfig
from gllm.model.model import Model
from gllm.ops.attention.flash_attention import flash_attention
from gllm.ops.attention.reference_attention import reference_attention


@pytest.mark.asyncio
@pytest.mark.parametrize("B, T", [
    (
        1, 16
    ),
])
async def test_model_backward(
    B: int,
    T: int,
):
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    
    # Create model.
    model = Model(
        hf_model=HuggingFaceModel.LLAMA_3_2_1B_INSTUCT,
        gen_config=GeneratorConfig(
            block_size=16,
            max_batch_size=8,
            max_seq_len=256,
        ),
        device="cuda",
    )
    model.training = True
    
    # [B, T]
    input_ids  = torch.randint(0, vocab_size, (B, T))
    target_ids = torch.randint(0, vocab_size, (B, T))
    
    # [B]
    seq_lens = torch.tensor([T] * B, dtype=torch.int32, device=model.device)
    
    # Calculate token positions.
    # [T]
    positions = torch.arange(T, device=model.device)
    # [B, T]
    positions = positions.unsqueeze(0).expand(B, -1)
    
    # Calculate causal attention bias.
    # [B, T, T]
    bias = torch.full(
        (B, T, T),
        float("-inf"),
        dtype=model.dtype,
        device=model.device,
    )
    bias.triu_(diagonal=1)
    
    attention_metadata = AttentionMetadata(
        positions=positions,
        query_lens=seq_lens_tensor,
        seq_lens=seq_lens_tensor,
        bias=bias,
        # No KV caching for training.
        block_table=None,
        slot_mapping=None,
        query_slot_mapping=None,
    )
    
    eps = 1e-5
    for param in model.parameters():
        W = params.weights
        
        # L(W + eps)
        params.weights = W + eps
        logits = model.forward(
            input_ids,
            attention_metadata,
        )
        loss_plus_eps = loss_fn.forward(
            logits,
            target_ids,
        )
        
        # L(W - eps)
        params.weights = W - eps
        logits = model.forward(
            input_ids,
            attention_metadata,
        )
        loss_minus_eps = loss_fn.forward(
            logits,
            target_ids,
        )
        
        # dL/dW = (L(W + eps) - (L - eps)) / (2 * eps)
        expected_grad = (loss_plus_eps - loss_minus_eps) / (2 * eps)
        
        # Run model forward pass with original weights.
        params.weights = W
        logits = model.forward(
            input_ids,
            attention_metadata,
        )
        # Run loss forward & backward to get dL/dy
        loss_fn.forward(
            logits,
            target_ids,
        )
        dL_dy = loss_fn.backward()
        # Run model backward pass to compute parameter gradients.
        model.backward(dL_dy)
        
        # Compare expected numeric gradient with actual from backward pass.
        assert expected_grad == param.grad
        
        # Zero all grads.
        for p in self.params:
            p.grad = None
    
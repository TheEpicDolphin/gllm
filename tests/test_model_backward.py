import math
import pytest

import torch

from gllm.model.layers.attention import AttentionMetadata
from gllm.model.model import Model, HuggingFaceModel
from gllm.training.loss.cross_entropy_loss import CrossEntropyLoss


@pytest.mark.parametrize("B, T", [
    (
        1, 16
    ),
])
def test_model_backward_correctness(
    B: int,
    T: int,
):
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    
    # Create model.
    model = Model(
        hf_model=HuggingFaceModel.LLAMA_3_2_1B_INSTUCT,
        max_seq_len=1024,
        device="cpu",
        dtype="float64",
    )
    model.training = True
    
    # [B, T]
    input_ids  = torch.randint(0, model.vocab_size, (B, T), device=model.device)
    target_ids = torch.randint(0, model.vocab_size, (B, T), device=model.device)
    
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
        query_lens=seq_lens,
        seq_lens=seq_lens,
        bias=bias,
        # No KV caching for training.
        block_table=None,
        slot_mapping=None,
        query_slot_mapping=None,
    )
    
    loss_fn = CrossEntropyLoss()
    
    # Disable gradients tracking for all parameters.
    for p in model.parameters():
        p.requires_grad = False
    
    eps = 1e-3
    for param in model.parameters():
        W = param.weights
        # Create a random direction to perturb the weights along.
        u = torch.randn_like(param.weights)
        u /= u.norm()
        
        # Enable gradients tracking for this parameter.
        param.requires_grad = True
        # Run model forward pass with original weights.
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
        # Compute magnitude of gradient along the random direction.
        actual_grad = torch.sum(param.grad * u).item()
        # Zero this parameter's gradients.
        param.requires_grad = False
        param.grad = None
        
        # Caculate approximate numerical gradient magnitude along the random direction.
        # L(W + eps)
        #param.weights += eps * u
        param.weights = W + eps * u
        logits = model.forward(
            input_ids,
            attention_metadata,
        )
        loss_plus_eps = loss_fn.forward(
            logits,
            target_ids,
        ).item()
        # L(W - eps)
        #param.weights -= 2 * eps * u
        param.weights = W - eps * u
        logits = model.forward(
            input_ids,
            attention_metadata,
        )
        loss_minus_eps = loss_fn.forward(
            logits,
            target_ids,
        ).item()
        # dL/dW ~ (L(W_i + eps) - L(W_i - eps)) / (2 * eps)
        numerical_grad = (loss_plus_eps - loss_minus_eps) / (2 * eps)
        
        # Compare approximate numerical gradient with actual gradient from backward pass.
        abs_err = abs(numerical_grad - actual_grad)
        rel_err = abs_err / abs(numerical_grad)
        assert abs_err < 1e-4 and rel_err < 1e-4
    
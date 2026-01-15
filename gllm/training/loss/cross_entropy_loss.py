import torch
import torch.nn.functional as F


class CrossEntropyLoss:
    def __init__(self):
        self.logits = None
        self.target_ids = None
        self.completion_mask = None


    def forward(
        self,
        # [B, T, vocab_size]
        logits,
        # [B, T]
        target_ids,
        # [B, T]
        completion_mask: torch.Tensor | None = None,
    ) -> float:
        self.logits = logits
        self.target_ids = target_ids
        self.completion_mask = completion_mask

        B, T, vocab_size = logits.shape
        # [B * T, vocab_size]
        logits_flat = logits.view(-1, vocab_size)
        # [B * T]
        target_ids_flat = target_ids.view(-1)
        
        # L = -(1/N) * sum(y_i * log(p_i))
        # One-hot y is zero for i != target.
        log_probs = F.log_softmax(logits_flat, dim=-1)
        loss = -log_probs[range(B*T), target_ids_flat]
        
        if completion_mask is not None:
            # Ignore prompt tokens during loss calculation.
            loss = loss[completion_mask.view(-1)]
        
        return torch.mean(loss)
        

    def backward(self) -> torch.Tensor:
        B, T, vocab_size = self.logits.shape
        probs = F.softmax(self.logits, dim=-1)
        
        # Theory:
        # dp_i/dz_j = (i == j) * p_i - p_i * p_j
        # dL/dz_i = dL/dp_1 * dp_1/dz_i + dL/dp_2 * dp_2/dz_i + ... + dL/dp_V * dp_V/dz_i
        # dL/dz_j = -(1/N) * ((y_1/p_1) * dp_1/dz_j + (y_2/p_2) * dp_2/dz_j + ... + (y_V/p_V) * dp_V/dz_j)
        # dL/dz_j = (1/N) * (p_j - y_j)
        # One-hot y is 1 for j == target, and zero otherwise.
        target_one_hot = F.one_hot(self.target_ids, num_classes=vocab_size).float()
        dL_dy = probs - target_one_hot
        
        N = B * T
        if self.completion_mask is not None:
            # Mask out loss for prompt tokens.
            dL_dy[~self.completion_mask] = 0
            # Only count completion tokens.
            N = self.completion_mask.sum()
        
        # Normalize by number of tokens.
        dL_dy /= N
        return dL_dy

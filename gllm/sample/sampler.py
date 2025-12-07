from typing import NamedTuple

import torch
import torch.nn.functional as F

from gllm.sample.sampling_metadata import SamplingMetadata


class SamplerOutput(NamedTuple):
    # [B]
    sampled_token_ids: torch.Tensor
    # [B, num_logprobs]
    logprobs: list[list[float]] | None


class Sampler:
    def __init__(
        self,
        max_batch_size: int,
        device: torch.device,
    ):
        self.device = device
    
    
    def apply_temperature(
        self,
        # [B, 1, vocab_size]
        logits: torch.Tensor,
        # [B]
        temp: torch.Tensor,
    ) -> torch.Tensor:
        return logits.div_(temp.view(-1, 1))
    

    def sample_top_k(
        self,
        # [B, vocab_size]
        logits: torch.Tensor,
        # [B]
        top_k: torch.Tensor,
    ) -> torch.Tensor:
        B, vocab_size = logits.shape
        # If top_k is 0, sample from the entire vocabulary.
        top_k[top_k == 0] = vocab_size
        # NOTE: We are using the max K in the batch as the last dimension.
        # This is done for simplicity, and means that we may get more than
        # k top logits for some requests.
        max_top_k = top_k.max()
        top_k_logits, top_k_indices = torch.topk(logits, max_top_k, dim=-1, sorted=True)
        # Ensure that probs past k for each request will be zero.
        idxs = torch.arange(max_top_k, device=self.device).expand(B, -1)
        mask = idxs >= top_k.unsqueeze(1)
        top_k_logits[mask] = float("-inf")
        return top_k_logits, top_k_indices
    

    def sample_top_p(
        self,
        # [B, K_max]
        sorted_logits: torch.Tensor,
        # [B]
        top_p: torch.Tensor,
    ) -> torch.Tensor:
        # [B, K_max]
        probs = F.softmax(sorted_logits, dim=-1)
        # [B, K_max]
        cum_probs = torch.cumsum(probs, dim=-1)
        top_p_mask = cum_probs > top_p.unsqueeze(1)
        # Always accepted top-1.
        top_p_mask[:, 0] = False
        probs[top_p_mask] = 0.0
        # Renormalize.
        return probs.div_(probs.sum(dim=-1, keepdim=True))

    
    def forward(
        self,
        # [B, vocab_size]
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> SamplerOutput:
        B, _ = logits.shape
        # Use float32 for the logits.
        logits = logits.to(torch.float32)
        # Apply temperature.
        logits = self.apply_temperature(logits, sampling_metadata.temperature)
        # Sample top-k first (more efficient).
        # [B, K_max], [B, K_max]
        top_k_logits, top_k_token_ids = self.sample_top_k(logits, sampling_metadata.top_k)
        # Sample top-p.
        # [B, K_max]
        probs = self.sample_top_p(top_k_logits, sampling_metadata.top_p)
        # Sample a token for each request.
        sampled_indices = torch.multinomial(probs, num_samples=1)
        sampled_token_ids = top_k_token_ids.gather(1, sampled_indices).view(-1)
        return SamplerOutput(
            sampled_token_ids=sampled_token_ids,
            logprobs=None,
        )
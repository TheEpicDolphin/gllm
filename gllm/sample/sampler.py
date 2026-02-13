from typing import NamedTuple

import torch
import torch.nn.functional as F

from gllm.sample.sampling_metadata import SamplingMetadata
    

class SamplerOutput(NamedTuple):
    # [B, T_q]
    sampled_token_ids: torch.Tensor
    # [B, T_q, top_logprobs]
    top_logprobs: torch.Tensor
    # [B, T_q, top_logprobs]
    top_logprobs_token_ids: torch.Tensor


class Sampler:
    def __init__(
        self,
        max_batch_size: int,
        device: torch.device,
    ):
        self.device = device
    
    
    def sample_top_k(
        self,
        # [B, T_q, vocab_size]
        logits: torch.Tensor,
        # [B]
        top_k: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, T_q, vocab_size = logits.shape
        # If top_k is 0, sample from the entire vocabulary.
        top_k[top_k == 0] = vocab_size
        # NOTE: We are using the max K in the batch as the last dimension.
        # This is done for simplicity, and means that we may get more than
        # k top logits for some requests.
        max_top_k = top_k.max()
        top_k_logits, top_k_token_ids = torch.topk(logits, k=max_top_k, dim=-1, sorted=True)
        # Ensure that probs past k for each request will be zero.
        top_k_idxs = torch.arange(max_top_k, device=self.device).expand(B, T_q, -1)
        mask = top_k_idxs >= top_k.view(-1, 1, 1)
        top_k_logits[mask] = float("-inf")
        return top_k_logits, top_k_token_ids
    

    def apply_top_p(
        self,
        # [B, T_q, K_max]
        sorted_logits: torch.Tensor,
        # [B]
        top_p: torch.Tensor,
    ) -> None:
        # [B, T_q, K_max]
        probs = F.softmax(sorted_logits, dim=-1)
        # [B, T_q, K_max]
        cum_probs = torch.cumsum(probs, dim=-1)
        top_p_mask = cum_probs <= top_p.view(-1, 1, 1)
        # Always accepted top-1.
        top_p_mask[:, :, 0] = True
        # Only keep the top-p logits.
        sorted_logits[~top_p_mask] = float("-inf")

    
    def forward(
        self,
        # [B, T_q, vocab_size]
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> SamplerOutput:
        B, T_q, _ = logits.shape
        # Use float32 for the logits.
        raw_logits = logits.to(torch.float32)

        # Compute top logprobs before temperature/top-k/top-p are applied.
        logprobs = F.log_softmax(raw_logits, dim=-1)
        # Use the maximum number of top logprobs for the whole batch.
        num_top_logprobs = sampling_metadata.num_top_logprobs.max()
        top_logprobs, top_logprobs_token_ids = torch.topk(
            logprobs,
            k=num_top_logprobs,
            dim=-1,
            sorted=True
        )

        # Apply temperature.
        processed_logits = raw_logits / sampling_metadata.temperature.view(-1, 1, 1)
        # Sample top-k first (more efficient).
        # [B, T_q, K_max], [B, T_q, K_max]
        top_k_logits, top_k_token_ids = self.sample_top_k(processed_logits, sampling_metadata.top_k)
        # Sample top-p.
        self.apply_top_p(top_k_logits, sampling_metadata.top_p)
        # Compute probs from logits.
        # [B, T_q, K_max]
        probs = F.softmax(top_k_logits, dim=-1)
        # Sample a token for each request.
        # [B * T_q, 1]
        sampled_idxs = torch.multinomial(probs.view(B * T_q, -1), num_samples=1)
        # [B, T_q, 1]
        sampled_idxs = sampled_idxs.view(B, T_q, 1)
        # [B, T_q]
        sampled_token_ids = torch.gather(top_k_token_ids, dim=2, index=sampled_idxs).squeeze(-1)
        
        return SamplerOutput(
            sampled_token_ids=sampled_token_ids,
            top_logprobs=top_logprobs,
            top_logprobs_token_ids=top_logprobs_token_ids,
        )
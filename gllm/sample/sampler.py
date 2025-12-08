from typing import NamedTuple

import torch
import torch.nn.functional as F

from gllm.sample.logprobs import TokenLogProbs
from gllm.sample.sampling_metadata import SamplingMetadata
    

class SamplerOutput(NamedTuple):
    # [B]
    sampled_token_ids: torch.Tensor
    # [B]
    logprobs: list[TokenLogProbs]


class Sampler:
    def __init__(
        self,
        max_batch_size: int,
        device: torch.device,
    ):
        self.device = device
    
    
    def sample_top_k(
        self,
        # [B, vocab_size]
        logits: torch.Tensor,
        # [B]
        top_k: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, vocab_size = logits.shape
        # If top_k is 0, sample from the entire vocabulary.
        top_k[top_k == 0] = vocab_size
        # NOTE: We are using the max K in the batch as the last dimension.
        # This is done for simplicity, and means that we may get more than
        # k top logits for some requests.
        max_top_k = top_k.max()
        top_k_logits, top_k_token_ids = torch.topk(logits, max_top_k, dim=-1, sorted=True)
        # Ensure that probs past k for each request will be zero.
        top_k_idxs = torch.arange(max_top_k, device=self.device).expand(B, -1)
        mask = top_k_idxs >= top_k.unsqueeze(1)
        top_k_logits[mask] = float("-inf")
        return top_k_logits, top_k_token_ids, top_k_idxs
    

    def apply_top_p(
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
        top_p_mask = cum_probs <= top_p.unsqueeze(1)
        # Always accepted top-1.
        top_p_mask[:, 0] = True
        # Only keep the top-p logits.
        sorted_logits[~top_p_mask] = float("-inf")
        return top_p_mask.sum(dim=-1)

    
    def forward(
        self,
        # [B, vocab_size]
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> SamplerOutput:
        B, _ = logits.shape
        # Use float32 for the logits.
        raw_logits = logits.to(torch.float32)
        # Apply temperature.
        processed_logits = raw_logits / sampling_metadata.temperature.view(-1, 1)
        # Sample top-k first (more efficient).
        # [B, K_max], [B, K_max], [B, K_max]
        top_logits, top_k_token_ids, top_k_idxs = self.sample_top_k(processed_logits, sampling_metadata.top_k)
        # Sample top-p.
        # [B]
        num_candidates = self.apply_top_p(top_logits, sampling_metadata.top_p)
        # Compute probs from logits.
        probs = F.softmax(top_logits, dim=-1)
        # Sample a token for each request.
        sampled_idxs = torch.multinomial(probs, num_samples=1)
        sampled_token_ids = top_k_token_ids.gather(1, sampled_idxs).view(-1)
        # Compute logprobs.
        logprobs = F.log_softmax(raw_logits, dim=-1)
        top_k_logprobs = logprobs.gather(1, top_k_token_ids)
        num_logprobs = torch.min(sampling_metadata.max_num_logprobs, num_candidates)
        num_logprobs_list = num_logprobs.tolist()
        logprobs_mask = top_k_idxs < num_logprobs.unsqueeze(1)
        logprob_values = torch.split(
            top_k_logprobs[logprobs_mask],
            num_logprobs_list,
        )
        logprob_token_ids = torch.split(
            top_k_token_ids[logprobs_mask],
            num_logprobs_list,
        )
        
        return SamplerOutput(
            sampled_token_ids=sampled_token_ids,
            logprobs=list(zip(logprob_values, logprob_token_ids)),
        )
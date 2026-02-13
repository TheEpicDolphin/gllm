from typing import NamedTuple

import torch
import torch.nn.functional as F

from gllm.sample.spec_decode.sbd.sampling_metadata import SamplingMetadata
    

class SamplerOutput(NamedTuple):
    # [B, T_q]
    sampled_token_ids: torch.Tensor


class Sampler:
    def __init__(
        self,
        mask_token_id: int,
    ):
        self.mask_token_id = mask_token_id

    
    def forward(
        self,
        # [B, T_q, vocab_size]
        logits: torch.Tensor,
        # [B, T_q]
        query_token_ids: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> SamplerOutput:
        # Use float32 for the logits.
        logits = logits.to(torch.float32)
        # [B, T_q, vocab_size]
        probs = F.softmax(logits, dim=-1)
        # [B, T_q]
        sampled_token_ids = torch.argmax(probs, dim=-1)
        # [B, T_q, vocab_size]
        logprobs = F.log_softmax(logits, dim=-1)
        # Compute entropy: H(p) = -sum(p * log(p))
        # [B, T_q]
        entropies = -torch.sum(probs * logprobs, dim=-1)

        # Get the mask of currently revealed tokens.
        # [B, T_q]
        revealed_token_mask = query_token_ids != self.mask_token_id
        # Overwrite the sampled tokens with the currently revealed tokens.
        sampled_token_ids[revealed_token_mask] = query_token_ids[revealed_token_mask]
        # Zero out entropy for revealed tokens, ensuring that they never
        # exceed the entropy threshold.
        entropies[revealed_token_mask] = 0
        entropies[revealed_token_mask.nonzero()] = 0

        # Sort the entropies in ascending order.
        # [B, T_q], [B, T_q]
        sorted_entropies, indices = entropies.sort(dim=-1, descending=False)

        # Compute cumulative sum of sorted entropies. All tokens
        # exceeding the threshold are rejected.
        # [B, T_q]
        sorted_entropies_cumsum = sorted_entropies.cumsum(dim=-1)
        # [B, T_q]
        rejected_mask = sorted_entropies_cumsum[indices] > sampling_metadata.entropy_thresholds

        # Mask out rejected tokens.
        sampled_token_ids[rejected_mask] = self.mask_token_id
        return SamplerOutput(sampled_token_ids)
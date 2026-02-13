from dataclasses import dataclass

import torch


@dataclass
class SamplingMetadata:
    # [B]
    entropy_thresholds: torch.Tensor
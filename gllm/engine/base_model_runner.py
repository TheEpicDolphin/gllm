from abc import ABC, abstractmethod

import torch

from gllm.engine.batch_inputs import BatchInputs
from gllm.model.kv_cache.paged_kv_cache import PagedKVCache
from gllm.sample.logprobs import TokenLogProbs


class BaseModelRunner(ABC):
    @abstractmethod
    def prefill_step(
        self,
        batch: BatchInputs,
    ) -> tuple[torch.Tensor, list[TokenLogProbs]]:
        raise NotImplementedError


    @abstractmethod
    def decode_step(
        self,
        batch: BatchInputs,
    ) -> tuple[torch.Tensor, list[TokenLogProbs]]:
        raise NotImplementedError
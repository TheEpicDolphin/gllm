from abc import ABC, abstractmethod

from gllm.engine.batch_inputs import BatchInputs
from gllm.sample.sampler import SamplerOutput


class BaseModelRunner(ABC):
    @abstractmethod
    def prefill_step(
        self,
        batch: BatchInputs,
    ) -> SamplerOutput:
        raise NotImplementedError


    @abstractmethod
    def decode_step(
        self,
        batch: BatchInputs,
    ) -> SamplerOutput:
        raise NotImplementedError
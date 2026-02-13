from abc import ABC, abstractmethod

from gllm.engine.batch_inputs import BatchInputs
from gllm.engine.decode_output import DecodeOutput
from gllm.engine.prefill_output import PrefillOutput


class BaseModelRunner(ABC):
    @abstractmethod
    def prefill_step(
        self,
        batch: BatchInputs,
    ) -> PrefillOutput:
        raise NotImplementedError


    @abstractmethod
    def decode_step(
        self,
        batch: BatchInputs,
    ) -> DecodeOutput:
        raise NotImplementedError
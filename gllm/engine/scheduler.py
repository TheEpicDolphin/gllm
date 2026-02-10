import torch

from gllm.config.generator_config import GeneratorConfig
from gllm.engine.base_scheduler import BaseScheduler
from gllm.engine.llm_engine_base import TokenLogProbs
from gllm.model.model import Model
from gllm.sample.logprobs import TokenLogProbs


class Scheduler(BaseScheduler):
    def __init__(
        self,
        model: Model,
        gen_config: GeneratorConfig,
        device: str,
    ):
        super().__init__(
            model,
            gen_config,
            device,
        )
    

    def update(
        self,
        # [B, T_q]
        sampled_token_ids: torch.Tensor,
        sampled_logprobs: list[TokenLogProbs],
        req_offset: int = 0,
    ) -> None:
        super().update(
            sampled_token_ids,
            sampled_logprobs,
            req_offset,
        )

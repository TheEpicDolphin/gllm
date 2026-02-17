import torch

from gllm.engine.config import EngineConfig
from gllm.engine.base_scheduler import BaseScheduler
from gllm.engine.batch_inputs import BatchInputs
from gllm.engine.decode_output import DecodeOutput
from gllm.engine.prefill_output import PrefillOutput
from gllm.model.model import Model


class Scheduler(BaseScheduler):
    def __init__(
        self,
        model: Model,
        engine_config: EngineConfig,
        device: str,
    ):
        self.sbd_config = engine_config.sbd_config
        super().__init__(model, engine_config, device)
    

    def prepare_decode_batch(self) -> BatchInputs:
        # Temporarily augment sequence lengths by the block size so
        # that sufficient paged KV cache blocks are allocated.
        self.req_states.seq_lens[:self.batch_size] += self.sbd_config.block_size
        batch = super().prepare_decode_batch()
        self.req_states.seq_lens[:self.batch_size] -= self.sbd_config.block_size
        return batch
    

    def prefill_update(
        self,
        prefill_output: PrefillOutput,
        prefill_start_idx: int = 0,
    ) -> None:
        seq_lens = self.req_states.seq_lens[prefill_start_idx:self.batch_size]
        token_ids = self.req_states.token_ids[prefill_start_idx:self.batch_size]

        # Set all tokens after the prompt to mask tokens.
        # [1, T_max]
        positions = self.arange[:self.max_seq_len].unsquueze(0)
        # [B, T_max]
        gen_mask = positions > seq_lens.unsqueeze(1)
        token_ids[gen_mask] = self.model.config.mask_token_id

        super().prefill_update(prefill_output, prefill_start_idx=prefill_start_idx)


    def decode_update(
        self,
        decode_output: DecodeOutput,
    ) -> None:
        self._write_step_outputs(*decode_output)
        # Set entire row to dummy token ids if any masked token id still exists in the block.
        # This skips any actions for those requests.
        output_token_ids = decode_output.token_ids
        mask_token_mask = output_token_ids == self.model.config.mask_token_id
        any_mask_token_mask = mask_token_mask.any(dim=-1)
        output_token_ids[any_mask_token_mask, :] = -1
        self._finalize_step(output_token_ids)


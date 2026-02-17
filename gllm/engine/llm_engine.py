import asyncio
import queue
import time
from concurrent.futures import Future as ConcurrentFuture
from typing import NamedTuple, Type

from gllm.engine.config import EngineConfig, SetBlockDecoderConfig
from gllm.engine.llm_engine_base import GenerationRequest, GenerationResult, LLMEngineBase
from gllm.engine.base_model_runner import BaseModelRunner
from gllm.engine.base_scheduler import BaseScheduler
from gllm.engine.utils import complete_future_threadsafe
from gllm.model.model import Model


class QueuedRequest(NamedTuple):
    gen_req: GenerationRequest
    future: asyncio.Future
    timestamp: float


class LLMEngine(LLMEngineBase):
    def __init__(
        self,
        model_path: str,
        engine_config: EngineConfig,
        device: str,
    ):
        self.alive = False

        # Create the model.
        model = Model.from_path(
            model_path,
            device,
            max_seq_len=engine_config.max_seq_len,
            dtype_override=engine_config.model_dtype,
            offload_device=engine_config.offload_device,
        )

        # Create model runner and scheduler.
        runner_cls, scheduler_cls = self._get_runner_and_scheduler_cls(engine_config)
        self.runner = runner_cls(
            model=model,
            engine_config=engine_config,
            device=device,
        )
        self.scheduler = scheduler_cls(
            model=model,
            engine_config=engine_config,
            device=device,
        )

        self.max_queue_size = engine_config.max_queue_size
        self.request_queue = queue.Queue(self.max_queue_size)


    @staticmethod
    def _get_runner_and_scheduler_cls(
        engine_config: EngineConfig,
    ) -> tuple[Type[BaseModelRunner], Type[BaseScheduler]]:
        if engine_config.spec_decode_config is None:
            from gllm.engine.model_runner import ModelRunner
            from gllm.engine.scheduler import Scheduler

            return ModelRunner, Scheduler
        elif isinstance(engine_config.spec_decode_config, SetBlockDecoderConfig):
            from gllm.engine.spec_decode.sbd.model_runner import ModelRunner as SBDModelRunner
            from gllm.engine.spec_decode.sbd.scheduler import Scheduler as SBDScheduler

            return SBDModelRunner, SBDScheduler
        else:
            raise NotImplementedError(f"Unsupported spec decode config of type: {type(engine_config.spec_decode_config)}")
        

    def _get_queued_requests(
        self,
        max_count: int,
        blocking: bool,
    ) -> list[QueuedRequest]:
        if max_count == 0:
            return []
        
        reqs = []
        if blocking:
            req = self.request_queue.get(block=True)
            reqs.append(req)
            self.request_queue.task_done()
            max_count -= 1
        
        while not self.request_queue.empty() and max_count > 0:
            req = self.request_queue.get_nowait()
            reqs.append(req)
            self.request_queue.task_done()
            max_count -= 1
        return reqs
    
    
    def enqueue_request(self, req: GenerationRequest) -> asyncio.Future:
        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        if not self.alive:
            complete_future_threadsafe(
                fut,
                exception=RuntimeError("Engine is not running.")
            )
            return fut
        
        try:
            self.request_queue.put_nowait(QueuedRequest(req, fut, time.time()))
        except queue.Full:
            complete_future_threadsafe(
                fut,
                exception=RuntimeError(f"Queue is at max capacity: {self.max_queue_size}.")
            )
        finally:
            return fut
                

    def run(self):
        self.alive = True
        while self.alive:
            in_progress = self.scheduler.num_active_requests > 0
            req_allowance = max(self.scheduler.max_batch_size - self.scheduler.num_active_requests, 0)
            # Get enqueued requests.
            queued_reqs = self._get_queued_requests(req_allowance, blocking=not in_progress)
            if len(queued_reqs) > 0:
                prefill_reqs, futures, _ = zip(*queued_reqs)
                # Prefill new requests.
                prefill_batch, prefill_start_idx = self.scheduler.prepare_prefill_batch(prefill_reqs, futures)
                if prefill_batch is not None:
                    prefill_output = self.runner.prefill_step(prefill_batch)
                    self.scheduler.prefill_update(prefill_output, prefill_start_idx=prefill_start_idx)
            # Decode step for all requests.
            decode_batch = self.scheduler.prepare_decode_batch()
            decode_output = self.runner.decode_step(decode_batch)
            self.scheduler.decode_update(decode_output)
    
    
    def stop(self):
        self.alive = False
        
    
    def generate(self, reqs: list[GenerationRequest]) -> list[GenerationResult]:
        # Run prefill step.
        futures = [ConcurrentFuture() for _ in reqs]
        prefill_batch, prefill_start_idx = self.scheduler.prepare_prefill_batch(reqs, futures)
        if prefill_batch is not None:
            prefill_output = self.runner.prefill_step(prefill_batch)
            self.scheduler.prefill_update(prefill_output, prefill_start_idx=prefill_start_idx)

        # Run decode steps until all requests are finished.
        while self.scheduler.num_active_requests > 0:
            decode_batch = self.scheduler.prepare_decode_batch()
            decode_output = self.runner.decode_step(decode_batch)
            self.scheduler.decode_update(decode_output)
        return [future.result() for future in futures]

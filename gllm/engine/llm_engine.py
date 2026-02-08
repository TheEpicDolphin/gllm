import asyncio
import queue

from gllm.config.generator_config import GeneratorConfig
from gllm.engine.llm_engine_base import GenerationRequest, GenerationResult, LLMEngineBase
from gllm.engine.model_runner import ModelRunner
from gllm.engine.scheduler import Scheduler


class LLMEngine(LLMEngineBase):
    def __init__(
        self,
        model_path: str,
        gen_config: GeneratorConfig,
        device: str,
    ):
        self.alive = False
        self.runner = ModelRunner(
            model_path=model_path,
            gen_config=gen_config,
            device=device,
        )
        self.scheduler = Scheduler(
            model=self.runner.model,
            gen_config=gen_config,
            device=device,
        )
        self.request_queue = queue.Queue()
    

    def _get_enqueued_requests(
        self,
        blocking: bool,
    ) -> tuple[list[GenerationRequest], list[asyncio.Future]]:
        reqs = []
        futures = []
        if blocking:
            req, future = self.request_queue.get(block=True)
            reqs.append(req)
            futures.append(future)
            self.request_queue.task_done()
        
        while not self.request_queue.empty():
            req, future = self.request_queue.get_nowait()
            reqs.append(req)
            futures.append(future)
            self.request_queue.task_done()
        return reqs, futures
    
    
    def enqueue_request(self, req: GenerationRequest) -> asyncio.Future:
        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        if not self.alive:
            fut.set_exception(RuntimeError("Engine is not running"))
            return fut
        
        self.request_queue.put((req, fut))
        return fut
                

    def run(self):
        self.alive = True
        while self.alive:
            # Get enqueued requests.
            in_progress = self.scheduler.has_active_requests
            enqueued_reqs, futures = self._get_enqueued_requests(block=not in_progress)
            if len(enqueued_reqs) > 0:
                # Prefill new requests.
                prefill_batch, prefill_start = self.scheduler.prepare_prefill_batch(enqueued_reqs, futures)
                if prefill_batch is not None:
                    sampled_token_ids, logprobs = self.runner.prefill_step(prefill_batch)
                    self.scheduler.update(sampled_token_ids, logprobs, req_offset=prefill_start)
            # Decode step for all requests.
            decode_batch = self.scheduler.prepare_decode_batch()
            sampled_token_ids, logprobs = self.runner.decode_step(decode_batch)
            self.scheduler.update(sampled_token_ids, logprobs)
    
    
    def stop(self):
        self.alive = False
        
    
    def generate(self, reqs: list[GenerationRequest]) -> list[GenerationResult]:
        from concurrent.futures import Future

        # Run prefill step.
        futures = [Future() for _ in reqs]
        prefill_batch, prefill_start = self.scheduler.prepare_prefill_batch(reqs, futures)
        if prefill_batch is not None:
            sampled_token_ids, logprobs = self.runner.prefill_step(prefill_batch)
            self.scheduler.update(sampled_token_ids, logprobs, req_offset=prefill_start)

        # Run decode steps until all requests are finished.
        while self.scheduler.has_active_requests:
            decode_batch = self.scheduler.prepare_decode_batch()
            sampled_token_ids, logprobs = self.runner.decode_step(decode_batch)
            self.scheduler.update(sampled_token_ids, logprobs)
        return [future.result() for future in futures]

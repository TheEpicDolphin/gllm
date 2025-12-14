import asyncio
import uuid
import queue
from dataclasses import dataclass

from gllm.config.generator_params import GeneratorParams
from gllm.engine.llm_engine_base import GenerationRequest, GenerationResult, LLMEngineBase, TokenLogProbs
from gllm.llm.llm import LLM
from gllm.model.model import HuggingFaceModel
from gllm.scheduler.request_state import RequestState


class LLMEngine(LLMEngineBase):
    def __init__(
        self,
        hf_model: HuggingFaceModel,
        gen_params: GeneratorParams,
        device: str,
        local_cache_dir: str | None = None,
    ):
        self.alive = False
        self.llm = LLM(
            hf_model=hf_model,
            gen_params=gen_params,
            device=device,
            local_cache_dir=local_cache_dir,
        )
        self.request_queue = queue.Queue()
        

    def _gen_unique_id(self):
        return uuid.uuid4().int
    

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
    
    
    def _schedule_requests(
        self,
        num_active_reqs: int,
        reqs: list[GenerationRequest],
        futures: list[asyncio.Future] | None = None
    ) -> list[RequestState]:
        scheduled_reqs = []
        if futures is None:
            futures = [None for req in reqs]
        for req, future in zip(reqs, futures):
            try:
                # Generate unique request id.
                uid = self._gen_unique_id()
                # Combine user and model stop token ids.
                user_stop_tokens = "".join(req.stop_tokens)
                user_stop_token_ids = self.llm.tokenize(user_stop_tokens)
                stop_token_ids = set(self.llm.eos_token_ids + user_stop_token_ids)
                prompt_token_ids = self.llm.tokenize(req.prompt)
                allowed_num_new_tokens = self.llm.max_seq_len - len(prompt_token_ids)
                if allowed_num_new_tokens <= 0:
                    raise ValueError(
                        f"Prompt has {len(prompt_token_ids)} tokens, but the engine only supports {self.llm.max_seq_len}."
                    )
                
                if req.max_new_tokens <= 0:
                    raise ValueError(
                        f"max_new_tokens must be >= 0."
                    )
                    
                if (num_active_reqs + len(scheduled_reqs)) == self.llm.max_batch_size:
                    raise RuntimeError(
                        f"Failed to enqueue request with error: Batch size limit ({self.llm.max_batch_size}) has been reached."
                    )

                scheduled_reqs.append(
                    RequestState(
                        id=uid,
                        prompt_token_ids=prompt_token_ids,
                        generated_token_ids=[],
                        generated_logprobs=[],
                        is_finished=False,
                        max_new_tokens=min(allowed_num_new_tokens, req.max_new_tokens),
                        stop_token_ids=stop_token_ids,
                        temperature=req.temperature,
                        top_k=req.top_k,
                        top_p=req.top_p,
                        max_num_logprobs=req.max_num_logprobs,
                        future=future,
                    )
                )
                
                print(f"[LLMEngine] received request '{uid}'.")
            except Exception as e:
                if future:
                    future.set_exception(e)
                else:
                    raise
                    
        return scheduled_reqs
                
        
        
    def _update_requests(
        self,
        reqs: list[RequestState],
        sampled_token_ids: list[int],
        logprobs: list[TokenLogProbs],
    ):
        for req, sampled_token_id, top_logprobs in zip(reqs, sampled_token_ids, logprobs):
            req.generated_token_ids.append(sampled_token_id)
            req.generated_logprobs.append(top_logprobs)
            req.is_finished = sampled_token_id in req.stop_token_ids \
                          or len(req.generated_token_ids) >= req.max_new_tokens
    

    def _create_result(
        self,
        req: RequestState,
    ):
        generated_text = self.llm.detokenize(req.generated_token_ids)
        result = GenerationResult(
            token_ids=req.generated_token_ids,
            logprobs=req.generated_logprobs,
            text=generated_text,
        )
        if req.future:
            req.future.set_result(result)
        return result
    
    
    def enqueue_request(self, req: GenerationRequest) -> asyncio.Future:
        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        if not self.alive:
            fut.set_exception(RuntimeError("Engine is not running"))
            return fut
        
        self.request_queue.put((req, fut))
        return fut
        

    def run(self):
        active_reqs = []
        self.alive = True
        while self.alive:
            # Get enqueued requests.
            in_progress = len(active_reqs) > 0
            enqueued_reqs, futures = self._get_enqueued_requests(block=not in_progress)
            
            # Schedule enqueued requests. Some may be rejected.
            new_reqs = self._schedule_requests(len(active_reqs), enqueued_reqs, futures)
            active_reqs.extend(new_reqs)
            
            input_batch = self.llm.prepare_batch(active_reqs)
            sampled_token_ids, logprobs = self.llm.decode_step(input_batch)
            self._update_requests(active_reqs, sampled_token_ids, logprobs)
            
            # Remove finished requests.
            for idx in range(len(active_reqs) - 1, -1, -1):
                req = active_reqs[idx]
                if req.is_finished:
                    # The request is finished. Output the result and remove.
                    result = self._create_result(req)
                    req.future.set_result(result)
                    print(f"[LLMEngine] completed request '{req.id}'.")
                    active_reqs.pop(idx)
    
    
    def stop(self):
        self.alive = False
        
    
    def generate(self, reqs: list[GenerationRequest]) -> list[GenerationResult]:
        results = [None for req in reqs]
        active_reqs = self._schedule_requests(0, reqs)
        index_map = {req.id: idx for idx, req in enumerate(active_reqs)}
        while len(active_reqs) > 0:
            input_batch = self.llm.prepare_batch(active_reqs)
            sampled_token_ids, logprobs = self.llm.decode_step(input_batch)
            self._update_requests(active_reqs, sampled_token_ids, logprobs)
            
            # Remove finished requests.
            for idx in range(len(active_reqs) - 1, -1, -1):
                req = active_reqs[idx]
                if req.is_finished:
                    # The request is finished. Set the result and remove.
                    result = self._create_result(req)
                    results[index_map[req.id]] = result
                    print(f"[LLMEngine] completed request '{req.id}'.")
                    active_reqs.pop(idx)
        return results

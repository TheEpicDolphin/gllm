import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from gllm.config.generator_params import GeneratorParams
from gllm.engine.llm_engine_base import GenerationResult, GenerationRequest, LLMEngineBase
from gllm.model.model import HuggingFaceModel


class HuggingFaceLLMEngine(LLMEngineBase):
    def __init__(
        self,
        hf_model: HuggingFaceModel,
        device: str,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(hf_model)
        self.model = AutoModelForCausalLM.from_pretrained(hf_model)
        self.model.to(device)
            
    
    def generate(self, reqs: list[GenerationRequest]) -> list[GenerationResult]:
        results = [None for req in reqs]
        for idx, req in enumerate(reqs):
            inputs = self.tokenizer(
                req.prompt,
                return_tensors="pt",
                add_special_tokens=True,
            ).to(self.model.device)
        
            prompt_ids = inputs["input_ids"][0]
            with torch.inference_mode():  # disable gradients
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=req.max_new_tokens,
                    temperature=req.temperature,
                    top_p=req.top_p,
                    top_k=req.top_k,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
                
            output_ids = output[0]
            generated_token_ids = output_ids[len(prompt_ids):].tolist()
            generated_text = self.tokenizer.decode(generated_token_ids)
            results[idx] = GenerationResult(
                token_ids=generated_token_ids,
                text=generated_text,
                logprobs=None,
            )
        return results
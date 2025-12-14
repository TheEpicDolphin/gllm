import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from gllm.engine.llm_engine_base import GenerationResult, GenerationRequest, LLMEngineBase
from gllm.model.model import HuggingFaceModel


class HuggingFaceLLMEngine(LLMEngineBase):
    def __init__(
        self,
        hf_model: HuggingFaceModel,
        device: str,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(hf_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(hf_model)
        self.model.to(device)
            
    
    def generate(self, reqs: list[GenerationRequest]) -> list[GenerationResult]:
        prompts = [req.prompt for req in reqs]
        
        # Tokenize as a batch.
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=True,
        ).to(self.model.device)
    
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        
        # NOTE: HF only supports uniform generation params per call
        max_new_tokens = max(req.max_new_tokens for req in reqs)
        temperature = reqs[0].temperature
        top_p = reqs[0].top_p
        top_k = reqs[0].top_k
    
        with torch.inference_mode():  # disable gradients
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            
        results = [None for req in reqs]
        for idx, req in enumerate(reqs):
            prompt_len = attention_mask[idx].sum().item()
            output_ids = outputs[idx]
            generated_token_ids = output_ids[prompt_len:].tolist()
            generated_text = self.tokenizer.decode(generated_token_ids)
            results[idx] = GenerationResult(
                token_ids=generated_token_ids,
                text=generated_text,
                logprobs=None,
            )
        return results
import json
import os

import torch
import torch.nn.functional as F
from enum import StrEnum
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from tokenizers import Tokenizer
from torch.profiler import record_function

from gllm.config.generator_config import GeneratorConfig
from gllm.config.model_config import ModelConfig
from gllm.model.kv_cache.paged_kv_cache import PagedKVCache
from gllm.model.layers.attention import AttentionMetadata
from gllm.model.layers.base_module import BaseModule
from gllm.model.layers.embedding import Embedding
from gllm.model.layers.linear import Linear
from gllm.model.layers.norm import RMSNorm
from gllm.model.layers.transformer_layer import TransformerLayer

CPU_DEVICE = "cpu"


class HuggingFaceModel(StrEnum):
    LLAMA_3_2_1B = "meta-llama/Llama-3.2-1B"
    LLAMA_3_2_1B_INSTUCT = "meta-llama/Llama-3.2-1B-Instruct"
    

class Model(BaseModule):
    def __init__(
        self,
        hf_model: HuggingFaceModel,
        gen_config: GeneratorConfig,
        device: str,
        local_cache_dir: str | None = None,
    ):
        super().__init__(None)
        
        if local_cache_dir is None:
            # Use default cache directory.
            local_cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "gllm")
            
        self.gen_config = gen_config
        self.device = torch.device(device)
        
        # Create directories for model and tokenizer files.
        model_cache_dir = os.path.join(local_cache_dir, "models")
        os.makedirs(model_cache_dir, exist_ok=True)
        tokenizer_cache_dir = os.path.join(local_cache_dir, "tokenizers")
        os.makedirs(tokenizer_cache_dir, exist_ok=True)

        # Download tokenizer.
        local_tokenizer_path = hf_hub_download(
            repo_id=hf_model,
            filename="tokenizer.json",
            cache_dir=tokenizer_cache_dir,
        )
        self.tokenizer = Tokenizer.from_file(local_tokenizer_path)

        # Download model config.
        local_config_path = hf_hub_download(
            repo_id=hf_model,
            filename="config.json"
        )
        with open(local_config_path, "r") as f:
            config = json.load(f)
        self.model_config = ModelConfig(
            dtype=getattr(torch, gen_config.model_dtype or config["torch_dtype"]),
            hidden_size=config["hidden_size"],
            head_dim=config["head_dim"],
            intermediate_size=config["intermediate_size"],
            act_func=config["hidden_act"],
            num_layers=config["num_hidden_layers"],
            num_attn_heads=config["num_attention_heads"],
            num_kv_heads=config["num_key_value_heads"],
            rms_norm_eps=config["rms_norm_eps"],
            eos_token_ids=self.parse_eos_token_ids(config),
            kv_dtype=getattr(torch, gen_config.kv_dtype or config["torch_dtype"]),
            rope_theta=config["rope_theta"],
        )
        
        # Download model safetensors.
        local_model_path = hf_hub_download(
            repo_id=hf_model,
            filename="model.safetensors",
            cache_dir=model_cache_dir,
        )
        # If cpu_offloading is true, weights are kept in CPU RAM and loaded to GPU only when needed.
        initial_weights_device = CPU_DEVICE if self.cpu_offloading else device
        safetensors = load_file(local_model_path, device=initial_weights_device)
        
        # Initialize paged KV cache.
        self.paged_kv_cache = PagedKVCache(
            model_config=self.model_config,
            gen_config=gen_config,
            device=device,
        )

        # Initialize transformer layers.
        self.layers: list[TransformerLayer] = []
        for layer_idx in range(self.model_config.num_layers):
            self.layers.append(TransformerLayer(
                layer_idx,
                model_config=self.model_config,
                safetensors=safetensors,
            ))
        
        # Initialize final norm.
        final_norm_weights = safetensors[f"model.norm.weight"].to(self.dtype)
        self.final_norm = RMSNorm(
            weights=final_norm_weights,
            eps=self.model_config.rms_norm_eps
        )
        
        # Get embedding matrix. This is indexed into using the token
        # ids to get the embedding vectors. It is also used as an LM
        # heead by multiplying with the hidden states to obtain the
        # logits.
        self.embedding_weights = safetensors["model.embed_tokens.weight"].to(self.dtype)
        self.embed = Embedding(self.embedding_weights)
        self.unembed = Linear(self.embedding_weights)
        
        # Construct RoPE sin/cos caches for positions up to T_max.
        # [T_max]
        p = torch.arange(gen_config.max_seq_len, device=device)
        # [head_dim // 2]
        m = torch.arange(self.head_dim // 2, device=device)
        theta_m = self.rope_theta**(-2 * m / self.head_dim)
        # [T_max, head_dim // 2]
        p_theta_m = p.unsqueeze(1) * theta_m
        # [T_max, head_dim // 2]
        self.cos_pos_cache = torch.cos(p_theta_m).to(self.dtype)
        # [T_max, head_dim // 2]
        self.sin_pos_cache = torch.sin(p_theta_m).to(self.dtype)
        
        if self.cpu_offloading:
            # Stream for preloading weights to device.
            self.transfer_stream = torch.cuda.Stream()
            # Allocate staging buffers using the first transformer layer.
            # All layers are expected to have the same weight shapes.
            self.transformer_layer_staging_buffers = self.layers[0].allocate_staging_buffers()
        
        self.child_modules = [
            self.embed,
            *self.layers,
            self.final_norm,
            self.unembed,
        ]
        
        
    @property
    def cpu_offloading(self) -> bool:
        return self.device != "cpu" and self.gen_config.cpu_offloading

    @property
    def eos_token_ids(self) -> list[int]:
        return self.model_config.eos_token_ids
        
        
    @property
    def pad_token_id(self) -> int:
        return self.eos_token_ids[0]
        
        
    @property
    def dtype(self) -> str:
        return self.model_config.dtype
        
        
    @property
    def head_dim(self) -> int:
        return self.model_config.head_dim
        
    @property
    def rope_theta(self) -> float:
        return self.model_config.rope_theta
        
        
    def parse_eos_token_ids(self, config):
        value = config["eos_token_id"]
        if isinstance(value, list):
            return value
        elif isinstance(value, list):
            return [value]
        else:
            return []
        

    def _forward_impl(
        self,
        # [B, T_q]
        x: torch.Tensor,
        weights: torch.Tensor | None,
        attention_metadata: AttentionMetadata
    ) -> torch.Tensor:    
        # Get RoPE rotation matrix for each position.
        # [B, T_q, head_dim // 2, 2, 2]
        cos_pos = self.cos_pos_cache[attention_metadata.positions]
        sin_pos = self.sin_pos_cache[attention_metadata.positions]
        
        # [B, T_q, hidden_size]
        h = self.embed.forward(x)
        
        # Core transformer layer loop.
        for idx, layer in enumerate(self.layers):
            if self.cpu_offloading:
                # Wait for weights transfer to finish.
                torch.cuda.current_stream().wait_stream(self.transfer_stream)
                if idx < len(self.layers) - 1:
                    with (
                        torch.cuda.stream(self.transfer_stream),
                        record_function(f"model.layer.{idx + 1}.preload")
                    ):
                        # Preload the next layer's weights to device.
                        self.layers[idx + 1].preload_weights(x.device, self.transformer_layer_staging_buffers)
            
            with record_function(f"model.layer.{idx}.forward"):
                h = layer.forward(
                    h,
                    cos_pos,
                    sin_pos,
                    self.paged_kv_cache.get_layer_kv_cache(idx),
                    attention_metadata
                )
            
            if self.cpu_offloading:
                # Unload the current layer's weights from the device.
                layer.unload_weights()
        
        # Final layernorm.
        with record_function("model.final_norm"):
            h_normed = self.final_norm.forward(h)
        
        # Unembed. Computes output logits.
        with record_function("model.compute_logits"):
            # TODO: Only compute logits for last token hidden state during prefill.
            logits = self.unembed.forward(h_normed)
        return logits
        
            
    def _backward_impl(
        self,
        dL_dy: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # l = W_e @ h_n
        dL_dh_n = self.unembed.backward(dL_dy)
        
        # h_normed = RMSNorm(h_n)
        dL_dh = self.final_norm.backward(dL_dh_n)
        
        # Core transformer layer loop.
        for idx, layer in enumerate(reversed(self.layers)):
            with record_function(f"model.layer.{idx}.backward"):
                dL_dh = layer.backward(dL_dh)
                
        # h = W_e[x]
        dL_dx = self.embed.backward(dL_dh)
        return dL_dx, None
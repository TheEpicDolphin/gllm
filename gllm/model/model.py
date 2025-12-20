import json
import os

import torch
import torch.nn.functional as F
from enum import StrEnum
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from tokenizers import Tokenizer

from gllm.config.generator_params import GeneratorParams
from gllm.config.model_config import ModelConfig
from gllm.model.kv_cache.paged_kv_cache import PagedKVCache
from gllm.model.layers.attention import AttentionMetadata
from gllm.model.layers.norm import RMSNorm
from gllm.model.transformer import Transformer

CPU_DEVICE = "cpu"


class HuggingFaceModel(StrEnum):
    LLAMA_3_2_1B = "meta-llama/Llama-3.2-1B"
    LLAMA_3_2_1B_INSTUCT = "meta-llama/Llama-3.2-1B-Instruct"
    

class Model:
    def __init__(
        self,
        hf_model: HuggingFaceModel,
        gen_params: GeneratorParams,
        device: str,
        local_cache_dir: str | None = None,
    ):
        if local_cache_dir is None:
            # Use default cache directory.
            local_cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "gllm")
        
        # Create directories for model and tokenizer files.
        model_cache_dir = os.path.join(local_cache_dir, "models")
        os.makedirs(model_cache_dir, exist_ok=True)
        tokenizer_cache_dir = os.path.join(local_cache_dir, "tokenizers")
        os.makedirs(tokenizer_cache_dir, exist_ok=True)
    
        # Download model safetensors.
        local_model_path = hf_hub_download(
            repo_id=hf_model,
            filename="model.safetensors",
            cache_dir=model_cache_dir,
        )
        # Tensors are kept in CPU RAM and loaded on GPU only when needed.
        safetensors = load_file(local_model_path, device=CPU_DEVICE)

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
            dtype=getattr(torch, gen_params.model_dtype or config["torch_dtype"]),
            hidden_size=config["hidden_size"],
            head_dim=config["head_dim"],
            intermediate_size=config["intermediate_size"],
            act_func=config["hidden_act"],
            num_layers=config["num_hidden_layers"],
            num_attn_heads=config["num_attention_heads"],
            num_kv_heads=config["num_key_value_heads"],
            rms_norm_eps=config["rms_norm_eps"],
            eos_token_ids=self.parse_eos_token_ids(config),
            kv_dtype=getattr(torch, gen_params.kv_dtype or config["torch_dtype"]),
            rope_theta=config["rope_theta"],
        )
        
        # Initialize paged KV cache.
        self.paged_kv_cache = PagedKVCache(
            model_config=self.model_config,
            gen_params=gen_params,
            device=device,
        )

        # Initialize transformer layers.
        self.layers: list[Transformer] = []
        for layer_idx in range(self.model_config.num_layers):
            self.layers.append(Transformer(
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
        self.embedding = safetensors["model.embed_tokens.weight"].to(self.dtype)
        self.device = torch.device(device)
        
        # Construct RoPE sin/cos caches for positions up to T_max.
        # [T_max]
        p = torch.arange(gen_params.max_seq_len, device=device)
        # [head_dim // 2]
        m = torch.arange(self.head_dim // 2, device=device)
        theta_m = self.rope_theta**(-2 * m / self.head_dim)
        # [T_max, head_dim // 2]
        p_theta_m = p.unsqueeze(1) * theta_m
        # [T_max, head_dim // 2]
        self.cos_pos_cache = torch.cos(p_theta_m).to(self.dtype)
        # [T_max, head_dim // 2]
        self.sin_pos_cache = torch.sin(p_theta_m).to(self.dtype)
        
        # Allocate staging buffers using the first transformer layer.
        # All layers are expected to have the same weight shapes.
        self.staging_buffers = self.layers[0].allocate_staging_buffers()
        

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


    def get_token_embeddings(
        self,
        token_ids: torch.Tensor | list[int]
    ) -> torch.Tensor:
        # Load embedding weights to device.
        embedding_gpu = self.embedding.to(self.device)
        embeddings = embedding_gpu[token_ids]
        return embeddings
        

    def forward(
        self,
        # [B, T, hidden_size]
        hidden_states: torch.Tensor,
        # [B, T_q]
        positions: torch.Tensor,
        attention_metadata: AttentionMetadata
    ) -> torch.Tensor:
        # Get RoPE rotation matrix for each position.
        # [B, T_q, head_dim // 2, 2, 2]
        cos_pos = self.cos_pos_cache[positions]
        sin_pos = self.sin_pos_cache[positions]
        
        # Preload first layer's weights to device.
        device = hidden_states.device
        
        residual = None
        for idx, layer in enumerate(self.layers):
            if idx < len(self.layers) - 1:
                # Preload the next layer's weights to device.
                self.layers[idx + 1].preload_weights(device, self.staging_buffers)
                
            hidden_states, residual = layer.forward(
                hidden_states,
                residual,
                cos_pos,
                sin_pos,
                self.paged_kv_cache.get_layer_kv_cache(idx),
                attention_metadata
            )
            
            # Unload the current layer's weights from the device.
            layer.unload_weights()
        return self.final_norm.forward(hidden_states + residual)


    def compute_logits(
        self,
        # [B, T, hidden_size]
        x: torch.Tensor
    ) -> torch.Tensor:
        # Load embedding weights to device.
        embedding_gpu = self.embedding.to(self.device)
        return F.linear(x, embedding_gpu)
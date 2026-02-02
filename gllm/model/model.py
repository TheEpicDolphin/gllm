import json
import os

from contextlib import contextmanager
from pyexpat import model

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from tokenizers import Tokenizer
from torch.profiler import record_function

from gllm.config.model_config import ModelConfig
from gllm.model.kv_cache.paged_kv_cache import PagedKVCache
from gllm.model.layers.attention import AttentionMetadata
from gllm.model.layers.base_module import BaseModule
from gllm.model.layers.embedding import Embedding
from gllm.model.layers.linear import Linear
from gllm.model.layers.norm import RMSNorm
from gllm.model.layers.transformer_layer import TransformerLayer

CPU_DEVICE = "cpu"


class Model(BaseModule):
    def __init__(
        self,
        model_path: str,
        max_seq_len: int,
        device: str,
        dtype: str | None = None,
        kv_dtype: str | None = None,
        cpu_offloading: bool = False,
    ):
        super().__init__("model", None)
        
        self.device = torch.device(device)
        self.cpu_offloading = device != "cpu" and cpu_offloading
        
        if os.path.exists(model_path):
            # Get local filepaths for tokenizer, config, and weights.
            tokenizer_filepath = os.path.join(model_path, "tokenizer.json")
            config_filepath = os.path.join(model_path, "config.json")
            weights_filepath = os.path.join(model_path, "model.safetensors")
        else:
            # Assume model path is a HuggingFace repo id.
            from huggingface_hub import hf_hub_download
            
            # Download files for tokenizer, config, and weights.
            tokenizer_filepath = hf_hub_download(repo_id=model_path, filename="tokenizer.json")
            config_filepath = hf_hub_download(repo_id=model_path, filename="config.json")
            weights_filepath = hf_hub_download(repo_id=model_path, filename="model.safetensors")
        
        # Load tokenizer.
        self.tokenizer = Tokenizer.from_file(tokenizer_filepath)

        # Load model config.
        with open(config_filepath, "r") as f:
            config = json.load(f)
        self.config = ModelConfig(
            dtype=getattr(torch, dtype or config["torch_dtype"]),
            hidden_size=config["hidden_size"],
            head_dim=config["head_dim"],
            intermediate_size=config["intermediate_size"],
            act_func=config["hidden_act"],
            num_layers=config["num_hidden_layers"],
            num_attn_heads=config["num_attention_heads"],
            num_kv_heads=config["num_key_value_heads"],
            rms_norm_eps=config["rms_norm_eps"],
            eos_token_ids=self.parse_eos_token_ids(config),
            kv_dtype=getattr(torch, kv_dtype or config["torch_dtype"]),
            rope_theta=config["rope_theta"],
            vocab_size=config["vocab_size"],
        )
        
        # Load model safetensors.
        # If cpu_offloading is true, weights are kept in CPU RAM and loaded to GPU only when needed.
        initial_weights_device = CPU_DEVICE if self.cpu_offloading else device
        safetensors = load_file(weights_filepath, device=initial_weights_device)

        # Initialize transformer layers.
        self.layers: list[TransformerLayer] = []
        for layer_idx in range(self.config.num_layers):
            self.layers.append(TransformerLayer(
                f"{self._id}.layers.{layer_idx}",
                model_config=self.config,
                safetensors=safetensors,
            ))
        
        # Initialize final norm.
        final_norm_id = f"{self._id}.norm"
        final_norm_weights = safetensors[f"{final_norm_id}.weight"].to(self.dtype)
        self.final_norm = RMSNorm(
            final_norm_id,
            weights=final_norm_weights,
            eps=self.config.rms_norm_eps
        )
        
        # Get embedding matrix. This is indexed into using the token
        # ids to get the embedding vectors. It is also used as an LM
        # head by multiplying with the hidden states to obtain the
        # logits.
        embedding_id = f"{self._id}.embed_tokens"
        self.embedding_weights = safetensors[f"{embedding_id}.weight"].to(self.dtype)
        self.embed = Embedding(embedding_id, self.embedding_weights)
        self.lm_head = Linear(embedding_id, self.embedding_weights)
        
        # Construct RoPE sin/cos caches for positions up to T_max.
        # [T_max]
        p = torch.arange(max_seq_len, device=device)
        # [head_dim // 2]
        m = torch.arange(self.head_dim // 2, device=device)
        theta_m = self.config.rope_theta**(-2 * m / self.head_dim)
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
            self.lm_head,
        ]
        

    @property
    def eos_token_ids(self) -> list[int]:
        return self.config.eos_token_ids
        
        
    @property
    def pad_token_id(self) -> int:
        return self.eos_token_ids[0]
        
        
    @property
    def dtype(self) -> str:
        return self.config.dtype
        
        
    @property
    def head_dim(self) -> int:
        return self.config.head_dim
    
    
    @property
    def vocab_size(self) -> int:
        return self.config.vocab_size
        
        
    def parse_eos_token_ids(self, config):
        value = config["eos_token_id"]
        if isinstance(value, list):
            return value
        elif isinstance(value, list):
            return [value]
        else:
            return []
            

    @contextmanager
    def training_mode(self):
        try:
            for module in self.modules():
                module.was_training = module._training
                module._training = True
            yield
        finally:
            for module in self.modules():
                module._training = module.was_training
                del module.was_training

            
    def save(self, dir: str):
        from safetensors.torch import save_file

        weights_dict = {}
        self.save_tensors(weights_dict)
        save_file(weights_dict, os.path.join(dir, "model.safetensors"))
        
        
    def retokenize(self, new_tokenizer):
        # TODO: Construct a new embedding table by averaging together the
        # subword embeddings for each token in the new tokenizer's vocabulary.
        # TODO: Update model config with new vocabulary and bos/pad/eos token ids.
        self.tokenizer = new_tokenizer
        

    def _forward_impl(
        self,
        # [B, T_q]
        x: torch.Tensor,
        weights: torch.Tensor | None,
        attention_metadata: AttentionMetadata,
        kv_cache: PagedKVCache | None = None,
    ) -> torch.Tensor:    
        # Get RoPE rotation matrix for each position.
        # [B, T_q, head_dim // 2]
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
                    kv_cache.get_layer_kv_cache(idx) if kv_cache is not None else None,
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
            logits = self.lm_head.forward(h_normed)
        return logits
        
            
    def _backward_impl(
        self,
        dL_dy: torch.Tensor,
        weights: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # l = W_e @ h_n
        dL_dh_n = self.lm_head.backward(dL_dy)
        
        # h_normed = RMSNorm(h)
        dL_dh = self.final_norm.backward(dL_dh_n)
        
        # Core transformer layer loop.
        for idx, layer in enumerate(reversed(self.layers)):
            with record_function(f"model.layer.{idx}.backward"):
                dL_dh = layer.backward(dL_dh)
                
        # h = W_e[x]
        dL_dx = self.embed.backward(dL_dh)
        return dL_dx, None
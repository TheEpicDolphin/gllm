import os

from contextlib import contextmanager

import torch
from tokenizers import Tokenizer
from torch.profiler import record_function

from gllm.model.config import ModelConfig
from gllm.model.kv_cache.paged_kv_cache import PagedKVCache
from gllm.model.layers.attention import AttentionMetadata
from gllm.model.layers.base_module import BaseModule
from gllm.model.layers.embedding import Embedding
from gllm.model.layers.linear import Linear
from gllm.model.layers.norm import RMSNorm
from gllm.model.layers.transformer_layer import TransformerLayer
from gllm.model.params import ModelParams


class Model(BaseModule):
    def __init__(
        self,
        config: ModelConfig,
        params: ModelParams,
        tokenizer: None,
        device: str,
        max_seq_len: int,
        offload_device: str | None = None,
    ):
        super().__init__(None)
        
        self.config = config
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.cpu_offloading = self.device.type == "cuda" and offload_device == "cpu"

        # Initialize transformer layers.
        self.layers: list[TransformerLayer] = []
        for idx in range(self.config.num_layers):
            self.layers.append(TransformerLayer(
                self.config,
                params.layers[idx],
            ))
        # Initialize embedding. This is indexed into using the token
        # ids to get the embedding vectors.
        self.embed = Embedding(params.embed)
        # Initialize the LM head. This is multiplied with the output
        # hidden states to obtain the logits.
        self.lm_head = Linear(params.lm_head)
        # Initialize final norm.
        self.final_norm = RMSNorm(
            params.final_norm,
            eps=self.config.rms_norm_eps
        )
        
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
        
    
    @classmethod
    def from_path(
        cls,
        path: str,
        device: str,
        max_seq_len: int,
        dtype_override: torch.dtype | None = None,
        offload_device: str | None = None,
    ):
        if os.path.exists(path):
            # Get local filepaths for tokenizer, config, and weights.
            tokenizer_filepath = os.path.join(path, "tokenizer.json")
            config_filepath = os.path.join(path, "config.json")
            weights_filepath = os.path.join(path, "model.safetensors")
        else:
            # Assume model path is a HuggingFace repo id.
            from huggingface_hub import hf_hub_download
            
            # Download files for tokenizer, config, and weights.
            tokenizer_filepath = hf_hub_download(repo_id=path, filename="tokenizer.json")
            config_filepath = hf_hub_download(repo_id=path, filename="config.json")
            weights_filepath = hf_hub_download(repo_id=path, filename="model.safetensors")

        # Load the model config.
        model_config = ModelConfig.from_json(
            config_filepath,
            dtype_override=dtype_override,
        )
        # Load the model weights.
        weights = ModelParams.from_safetensors(
            weights_filepath,
            model_config,
            initial_device=offload_device or device,
        )
        # Load the tokenizer.
        tokenizer = Tokenizer.from_file(tokenizer_filepath)
        # Create the model.
        return Model(
            model_config,
            weights,
            tokenizer,
            device,
            max_seq_len=max_seq_len,
            offload_device=offload_device,
        )
            

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


    def tokenize(self, tokens: str) -> list[int]:
        return self.tokenizer.encode(tokens).ids


    def detokenize(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids)
        
        
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
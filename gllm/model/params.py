from dataclasses import dataclass

import torch

from safetensors.torch import load_file

from gllm.model.config import ModelConfig, ModelType


@dataclass
class Parameter:
    id: str
    weights: torch.Tensor
    grad: torch.Tensor | None = None
    requires_grad: bool = True


    @classmethod
    def from_safetensors(
        cls,
        safetensors,
        key: str,
        dtype: torch.dtype
    ):
        return Parameter(key, safetensors[f"{key}.weight"].to(dtype))


@dataclass
class AttentionParams:
    q_proj: Parameter
    k_proj: Parameter
    v_proj: Parameter
    o_proj: Parameter


@dataclass
class FFNParams:
    up_proj: Parameter
    down_proj: Parameter
    gate_proj: Parameter | None


@dataclass
class TransformerLayerParams:
    id: str
    input_norm: Parameter
    attention: AttentionParams
    post_attn_norm: Parameter
    ffn: FFNParams


@dataclass
class ModelParams:
    embed: Parameter
    layers: list[TransformerLayerParams]
    final_norm: Parameter
    lm_head: Parameter


    @classmethod
    def from_safetensors(
        cls,
        path: str,
        model_config: ModelConfig,
        initial_device: str,
    ):
        safetensors = load_file(path, device=initial_device)
        dtype = model_config.dtype
        if model_config.type == ModelType.LLAMA:
            layers = []
            for layer_idx in range(model_config.num_layers):
                layer_id = f"model.layers.{layer_idx}"
                layers.append(TransformerLayerParams(
                    id=layer_id,
                    input_norm=Parameter.from_safetensors(safetensors, f"{layer_id}.input_layernorm", dtype),
                    attention=AttentionParams(
                        q_proj=Parameter.from_safetensors(safetensors, f"{layer_id}.self_attn.q_proj", dtype),
                        k_proj=Parameter.from_safetensors(safetensors, f"{layer_id}.self_attn.k_proj", dtype),
                        v_proj=Parameter.from_safetensors(safetensors, f"{layer_id}.self_attn.v_proj", dtype),
                        o_proj=Parameter.from_safetensors(safetensors, f"{layer_id}.self_attn.o_proj", dtype),
                    ),
                    post_attn_norm=Parameter.from_safetensors(safetensors, f"{layer_id}.post_attention_layernorm", dtype),
                    ffn=FFNParams(
                        up_proj=Parameter.from_safetensors(safetensors, f"{layer_id}.mlp.up_proj", dtype),
                        down_proj=Parameter.from_safetensors(safetensors, f"{layer_id}.mlp.down_proj", dtype),
                        gate_proj=Parameter.from_safetensors(safetensors, f"{layer_id}.mlp.gate_proj", dtype),
                    )
                ))
            embed_param = Parameter.from_safetensors(safetensors, f"model.embed_tokens", dtype)
            return ModelParams(
                embed=embed_param,
                layers=layers,
                final_norm=Parameter.from_safetensors(safetensors, f"model.norm", dtype),
                # LM head is tied to embedding for llama.
                lm_head=embed_param,
            )
        elif model_config.type == ModelType.LLADA:
            layers = []
            for layer_idx in range(model_config.num_layers):
                layer_id = f"model.transformer.blocks.{layer_idx}"
                layers.append(TransformerLayerParams(
                    id=layer_id,
                    input_norm=Parameter.from_safetensors(safetensors, f"{layer_id}.attn_norm", dtype),
                    attention=AttentionParams(
                        q_proj=Parameter.from_safetensors(safetensors, f"{layer_id}.q_proj", dtype),
                        k_proj=Parameter.from_safetensors(safetensors, f"{layer_id}.k_proj", dtype),
                        v_proj=Parameter.from_safetensors(safetensors, f"{layer_id}.v_proj", dtype),
                        o_proj=Parameter.from_safetensors(safetensors, f"{layer_id}.attn_out", dtype),
                    ),
                    post_attn_norm=Parameter.from_safetensors(safetensors, f"{layer_id}.ff_norm", dtype),
                    ffn=FFNParams(
                        up_proj=Parameter.from_safetensors(safetensors, f"{layer_id}.up_proj", dtype),
                        down_proj=Parameter.from_safetensors(safetensors, f"{layer_id}.ff_out", dtype),
                        gate_proj=Parameter.from_safetensors(safetensors, f"{layer_id}.ff_proj", dtype),
                    )
                ))
            return ModelParams(
                embed=Parameter.from_safetensors(safetensors, f"model.transformer.wte", dtype),
                layers=layers,
                final_norm=Parameter.from_safetensors(safetensors, f"model.transformer.ln_f", dtype),
                lm_head=Parameter.from_safetensors(safetensors, f"model.transformer.ff_out", dtype),
            )
        else:
            raise NotImplementedError(f"Attempted to load weights for unsupported model type: {model_config.type}")
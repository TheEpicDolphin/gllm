import json
from dataclasses import dataclass
from enum import StrEnum

import torch


class ModelType(StrEnum):
    LLAMA = "llama"
    LLADA = "llada"


class ActivationFunction(StrEnum):
    SILU = "silu"
    RELU = "relu"
    GELU = "gelu"
    SWIGLU = "swiglu"


@dataclass
class ModelConfig:
    type: ModelType
    dtype: torch.dtype
    hidden_size: int
    head_dim: int
    intermediate_size: int
    act_func: ActivationFunction
    num_layers: int
    num_attn_heads: int
    num_kv_heads: int
    rms_norm_eps: float
    eos_token_ids: list[int]
    pad_token_id: int
    rope_theta: float
    vocab_size: int

    mask_token_id: int | None = None


    @staticmethod
    def _parse_dtype(value: str) -> torch.dtype:
        if value.endswith("bf16"):
            return torch.bfloat16
        return getattr(torch, value)


    @staticmethod
    def _parse_eos_token_ids(value):
        if isinstance(value, list):
            return value
        elif isinstance(value, int):
            return [value]
        else:
            raise RuntimeError(f"Unsupported eos token id value type: {type(value)}")
        
    
    @classmethod
    def from_json(
        cls,
        path: str,
        dtype_override: torch.dtype | None
    ):
        with open(path, "r") as f:
            config = json.load(f)

        model_type = config["model_type"]
        if model_type == "llama":
            eos_token_ids = cls._parse_eos_token_ids(config["eos_token_id"])
            return ModelConfig(
                type=model_type,
                dtype=dtype_override or cls._parse_dtype(config["torch_dtype"]),
                hidden_size=config["hidden_size"],
                head_dim=config["head_dim"],
                intermediate_size=config["intermediate_size"],
                act_func=config["hidden_act"],
                num_layers=config["num_hidden_layers"],
                num_attn_heads=config["num_attention_heads"],
                num_kv_heads=config["num_key_value_heads"],
                rms_norm_eps=config["rms_norm_eps"],
                eos_token_ids=eos_token_ids,
                pad_token_id=eos_token_ids[0],
                rope_theta=config["rope_theta"],
                vocab_size=config["vocab_size"],
            )
        elif model_type == "llada":
            return ModelConfig(
                type=model_type,
                dtype=dtype_override or cls._parse_dtype(config["precision"]),
                hidden_size=config["d_model"],
                head_dim=config["d_model"] // config["n_heads"],
                intermediate_size=config["mlp_hidden_size"],
                act_func=config["activation_type"],
                num_layers=config["n_layers"],
                num_attn_heads=config["n_heads"],
                num_kv_heads=config["n_kv_heads"],
                rms_norm_eps=config["rms_norm_eps"],
                eos_token_ids=cls._parse_eos_token_ids(config["eos_token_id"]),
                pad_token_id=config["pad_token_id"],
                mask_token_id=config["mask_token_id"],
                rope_theta=config["rope_theta"],
                vocab_size=config["vocab_size"],
            )
        else:
            raise NotImplementedError(f"Attempted to load config unsupported model type: {model_type}")
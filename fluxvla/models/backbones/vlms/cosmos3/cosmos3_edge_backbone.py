# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Compact Cosmos3-Edge Nemotron MoT backbone.

The Nemotron configuration, RMSNorm, ReLU2 MLP, rotary embedding, and
56-block-to-28-layer folding follow NVIDIA cosmos-framework's public
``nemotron_3_dense_vl`` and ``unified_mot`` implementations.  The wrapper is
kept intentionally small and exposes the same FluxVLA packed-forward contract
as :class:`Cosmos3MoTBackbone`.
"""

from __future__ import annotations

import copy
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping, Type

import torch
import torch.nn.functional as F
from torch import nn
from transformers.configuration_utils import PretrainedConfig
from transformers.initialization import no_init_weights

from fluxvla.engines import VLM_BACKBONES
from fluxvla.models.third_party_models.cosmos3.data.vfm.sequence_packing import (
    FactoredSequencePack, from_joint, get_device_and_dtype, get_gen_seq,
    get_und_seq, set_gen_seq, set_und_seq, zeros_like)
from .cosmos3_mot_layer import Cosmos3TextDecoderLayer


COSMOS3_EDGE_TEXT_CONFIG = {
    'model_type': 'nemotron_3_dense_vl_text',
    'vocab_size': 131072,
    'tie_word_embeddings': False,
    'hidden_size': 2048,
    'intermediate_size': 9216,
    'num_hidden_layers': 28,
    'num_attention_heads': 16,
    'head_dim': 128,
    'num_key_value_heads': 8,
    'mlp_hidden_act': 'relu2',
    'attention_bias': False,
    'mlp_bias': False,
    'initializer_range': 0.02,
    'layer_norm_epsilon': 1e-5,
    'residual_in_fp32': False,
    'use_cache': True,
    'num_logits_to_keep': 1,
    'pad_token_id': 11,
    'bos_token_id': 1,
    'eos_token_id': 11,
    'sliding_window': None,
    'max_position_embeddings': 131072,
    'attention_dropout': 0.0,
    'hidden_dropout': 0.0,
    'enable_rope': True,
    'rope_scaling': None,
    'rope_theta': 100_000_000.0,
    'enable_mrope': True,
    'mrope_section': [24, 20, 20],
    'use_und_k_norm_for_gen': True,
    'torch_dtype': 'bfloat16',
}

COSMOS3_EDGE_SPECIAL_TOKENS = {
    'eos_token_id': 11,
    'start_of_generation': 20,
    'end_of_generation': 21,
}


def _parameter_dtype(value: str | torch.dtype | None) -> torch.dtype:
    if isinstance(value, torch.dtype):
        return value
    if value is None:
        return torch.get_default_dtype()
    names = {
        'bfloat16': torch.bfloat16,
        'bf16': torch.bfloat16,
        'float16': torch.float16,
        'fp16': torch.float16,
        'float32': torch.float32,
        'fp32': torch.float32,
    }
    name = str(value).lower()
    if name not in names:
        raise ValueError(f'Unsupported Edge parameter dtype: {value!r}.')
    return names[name]


@contextmanager
def _temporary_default_dtype(dtype: torch.dtype):
    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


class Nemotron3DenseVLTextConfig(PretrainedConfig):
    """Nemotron-H config after pairing 56 hybrid blocks into 28 layers."""

    model_type = 'nemotron_3_dense_vl_text'

    def __init__(
        self,
        vocab_size: int = 131072,
        tie_word_embeddings: bool = False,
        hidden_size: int = 2048,
        intermediate_size: int = 9216,
        num_hidden_layers: int = 28,
        num_attention_heads: int = 16,
        head_dim: int = 128,
        num_key_value_heads: int = 8,
        mlp_hidden_act: str = 'relu2',
        attention_bias: bool = False,
        mlp_bias: bool = False,
        initializer_range: float = 0.02,
        layer_norm_epsilon: float = 1e-5,
        residual_in_fp32: bool = False,
        use_cache: bool = True,
        num_logits_to_keep: int = 1,
        pad_token_id: int = 11,
        bos_token_id: int = 1,
        eos_token_id: int = 11,
        sliding_window: int | None = None,
        max_position_embeddings: int = 131072,
        attention_dropout: float = 0.0,
        hidden_dropout: float = 0.0,
        enable_rope: bool = True,
        rope_scaling: dict | None = None,
        rope_theta: float = 100_000_000.0,
        enable_mrope: bool = True,
        mrope_section: list[int] | None = None,
        use_und_k_norm_for_gen: bool = True,
        torch_dtype: str = 'bfloat16',
        **kwargs,
    ) -> None:
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.num_key_value_heads = num_key_value_heads
        self.mlp_hidden_act = mlp_hidden_act
        self.attention_bias = attention_bias
        self.mlp_bias = mlp_bias
        self.initializer_range = initializer_range
        self.layer_norm_epsilon = layer_norm_epsilon
        self.residual_in_fp32 = residual_in_fp32
        self.use_cache = use_cache
        self.num_logits_to_keep = num_logits_to_keep
        self.sliding_window = sliding_window
        self.max_position_embeddings = max_position_embeddings
        self.attention_dropout = attention_dropout
        self.hidden_dropout = hidden_dropout
        self.enable_rope = enable_rope
        self.rope_scaling = rope_scaling
        self.rope_theta = rope_theta
        self.enable_mrope = enable_mrope
        self.mrope_section = ([24, 20, 20]
                              if mrope_section is None else mrope_section)
        self.use_und_k_norm_for_gen = bool(use_und_k_norm_for_gen)
        self.torch_dtype = torch_dtype
        self._attn_implementation = kwargs.pop('_attn_implementation', 'eager')
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    @property
    def rms_norm_eps(self) -> float:
        return self.layer_norm_epsilon


def relu2(x: torch.Tensor) -> torch.Tensor:
    """Nemotron's squared-ReLU activation."""

    return F.relu(x).square()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_partial(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply Nemotron partial RoPE; Edge-2B rotates the full head."""

    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    rot_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rot_dim], q[..., rot_dim:]
    k_rot, k_pass = k[..., :rot_dim], k[..., rot_dim:]
    q_embed = (q_rot * cos) + (_rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (_rotate_half(k_rot) * sin)
    return (torch.cat((q_embed, q_pass), dim=-1),
            torch.cat((k_embed, k_pass), dim=-1))


class Nemotron3DenseVLRMSNorm(nn.Module):

    def __init__(self, hidden_size: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        normalized = hidden_states.float()
        variance = normalized.pow(2).mean(-1, keepdim=True)
        normalized = normalized * torch.rsqrt(variance +
                                               self.variance_epsilon)
        return (self.weight.float() * normalized).to(input_dtype)


class Nemotron3DenseVLMLP(nn.Module):

    def __init__(self, config: Nemotron3DenseVLTextConfig) -> None:
        super().__init__()
        self.up_proj = nn.Linear(
            config.hidden_size,
            config.intermediate_size,
            bias=config.mlp_bias,
        )
        self.down_proj = nn.Linear(
            config.intermediate_size,
            config.hidden_size,
            bias=config.mlp_bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(relu2(self.up_proj(x)))


class MultiModalRotaryEmbedding(nn.Module):
    """Nemotron multimodal RoPE used by the Edge packed transformer."""

    def __init__(self, config: Nemotron3DenseVLTextConfig) -> None:
        super().__init__()
        self.config = config
        self.mrope_section = config.mrope_section
        if sum(self.mrope_section) != config.head_dim // 2:
            raise ValueError(
                'Nemotron mrope_section must sum to head_dim / 2, got '
                f'{self.mrope_section} for head_dim={config.head_dim}.')
        inv_freq = 1.0 / (
            config.rope_theta**(
                torch.arange(0, config.head_dim, 2, dtype=torch.float32) /
                config.head_dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)
        self.register_buffer(
            'original_inv_freq', inv_freq.clone(), persistent=False)

    def _apply_interleaved_mrope(self, freqs: torch.Tensor) -> torch.Tensor:
        freqs_t = freqs[0].clone()
        for dim, offset in enumerate((1, 2), start=1):
            length = self.mrope_section[dim] * 3
            freqs_t[..., slice(offset, length, 3)] = freqs[
                dim, ..., slice(offset, length, 3)]
        return freqs_t

    @torch.no_grad()
    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if position_ids.ndim == 1:
            position_ids = position_ids.unsqueeze(0)
        if position_ids.ndim == 2:
            position_ids = position_ids.unsqueeze(0).expand(
                3, position_ids.shape[0], -1)
        if position_ids.ndim != 3 or position_ids.shape[0] != 3:
            raise ValueError(
                'Nemotron mRoPE expects [N], [B,N], or [3,B,N] position '
                f'ids, got {tuple(position_ids.shape)}.')

        inv_freq = self.inv_freq[None, None, :, None].float().expand(
            3, position_ids.shape[1], -1, 1)
        pos = position_ids[:, :, None, :].float()
        device_type = x.device.type if x.device.type != 'mps' else 'cpu'
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq @ pos).transpose(2, 3)
            freqs = self._apply_interleaved_mrope(freqs)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos, sin = emb.cos(), emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class Cosmos3EdgeTextDecoderLayer(Cosmos3TextDecoderLayer):
    """Nemotron dense dual-pathway MoT layer for Cosmos3-Edge."""

    def __init__(
        self,
        config: Nemotron3DenseVLTextConfig,
        layer_idx: int,
        *,
        qk_norm_for_text: bool = False,
        qk_norm_for_diffusion: bool = True,
    ) -> None:
        super().__init__(
            config,
            layer_idx,
            qk_norm_for_text=qk_norm_for_text,
            qk_norm_for_diffusion=qk_norm_for_diffusion,
            mlp_cls=Nemotron3DenseVLMLP,
            rms_norm_cls=Nemotron3DenseVLRMSNorm,
            apply_rotary_pos_emb=apply_rotary_pos_emb_partial,
        )


class _Cosmos3EdgeTextModel(nn.Module):

    def __init__(
        self,
        config: Nemotron3DenseVLTextConfig,
        *,
        qk_norm_for_text: bool,
        qk_norm_for_diffusion: bool,
    ) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([
            Cosmos3EdgeTextDecoderLayer(
                config,
                layer_idx,
                qk_norm_for_text=qk_norm_for_text,
                qk_norm_for_diffusion=qk_norm_for_diffusion,
            ) for layer_idx in range(config.num_hidden_layers)
        ])
        self.norm = Nemotron3DenseVLRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps)
        self.norm_moe_gen = Nemotron3DenseVLRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = MultiModalRotaryEmbedding(config)

    def forward_packed(
        self,
        pack: FactoredSequencePack,
        attention_mask,
        position_ids: torch.Tensor,
    ) -> tuple[FactoredSequencePack, dict[str, Any]]:
        device, dtype = get_device_and_dtype(pack)
        meta_tensor = torch.tensor([], dtype=dtype, device=device)
        rope_ids = (position_ids.unsqueeze(0)
                    if position_ids.ndim == 1 else position_ids.unsqueeze(1))
        cos, sin = self.rotary_emb(meta_tensor, position_ids=rope_ids)
        position_embeddings = (from_joint(cos.squeeze(0), pack),
                               from_joint(sin.squeeze(0), pack))

        hidden_states = pack
        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                packed_sequence=hidden_states,
                packed_position_embeddings=position_embeddings,
                packed_attention_mask=attention_mask,
            )

        output = zeros_like(hidden_states)
        set_und_seq(output, self.norm(get_und_seq(hidden_states)))
        set_gen_seq(output,
                    self.norm_moe_gen(get_gen_seq(hidden_states)))
        return output, {}


class _Cosmos3EdgeModel(nn.Module):

    def __init__(self, language_model: _Cosmos3EdgeTextModel) -> None:
        super().__init__()
        self.language_model = language_model


@VLM_BACKBONES.register_module()
class Cosmos3EdgeBackbone(nn.Module):
    """FluxVLA-facing Cosmos3-Edge Nemotron generator backbone."""

    architecture_family = 'edge_nemotron'

    def __init__(
        self,
        vlm_config: Mapping[str, Any] | str | None = None,
        *,
        include_visual: bool = False,
        packed_attention_backend: str = 'flash2',
        text_config_overrides: Mapping[str, Any] | None = None,
        vision_encoder_path: str | Path | None = None,
        skip_init_weights: bool = False,
    ) -> None:
        super().__init__()
        if include_visual:
            raise NotImplementedError(
                'Cosmos3EdgeBackbone currently implements the Nemotron '
                'generator tower only; Edge SigLIP2 reasoner inputs are not '
                'yet supported.')
        if vision_encoder_path is not None:
            raise ValueError(
                'vision_encoder_path is only valid with include_visual=True.')

        config_dict = self._load_vlm_config(vlm_config)
        nested = config_dict.get('text_config')
        if isinstance(nested, Mapping):
            config_dict = dict(nested)
        config_dict['packed_attention_backend'] = packed_attention_backend
        if text_config_overrides:
            config_dict.update(dict(text_config_overrides))
        if config_dict.get('num_hidden_layers') == 56:
            config_dict['num_hidden_layers'] = 28
        config_dict['tie_word_embeddings'] = False
        text_config = Nemotron3DenseVLTextConfig(**config_dict)

        parameter_dtype = _parameter_dtype(text_config.torch_dtype)
        with _temporary_default_dtype(parameter_dtype):
            if skip_init_weights:
                with no_init_weights():
                    self._init_model(text_config)
            else:
                self._init_model(text_config)
        self.config = text_config
        self.vlm_config_dict = config_dict
        self.include_visual = False
        self.packed_attention_backend = packed_attention_backend

    def _init_model(self, text_config: Nemotron3DenseVLTextConfig) -> None:
        language_model = _Cosmos3EdgeTextModel(
            text_config,
            qk_norm_for_text=False,
            qk_norm_for_diffusion=True,
        )
        self.model = _Cosmos3EdgeModel(language_model)
        self.lm_head = nn.Linear(
            text_config.hidden_size, text_config.vocab_size, bias=False)

    @staticmethod
    def _load_vlm_config(
        vlm_config: Mapping[str, Any] | str | None,
    ) -> dict[str, Any]:
        if vlm_config is None:
            return copy.deepcopy(COSMOS3_EDGE_TEXT_CONFIG)
        if isinstance(vlm_config, (str, Path)):
            with open(vlm_config, encoding='utf-8') as reader:
                return json.load(reader)
        return copy.deepcopy(dict(vlm_config))

    @property
    def text_config(self) -> Nemotron3DenseVLTextConfig:
        return self.model.language_model.config

    @property
    def decoder_layers(self) -> nn.ModuleList:
        return self.model.language_model.layers

    @property
    def token_embedding(self) -> nn.Embedding:
        return self.model.language_model.embed_tokens

    @property
    def transformer_layer_cls(self) -> Type[nn.Module]:
        return Cosmos3EdgeTextDecoderLayer

    def embed_text_ids(self, text_ids: torch.Tensor) -> torch.Tensor:
        return self.token_embedding(text_ids)

    def forward_packed(self, pack, attention_mask, position_ids):
        return self.model.language_model.forward_packed(
            pack=pack,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

    @classmethod
    def fsdp_transformer_layer_cls(cls):
        return {Cosmos3EdgeTextDecoderLayer}

    def generate(self, *args, **kwargs):
        raise NotImplementedError(
            'Autoregressive Edge reasoner generation is outside the minimal '
            'generator-backbone implementation.')


__all__ = [
    'COSMOS3_EDGE_SPECIAL_TOKENS',
    'COSMOS3_EDGE_TEXT_CONFIG',
    'Cosmos3EdgeBackbone',
    'Cosmos3EdgeTextDecoderLayer',
    'MultiModalRotaryEmbedding',
    'Nemotron3DenseVLMLP',
    'Nemotron3DenseVLRMSNorm',
    'Nemotron3DenseVLTextConfig',
    'apply_rotary_pos_emb_partial',
    'relu2',
]

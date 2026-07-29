# Copyright 2026 Limx Dynamics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from collections.abc import Mapping
from pathlib import Path
import warnings

import torch
import torch.nn as nn

from fluxvla.engines import TOKENIZERS
from fluxvla.models.third_party_models.cosmos3.model.vfm.tokenizers import \
    wan2pt2_vae_4x16x16 as wan_vae

_DEFAULT_ENCODE_CHUNK_FRAMES = {'256': 68, '480': 24, '720': 12}


@TOKENIZERS.register_module()
class Cosmos3Wan22VAE(nn.Module):
    """Frozen Wan2.2 VAE used as the Cosmos3 video latent tokenizer."""

    def __init__(
        self,
        pretrained_name_or_path: str | None = None,
        dtype: torch.dtype | str = torch.bfloat16,
        encode_chunk_frames: Mapping[str, int] | None = None,
        encode_exact_durations: list[int] | None = None,
        spatial_compression_factor: int = 16,
        temporal_compression_factor: int = 4,
    ) -> None:
        super().__init__()

        dtype = self._resolve_dtype(dtype)
        if encode_chunk_frames is None:
            encode_chunk_frames = dict(_DEFAULT_ENCODE_CHUNK_FRAMES)
        if any(chunk % 4 != 0 for chunk in encode_chunk_frames.values()):
            raise ValueError(
                'encode_chunk_frames values must be multiples of 4.')

        resolved_path = None
        if pretrained_name_or_path is not None:
            try:
                resolved_path = str(
                    self.resolve_vae_path(pretrained_name_or_path))
            except FileNotFoundError:
                # An all-in-one Cosmos3 safetensors checkpoint may contain
                # ``vision_vae.*``. Build an empty Wan module now and let the
                # strict parent checkpoint loader populate every parameter.
                # encode/decode remain blocked until that happens.
                warnings.warn(
                    'External Wan2.2 VAE checkpoint was not found at '
                    f'{pretrained_name_or_path}; waiting for embedded '
                    'vision_vae weights from the parent checkpoint.',
                    RuntimeWarning,
                )

        wan = wan_vae.WanVAE(
            dtype=dtype,
            is_amp=False,
            vae_pth=resolved_path,
            temporal_window=encode_chunk_frames,
            encode_exact_durations=encode_exact_durations,
        )
        self.vae = wan.model.eval().requires_grad_(False)
        self._weights_loaded = resolved_path is not None
        self.requested_pretrained_path = pretrained_name_or_path
        self.register_buffer(
            'scale_mean', wan.scale[0].clone(), persistent=False)
        self.register_buffer(
            'scale_inv_std', wan.scale[1].clone(), persistent=False)

        self.encode_chunk_frames = encode_chunk_frames
        self.encode_exact_durations = encode_exact_durations
        self.spatial_compression_factor = spatial_compression_factor
        self.temporal_compression_factor = temporal_compression_factor

    @staticmethod
    def _resolve_dtype(dtype: torch.dtype | str) -> torch.dtype:
        if isinstance(dtype, torch.dtype):
            return dtype
        mapping = {
            'float16': torch.float16,
            'fp16': torch.float16,
            'bfloat16': torch.bfloat16,
            'bf16': torch.bfloat16,
            'float32': torch.float32,
            'fp32': torch.float32,
        }
        if dtype not in mapping:
            raise ValueError(f'Unsupported dtype: {dtype}')
        return mapping[dtype]

    @staticmethod
    def resolve_vae_path(path: str | Path) -> Path:
        path = Path(path)
        if path.is_file():
            return path
        candidate = path / 'Wan2.2_VAE.pth'
        if candidate.is_file():
            return candidate
        raise FileNotFoundError(
            'Could not find Wan2.2_VAE.pth. FluxVLA expects the original '
            f'Cosmos3/Wan2.2 VAE checkpoint, got {path}.')

    @property
    def dtype(self) -> torch.dtype:
        return self.scale_mean.dtype

    @property
    def device(self) -> torch.device:
        return next(self.vae.parameters()).device

    def train(self, mode: bool = True):
        super().train(mode)
        self.vae.eval()
        return self

    def load_state_dict(self, state_dict, strict: bool = True, assign=False):
        """Load a standalone VAE and mark it ready only after an exact load.

        Recursive parent loads invoke ``_load_from_state_dict`` directly, so
        they deliberately do not mark the VAE here.  The strict Cosmos3 parent
        checkpoint loader calls :meth:`mark_weights_loaded` only after its
        per-parameter coverage gate succeeds.
        """
        result = super().load_state_dict(
            state_dict, strict=strict, assign=assign)
        if strict and not result.missing_keys and not result.unexpected_keys:
            self._weights_loaded = True
        return result

    def mark_weights_loaded(self) -> None:
        """Mark a direct parameter-copy checkpoint load as complete."""
        self._weights_loaded = True

    def _require_weights(self) -> None:
        if not self._weights_loaded:
            raise RuntimeError(
                'Wan2.2 VAE weights are not loaded. Provide an external '
                'Wan2.2_VAE.pth or a checkpoint containing vision_vae.*.')

    def requires_grad_(self, requires_grad: bool = False):
        super().requires_grad_(requires_grad)
        self.vae.requires_grad_(requires_grad)
        return self

    @property
    def _scale(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.scale_mean, self.scale_inv_std

    @torch.no_grad()
    def encode(self, state: torch.Tensor) -> torch.Tensor:
        self._require_weights()
        in_dtype = state.dtype
        state = state.to(device=self.device, dtype=self.dtype)
        latents = self.vae.encode(state, self._scale)
        return latents.to(in_dtype)

    @torch.no_grad()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        self._require_weights()
        in_dtype = latent.dtype
        latent = latent.to(device=self.device, dtype=self.dtype)
        video = self.vae.decode(
            latent,
            self._scale,
            clear_decoder_cache=True,
        )
        return video.to(in_dtype)

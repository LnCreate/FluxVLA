# Copyright 2026 Limx Dynamics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
# flake8: noqa

from __future__ import annotations
from typing import List, Optional

import torch
import torch.nn.functional as F


class Cosmos3LossMixin:
    """Flow-matching losses for action and Wan latent predictions."""

    @staticmethod
    def _sample_valid_mask(
        valid_mask,
        index: int,
        *,
        expected_length: Optional[int] = None,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        """Return one sample's 1-D bool mask with strict shape checking."""
        if valid_mask is None:
            return None
        if isinstance(valid_mask, torch.Tensor):
            if valid_mask.ndim == 1:
                if index != 0:
                    raise ValueError(
                        'A 1-D validity mask can only represent batch size 1.')
                sample = valid_mask
            elif valid_mask.ndim == 2:
                if index >= valid_mask.shape[0]:
                    raise ValueError(
                        f'Validity mask has batch size {valid_mask.shape[0]}, '
                        f'cannot read sample {index}.')
                sample = valid_mask[index]
            else:
                raise ValueError(
                    'Validity masks must have shape [T] or [B,T], got '
                    f'{tuple(valid_mask.shape)}.')
        else:
            if index >= len(valid_mask):
                raise ValueError(
                    f'Validity mask has {len(valid_mask)} samples, cannot '
                    f'read sample {index}.')
            sample = valid_mask[index]

        sample = torch.as_tensor(sample, device=device, dtype=torch.bool)
        if sample.ndim != 1:
            raise ValueError(
                'Each validity mask must be 1-D, got '
                f'{tuple(sample.shape)} for sample {index}.')
        if expected_length is not None and sample.numel() != expected_length:
            raise ValueError(
                f'Validity mask length {sample.numel()} does not match '
                f'prediction length {expected_length} for sample {index}.')
        return sample

    def _compute_action_loss(
        self,
        preds_action: List[torch.Tensor],
        target_action: Optional[List[torch.Tensor]],
        raw_action_dim: Optional[List[torch.Tensor]],
        condition_mask: Optional[List[torch.Tensor]] = None,
        valid_mask=None,
    ) -> torch.Tensor:
        if target_action is None:
            if preds_action:
                return 0.0 * sum(pred.sum() for pred in preds_action)
            return next(self.parameters()).sum() * 0.0
        if not preds_action:
            return next(self.parameters()).sum() * 0.0

        normalize_by_active = bool(
            self.rectified_flow_training_config['normalize_loss_by_active'])
        losses = []
        for index, (pred,
                    target) in enumerate(zip(preds_action, target_action)):
            valid_dim = int(raw_action_dim[index].item()
                            ) if raw_action_dim is not None else pred.shape[-1]
            sqerr = (pred[:, :valid_dim].float() -
                     target[:, :valid_dim].float())**2
            sample_valid = self._sample_valid_mask(
                valid_mask,
                index,
                expected_length=pred.shape[0],
                device=sqerr.device,
            )
            if condition_mask is None and sample_valid is None:
                losses.append(sqerr.mean())
                continue
            if condition_mask is None:
                noisy_mask = torch.ones(
                    (pred.shape[0], 1),
                    device=sqerr.device,
                    dtype=sqerr.dtype,
                )
            else:
                noisy_mask = 1.0 - condition_mask[index].to(
                    device=sqerr.device, dtype=sqerr.dtype)
                noisy_mask = noisy_mask.reshape(pred.shape[0], 1)
            if sample_valid is not None:
                noisy_mask = noisy_mask * sample_valid.to(
                    dtype=sqerr.dtype).view(-1, 1)
            if normalize_by_active or sample_valid is not None:
                active_count = (noisy_mask.sum() *
                                (sqerr.numel() // noisy_mask.numel())).clamp(
                                    min=1)
                losses.append((sqerr * noisy_mask).sum() / active_count)
            else:
                losses.append((sqerr * noisy_mask).mean())
        return torch.stack(losses).mean() if losses else next(
            self.parameters()).sum() * 0.0

    def _compute_vision_loss(
        self,
        preds_vision: List[torch.Tensor],
        target_vision: Optional[List[torch.Tensor]],
        condition_mask: Optional[List[torch.Tensor]] = None,
        valid_mask=None,
    ) -> torch.Tensor:
        if target_vision is None:
            if preds_vision:
                return 0.0 * sum(pred.sum() for pred in preds_vision)
            return next(self.parameters()).sum() * 0.0
        if not preds_vision:
            return next(self.parameters()).sum() * 0.0
        normalize_by_active = bool(
            self.rectified_flow_training_config['normalize_loss_by_active'])
        losses = []
        for index, (pred,
                    target) in enumerate(zip(preds_vision, target_vision)):
            sqerr = (pred.float() - target.float())**2
            latent_t = int(pred.shape[2])
            sample_valid = self._sample_valid_mask(
                valid_mask,
                index,
                device=sqerr.device,
            )
            if sample_valid is not None and sample_valid.numel() != latent_t:
                # The input mask is in pixel-frame time while the loss is in
                # causal-VAE latent time.  A latent timestep is valid only if
                # every source frame in its temporal bin is valid.
                sample_valid = F.adaptive_avg_pool1d(
                    sample_valid.float().view(1, 1, -1), latent_t,
                ).view(-1).ge(1.0 - 1e-6)
            if condition_mask is None and sample_valid is None:
                losses.append(sqerr.mean())
                continue
            if condition_mask is None:
                noisy_mask = torch.ones(
                    latent_t, device=sqerr.device, dtype=sqerr.dtype)
            else:
                noisy_mask = 1.0 - condition_mask[index].to(
                    device=sqerr.device, dtype=sqerr.dtype)
                noisy_mask = noisy_mask.reshape(latent_t)
            if sample_valid is not None:
                noisy_mask = noisy_mask * sample_valid.to(dtype=sqerr.dtype)
            noisy_mask = noisy_mask.view(1, 1, latent_t, 1, 1)
            if normalize_by_active or sample_valid is not None:
                active_count = (noisy_mask.sum() *
                                (sqerr.numel() // noisy_mask.numel())).clamp(
                                    min=1)
                losses.append((sqerr * noisy_mask).sum() / active_count)
            else:
                losses.append((sqerr * noisy_mask).mean())
        return torch.stack(losses).mean() if losses else next(
            self.parameters()).sum() * 0.0

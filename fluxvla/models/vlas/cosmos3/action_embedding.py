# Copyright 2026 Limx Dynamics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Checkpoint-compatible action modality embedding."""

from __future__ import annotations

import torch
import torch.nn as nn


class ActionModalityEmbedding(nn.Module):
    """Expose the embedding as ``action_modality_embed.weight``.

    Existing FluxVLA Edge fine-tuning checkpoints use this state-dict key.
    """

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(hidden_size))

    def forward(self) -> torch.Tensor:
        return self.weight


__all__ = ['ActionModalityEmbedding']

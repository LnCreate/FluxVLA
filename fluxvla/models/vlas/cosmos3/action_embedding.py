# Copyright 2026 Limx Dynamics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""PEFT-compatible action modality embedding."""

from __future__ import annotations

import torch
import torch.nn as nn


class ActionModalityEmbedding(nn.Module):
    """One-token module so PEFT can train, save, and reload the embedding."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(hidden_size))

    def forward(self) -> torch.Tensor:
        return self.weight


__all__ = ['ActionModalityEmbedding']

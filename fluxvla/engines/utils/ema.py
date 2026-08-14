# Copyright 2026 Limx Dynamics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, Iterator

import numpy as np
import torch


class ShardedPowerEMA:
    """FP32 Power-EMA shadows for the local trainable parameter shards."""

    def __init__(self,
                 model: torch.nn.Module,
                 rate: float = 0.1,
                 iteration_shift: int = 0) -> None:
        if rate <= 0:
            raise ValueError(f'EMA rate must be positive, got {rate}.')
        self.rate = float(rate)
        self.iteration_shift = int(iteration_shift)
        self.exp_coefficient = float(
            np.roots([1, 7, 16 - rate**-2, 12 - rate**-2]).real.max())
        self.shadow: Dict[str, torch.Tensor] = {
            name: param.detach().float().clone()
            for name, param in model.named_parameters() if param.requires_grad
        }
        self._is_swapped = False

    def beta(self, iteration: int) -> float:
        iteration = int(iteration) + self.iteration_shift
        if iteration < 1:
            return 0.0
        return float((1 - 1 / (iteration + 1))**(self.exp_coefficient + 1))

    @torch.no_grad()
    def update(self, model: torch.nn.Module, iteration: int) -> None:
        beta = self.beta(iteration)
        parameters = dict(model.named_parameters())
        for name, shadow in self.shadow.items():
            parameter = parameters[name].detach()
            if parameter.shape != shadow.shape:
                raise RuntimeError(
                    f'EMA shard shape changed for {name}: '
                    f'{tuple(shadow.shape)} -> {tuple(parameter.shape)}.')
            shadow.mul_(beta).add_(parameter.float(), alpha=1.0 - beta)

    @torch.no_grad()
    def copy_from_model(self, model: torch.nn.Module) -> None:
        parameters = dict(model.named_parameters())
        for name, shadow in self.shadow.items():
            shadow.copy_(parameters[name].detach().float())

    @contextmanager
    @torch.no_grad()
    def swap_into(self, model: torch.nn.Module) -> Iterator[None]:
        if self._is_swapped:
            raise RuntimeError('EMA weights are already active.')
        parameters = dict(model.named_parameters())
        regular = {
            name: parameters[name].detach().clone()
            for name in self.shadow
        }
        self._is_swapped = True
        try:
            for name, shadow in self.shadow.items():
                parameters[name].copy_(shadow.to(dtype=parameters[name].dtype))
            yield
        finally:
            for name, value in regular.items():
                parameters[name].copy_(value)
            self._is_swapped = False

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

import unittest

import torch

from fluxvla.engines.utils.ema import ShardedPowerEMA


class TestShardedPowerEMA(unittest.TestCase):

    def test_first_update_copies_trainable_weights(self):
        model = torch.nn.Linear(2, 1)
        ema = ShardedPowerEMA(model, rate=0.1)
        with torch.no_grad():
            model.weight.fill_(3.0)
            model.bias.fill_(4.0)

        ema.update(model, iteration=0)

        torch.testing.assert_close(ema.shadow['weight'], model.weight)
        torch.testing.assert_close(ema.shadow['bias'], model.bias)

    def test_swap_restores_regular_weights(self):
        model = torch.nn.Linear(2, 1, bias=False)
        ema = ShardedPowerEMA(model, rate=0.1)
        regular = model.weight.detach().clone()
        ema.shadow['weight'].fill_(7.0)

        with ema.swap_into(model):
            torch.testing.assert_close(model.weight,
                                       torch.full_like(model.weight, 7.0))

        torch.testing.assert_close(model.weight, regular)


if __name__ == '__main__':
    unittest.main()

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

import os
import subprocess
import sys
import unittest

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullStateDictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy, StateDictType

from fluxvla.engines.utils.ema import ShardedPowerEMA


def _gather_full_state_dict(model):
    policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, policy):
        return model.state_dict()


def _distributed_worker():
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend='nccl')
    try:
        torch.manual_seed(7)
        model = torch.nn.Sequential(
            torch.nn.Linear(8, 16),
            torch.nn.GELU(),
            torch.nn.Linear(16, 4),
        ).cuda(local_rank)
        model = FSDP(
            model,
            device_id=local_rank,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            use_orig_params=True,
        )
        ema = ShardedPowerEMA(model, rate=0.1)

        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(float(local_rank + 1))
        ema.update(model, iteration=0)

        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(10.0)
        regular_full = _gather_full_state_dict(model)
        with ema.swap_into(model):
            ema_full = _gather_full_state_dict(model)

        if dist.get_rank() == 0:
            for name in regular_full:
                torch.testing.assert_close(
                    regular_full[name],
                    ema_full[name] + 10.0,
                )

        payload = [ema_full if dist.get_rank() == 0 else None]
        dist.broadcast_object_list(payload, src=0)
        ema_full = payload[0]

        restored_ema = ShardedPowerEMA(model, rate=0.1)
        with restored_ema.swap_into(model):
            policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT,
                                      policy):
                model.load_state_dict(ema_full, strict=True)
            restored_ema.copy_from_model(model)

        regular_after_restore = _gather_full_state_dict(model)
        with restored_ema.swap_into(model):
            ema_after_restore = _gather_full_state_dict(model)

        if dist.get_rank() == 0:
            for name in regular_full:
                torch.testing.assert_close(regular_after_restore[name],
                                           regular_full[name])
                torch.testing.assert_close(ema_after_restore[name],
                                           ema_full[name])
    finally:
        dist.destroy_process_group()


@unittest.skipUnless(torch.cuda.device_count() >= 2,
                     'FSDP EMA integration test requires two CUDA devices.')
class TestShardedPowerEMAFSDP(unittest.TestCase):

    def test_update_swap_and_restore_on_two_ranks(self):
        env = os.environ.copy()
        env.setdefault('OMP_NUM_THREADS', '1')
        subprocess.run(
            [
                sys.executable,
                '-m',
                'torch.distributed.run',
                '--standalone',
                '--nproc-per-node=2',
                __file__,
            ],
            check=True,
            env=env,
            timeout=120,
        )


if __name__ == '__main__':
    _distributed_worker()

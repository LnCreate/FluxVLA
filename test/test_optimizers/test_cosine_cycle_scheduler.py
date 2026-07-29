from types import SimpleNamespace

import pytest
import torch

from fluxvla.optimizers.lr_scheduler_policies import (
    LinearWarmupCosineCycleLRScheduler,
)


def test_cosine_cycle_can_extend_beyond_training_run():
    parameter = torch.nn.Parameter(torch.ones(()))
    optimizer = torch.optim.AdamW([parameter], lr=4e-5)
    policy = LinearWarmupCosineCycleLRScheduler(
        warmup_steps=500, cycle_steps=16_000)
    scheduler = policy.build_scheduler(
        SimpleNamespace(num_training_steps=10_000), optimizer)

    assert optimizer.param_groups[0]['lr'] == 0.0
    for _ in range(10_000):
        optimizer.step()
        scheduler.step()
    assert scheduler.get_last_lr()[0] == pytest.approx(
        1.30538949e-5, rel=1e-6)


def test_cosine_cycle_rejects_cycle_shorter_than_run():
    parameter = torch.nn.Parameter(torch.ones(()))
    optimizer = torch.optim.AdamW([parameter], lr=4e-5)
    policy = LinearWarmupCosineCycleLRScheduler(
        warmup_steps=500, cycle_steps=2_000)

    with pytest.raises(ValueError, match='shorter than the training run'):
        policy.build_scheduler(
            SimpleNamespace(num_training_steps=10_000), optimizer)

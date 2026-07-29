from types import SimpleNamespace

import pytest
import torch

from fluxvla.engines.runners.base_train_runner import (
    BaseTrainRunner, OptimizerParamGroupTopologyError,
    SchedulerParamGroupTopologyError, build_optimizer_param_group_topology,
    load_scheduler_state_strict, validate_optimizer_param_group_topology)


class _ToyModel(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.first = torch.nn.Parameter(torch.tensor([1.0]))
        self.second = torch.nn.Parameter(torch.tensor([2.0]))
        self.third = torch.nn.Parameter(torch.tensor([3.0]))
        self.fourth = torch.nn.Parameter(torch.tensor([4.0]))


def _two_group_optimizer(model):
    return torch.optim.AdamW([
        {
            'params': [model.first, model.second],
            'lr': 2e-4,
        },
        {
            'params': [model.third, model.fourth],
            'lr': 2e-4,
        },
    ])


def _four_group_optimizer(model):
    return torch.optim.AdamW([
        {
            'params': [model.first],
            'lr': 4e-5,
        },
        {
            'params': [model.second],
            'lr': 4e-5,
        },
        {
            'params': [model.third],
            'lr': 2e-4,
        },
        {
            'params': [model.fourth],
            'lr': 2e-4,
        },
    ])


def _scheduler(optimizer):
    return torch.optim.lr_scheduler.LambdaLR(optimizer, [lambda _: 1.0] *
                                             len(optimizer.param_groups))


def test_resume_rejects_two_group_checkpoint_for_four_group_optimizer():
    checkpoint_model = _ToyModel()
    checkpoint_optimizer = _two_group_optimizer(checkpoint_model)
    checkpoint_topology = build_optimizer_param_group_topology(
        checkpoint_model, checkpoint_optimizer)
    checkpoint_scheduler_state = _scheduler(checkpoint_optimizer).state_dict()

    current_model = _ToyModel()
    current_optimizer = _four_group_optimizer(current_model)
    current_topology = build_optimizer_param_group_topology(
        current_model, current_optimizer)

    with pytest.raises(
            OptimizerParamGroupTopologyError,
            match='param-group count mismatch'):
        validate_optimizer_param_group_topology(checkpoint_topology,
                                                current_topology)

    current_scheduler = _scheduler(current_optimizer)
    with pytest.raises(
            SchedulerParamGroupTopologyError, match='group count mismatch'):
        load_scheduler_state_strict(current_scheduler,
                                    checkpoint_scheduler_state,
                                    current_optimizer)


def test_resume_restores_same_optimizer_and_scheduler_topology():
    checkpoint_model = _ToyModel()
    checkpoint_optimizer = _two_group_optimizer(checkpoint_model)
    checkpoint_scheduler = _scheduler(checkpoint_optimizer)
    loss = sum(parameter.square().sum()
               for parameter in checkpoint_model.parameters())
    loss.backward()
    checkpoint_optimizer.step()
    checkpoint_scheduler.step()

    current_model = _ToyModel()
    current_optimizer = _two_group_optimizer(current_model)
    current_scheduler = _scheduler(current_optimizer)
    current_topology = build_optimizer_param_group_topology(
        current_model, current_optimizer)

    def load_optimizer(state_dict):
        current_optimizer.load_state_dict(state_dict)
        return True

    runner = SimpleNamespace(
        optimizer=current_optimizer,
        lr_scheduler=current_scheduler,
        optimizer_state_loaded=False,
        _current_checkpoint_info=None,
        _optimizer_param_group_topology=lambda: current_topology,
        _load_optimizer_state=load_optimizer,
    )
    checkpoint = dict(
        optimizer_state_dict=checkpoint_optimizer.state_dict(),
        optimizer_param_group_topology=build_optimizer_param_group_topology(
            checkpoint_model, checkpoint_optimizer),
        scheduler_state_dict=checkpoint_scheduler.state_dict(),
    )
    BaseTrainRunner._restore_optimizer_and_scheduler(runner, checkpoint)

    assert len(current_optimizer.state) == len(checkpoint_optimizer.state)
    assert runner.optimizer_state_loaded is True
    assert current_scheduler.last_epoch == checkpoint_scheduler.last_epoch
    current_optimizer.step()
    current_scheduler.step()
    assert len(current_scheduler.get_last_lr()) == 2


class _RecordingScheduler:

    def __init__(self):
        self.loaded = False

    def load_state_dict(self, state_dict):
        self.loaded = True

    def state_dict(self):
        return {'base_lrs': [2e-4, 2e-4], '_last_lr': [2e-4, 2e-4]}


def test_failed_optimizer_restore_does_not_load_scheduler():
    model = _ToyModel()
    optimizer = _two_group_optimizer(model)
    topology = build_optimizer_param_group_topology(model, optimizer)
    scheduler = _RecordingScheduler()
    runner = SimpleNamespace(
        optimizer=optimizer,
        lr_scheduler=scheduler,
        optimizer_state_loaded=False,
        _current_checkpoint_info=None,
        _optimizer_param_group_topology=lambda: topology,
        _load_optimizer_state=lambda _: False,
    )
    checkpoint = {
        'optimizer_state_dict': {},
        'optimizer_param_group_topology': topology,
        'scheduler_state_dict': scheduler.state_dict(),
    }

    with pytest.raises(RuntimeError, match='Optimizer state restoration'):
        BaseTrainRunner._restore_optimizer_and_scheduler(runner, checkpoint)

    assert scheduler.loaded is False

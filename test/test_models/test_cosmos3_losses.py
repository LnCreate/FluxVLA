from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest
import torch
from torch import nn


def _load_loss_mixin():
    path = (Path(__file__).parents[2] /
            'fluxvla/models/vlas/cosmos3/loss_mixin.py')
    spec = spec_from_file_location('_cosmos3_loss_mixin_test', path)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.Cosmos3LossMixin


Cosmos3LossMixin = _load_loss_mixin()


class _LossModel(Cosmos3LossMixin, nn.Module):

    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.rectified_flow_training_config = {
            'normalize_loss_by_active': True,
        }


def test_action_validity_mask_excludes_padded_rows_and_dimensions():
    model = _LossModel()
    pred = torch.tensor([
        [1.0, 1.0, 999.0],
        [1.0, 1.0, 999.0],
        [1000.0, 1000.0, 999.0],
        [1000.0, 1000.0, 999.0],
    ])
    target = torch.zeros_like(pred)

    loss = model._compute_action_loss(
        [pred],
        [target],
        [torch.tensor(2)],
        [torch.zeros(4, 1)],
        valid_mask=torch.tensor([[True, True, False, False]]),
    )

    torch.testing.assert_close(loss, torch.tensor(1.0))


def test_conditioned_state_and_invalid_tail_do_not_contribute_action_loss():
    model = _LossModel()
    pred = torch.tensor([[1000.0], [2.0], [2.0], [1000.0]])

    loss = model._compute_action_loss(
        [pred],
        [torch.zeros_like(pred)],
        [torch.tensor(1)],
        [torch.tensor([[1.0], [0.0], [0.0], [0.0]])],
        valid_mask=torch.tensor([[True, True, True, False]]),
    )

    torch.testing.assert_close(loss, torch.tensor(4.0))


def test_frame_mask_is_conservatively_downsampled_to_vae_latent_time():
    model = _LossModel()
    pred = torch.tensor([1.0, 1000.0]).reshape(1, 1, 2, 1, 1)
    target = torch.zeros_like(pred)

    loss = model._compute_vision_loss(
        [pred],
        [target],
        [torch.zeros(2, 1, 1)],
        valid_mask=torch.tensor([[True, True, True, False, False]]),
    )

    torch.testing.assert_close(loss, torch.tensor(1.0))


def test_action_mask_length_mismatch_fails_fast():
    model = _LossModel()
    with pytest.raises(ValueError, match='does not match prediction length'):
        model._compute_action_loss(
            [torch.zeros(4, 2)],
            [torch.zeros(4, 2)],
            [torch.tensor(2)],
            valid_mask=torch.ones(1, 3, dtype=torch.bool),
        )

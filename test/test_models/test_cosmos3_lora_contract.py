from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import torch
import pytest
from peft import LoraConfig, PeftModel, get_peft_model
from torch import nn


def _action_embedding_class():
    path = (Path(__file__).parents[2] /
            'fluxvla/models/vlas/cosmos3/action_embedding.py')
    spec = spec_from_file_location('_cosmos3_action_embedding_test', path)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.ActionModalityEmbedding


def _checkpoint_helpers():
    path = (Path(__file__).parents[2] /
            'fluxvla/engines/runners/peft_checkpoint.py')
    spec = spec_from_file_location('_cosmos3_peft_checkpoint_test', path)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class _TinyPolicy(nn.Module):

    def __init__(self, embedding_cls):
        super().__init__()
        self.backbone = nn.Linear(4, 4, bias=False)
        self.action_in_proj = nn.Linear(4, 4, bias=False)
        self.action_out_proj = nn.Linear(4, 4, bias=False)
        self.action_modality_embed = embedding_cls(4)

    def forward(self, values):
        hidden = self.backbone(values) + self.action_modality_embed()
        return self.action_out_proj(self.action_in_proj(hidden))


def _wrap(model):
    return get_peft_model(
        model,
        LoraConfig(
            r=2,
            lora_alpha=4,
            target_modules=['backbone'],
            modules_to_save=[
                'action_in_proj',
                'action_out_proj',
                'action_modality_embed',
            ],
        ),
    )


def test_lora_keeps_complete_action_interface_trainable_and_reloadable(
        tmp_path):
    embedding_cls = _action_embedding_class()
    model = _wrap(_TinyPolicy(embedding_cls))
    trainable = {
        name for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    assert any('action_in_proj.modules_to_save' in name for name in trainable)
    assert any('action_out_proj.modules_to_save' in name for name in trainable)
    assert any(
        'action_modality_embed.modules_to_save' in name
        for name in trainable)

    saved_value = torch.full((4, ), 0.25)
    with torch.no_grad():
        model.base_model.model.action_modality_embed.modules_to_save[
            'default'].weight.copy_(saved_value)
    model.save_pretrained(tmp_path)

    reloaded = PeftModel.from_pretrained(
        _TinyPolicy(embedding_cls), tmp_path)
    torch.testing.assert_close(
        reloaded.base_model.model.action_modality_embed.modules_to_save[
            'default'].weight,
        saved_value,
    )


def test_peft_step_state_resumes_exactly_and_rejects_truncation():
    embedding_cls = _action_embedding_class()
    helpers = _checkpoint_helpers()
    torch.manual_seed(3)
    base = _TinyPolicy(embedding_cls)
    base_state = {
        key: value.clone() for key, value in base.state_dict().items()
    }
    trained = _wrap(base)
    with torch.no_grad():
        for name, parameter in trained.named_parameters():
            if parameter.requires_grad:
                parameter.add_(0.125)
    training_state = helpers.get_peft_training_state(trained)

    fresh_base = _TinyPolicy(embedding_cls)
    fresh_base.load_state_dict(base_state)
    resumed = _wrap(fresh_base)
    helpers.load_peft_training_state(resumed, training_state)

    resumed_state = helpers.get_peft_training_state(resumed)
    assert resumed_state.keys() == training_state.keys()
    for key in training_state:
        torch.testing.assert_close(resumed_state[key], training_state[key])
    inputs = torch.randn(2, 4)
    torch.testing.assert_close(resumed(inputs), trained(inputs))

    truncated = dict(training_state)
    truncated.pop(next(iter(truncated)))
    with pytest.raises(ValueError, match='contract mismatch'):
        helpers.load_peft_training_state(resumed, truncated)

# Copyright 2026 Limx Dynamics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Small PEFT training-checkpoint helpers, independent of runner setup."""

from peft import (PeftModel, get_peft_model_state_dict,
                  set_peft_model_state_dict)


def get_peft_training_state(model: PeftModel) -> dict:
    """Return adapter/modules-to-save tensors needed for exact resume."""
    return {
        key: value.detach().cpu().clone()
        for key, value in get_peft_model_state_dict(model).items()
    }


def load_peft_training_state(model: PeftModel, state_dict: dict) -> None:
    """Restore PEFT state with an exact adapter-key contract."""
    expected = set(get_peft_model_state_dict(model))
    provided = set(state_dict)
    if expected != provided:
        raise ValueError(
            'PEFT resume tensor contract mismatch: '
            f'missing={sorted(expected - provided)[:20]}, '
            f'unexpected={sorted(provided - expected)[:20]}.')
    incompatible = set_peft_model_state_dict(model, state_dict)
    if incompatible.unexpected_keys:
        raise ValueError(
            'PEFT resume contains unexpected tensors: '
            f'{incompatible.unexpected_keys[:20]}')


__all__ = ['get_peft_training_state', 'load_peft_training_state']

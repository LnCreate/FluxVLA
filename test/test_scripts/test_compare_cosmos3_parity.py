from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pytest


def _load_module():
    path = Path(__file__).parents[2] / 'scripts/compare_cosmos3_parity.py'
    spec = spec_from_file_location('_compare_cosmos3_parity_test', path)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _bundle(path, velocity, input_action=None):
    payload = {
        'velocity_action': np.asarray(velocity, dtype=np.float32),
        'text_token_ids': np.asarray([[1, 2]], dtype=np.int64),
        'timestep': np.asarray(0.5, dtype=np.float32),
        'embodiment_id': np.asarray(8, dtype=np.int64),
        'raw_action_dim': np.asarray(8, dtype=np.int64),
        'sequence_has_text': np.asarray(True),
        'sequence_has_vision': np.asarray(True),
        'sequence_has_action': np.asarray(True),
        'sequence_condition_vision': np.asarray([0], dtype=np.int64),
        'sequence_condition_action': np.asarray([0], dtype=np.int64),
        'sequence_action_start_offset': np.asarray(0, dtype=np.int64),
    }
    if input_action is not None:
        payload['input_action'] = np.asarray(input_action, dtype=np.float32)
    np.savez(path, **payload)


def test_parity_comparator_checks_velocity_and_exact_inputs(tmp_path):
    module = _load_module()
    reference = tmp_path / 'official.npz'
    candidate = tmp_path / 'flux.npz'
    _bundle(reference, [1.0, 2.0], [0.5])
    _bundle(candidate, [1.001, 1.999], [0.5])

    report = module.compare_bundles(reference, candidate)

    assert report['ok'] is True
    assert report['velocities']['velocity_action']['max_abs_error'] < 0.01


def test_parity_comparator_rejects_input_or_velocity_drift(tmp_path):
    module = _load_module()
    reference = tmp_path / 'official.npz'
    candidate = tmp_path / 'flux.npz'
    _bundle(reference, [1.0], [0.5])
    _bundle(candidate, [2.0], [0.6])

    report = module.compare_bundles(reference, candidate)

    assert report['ok'] is False
    assert report['input_mismatches'] == ['input_action']
    assert report['checks']['velocities_close'] is False


@pytest.mark.parametrize(
    ('reference_input', 'candidate_input', 'message'),
    [
        (None, None, 'input_vision and/or input_action'),
        ([0.5], None, 'same non-empty input key set'),
    ],
)
def test_parity_comparator_requires_same_nonempty_input_keys(
        tmp_path, reference_input, candidate_input, message):
    module = _load_module()
    reference = tmp_path / 'official.npz'
    candidate = tmp_path / 'flux.npz'
    _bundle(reference, [1.0], reference_input)
    _bundle(candidate, [1.0], candidate_input)

    with pytest.raises(ValueError, match=message):
        module.compare_bundles(reference, candidate)


def test_parity_comparator_requires_complete_transformer_inputs(tmp_path):
    module = _load_module()
    reference = tmp_path / 'official.npz'
    candidate = tmp_path / 'flux.npz'
    _bundle(reference, [1.0], [0.5])
    _bundle(candidate, [1.0], [0.5])
    for path in (reference, candidate):
        with np.load(path, allow_pickle=False) as bundle:
            payload = {
                key: bundle[key]
                for key in bundle.files if key != 'timestep'
            }
        np.savez(path, **payload)

    with pytest.raises(ValueError, match='missing required'):
        module.compare_bundles(reference, candidate)

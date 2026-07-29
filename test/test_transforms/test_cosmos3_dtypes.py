import numpy as np

from fluxvla.transforms.normalize import (LiberoProprioFromInputs,
                                          NormalizeStatesAndActions)
from fluxvla.transforms.transform_images import SimpleNormalizeImages


def test_simple_normalize_images_keeps_float32():
    transform = SimpleNormalizeImages()
    images = np.zeros((17, 3, 8, 8), dtype=np.uint8)

    output = transform({'images': images})['images']

    assert output.shape == (51, 8, 8)
    assert output.dtype == np.float32


def test_state_action_normalization_keeps_float32():
    transform = NormalizeStatesAndActions(
        state_key='state', action_key='action', norm_type='mean_std')
    stats = {
        'state': {
            'mean': [0.0, 0.0],
            'std': [1.0, 1.0],
        },
        'action': {
            'mean': [0.0, 0.0],
            'std': [1.0, 1.0],
        },
    }

    output = transform({
        'states': np.ones(2, dtype=np.float32),
        'actions': np.ones((3, 2), dtype=np.float32),
        'stats': stats,
    })

    assert output['states'].dtype == np.float32
    assert output['actions'].dtype == np.float32


def test_libero_proprio_keeps_float32_with_float64_stats():
    transform = LiberoProprioFromInputs(norm_type='mean_std', state_dim=8)
    stats = {
        'mean': [0.0] * 8,
        'std': [1.0] * 8,
        'mask': [True] * 8,
    }
    output = transform({
        'robot0_eef_pos': np.zeros(3, dtype=np.float32),
        'robot0_eef_quat': np.array([0, 0, 0, 1], dtype=np.float32),
        'robot0_gripper_qpos': np.zeros(2, dtype=np.float32),
        'norm_stats': {
            'proprio': stats,
        },
    })

    assert output['states'].dtype == np.float32

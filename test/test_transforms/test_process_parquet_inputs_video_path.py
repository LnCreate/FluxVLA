from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType
import sys

import numpy as np
import pytest
import torch


class _Registry:

    @staticmethod
    def register_module():

        def decorator(cls):
            return cls

        return decorator


def _package(name):
    module = ModuleType(name)
    module.__path__ = []
    return module


def _load_module(name, path):
    spec = spec_from_file_location(name, path)
    module = module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope='module')
def transform_inputs_module():
    """Load the transform and shared path resolver without GPU imports."""
    root = Path(__file__).parents[2]
    names = [
        'av',
        'torchvision',
        'fluxvla',
        'fluxvla.datasets',
        'fluxvla.datasets.utils',
        'fluxvla.datasets.utils.video_decode',
        'fluxvla.engines',
        'fluxvla.engines.utils',
        'fluxvla.engines.utils.eval_utils',
        'fluxvla.transforms',
        'fluxvla.transforms.transform_images',
        'fluxvla.transforms.transform_inputs',
        'fluxvla.transforms.utils',
    ]
    previous = {name: sys.modules.get(name) for name in names}

    sys.modules['av'] = ModuleType('av')
    sys.modules['torchvision'] = ModuleType('torchvision')
    sys.modules['fluxvla'] = _package('fluxvla')
    sys.modules['fluxvla.datasets'] = _package('fluxvla.datasets')
    sys.modules['fluxvla.datasets.utils'] = _package('fluxvla.datasets.utils')
    sys.modules['fluxvla.transforms'] = _package('fluxvla.transforms')
    sys.modules['fluxvla.engines.utils'] = _package('fluxvla.engines.utils')

    engines = ModuleType('fluxvla.engines')
    engines.TRANSFORMS = _Registry()
    engines.initialize_overwatch = lambda name: object()
    sys.modules['fluxvla.engines'] = engines

    eval_utils = ModuleType('fluxvla.engines.utils.eval_utils')
    eval_utils.crop_and_resize = lambda *args, **kwargs: None
    sys.modules['fluxvla.engines.utils.eval_utils'] = eval_utils

    transform_images = ModuleType('fluxvla.transforms.transform_images')
    transform_images._resize_hwc_lanczos3_numpy = (
        lambda image, height, width: image)
    transform_images._resize_hwc_lanczos3_tensorflow = (
        lambda image, height, width, jpeg_roundtrip=False: image)
    sys.modules['fluxvla.transforms.transform_images'] = transform_images

    transform_utils = ModuleType('fluxvla.transforms.utils')
    transform_utils.pad_to_dim = lambda value, dim: value
    transform_utils.parse_image = lambda value: value
    sys.modules['fluxvla.transforms.utils'] = transform_utils

    _load_module('fluxvla.datasets.utils.video_decode',
                 root / 'fluxvla/datasets/utils/video_decode.py')
    module = _load_module('fluxvla.transforms.transform_inputs',
                          root / 'fluxvla/transforms/transform_inputs.py')
    yield module

    for name, previous_module in previous.items():
        if previous_module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous_module


def _run_transform(transform_inputs_module, monkeypatch, data):
    transform = transform_inputs_module.ProcessParquetInputs(
        parquet_keys=['timestamp'],
        video_keys=['observation.images.front'],
    )
    decoded_paths = []
    decoded_timestamps = []

    def fake_decode(video_path, timestamps, tolerance_s):
        decoded_paths.append(video_path)
        decoded_timestamps.append(list(timestamps))
        return torch.zeros((len(timestamps), 3, 2, 2), dtype=torch.uint8)

    monkeypatch.setattr(transform_inputs_module.os.path, 'exists',
                        lambda path: True)
    monkeypatch.setattr(transform, 'decode_video_frames_torchvision',
                        fake_decode)

    output = transform(data)
    assert len(output['images']) == 1
    return decoded_paths, decoded_timestamps


def test_process_parquet_inputs_keeps_v21_video_paths(transform_inputs_module,
                                                      monkeypatch):
    paths, timestamps = _run_transform(
        transform_inputs_module,
        monkeypatch,
        {
            'timestamp': 0.0,
            'episode_index': 1234,
            'data_root': '/dataset',
            'info': {
                'chunks_size':
                1000,
                'video_path': ('videos/chunk-{episode_chunk:03d}/{video_key}/'
                               'episode_{episode_index:06d}.mp4'),
            },
        },
    )

    assert paths == [
        '/dataset/videos/chunk-001/observation.images.front/'
        'episode_001234.mp4'
    ]
    assert timestamps == [[0.0]]


def test_process_parquet_inputs_resolves_native_v3_video_paths(
        transform_inputs_module, monkeypatch):
    video_key = 'observation.images.front'
    paths, timestamps = _run_transform(
        transform_inputs_module,
        monkeypatch,
        {
            'timestamp': 0.0,
            'episode_index': np.int64(42),
            'data_root': '/dataset',
            'info': {
                'video_path': ('videos/{video_key}/chunk-{chunk_index:03d}/'
                               'file-{file_index:03d}.mp4'),
            },
            'episode_meta': {
                f'videos/{video_key}/chunk_index': 4,
                f'videos/{video_key}/file_index': 9,
                f'videos/{video_key}/from_timestamp': 12.5,
            },
        },
    )

    assert paths == [
        '/dataset/videos/observation.images.front/chunk-004/file-009.mp4'
    ]
    assert timestamps == [[12.5]]

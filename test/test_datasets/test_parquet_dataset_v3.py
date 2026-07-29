from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType
import sys

import numpy as np
import pytest


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
def parquet_v3_module():
    """Load dataset classes without importing GPU model registrations."""
    root = Path(__file__).parents[2]
    names = [
        'fluxvla',
        'fluxvla.datasets',
        'fluxvla.engines',
        'fluxvla.datasets.parquet_dataset',
        'fluxvla.datasets.parquet_dataset_v3',
    ]
    previous = {name: sys.modules.get(name) for name in names}
    sys.modules['fluxvla'] = _package('fluxvla')
    sys.modules['fluxvla.datasets'] = _package('fluxvla.datasets')
    engines = ModuleType('fluxvla.engines')
    engines.DATASETS = _Registry()
    engines.build_transform_from_cfg = lambda config: config
    sys.modules['fluxvla.engines'] = engines

    _load_module(
        'fluxvla.datasets.parquet_dataset',
        root / 'fluxvla/datasets/parquet_dataset.py')
    module = _load_module(
        'fluxvla.datasets.parquet_dataset_v3',
        root / 'fluxvla/datasets/parquet_dataset_v3.py')
    yield module

    for name, previous_module in previous.items():
        if previous_module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous_module


def _dataset_with_episode_boundary(parquet_v3_module, require_full_window):
    dataset_cls = parquet_v3_module.ParquetDatasetV3
    dataset = dataset_cls.__new__(dataset_cls)
    dataset.dataset = [
        {
            'episode_index': 0,
            'task': 'move',
        } for _ in range(6)
    ] + [{
        'episode_index': 1,
        'task': 'move',
    } for _ in range(6)]
    dataset.dataset_cumulative_sizes = np.array([0, len(dataset.dataset)])
    dataset.require_full_window = require_full_window
    dataset.window_start_idx = 1
    dataset.action_window_size = 2
    dataset.frame_window_size = 3
    dataset.frame_sample_stride = 3
    return dataset


def test_v3_full_window_accounts_for_frame_stride(parquet_v3_module):
    dataset = _dataset_with_episode_boundary(
        parquet_v3_module, require_full_window=True)

    # The action window ends at row 2, but the strided frame window ends at
    # row 6 in the next episode (offsets 0, 3, 6).
    assert dataset._invalid_start_index(0, 0, dataset.dataset[0])


def test_v3_can_keep_padded_tail_window_when_not_required(parquet_v3_module):
    dataset = _dataset_with_episode_boundary(
        parquet_v3_module, require_full_window=False)

    assert not dataset._invalid_start_index(0, 0, dataset.dataset[0])


def test_v3_full_window_accounts_for_skipped_static_rows(parquet_v3_module):
    dataset = _dataset_with_episode_boundary(
        parquet_v3_module, require_full_window=True)
    dataset.dataset = dataset.dataset[:4] + dataset.dataset[6:]
    dataset.dataset[2]['task'] = 'static'
    dataset.dataset_cumulative_sizes = np.array([0, len(dataset.dataset)])
    dataset.action_window_size = 3
    dataset.frame_window_size = 1

    # Nominal action offsets 1..3 fit episode 0. Skipping static row 2 needs
    # offset 4, which is already in episode 1 and must therefore be rejected.
    assert dataset._invalid_start_index(0, 0, dataset.dataset[0])


def test_v3_task_indices_select_single_task(parquet_v3_module):
    dataset_cls = parquet_v3_module.ParquetDatasetV3
    dataset = dataset_cls.__new__(dataset_cls)
    dataset.dataset = [
        {'task_index': 0},
        {'task_index': 1},
        {'task': np.array(0)},
    ]

    selected = dataset._filter_task_indices(
        np.arange(3, dtype=np.int64), [0])

    np.testing.assert_array_equal(selected, [0, 2])


def test_v3_task_indices_reject_empty_selection(parquet_v3_module):
    dataset_cls = parquet_v3_module.ParquetDatasetV3
    dataset = dataset_cls.__new__(dataset_cls)
    dataset.dataset = [{'task_index': 0}]

    with pytest.raises(ValueError, match='No samples match'):
        dataset._filter_task_indices(np.array([0]), [7])


@pytest.mark.parametrize('dataset_version', ['v2.1', 'v3'])
def test_max_windows_selects_exact_deterministic_prefix(
        parquet_v3_module, dataset_version):
    dataset_cls = (parquet_v3_module.ParquetDataset
                   if dataset_version == 'v2.1' else
                   parquet_v3_module.ParquetDatasetV3)

    selected = dataset_cls._limit_sample_windows(
        np.array([9, 2, 7, 4], dtype=np.int64), 3)

    np.testing.assert_array_equal(selected, [9, 2, 7])


@pytest.mark.parametrize('dataset_version', ['v2.1', 'v3'])
def test_max_windows_fails_when_exact_subset_is_unavailable(
        parquet_v3_module, dataset_version):
    dataset_cls = (parquet_v3_module.ParquetDataset
                   if dataset_version == 'v2.1' else
                   parquet_v3_module.ParquetDatasetV3)

    with pytest.raises(ValueError, match='only 2 valid starts'):
        dataset_cls._limit_sample_windows(
            np.array([0, 1], dtype=np.int64), 3)

    for invalid in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match='positive integer'):
            dataset_cls._limit_sample_windows(
                np.array([0, 1], dtype=np.int64), invalid)


def test_v2_full_window_accounts_for_stride_and_static(parquet_v3_module):
    dataset_cls = parquet_v3_module.ParquetDataset
    dataset = dataset_cls.__new__(dataset_cls)
    dataset.dataset = [{
        'episode_index': 0,
        'task_index': 0,
    } for _ in range(5)] + [{
        'episode_index': 1,
        'task_index': 0,
    } for _ in range(5)]
    dataset.dataset[2]['task_index'] = 1
    dataset.tasks = [[{'task': 'move'}, {'task': 'static'}]]
    dataset.dataset_cumulative_sizes = np.array([0, len(dataset.dataset)])
    dataset.require_full_window = True
    dataset.window_start_idx = 1
    dataset.action_window_size = 4
    dataset.frame_window_size = 3
    dataset.frame_sample_stride = 2

    # Frames use rows 0,2,4, but the fourth non-static action would be row 5
    # in the next episode after row 2 is skipped.
    assert dataset._invalid_start_index(0, 0, dataset.dataset[0])


@pytest.mark.parametrize('dataset_version', ['v2.1', 'v3'])
def test_require_full_window_samples_only_valid_starts(parquet_v3_module,
                                                       dataset_version):
    dataset_cls = (parquet_v3_module.ParquetDataset
                   if dataset_version == 'v2.1' else
                   parquet_v3_module.ParquetDatasetV3)
    dataset = dataset_cls.__new__(dataset_cls)
    dataset.dataset = [
        {
            'episode_index': episode,
            'task_index': 0,
        } for episode in (0, 0, 0, 0, 1, 1, 1, 1)
    ]
    dataset.dataset_cumulative_sizes = np.array([0, len(dataset.dataset)])
    dataset.tasks = ([[{'task': 'move'}]] if dataset_version == 'v2.1' else
                     [{0: 'move'}])
    dataset.require_full_window = True
    dataset.window_start_idx = 1
    dataset.action_window_size = 2
    dataset.frame_window_size = 3
    dataset.frame_sample_stride = 1

    selected = dataset._filter_valid_start_indices(
        np.arange(len(dataset.dataset), dtype=np.int64))

    np.testing.assert_array_equal(selected, [0, 1, 4, 5])


@pytest.mark.parametrize('dataset_version', ['v2.1', 'v3'])
def test_require_full_window_fails_fast_without_valid_start(
        parquet_v3_module, dataset_version):
    dataset_cls = (parquet_v3_module.ParquetDataset
                   if dataset_version == 'v2.1' else
                   parquet_v3_module.ParquetDatasetV3)
    dataset = dataset_cls.__new__(dataset_cls)
    dataset.dataset = [{
        'episode_index': 0,
        'task_index': 0,
    } for _ in range(2)]
    dataset.dataset_cumulative_sizes = np.array([0, len(dataset.dataset)])
    dataset.tasks = ([[{'task': 'move'}]] if dataset_version == 'v2.1' else
                     [{0: 'move'}])
    dataset.require_full_window = True
    dataset.window_start_idx = 1
    dataset.action_window_size = 2
    dataset.frame_window_size = 3
    dataset.frame_sample_stride = 1

    with pytest.raises(ValueError, match='No valid sample starts remain'):
        dataset._filter_valid_start_indices(
            np.arange(len(dataset.dataset), dtype=np.int64))


def test_v3_episode_fraction_uses_complete_leading_episodes(
        parquet_v3_module):
    dataset_cls = parquet_v3_module.ParquetDatasetV3
    dataset = dataset_cls.__new__(dataset_cls)
    dataset.dataset = [
        {'episode_index': episode}
        for episode in (0, 0, 1, 1, 2, 2, 3, 3)
    ]
    dataset.full_length = len(dataset.dataset)
    dataset.dataset_cumulative_sizes = np.array([0, len(dataset.dataset)])

    selected = dataset._build_sample_indices(0.5)

    np.testing.assert_array_equal(selected, [0, 1, 2, 3])


def _multi_root_split_dataset(parquet_v3_module, dataset_version):
    dataset_cls = (
        parquet_v3_module.ParquetDataset
        if dataset_version == 'v2.1' else parquet_v3_module.ParquetDatasetV3)
    dataset = dataset_cls.__new__(dataset_cls)
    first_root_order = [4, 1, 9, 3, 8, 2, 7, 0, 6, 5]
    second_root_order = [104, 101, 109, 103, 108, 102, 107, 100, 106, 105]
    dataset.dataset = [{
        'episode_index': episode
    } for episode in first_root_order
                       for _ in range(2)] + [{
                           'episode_index': episode
                       } for episode in second_root_order for _ in range(2)]
    dataset.full_length = len(dataset.dataset)
    dataset.dataset_cumulative_sizes = np.array([0, 20, 40])
    return dataset


@pytest.mark.parametrize('dataset_version', ['v2.1', 'v3'])
def test_episode_fraction_range_supports_disjoint_80_10_10_splits(
        parquet_v3_module, dataset_version):
    dataset = _multi_root_split_dataset(parquet_v3_module, dataset_version)

    train = dataset._build_sample_indices(episode_fraction_range=(0.0, 0.8))
    validation = dataset._build_sample_indices(
        episode_fraction_range=(0.8, 0.9))
    test = dataset._build_sample_indices(episode_fraction_range=(0.9, 1.0))

    np.testing.assert_array_equal(train, np.r_[np.arange(0, 16),
                                               np.arange(20, 36)])
    np.testing.assert_array_equal(validation, [16, 17, 36, 37])
    np.testing.assert_array_equal(test, [18, 19, 38, 39])
    assert not set(train) & set(validation)
    assert not set(train) & set(test)
    assert not set(validation) & set(test)
    assert set(train) | set(validation) | set(test) == set(range(40))


@pytest.mark.parametrize('dataset_version', ['v2.1', 'v3'])
def test_episode_fraction_range_is_mutually_exclusive_with_legacy_fraction(
        parquet_v3_module, dataset_version):
    dataset = _multi_root_split_dataset(parquet_v3_module, dataset_version)

    with pytest.raises(ValueError, match='mutually exclusive'):
        dataset._build_sample_indices(0.8, episode_fraction_range=(0.0, 0.8))


@pytest.mark.parametrize('dataset_version', ['v2.1', 'v3'])
def test_episode_fraction_range_fails_fast_for_empty_per_root_split(
        parquet_v3_module, dataset_version):
    dataset = _multi_root_split_dataset(parquet_v3_module, dataset_version)
    dataset.dataset = dataset.dataset[:8]
    dataset.full_length = len(dataset.dataset)
    dataset.dataset_cumulative_sizes = np.array([0, 8])

    with pytest.raises(ValueError, match='selects no episodes'):
        dataset._build_sample_indices(episode_fraction_range=(0.8, 0.9))


@pytest.mark.parametrize('dataset_version', ['v2.1', 'v3'])
def test_legacy_episode_fraction_keeps_one_episode_per_root(
        parquet_v3_module, dataset_version):
    dataset = _multi_root_split_dataset(parquet_v3_module, dataset_version)

    selected = dataset._build_sample_indices(0.01)

    np.testing.assert_array_equal(selected, [0, 1, 20, 21])


def test_v3_getitem_resolves_split_index_to_selected_physical_row(
        parquet_v3_module):
    dataset_cls = parquet_v3_module.ParquetDatasetV3
    dataset = dataset_cls.__new__(dataset_cls)
    dataset.dataset = [{
        'episode_index': row // 2,
        'task_index': 0,
        'timestamp': float(row),
        'action': [float(row)],
    } for row in range(6)]
    dataset.sample_indices = np.array([2, 3], dtype=np.int64)
    dataset.dataset_cumulative_sizes = np.array([0, 6])
    dataset.tasks = [{0: 'move'}]
    dataset.action_window_size = 1
    dataset.action_key = 'action'
    dataset.use_delta = False
    dataset.statistic_name = 'private'
    dataset.window_start_idx = 0
    dataset.frame_window_size = 1
    dataset.frame_sample_stride = 1
    dataset.require_full_window = False
    dataset.expose_index = True
    dataset.info = [{}]
    dataset.data_root_path = ['/dataset']
    episode_meta = {
        'episode_index': 1,
        'videos/observation.images.front/chunk_index': 3,
        'videos/observation.images.front/file_index': 7,
    }
    dataset.episode_metadata_by_dataset = [{1: episode_meta}]
    dataset.transforms = []

    sample = dataset.__getitem__(0, {'private': {}})

    assert sample['index'].item() == 2
    assert sample['episode_index'] == 1
    assert sample['episode_meta'] == episode_meta
    np.testing.assert_array_equal(sample['actions'], [[2.0]])


def test_v3_droid_alignment_uses_future_commands_after_current_state(
        parquet_v3_module):
    dataset_cls = parquet_v3_module.ParquetDatasetV3
    dataset = dataset_cls.__new__(dataset_cls)
    dataset.dataset = [{
        'episode_index': 0,
        'task_index': 0,
        'timestamp': float(row),
        'action': [float(row)] * 8,
    } for row in range(5)]
    dataset.sample_indices = np.arange(5, dtype=np.int64)
    dataset.dataset_cumulative_sizes = np.array([0, 5])
    dataset.tasks = [{0: 'move'}]
    dataset.action_window_size = 2
    dataset.action_key = 'action'
    dataset.use_delta = False
    dataset.statistic_name = 'private'
    dataset.window_start_idx = 1
    dataset.frame_window_size = 3
    dataset.frame_sample_stride = 1
    dataset.require_full_window = True
    dataset.expose_index = False
    dataset.info = [{}]
    dataset.data_root_path = ['/dataset']
    dataset.episode_metadata_by_dataset = [{0: {'episode_index': 0}}]
    dataset.transforms = []

    sample = dataset.__getitem__(0, {'private': {}})

    # A later transform may prepend state_0; labels begin at the configured
    # action offset and contain action_1 and action_2.
    np.testing.assert_array_equal(sample['actions'],
                                  [[1.0] * 8, [2.0] * 8])

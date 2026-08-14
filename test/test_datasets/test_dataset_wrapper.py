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

import unittest

import numpy as np

from fluxvla.datasets.dataset_wrapper import DistributedRepeatingDataset
from fluxvla.datasets.parquet_dataset import ParquetDataset


class TestDistributedRepeatingDatasetStatistics(unittest.TestCase):

    def _make_wrapper(self, dim=None):
        wrapper = DistributedRepeatingDataset.__new__(
            DistributedRepeatingDataset)
        wrapper.statistic_name = 'private'
        wrapper.dim = dim
        return wrapper

    def test_combines_weighted_mean_and_std_with_scalar_counts(self):
        wrapper = self._make_wrapper()
        stats = [
            {
                'stats': {
                    'action': {
                        'min': [0.0, 8.0],
                        'max': [2.0, 12.0],
                        'mean': [1.0, 10.0],
                        'std': [0.5, 1.0],
                        'count': 2,
                        'q01': [0.0, 8.0],
                        'q99': [2.0, 12.0],
                    }
                }
            },
            {
                'stats': {
                    'action': {
                        'min': [3.0, 11.0],
                        'max': [7.0, 17.0],
                        'mean': [5.0, 14.0],
                        'std': [1.5, 2.0],
                        'count': 6,
                        'q01': [3.0, 11.0],
                        'q99': [7.0, 17.0],
                    }
                }
            },
        ]

        combined = wrapper.get_dataset_statistics(
            stats, ['action'])['private']['action']

        np.testing.assert_allclose(combined['mean'], [4.0, 13.0])
        np.testing.assert_allclose(combined['std'], [np.sqrt(4.75), 2.5])
        np.testing.assert_allclose(combined['q01'], [2.25, 10.25])
        np.testing.assert_allclose(combined['q99'], [5.75, 15.75])

    def test_unweighted_std_includes_between_dataset_variance(self):
        wrapper = self._make_wrapper()
        stats = [
            {
                'stats': {
                    'action': {
                        'min': [0.0],
                        'max': [0.0],
                        'mean': [0.0],
                        'std': [0.0],
                    }
                }
            },
            {
                'stats': {
                    'action': {
                        'min': [10.0],
                        'max': [10.0],
                        'mean': [10.0],
                        'std': [0.0],
                    }
                }
            },
        ]

        combined = wrapper.get_dataset_statistics(
            stats, ['action'])['private']['action']

        np.testing.assert_allclose(combined['mean'], [5.0])
        np.testing.assert_allclose(combined['std'], [5.0])

    def test_combines_weighted_mean_and_std_with_vector_counts(self):
        wrapper = self._make_wrapper()
        stats = [
            {
                'stats': {
                    'action': {
                        'min': [0.0, 10.0],
                        'max': [0.0, 10.0],
                        'mean': [0.0, 10.0],
                        'std': [0.0, 0.0],
                        'count': [1, 9],
                    }
                }
            },
            {
                'stats': {
                    'action': {
                        'min': [10.0, 20.0],
                        'max': [10.0, 20.0],
                        'mean': [10.0, 20.0],
                        'std': [0.0, 0.0],
                        'count': [9, 1],
                    }
                }
            },
        ]

        combined = wrapper.get_dataset_statistics(
            stats, ['action'])['private']['action']

        np.testing.assert_allclose(combined['mean'], [9.0, 11.0])
        np.testing.assert_allclose(combined['std'], [3.0, 3.0])

    def test_incomplete_counts_fall_back_to_unweighted_merge(self):
        wrapper = self._make_wrapper()
        stats = [
            {
                'stats': {
                    'action': {
                        'min': [0.0],
                        'max': [0.0],
                        'mean': [0.0],
                        'std': [0.0],
                        'count': 100,
                    }
                }
            },
            {
                'stats': {
                    'action': {
                        'min': [10.0],
                        'max': [10.0],
                        'mean': [10.0],
                        'std': [0.0],
                    }
                }
            },
        ]

        combined = wrapper.get_dataset_statistics(
            stats, ['action'])['private']['action']

        np.testing.assert_allclose(combined['mean'], [5.0])
        np.testing.assert_allclose(combined['std'], [5.0])

    def test_padding_applies_to_quantiles_and_vector_counts(self):
        wrapper = self._make_wrapper(dim=4)
        stats = [
            {
                'stats': {
                    'action': {
                        'min': [0.0, 1.0, 2.0],
                        'max': [0.0, 1.0, 2.0],
                        'mean': [0.0, 1.0, 2.0],
                        'std': [0.0, 0.0, 0.0],
                        'count': [1, 1, 1],
                        'q25': [0.0, 1.0, 2.0],
                    }
                }
            },
            {
                'stats': {
                    'action': {
                        'min': [4.0, 5.0, 6.0],
                        'max': [4.0, 5.0, 6.0],
                        'mean': [4.0, 5.0, 6.0],
                        'std': [0.0, 0.0, 0.0],
                        'count': [3, 3, 3],
                        'q25': [4.0, 5.0, 6.0],
                    }
                }
            },
        ]

        combined = wrapper.get_dataset_statistics(
            stats, ['action'])['private']['action']

        np.testing.assert_allclose(combined['mean'], [3.0, 4.0, 5.0, 3.0])
        np.testing.assert_allclose(combined['q25'], [3.0, 4.0, 5.0, 3.0])


class _EpisodeBlockDataset:

    stats = []

    def __init__(self):
        self.values = list(range(8))

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index, _statistics):
        return self.values[index]

    def get_shuffle_blocks(self):
        return [(0, 3), (3, 2), (5, 3)]


class TestDistributedRepeatingDatasetEpisodeShuffle(unittest.TestCase):

    def test_preserves_order_inside_episode_blocks(self):
        wrapper = DistributedRepeatingDataset.__new__(
            DistributedRepeatingDataset)
        wrapper.shuffle = True
        wrapper.reshuffle_each_epoch = True
        wrapper.shuffle_by_episode = True
        wrapper.seed = 42
        wrapper.rank = 0
        wrapper.world_size = 1
        wrapper._epoch = 0
        wrapper.is_grouped = False
        wrapper.is_list = False
        wrapper.dataset = _EpisodeBlockDataset()
        wrapper.dataset_statistics = {}
        wrapper.total_len = len(wrapper.dataset)

        iterator = iter(wrapper)
        first_epoch = [next(iterator) for _ in range(8)]
        second_epoch = [next(iterator) for _ in range(8)]

        valid_blocks = ([0, 1, 2], [3, 4], [5, 6, 7])
        for epoch in (first_epoch, second_epoch):
            cursor = 0
            seen = []
            while cursor < len(epoch):
                block = next(block for block in valid_blocks
                             if block[0] == epoch[cursor])
                self.assertEqual(epoch[cursor:cursor + len(block)], block)
                seen.append(block[0])
                cursor += len(block)
            self.assertCountEqual(seen, [0, 3, 5])
        self.assertNotEqual(first_epoch, second_epoch)


class TestParquetDatasetFullWindowIndex(unittest.TestCase):

    def test_filters_episode_tails_before_sampling(self):
        dataset = ParquetDataset.__new__(ParquetDataset)
        dataset.full_length = 38
        dataset._episode_indices = np.array([0] * 20 + [1] * 18)
        dataset.dataset_cumulative_sizes = np.array([0, 38])
        dataset.frame_window_size = 17
        dataset.window_start_idx = 0
        dataset.action_window_size = 16
        dataset.require_full_window = True
        dataset.repeat_to_full_length = False

        dataset.sample_indices = dataset._build_sample_indices(1.0)

        np.testing.assert_array_equal(dataset.sample_indices,
                                      [0, 1, 2, 3, 20, 21])
        self.assertEqual(dataset.get_shuffle_blocks(), [(0, 4), (4, 2)])


if __name__ == '__main__':
    unittest.main()

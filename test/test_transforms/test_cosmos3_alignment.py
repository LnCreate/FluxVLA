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

import json
import unittest

import numpy as np

from fluxvla.transforms.transform_actions import \
    LiberoFramewiseActionToRot6D
from fluxvla.transforms.transform_cosmos3 import ProcessCosmos3Prompt


class TestCosmos3LiberoAlignment(unittest.TestCase):

    def test_idle_frames_are_computed_before_normalization(self):
        actions = np.zeros((6, 7), dtype=np.float32)
        actions[3, 0] = 0.1
        transform = LiberoFramewiseActionToRot6D(compute_idle_frames=True)

        result = transform({'actions': actions})

        self.assertEqual(int(result['idle_frames']), 3)
        self.assertEqual(int(result['idle_frames_total']), 6)

    def test_json_prompt_uses_idle_frames_and_view_description(self):
        transform = ProcessCosmos3Prompt.__new__(ProcessCosmos3Prompt)
        transform.action_metadata = {
            'conditioning_fps':
            20.0,
            'frame_window_size':
            17,
            'video_height':
            192,
            'video_width':
            320,
            'viewpoint':
            'concat_view',
            'viewpoint_description':
            ('The left half shows the third-person view; the right half '
             'shows the wrist-mounted camera.'),
        }
        data = {
            'idle_frames': np.array(5),
            'idle_frames_total': np.array(16),
        }

        prompt = json.loads(
            transform._format_action_json_prompt('pick up the mug', data))

        self.assertEqual(prompt['actions'][0]['idle_frame'], '5 out of 16.')
        self.assertIn('left half', prompt['cinematography']['framing'])
        self.assertIn('right half', prompt['cinematography']['framing'])


if __name__ == '__main__':
    unittest.main()

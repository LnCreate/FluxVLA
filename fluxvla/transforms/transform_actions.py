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

from typing import Dict, List

import numpy as np

from fluxvla.engines import TRANSFORMS
from fluxvla.transforms.normalize import DenormalizeLiberoAction


def _axisangle_to_matrix(axisangle: np.ndarray) -> np.ndarray:
    rotvec = np.asarray(axisangle, dtype=np.float32)
    if rotvec.shape[-1] != 3:
        raise ValueError(f'axisangle must have last dim 3, got {rotvec.shape}')

    original_shape = rotvec.shape[:-1]
    flat = rotvec.reshape(-1, 3)
    theta = np.linalg.norm(flat, axis=-1, keepdims=True)
    axis = np.divide(
        flat,
        theta,
        out=np.zeros_like(flat),
        where=theta > 1e-8,
    )
    x, y, z = axis[:, 0], axis[:, 1], axis[:, 2]

    skew = np.zeros((flat.shape[0], 3, 3), dtype=np.float32)
    skew[:, 0, 1] = -z
    skew[:, 0, 2] = y
    skew[:, 1, 0] = z
    skew[:, 1, 2] = -x
    skew[:, 2, 0] = -y
    skew[:, 2, 1] = x

    eye = np.eye(3, dtype=np.float32)[None, :, :]
    sin_theta = np.sin(theta)[:, :, None]
    cos_theta = np.cos(theta)[:, :, None]
    matrix = eye + sin_theta * skew + (1.0 - cos_theta) * (skew @ skew)
    matrix[(theta[:, 0] <= 1e-8)] = eye
    return matrix.reshape(*original_shape, 3, 3)


def _normalize_rotation_matrices(matrices: np.ndarray) -> np.ndarray:
    original_shape = matrices.shape[:-2]
    flat = matrices.reshape(-1, 3, 3).astype(np.float32, copy=False)
    u, _, vh = np.linalg.svd(flat)
    projected = u @ vh
    reflection = np.linalg.det(projected) < 0
    if np.any(reflection):
        u[reflection, :, -1] *= -1
        projected[reflection] = u[reflection] @ vh[reflection]
    return projected.astype(
        np.float32, copy=False).reshape(*original_shape, 3, 3)


def _matrix_to_rot6d(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.shape[-2:] != (3, 3):
        raise ValueError(f'rotation matrix must end with [3,3], got '
                         f'{matrix.shape}')
    return matrix[..., :, :2].swapaxes(-1, -2).reshape(*matrix.shape[:-2], 6)


def _rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    rot6d = np.asarray(rot6d, dtype=np.float32)
    if rot6d.shape[-1] != 6:
        raise ValueError(f'rot6d must have last dim 6, got {rot6d.shape}')
    original_shape = rot6d.shape[:-1]
    flat = rot6d.reshape(-1, 6)
    col0 = flat[:, :3]
    col1 = flat[:, 3:]
    col2 = np.cross(col0, col1, axis=-1)
    matrix = np.stack((col0, col1, col2), axis=-1)
    return _normalize_rotation_matrices(matrix).reshape(*original_shape, 3, 3)


def _quaternion_xyzw_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float32)
    quaternion = quaternion / np.linalg.norm(
        quaternion, axis=-1, keepdims=True).clip(min=1e-8)
    x, y, z, w = np.moveaxis(quaternion, -1, 0)
    return np.stack((
        1 - 2 * (y * y + z * z),
        2 * (x * y - z * w),
        2 * (x * z + y * w),
        2 * (x * y + z * w),
        1 - 2 * (x * x + z * z),
        2 * (y * z - x * w),
        2 * (x * z - y * w),
        2 * (y * z + x * w),
        1 - 2 * (x * x + y * y),
    ),
                    axis=-1).reshape(*quaternion.shape[:-1], 3, 3)


def _matrix_to_axisangle(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.shape[-2:] != (3, 3):
        raise ValueError(f'rotation matrix must end with [3,3], got '
                         f'{matrix.shape}')

    original_shape = matrix.shape[:-2]
    flat = matrix.reshape(-1, 3, 3)
    qw = np.sqrt(
        np.maximum(0.0,
                   1.0 + flat[:, 0, 0] + flat[:, 1, 1] + flat[:, 2, 2])) * 0.5
    qx = np.sqrt(
        np.maximum(0.0,
                   1.0 + flat[:, 0, 0] - flat[:, 1, 1] - flat[:, 2, 2])) * 0.5
    qy = np.sqrt(
        np.maximum(0.0,
                   1.0 - flat[:, 0, 0] + flat[:, 1, 1] - flat[:, 2, 2])) * 0.5
    qz = np.sqrt(
        np.maximum(0.0,
                   1.0 - flat[:, 0, 0] - flat[:, 1, 1] + flat[:, 2, 2])) * 0.5
    qx = np.copysign(qx, flat[:, 2, 1] - flat[:, 1, 2])
    qy = np.copysign(qy, flat[:, 0, 2] - flat[:, 2, 0])
    qz = np.copysign(qz, flat[:, 1, 0] - flat[:, 0, 1])

    quat = np.stack((qx, qy, qz, qw), axis=-1).astype(np.float32)
    quat_norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    quat = np.divide(
        quat,
        quat_norm,
        out=np.zeros_like(quat),
        where=quat_norm > 1e-8,
    )
    quat = np.where(quat[:, 3:4] < 0.0, -quat, quat)

    vec = quat[:, :3]
    w = np.clip(quat[:, 3], -1.0, 1.0)
    sin_half = np.linalg.norm(vec, axis=-1)
    angle = 2.0 * np.arctan2(sin_half, w)
    axis = np.divide(
        vec,
        sin_half[:, None],
        out=np.zeros_like(vec),
        where=sin_half[:, None] > 1e-8,
    )
    rotvec = axis * angle[:, None]
    small = sin_half <= 1e-8
    rotvec[small] = 2.0 * vec[small]
    return rotvec.astype(np.float32, copy=False).reshape(*original_shape, 3)


def _libero_axisangle_action_to_rot6d(action: np.ndarray) -> np.ndarray:
    action = np.asarray(action, dtype=np.float32)
    if action.shape[-1] != 7:
        raise ValueError('LIBERO frame-wise action must have shape [..., 7], '
                         f'got {action.shape}.')
    rotation = _matrix_to_rot6d(_axisangle_to_matrix(action[..., 3:6]))
    return np.concatenate((action[..., :3], rotation, action[..., 6:7]),
                          axis=-1)


def _libero_rot6d_action_to_axisangle(action: np.ndarray) -> np.ndarray:
    action = np.asarray(action, dtype=np.float32)
    if action.shape[-1] != 10:
        raise ValueError('LIBERO rot6d action must have shape [..., 10], '
                         f'got {action.shape}.')
    rotation = _matrix_to_axisangle(_rot6d_to_matrix(action[..., 3:9]))
    return np.concatenate((action[..., :3], rotation, action[..., 9:10]),
                          axis=-1)


@TRANSFORMS.register_module()
class ProcessLiberoActions:

    def __init__(self, mask: List[bool] = None) -> None:
        """ProcessLiberoActions is a transform
        that modifies the actions in the data
        by subtracting the state values from
        the actions based on a mask.

        Args:
            mask (List[bool], optional): A list
                indicating which dimensions
                of the state should be subtracted from
                the actions.
                If None, no subtraction is performed.
        """
        self.mask = np.asarray(mask, dtype=bool)

    def __call__(self, data: Dict) -> Dict:
        if 'actions' not in data or self.mask is None:
            return data

        states, actions = data['states'], data['actions']
        mask = np.asarray(self.mask)
        dims = mask.shape[-1]
        actions[..., :dims] -= np.expand_dims(
            np.where(mask, states[..., :dims], 0), axis=-2)
        data['actions'] = actions

        return data


@TRANSFORMS.register_module()
class LiberoFramewiseActionToRot6D:
    """Convert LIBERO stored 7D frame-wise deltas to official 10D rot6d."""

    def __init__(self, action_key: str = 'actions') -> None:
        self.action_key = action_key

    def __call__(self, data: Dict) -> Dict:
        if self.action_key not in data:
            return data
        data[self.action_key] = _libero_axisangle_action_to_rot6d(
            data[self.action_key])
        return data


@TRANSFORMS.register_module()
class DualAbsoluteEEPoseToRelativeRot6D:
    """Convert future dual-arm eeposes to frame-wise 20D actions."""

    def __init__(self,
                 state_key: str = 'states',
                 action_key: str = 'actions',
                 frame_stride: int = 2,
                 gripper_open_threshold: float = 0.06) -> None:
        self.state_key = state_key
        self.action_key = action_key
        self.frame_stride = frame_stride
        self.gripper_open_threshold = gripper_open_threshold

    def __call__(self, data: Dict) -> Dict:
        current = np.asarray(data[self.state_key], dtype=np.float32)
        future = np.asarray(data[self.action_key], dtype=np.float32)
        if current.shape[-1] != 16 or future.shape[-1] != 16:
            raise ValueError('Dual-arm eepose with grippers must have '
                             '16 dimensions.')

        future = future[self.frame_stride - 1::self.frame_stride]
        previous = np.concatenate((current[None], future[:-1]), axis=0)
        arm_actions = []
        for offset in (0, 8):
            delta_position = (
                future[:, offset:offset + 3] - previous[:, offset:offset + 3])
            previous_rotation = _quaternion_xyzw_to_matrix(
                previous[:, offset + 3:offset + 7])
            future_rotation = _quaternion_xyzw_to_matrix(future[:, offset +
                                                                3:offset + 7])
            relative_rotation = (
                future_rotation @ previous_rotation.swapaxes(-1, -2))
            gripper = (future[:, offset + 7:offset + 8]
                       >= self.gripper_open_threshold).astype(np.float32)
            arm_actions.append(
                np.concatenate((delta_position,
                                _matrix_to_rot6d(relative_rotation), gripper),
                               axis=-1))

        data[self.action_key] = np.concatenate(arm_actions, axis=-1)
        if 'action_masks' in data:
            data['action_masks'] = data['action_masks'][self.frame_stride -
                                                        1::self.frame_stride]
        return data


@TRANSFORMS.register_module()
class DenormalizeLiberoFramewiseRot6DAction:
    """Denormalize official 10D rot6d LIBERO actions to 7D env commands."""

    def __init__(self, *args, norm_type: str = 'quantile', **kwargs) -> None:
        if norm_type == 'quantile_rot':
            norm_type = 'quantile'
        self.denormalize = DenormalizeLiberoAction(
            *args, norm_type=norm_type, **kwargs)

    def __call__(self, data: Dict) -> np.ndarray:
        return _libero_rot6d_action_to_axisangle(self.denormalize(data))

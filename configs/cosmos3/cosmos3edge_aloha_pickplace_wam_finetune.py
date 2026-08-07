# Copyright 2026 Limx Dynamics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Cosmos3-Edge WAM post-training on dual-ALOHA pick-and-place."""

_base_ = ['./cosmos3edge_libero_full_finetune.py']

_dataset_root = (
    '/mnt/data/cpfs/users/leo/data/'
    'RealRobot_AgileX_aloha_pickplace_rot6d/train')
_statistic_name = 'aloha_dual_rot6d'
_embodiment_id = 21
_raw_action_dim = 20
_frame_window_size = 17
_frame_stride = 2
_conditioning_fps = 15.0

_tokenizer = dict(
    model_max_length=4096,
    model_path='./checkpoints/Cosmos3-Edge',
    padding_side='right',
    trust_remote_code=True,
    type='PretrainedTokenizer')

_action_stats = dict(
    q01=[
        -0.01271332, -0.01283099, -0.010491, 0.99832931, -0.04364049,
        -0.02785018, -0.04549627, 0.99814586, -0.03744125, 0.0,
        -0.01285796, -0.010133, -0.01074292, 0.99856134, -0.03282509,
        -0.02877802, -0.04402629, 0.9984049, -0.02858664, 0.0
    ],
    q99=[
        0.01273898, 0.01034296, 0.01407696, 1.00000036, 0.04548698,
        0.03168717, 0.0436939, 1.00000024, 0.02950429, 1.0,
        0.01343395, 0.01283732, 0.01379932, 1.00000036, 0.04397862,
        0.03540704, 0.03271011, 1.00000024, 0.03470469, 1.0
    ])

_transforms = [
    dict(
        embodiment_id=_embodiment_id,
        name_mappings={'observation.eepose_gripper': ['states']},
        parquet_keys=[
            'observation.eepose_gripper',
            'timestamp',
            'actions',
            'info',
            'stats',
            'action_masks',
        ],
        type='ProcessParquetInputs',
        video_keys=[
            'observation.images.cam_high',
            'observation.images.cam_left_wrist',
            'observation.images.cam_right_wrist',
        ]),
    dict(
        frame_stride=_frame_stride,
        gripper_open_threshold=0.05,
        type='DualAbsoluteEEPoseToRelativeRot6D'),
    dict(
        action_metadata=dict(
            append_viewpoint=False,
            conditioning_fps=_conditioning_fps,
            frame_window_size=_frame_window_size,
            video_height=384,
            video_width=256),
        cfg_dropout_rate=0.1,
        max_len=512,
        tokenizer=_tokenizer,
        type='ProcessCosmos3Prompt'),
    dict(height=256, type='ResizeImages', width=256),
    dict(type='SimpleNormalizeImages'),
    dict(
        action_dim=64,
        action_key='action',
        action_norm_type='quantile',
        state_dim=64,
        state_key='proprio',
        state_norm_type='none',
        type='NormalizeStatesAndActions'),
    dict(
        conditioning_fps=_conditioning_fps,
        frame_window_size=_frame_window_size,
        mode='wam',
        prepend_state_to_action=False,
        raw_action_dim=_raw_action_dim,
        type='BuildCosmos3Sequence'),
    dict(
        bottom_height_ratio=0.5,
        bottom_views=(1, 2),
        frame_window_size=_frame_window_size,
        num_views=3,
        tile_direction='top_bottom_pair',
        top_view=0,
        type='PrepareVideo'),
]

model = dict(ori_action_dim=_raw_action_dim)
inference_model = None
eval = None

train_dataloader = dict(
    dataset=dict(
        _delete_=True,
        datasets=dict(
            action_key='observation.eepose_gripper',
            action_window_size=32,
            data_root_path=_dataset_root,
            frame_sample_stride=_frame_stride,
            frame_window_size=_frame_window_size,
            require_full_window=True,
            statistic_name=_statistic_name,
            transforms=_transforms,
            type='ParquetDataset',
            use_delta=False,
            window_start_idx=1),
        name_mappings={'observation.eepose_gripper': ['proprio']},
        statistic_keys=['observation.eepose_gripper', 'timestamp'],
        statistic_name=_statistic_name,
        statistics_overrides={_statistic_name: {
            'action': _action_stats,
        }},
        type='DistributedRepeatingDataset'),
    per_device_batch_size=16,
    per_device_num_workers=4)

runner = dict(
    grad_accumulation_steps=16,
    max_keep_ckpts=4,
    max_steps=2000,
    metric=dict(grad_accumulation_steps=16),
    save_iter_interval=500)

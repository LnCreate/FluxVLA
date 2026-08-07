# Copyright 2026 Limx Dynamics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Cosmos3-Edge WAM post-training on four dual-Franka tasks."""

_base_ = ['./cosmos3edge_libero_full_finetune.py']

_dataset_root = ('/mnt/data/cpfs/users/leo/data/'
                 'RealRobot_franka_dual_lerobotv2.1_clean_4tasks_300')
_statistic_name = 'franka_dual_rot6d'
_embodiment_id = 22
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
        -0.01322838, -0.01451273, -0.01207803, 0.99772364, -0.05481346,
        -0.01714929, -0.05188668, 0.9961458, -0.0213582, 0.0, -0.01429188,
        -0.01861186, -0.01272357, 0.99962902, -0.01423965, -0.02244734,
        -0.01420859, 0.99675404, -0.02168916, 0.0
    ],
    q99=[
        0.01362636, 0.01386914, 0.01437228, 1.00000024, 0.0519619, 0.01807653,
        0.05498469, 1.00000036, 0.02378323, 1.0, 0.01399703, 0.01526304,
        0.01725417, 1.00000012, 0.01405056, 0.01831199, 0.01430536, 1.00000036,
        0.02625485, 1.0
    ])

_transforms = [
    dict(
        embodiment_id=_embodiment_id,
        name_mappings={'observation.eepose': ['states']},
        parquet_keys=[
            'observation.eepose',
            'timestamp',
            'actions',
            'info',
            'stats',
            'action_masks',
        ],
        type='ProcessParquetInputs',
        video_keys=[
            'observation.images.cam_front',
            'observation.images.cam_wrist_left',
            'observation.images.cam_wrist_right',
        ]),
    dict(
        frame_stride=_frame_stride,
        gripper_open_threshold=0.06,
        type='DualAbsoluteEEPoseToRelativeRot6D'),
    dict(
        action_metadata=dict(
            append_viewpoint=False,
            conditioning_fps=_conditioning_fps,
            frame_window_size=_frame_window_size,
            video_height=256,
            video_width=768),
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
        frame_window_size=_frame_window_size,
        num_views=3,
        tile_direction='horizontal',
        type='PrepareVideo'),
]


def _dataset(date):
    return dict(
        action_key='observation.eepose',
        action_window_size=32,
        data_root_path=f'{_dataset_root}/{date}_dual_franka_teleop',
        frame_sample_stride=_frame_stride,
        frame_window_size=_frame_window_size,
        require_full_window=True,
        statistic_name=_statistic_name,
        transforms=_transforms,
        type='ParquetDataset',
        use_delta=False,
        window_start_idx=1)


model = dict(ori_action_dim=_raw_action_dim)
inference_model = None
eval = None

train_dataloader = dict(
    dataset=dict(
        _delete_=True,
        datasets=[
            _dataset('20260527'),
            _dataset('20260528'),
            _dataset('20260602'),
            _dataset('20260616'),
            _dataset('20260617'),
            _dataset('20260618'),
            _dataset('20260623'),
            _dataset('20260624'),
            _dataset('20260626'),
            _dataset('20260702'),
            _dataset('20260720'),
            _dataset('20260803'),
        ],
        name_mappings={'observation.eepose': ['proprio']},
        statistic_keys=['observation.eepose', 'timestamp'],
        statistic_name=_statistic_name,
        statistics_overrides={_statistic_name: {
            'action': _action_stats,
        }},
        type='DistributedRepeatingDataset'),
    per_device_batch_size=16,
    per_device_num_workers=4)

runner = dict(
    grad_accumulation_steps=16,
    max_keep_ckpts=2,
    max_steps=2000,
    metric=dict(grad_accumulation_steps=16),
    save_iter_interval=500)

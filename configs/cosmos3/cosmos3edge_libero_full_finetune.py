"""JiKun-aligned full LIBERO post-training for Cosmos3-Edge.

"Full" follows the Nano recipe terminology: train on all four LIBERO suites
for the complete 12-epoch schedule.  The parameter selection remains the
Nano-style generation/action partial SFT; the Edge/Nemotron architecture,
tokenizer, checkpoint loader, and special tokens are the core-model changes.
"""

from runpy import run_path as _run_path

_common = _run_path('{{fileDirname}}/cosmos3edge_common.py')
_build = _common['build_libero_config']

_data_root = (
    '/mnt/data/cpfs/mnt/data/yanis/FastWAM/data/libero_mujoco3.3.2'
)
_data_roots = [
    f'{_data_root}/libero_spatial_no_noops_lerobot',
    f'{_data_root}/libero_object_no_noops_lerobot',
    f'{_data_root}/libero_goal_no_noops_lerobot',
    f'{_data_root}/libero_10_no_noops_lerobot',
]
_statistic_name = 'all_libero_no_noops'

# max_steps is replaced below by the Nano recipe's epoch-based schedule.
_config = _build(
    suite='libero_10',
    tuning='partial',
    max_steps=1,
    eval_trials=50,
    data_root_path=_data_roots,
    action_init='checkpoint',
)

for _model_key in ('model', 'inference_model'):
    _model = _config[_model_key]
    _model['rectified_flow_training_config'].update(
        action_loss_weight=10.0,
        vision_loss_weight=1.0,
        normalize_loss_by_active=False,
    )
    _model['rectified_flow_inference_config']['num_steps'] = 30

_train = _config['train_dataloader']
_train.update(per_device_batch_size=8, per_device_num_workers=4)
_wrapper = _train['dataset']
_wrapper['statistic_name'] = _statistic_name
_dataset = _wrapper['datasets']
_dataset['statistic_name'] = _statistic_name
# JiKun's loader counted all 277,713 rows in each epoch and resampled invalid
# episode-tail starts.  Keep that epoch/sample budget while the hardened
# loader deterministically repeats only prevalidated full-window starts.
_dataset['repeat_to_full_length'] = True

for _index, _transform in enumerate(_dataset['transforms']):
    if _transform['type'] == 'ProcessCosmos3Prompt':
        _transform['action_metadata'].update(
            video_height=256,
            video_width=128,
        )
    elif _transform['type'] == 'ResizeImagesWithPad':
        _dataset['transforms'][_index] = dict(
            type='ResizeImages', height=128, width=128)
    elif _transform['type'] == 'BuildCosmos3Sequence':
        _transform['mode'] = 'joint'

_runner = _config['runner']
_runner.update(
    max_steps=None,
    max_epochs=12,
    save_epoch_interval=1,
    save_iter_interval=10_000,
    max_keep_ckpts=2,
    grad_accumulation_steps=1,
)
_runner['optimizer']['lr'] = 8e-5
_runner['optimizer']['paramwise_learning_rate'] = {
    'action_in_proj.': 4e-4,
    'action_out_proj.': 4e-4,
    'action_modality_embed': 4e-4,
}
_runner['metric'].update(
    active_trackers=('jsonl', 'wandb'),
    grad_accumulation_steps=1,
)
_runner['lr_scheduler'] = dict(
    type='linear-warmup+cosine-decay',
    warmup_ratio=0.0,
)
_runner['collator'] = dict(
    type='Cosmos3Collator',
    tensor_keys=[
        'images',
        'actions',
        'embodiment_ids',
        'raw_action_dim',
        'conditioning_fps',
        'action_fps',
    ],
    sequence_keys=['text_token_ids'],
    list_keys=['sequence_plan'],
    meta_keys=['task_description', 'stats', 'info', 'timestamp'],
    pad_id=11,
)

_eval = _config['eval']
_eval.update(
    task_suite_name='libero_10',
    norm_stats_key=_statistic_name,
    resize_size=128,
    num_trials_per_task=50,
    task_ids=None,
    num_inference_steps=30,
)
for _transform in _eval['dataset']['transforms']:
    if _transform['type'] == 'TransformImage':
        _transform.update(
            image_resize_strategy='resize-naive',
            input_sizes=[[3, 128, 128], [3, 128, 128]],
        )
    elif _transform['type'] == 'ProcessCosmos3Prompt':
        _transform['action_metadata'].update(
            video_height=256,
            video_width=128,
        )

globals().update(_config)
del _build, _common, _config, _data_root, _data_roots
del _dataset, _eval, _index, _model, _model_key, _run_path
del _runner, _statistic_name, _train, _transform, _wrapper

"""Official-style partial LIBERO tuning warm-started from action-only step 2k."""

from runpy import run_path as _run_path

_build = _run_path('{{fileDirname}}/cosmos3edge_common.py')[
    'build_libero_config']

_warmstart = (
    './work_dirs/cosmos3edge_libero_task0_action_10k/checkpoints/'
    'step-002000-epoch-08-loss=8.6627.safetensors'
)
_data_root = (
    '/mnt/data/cpfs/mnt/data/yanis/FastWAM/data/'
    'libero_mujoco3.3.2/libero_spatial_no_noops_lerobot'
)
_config = _build(
    suite='libero_spatial',
    tuning='partial',
    max_steps=10_000,
    save_interval=2_000,
    task_indices=[0],
    eval_task_ids=[0],
    eval_trials=20,
    data_root_path=_data_root,
)

for _model_key in ('model', 'inference_model'):
    _config[_model_key]['pretrained_name_or_path'] = _warmstart
    _config[_model_key]['rectified_flow_training_config'][
        'vision_loss_weight'] = 10.0

_config['runner']['max_keep_ckpts'] = 2
_config['runner']['lr_scheduler'] = dict(
    type='linear-warmup+cosine-cycle',
    warmup_steps=500,
    cycle_steps=16_000,
)
_config['runner']['metric']['active_trackers'] = (
    'jsonl', 'wandb', 'tensorboard')

globals().update(_config)
del _build, _config, _data_root, _model_key, _run_path, _warmstart

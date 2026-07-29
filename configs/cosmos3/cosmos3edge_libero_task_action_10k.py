"""Cosmos3-Edge action-only post-training on LIBERO-Spatial task 0."""

from runpy import run_path as _run_path

_build = _run_path('{{fileDirname}}/cosmos3edge_common.py')[
    'build_libero_config']

_config = _build(
    suite='libero_spatial',
    tuning='action',
    max_steps=10_000,
    save_interval=2_000,
    task_indices=[0],
    eval_task_ids=[0],
    eval_trials=20,
)
_config['runner']['max_keep_ckpts'] = 2
_config['runner']['metric']['active_trackers'] = (
    'jsonl', 'wandb', 'tensorboard')
globals().update(_config)
del _build, _config, _run_path

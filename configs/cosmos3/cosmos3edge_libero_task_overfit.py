"""Cosmos3-Edge deterministic 32-window/100-step LIBERO overfit gate."""

from runpy import run_path as _run_path

_build = _run_path('{{fileDirname}}/cosmos3edge_common.py')[
    'build_libero_config']

globals().update(
    _build(
        suite='libero_spatial',
        tuning='action',
        max_steps=100,
        task_indices=[0],
        max_windows=32,
        eval_task_ids=[0],
        eval_trials=20,
        seed=7,
        cfg_dropout_rate=0.0,
    ))
del _build, _run_path

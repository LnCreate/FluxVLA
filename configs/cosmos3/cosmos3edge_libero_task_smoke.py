"""Cosmos3-Edge action-only, task-0 LIBERO-Spatial 500-step smoke run."""

from runpy import run_path as _run_path

_build = _run_path('{{fileDirname}}/cosmos3edge_common.py')[
    'build_libero_config']

globals().update(
    _build(
        suite='libero_spatial',
        tuning='action',
        max_steps=500,
        task_indices=[0],
        eval_task_ids=[0],
        eval_trials=20,
    ))
del _build, _run_path

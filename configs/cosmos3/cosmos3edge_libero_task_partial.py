"""Cosmos3-Edge generation-tower continuation for the LIBERO smoke task."""

from runpy import run_path as _run_path

_build = _run_path('{{fileDirname}}/cosmos3edge_common.py')[
    'build_libero_config']

globals().update(
    _build(
        suite='libero_spatial',
        tuning='partial',
        max_steps=2000,
        task_indices=[0],
        eval_task_ids=[0],
        eval_trials=20,
    ))
del _build, _run_path

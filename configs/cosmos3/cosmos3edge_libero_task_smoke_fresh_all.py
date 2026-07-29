"""LIBERO task-0 smoke for an Edge checkpoint without action tensors.

The explicit `(64,32)` layout is an experiment contract, not an inferred
property. Change it only after checking the released checkpoint config.
"""

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
        action_init='fresh_all',
        action_layout=(64, 32),
    ))
del _build, _run_path

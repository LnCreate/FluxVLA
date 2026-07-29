"""LIBERO task-0 smoke with explicit reinitialization of domain row 5.

Use only when the Edge checkpoint contains a shape-compatible action policy
head and domain 5 should be relearned for LIBERO.
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
        action_init='fresh_domain',
        fresh_action_domain_ids=[5],
    ))
del _build, _run_path

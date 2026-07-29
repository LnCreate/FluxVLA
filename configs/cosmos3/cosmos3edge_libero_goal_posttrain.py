from runpy import run_path as _run_path

_build = _run_path('{{fileDirname}}/cosmos3edge_common.py')[
    'build_libero_config']

globals().update(
    _build(
        suite='libero_goal',
        tuning='partial',
        max_steps=30_000,
        save_interval=3_000,
    ))
del _build, _run_path

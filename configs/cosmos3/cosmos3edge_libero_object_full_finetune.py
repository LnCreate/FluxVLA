"""JiKun-aligned Cosmos3-Edge single-suite training: LIBERO-Object."""

from runpy import run_path as _run_path

_build = _run_path('{{fileDirname}}/cosmos3edge_common.py')[
    'build_jikun_libero_single_config']

globals().update(_build('libero_object'))
del _build, _run_path

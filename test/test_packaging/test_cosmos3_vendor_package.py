import ast
from pathlib import Path

from setuptools import Distribution, find_packages
from setuptools.command.build_py import build_py


REPO_ROOT = Path(__file__).resolve().parents[2]
VENDOR_PACKAGE = 'fluxvla.models.third_party_models.cosmos3'
VENDOR_PATH = REPO_ROOT / Path(*VENDOR_PACKAGE.split('.'))


def _package_data_from_setup():
    tree = ast.parse((REPO_ROOT / 'setup.py').read_text(encoding='utf-8'))
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == 'setup'):
            continue
        for keyword in node.keywords:
            if keyword.arg == 'package_data':
                return ast.literal_eval(keyword.value)
    raise AssertionError('setup.py does not declare package_data')


def test_all_cosmos3_vendor_namespaces_are_discoverable():
    packages = set(find_packages(where=str(REPO_ROOT)))
    expected = {
        VENDOR_PACKAGE,
        f'{VENDOR_PACKAGE}.data',
        f'{VENDOR_PACKAGE}.data.vfm',
        f'{VENDOR_PACKAGE}.model',
        f'{VENDOR_PACKAGE}.model.vfm',
        f'{VENDOR_PACKAGE}.model.vfm.algorithm',
        f'{VENDOR_PACKAGE}.model.vfm.algorithm.loss',
        f'{VENDOR_PACKAGE}.model.vfm.diffusion',
        f'{VENDOR_PACKAGE}.model.vfm.diffusion.samplers',
        f'{VENDOR_PACKAGE}.model.vfm.mot',
        f'{VENDOR_PACKAGE}.model.vfm.tokenizers',
        f'{VENDOR_PACKAGE}.model.vfm.utils',
    }

    assert expected <= packages


def test_wheel_python_payload_contains_vendor_code_and_legal_files(
        tmp_path, monkeypatch):
    """Exercise setuptools' pure-Python payload used by wheel builds."""
    package_data = _package_data_from_setup()
    monkeypatch.chdir(REPO_ROOT)

    distribution = Distribution({
        'packages': find_packages(where=str(REPO_ROOT)),
        'package_data': package_data,
    })
    distribution.script_name = 'setup.py'
    command = build_py(distribution)
    command.ensure_finalized()
    command.build_lib = str(tmp_path / 'wheel-payload')
    command.run()

    built_vendor = Path(command.build_lib) / VENDOR_PATH.relative_to(REPO_ROOT)
    expected_files = [
        path.relative_to(VENDOR_PATH)
        for path in VENDOR_PATH.rglob('*.py')
    ]
    expected_files.extend(
        Path(filename) for filename in ('ATTRIBUTION.txt', 'LICENSE', 'NOTICE'))

    missing = [
        str(relative_path) for relative_path in expected_files
        if not (built_vendor / relative_path).is_file()
    ]
    assert not missing, f'missing from wheel Python payload: {missing}'

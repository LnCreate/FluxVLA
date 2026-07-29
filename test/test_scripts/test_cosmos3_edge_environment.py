from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def _load_module():
    path = (Path(__file__).parents[2] /
            'scripts/check_cosmos3_edge_environment.py')
    spec = spec_from_file_location('_cosmos3_environment_test', path)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_torchrun_local_rank_binds_matching_cuda_device(monkeypatch):
    module = _load_module()
    selected = []
    monkeypatch.setenv('WORLD_SIZE', '2')
    monkeypatch.setenv('LOCAL_RANK', '1')
    monkeypatch.setattr(module.torch.cuda, 'set_device', selected.append)
    errors = []

    local_rank = module._bind_local_cuda_device(errors, device_count=2)

    assert local_rank == 1
    assert selected == [1]
    assert errors == []


def test_torchrun_rejects_out_of_range_local_rank(monkeypatch):
    module = _load_module()
    monkeypatch.setenv('WORLD_SIZE', '2')
    monkeypatch.setenv('LOCAL_RANK', '2')
    errors = []

    local_rank = module._bind_local_cuda_device(errors, device_count=2)

    assert local_rank is None
    assert 'outside 2 CUDA devices' in errors[0]


def test_flux_runtime_accepts_only_repository_cuda_torch_pairs(monkeypatch):
    module = _load_module()
    monkeypatch.setattr(module.torch, '__version__', '2.6.0+cu124')
    monkeypatch.setattr(module.torch.version, 'cuda', '12.4')
    errors = []
    module._validate_flux_runtime('flux', errors)
    assert errors == []

    monkeypatch.setattr(module.torch, '__version__', '2.10.0+cu130')
    monkeypatch.setattr(module.torch.version, 'cuda', '13.0')
    module._validate_flux_runtime('flux', errors)
    assert 'repository-supported runtime pair' in errors[-1]


def test_flux5090_runtime_requires_cu128_stack(monkeypatch):
    module = _load_module()
    monkeypatch.setattr(module.torch, '__version__', '2.6.0+cu124')
    monkeypatch.setattr(module.torch.version, 'cuda', '12.4')
    errors = []
    module._validate_flux_runtime('flux5090', errors)
    assert len(errors) == 2


def test_oracle_runtime_requires_torch210_cu128(monkeypatch):
    module = _load_module()
    monkeypatch.setattr(module.torch, '__version__', '2.10.0+cu128')
    monkeypatch.setattr(module.torch.version, 'cuda', '12.8')
    errors = []
    module._validate_oracle_runtime(errors)
    assert errors == []

    monkeypatch.setattr(module.torch.version, 'cuda', '13.0')
    module._validate_oracle_runtime(errors)
    assert 'CUDA 12.8 runtime' in errors[-1]

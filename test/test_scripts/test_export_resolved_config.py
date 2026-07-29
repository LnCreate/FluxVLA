from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from mmengine import Config


def _load_module():
    path = (Path(__file__).parents[2] / 'scripts' /
            'export_resolved_config.py')
    spec = spec_from_file_location('_export_resolved_config_test', path)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_export_libero_config_is_self_contained(tmp_path):
    module = _load_module()
    root = Path(__file__).parents[2]
    source = (root / 'configs/cosmos3' /
              'cosmos3edge_libero_task_smoke.py')
    output = tmp_path / 'config.json'

    exported = module.export_config(source, output)
    reloaded = Config.fromfile(output)

    assert exported['model']['action_horizon'] == 16
    assert reloaded.model.action_horizon == 16
    assert reloaded.model.ori_action_dim == 7
    assert reloaded.eval.task_suite_name == 'libero_spatial'
    assert reloaded.eval.num_inference_steps == 10

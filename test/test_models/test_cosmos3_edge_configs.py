from pathlib import Path
from runpy import run_path

import draccus
from mmengine import Config
import torch
from safetensors.torch import save_file


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = REPO_ROOT / 'configs/cosmos3'


def _load(name, monkeypatch):
    # Entry configs must resolve their shared builder independently of cwd.
    monkeypatch.chdir('/tmp')
    return Config.fromfile(CONFIG_ROOT / name)


def test_all_edge_experiment_configs_load(monkeypatch):
    names = [
        'cosmos3edge_libero_task_smoke.py',
        'cosmos3edge_libero_task_overfit.py',
        'cosmos3edge_libero_task_smoke_fresh_domain.py',
        'cosmos3edge_libero_task_smoke_fresh_all.py',
        'cosmos3edge_libero_task_partial.py',
        'cosmos3edge_libero_task_partial_10k_from_action2k.py',
        'cosmos3edge_libero_full_finetune.py',
        'cosmos3edge_libero_spatial_posttrain.py',
        'cosmos3edge_libero_object_posttrain.py',
        'cosmos3edge_libero_goal_posttrain.py',
        'cosmos3edge_libero_10_posttrain.py',
    ]

    for name in names:
        cfg = _load(name, monkeypatch)
        assert cfg.model.type == 'Cosmos3FlowMatching'
        assert cfg.model.vlm_backbone.type == 'Cosmos3EdgeBackbone'
        assert cfg.model.strict_mapping is True
        assert cfg.model.checkpoint_min_coverage == 1.0
        # scripts/train.py persists the resolved config before model build.
        draccus.dump(cfg.to_dict())


def test_libero_smoke_keeps_native_7d_action_and_single_task(monkeypatch):
    cfg = _load('cosmos3edge_libero_task_smoke.py', monkeypatch)

    assert cfg.model.ori_action_dim == 7
    assert cfg.model.action_horizon == 16
    assert cfg.model.freeze_non_action_components is True
    assert cfg.model.enable_vision_loss is False
    assert cfg.seed == 7
    assert cfg.model.vlm_backbone.vlm_config.pad_token_id == 11
    assert cfg.model.vlm_backbone.vlm_config.use_und_k_norm_for_gen is True
    assert cfg.runner.collator.pad_id == 11
    dataset = cfg.train_dataloader.dataset.datasets
    assert dataset.task_indices == [0]
    assert dataset.require_full_window is True
    assert cfg.eval.task_ids == [0]


def test_libero_overfit_gate_is_exact_and_deterministic(monkeypatch):
    cfg = _load('cosmos3edge_libero_task_overfit.py', monkeypatch)

    dataset = cfg.train_dataloader.dataset.datasets
    prompt = next(
        transform for transform in dataset.transforms
        if transform.type == 'ProcessCosmos3Prompt')
    assert cfg.seed == 7
    assert cfg.runner.max_steps == 100
    assert dataset.task_indices == [0]
    assert dataset.max_windows == 32
    assert prompt.cfg_dropout_rate == 0.0


def test_libero_action_initialization_variants_are_explicit(monkeypatch):
    fresh_domain = _load(
        'cosmos3edge_libero_task_smoke_fresh_domain.py', monkeypatch)
    fresh_all = _load(
        'cosmos3edge_libero_task_smoke_fresh_all.py', monkeypatch)

    assert fresh_domain.model.action_init == 'fresh_domain'
    assert fresh_domain.model.fresh_action_domain_ids == [5]
    assert fresh_all.model.action_init == 'fresh_all'
    assert fresh_all.model.max_action_dim == 64
    assert fresh_all.model.num_embodiment_domains == 32


def test_libero_suite_posttrain_configs_use_convergence_schedule(monkeypatch):
    names = [
        'cosmos3edge_libero_spatial_posttrain.py',
        'cosmos3edge_libero_object_posttrain.py',
        'cosmos3edge_libero_goal_posttrain.py',
        'cosmos3edge_libero_10_posttrain.py',
    ]

    for name in names:
        cfg = _load(name, monkeypatch)
        assert cfg.runner.max_steps == 30_000
        assert cfg.runner.save_iter_interval == 3_000
        assert cfg.runner.grad_accumulation_steps == 8
        assert cfg.runner.sharding_strategy == 'full-shard'


def test_libero_partial_10k_uses_action_warmstart_and_official_style_schedule(
        monkeypatch):
    cfg = _load(
        'cosmos3edge_libero_task_partial_10k_from_action2k.py', monkeypatch)

    assert cfg.model.pretrained_name_or_path.endswith(
        'step-002000-epoch-08-loss=8.6627.safetensors')
    assert cfg.model.freeze_non_moe_vlm_backbone is True
    assert cfg.model.freeze_non_action_components is False
    assert cfg.model.enable_vision_loss is True
    assert cfg.model.rectified_flow_training_config.vision_loss_weight == 10.0
    assert cfg.model.rectified_flow_training_config.action_loss_weight == 10.0
    assert cfg.runner.max_steps == 10_000
    assert cfg.runner.save_iter_interval == 2_000
    assert cfg.runner.max_keep_ckpts == 2
    assert cfg.runner.lr_scheduler.type == 'linear-warmup+cosine-cycle'
    assert cfg.runner.lr_scheduler.warmup_steps == 500
    assert cfg.runner.lr_scheduler.cycle_steps == 16_000
    assert cfg.train_dataloader.dataset.datasets.data_root_path == (
        '/mnt/data/cpfs/mnt/data/yanis/FastWAM/data/'
        'libero_mujoco3.3.2/libero_spatial_no_noops_lerobot')


def test_edge_full_libero_recipe_matches_jikun_training_contract(monkeypatch):
    cfg = _load('cosmos3edge_libero_full_finetune.py', monkeypatch)
    dataset = cfg.train_dataloader.dataset.datasets
    transforms = {transform.type: transform for transform in dataset.transforms}

    assert cfg.model.pretrained_name_or_path == './checkpoints/Cosmos3-Edge'
    assert cfg.model.action_init == 'checkpoint'
    assert cfg.model.freeze_non_moe_vlm_backbone is True
    assert cfg.model.freeze_non_action_components is False
    assert cfg.model.rectified_flow_training_config.action_loss_weight == 10.0
    assert cfg.model.rectified_flow_training_config.vision_loss_weight == 1.0
    assert (cfg.model.rectified_flow_training_config.normalize_loss_by_active
            is False)
    assert cfg.model.rectified_flow_inference_config.num_steps == 30

    assert cfg.train_dataloader.per_device_batch_size == 8
    assert cfg.train_dataloader.per_device_num_workers == 4
    assert tuple(dataset.data_root_path) == (
        '/mnt/data/cpfs/mnt/data/yanis/FastWAM/data/libero_mujoco3.3.2/libero_spatial_no_noops_lerobot',
        '/mnt/data/cpfs/mnt/data/yanis/FastWAM/data/libero_mujoco3.3.2/libero_object_no_noops_lerobot',
        '/mnt/data/cpfs/mnt/data/yanis/FastWAM/data/libero_mujoco3.3.2/libero_goal_no_noops_lerobot',
        '/mnt/data/cpfs/mnt/data/yanis/FastWAM/data/libero_mujoco3.3.2/libero_10_no_noops_lerobot',
    )
    assert dataset.require_full_window is True
    assert dataset.repeat_to_full_length is True
    assert transforms['ResizeImages'].height == 128
    assert transforms['ResizeImages'].width == 128
    assert transforms['BuildCosmos3Sequence'].mode == 'joint'

    assert cfg.runner.max_steps is None
    assert cfg.runner.max_epochs == 12
    assert cfg.runner.grad_accumulation_steps == 1
    assert cfg.runner.optimizer.lr == 8e-5
    assert cfg.runner.optimizer.paramwise_learning_rate[
        'action_in_proj.'] == 4e-4
    assert cfg.runner.lr_scheduler.warmup_ratio == 0.0
    assert cfg.runner.sharding_strategy == 'full-shard'
    assert cfg.eval.task_suite_name == 'libero_10'
    assert cfg.eval.norm_stats_key == 'all_libero_no_noops'
    assert cfg.eval.num_trials_per_task == 50
    assert cfg.eval.num_inference_steps == 30


def test_edge_config_reads_action_width_and_domains_from_checkpoint_header(
        tmp_path):
    checkpoint = tmp_path / 'Cosmos3-Edge'
    checkpoint.mkdir()
    save_file(
        {
            'model.net.llm2action.bias.weight': torch.zeros(40, 96),
            'model.net.language_model.model.layers.0.mlp.up_proj.weight':
            torch.zeros(2, 2),
            'model.net.language_model.model.layers.0.mlp.down_proj.weight':
            torch.zeros(2, 2),
        }, checkpoint / 'model.safetensors')
    common = run_path(str(CONFIG_ROOT / 'cosmos3edge_common.py'))

    assert common['_checkpoint_action_layout'](checkpoint) == (96, 40)


def test_action_layout_can_come_from_explicit_checkpoint_config(tmp_path):
    checkpoint = tmp_path / 'Cosmos3-Edge'
    checkpoint.mkdir()
    save_file(
        {
            'model.net.language_model.model.layers.0.mlp.up_proj.weight':
            torch.zeros(2, 2),
            'model.net.language_model.model.layers.0.mlp.down_proj.weight':
            torch.zeros(2, 2),
        }, checkpoint / 'model.safetensors')
    (checkpoint / 'config.json').write_text(
        '{"max_action_dim": 80, "num_embodiment_domains": 48}',
        encoding='utf-8')
    common = run_path(str(CONFIG_ROOT / 'cosmos3edge_common.py'))

    assert common['_checkpoint_action_layout'](checkpoint) == (80, 48)

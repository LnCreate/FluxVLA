# Copyright 2026 Limx Dynamics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import importlib
import importlib.util
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_checkpoint_module():
    path = (REPO_ROOT / 'fluxvla/models/vlas/cosmos3/checkpoint_mixin.py')
    name = '_fluxvla_cosmos3_checkpoint_mixin_test'
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_checkpoint_module = _load_checkpoint_module()
Cosmos3CheckpointMixin = _checkpoint_module.Cosmos3CheckpointMixin
inspect_cosmos3_checkpoint = _checkpoint_module.inspect_cosmos3_checkpoint


class _Registry:

    def register_module(self, *args, **kwargs):
        del args, kwargs
        return lambda cls: cls


def _import_edge_module_without_fluxvla_root():
    """Import only Cosmos3 modules; FluxVLA's global registry probes CUDA."""

    saved = {
        name: module
        for name, module in sys.modules.items()
        if name == 'fluxvla' or name.startswith('fluxvla.')
    }
    for name in saved:
        sys.modules.pop(name, None)

    package_paths = {
        'fluxvla': REPO_ROOT / 'fluxvla',
        'fluxvla.models': REPO_ROOT / 'fluxvla/models',
        'fluxvla.models.backbones': REPO_ROOT / 'fluxvla/models/backbones',
        'fluxvla.models.backbones.vlms':
        REPO_ROOT / 'fluxvla/models/backbones/vlms',
        'fluxvla.models.backbones.vlms.cosmos3':
        REPO_ROOT / 'fluxvla/models/backbones/vlms/cosmos3',
        'fluxvla.models.third_party_models':
        REPO_ROOT / 'fluxvla/models/third_party_models',
        'fluxvla.models.third_party_models.cosmos3':
        REPO_ROOT / 'fluxvla/models/third_party_models/cosmos3',
        'fluxvla.models.third_party_models.cosmos3.data':
        REPO_ROOT / 'fluxvla/models/third_party_models/cosmos3/data',
        'fluxvla.models.third_party_models.cosmos3.data.vfm':
        REPO_ROOT / 'fluxvla/models/third_party_models/cosmos3/data/vfm',
        'fluxvla.models.third_party_models.cosmos3.model':
        REPO_ROOT / 'fluxvla/models/third_party_models/cosmos3/model',
        'fluxvla.models.third_party_models.cosmos3.model.vfm':
        REPO_ROOT / 'fluxvla/models/third_party_models/cosmos3/model/vfm',
        'fluxvla.models.third_party_models.cosmos3.model.vfm.mot':
        REPO_ROOT / 'fluxvla/models/third_party_models/cosmos3/model/vfm/mot',
        'fluxvla.models.third_party_models.cosmos3.model.vfm.utils':
        REPO_ROOT / 'fluxvla/models/third_party_models/cosmos3/model/vfm/utils',
    }
    for name, path in package_paths.items():
        package = types.ModuleType(name)
        package.__path__ = [str(path)]
        sys.modules[name] = package

    engines = types.ModuleType('fluxvla.engines')
    engines.VLM_BACKBONES = _Registry()
    sys.modules['fluxvla.engines'] = engines
    engine_utils = types.ModuleType('fluxvla.engines.utils')
    engine_utils.initialize_overwatch = lambda *args, **kwargs: SimpleNamespace(
        info=lambda *a, **k: None, warning=lambda *a, **k: None)
    sys.modules['fluxvla.engines.utils'] = engine_utils

    try:
        module = importlib.import_module(
            'fluxvla.models.backbones.vlms.cosmos3.cosmos3_edge_backbone')
        attention = importlib.import_module(
            'fluxvla.models.backbones.vlms.cosmos3.cosmos3_attention')
        return module, attention.build_packed_sequence
    finally:
        for name in list(sys.modules):
            if name == 'fluxvla' or name.startswith('fluxvla.'):
                sys.modules.pop(name, None)
        sys.modules.update(saved)


def _write_nested_checkpoint(root, tensors, config=None):
    transformer = root / 'transformer'
    transformer.mkdir(parents=True)
    shard = transformer / 'diffusion_pytorch_model-00001-of-00001.safetensors'
    save_file(tensors, shard)
    weight_map = {key: f'transformer/{shard.name}' for key in tensors}
    (root / 'model.safetensors.index.json').write_text(
        json.dumps({'weight_map': weight_map}), encoding='utf-8')
    if config is not None:
        (root / 'config.json').write_text(
            json.dumps(config), encoding='utf-8')
    return root


class _TinyCheckpointModel(Cosmos3CheckpointMixin, nn.Module):

    def __init__(self, checkpoint, *, bias=False, architecture='qwen3_vl'):
        nn.Module.__init__(self)
        self.layers = nn.ModuleList([nn.Linear(2, 2, bias=bias)])
        self.vlm_backbone = SimpleNamespace(
            architecture_family=architecture)
        self.pretrained_name_or_path = str(checkpoint)
        self.name_mapping = None
        self.strict_mapping = False
        self.checkpoint_min_coverage = 1.0
        self.checkpoint_missing_allowlist = ()


class _TinyVisionModel(Cosmos3CheckpointMixin, nn.Module):

    def __init__(self, checkpoint):
        nn.Module.__init__(self)
        self.vision_in_proj = nn.Module()
        self.vision_in_proj.projector = nn.Linear(2, 2, bias=False)
        self.vision_out_proj = nn.Module()
        self.vision_out_proj.projector = nn.Linear(2, 2, bias=False)
        self.vlm_backbone = SimpleNamespace(architecture_family='qwen3_vl')
        self.pretrained_name_or_path = str(checkpoint)
        self.name_mapping = None
        self.strict_mapping = False
        self.checkpoint_min_coverage = 1.0
        self.checkpoint_missing_allowlist = ()


class _TinyFullCheckpointModel(_TinyCheckpointModel):

    def __init__(self, checkpoint):
        super().__init__(checkpoint)
        self.vision_vae = nn.Linear(2, 2, bias=False)


class _TinyHybridModel(Cosmos3CheckpointMixin, nn.Module):

    def __init__(self, checkpoint):
        nn.Module.__init__(self)
        self.vlm_backbone = nn.Module()
        self.vlm_backbone.architecture_family = 'edge_nemotron'
        self.vlm_backbone.model = nn.Module()
        self.vlm_backbone.model.language_model = nn.Module()
        layer = nn.Module()
        layer.self_attn = nn.Module()
        layer.self_attn.q_proj = nn.Linear(2, 2, bias=False)
        layer.self_attn.q_proj_moe_gen = nn.Linear(2, 2, bias=False)
        layer.self_attn.q_norm_moe_gen = nn.Module()
        layer.self_attn.q_norm_moe_gen.weight = nn.Parameter(torch.ones(2))
        self.vlm_backbone.model.language_model.layers = nn.ModuleList([layer])
        self.pretrained_name_or_path = str(checkpoint)
        self.name_mapping = None
        self.strict_mapping = False
        self.checkpoint_min_coverage = 1.0


def test_edge_backbone_constructs_tiny_nemotron_architecture():
    edge, build_packed_sequence = _import_edge_module_without_fluxvla_root()
    from transformers.models.qwen3_vl.configuration_qwen3_vl import (
        Qwen3VLTextConfig)
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        Qwen3VLTextMLP, Qwen3VLTextRMSNorm)
    config = dict(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        mrope_section=[1, 1, 0],
        max_position_embeddings=64,
        torch_dtype='float32',
    )
    backbone = edge.Cosmos3EdgeBackbone(
        vlm_config=config, packed_attention_backend='sdpa')

    assert len(backbone.decoder_layers) == 2
    layer = backbone.decoder_layers[0]
    assert isinstance(layer.mlp, edge.Nemotron3DenseVLMLP)
    assert isinstance(layer.input_layernorm, edge.Nemotron3DenseVLRMSNorm)
    assert isinstance(layer.self_attn.q_norm, nn.Identity)
    assert isinstance(layer.self_attn.q_norm_moe_gen,
                      edge.Nemotron3DenseVLRMSNorm)
    assert isinstance(layer.self_attn.k_norm_und_for_gen,
                      edge.Nemotron3DenseVLRMSNorm)
    assert backbone.text_config.pad_token_id == 11
    assert backbone.embed_text_ids(torch.tensor([1, 2])).shape == (2, 16)
    state_keys = set(backbone.state_dict())
    assert ('model.language_model.layers.0.mlp.up_proj.weight' in state_keys)
    assert ('model.language_model.layers.0.mlp_moe_gen.up_proj.weight'
            in state_keys)
    assert ('model.language_model.layers.0.self_attn.q_proj_moe_gen.weight'
            in state_keys)
    assert ('model.language_model.layers.0.self_attn.'
            'k_norm_und_for_gen.weight' in state_keys)
    torch.testing.assert_close(
        edge.relu2(torch.tensor([-2.0, 3.0])), torch.tensor([0.0, 9.0]))

    hidden_states = torch.randn(4, 16)
    pack, attention_mask = build_packed_sequence(
        packed_sequence=hidden_states,
        attn_modes=['causal', 'full'],
        split_lens=[2, 2],
        sample_lens=[4],
        packed_und_token_indexes=torch.tensor([0, 1]),
        packed_gen_token_indexes=torch.tensor([2, 3]),
    )
    output, metadata = backbone.forward_packed(
        pack,
        attention_mask,
        position_ids=torch.arange(4),
    )
    assert output['causal_seq'].shape == (2, 16)
    assert output['full_only_seq'].shape == (2, 16)
    assert torch.isfinite(output['causal_seq']).all()
    assert torch.isfinite(output['full_only_seq']).all()
    assert metadata == {}

    qwen_config = Qwen3VLTextConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
    )
    qwen_layer = edge.Cosmos3TextDecoderLayer(
        qwen_config,
        0,
        qk_norm_for_text=True,
        qk_norm_for_diffusion=False,
    )
    assert isinstance(qwen_layer.mlp, Qwen3VLTextMLP)
    assert isinstance(qwen_layer.input_layernorm, Qwen3VLTextRMSNorm)

    no_init_backbone = edge.Cosmos3EdgeBackbone(
        vlm_config={**config, 'num_hidden_layers': 1},
        packed_attention_backend='sdpa',
        skip_init_weights=True,
    )
    torch.testing.assert_close(
        no_init_backbone.decoder_layers[0].self_attn.q_norm_moe_gen.weight,
        torch.ones(4),
    )
    folded_backbone = edge.Cosmos3EdgeBackbone(
        vlm_config=config,
        text_config_overrides={'num_hidden_layers': 56},
        packed_attention_backend='sdpa',
        skip_init_weights=True,
    )
    assert len(folded_backbone.decoder_layers) == 28


def test_edge_lora_targets_each_match_generation_tower_module():
    edge, _ = _import_edge_module_without_fluxvla_root()
    backbone = edge.Cosmos3EdgeBackbone(
        vlm_config=dict(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=24,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=4,
            mrope_section=[1, 1, 0],
            max_position_embeddings=64,
            torch_dtype='float32',
        ),
        packed_attention_backend='sdpa',
    )
    targets = [
        'self_attn.q_proj_moe_gen',
        'self_attn.k_proj_moe_gen',
        'self_attn.v_proj_moe_gen',
        'self_attn.o_proj_moe_gen',
        'mlp_moe_gen.up_proj',
        'mlp_moe_gen.down_proj',
    ]
    names = [name for name, _ in backbone.named_modules()]

    for target in targets:
        matches = [name for name in names if name.endswith(target)]
        assert matches, target
        assert all('moe_gen' in name for name in matches)


def test_checkpoint_loader_reads_root_index_and_nested_shard(tmp_path):
    expected = torch.arange(4, dtype=torch.float32).reshape(2, 2)
    checkpoint = _write_nested_checkpoint(
        tmp_path / 'Cosmos3-Nano',
        {
            '_orig_mod._checkpoint_wrapped_module.transformer.'
            'layers.0.weight': expected,
        })
    # The canonical root manifest deliberately takes precedence over a
    # component-local manifest for the same nested shard.
    nested_index = checkpoint / 'transformer' / (
        'diffusion_pytorch_model.safetensors.index.json')
    nested_index.write_text(
        json.dumps({
            'weight_map': {
                '_orig_mod._checkpoint_wrapped_module.transformer.'
                'layers.0.weight':
                'diffusion_pytorch_model-00001-of-00001.safetensors',
            }
        }),
        encoding='utf-8',
    )
    layout = inspect_cosmos3_checkpoint(checkpoint)
    assert layout.index_path == checkpoint / 'model.safetensors.index.json'
    assert layout.shard_paths == (
        checkpoint / 'transformer' /
        'diffusion_pytorch_model-00001-of-00001.safetensors', )

    model = _TinyCheckpointModel(checkpoint)
    report = model.from_pretrained()
    torch.testing.assert_close(model.layers[0].weight, expected)
    assert report.coverage == 1.0
    assert report.overall_coverage == 1.0
    assert report.allowed_missing_keys == ()


def test_edge_prefers_complete_nested_transformer_over_pipeline_root_manifest(
        tmp_path):
    checkpoint = tmp_path / 'Cosmos3-Edge'
    transformer = checkpoint / 'transformer'
    transformer.mkdir(parents=True)
    shard = transformer / (
        'diffusion_pytorch_model-00001-of-00001.safetensors')
    key = 'layers.0.self_attn.k_norm_und_for_gen.weight'
    save_file({key: torch.ones(2)}, shard)
    nested_index = transformer / (
        'diffusion_pytorch_model.safetensors.index.json')
    nested_index.write_text(
        json.dumps({'weight_map': {key: shard.name}}), encoding='utf-8')
    (checkpoint / 'model.safetensors.index.json').write_text(
        json.dumps({
            'weight_map': {
                # A pipeline root manifest may contain malformed aliases;
                # the complete nested transformer manifest is authoritative.
                'layers.layers.0.self_attn.k_norm_und_for_gen.weight':
                f'transformer/{shard.name}',
            }
        }),
        encoding='utf-8')
    (checkpoint / 'config.json').write_text(
        json.dumps({'model_type': 'cosmos3_edge'}), encoding='utf-8')

    layout = inspect_cosmos3_checkpoint(checkpoint)

    assert layout.index_path == nested_index
    assert key in layout.weight_map
    assert layout.architecture == 'edge_nemotron'


def test_checkpoint_inspection_rejects_ambiguous_indexes_without_root_manifest(
        tmp_path):
    checkpoint = tmp_path / 'Cosmos3-Nano'
    transformer = checkpoint / 'transformer'
    transformer.mkdir(parents=True)
    (checkpoint / 'diffusion_pytorch_model.safetensors.index.json').write_text(
        json.dumps({'weight_map': {}}), encoding='utf-8')
    (transformer / 'custom.safetensors.index.json').write_text(
        json.dumps({'weight_map': {}}), encoding='utf-8')

    with pytest.raises(ValueError, match='Ambiguous Cosmos3 safetensors'):
        inspect_cosmos3_checkpoint(checkpoint)


def test_checkpoint_loader_fails_below_coverage_threshold(tmp_path):
    checkpoint = _write_nested_checkpoint(
        tmp_path / 'Cosmos3-Nano',
        {'transformer.layers.0.weight': torch.ones(2, 2)})
    model = _TinyCheckpointModel(checkpoint, bias=True)
    with pytest.raises(ValueError, match='coverage'):
        model.from_pretrained()


def test_checkpoint_loader_rejects_edge_checkpoint_for_qwen_backbone(tmp_path):
    checkpoint = _write_nested_checkpoint(
        tmp_path / 'Cosmos3-Edge',
        {'transformer.layers.0.weight': torch.ones(2, 2)},
        config={'model_type': 'nemotron_3_dense_vl_text'},
    )
    model = _TinyCheckpointModel(checkpoint, architecture='qwen3_vl')
    with pytest.raises(ValueError, match='Edge/Nemotron'):
        model.from_pretrained()


def test_checkpoint_inspection_detects_pure_nemotron_hf(tmp_path):
    checkpoint = _write_nested_checkpoint(
        tmp_path / 'checkpoint',
        {'model.embeddings.weight': torch.ones(4, 2)})

    layout = inspect_cosmos3_checkpoint(checkpoint)

    assert layout.architecture == 'edge_nemotron'
    assert layout.checkpoint_format == 'nemotron_llm_hf'


def test_checkpoint_inspection_detects_native_fluxvla_architecture(tmp_path):
    qwen = _write_nested_checkpoint(
        tmp_path / 'qwen', {
            'vlm_backbone.model.language_model.layers.0.mlp.'
            'gate_proj.weight': torch.ones(2, 2),
        })
    edge = _write_nested_checkpoint(
        tmp_path / 'edge', {
            'vlm_backbone.model.language_model.layers.0.mlp.'
            'up_proj.weight': torch.ones(2, 2),
            'vlm_backbone.model.language_model.layers.0.mlp.'
            'down_proj.weight': torch.ones(2, 2),
        })

    qwen_layout = inspect_cosmos3_checkpoint(qwen)
    edge_layout = inspect_cosmos3_checkpoint(edge)

    assert qwen_layout.checkpoint_format == 'native_fluxvla'
    assert qwen_layout.architecture == 'qwen3_vl'
    assert edge_layout.checkpoint_format == 'native_fluxvla'
    assert edge_layout.architecture == 'edge_nemotron'


def test_checkpoint_loader_pairs_nemotron_hybrid_blocks(tmp_path):
    expected = torch.arange(4, dtype=torch.float32).reshape(2, 2)
    checkpoint = _write_nested_checkpoint(
        tmp_path / 'Cosmos3-Edge', {
            'model.language_model.layers.0.mixer.q_proj.weight': expected,
        })
    model = _TinyHybridModel(checkpoint)
    report = model.from_pretrained()
    loaded = model.vlm_backbone.model.language_model.layers[
        0].self_attn.q_proj.weight
    initialized = model.vlm_backbone.model.language_model.layers[
        0].self_attn.q_proj_moe_gen.weight
    torch.testing.assert_close(loaded, expected)
    torch.testing.assert_close(initialized, expected)
    assert report.checkpoint_format == 'nemotron_hybrid_hf'
    assert report.coverage == 1.0
    assert report.initialized_keys == (
        'vlm_backbone.model.language_model.layers.0.self_attn.'
        'q_proj_moe_gen.weight', )
    assert report.allowed_missing_keys == (
        'vlm_backbone.model.language_model.layers.0.self_attn.'
        'q_norm_moe_gen.weight', )


def test_checkpoint_loader_rejects_ambiguous_shape_compatible_sources(
        tmp_path):
    checkpoint = _write_nested_checkpoint(
        tmp_path / 'Cosmos3-Nano', {
            'layers.0.weight': torch.ones(2, 2),
            'transformer.layers.0.weight': torch.zeros(2, 2),
        })

    with pytest.raises(ValueError, match='Multiple shape-compatible'):
        _TinyCheckpointModel(checkpoint).from_pretrained()


def test_checkpoint_loader_rejects_duplicate_hybrid_target(tmp_path):
    source = 'model.language_model.layers.0.mixer.q_proj.weight'
    checkpoint = _write_nested_checkpoint(
        tmp_path / 'Cosmos3-Edge', {
            source: torch.ones(2, 2),
            f'transformer.{source}': torch.zeros(2, 2),
        })

    with pytest.raises(ValueError, match='Multiple Nemotron hybrid'):
        _TinyHybridModel(checkpoint).from_pretrained()


def test_checkpoint_loader_rejects_out_of_range_hybrid_block(tmp_path):
    checkpoint = _write_nested_checkpoint(
        tmp_path / 'Cosmos3-Edge', {
            'model.language_model.layers.56.mixer.q_proj.weight':
            torch.ones(2, 2),
        })

    with pytest.raises(ValueError, match='below 56'):
        _TinyHybridModel(checkpoint).from_pretrained()


def test_checkpoint_loader_reports_explicit_allowed_missing(tmp_path):
    checkpoint = _write_nested_checkpoint(
        tmp_path / 'Cosmos3-Nano',
        {'transformer.layers.0.weight': torch.ones(2, 2)})
    model = _TinyCheckpointModel(checkpoint, bias=True)
    model.checkpoint_missing_allowlist = (r'layers\.0\.bias', )

    report = model.from_pretrained()

    assert report.coverage == 1.0
    assert report.overall_coverage == pytest.approx(4 / 6)
    assert report.allowed_missing_keys == ('layers.0.bias', )


def test_checkpoint_allowlist_can_explicitly_reinitialize_shape_mismatch(
        tmp_path):
    checkpoint = _write_nested_checkpoint(
        tmp_path / 'Cosmos3-Nano',
        {'transformer.layers.0.weight': torch.ones(3, 2)})
    model = _TinyCheckpointModel(checkpoint)
    model.checkpoint_missing_allowlist = (r'layers\.0\.weight', )

    report = model.from_pretrained()

    assert report.coverage == 1.0
    assert report.overall_coverage == 0.0
    assert report.allowed_missing_keys == ('layers.0.weight', )


def test_checkpoint_loader_maps_native_vision_projectors(tmp_path):
    vae_to_llm = torch.arange(4, dtype=torch.float32).reshape(2, 2)
    llm_to_vae = vae_to_llm.flip(0)
    checkpoint = _write_nested_checkpoint(
        tmp_path / 'Cosmos3-Nano', {
            'model.net.vae2llm.weight': vae_to_llm,
            'model.net.llm2vae.weight': llm_to_vae,
        }, config={'model_type': 'qwen3_vl'})
    model = _TinyVisionModel(checkpoint)

    report = model.from_pretrained()

    torch.testing.assert_close(model.vision_in_proj.projector.weight,
                               vae_to_llm)
    torch.testing.assert_close(model.vision_out_proj.projector.weight,
                               llm_to_vae)
    assert report.coverage == 1.0


def test_checkpoint_loader_maps_diffusers_edge_attention_names(tmp_path):
    reasoner = torch.arange(4, dtype=torch.float32).reshape(2, 2)
    generator = reasoner.flip(0)
    generator_norm = torch.tensor([2.0, 3.0])
    checkpoint = _write_nested_checkpoint(
        tmp_path / 'Cosmos3-Edge', {
            'layers.0.self_attn.to_q.weight': reasoner,
            'layers.0.self_attn.add_q_proj.weight': generator,
            'layers.0.self_attn.norm_added_q.weight': generator_norm,
        })
    model = _TinyHybridModel(checkpoint)

    report = model.from_pretrained()

    attention = model.vlm_backbone.model.language_model.layers[0].self_attn
    torch.testing.assert_close(attention.q_proj.weight, reasoner)
    torch.testing.assert_close(attention.q_proj_moe_gen.weight, generator)
    torch.testing.assert_close(attention.q_norm_moe_gen.weight,
                               generator_norm)
    assert report.checkpoint_format == 'diffusers_transformer'
    assert report.coverage == 1.0


def test_native_full_checkpoint_keeps_and_loads_exact_vae(tmp_path):
    layer_weight = torch.arange(4, dtype=torch.float32).reshape(2, 2)
    vae_weight = layer_weight.flip(1)
    checkpoint = _write_nested_checkpoint(
        tmp_path / 'Cosmos3-Nano', {
            'layers.0.weight': layer_weight,
            'vision_vae.weight': vae_weight,
        })
    model = _TinyFullCheckpointModel(checkpoint)
    original_vae = model.vision_vae

    report = model.from_pretrained()

    assert model.vision_vae is original_vae
    torch.testing.assert_close(model.vision_vae.weight, vae_weight)
    assert report.coverage == 1.0


def test_checkpoint_loader_rejects_unknown_format_and_edge_pt(tmp_path):
    unknown = _write_nested_checkpoint(
        tmp_path / 'unknown', {'linear.weight': torch.ones(2, 2)},
        config={'model_type': 'qwen3_vl'})
    with pytest.raises(ValueError, match='safetensors format'):
        _TinyCheckpointModel(unknown).from_pretrained()

    legacy = tmp_path / 'edge.pth'
    torch.save({'layers.0.weight': torch.ones(2, 2)}, legacy)
    edge_model = _TinyCheckpointModel(
        legacy, architecture='edge_nemotron')
    with pytest.raises(ValueError, match='must use inspected safetensors'):
        edge_model.from_pretrained()


def test_checkpoint_inspection_rejects_distributed_checkpoint(tmp_path):
    checkpoint = tmp_path / 'dcp' / 'model'
    checkpoint.mkdir(parents=True)
    (checkpoint / '.metadata').write_bytes(b'dcp')
    (checkpoint / '__0_0.distcp').write_bytes(b'dcp')

    with pytest.raises(ValueError, match='distributed checkpoint'):
        inspect_cosmos3_checkpoint(checkpoint)

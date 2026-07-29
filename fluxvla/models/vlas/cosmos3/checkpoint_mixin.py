# Copyright 2026 Limx Dynamics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Fail-fast Cosmos3 safetensors checkpoint inspection and loading."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable

import torch
from safetensors import safe_open


_INDEX_NAMES = (
    'model.safetensors.index.json',
    'diffusion_pytorch_model.safetensors.index.json',
)


def _canonical_checkpoint_key(name: str) -> str:
    """Strip compile/checkpoint wrapper segments without changing layout."""

    return name.replace('_orig_mod.', '').replace(
        '_checkpoint_wrapped_module.', '')


@dataclass(frozen=True)
class Cosmos3CheckpointLayout:
    root: Path
    checkpoint_format: str
    architecture: str
    index_path: Path | None
    weight_map: dict[str, Path]
    tensor_shapes: dict[str, tuple[int, ...]]

    @property
    def shard_paths(self) -> tuple[Path, ...]:
        return tuple(sorted(set(self.weight_map.values())))


@dataclass(frozen=True)
class Cosmos3CheckpointLoadReport:
    checkpoint_format: str
    architecture: str
    loaded_keys: tuple[str, ...]
    initialized_keys: tuple[str, ...]
    missing_keys: tuple[str, ...]
    allowed_missing_keys: tuple[str, ...]
    coverage: float
    overall_coverage: float


def _read_weight_map(index_path: Path) -> dict[str, str]:
    payload = json.loads(index_path.read_text(encoding='utf-8'))
    weight_map = payload.get('weight_map')
    if not isinstance(weight_map, dict):
        raise ValueError(
            f'{index_path} does not contain a safetensors weight_map.')
    if not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in weight_map.items()):
        raise TypeError(
            f'{index_path} weight_map must contain string keys and values.')
    return weight_map


def _index_candidates(root: Path) -> list[Path]:
    # Inspect every manifest in the two layouts supported by this loader.
    # Looking only for the well-known names first can silently hide a second,
    # custom-named manifest and make the selected checkpoint depend on list
    # order.
    candidates = list(root.glob('*.safetensors.index.json'))
    transformer_root = root / 'transformer'
    if transformer_root.is_dir():
        candidates.extend(
            transformer_root.glob('*.safetensors.index.json'))
    return sorted(set(candidates))


def _checkpoint_config_payloads(root: Path) -> list[dict]:
    candidates = [
        root / 'config.json',
        root / 'model_index.json',
        root / 'transformer' / 'config.json',
    ]
    if root.name == 'transformer':
        candidates.extend([
            root.parent / 'config.json',
            root.parent / 'model_index.json',
        ])
    payloads = []
    for path in candidates:
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            payloads.append(payload)
    return payloads


def _detect_architecture(root: Path, keys: Iterable[str]) -> str:
    normalized_keys = [
        _canonical_checkpoint_key(key).removeprefix('transformer.')
        for key in keys
    ]
    if any('model.language_model.layers.' in key and '.mixer.q_proj' in key
           for key in normalized_keys):
        return 'edge_nemotron'
    if 'model.embeddings.weight' in normalized_keys:
        return 'edge_nemotron'

    # Native FluxVLA/VFM and diffusers checkpoints have already-folded MoT
    # layers.  Qwen's gated MLP has gate/up/down projections, while Edge's
    # Nemotron ReLU2 MLP has only up/down projections.
    if any('.mlp.gate_proj.' in key for key in normalized_keys):
        return 'qwen3_vl'
    if (any('.mlp.up_proj.' in key for key in normalized_keys)
            and any('.mlp.down_proj.' in key for key in normalized_keys)):
        return 'edge_nemotron'

    config_text = json.dumps(
        _checkpoint_config_payloads(root), sort_keys=True).lower()
    path_text = str(root).lower()
    if 'nemotron' in config_text or 'cosmos3-edge' in config_text:
        return 'edge_nemotron'
    if 'qwen3' in config_text or 'qwen3_vl' in config_text:
        return 'qwen3_vl'
    if any(name in path_text
           for name in ('cosmos3-edge', 'cosmos3_edge', 'cosmos3edge')):
        return 'edge_nemotron'
    if any(name in path_text for name in (
            'cosmos3-nano',
            'cosmos3_nano',
            'cosmos3nano',
            'cosmos3-super',
            'cosmos3_super',
            'cosmos3super',
    )):
        return 'qwen3_vl'

    if any(key.startswith(('model.embed_tokens.', 'model.layers.'))
           for key in normalized_keys):
        return 'qwen3_vl'
    if any('model.language_model.layers.' in key and '.self_attn.' in key
           for key in normalized_keys):
        return 'qwen3_vl'
    return 'unknown'


def _detect_format(keys: Iterable[str]) -> str:
    normalized = {
        _canonical_checkpoint_key(key).removeprefix('transformer.')
        for key in keys
    }
    if any('model.language_model.layers.' in key and '.mixer.q_proj' in key
           for key in normalized):
        return 'nemotron_hybrid_hf'
    if 'model.embeddings.weight' in normalized:
        return 'nemotron_llm_hf'
    if any(key.startswith((
            'vlm_backbone.',
            'vision_in_proj.',
            'vision_out_proj.',
            'action_in_proj.',
            'action_out_proj.',
    )) for key in normalized):
        return 'native_fluxvla'
    if any(key.startswith(('language_model.', 'model.net.'))
           for key in normalized):
        return 'native_vfm'
    if any(
            key.startswith((
                'layers.',
                'embed_tokens.',
                'action_proj_in.',
                'action_proj_out.',
                'proj_in.',
                'proj_out.',
                'time_embedder.',
            )) for key in normalized):
        return 'diffusers_transformer'
    if any(key.startswith(('model.', 'lm_head.')) for key in normalized):
        return 'hf_language_model'
    return 'unknown_safetensors'


def _indexed_layout(root: Path,
                    index_path: Path) -> tuple[dict[str, Path], dict[str,
                                                                    tuple]]:
    indexed = _read_weight_map(index_path)
    shard_to_indexed_keys: dict[Path, list[str]] = {}
    for source_key, relative_path in indexed.items():
        shard_path = index_path.parent / relative_path
        shard_to_indexed_keys.setdefault(shard_path, []).append(source_key)

    weight_map: dict[str, Path] = {}
    shapes: dict[str, tuple[int, ...]] = {}
    for shard_path, indexed_keys in shard_to_indexed_keys.items():
        if not shard_path.is_file():
            raise FileNotFoundError(
                f'Cosmos3 checkpoint shard not found: {shard_path}')
        with safe_open(str(shard_path), framework='pt', device='cpu') as shard:
            actual_keys = set(shard.keys())
            for indexed_key in indexed_keys:
                actual_key = indexed_key
                if actual_key not in actual_keys and actual_key.startswith(
                        'transformer.'):
                    stripped = actual_key.removeprefix('transformer.')
                    if stripped in actual_keys:
                        actual_key = stripped
                if actual_key not in actual_keys and not actual_key.startswith(
                        'transformer.'):
                    prefixed = 'transformer.' + actual_key
                    if prefixed in actual_keys:
                        actual_key = prefixed
                if actual_key not in actual_keys:
                    raise KeyError(
                        f'{shard_path} is missing indexed tensor '
                        f'{indexed_key!r}.')
                if actual_key in weight_map and weight_map[
                        actual_key] != shard_path:
                    raise KeyError(
                        f'Checkpoint tensor {actual_key!r} occurs in multiple '
                        'shards.')
                weight_map[actual_key] = shard_path
                shapes[actual_key] = tuple(
                    shard.get_slice(actual_key).get_shape())
    return weight_map, shapes


def _unindexed_layout(shards: Iterable[Path]) -> tuple[dict[str, Path],
                                                       dict[str, tuple]]:
    weight_map: dict[str, Path] = {}
    shapes: dict[str, tuple[int, ...]] = {}
    for shard_path in sorted(shards):
        with safe_open(str(shard_path), framework='pt', device='cpu') as shard:
            for key in shard.keys():
                if key in weight_map:
                    raise KeyError(
                        f'Checkpoint tensor {key!r} occurs in both '
                        f'{weight_map[key]} and {shard_path}.')
                weight_map[key] = shard_path
                shapes[key] = tuple(shard.get_slice(key).get_shape())
    return weight_map, shapes


def inspect_cosmos3_checkpoint(
        checkpoint_path: str | Path) -> Cosmos3CheckpointLayout:
    """Inspect an HF/diffusers Cosmos3 safetensors checkpoint without loading it."""

    path = Path(checkpoint_path).expanduser()
    if path.is_file():
        if path.suffix == '.distcp' or path.name == '.metadata':
            raise ValueError(
                'PyTorch distributed checkpoint (DCP) format is not '
                'supported by the FluxVLA Cosmos3 loader. Convert it to '
                'indexed safetensors before loading.')
        if path.suffix != '.safetensors':
            raise ValueError(f'Unsupported Cosmos3 checkpoint file: {path}')
        weight_map, shapes = _unindexed_layout([path])
        root = path.parent
        index_path = None
    elif path.is_dir():
        root = path
        indexes = _index_candidates(root)
        # Edge publishes a root conditional-generation manifest and a nested
        # diffusion-transformer manifest. FluxVLA's policy model consumes the
        # latter because the Base root manifest omits generation/action
        # tensors. Nano/Super retain the canonical-root priority used by
        # PR #51.
        root_index = root / 'model.safetensors.index.json'
        edge_transformer_index = root / 'transformer' / (
            'diffusion_pytorch_model.safetensors.index.json')
        if (edge_transformer_index in indexes
                and _detect_architecture(root, ()) == 'edge_nemotron'):
            index_path = edge_transformer_index
        elif root_index in indexes:
            index_path = root_index
        elif len(indexes) > 1:
            relative_indexes = [str(index.relative_to(root))
                                for index in indexes]
            raise ValueError(
                'Ambiguous Cosmos3 safetensors indexes: found multiple '
                f'manifests under {root}: {relative_indexes}. Provide a '
                'canonical root model.safetensors.index.json or remove the '
                'unrelated manifests; refusing to select one by path order.')
        else:
            index_path = indexes[0] if indexes else None
        if index_path is not None:
            weight_map, shapes = _indexed_layout(root, index_path)
        else:
            shard_root = (root / 'transformer'
                          if (root / 'transformer').is_dir() else root)
            shards = sorted(shard_root.glob('*.safetensors'))
            if not shards:
                dcp_roots = (root, root / 'model')
                if any(
                        (candidate / '.metadata').is_file()
                        or next(candidate.glob('*.distcp'), None) is not None
                        for candidate in dcp_roots if candidate.is_dir()):
                    raise ValueError(
                        'PyTorch distributed checkpoint (DCP) format is not '
                        'supported by the FluxVLA Cosmos3 loader. Convert it '
                        'to indexed safetensors before loading.')
                raise FileNotFoundError(
                    f'No Cosmos3 safetensors or index found at {root}.')
            weight_map, shapes = _unindexed_layout(shards)
    else:
        raise FileNotFoundError(f'Cosmos3 checkpoint does not exist: {path}')

    return Cosmos3CheckpointLayout(
        root=root,
        checkpoint_format=_detect_format(weight_map),
        architecture=_detect_architecture(root, weight_map),
        index_path=index_path,
        weight_map=weight_map,
        tensor_shapes=shapes,
    )


def infer_cosmos3_action_layout(
        checkpoint: str | Path | Cosmos3CheckpointLayout) -> tuple[int, int]:
    """Return ``(Dmax, num_domains)`` from weights or an explicit config."""
    layout = (checkpoint if isinstance(checkpoint, Cosmos3CheckpointLayout)
              else inspect_cosmos3_checkpoint(checkpoint))
    candidates = []
    suffixes = (
        'action_out_proj.bias.weight',
        'llm2action.bias.weight',
        'action_proj_out.bias.weight',
    )
    for name, shape in layout.tensor_shapes.items():
        canonical = _canonical_checkpoint_key(name)
        if canonical.endswith(suffixes) and len(shape) == 2:
            candidates.append((int(shape[1]), int(shape[0]), name))
    unique = {(dmax, domains) for dmax, domains, _ in candidates}
    config_candidates = set()

    def visit(value):
        if isinstance(value, dict):
            action_dim = value.get('max_action_dim', value.get('action_dim'))
            domains = value.get('num_embodiment_domains')
            if (isinstance(action_dim, int) and action_dim > 0
                    and isinstance(domains, int) and domains > 0):
                config_candidates.add((action_dim, domains))
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    for payload in _checkpoint_config_payloads(layout.root):
        visit(payload)

    if len(unique) == 1:
        tensor_layout = next(iter(unique))
        if config_candidates and config_candidates != {tensor_layout}:
            raise ValueError(
                'Cosmos3 action layout disagrees between checkpoint tensors '
                f'{tensor_layout} and config values {sorted(config_candidates)}.')
        return tensor_layout
    if not unique and len(config_candidates) == 1:
        return next(iter(config_candidates))
    if len(unique) != 1:
        raise ValueError(
            'Expected exactly one Cosmos3 action layout in checkpoint '
            f'{layout.root}, found tensor candidates {candidates} and config '
            f'candidates {sorted(config_candidates)}. Refusing to guess Dmax '
            'or the embodiment-domain count.')


def _edge_hybrid_target(source_name: str) -> str | None:
    """Map an official 56-block Nemotron VL key to FluxVLA's 28-layer tree."""

    name = _canonical_checkpoint_key(source_name).removeprefix('transformer.')
    if name in ('lm_head.weight', 'model.lm_head.weight'):
        return 'vlm_backbone.lm_head.weight'
    if name == 'model.language_model.embeddings.weight':
        return 'vlm_backbone.model.language_model.embed_tokens.weight'
    if name == 'model.language_model.norm_f.weight':
        return 'vlm_backbone.model.language_model.norm.weight'

    match = re.fullmatch(
        r'model\.language_model\.layers\.(\d+)\.norm\.weight', name)
    if match:
        block = int(match.group(1))
        if block >= 56:
            raise ValueError(
                f'Nemotron hybrid block index must be below 56: {name}')
        norm = ('input_layernorm'
                if block % 2 == 0 else 'post_attention_layernorm')
        return ('vlm_backbone.model.language_model.layers.'
                f'{block // 2}.{norm}.weight')

    match = re.fullmatch(
        r'model\.language_model\.layers\.(\d+)\.mixer\.'
        r'(q_proj|k_proj|v_proj|o_proj)\.weight', name)
    if match:
        block = int(match.group(1))
        if block >= 56:
            raise ValueError(
                f'Nemotron hybrid block index must be below 56: {name}')
        if block % 2:
            raise ValueError(
                f'Nemotron attention tensor is in odd block: {name}')
        return ('vlm_backbone.model.language_model.layers.'
                f'{block // 2}.self_attn.{match.group(2)}.weight')

    match = re.fullmatch(
        r'model\.language_model\.layers\.(\d+)\.mixer\.'
        r'(up_proj|down_proj)\.weight', name)
    if match:
        block = int(match.group(1))
        if block >= 56:
            raise ValueError(
                f'Nemotron hybrid block index must be below 56: {name}')
        if block % 2 == 0:
            raise ValueError(f'Nemotron MLP tensor is in even block: {name}')
        return ('vlm_backbone.model.language_model.layers.'
                f'{block // 2}.mlp.{match.group(2)}.weight')
    return None


def _native_aliases(target_name: str) -> list[str]:
    aliases = []
    language_prefix = 'vlm_backbone.model.language_model.'
    if target_name.startswith(language_prefix):
        tail = target_name.removeprefix(language_prefix)
        aliases.extend([
            tail,
            f'model.{tail}',
            f'language_model.{tail}',
            f'language_model.model.{tail}',
            f'model.language_model.{tail}',
            f'model.net.language_model.model.{tail}',
        ])
        if tail == 'embed_tokens.weight':
            aliases.extend([
                'model.embeddings.weight',
                'model.language_model.embeddings.weight',
            ])
        # Official diffusers transformers keep the folded ``layers.*`` tree
        # but use diffusers attention projection names.
        diffusers_tail = tail
        for target_part, source_part in (
                ('.self_attn.q_proj_moe_gen.', '.self_attn.add_q_proj.'),
                ('.self_attn.k_proj_moe_gen.', '.self_attn.add_k_proj.'),
                ('.self_attn.v_proj_moe_gen.', '.self_attn.add_v_proj.'),
                ('.self_attn.o_proj_moe_gen.', '.self_attn.to_add_out.'),
                ('.self_attn.q_norm_moe_gen.', '.self_attn.norm_added_q.'),
                ('.self_attn.k_norm_moe_gen.', '.self_attn.norm_added_k.'),
                ('.self_attn.q_proj.', '.self_attn.to_q.'),
                ('.self_attn.k_proj.', '.self_attn.to_k.'),
                ('.self_attn.v_proj.', '.self_attn.to_v.'),
                ('.self_attn.o_proj.', '.self_attn.to_out.'),
                ('.self_attn.q_norm.', '.self_attn.norm_q.'),
                ('.self_attn.k_norm.', '.self_attn.norm_k.'),
        ):
            diffusers_tail = diffusers_tail.replace(target_part, source_part)
        aliases.append(diffusers_tail)
    elif target_name.startswith('vlm_backbone.lm_head.'):
        tail = target_name.removeprefix('vlm_backbone.')
        aliases.extend([
            tail,
            f'model.{tail}',
            f'language_model.{tail}',
            f'model.net.language_model.{tail}',
        ])

    native_prefixes = {
        'vision_in_proj.projector.': 'vae2llm.',
        'vision_out_proj.projector.': 'llm2vae.',
        'action_in_proj.': 'action2llm.',
        'action_out_proj.': 'llm2action.',
    }
    for target_prefix, source_prefix in native_prefixes.items():
        if target_name.startswith(target_prefix):
            tail = target_name.removeprefix(target_prefix)
            aliases.extend([
                source_prefix + tail,
                'model.net.' + source_prefix + tail,
            ])
    diffusers_prefixes = {
        'vision_in_proj.projector.': 'proj_in.',
        'vision_out_proj.projector.': 'proj_out.',
        'action_in_proj.': 'action_proj_in.',
        'action_out_proj.': 'action_proj_out.',
        'time_embedder.mlp.0.': 'time_embedder.linear_1.',
        'time_embedder.mlp.2.': 'time_embedder.linear_2.',
    }
    for target_prefix, source_prefix in diffusers_prefixes.items():
        if target_name.startswith(target_prefix):
            aliases.append(
                source_prefix + target_name.removeprefix(target_prefix))
    if target_name.startswith('time_embedder.'):
        aliases.extend([
            target_name,
            'model.net.' + target_name,
        ])
    if target_name in {
            'action_modality_embed', 'action_modality_embed.weight'
    }:
        aliases.extend([
            'action_modality_embed',
            'action_modality_embed.weight',
            'model.net.action_modality_embed',
            'model.net.action_modality_embed.weight',
        ])
    return aliases


class Cosmos3CheckpointMixin:
    """Cosmos3 loader with architecture checks and parameter coverage gates."""

    checkpoint_min_coverage: float = 1.0
    checkpoint_missing_allowlist: tuple[str, ...] = ()

    def _checkpoint_source_candidates(
        self,
        target_name: str,
        hybrid_targets: dict[str, str],
    ) -> list[str]:
        candidates = [target_name]
        mapper = getattr(self, '_mapped_name_candidates', None)
        if callable(mapper) and getattr(self, 'name_mapping', None):
            candidates.extend(name for _, name in mapper(target_name))
        candidates.extend(_native_aliases(target_name))
        hybrid_source = hybrid_targets.get(target_name)
        if hybrid_source is not None:
            candidates.append(hybrid_source)

        expanded = []
        for name in candidates:
            expanded.append(name)
            if not name.startswith('transformer.'):
                expanded.append('transformer.' + name)
        return list(dict.fromkeys(expanded))

    def _validate_checkpoint_architecture(
            self, layout: Cosmos3CheckpointLayout) -> None:
        backbone = getattr(self, 'vlm_backbone', None)
        expected = getattr(backbone, 'architecture_family', 'unknown')
        actual = layout.architecture
        if actual == 'unknown':
            raise ValueError(
                'Unable to determine the Cosmos3 checkpoint architecture; '
                'refusing to guess between Edge/Nemotron and Nano/Super '
                'Qwen.')
        if layout.checkpoint_format == 'unknown_safetensors':
            raise ValueError(
                'Unable to determine the Cosmos3 safetensors format; '
                'refusing an unsafe best-effort load.')
        if actual == 'edge_nemotron' and expected != 'edge_nemotron':
            raise ValueError(
                'Cosmos3-Edge/Nemotron checkpoint cannot be loaded into the '
                f'{expected} backbone. Select model_size="edge" and '
                'Cosmos3EdgeBackbone explicitly.')
        if actual == 'qwen3_vl' and expected == 'edge_nemotron':
            raise ValueError(
                'A Cosmos3 Nano/Super Qwen checkpoint cannot be loaded into '
                'Cosmos3EdgeBackbone.')

    def _load_cosmos3_safetensors(
        self,
        layout: Cosmos3CheckpointLayout,
    ) -> Cosmos3CheckpointLoadReport:
        self._validate_checkpoint_architecture(layout)
        threshold = (1.0 if getattr(self, 'strict_mapping', False) else float(
            getattr(self, 'checkpoint_min_coverage', 1.0)))
        if not 0.0 < threshold <= 1.0:
            raise ValueError(
                'checkpoint_min_coverage must be in (0, 1], got '
                f'{threshold}.')

        hybrid_targets = {}
        canonical_sources: dict[str, str] = {}
        for source_name in layout.weight_map:
            canonical_name = _canonical_checkpoint_key(source_name)
            if (canonical_name in canonical_sources
                    and canonical_sources[canonical_name] != source_name):
                raise ValueError(
                    'Cosmos3 checkpoint keys collide after wrapper-prefix '
                    f'normalization: {canonical_sources[canonical_name]!r} '
                    f'and {source_name!r}.')
            canonical_sources[canonical_name] = source_name
            target_name = _edge_hybrid_target(source_name)
            if target_name is not None:
                previous_source = hybrid_targets.get(target_name)
                if (previous_source is not None
                        and previous_source != canonical_name):
                    raise ValueError(
                        'Multiple Nemotron hybrid checkpoint tensors map to '
                        f'target {target_name!r}: {previous_source!r} and '
                        f'{canonical_name!r}.')
                hybrid_targets[target_name] = canonical_name

        target_parameters = dict(self.named_parameters())
        target_to_source: dict[str, str] = {}
        target_to_initialized_target: dict[str, str] = {}
        used_sources: set[str] = set()
        mismatches: list[str] = []
        missing: list[str] = []
        allowed_missing: list[str] = []
        loaded_numel = 0
        allowed_missing_numel = 0
        total_numel = sum(param.numel()
                          for param in target_parameters.values())

        allowlist = getattr(self, 'checkpoint_missing_allowlist', ()) or ()
        try:
            allowed_missing_patterns = [
                re.compile(pattern) for pattern in allowlist
            ]
        except re.error as error:
            raise ValueError(
                'checkpoint_missing_allowlist contains an invalid regular '
                f'expression: {error}') from error

        for target_name, parameter in target_parameters.items():
            candidates = self._checkpoint_source_candidates(
                target_name, hybrid_targets)
            present = [
                canonical_sources[name] for name in candidates
                if name in canonical_sources
            ]
            matching = [
                name for name in present
                if layout.tensor_shapes[name] == tuple(parameter.shape)
            ]
            if len(matching) > 1:
                raise ValueError(
                    'Multiple shape-compatible Cosmos3 checkpoint tensors map '
                    f'to target {target_name!r}: {matching}.')
            if not matching:
                is_allowed = any(
                    pattern.fullmatch(target_name)
                    for pattern in allowed_missing_patterns)
                if is_allowed:
                    allowed_missing.append(target_name)
                    allowed_missing_numel += parameter.numel()
                elif present:
                    shapes = {
                        name: layout.tensor_shapes[name]
                        for name in present
                    }
                    mismatches.append(
                        f'{target_name}: model={tuple(parameter.shape)}, '
                        f'checkpoint={shapes}')
                else:
                    missing.append(target_name)
                continue
            source_name = matching[0]
            if source_name in used_sources:
                raise ValueError(
                    f'Checkpoint tensor {source_name!r} maps to multiple model '
                    'parameters.')
            used_sources.add(source_name)
            target_to_source[target_name] = source_name
            loaded_numel += parameter.numel()

        raw_edge_backbone = layout.checkpoint_format in {
            'nemotron_hybrid_hf',
            'nemotron_llm_hf',
        }
        if raw_edge_backbone:
            still_missing = []
            for target_name in missing:
                source_target = target_name.replace('_moe_gen', '')
                if ('_moe_gen' in target_name
                        and source_target in target_to_source
                        and target_parameters[source_target].shape
                        == target_parameters[target_name].shape):
                    target_to_initialized_target[target_name] = source_target
                    loaded_numel += target_parameters[target_name].numel()
                elif re.search(
                        r'\.self_attn\.[qk]_norm_moe_gen\.weight$',
                        target_name):
                    # Edge uses Identity for reasoner QK norms, so there is no
                    # understanding parameter to copy.  Official init_moe()
                    # deliberately leaves these generation RMSNorms freshly
                    # initialized.
                    allowed_missing.append(target_name)
                    allowed_missing_numel += target_parameters[
                        target_name].numel()
                else:
                    still_missing.append(target_name)
            missing = still_missing

        if mismatches:
            raise ValueError(
                'Cosmos3 checkpoint contains incompatible tensor shapes. '
                f'First up to 10: {mismatches[:10]}')
        required_numel = total_numel - allowed_missing_numel
        coverage = (1.0 if required_numel == 0 else
                    loaded_numel / required_numel)
        overall_coverage = (1.0 if total_numel == 0 else
                            loaded_numel / total_numel)
        if coverage < threshold:
            raise ValueError(
                'Cosmos3 checkpoint coverage is below the fail-fast threshold: '
                f'{coverage:.2%} < {threshold:.2%}; missing '
                f'{len(missing)} parameter(s). First up to 10: '
                f'{missing[:10]}')

        targets_by_shard: dict[Path, list[tuple[str, str]]] = {}
        for target_name, source_name in target_to_source.items():
            targets_by_shard.setdefault(layout.weight_map[source_name],
                                        []).append((target_name, source_name))
        with torch.no_grad():
            for shard_path, assignments in targets_by_shard.items():
                with safe_open(
                        str(shard_path), framework='pt', device='cpu') as shard:
                    for target_name, source_name in assignments:
                        tensor = shard.get_tensor(source_name)
                        parameter = target_parameters[target_name]
                        parameter.copy_(tensor.to(
                            device=parameter.device, dtype=parameter.dtype))
            for target_name, source_target in (
                    target_to_initialized_target.items()):
                target_parameters[target_name].copy_(
                    target_parameters[source_target])

        return Cosmos3CheckpointLoadReport(
            checkpoint_format=layout.checkpoint_format,
            architecture=layout.architecture,
            loaded_keys=tuple(sorted(target_to_source)),
            initialized_keys=tuple(sorted(target_to_initialized_target)),
            missing_keys=tuple(sorted(missing)),
            allowed_missing_keys=tuple(sorted(allowed_missing)),
            coverage=coverage,
            overall_coverage=overall_coverage,
        )

    def from_pretrained(self):
        checkpoint_path = getattr(self, 'pretrained_name_or_path', None)
        if checkpoint_path is None:
            return None
        path_text = str(checkpoint_path)
        if path_text.lower().endswith(('.pt', '.pth')):
            architecture = getattr(
                getattr(self, 'vlm_backbone', None),
                'architecture_family',
                'unknown',
            )
            if architecture == 'edge_nemotron':
                raise ValueError(
                    'Cosmos3-Edge checkpoints must use inspected '
                    'safetensors; legacy .pt/.pth loading cannot verify the '
                    'Nemotron architecture or tensor coverage.')
            return super().from_pretrained()

        layout = inspect_cosmos3_checkpoint(path_text)
        normalized_source_keys = {
            _canonical_checkpoint_key(key).removeprefix('transformer.')
            for key in layout.weight_map
        }
        has_exact_vae = any(
            key.startswith('vision_vae.') for key in normalized_source_keys)
        has_exact_visual = any(
            key.startswith('vlm_backbone.model.visual.')
            for key in normalized_source_keys)

        vision_vae_module = getattr(self, 'vision_vae', None)
        if (vision_vae_module is not None and not has_exact_vae
                and not getattr(vision_vae_module, '_weights_loaded', True)):
            requested = getattr(vision_vae_module,
                                'requested_pretrained_path', None)
            raise FileNotFoundError(
                'Cosmos3 checkpoint contains no embedded vision_vae weights '
                'and the external Wan2.2 VAE is unavailable. Expected '
                f'Wan2.2_VAE.pth at {requested!r}.')

        vision_vae = (None if has_exact_vae else
                      self._modules.pop('vision_vae', None))
        visual = None
        backbone_model = getattr(getattr(self, 'vlm_backbone', None), 'model',
                                 None)
        if (not has_exact_visual
                and isinstance(backbone_model, torch.nn.Module)):
            visual = backbone_model._modules.pop('visual', None)
        try:
            report = self._load_cosmos3_safetensors(layout)
            if has_exact_vae and vision_vae_module is not None:
                mark_loaded = getattr(vision_vae_module,
                                      'mark_weights_loaded', None)
                if callable(mark_loaded):
                    mark_loaded()
            initialize_action = getattr(
                self, '_reinitialize_action_policy', None)
            if callable(initialize_action):
                initialized = tuple(initialize_action())
                if initialized:
                    report = replace(
                        report,
                        initialized_keys=tuple(sorted(
                            (*report.initialized_keys, *initialized))),
                    )
            self.checkpoint_load_report = report
            return report
        finally:
            if visual is not None:
                backbone_model._modules['visual'] = visual
            if vision_vae is not None:
                self._modules['vision_vae'] = vision_vae


__all__ = [
    'Cosmos3CheckpointLayout',
    'Cosmos3CheckpointLoadReport',
    'Cosmos3CheckpointMixin',
    'infer_cosmos3_action_layout',
    'inspect_cosmos3_checkpoint',
]

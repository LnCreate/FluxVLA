#!/usr/bin/env python3
"""Fail-fast CUDA/NCCL check for Cosmos3-Edge training and inference.

Run this script from the exact Python environment used by a job.  For the
two-GPU training gate, launch it with ``torchrun --nproc-per-node=2`` and pass
``--require-distributed`` so a real NCCL all-reduce is exercised.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import sys
from typing import Any

import torch
import torch.distributed as dist


def _version_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in re.findall(r'\d+', value)[:3])


def _driver_version() -> str | None:
    path = Path('/proc/driver/nvidia/version')
    if not path.exists():
        return None
    match = re.search(r'Kernel Module\s+([0-9.]+)',
                      path.read_text(errors='replace'))
    return match.group(1) if match else None


def _distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _validate_flux_runtime(profile: str, errors: list[str]) -> None:
    """Validate one of the repository's two supported CUDA/Torch pairs."""
    torch_version = _version_tuple(torch.__version__)
    cuda_version = (
        _version_tuple(torch.version.cuda)
        if torch.version.cuda is not None else ())
    if profile == 'flux5090':
        if torch_version[:2] != (2, 8):
            errors.append(
                'Flux RTX 5090 profile requires the repository cu128 '
                'stack (Torch 2.8.x).')
        if cuda_version < (12, 8):
            errors.append(
                'Flux RTX 5090 profile requires a CUDA >=12.8 runtime.')
        return

    supported = (
        torch_version[:2] == (2, 6) and cuda_version[:2] == (12, 4),
        torch_version[:2] == (2, 8) and cuda_version[:2] == (12, 8),
    )
    if not any(supported):
        errors.append(
            'Flux profile requires a repository-supported runtime pair: '
            'Torch 2.6.x + CUDA 12.4 or Torch 2.8.x + CUDA 12.8.')


def _validate_oracle_runtime(errors: list[str]) -> None:
    """Validate the documented NVIDIA cu128 oracle runtime."""
    if not torch.__version__.startswith('2.10.'):
        errors.append('Oracle profile requires the official Torch 2.10.x stack.')
    if (torch.version.cuda is None
            or _version_tuple(torch.version.cuda)[:2] != (12, 8)):
        errors.append('Oracle cu128 profile requires a CUDA 12.8 runtime.')


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _checkpoint_manifest(paths: list[str],
                         errors: list[str]) -> list[dict[str, Any]]:
    manifest = []
    if not paths:
        errors.append(
            'At least one --hash-path is required for the checkpoint gate.')
        return manifest
    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve()
        if path.is_file():
            files = [path]
            root = path.parent
        elif path.is_dir():
            files = sorted(item for item in path.rglob('*') if item.is_file())
            root = path
        else:
            manifest.append({'path': str(path), 'error': 'not found'})
            errors.append(f'Checkpoint hash path does not exist: {path}')
            continue
        if not files:
            errors.append(f'Checkpoint hash path contains no files: {path}')
        manifest.append({
            'path': str(path),
            'files': [{
                'path': str(item.relative_to(root)),
                'bytes': item.stat().st_size,
                'sha256': _sha256(item),
            } for item in files],
        })
    return manifest


def _bind_local_cuda_device(errors: list[str], device_count: int) -> int | None:
    """Bind each torchrun worker to its declared local CUDA device."""
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    local_rank_text = os.environ.get('LOCAL_RANK')
    if world_size > 1 and local_rank_text is None:
        errors.append('WORLD_SIZE > 1 requires LOCAL_RANK from torchrun.')
        return None
    try:
        local_rank = int(local_rank_text or '0')
    except ValueError:
        errors.append(f'Invalid LOCAL_RANK={local_rank_text!r}.')
        return None
    if not 0 <= local_rank < device_count:
        errors.append(
            f'LOCAL_RANK={local_rank} is outside {device_count} CUDA devices.')
        return None
    torch.cuda.set_device(local_rank)
    return local_rank


def _distributed_all_reduce(errors: list[str],
                            local_rank: int | None) -> dict[str, Any]:
    if int(os.environ.get('WORLD_SIZE', '1')) <= 1:
        return {'executed': False}
    if local_rank is None:
        return {'executed': False, 'error': 'no valid local CUDA rank'}
    try:
        dist.init_process_group(backend='nccl')
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = torch.device('cuda', local_rank)
        value = torch.tensor(float(rank + 1), device=device)
        dist.all_reduce(value)
        expected = world_size * (world_size + 1) / 2
        if value.item() != expected:
            errors.append(
                f'NCCL all-reduce returned {value.item()}, expected {expected}.')
        return {
            'executed': True,
            'rank': rank,
            'local_rank': local_rank,
            'device': str(device),
            'world_size': world_size,
            'sum': value.item(),
        }
    except Exception as exc:  # diagnostic must report the original failure
        errors.append(f'NCCL all-reduce failed: {exc!r}')
        return {'executed': True, 'error': repr(exc)}
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def _flash_attn_smoke(errors: list[str], local_rank: int | None) -> bool:
    """Import flash-attn and execute a BF16 forward kernel on this rank."""
    try:
        from flash_attn import flash_attn_func
    except Exception as exc:
        errors.append(f'flash-attn import failed: {exc!r}')
        return False
    if local_rank is None:
        errors.append('flash-attn kernel smoke has no valid CUDA device.')
        return False
    try:
        device = torch.device('cuda', local_rank)
        q = torch.randn(
            1, 8, 2, 16, device=device, dtype=torch.bfloat16)
        output = flash_attn_func(q, q, q, dropout_p=0.0, causal=False)
        torch.cuda.synchronize(device)
        if output.shape != q.shape or not torch.isfinite(output).all():
            errors.append(
                'flash-attn BF16 smoke returned an invalid output.')
            return False
    except Exception as exc:
        errors.append(f'flash-attn BF16 kernel smoke failed: {exc!r}')
        return False
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--profile', choices=('flux', 'flux5090', 'oracle'), default='flux')
    parser.add_argument('--require-gpus', type=int, default=2)
    parser.add_argument('--require-distributed', action='store_true')
    parser.add_argument('--require-flash-attn', action='store_true')
    parser.add_argument('--min-driver-version', default='525.60.13')
    parser.add_argument('--hash-path', action='append', default=[])
    parser.add_argument(
        '--report-dir',
        default=None,
        help='Write one JSON report per torchrun rank into this directory.')
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    errors: list[str] = []
    process_rank = int(os.environ.get('RANK', '0'))
    driver = _driver_version()
    transformers_version = _distribution_version('transformers')
    flash_attn_version = _distribution_version('flash-attn')

    if driver is None:
        errors.append('NVIDIA kernel driver was not detected.')
    elif _version_tuple(driver) < _version_tuple(args.min_driver_version):
        errors.append(
            f'NVIDIA driver {driver} is below required '
            f'{args.min_driver_version}.')

    cuda_available = torch.cuda.is_available()
    device_count = torch.cuda.device_count() if cuda_available else 0
    if not cuda_available:
        errors.append('torch.cuda.is_available() is false.')
    if device_count < args.require_gpus:
        errors.append(
            f'Found {device_count} CUDA devices; {args.require_gpus} required.')

    devices: list[dict[str, Any]] = []
    if cuda_available:
        for index in range(device_count):
            properties = torch.cuda.get_device_properties(index)
            with torch.cuda.device(index):
                bf16_supported = bool(torch.cuda.is_bf16_supported())
            device = {
                'index': index,
                'name': properties.name,
                'total_memory_bytes': properties.total_memory,
                'capability': list(torch.cuda.get_device_capability(index)),
                'bf16_supported': bf16_supported,
            }
            devices.append(device)
            if not bf16_supported:
                errors.append(f'CUDA device {index} does not report BF16 support.')
            try:
                left = torch.ones((64, 64), dtype=torch.bfloat16,
                                  device=index)
                result = left @ left
                if not torch.isfinite(result).all():
                    errors.append(
                        f'CUDA device {index} produced non-finite BF16 output.')
            except Exception as exc:
                errors.append(f'CUDA device {index} BF16 matmul failed: {exc!r}')

    local_rank = (
        _bind_local_cuda_device(errors, device_count)
        if cuda_available else None)

    if args.profile == 'oracle':
        _validate_oracle_runtime(errors)
        if (transformers_version is None
                or _version_tuple(transformers_version) < (4, 57, 1)
                or _version_tuple(transformers_version) >= (5, 0, 0)):
            errors.append(
                'Oracle profile requires transformers>=4.57.1,<5.0.0.')
    else:
        if transformers_version != '5.3.0':
            errors.append(
                'Flux profile must match requirements-base.txt '
                '(transformers==5.3.0).')
        _validate_flux_runtime(args.profile, errors)
    if args.profile == 'flux5090':
        if not any(device['capability'][0] >= 12 for device in devices):
            errors.append(
                'Flux RTX 5090 profile requires a Blackwell CUDA device '
                '(compute capability >= 12.0).')

    flash_attn_kernel_ok = None
    if args.require_flash_attn:
        if flash_attn_version is None:
            errors.append('flash-attn is required but is not installed.')
            flash_attn_kernel_ok = False
        else:
            flash_attn_kernel_ok = _flash_attn_smoke(errors, local_rank)

    distributed = _distributed_all_reduce(errors, local_rank)
    if args.require_distributed and not distributed.get('executed'):
        errors.append(
            'Distributed gate was requested; launch with torchrun and '
            'WORLD_SIZE > 1.')

    if process_rank == 0:
        checkpoint_manifest = _checkpoint_manifest(args.hash_path, errors)
    else:
        checkpoint_manifest = [{
            'skipped': 'checkpoint hashing is performed by rank 0 only'
        }]
    report = {
        'ok': not errors,
        'profile': args.profile,
        'process_rank': process_rank,
        'python': sys.version.split()[0],
        'torch': torch.__version__,
        'torch_cuda_runtime': torch.version.cuda,
        'transformers': transformers_version,
        'flash_attn': flash_attn_version,
        'flash_attn_kernel_ok': flash_attn_kernel_ok,
        'driver': driver,
        'local_rank': local_rank,
        'devices': devices,
        'distributed': distributed,
        'checkpoint_manifest': checkpoint_manifest,
        'errors': errors,
    }
    report_text = json.dumps(report, indent=2, sort_keys=True)
    if args.report_dir is not None:
        report_dir = Path(args.report_dir).expanduser()
        report_dir.mkdir(parents=True, exist_ok=True)
        (report_dir / f'rank-{process_rank}.json').write_text(
            report_text + '\n', encoding='utf-8')
    if process_rank == 0:
        print(report_text)
    return 0 if report['ok'] else 1


if __name__ == '__main__':
    raise SystemExit(main())

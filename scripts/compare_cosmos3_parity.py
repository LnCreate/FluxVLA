#!/usr/bin/env python3
"""Compare official-oracle and FluxVLA Cosmos3 velocity NPZ bundles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def compare_bundles(reference: Path,
                    candidate: Path,
                    *,
                    rtol: float = 1e-2,
                    atol: float = 1e-2) -> dict:
    with np.load(reference, allow_pickle=False) as ref_file, np.load(
            candidate, allow_pickle=False) as cand_file:
        ref = {key: ref_file[key] for key in ref_file.files}
        cand = {key: cand_file[key] for key in cand_file.files}
    ref_velocity = {key for key in ref if key.startswith('velocity_')}
    cand_velocity = {key for key in cand if key.startswith('velocity_')}
    if not ref_velocity or ref_velocity != cand_velocity:
        raise ValueError(
            'Parity bundles must contain the same non-empty velocity_* keys; '
            f'reference={sorted(ref_velocity)}, '
            f'candidate={sorted(cand_velocity)}.')

    ref_inputs = set(ref) - ref_velocity
    cand_inputs = set(cand) - cand_velocity
    if not ref_inputs or ref_inputs != cand_inputs:
        raise ValueError(
            'Parity bundles must contain the same non-empty input key set; '
            f'reference={sorted(ref_inputs)}, candidate={sorted(cand_inputs)}.')

    required_inputs = {
        'text_token_ids',
        'timestep',
        'embodiment_id',
        'raw_action_dim',
        'sequence_has_text',
        'sequence_has_vision',
        'sequence_has_action',
        'sequence_condition_vision',
        'sequence_condition_action',
        'sequence_action_start_offset',
    }
    missing_required = required_inputs - ref_inputs
    if missing_required:
        raise ValueError(
            'Parity bundles are missing required transformer inputs: '
            f'{sorted(missing_required)}.')
    if not ({'input_vision', 'input_action'} & ref_inputs):
        raise ValueError(
            'Parity bundles must contain input_vision and/or input_action.')

    input_keys = ref_inputs
    input_mismatches = []
    for key in sorted(input_keys):
        if ref[key].shape != cand[key].shape or not np.array_equal(
                ref[key], cand[key]):
            input_mismatches.append(key)

    comparisons = {}
    for key in sorted(ref_velocity):
        ref_value = np.asarray(ref[key], dtype=np.float64)
        cand_value = np.asarray(cand[key], dtype=np.float64)
        if ref_value.shape != cand_value.shape:
            comparisons[key] = {
                'ok': False,
                'reference_shape': list(ref_value.shape),
                'candidate_shape': list(cand_value.shape),
            }
            continue
        finite = np.all(np.isfinite(ref_value)) and np.all(
            np.isfinite(cand_value))
        difference = np.abs(cand_value - ref_value)
        relative = difference / np.maximum(np.abs(ref_value), atol)
        comparisons[key] = {
            'ok': bool(finite and np.allclose(
                cand_value, ref_value, rtol=rtol, atol=atol)),
            'shape': list(ref_value.shape),
            'max_abs_error': float(difference.max(initial=0.0)),
            'max_relative_error': float(relative.max(initial=0.0)),
        }
    checks = {
        'shared_inputs_exact': not input_mismatches,
        'velocities_close': all(value['ok']
                                for value in comparisons.values()),
    }
    return {
        'ok': all(checks.values()),
        'reference': str(reference),
        'candidate': str(candidate),
        'rtol': rtol,
        'atol': atol,
        'checks': checks,
        'input_mismatches': input_mismatches,
        'velocities': comparisons,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--rtol', type=float, default=1e-2)
    parser.add_argument('--atol', type=float, default=1e-2)
    parser.add_argument('--output-json', type=Path)
    args = parser.parse_args()
    report = compare_bundles(
        args.reference, args.candidate, rtol=args.rtol, atol=args.atol)
    output = json.dumps(report, indent=2, sort_keys=True)
    print(output)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(output + '\n', encoding='utf-8')
    return 0 if report['ok'] else 1


if __name__ == '__main__':
    raise SystemExit(main())

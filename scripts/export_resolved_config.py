#!/usr/bin/env python3
"""Materialize an MMEngine experiment config as a relocatable JSON file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mmengine import Config


def export_config(source: Path, output: Path) -> dict:
    config = Config.fromfile(str(source)).to_dict()
    # Normalize tuples/ConfigDict values through JSON before comparing.
    payload = json.loads(json.dumps(config))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + '\n',
        encoding='utf-8')
    # Prove the output is consumable by the same loader used by serving.
    reloaded = Config.fromfile(str(output)).to_dict()
    if reloaded != payload:
        raise RuntimeError('Resolved config changed during JSON round-trip.')
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    export_config(args.config.resolve(), args.output.resolve())
    print(args.output.resolve())
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

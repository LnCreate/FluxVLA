#!/usr/bin/env python3
"""Mirror FluxVLA JSONL training metrics into TensorBoard event files."""

import argparse
import json
import math
import time
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True,
                        help='FluxVLA per-step JSONL metrics file.')
    parser.add_argument('--log-dir', type=Path, required=True,
                        help='TensorBoard event output directory.')
    parser.add_argument('--follow', action='store_true',
                        help='Keep following appended JSONL records.')
    parser.add_argument('--poll-interval', type=float, default=1.0)
    return parser.parse_args()


def write_record(writer, record):
    step = record.get('VLA Train/Step', record.get('step'))
    if not isinstance(step, (int, float)):
        return False
    for key, value in record.items():
        if key in {'VLA Train/Step', 'step'}:
            continue
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            scalar = float(value)
        elif isinstance(value, str):
            try:
                scalar = float(value)
            except ValueError:
                continue
        else:
            continue
        if math.isfinite(scalar):
            writer.add_scalar(key, scalar, int(step))
    return True


def main():
    args = parse_args()
    args.log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(args.log_dir))
    written = 0
    with args.input.open(encoding='utf-8') as stream:
        while True:
            offset = stream.tell()
            line = stream.readline()
            if not line:
                if not args.follow:
                    break
                writer.flush()
                time.sleep(args.poll_interval)
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                if args.follow:
                    stream.seek(offset)
                    time.sleep(args.poll_interval)
                    continue
                raise
            written += int(write_record(writer, record))
            if written % 100 == 0:
                writer.flush()
    writer.flush()
    writer.close()


if __name__ == '__main__':
    main()

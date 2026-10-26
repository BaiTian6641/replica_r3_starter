#!/usr/bin/env python3
"""Prepare exact prepared02 copies and update-dynamics pins, without a model.

The command copies the already-admitted prepared02 episode and verifier-label
bytes, regenerates the deterministic variant inventory, and pins the sealed
source, frozen rejected writer extension, configuration and implementation.
It loads no tokenizer, no model weight and performs no native computation.

``--admit`` re-verifies every pin, regenerates the variant inventory and writes
an admission receipt; a rejection is preserved rather than hidden.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from replica_lab.memory_update_dynamics import (  # noqa: E402
    ADMISSION_FORMAT,
    CONFIG_FORMAT,
    admit_update_dynamics_prepared,
    prepare_update_dynamics,
)


def _fresh(path):
    path = Path(path)
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise SystemExit(f'Use a fresh preparation output path: {path}')
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path,
                        default=ROOT / 'configs/memory_update_dynamics_v1.json')
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'data/memory_update_dynamics_20261025/prepared01')
    parser.add_argument('--source-prepared', type=Path,
                        help='override the configured prepared02 directory')
    parser.add_argument('--admit', action='store_true',
                        help='verify all pins and write admission.json')
    args = parser.parse_args(argv)
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    if config.get('format') != CONFIG_FORMAT:
        parser.error(f'configuration format must be {CONFIG_FORMAT}')
    output = _fresh(args.output)
    manifest = prepare_update_dynamics(
        output, config_path=config_path, config=config,
        source_prepared=args.source_prepared)
    receipt = None
    if args.admit:
        receipt = admit_update_dynamics_prepared(output)
        if receipt.get('format') != ADMISSION_FORMAT or receipt.get('status') != 'admitted':
            raise SystemExit('admission did not return an admitted receipt')
    print(json.dumps({'manifest': manifest, 'admission': receipt},
                     indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

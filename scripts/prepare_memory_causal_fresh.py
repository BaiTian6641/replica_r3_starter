#!/usr/bin/env python3
"""Prepare the fresh disjoint causal-selector fixture and every pin.

This command performs no model-weight load, no neural forward and no optimizer
step.  It writes deterministic episodes, evaluation-only labels, live
revocation controls, an inventory and a self-contained protocol snapshot.
``--verify`` re-derives every artifact and pin.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
    os.environ.setdefault(key, "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from replica_lab.memory_causal_fresh import (  # noqa: E402
    CONFIG_FORMAT, prepare_fresh, verify_fresh_prepared,
)


def _fresh(path):
    path = Path(path)
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise SystemExit(f"Use a fresh preparation output path: {path}")
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=ROOT / "configs/memory_causal_fresh_v1.json")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "data/memory_causal_fresh_20261026/prepared01")
    parser.add_argument("--verify", action="store_true",
                        help="re-derive and verify the prepared fixture instead of writing")
    args = parser.parse_args(argv)
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    if config.get("format") != CONFIG_FORMAT:
        parser.error(f"configuration format must be {CONFIG_FORMAT}")
    if args.verify:
        receipt = verify_fresh_prepared(args.output.resolve())
        print(json.dumps({"verification": receipt}, indent=2, sort_keys=True))
        return 0
    manifest = prepare_fresh(_fresh(args.output), config_path=config_path,
                             config=config)
    print(json.dumps({"manifest": manifest}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

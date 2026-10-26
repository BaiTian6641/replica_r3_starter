#!/usr/bin/env python3
"""Prepare deterministic delayed visual-memory episodes without a model.

The command renders owned PNG scenes, writes label-free episodes, queries and
checked source annotations, keeps evaluator expectations in a separate labels
file, and pins the configuration, protocol and implementation. It loads no
model, imports no torch and performs no native computation.

``--admit`` re-verifies every pin and writes an admission receipt. A rejection
is preserved in ``admission.json`` and the command exits nonzero.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from replica_lab.visual_memory_episodes import (  # noqa: E402
    ADMISSION_FORMAT,
    CONFIG_FORMAT,
    admit_visual_memory_prepared,
    prepare_visual_memory,
    validate_config,
)


def _fresh(path: Path) -> Path:
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise SystemExit(f"Use a fresh preparation output path: {path}")
    return path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/visual_memory_v1.json")
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "data/visual_memory_20261025/prepared01")
    parser.add_argument("--admit", action="store_true",
                        help="verify all pins and write admission.json")
    args = parser.parse_args(argv)
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("format") != CONFIG_FORMAT:
        parser.error(f"configuration format must be {CONFIG_FORMAT}")
    validate_config(config)
    output = _fresh(args.output.resolve())
    manifest = prepare_visual_memory(
        output, config_path=config_path, config=config)
    receipt = None
    if args.admit:
        receipt = admit_visual_memory_prepared(output)
        if (receipt.get("format") != ADMISSION_FORMAT
                or receipt.get("status") != "admitted"):
            print(json.dumps({"manifest": manifest, "admission": receipt},
                             indent=2, sort_keys=True))
            return 2
    print(json.dumps({"manifest": manifest, "admission": receipt},
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

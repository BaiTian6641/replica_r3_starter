#!/usr/bin/env python3
"""Rescore immutable memory rankings through memory.acceptance.v2."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from replica_lab.memory_acceptance import RECEIPT_FORMAT, rescore_ranking_file  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rankings", type=Path, required=True)
    parser.add_argument("--config", type=Path,
                        default=ROOT / "configs/memory_acceptance_v2.json")
    parser.add_argument("--generation", type=Path,
                        help="optional historical generation JSONL, reference only")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Use a fresh output receipt")
    config = json.loads(args.config.read_text())
    receipt = rescore_ranking_file(args.rankings, config,
                                   generation_path=args.generation)
    receipt["config_sha256"] = hashlib.sha256(args.config.read_bytes()).hexdigest()
    if receipt["format"] != RECEIPT_FORMAT:
        raise RuntimeError("Unexpected receipt format")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"decision": receipt["verdict"]["decision"],
                      "accepted_correct": receipt["metrics"][receipt["candidate_variant"]]["query"]["accepted_correct"],
                      "accepted_precision": receipt["metrics"][receipt["candidate_variant"]]["query"]["accepted_precision"],
                      "acceptance_coverage": receipt["metrics"][receipt["candidate_variant"]]["query"]["acceptance_coverage"],
                      "output": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

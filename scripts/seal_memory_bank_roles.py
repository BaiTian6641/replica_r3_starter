#!/usr/bin/env python3
"""Project the pinned prepared labels into role-specific sealed files.

The prepared04 fixture has one ``labels.json``.  This preparation-time script
writes disjoint ``labels_dev.json``, ``labels_heldout.json`` and
``labels_train.json`` files plus a manifest with per-file SHA256 digests.  The
readout runner opens the dev role before the development gates and the heldout
role only after every development gate passes.

No model, tokenizer, native computation or evaluation decision is used.  The
output directory must be fresh.

    .venv/bin/python scripts/seal_memory_bank_roles.py \\
      --prepared data/memory_causal_fresh_20261026/prepared04 \\
      --output data/memory_causal_fresh_20261026/prepared04_roles

Pin the printed per-role hashes in ``configs/memory_bank_readout_v2.json``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main(argv=None):
    from replica_lab.memory_bank_readout_v2 import seal_role_labels

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prepared", type=Path,
        default=ROOT / "data/memory_causal_fresh_20261026/prepared04")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    prepared = args.prepared.resolve()
    output = args.output.resolve()
    if prepared == output:
        parser.error("Use a fresh output directory")
    manifest = seal_role_labels(prepared, output)
    summary = {
        "output": str(output),
        "source_labels_sha256": manifest["source_labels_sha256"],
        "role_sets_disjoint": manifest["role_sets_disjoint"],
        "roles": {
            role: {
                "filename": entry["filename"],
                "sha256": entry["sha256"],
                "episodes": entry["episodes"],
                "states": entry["states"],
            }
            for role, entry in manifest["roles"].items()
        },
        "manifest_sha256": manifest["manifest_sha256"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return manifest


if __name__ == "__main__":
    main()

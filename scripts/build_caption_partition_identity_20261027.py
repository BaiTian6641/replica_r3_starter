#!/usr/bin/env python3
"""Write the deterministic caption partition identity sidecar.

The sidecar is additive: it never rewrites the lineage manifest, rows,
exclusions, configs, receipt or evaluation seal.  It records identities for the
nine admitted rows, all twenty-one exclusions and the eight quarantined exposed
rows, with parent/exposure pins, provenance, dedup keys and admission roles.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))

import validate_caption_lineage_and_eval_20261027 as validator  # noqa: E402


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lineage-config", default=str(
        ROOT / "configs/caption_lineage_correction_20261027.json"))
    parser.add_argument("--eval-config", default=str(
        ROOT / "configs/synthetic_scene_eval_20261027.json"))
    parser.add_argument("--output", default=str(
        ROOT / validator.PARTITION_IDENTITY_PATH))
    args = parser.parse_args(argv)
    target = Path(args.output)
    identity = validator.build_partition_identity(
        args.lineage_config, args.eval_config, project_root=ROOT)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(identity, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    print(json.dumps({
        "status": "written",
        "path": str(target.relative_to(ROOT)),
        "counts": identity["counts"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

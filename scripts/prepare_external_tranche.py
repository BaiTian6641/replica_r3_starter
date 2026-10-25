#!/usr/bin/env python3
"""Prepare or verify one bounded external caption/grounding tranche.

Acquisition is limited to a pinned dataset revision, a bounded row window, the
ZIP central directory and the selected image members. Nothing is admitted unless
the card digest, member CRCs, image inspection, split grouping, duplicate checks
and label/input separation all pass.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from replica_lab.external_intake import (  # noqa: E402
    ExternalIntakeError,
    build_sources,
    load_config,
    prepare_tranche,
    verify_tranche,
)


def _summary(manifest: dict) -> dict:
    return {
        "status": manifest["status"],
        "intake_id": manifest["intake_id"],
        "source": f"{manifest['dataset']['id']}@{manifest['dataset']['revision']}",
        "counts": manifest["counts"],
        "exclusions": manifest["exclusions"],
        "acquired_bytes": manifest["acquisition"]["acquired_bytes"],
        "budget_bytes": manifest["acquisition"]["budget_bytes"],
        "leakage": manifest["leakage"],
        "artifacts": manifest["artifacts"],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=ROOT / "configs/external_caption_tranche_v1.json")
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--data-root", type=Path, default=None,
                        help="override the configured data root (review/fixtures)")
    parser.add_argument("--report", type=Path, default=None,
                        help="override the configured report path")
    parser.add_argument("--verify-only", action="store_true",
                        help="re-verify an existing tranche without acquiring anything")
    args = parser.parse_args(argv)
    try:
        config, config_sha256 = load_config(args.config)
        data_root = args.data_root or args.project_root / config["outputs"]["data_root"]
        if args.verify_only:
            result = verify_tranche(data_root, config=config,
                                    config_sha256=config_sha256)
            print(json.dumps({"status": result["status"],
                              "rows": result["rows"],
                              "roles": result["roles"],
                              "leakage": result["leakage"]}, indent=2))
            return 0
        manifest = prepare_tranche(
            config, project_root=args.project_root, config_sha256=config_sha256,
            config_path=args.config, sources=build_sources(config),
            data_root=args.data_root, report_path=args.report)
        print(json.dumps(_summary(manifest), indent=2))
        return 0
    except ExternalIntakeError as error:
        print(f"external intake refused: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

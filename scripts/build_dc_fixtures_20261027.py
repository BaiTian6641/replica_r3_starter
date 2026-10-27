#!/usr/bin/env python3
"""Build fresh deterministic DC-P1/DC-R1 role-sealed fixtures.

This command only writes text/JSON fixtures.  It creates a fresh directory,
refuses to overwrite existing evidence and performs no model, tokenizer,
optimizer, gradient, checkpoint or native work.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from dc_p1_native_response_20261027 import (  # noqa: E402
    CONFIG_PATH as P1_CONFIG, PREPARED_DIR as P1_PREPARED,
    build_fixture as build_p1,
)
from dc_r1_native_value_20261027 import (  # noqa: E402
    CONFIG_PATH as R1_CONFIG, PREPARED_DIR as R1_PREPARED,
    build_fixture as build_r1,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", choices=("p1", "r1", "all"),
                        default="all")
    parser.add_argument("--p1-output", type=Path, default=P1_PREPARED)
    parser.add_argument("--r1-output", type=Path, default=R1_PREPARED)
    args = parser.parse_args(argv)
    results = {}
    try:
        if args.protocol in ("p1", "all"):
            manifest = build_p1(P1_CONFIG, args.p1_output)
            results["p1"] = {
                "manifest_sha256": manifest["manifest_sha256"],
                "counts": manifest["counts"],
                "path": str(args.p1_output),
            }
        if args.protocol in ("r1", "all"):
            manifest = build_r1(R1_CONFIG, args.r1_output)
            results["r1"] = {
                "manifest_sha256": manifest["manifest_sha256"],
                "counts": manifest["counts"],
                "path": str(args.r1_output),
            }
    except Exception as error:  # noqa: BLE001 - CLI refuses with a stable code
        print(json.dumps({"status": "refused", "error": str(error)},
                         indent=2, sort_keys=True))
        return 2
    print(json.dumps({"status": "built", "results": results},
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

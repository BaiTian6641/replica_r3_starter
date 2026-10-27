#!/usr/bin/env python3
"""Prepare the deterministic value-readout abstention fixture.

This command performs no model load, tokenizer load, neural forward, optimizer
step, checkpoint write, download or native job.  It writes chronological
episodes, disjoint training/replay cases, sealed development and heldout
labels, control-arm rows and a manifest.  A fresh output path is required.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from value_readout_abstention_contract_v1 import (  # noqa: E402
    CONFIG_PATH, PREPARED_DIR, ProtocolError, build_fixture, read_json,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--output", type=Path, default=PREPARED_DIR)
    args = parser.parse_args(argv)
    try:
        config = read_json(args.config)
        manifest = build_fixture(config, args.config.resolve(), args.output)
    except ProtocolError as error:
        print(json.dumps({"status": "refused", "check": error.check,
                          "detail": error.detail}, indent=2, sort_keys=True))
        return 2
    print(json.dumps({
        "status": "prepared",
        "protocol_id": manifest["protocol_id"],
        "prepared_manifest_sha256": manifest["manifest_sha256"],
        "counts": manifest["counts"],
        "native_computation": manifest["native_computation"],
        "model_loaded": manifest["model_loaded"],
        "downloads": manifest["downloads"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

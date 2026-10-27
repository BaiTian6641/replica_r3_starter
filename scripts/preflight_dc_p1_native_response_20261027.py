#!/usr/bin/env python3
"""Review-only preflight for DC-P1; launch fails closed.

``--review`` validates the config, resource guards, tokenizer binding, manifest
and file hashes, then writes a fresh deterministic receipt.  It parses no
sealed evaluation label and executes no model, tokenizer or native work.
``--launch`` requires a separate authorization receipt binding the exact config
and fixture bytes; even then it refuses, because this package intentionally
contains no native runner.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from dc_p1_native_response_20261027 import (  # noqa: E402
    CONFIG_PATH, PREPARED_DIR, preflight,
)
from dc_protocol_common_20261027 import (  # noqa: E402
    ProtocolError, require_fresh_path, write_json_atomic,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--prepared", type=Path, default=PREPARED_DIR)
    parser.add_argument("--receipt", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--review", action="store_true")
    mode.add_argument("--launch", action="store_true")
    parser.add_argument("--authorization", type=Path, default=None)
    args = parser.parse_args(argv)
    try:
        report = preflight(
            args.config, args.prepared, launch=bool(args.launch),
            authorization=args.authorization)
        target = require_fresh_path(args.receipt, "preflight_receipt")
        write_json_atomic(target, report)
    except ProtocolError as error:
        print(json.dumps({"status": "refused", **error.as_dict()},
                         indent=2, sort_keys=True))
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return int(report.get("exit_code", 2))


if __name__ == "__main__":
    raise SystemExit(main())

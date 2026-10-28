#!/usr/bin/env python3
"""Review-ready DC-P1 v2 preflight; launch requires an external receipt.

``--review`` validates the protocol config, the v1 lineage preservation, the
resource guards, the tokenizer binding, the prepared manifest and every file
hash, the derived physical-work schedule, the token-length artifact state and the
declared runner path, then writes a fresh deterministic receipt.  It reads no
sealed evaluation label and executes no model, tokenizer or native work.

``--launch`` additionally requires an externally issued authorization receipt of
format ``replica.dc_protocol.authorization.v2`` that binds the exact config
bytes, the exact prepared manifest, the exact runner bytes and a full protocol
commit, and verifies those committed bytes offline with ``git show``.  Even then
this command only reports ``launch_admitted`` and prints the runner command: it
never executes the native runner and never creates an authorization.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from dc_protocol_common_20261027 import (  # noqa: E402
    ProtocolError, require_fresh_path, write_json_atomic,
)
import dc_p1_native_response_v2_20261027 as protocol  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=protocol.CONFIG_PATH)
    parser.add_argument("--prepared", type=Path, default=protocol.PREPARED_DIR)
    parser.add_argument("--receipt", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--review", action="store_true")
    mode.add_argument("--launch", action="store_true")
    parser.add_argument("--authorization", type=Path, default=None)
    args = parser.parse_args(argv)
    try:
        report = protocol.preflight(
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

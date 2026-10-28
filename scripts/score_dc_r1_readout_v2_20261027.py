#!/usr/bin/env python3
"""Score already-generated DC-R1 v2 per-layer readout outputs.

Two explicit split entry points share one scorer:

* ``--split dev`` uses development cases and the development labels.
* ``--split sealed_final`` requires ``--dev-report`` and admits that report
  before any final label is opened.

Every layer's generated ids are verified against the sealed MiniCPM5 tokenizer
and decode policy, and abstention/non-abstention/hallucination outcomes are
recomputed from the re-decoded text.  Host abstention flags are rejected.  The
host compressed reconstruction is reported only as an audit/control; native
compressed gates remain blocked without a native compressed-channel receipt.

The command loads a tokenizer but no model and performs no generation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from dc_r1_native_value_v2_20261027 import (  # noqa: E402
    CONFIG_PATH, PREPARED_DIR, score_r1_v2,
)
from dc_protocol_common_20261027 import (  # noqa: E402
    ProtocolError, read_json, require_fresh_path, sha256_file,
    validate_tokenizer_binding, validate_tokenizer_directory,
    write_json_atomic,
)


def _decode_fn(tokenizer_dir: Path, binding: dict):
    validate_tokenizer_directory(binding, tokenizer_dir)
    from transformers import AutoTokenizer  # local import; no model load

    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_dir), local_files_only=True, trust_remote_code=False,
        revision=binding["revision"])
    policy = binding["decode_policy"]

    def decode(ids):
        return tokenizer.decode(
            list(ids), skip_special_tokens=policy["skip_special_tokens"],
            clean_up_tokenization_spaces=policy[
                "clean_up_tokenization_spaces"])

    return decode


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--split", choices=("dev", "sealed_final"),
                        required=True)
    parser.add_argument("--dev-report", type=Path, default=None,
                        help="required for --split sealed_final")
    parser.add_argument("--dev-predictions", type=Path, default=None,
                        help="original development predictions required for final admission")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--prepared", type=Path, default=PREPARED_DIR)
    parser.add_argument("--tokenizer-dir", type=Path,
                        default=ROOT / "checkpoints" / "minicpm5_source")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        config = read_json(args.config)
        if args.split == "sealed_final":
            from dc_r1_native_value_v2_20261027 import (
                check_dev_report_boundary, _manifest_hash)
            check_dev_report_boundary(
                args.dev_report, config_path=args.config,
                prepared_manifest_sha256=_manifest_hash(
                    read_json(args.config), args.prepared))
        binding = validate_tokenizer_binding(config["tokenizer_binding"])
        decode = _decode_fn(args.tokenizer_dir, binding)
        report = score_r1_v2(
            args.predictions, split=args.split, prepared=args.prepared,
            config_path=args.config, decode_fn=decode,
            dev_report_path=args.dev_report,
            dev_predictions_path=args.dev_predictions)
        target = require_fresh_path(args.report, "score_report")
        write_json_atomic(target, report)
        report["report"] = str(target)
        report["report_sha256"] = sha256_file(target)
    except (ProtocolError, OSError, ValueError) as error:
        print(json.dumps({"status": "refused", "error": str(error)},
                         indent=2, sort_keys=True))
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.split == "dev":
        return 0 if report["development_evidence_admissible"] else 4
    return 0 if report["readout_useful"] else 4


if __name__ == "__main__":
    raise SystemExit(main())

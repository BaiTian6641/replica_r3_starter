#!/usr/bin/env python3
"""Score already-generated DC-R1 per-layer readout outputs.

The command verifies every layer's generated ids against the sealed MiniCPM5
tokenizer and decode policy, binds each prompt to its sealed condition, rejects
answer-index metadata, and only then computes accepted-layer, complete-group,
wrong-value, no-memory and exact/compressed reconstruction metrics.  It loads a
tokenizer but no model and performs no generation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from dc_r1_native_value_20261027 import (  # noqa: E402
    CONDITIONS, CONFIG_PATH, FILES, MANIFEST_FORMAT, PREPARED_DIR,
    PROTOCOL_ID, score_r1, validate_config,
)
from dc_protocol_common_20261027 import (  # noqa: E402
    ProtocolError, read_json, require_fresh_path, sha256_file,
    validate_manifest, validate_tokenizer_binding,
    validate_tokenizer_directory, write_json_atomic,
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
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--prepared", type=Path, default=PREPARED_DIR)
    parser.add_argument("--tokenizer-dir", type=Path,
                        default=ROOT / "checkpoints" / "minicpm5_source")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        config = read_json(args.config)
        validate_config(config)
        validate_manifest(
            args.prepared, config, expected_files=FILES,
            manifest_format=MANIFEST_FORMAT, protocol_id=PROTOCOL_ID)
        binding = validate_tokenizer_binding(config["tokenizer_binding"])
        decode = _decode_fn(args.tokenizer_dir, binding)
        conditions = {
            condition: args.prepared / f"dev_conditions_{condition}.jsonl"
            for condition in CONDITIONS
        }
        report = score_r1(
            args.predictions,
            dev_cases_path=args.prepared / "dev_cases.jsonl",
            dev_labels_path=args.prepared / "sealed_labels_dev.jsonl",
            dev_conditions=conditions, config_path=args.config,
            decode_fn=decode)
        target = require_fresh_path(args.report, "score_report")
        write_json_atomic(target, report)
        report["report"] = str(target)
        report["report_sha256"] = sha256_file(target)
    except (ProtocolError, OSError, ValueError) as error:
        print(json.dumps({"status": "refused", "error": str(error)},
                         indent=2, sort_keys=True))
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "readout_useful" else 4


if __name__ == "__main__":
    raise SystemExit(main())

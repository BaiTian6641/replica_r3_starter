#!/usr/bin/env python3
"""Score already-generated DC-P1 answers per case.

The command verifies the posted generated ids against the sealed MiniCPM5
tokenizer and decode policy, rejects source-retained/KV-retained rows, rejects
answer-index metadata, binds prompts to sealed cases, and only then compares the
normalized generated answer with the sealed label.  It loads a tokenizer but no
model and performs no generation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from dc_p1_native_response_20261027 import (  # noqa: E402
    CONFIG_PATH, FILES, MANIFEST_FORMAT, PREPARED_DIR, PROTOCOL_ID,
    score_p1, validate_config,
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
        report = score_p1(
            args.predictions,
            fit_cases_path=args.prepared / "fit_cases.jsonl",
            fit_labels_path=args.prepared / "sealed_labels_fit.jsonl",
            dev_cases_path=args.prepared / "dev_cases.jsonl",
            dev_labels_path=args.prepared / "sealed_labels_dev.jsonl",
            config_path=args.config, decode_fn=decode)
        target = require_fresh_path(args.report, "score_report")
        write_json_atomic(target, report)
        report["report"] = str(target)
        report["report_sha256"] = sha256_file(target)
    except (ProtocolError, OSError, ValueError) as error:
        print(json.dumps({"status": "refused", "error": str(error)},
                         indent=2, sort_keys=True))
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "passed" else 4


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Review and launch-preflight for the additive DC-R1 v3 protocol."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from dc_protocol_common_20261027 import (  # noqa: E402
    ProtocolError, read_json, require_fresh_path, sha256_file, write_json_atomic,
)
from replica_lab import dc_r1_v3_scoring as scoring  # noqa: E402
from replica_lab import dc_r1_v3_runner as runner  # noqa: E402

CONFIG_PATH = scoring.CONFIG_PATH
PREPARED_DIR = scoring.PREPARED_DIR


def workload(config: dict) -> dict:
    schedule = config["schedule"]
    max_new = int(schedule["max_generated_tokens_per_request"])
    prompt = int(schedule["prefill_token_ceiling_per_request"])
    splits = {}
    for split, count in (("dev", 12), ("sealed_final", 4)):
        requests = count * len(scoring.CONDITIONS) * len(scoring.LAYERS)
        splits[split] = {
            "cases": count,
            "generation_requests": requests,
            "prefill_forwards": requests,
            "decode_forwards": requests * (max_new - 1),
            "forward_calls": requests * max_new,
            "generated_tokens": requests * max_new,
            "prefill_tokens": requests * prompt,
            "processed_positions": requests * (max_new * prompt + max_new * (max_new - 1) // 2),
            "layer_readout_streams": requests,
        }
    total = {name: splits["dev"][name] + splits["sealed_final"][name]
             for name in splits["dev"] if name != "cases"}
    total["cases"] = 16
    total["conditions"] = len(scoring.CONDITIONS)
    total["layers"] = len(scoring.LAYERS)
    return {"splits": splits, "total": total,
            "recovery": {
                "restarts": schedule["recovery_max"],
                "model_loads": schedule["recovery_model_loads"],
                "tokenizer_loads": schedule["recovery_tokenizer_loads"],
                "replay_layer_streams_max": schedule["recovery_replay_layer_streams_max"],
                "forward_calls_max": schedule["recovery_forward_calls_max"],
                "generated_tokens_max": schedule["recovery_generated_tokens_max"],
                "processed_positions_max": schedule["recovery_processed_positions_max"],
            }}


def _command(context: runner.RunContext, stage: str, receipt: Path) -> str:
    report_root = receipt.parent
    continuation = ["--continue-existing-output"] if stage == "sealed_final" else []
    return shlex.join([
        "PYTHONDONTWRITEBYTECODE=1", "python3",
        "scripts/run_capacity_preflight.py", "--entry-point",
        "run_dc_r1_native_value_v3_20261028.py", "--report",
        str(report_root / f"guard_{stage}01.json"), "--",
        "--stage", stage, "--device", "mps",
        "--config", str(context.config_path),
        "--prepared", str(context.prepared),
        "--output", str(context.output),
        "--authorization", str(context.authorization.get("path", "")),
        "--report", str(report_root / f"{stage}_result01.json"),
        *continuation,
    ])


def preflight(config_path: Path = CONFIG_PATH, prepared: Path = PREPARED_DIR,
              *, launch: bool = False, authorization: Path | None = None,
              output: Path | None = None, require_head: bool = True,
              commit_verifier=None) -> dict:
    config = scoring.load_config(config_path)
    manifest = scoring.validate_prepared(config, defer_final_labels=True)
    output = (Path(output) if output is not None
              else ROOT / config["launch"]["output_root"])
    if not output.is_absolute():
        output = ROOT / output
    output = output.resolve()
    report = {
        "format": "replica.dc_r1_native_value.review.v3",
        "protocol_id": scoring.PROTOCOL_ID,
        "status": "review_ready",
        "config_sha256": sha256_file(config_path),
        "prepared_manifest_sha256": manifest["manifest_sha256"],
        "workload": workload(config),
        "physical_limits": config["budget"]["physical_limits"],
        "authorized_run_ceiling": config["budget"]["authorized_run_ceiling"],
        "native_execution": False,
        "model_loaded": False,
        "tokenizer_loaded": False,
        "heldout_opened": False,
        "native_compressed_gate_status": config["compressed_channel"]["native_gate_status"],
        "authorization_receipt_present": authorization is not None,
        "missing_receipt_fields": (
            config["launch"]["authorization_fields"] if authorization is None else []),
        "output_root": str(output),
        "fresh_output_available": not output.exists(),
        "runner_path": config["launch"]["runner_path"],
        "native_command": None,
        "sealed_final_command": None,
    }
    if not launch:
        report["native_execution_status"] = "refused_preparation_only"
        report["exit_code"] = 0
        return report
    if authorization is None:
        raise ProtocolError("authorization", "an external authorization receipt is required")
    context, _ = runner.verify_authorization(
        config_path=config_path, prepared=prepared,
        authorization_path=authorization, output=output,
        require_head=require_head, commit_verifier=commit_verifier, register=False)
    context.authorization["path"] = str(authorization)
    report.update({
        "status": "launch_admitted",
        "native_execution_status": "external authorization admitted; guard executes separately",
        "authorization_receipt_sha256": context.authorization_sha256,
        "protocol_commit": context.protocol_commit,
        "attempt_id": context.attempt_id,
        "native_command": _command(context, "dev", output.parent / "preflight_receipt.json"),
        "sealed_final_command": _command(context, "sealed_final", output.parent / "preflight_receipt.json"),
        "exit_code": 0,
    })
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--prepared", type=Path, default=PREPARED_DIR)
    parser.add_argument("--receipt", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--review", action="store_true")
    mode.add_argument("--launch", action="store_true")
    parser.add_argument("--authorization", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    try:
        report = preflight(
            args.config, args.prepared, launch=bool(args.launch),
            authorization=args.authorization, output=args.output,
            require_head=True)
        target = require_fresh_path(args.receipt, "preflight_receipt")
        write_json_atomic(target, report)
    except (ProtocolError, OSError, ValueError) as error:
        detail = error.as_dict() if hasattr(error, "as_dict") else {
            "check": "preflight", "detail": str(error)}
        print(json.dumps({"status": "refused", **detail}, indent=2, sort_keys=True))
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return int(report.get("exit_code", 2))


if __name__ == "__main__":
    raise SystemExit(main())

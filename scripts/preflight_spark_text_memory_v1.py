#!/usr/bin/env python3
"""Review and launch-preflight for the bounded Spark text-memory protocol."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/spark_text_memory_protocol_v1.json"
PROTOCOL_ID = "spark_text_memory_v1"
EXPECTED_FORMAT = "replica.spark_text_memory.protocol.v1"


class ProtocolError(ValueError):
    def __init__(self, check: str, detail: str) -> None:
        super().__init__(detail)
        self.check = check
        self.detail = detail

    def as_dict(self) -> dict[str, str]:
        return {"check": self.check, "detail": self.detail}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ProtocolError("json", f"cannot read {path}: {error}") from error
    if not isinstance(value, dict):
        raise ProtocolError("json", f"{path} must contain an object")
    return value


def _require(condition: bool, check: str, detail: str) -> None:
    if not condition:
        raise ProtocolError(check, detail)


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else root / path).resolve()


def _current_head(root: Path) -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                            text=True, capture_output=True, check=True)
    return result.stdout.strip()


def _source_validation(config: Mapping[str, Any], root: Path,
                       *, full_weight_hash: bool) -> dict[str, Any]:
    source = config["source_identities"]["official_fp8"]
    source_root = _resolve(root, source["local_path"])
    _require(source_root.is_dir(), "source", f"missing sealed FP8 source {source_root}")
    config_path = source_root / "config.json"
    weight_path = source_root / "model.safetensors"
    _require(config_path.is_file() and weight_path.is_file(),
             "source", "official FP8 config or weight payload is missing")
    _require(sha256_file(config_path) == source["config_sha256"],
             "source", "official FP8 config hash differs")
    result = {
        "source_identity": "official_fp8",
        "source_root": str(source_root),
        "config_sha256": source["config_sha256"],
        "weight_path": str(weight_path),
        "weight_sha256_checked": False,
        "execution_format": source["execution_format"],
        "required_backend": source["required_backend"],
    }
    if full_weight_hash:
        _require(sha256_file(weight_path) == source["weight_sha256"],
                 "source", "official FP8 weight hash differs")
        result["weight_sha256_checked"] = True
    for filename, expected in (
        ("tokenizer.json", config["runtime"]["tokenizer"]["tokenizer_json_sha256"]),
        ("tokenizer_config.json", config["runtime"]["tokenizer"]["tokenizer_config_sha256"]),
        ("chat_template.jinja", config["runtime"]["tokenizer"]["chat_template_sha256"]),
    ):
        path = source_root / filename
        _require(path.is_file(), "tokenizer", f"missing {filename}")
        _require(sha256_file(path) == expected, "tokenizer", f"{filename} hash differs")
    result["tokenizer_checked"] = True
    return result



def derive_work(config: Mapping[str, Any]) -> dict[str, int]:
    schedule = config["work_and_cost"]["native_schedule"]
    runtime = config["runtime"]
    fit = schedule["fit_events"]
    cases = schedule["development_cases_per_arm"]
    arms = len(config["controls"]["arms"])
    requests = (schedule["parity_requests"] + schedule["restart_parity_requests"]
                + arms * cases)
    prompt = runtime["context_tokens_max"]
    new_tokens = runtime["generation"]["max_new_tokens"]
    positions_per_request = prompt * new_tokens + new_tokens * (new_tokens - 1) // 2
    restarts = schedule["restarts"]
    expected = {
        "model_loads_max": 1 + restarts,
        "tokenizer_loads_max": 1 + restarts,
        "training_forwards_max": fit,
        "plastic_forward_calls_max": fit,
        "plastic_backward_calls_max": fit,
        "query_sensing_forwards_max": 2 * fit + requests * new_tokens,
        "native_query_passes_max": 2 * fit + requests * new_tokens,
        "backward_calls_max": 2 * fit,
        "backward_processed_positions_max": fit * prompt,
        "plastic_feature_vectors_max": fit * prompt,
        "plastic_backward_feature_vectors_max": fit * prompt,
        "optimizer_steps_max": fit * config["trainable_groups"]["updates_per_fit_event"],
        "generation_requests_max": requests,
        "autoregressive_forwards_max": requests * new_tokens,
        "forward_calls_max": 4 * fit + 2 * requests * new_tokens,
        "generated_tokens_max": requests * new_tokens,
        "processed_text_positions_max": (3 * fit * prompt
            + 2 * requests * positions_per_request),
        "retrieval_calls_max": fit + requests,
        "verification_calls_max": fit + arms * cases,
        "checkpoint_writes_max": fit + 1,
        "retained_checkpoint_artifacts_max": fit,
        "reloads_max": restarts,
        "cache_hits_max": 0,
        "cache_misses_max": fit + requests,
    }
    for key, value in expected.items():
        _require(schedule.get(key) == value, "budget",
                 f"{key}={schedule.get(key)} differs from derived {value}")
    for key in ("teacher_passes_max", "replay_events_max", "consolidation_events_max",
                "native_retries"):
        _require(schedule.get(key) == 0, "budget", f"{key} must be zero for this pilot")
    _require(restarts == 1, "budget", "exactly one restart is required")
    _require(schedule["artifact_bytes_max"] > 0 and schedule["wall_seconds_max"] > 0,
             "budget", "artifact and wall ceilings must be positive")
    return expected
def validate_config(config: Mapping[str, Any], root: Path = ROOT) -> dict[str, Any]:
    _require(config.get("format") == EXPECTED_FORMAT, "config", "format differs")
    _require(config.get("protocol_id") == PROTOCOL_ID, "config", "protocol id differs")
    _require(config["active_direction"]["source_baseline"] == "official_fp8",
             "source", "active baseline must remain official_fp8")
    _require(config["active_direction"]["internal_spine"] == "early_permanent_required",
             "spine", "internal spine must be early and permanent")
    _require(config["trainable_groups"]["optimizer"] == "NativeFusedAdamW",
             "optimizer", "only NativeFusedAdamW is admitted")
    _require(config["trainable_groups"]["frozen_original_backbone"] is True,
             "optimizer", "pilot must keep the original backbone frozen")
    memory = config["memory_contract"]
    for key in ("evidence_storage_records_max", "retained_history_versions_max",
                "active_read_slots_max", "snapshot_cache_entries_max",
                "metadata_bytes_max", "replay_bytes_max", "journal_records_max",
                "journal_bytes_max", "checkpoint_transient_bytes_max"):
        _require(type(memory[key]) is int and memory[key] > 0,
                 "memory", f"{key} must be positive")
    _require(memory["active_read_slots_max"] <= 8,
             "memory", "pilot read budget cannot exceed eight slots")
    _require(memory["evidence_storage_records_max"] > memory["active_read_slots_max"],
             "memory", "storage and read budgets must be distinct")
    _require(memory["representation_codec_mismatch"] == "reject_without_silent_conversion",
             "memory", "incompatible representations must be rejected")
    _require(config["runtime"]["spine_sites"] == [3, 7], "spine", "pilot sites differ")
    _require(config["runtime"]["generation"]["greedy"] is True
             and config["runtime"]["generation"]["sampling"] is False,
             "generation", "pilot must use greedy decoding")
    _require(config["runtime"]["generation"]["enable_thinking"] is False,
             "generation", "thinking level must be pinned off")
    _require(config["dataset"]["sealed_final_labels_closed"] is True,
             "access", "sealed-final labels must remain closed")
    required_arms = {arm["id"] for arm in config["controls"]["arms"]}
    _require(required_arms == {"full_eligible_context", "selective_exact_and_procedure",
                               "spine_E_only", "source_free_P_only", "E_plus_P",
                               "no_update", "reset_P", "shuffled_memory",
                               "inapplicable_procedure"},
             "controls", "required comparison arms differ")
    _require(config["success_criteria"]["reasoning_reuse"]["accuracy_gain_required"] is False,
             "success", "reasoning noninferiority must not require an accuracy gain")
    guards = config["work_and_cost"]["guards"]
    _require(guards["driver_limit_bytes"] == 53687091200
             and guards["sustained_swap_growth_limit_bytes"] == 536870912
             and guards["minimum_host_free_fraction"] == 0.15
             and guards["exclusive_native_worker"] is True
             and guards["cpu_fallback"] is False,
             "guards", "native resource guard changed")
    launch = config["launch"]
    _require(launch["runner_status"] == "implemented", "launch", "runner is not implemented")
    for relative in (launch["runner"], launch["preflight"], *launch["implementation_files"]):
        _require((root / relative).is_file(), "implementation", f"missing {relative}")
    work_units = derive_work(config)
    return {
        "source": _source_validation(config, root, full_weight_hash=False),
        "implementation_sha256": {
            relative: sha256_file(root / relative)
            for relative in launch["implementation_files"]
        },
        "work_units": work_units,
        "native_execution": False,
        "model_loaded": False,
        "tokenizer_loaded": False,
        "labels_read": False,
        "downloads": 0,
    }


def _missing_bindings(config: Mapping[str, Any], report: Mapping[str, Any],
                      authorization: Path | None) -> list[str]:
    missing = []
    launch = config["launch"]
    if launch.get("protocol_commit") is None:
        missing.append("protocol_commit")
    if config["dataset"].get("manifest_sha256") is None:
        missing.append("prepared_manifest_sha256")
    if authorization is None:
        missing.append("external_authorization_receipt")
    if not launch.get("implementation_sha256"):
        missing.append("implementation_sha256")
    backend_path = _resolve(ROOT, launch["backend_evidence_path"])
    if not backend_path.is_file():
        missing.append("native_fp8_backend_evidence")
    if report["source"]["required_backend"] and not launch["required_execution_binding"].get(
            "execution_backend"):
        missing.append("execution_backend")
    return missing


def _validate_authorization(config: Mapping[str, Any], authorization: Path,
                            output: Path, root: Path,
                            implementation_hashes: Mapping[str, str], *,
                            config_path: Path = CONFIG_PATH) -> dict[str, Any]:
    launch = config["launch"]
    receipt = read_json(authorization)
    for key in launch["required_authorization_fields"]:
        _require(key in receipt and receipt[key] not in (None, "", []),
                 "authorization", f"missing {key}")
    _require(receipt["protocol_id"] == PROTOCOL_ID, "authorization", "protocol id differs")
    _require(receipt["protocol_sha256"] == sha256_file(config_path),
             "authorization", "protocol hash differs")
    head = _current_head(root)
    protocol_commit = launch.get("protocol_commit")
    _require(isinstance(protocol_commit, str) and len(protocol_commit) == 40,
             "authorization", "protocol implementation commit is missing")
    ancestry = subprocess.run(
        ["git", "merge-base", "--is-ancestor", protocol_commit, head],
        cwd=root, capture_output=True, check=False)
    _require(ancestry.returncode == 0, "authorization",
             "protocol implementation commit is not an ancestor of HEAD")
    drift = subprocess.run(
        ["git", "diff", "--quiet", protocol_commit, head, "--",
         *launch["implementation_files"]], cwd=root, capture_output=True, check=False)
    _require(drift.returncode == 0, "authorization",
             "implementation files changed after the protocol commit")
    _require(receipt["protocol_commit"] == protocol_commit,
             "authorization", "receipt and protocol implementation commits differ")
    _require(receipt["implementation_sha256"] == dict(implementation_hashes),
             "authorization", "implementation hashes differ")
    _require(launch.get("implementation_sha256") == dict(implementation_hashes),
             "authorization", "config implementation binding differs")
    _require(receipt["prepared_manifest_sha256"] == config["dataset"]["manifest_sha256"],
             "authorization", "prepared manifest hash differs")
    manifest_path = _resolve(root, config["dataset"]["manifest"])
    _require(manifest_path.is_file(), "prepared", "prepared manifest is missing")
    _require(sha256_file(manifest_path) == config["dataset"]["manifest_sha256"],
             "prepared", "prepared manifest content hash differs")
    binding = launch["required_execution_binding"]
    for key, expected in binding.items():
        _require(receipt[key] == expected, "authorization",
                 f"execution binding {key} differs; no precision/backend substitution is allowed")
    backend_path = _resolve(root, launch["backend_evidence_path"])
    _require(backend_path.is_file(), "authorization", "native FP8 backend evidence is missing")
    backend = read_json(backend_path)
    for key, expected in launch["required_backend_evidence"].items():
        _require(backend.get(key) == expected, "authorization",
                 f"native FP8 backend evidence {key} differs")
    _require(receipt["backend_evidence_sha256"] == sha256_file(backend_path),
             "authorization", "backend evidence hash differs")
    _require(receipt["source_identity"] == "official_fp8",
             "authorization", "active source must remain official_fp8")
    _require(receipt["output_root"] == str(output),
             "authorization", "output root differs")
    _require(not output.exists(), "output", "native output must be fresh")
    _require(isinstance(receipt["issued_by"], str) and receipt["issued_by"].strip(),
             "authorization", "issued_by is required")
    try:
        expiry = datetime.fromisoformat(receipt["expires_at_utc"].replace("Z", "+00:00"))
    except (TypeError, ValueError) as error:
        raise ProtocolError("authorization", "expires_at_utc is invalid") from error
    _require(expiry.tzinfo is not None and expiry > datetime.now(timezone.utc),
             "authorization", "receipt is expired")
    _source_validation(config, root, full_weight_hash=True)
    return receipt


def guarded_command(config: Mapping[str, Any], output: Path,
                    authorization: Path, *, config_path: Path = CONFIG_PATH) -> str:
    report = output.parent / "guard_native01.json"
    return shlex.join([
        "PYTHONDONTWRITEBYTECODE=1", ".venv/bin/python",
        "scripts/run_capacity_preflight.py", "--entry-point", "run_spark_text_memory_v1.py",
        "--report", str(report), "--", "--device", "mps",
        "--output", str(output), "--authorization", str(authorization),
        "--config", str(config_path), "--prepared", str(_resolve(ROOT, config["dataset"]["root"])),
    ])


def preflight(config_path: Path = CONFIG_PATH, *, authorization: Path | None = None,
              output: Path | None = None, launch: bool = False) -> dict[str, Any]:
    config = read_json(config_path)
    report = validate_config(config, config_path.parent.parent if config_path.parent.name == "configs" else ROOT)
    output = (_resolve(ROOT, output) if output is not None
              else _resolve(ROOT, config["launch"]["output_root"]))
    report.update({
        "format": "replica.spark_text_memory.review.v1",
        "protocol_id": PROTOCOL_ID,
        "protocol_sha256": sha256_file(config_path),
        "protocol_commit": _current_head(ROOT),
        "status": "review_ready_preparation_only",
        "native_execution": False,
        "launch_allowed": False,
        "native_command": None,
        "output_root": str(output),
        "fresh_output_available": not output.exists(),
        "missing_bindings": _missing_bindings(config, report, authorization),
        "remaining_prerequisites": [
            "commit the complete implementation and bind protocol_commit and implementation_sha256",
            "materialize and hash the fit/development fixture manifest without opening sealed-final data",
            "issue an external authorization for official_fp8 and the exact verified execution backend",
            "preserve one exclusive worker and the 50 GiB / 512 MiB / 15% host-free guards",
        ],
    })
    if launch:
        _require(authorization is not None, "authorization",
                 "an external authorization receipt is required")
        receipt = _validate_authorization(
            config, authorization, output, ROOT, report["implementation_sha256"],
            config_path=config_path)
        report.update({
            "status": "launch_admitted",
            "launch_allowed": True,
            "native_command": guarded_command(
                config, output, authorization, config_path=config_path),
            "authorization_id": receipt["authorization_id"],
            "attempt_id": receipt["attempt_id"],
            "execution_format": receipt["execution_format"],
            "execution_backend": receipt["execution_backend"],
            "weight_sha256_checked": True,
        })
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--authorization", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = preflight(args.config, authorization=args.authorization,
                           output=args.output, launch=args.launch)
        if args.receipt.exists():
            raise ProtocolError("receipt", "receipt path must be fresh")
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                                encoding="utf-8")
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    except (ProtocolError, OSError, ValueError, subprocess.CalledProcessError) as error:
        detail = error.as_dict() if isinstance(error, ProtocolError) else {
            "check": "preflight", "detail": str(error)}
        print(json.dumps({"status": "refused", **detail}, indent=2, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

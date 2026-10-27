#!/usr/bin/env python3
"""Read-only teacher-forced diagnosis for the nested-text response readout.

The sense frame is built from the prompt alone. Gold response tokens are used
only for a no-grad teacher-forced action-pass measurement; no optimizer,
controller, coordinator, generation or held-out evaluation is constructed.
Launch through run_capacity_preflight.py --entry-point diagnose_nested_response.py.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
for _name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
    os.environ[_name] = "1"

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import run_nested_text as nested  # noqa: E402
from replica_lab.architecture.loading import exclusive_mps_worker, load_local_mps  # noqa: E402
from replica_lab.memory_system.ledger import RecordLedger  # noqa: E402
from replica_lab.memory_system.session import MemorySession  # noqa: E402
from replica_lab.nested_text import P_ONLY_KEYS  # noqa: E402

DIAGNOSTIC_FORMAT = "replica.nested_text.response_diagnostic.v1"


def _path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    nested._require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def _validate_declaration(config: dict, parent: dict, parent_path: Path) -> None:
    nested._require(config.get("format") == DIAGNOSTIC_FORMAT,
                    "unsupported response diagnostic protocol")
    nested._require(config.get("training_admitted") is False
                    and config.get("promotion") is False,
                    "response diagnosis must remain read-only and non-promoting")
    nested._require(config.get("parent_protocol_sha256")
                    == nested._sha256_file(parent_path),
                    "parent behavioral protocol digest differs")
    nested._require(parent.get("format") == nested.CONFIG_FORMAT_BEHAVIORAL,
                    "response diagnosis requires the behavioral parent protocol")
    nested.validate_v2_config(parent)
    episodes = config["episodes"]
    nested._require(episodes["root"] == parent["episodes"]["root"]
                    and episodes["manifest_sha256"]
                    == parent["episodes"]["manifest_sha256"],
                    "diagnostic episode lineage differs from the parent")
    workload = config["workload"]
    nested._require(int(episodes["attempts"]) == 28
                    and int(workload["teacher_forced_attempts_per_state"]) == 28
                    and int(workload["updates"]) == 0
                    and int(workload["backward_calls"]) == 0
                    and workload["new_generation"] is False
                    and workload["heldout_opened"] is False,
                    "diagnostic workload must be 28 read-only rows with no held-out")
    nested._require(int(workload["max_new_tokens_in_reused_greedy_report"])
                    == int(parent["generation"]["max_new_tokens"]),
                    "reused greedy generation budget differs from the parent")
    resource = config["resource_policy"]
    nested._require(resource["device"] == "mps"
                    and resource["cpu_fallback"] is False
                    and resource["exclusive_worker"] is True
                    and int(resource["max_driver_bytes"]) == 53687091200
                    and int(resource["max_swap_growth_bytes"]) == 536870912
                    and int(resource["minimum_free_percent"]) == 15
                    and int(resource["sustained_host_samples"]) == 3,
                    "diagnostic resource policy differs from the native guard")
    nested._require(config["controls"]["prompt_only_sense_frame"] is True
                    and config["controls"]["gold_response_action_pass_only"] is True
                    and config["controls"]["no_new_training_or_tuning"] is True,
                    "diagnostic leakage/read-only controls are not declared")


def _validate_input(path: Path, expected_sha: str) -> None:
    nested._require(path.is_file(), f"diagnostic input is missing: {path}")
    nested._require(nested._sha256_file(path) == expected_sha,
                    f"diagnostic input digest differs: {path}")


def _row_metrics(output, ids: torch.Tensor, response_mask: torch.Tensor):
    logits = output.logits[0].float()
    positions = response_mask[0].nonzero(as_tuple=False).flatten()
    nested._require(positions.numel() > 0 and bool((positions > 0).all()),
                    "teacher-forced response positions must follow a context token")
    targets = ids[0, positions]
    prediction_logits = logits[positions - 1]
    predictions = prediction_logits.argmax(dim=-1)
    correct = predictions == targets
    nll = F.cross_entropy(prediction_logits, targets, reduction="mean")
    return {
        "response_tokens": int(positions.numel()),
        "response_nll": float(nll.detach().cpu()),
        "model_loss": float(output.loss.detach().cpu()),
        "response_token_accuracy": float(correct.float().mean().detach().cpu()),
        "teacher_forced_exact": bool(correct.all().detach().cpu()),
        "predicted_ids": predictions.detach().cpu().tolist(),
        "target_ids": targets.detach().cpu().tolist(),
    }


def teacher_forced(learner, session: MemorySession, episodes: list,
                   *, label: str) -> dict:
    """Measure prompt-conditioned response logits without constructing a graph."""
    rows = []
    for episode in episodes:
        request = None
        started = time.perf_counter()
        row = {"id": episode.row["id"], "label": label,
               "expected": episode.expected,
               "prompt_sha256": episode.row["prompt_sha256"]}
        try:
            request, audit = nested.build_read_request(
                learner, session, episode, route_mode=None,
                task_id=f"teacher-forced:{label}:{episode.row['id']}")
            device = session.device
            ids = episode.input_ids.to(device)
            attention = episode.attention_mask.to(device)
            response = episode.response_mask.to(device)
            labels = ids.clone()
            labels[~response] = -100
            with torch.no_grad():
                output = session.forward(request, ids, labels=labels,
                                         attention_mask=attention)
                nested._require(output.loss is not None,
                                "teacher-forced response loss is missing")
                row.update(_row_metrics(output, ids, response))
            row["loss_alignment_abs"] = abs(row["model_loss"] - row["response_nll"])
            row["request_audit"] = audit
            row["prompt_positions"] = int(episode.prompt_ids.shape[1])
            row["full_positions"] = int(episode.input_ids.shape[1])
            row["predicted_text"] = session.tokenizer.decode(
                row["predicted_ids"], skip_special_tokens=True)
            row["target_text"] = session.tokenizer.decode(
                row["target_ids"], skip_special_tokens=True)
        except BaseException as error:
            row["error"] = f"{type(error).__name__}: {error}"
            row["traceback"] = traceback.format_exc()
        finally:
            if request is not None:
                request.release()
        row["total_seconds"] = time.perf_counter() - started
        rows.append(row)
    successful = [row for row in rows if "error" not in row]
    token_count = sum(row["response_tokens"] for row in successful)
    total_seconds = [row["total_seconds"] for row in rows]
    work = {
        "requests": len(rows),
        "successful_requests": len(successful),
        "plastic_feature_forward_calls": len(successful),
        "session_sense_forward_calls": len(successful),
        "teacher_forced_forward_calls": len(successful),
        "forward_calls": {
            "plastic_read": len(successful),
            "session_prepare": len(successful),
            "teacher_forced": len(successful),
            "total": 3 * len(successful),
        },
        "plastic_feature_prompt_positions": sum(
            row.get("prompt_positions", 0) for row in successful),
        "session_sense_prompt_positions": sum(
            row.get("prompt_positions", 0) for row in successful),
        "teacher_forced_positions": sum(
            row.get("full_positions", 0) for row in successful),
        "processed_positions": sum(
            2 * row.get("prompt_positions", 0) + row.get("full_positions", 0)
            for row in successful),
        "response_supervised_tokens": token_count,
        "backward_calls": 0,
        "replay_calls": 0,
        "new_generated_ids": 0,
    }
    return {
        "label": label,
        "count": len(rows),
        "successful": len(successful),
        "failures": len(rows) - len(successful),
        "mean_response_nll": (sum(row["response_nll"] for row in successful)
                              / len(successful) if successful else None),
        "response_token_accuracy": (
            sum(row["response_token_accuracy"] * row["response_tokens"]
                for row in successful) / token_count if token_count else None),
        "teacher_forced_exact_response": (
            sum(int(row["teacher_forced_exact"]) for row in successful)
            / len(successful) if successful else None),
        "teacher_forced_exact_count": sum(
            int(row["teacher_forced_exact"]) for row in successful),
        "loss_alignment_max_abs": max(
            (row["loss_alignment_abs"] for row in successful), default=None),
        "latency_ms": {"total_seconds": {
            "count": len(rows),
            "p50_ms": nested.p50(total_seconds) * 1000.0 if total_seconds else 0.0,
            "p95_ms": nested.p95(total_seconds) * 1000.0 if total_seconds else 0.0,
        }},
        "work": work,
        "results": rows,
    }


def _greedy_summary(path: Path, expected_sha: str, parent_sha: str,
                    episode_ids: set[str]) -> dict:
    _validate_input(path, expected_sha)
    report = _load_json(path)
    nested._require(report.get("config_sha256") == parent_sha,
                    "reused greedy report belongs to another parent protocol")
    scores = report["scores"]
    rows = scores["results"]
    nested._require(scores["label"] == "fit" and len(rows) == len(episode_ids)
                    and {row["id"] for row in rows} == episode_ids,
                    "reused greedy report rows differ from the diagnostic workload")
    return {
        "path": str(path), "sha256": expected_sha,
        "count": len(rows), "correct": int(scores["correct"]),
        "accuracy": float(scores["accuracy"]),
        "failures": int(scores["failures"]),
        "new_generation": False,
        "stop_reasons": report["work"]["stop_reasons"],
        "truncated_attempts": report["work"]["truncated_attempts"],
        "generated_ids": int(report["work"]["generated_ids"]),
    }


def run(args) -> int:
    output = args.output
    config = _load_json(args.protocol)
    parent_path = _path(config["parent_protocol"])
    parent = _load_json(parent_path)
    _validate_declaration(config, parent, parent_path)
    manifest = nested.load_episode_manifest(args.episodes)
    nested.validate_episode_binding(args, parent, manifest)
    inputs = config["inputs"]
    trained_path = args.p_state or _path(inputs["trained_p_state"])
    baseline_path = args.baseline_p_state or _path(inputs["baseline_p_state"])
    greedy_path = args.greedy_report or _path(inputs["greedy_report"])
    _validate_input(trained_path, inputs["trained_p_state_sha256"])
    _validate_input(baseline_path, inputs["baseline_p_state_sha256"])
    evaluation_started = time.perf_counter()
    with exclusive_mps_worker(ROOT):
        adapter, tokenizer, lock = load_local_mps(args.model_path, args.model_lock)
        protocol = nested.protocol_for_world(
            parent, lock, hidden_size=adapter.base.config.hidden_size,
            world_uid=manifest["primary_world"])
        adapter.install_plastic(protocol.plastic_spec)
        episodes = nested.episodes_for_teaching(
            args.episodes, manifest, tokenizer, protocol,
            max_prompt_tokens=int(config["workload"]["max_prompt_tokens"]),
            max_response_tokens=int(config["workload"]["max_response_tokens"]),
            limit=int(config["episodes"]["attempts"]))
        nested._require(len(episodes) == int(config["episodes"]["attempts"]),
                        "diagnostic did not load exactly 28 teaching rows")
        trained_state = torch.load(trained_path, map_location="cpu", weights_only=True)
        baseline_state = torch.load(baseline_path, map_location="cpu", weights_only=True)
        nested._require(set(trained_state) == P_ONLY_KEYS
                        and set(baseline_state) == P_ONLY_KEYS,
                        "diagnostic states must be strict P-only payloads")
        trained = nested.build_read_only_learner(protocol, parent, adapter, trained_state)
        baseline = nested.build_read_only_learner(protocol, parent, adapter, baseline_state)
        (output / "trained").mkdir(parents=True, exist_ok=True)
        (output / "baseline").mkdir(parents=True, exist_ok=True)
        trained_session = MemorySession(
            adapter, tokenizer, RecordLedger(output / "trained" / "ledger.sqlite"))
        baseline_session = MemorySession(
            adapter, tokenizer, RecordLedger(output / "baseline" / "ledger.sqlite"))
        trained_result = teacher_forced(trained, trained_session, episodes, label="trained")
        baseline_result = teacher_forced(baseline, baseline_session, episodes,
                                         label="null_parent_baseline")
        episode_ids = {episode.row["id"] for episode in episodes}
        greedy = _greedy_summary(greedy_path, inputs["greedy_report_sha256"],
                                 nested._sha256_file(parent_path), episode_ids)
        all_rows = trained_result["results"] + baseline_result["results"]
        source_free = all(
            row.get("request_audit", {}).get("sources_present") is False
            and row["request_audit"].get("memory_spans") == 0
            and row["request_audit"].get("selection_records") == 0
            and row["request_audit"].get("ledger_records") == 0
            and row["request_audit"].get("controller_constructed") is False
            and row["request_audit"].get("coordinator_constructed") is False
            and row["request_audit"].get("updates_enabled") is False
            for row in all_rows if "error" not in row)
        complete = (trained_result["failures"] == 0
                    and baseline_result["failures"] == 0 and source_free)
        delta = None
        if complete:
            delta = {
                "mean_response_nll": (trained_result["mean_response_nll"]
                                      - baseline_result["mean_response_nll"]),
                "response_token_accuracy": (trained_result["response_token_accuracy"]
                                            - baseline_result["response_token_accuracy"]),
                "teacher_forced_exact_response": (
                    trained_result["teacher_forced_exact_response"]
                    - baseline_result["teacher_forced_exact_response"]),
            }
        report = {
            "format": DIAGNOSTIC_FORMAT,
            "diagnostic_complete": complete,
            "status": "completed" if complete else "failed",
            "parent_protocol": str(parent_path),
            "parent_protocol_sha256": nested._sha256_file(parent_path),
            "trained_p_state": str(trained_path),
            "baseline_p_state": str(baseline_path),
            "trained": trained_result,
            "null_parent_baseline": baseline_result,
            "trained_minus_baseline": delta,
            "reused_greedy": greedy,
            "comparison_note": (
                "Teacher-forced exact response and reused greedy scored answer are "
                "different readout conditions; no cross-condition accuracy delta is claimed."),
            "source_free": source_free,
            "updates": 0,
            "backward_calls": 0,
            "new_generation": False,
            "heldout_opened": False,
            "sealed_final_rows_parsed": False,
            **nested.version_metadata(args),
        }
        nested._write_json(output / "response_diagnostic.json", report)
        nested._write_json(output / "cost.json", {
            "evaluation_end_to_end_seconds": time.perf_counter() - evaluation_started,
            "peak_driver_bytes": nested.peak_driver_bytes(),
            "model_loads": 1, "tokenizer_loads": 1,
            "updates": 0, "backward_calls": 0, "new_generated_ids": 0,
            "reused_greedy_generated_ids": greedy["generated_ids"],
            "trained_work": trained_result["work"],
            "baseline_work": baseline_result["work"],
            **nested.version_metadata(args),
        })
        nested._write_json(output / "run.json", {
            "format": "replica.nested_text.run.v1",
            "mode": "response_diagnostic", "status": "completed" if complete else "failed",
            "attempts_per_state": len(episodes), "updates": 0,
            "backward_calls": 0, "new_generation": False,
            "heldout_opened": False, "source_free": source_free,
            "trained_mean_response_nll": trained_result["mean_response_nll"],
            "baseline_mean_response_nll": baseline_result["mean_response_nll"],
            "trained_response_token_accuracy": trained_result["response_token_accuracy"],
            "baseline_response_token_accuracy": baseline_result["response_token_accuracy"],
            **nested.version_metadata(args),
        })
    return 0 if complete else 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("mps",), default="mps")
    parser.add_argument("--p-state", type=Path)
    parser.add_argument("--baseline-p-state", type=Path)
    parser.add_argument("--greedy-report", type=Path)
    args = parser.parse_args()
    try:
        nested._fresh_output(args.output)
        from replica_lab.profiling import RunProfiler
        with RunProfiler("mps", args.output):
            return run(args)
    except BaseException as error:
        nested._write_json(args.output / "failure.json", {
            "status": "failed", "mode": "response_diagnostic",
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
            "raw_outputs_preserved": True,
            "heldout_opened": False,
        })
        return 1


if __name__ == "__main__":
    sys.exit(main())

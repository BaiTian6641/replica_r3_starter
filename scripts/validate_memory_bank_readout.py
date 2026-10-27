#!/usr/bin/env python3
"""Deterministic, fail-closed replay of the preserved H1 readout evidence.

This script validates the already-produced ``first02`` and ``restart02``
reports and the H1 summary without rerunning a native job and without
reopening the role-sealed heldout labels.  It:

* verifies report/summary formats, identities, protocol pins and raw hashes;
* rebuilds every decision row (456 rows) from the pinned prepared inputs and
  compares the canonical bytes with the frozen raw decision files;
* replays every development phase, literal, state and restart-accounting gate
  from the raw generation rows and the pinned development labels;
* checks the heldout row/count/format structure and the reported miss
  arithmetic, but explicitly does *not* re-read or re-score heldout labels;
* validates the memory-disabled classification and the zero gradient /
  optimizer-update / checkpoint-write / download claims;
* emits a deterministic receipt (no wall-clock timestamp) and exits non-zero
  on any mismatch.

The replay uses the project's existing readout scoring utilities.  They import
PyTorch, so run this script with the project virtual environment::

    .venv/bin/python scripts/validate_memory_bank_readout.py \\
        --receipt reports/memory_bank_readout_20261026/validation_receipt.json

No model, tokenizer or native process is constructed.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]

FIRST_DIR = "runs/memory_bank_readout_20261026/first02"
RESTART_DIR = "runs/memory_bank_readout_20261026/restart02"
SUMMARY_PATH = "reports/memory_bank_readout_20261026/summary.json"
CONFIG_PATH = "configs/memory_bank_readout_v2.json"
PREPARED_PATH = "data/memory_causal_fresh_20261026/prepared04"
ROLES_PATH = "data/memory_causal_fresh_20261026/prepared04_roles"

REPORT_FORMAT = "replica.memory_bank_readout.report.v2"
SUMMARY_FORMAT = "replica.memory_bank_readout.h1.summary.v1"
RECEIPT_FORMAT = "replica.memory_bank_readout.h1.replay_receipt.v1"
DECISIONS_FORMAT = "replica.memory_address_bank.decisions.v1"
GENERATIONS_FORMAT = "replica.memory_bank_readout.generations.v2"

EXPECTED_CLASSIFICATION = {
    "answer_source": "sealed_minicpm5_greedy_readout",
    "bank_plus_compressed_arm": False,
    "calibrated": False,
    "checkpoint_state_role": "validated_only_not_consumed_by_generation",
    "compressed_arm": "separate_reference_not_combined",
    "few_shot": False,
    "generation": "model_readout_only",
    "host_selected_answer": False,
    "learned_selector": False,
    "model_execution": True,
    "native": True,
    "oracle": "diagnostic_only_not_executed",
    "package": "selected_record_literal_value_and_model_readout_v2",
    "persistent_learning": False,
    "retries": False,
    "structured_decoding": False,
    "tokenizer_load": True,
}

EXPECTED_GENERATION_CONTRACT = {
    "cap_hits_max": 0,
    "core_impl": "none",
    "empty_ledger": True,
    "enabled_branches": ["memory"],
    "enabled_ports": [],
    "ended_by_eos_required": True,
    "max_new_tokens": 32,
    "record_memory": False,
    "single_line_required": True,
    "stop_token_ids": [1, 130073],
    "workspace": False,
}

EXPECTED_PROMPT_CONTRACT = {
    "answer_normalization": "strip+collapse_whitespace+casefold",
    "contract_lines": [
        "Response contract: Output exactly one line and then stop.",
        "Do not explain, reason, quote, use Markdown, or add a prefix.",
        "Do not invent values.",
        "If a record value is shown, output exactly that value.",
        "If a record status is shown, output exactly that status word.",
        "Permitted status words: UNKNOWN, REVOKED, AMBIGUOUS, SUPERSEDED.",
    ],
    "evaluation_labels_read": False,
    "expected_answer_included": False,
    "few_shot": False,
    "fields": [
        "response_contract",
        "query_text",
        "selected_record_value_or_runtime_status",
    ],
    "format": "replica.memory_bank_readout.prompt.v2",
    "host_selected_answer": False,
    "status_vocabulary": ["unknown", "revoked", "ambiguous", "superseded"],
    "structured_decoding": False,
}

ZERO_RESOURCE_KEYS = (
    "gradients",
    "optimizer_updates",
    "checkpoints_written",
    "base_exports",
    "extension_exports",
    "downloads",
)

EXPECTED_RAW_KEYS = {
    "first": ("dev", "dev_first_process"),
    "restart": (
        "dev",
        "dev_restart",
        "dev_revocation",
        "heldout",
        "heldout_generations",
        "train",
    ),
}

FIRST_OUTCOME = {
    "status": "first_process_passed_restart_pending",
    "literal_dev": "passed",
    "model": "first_process_passed",
    "heldout": "not_run",
    "train": "not_run",
}

RESTART_OUTCOME = {
    "status": "dev_passed_heldout_evaluated",
    "literal_dev": "passed",
    "model": "restart_revocation_passed",
    "heldout": "evaluated",
    "train": "descriptive_only",
}

GATING_ORDER = {
    "first": ["literal_dev", "first_process"],
    "restart": [
        "first_receipt",
        "literal_dev",
        "restart",
        "revocation",
        "state_denominators",
        "restart_accounting",
        "heldout",
        "train",
    ],
}

HELDOUT_GATE = {
    "enforced": False,
    "metrics_role": "post_freeze_confirmation_only",
    "retried": False,
    "single_shot": True,
    "thresholds_predeclared": True,
    "tuned": False,
}


class ReplayError(ValueError):
    """One fail-closed replay check failed."""

    def __init__(self, check: str, detail: str,
                 evidence: Mapping[str, Any] | None = None):
        super().__init__(f"{check}: {detail}")
        self.check = check
        self.detail = detail
        self.evidence = dict(evidence or {})


# ---------------------------------------------------------------------------
# Small deterministic helpers
# ---------------------------------------------------------------------------

def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False)


def sha_json(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _short(value: Any, limit: int = 400) -> str:
    try:
        text = canonical(value)
    except (TypeError, ValueError):
        text = repr(value)
    return text if len(text) <= limit else text[:limit] + "…"


def _expect(actual: Any, expected: Any, check: str) -> None:
    if actual != expected:
        raise ReplayError(
            check, f"expected {_short(expected)}; found {_short(actual)}")


def _expect_true(condition: bool, check: str, detail: str) -> None:
    if not condition:
        raise ReplayError(check, detail)


def _read_json(path: Path, check: str) -> dict:
    _expect_true(Path(path).is_file(), check, f"missing file: {path}")
    try:
        doc = json.loads(Path(path).read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReplayError(check, f"unreadable JSON {path}: {error}") from error
    _expect_true(isinstance(doc, dict), check,
                 f"JSON root is not an object: {path}")
    return doc


def _read_jsonl(path: Path, check: str) -> tuple[dict, ...]:
    _expect_true(Path(path).is_file(), check, f"missing file: {path}")
    rows = []
    try:
        lines = Path(path).read_text().splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise ReplayError(check, f"unreadable JSONL {path}: {error}") from error
    for number, line in enumerate(lines, 1):
        if not line.strip():
            raise ReplayError(check,
                              f"blank JSONL line {number} in {path.name}")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ReplayError(
                check, f"invalid JSONL line {number} in {path.name}: {error}"
            ) from error
        _expect_true(isinstance(row, dict), check,
                     f"JSONL line {number} in {path.name} is not an object")
        rows.append(row)
    return tuple(rows)


def _ratio(numerator: int, denominator: int) -> float | None:
    return None if not denominator else numerator / denominator


def _relative(path: Path, base: Path) -> str:
    try:
        return Path(path).resolve().relative_to(Path(base).resolve()).as_posix()
    except ValueError:
        return Path(path).resolve().as_posix()


def _resolve(path: str | Path, base: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else Path(base) / candidate


def decision_identity(episode_id: str, state: str, phase: str,
                      query_id: str) -> str:
    """The address-bank's frozen decision identity for one decision key."""
    return "decision-" + sha_json(
        [episode_id, state, phase, query_id])[:16]


# ---------------------------------------------------------------------------
# Report shape, classification and resource claims
# ---------------------------------------------------------------------------

def validate_classification(report: Mapping[str, Any]) -> None:
    """The H1 classification must be the literal readout classification."""
    _expect(dict(report.get("classification") or {}), EXPECTED_CLASSIFICATION,
            "report.classification")


def validate_memory_disabled(report: Mapping[str, Any],
                             config: Mapping[str, Any]) -> None:
    """H1 is a literal readout, not retrieval from a live memory store."""
    protocol = report.get("protocol") or {}
    _expect(dict(protocol.get("generation_contract") or {}),
            EXPECTED_GENERATION_CONTRACT,
            "report.protocol.generation_contract")
    _expect(dict(protocol.get("prompt_contract") or {}),
            EXPECTED_PROMPT_CONTRACT, "report.protocol.prompt_contract")

    generation = config.get("generation") or {}
    for key, value in (
            ("record_memory", False),
            ("workspace", False),
            ("enabled_ports", []),
            ("core_impl", "none"),
            ("few_shot", False),
            ("structured_decoding", False),
            ("host_selected_answer", False),
            ("training", False),
    ):
        _expect(generation.get(key), value, f"config.generation.{key}")

    classification = report.get("classification") or {}
    for key in ("persistent_learning", "learned_selector", "calibrated",
                "bank_plus_compressed_arm", "retries"):
        _expect(classification.get(key), False,
                f"report.classification.{key}")


def validate_zero_updates(report: Mapping[str, Any],
                          config: Mapping[str, Any]) -> None:
    """No gradient, optimizer update, checkpoint write or download occurred."""
    usage = report.get("resource_usage") or {}
    for key in ZERO_RESOURCE_KEYS:
        _expect(usage.get(key), 0, f"report.resource_usage.{key}")
    _expect(usage.get("fallback"), "disabled",
            "report.resource_usage.fallback")

    baseline = report.get("baseline") or {}
    for label, item in baseline.items():
        checkpoint = (item or {}).get("checkpoint") or {}
        _expect(checkpoint.get("written"), False,
                f"report.baseline.{label}.checkpoint.written")
        _expect(checkpoint.get("loaded"), True,
                f"report.baseline.{label}.checkpoint.loaded")

    budget = config.get("budget") or {}
    for key in ("gradients", "optimizer_updates", "checkpoints_written",
                "base_export", "extension_export", "downloads"):
        _expect(budget.get(key), 0, f"config.budget.{key}")


def check_resource_usage(report: Mapping[str, Any],
                         generation_rows: Sequence[Mapping[str, Any]],
                         config: Mapping[str, Any]) -> dict:
    """Recompute the per-job resource block from the raw generation rows."""
    generation = config["generation"]
    expected = {
        "job": report["job"],
        "model_loads": 1,
        "tokenizer_loads": 1,
        "generation_rows": len(generation_rows),
        "generated_tokens": sum(len(row["generated_ids"])
                                for row in generation_rows),
        "optimizer_updates": 0,
        "gradients": 0,
        "checkpoints_written": 0,
        "base_exports": 0,
        "extension_exports": 0,
        "downloads": 0,
        "native_jobs": 1,
        "fallback": "disabled",
        "limits": {
            "max_model_loads_per_job":
                generation["model_loads_per_job"],
            "max_decisions_per_split":
                generation["max_decisions_per_split"],
            "max_new_tokens": generation["max_new_tokens"],
            "max_wall_seconds":
                generation["wall_budget_seconds_per_job"],
            "artifact_cap_bytes": generation["artifact_cap_bytes"],
        },
    }
    _expect(dict(report.get("resource_usage") or {}), expected,
            "report.resource_usage")
    return expected


# ---------------------------------------------------------------------------
# Raw outputs
# ---------------------------------------------------------------------------

def validate_manifest_entry(entry: Mapping[str, Any], directory: Path,
                            check: str) -> dict:
    """Hash and row-count one immutable raw output file."""
    filename = entry.get("filename")
    _expect_true(isinstance(filename, str) and filename, check + ".filename",
                 "manifest filename is missing")
    _expect_true("/" not in filename and "\\" not in filename,
                 check + ".filename", "manifest filename is not a basename")
    path = Path(directory) / filename
    _expect_true(path.is_file(), check,
                 f"raw output is missing: {path}")
    _expect(entry.get("rows"), len(_read_jsonl(path, check)),
            check + ".rows")
    _expect(entry.get("sha256"), sha256_file(path), check + ".sha256")
    _expect(entry.get("first_write"), True, check + ".first_write")
    _expect(entry.get("immutable"), True, check + ".immutable")
    return {"filename": filename, "rows": entry["rows"],
            "sha256": entry["sha256"]}


def validate_raw_manifest(report: Mapping[str, Any], directory: Path,
                          config: Mapping[str, Any]) -> dict:
    """Validate every raw output recorded by one report."""
    raw = report.get("raw_outputs") or {}
    _expect(dict(raw.get("policy") or {}),
            dict((config.get("reporting") or {}).get("raw_output_policy") or {}),
            "report.raw_outputs.policy")
    _expect(raw.get("verified_unchanged"), True,
            "report.raw_outputs.verified_unchanged")
    files = raw.get("files") or {}
    expected_keys = EXPECTED_RAW_KEYS[report["job"]]
    _expect(sorted(files), sorted(expected_keys), "report.raw_outputs.files")
    result = {}
    for key in expected_keys:
        result[key] = validate_manifest_entry(
            files[key], Path(directory), f"raw.{report['job']}.{key}")
    return result


def validate_decision_identities(rows: Sequence[Mapping[str, Any]],
                                 check: str = "decisions") -> dict:
    """Validate format, frozen identity formula and key uniqueness."""
    identities = set()
    keys = set()
    for row in rows:
        _expect(row.get("format"), DECISIONS_FORMAT, check + ".format")
        episode_id = row.get("episode_id")
        state = row.get("state")
        phase = row.get("phase")
        query_id = row.get("query_id")
        for name, value in (("episode_id", episode_id), ("state", state),
                            ("phase", phase), ("query_id", query_id)):
            _expect_true(isinstance(value, str) and bool(value),
                         check + "." + name, f"missing {name}")
        _expect(row.get("decision_id"),
                decision_identity(episode_id, state, phase, query_id),
                check + ".decision_id")
        key = (episode_id, state, phase, query_id)
        _expect_true(key not in keys, check,
                     f"repeated decision key: {key}")
        keys.add(key)
        identities.add(row["decision_id"])
    _expect(len(identities), len(rows), check + ".unique_ids")
    return {"rows": len(rows), "unique_ids": len(identities),
            "unique_keys": len(keys)}


# ---------------------------------------------------------------------------
# Gate / block comparisons
# ---------------------------------------------------------------------------

def compare_gate(recorded: Any, recomputed: Any, check: str) -> None:
    if recorded != recomputed:
        raise ReplayError(
            check,
            "recorded gate differs from the replay: "
            f"recorded {_short(recorded)}; recomputed {_short(recomputed)}")


def _validate_block_arithmetic(block: Mapping[str, Any], check: str) -> None:
    n = block.get("n")
    _expect_true(isinstance(n, int) and n >= 0, check + ".n",
                 "block count is not a non-negative integer")
    for prefix, rate_key in (("correct", "exact_answer_rate"),
                             ("status_correct", "status_correct_rate"),
                             ("false_value", "false_value_rate")):
        count = block.get(prefix + "_n")
        _expect_true(isinstance(count, int), check + f".{prefix}_n",
                     "block count is not an integer")
        _expect(block.get(rate_key), _ratio(count, n),
                check + f".{rate_key}")


def _count_pair(block: Mapping[str, Any]) -> list[int]:
    return [block["correct_n"], block["n"]]


# ---------------------------------------------------------------------------
# Summary claims
# ---------------------------------------------------------------------------

def validate_summary(summary: Mapping[str, Any],
                     reports: Mapping[str, Mapping[str, Any]],
                     evidence: Mapping[str, Any], *,
                     root: Path,
                     first_relative: str,
                     restart_relative: str) -> list[str]:
    """Validate every replayable summary claim; return un-replayed claims."""
    _expect(summary.get("format"), SUMMARY_FORMAT, "summary.format")
    first, restart = reports["first"], reports["restart"]

    selected = summary.get("selected_runs") or {}
    _expect(sorted(selected), ["first", "restart"],
            "summary.selected_runs")
    first_report_relative = Path(first_relative, "report.json").as_posix()
    restart_report_relative = Path(restart_relative, "report.json").as_posix()
    _expect(selected["first"], first_report_relative,
            "summary.selected_runs.first")
    _expect(selected["restart"], restart_report_relative,
            "summary.selected_runs.restart")
    _expect((Path(root) / selected["first"]).resolve(),
            (Path(root) / first_report_relative).resolve(),
            "summary.selected_runs.first.resolve")
    _expect((Path(root) / selected["restart"]).resolve(),
            (Path(root) / restart_report_relative).resolve(),
            "summary.selected_runs.restart.resolve")

    first_date = str(first["generated_at_utc"])[:10]
    restart_date = str(restart["generated_at_utc"])[:10]
    _expect(first_date, restart_date, "report.dates")
    _expect(summary.get("date_utc"), first_date, "summary.date_utc")

    outcome = summary.get("outcome") or {}
    expected_outcome = {
        "literal_dev": "passed",
        "first_process": "passed",
        "restart_revocation_state_accounting": "passed",
        "heldout": "evaluated_once_after_all_dev_gates",
        "persistent_learning": False,
        "calibrated_confidence": False,
        "general_personal_memory_claim": False,
    }
    _expect(dict(outcome), expected_outcome, "summary.outcome")

    literal_gate = evidence["literal_gate"]
    first_metrics = evidence["first_process"]["metrics"]
    restart_metrics = evidence["restart"]["metrics"]
    revocation_metrics = evidence["revocation"]["metrics"]
    heldout_metrics = evidence["heldout"]["metrics"]
    state_gate = evidence["state_gate"]

    expected_metrics = {
        "literal_fact_value_exact": [
            literal_gate["checks"]["fact_value_exact"]["value_exact_n"],
            literal_gate["checks"]["fact_value_exact"]["fact_n"],
        ],
        "first_exact_answers": _count_pair(first_metrics["overall"]),
        "first_correction": _count_pair(first_metrics["correction"]),
        "first_unaffected": _count_pair(first_metrics["unaffected"]),
        "first_unknown": _count_pair(first_metrics["unknown"]),
        "restart_exact_answers": _count_pair(restart_metrics["overall"]),
        "restart_correction": _count_pair(restart_metrics["correction"]),
        "restart_unaffected_retention":
            _count_pair(restart_metrics["unaffected"]),
        "restart_unknown": _count_pair(restart_metrics["unknown"]),
        "revocation_exact": _count_pair(revocation_metrics["revoked"]),
        "initial_fact_exact": [
            state_gate["checks"]["initial_fact_exact"]["correct_n"],
            state_gate["checks"]["initial_fact_exact"]["expected_n"],
        ],
        "corrected_fact_exact": [
            state_gate["checks"]["corrected_fact_exact"]["correct_n"],
            state_gate["checks"]["corrected_fact_exact"]["expected_n"],
        ],
        "heldout_exact_answers": _count_pair(heldout_metrics["overall"]),
        "heldout_exact_rate":
            heldout_metrics["overall"]["exact_answer_rate"],
        "single_line_rate":
            evidence["generation_format"]["single_line_rate"],
        "ended_by_eos_rate":
            evidence["generation_format"]["ended_by_eos_rate"],
        "cap_hits": evidence["generation_format"]["cap_hit_n"],
        "control_false_values": evidence["control_false_values"],
        "restart_text_disjoint_rate":
            evidence["accounting"]["disjoint_rate"],
        "old_observations_encoded":
            evidence["accounting"]["old_observations_encoded"],
        "old_first_process_queries_regenerated":
            evidence["accounting"]["old_first_process_queries_regenerated"],
    }
    metrics = summary.get("metrics") or {}
    _expect(sorted(metrics), sorted(expected_metrics), "summary.metrics.keys")
    for key in expected_metrics:
        _expect(metrics.get(key), expected_metrics[key], f"summary.metrics.{key}")

    misses = summary.get("heldout_misses") or {}
    _expect(sorted(misses), ["count", "false_value_generated", "kind", "note",
                             "runtime_status"], "summary.heldout_misses.keys")
    _expect(misses["count"],
            heldout_metrics["overall"]["n"]
            - heldout_metrics["overall"]["correct_n"],
            "summary.heldout_misses.count")
    _expect(misses["kind"], "predeclared_bank_abstentions",
            "summary.heldout_misses.kind")
    _expect(misses["false_value_generated"], False,
            "summary.heldout_misses.false_value_generated")
    _expect_true(isinstance(misses["runtime_status"], str)
                 and misses["runtime_status"], "summary.heldout_misses.status",
                 "runtime status is not a non-empty string")
    _expect_true(isinstance(misses["note"], str) and misses["note"],
                 "summary.heldout_misses.note", "note is not a string")

    resources = summary.get("resources") or {}
    resource_keys = [
        "first_guard_seconds", "restart_guard_seconds", "peak_driver_bytes",
        "peak_swap_growth_bytes", "minimum_observed_free_percent",
        "first_artifact_bytes", "restart_artifact_bytes",
        "model_loads_per_job", "tokenizer_loads_per_job", "gradients",
        "optimizer_updates", "checkpoint_writes", "downloads",
    ]
    _expect(sorted(resources), sorted(resource_keys),
            "summary.resources.keys")
    _expect(resources["first_artifact_bytes"],
            evidence["artifact_bytes"]["first"],
            "summary.resources.first_artifact_bytes")
    _expect(resources["restart_artifact_bytes"],
            evidence["artifact_bytes"]["restart"],
            "summary.resources.restart_artifact_bytes")
    _expect(resources["peak_driver_bytes"],
            evidence["profile"]["peak_driver_bytes"],
            "summary.resources.peak_driver_bytes")
    for key in ("model_loads_per_job", "tokenizer_loads_per_job"):
        _expect(resources[key], 1, f"summary.resources.{key}")
    for key, report_key in (("gradients", "gradients"),
                            ("optimizer_updates", "optimizer_updates"),
                            ("checkpoint_writes", "checkpoints_written"),
                            ("downloads", "downloads")):
        _expect(resources[key], evidence["resource_totals"][report_key],
                f"summary.resources.{key}")

    not_replayed = []
    guards = evidence.get("guards") or {}
    if guards.get("first") and guards.get("restart"):
        first_guard = guards["first"]
        restart_guard = guards["restart"]
        _expect(resources["first_guard_seconds"],
                first_guard["elapsed_seconds"],
                "summary.resources.first_guard_seconds")
        _expect(resources["restart_guard_seconds"],
                restart_guard["elapsed_seconds"],
                "summary.resources.restart_guard_seconds")
        _expect(resources["peak_swap_growth_bytes"],
                max(first_guard["guard"]["peak_swap_growth_bytes"],
                    restart_guard["guard"]["peak_swap_growth_bytes"]),
                "summary.resources.peak_swap_growth_bytes")
        _expect(resources["minimum_observed_free_percent"],
                min(first_guard["guard"]["minimum_observed_free_percent"],
                    restart_guard["guard"]["minimum_observed_free_percent"]),
                "summary.resources.minimum_observed_free_percent")
    else:
        not_replayed.extend([
            "resources.first_guard_seconds",
            "resources.restart_guard_seconds",
            "resources.peak_swap_growth_bytes",
            "resources.minimum_observed_free_percent",
        ])

    verification = summary.get("verification") or {}
    verification_keys = [
        "focused_tests", "independent_decision_rows_validated",
        "independent_dev_generations_validated",
        "independent_heldout_generations_validated",
        "metrics_replayed_match", "raw_hashes_match",
        "first_restart_implementation_hashes_match", "initial_failure",
    ]
    _expect(sorted(verification), sorted(verification_keys),
            "summary.verification.keys")
    _expect(verification["independent_decision_rows_validated"],
            evidence["decision_rows"],
            "summary.verification.independent_decision_rows_validated")
    _expect(verification["independent_dev_generations_validated"],
            evidence["dev_generation_rows"],
            "summary.verification.independent_dev_generations_validated")
    _expect(verification["independent_heldout_generations_validated"],
            evidence["heldout_generation_rows"],
            "summary.verification.independent_heldout_generations_validated")
    _expect(verification["metrics_replayed_match"], True,
            "summary.verification.metrics_replayed_match")
    _expect(verification["raw_hashes_match"], True,
            "summary.verification.raw_hashes_match")
    _expect(verification["first_restart_implementation_hashes_match"], True,
            "summary.verification.first_restart_implementation_hashes_match")
    _expect_true(isinstance(verification["focused_tests"], int)
                 and verification["focused_tests"] >= 0,
                 "summary.verification.focused_tests",
                 "focused test count is not a non-negative integer")
    _expect_true(isinstance(verification["initial_failure"], str)
                 and verification["initial_failure"],
                 "summary.verification.initial_failure",
                 "initial failure note is not a non-empty string")
    not_replayed.extend(["verification.focused_tests",
                         "verification.initial_failure"])

    non_claims = summary.get("non_claims") or []
    _expect_true(isinstance(non_claims, list) and non_claims
                 and all(isinstance(item, str) and item
                         for item in non_claims),
                 "summary.non_claims",
                 "non_claims is not a non-empty list of strings")
    return not_replayed


# ---------------------------------------------------------------------------
# Full replay
# ---------------------------------------------------------------------------

def _lazy_imports():
    """Import the existing CPU scoring utilities only when replaying."""
    from replica_lab.memory_bank_readout_v2 import (
        _format_block,
        build_readout_split_decisions,
        decision_counts,
        evaluate_literal_gate,
        evaluate_phase_gate,
        evaluate_restart_accounting,
        evaluate_state_gates,
        freeze_decisions,
        load_prepared,
        load_role_labels,
        query_text_register,
        restart_query_accounting,
        score_literal_values,
        score_phase,
        validate_decision_counts,
        validate_generation_rows,
        verify_role_manifest,
    )
    return {
        "build_readout_split_decisions": build_readout_split_decisions,
        "decision_counts": decision_counts,
        "evaluate_literal_gate": evaluate_literal_gate,
        "evaluate_phase_gate": evaluate_phase_gate,
        "evaluate_restart_accounting": evaluate_restart_accounting,
        "evaluate_state_gates": evaluate_state_gates,
        "freeze_decisions": freeze_decisions,
        "load_prepared": load_prepared,
        "load_role_labels": load_role_labels,
        "query_text_register": query_text_register,
        "restart_query_accounting": restart_query_accounting,
        "score_literal_values": score_literal_values,
        "score_phase": score_phase,
        "validate_decision_counts": validate_decision_counts,
        "validate_generation_rows": validate_generation_rows,
        "verify_role_manifest": verify_role_manifest,
        "_format_block": _format_block,
    }


def _load_guard(run_dir: Path) -> dict | None:
    path = run_dir.parent / f"guard_{run_dir.name}.json"
    if not path.is_file():
        return None
    doc = _read_json(path, "guards")
    guard = doc.get("guard") or {}
    _expect(doc.get("status"), "completed_within_declared_sampled_limits",
            "guards.status")
    _expect(doc.get("child_exit_code"), 0, "guards.child_exit_code")
    _expect(Path(str(doc.get("output"))).resolve(), run_dir.resolve(),
            "guards.output")
    _expect(guard.get("stop_reason"), None, "guards.stop_reason")
    return doc


def _validate_baseline(report: Mapping[str, Any],
                       config: Mapping[str, Any],
                       source_root: Path) -> dict:
    baseline = report.get("baseline") or {}
    config_baseline = config.get("baseline") or {}
    for label in ("first_process", "restart"):
        if label not in baseline:
            continue
        item = baseline[label] or {}
        pin = config_baseline.get(label) or {}
        checkpoint = item.get("checkpoint") or {}
        report_file = item.get("report") or {}
        ledger = item.get("ledger") or {}
        _expect(checkpoint.get("sha256"), pin.get("checkpoint_sha256"),
                f"baseline.{label}.checkpoint.sha256")
        _expect(report_file.get("sha256"), pin.get("report_sha256"),
                f"baseline.{label}.report.sha256")
        _expect(ledger.get("sha256"), pin.get("ledger_sha256"),
                f"baseline.{label}.ledger.sha256")
        _expect(checkpoint.get("written"), False,
                f"baseline.{label}.checkpoint.written")
        _expect(checkpoint.get("loaded"), True,
                f"baseline.{label}.checkpoint.loaded")
        if label == "restart":
            _expect(checkpoint.get("loaded_sha256"),
                    (config_baseline.get("first_process") or {}).get(
                        "checkpoint_sha256"),
                    "baseline.restart.checkpoint.loaded_sha256")
        for name, declared in (
                ("checkpoint", checkpoint), ("report", report_file),
                ("ledger", ledger)):
            path = _resolve(declared.get("path") or "", source_root)
            _expect_true(path.is_file(), f"baseline.{label}.{name}.path",
                         f"pinned baseline file is missing: {path}")
            _expect(sha256_file(path), declared.get("sha256"),
                    f"baseline.{label}.{name}.file_sha256")
    return baseline


def _validate_heldout_structure(
        report: Mapping[str, Any],
        heldout_rows: Sequence[Mapping[str, Any]],
        heldout_decisions: Sequence[Mapping[str, Any]],
        *, raw_rows_sha: str, module: Mapping[str, Any]) -> dict:
    heldout = report["gating"]["heldout"]
    _expect(heldout.get("status"), "evaluated", "heldout.status")
    _expect(heldout.get("evaluated_after_dev_pass"), True,
            "heldout.evaluated_after_dev_pass")
    _expect(heldout.get("reason"), None, "heldout.reason")
    _expect(heldout.get("fields"), 0, "heldout.fields")
    _expect(heldout.get("decisions"), len(heldout_decisions),
            "heldout.decisions")
    _expect(heldout.get("generations"), len(heldout_rows),
            "heldout.generations")
    _expect(heldout.get("raw_rows"), raw_rows_sha, "heldout.raw_rows")
    _expect(dict(heldout.get("gate") or {}), HELDOUT_GATE, "heldout.gate")

    metrics = heldout.get("metrics") or {}
    _expect(sorted(metrics), sorted([
        "overall", "by_phase", "by_state", "correction", "unaffected",
        "unknown", "revoked", "format", "counts",
    ]), "heldout.metrics.keys")
    counts = metrics.get("counts") or {}
    phases = Counter(row["phase"] for row in heldout_rows)
    states = Counter(row["state"] for row in heldout_rows)
    _expect(counts.get("rows"), len(heldout_rows), "heldout.counts.rows")
    _expect(counts.get("episodes"),
            len({row["episode_id"] for row in heldout_rows}),
            "heldout.counts.episodes")
    _expect(counts.get("states"),
            len({(row["episode_id"], row["state"]) for row in heldout_rows}),
            "heldout.counts.states")
    _expect(sorted(metrics["by_phase"]), sorted(phases),
            "heldout.metrics.by_phase.keys")
    _expect(sorted(metrics["by_state"]), sorted(states),
            "heldout.metrics.by_state.keys")
    for phase, count in phases.items():
        _expect(metrics["by_phase"][phase]["n"], count,
                f"heldout.by_phase.{phase}.n")
    for state, count in states.items():
        _expect(metrics["by_state"][state]["n"], count,
                f"heldout.by_state.{state}.n")

    for name in ("overall", "correction", "unaffected", "unknown", "revoked"):
        _validate_block_arithmetic(metrics[name], f"heldout.metrics.{name}")
    for name, block in metrics["by_phase"].items():
        _validate_block_arithmetic(block, f"heldout.by_phase.{name}")
    for name, block in metrics["by_state"].items():
        _validate_block_arithmetic(block, f"heldout.by_state.{name}")
    _expect(sum(block["correct_n"] for block in metrics["by_phase"].values()),
            metrics["overall"]["correct_n"], "heldout.by_phase.correct_sum")
    _expect(sum(block["n"] for block in metrics["by_phase"].values()),
            metrics["overall"]["n"], "heldout.by_phase.n_sum")

    # Format facts are label-free and can be fully recomputed.
    format_block = module["_format_block"](heldout_rows)
    _expect(dict(metrics["format"]), format_block, "heldout.metrics.format")
    _expect(metrics["overall"]["n"], len(heldout_rows), "heldout.overall.n")
    _expect(metrics["overall"]["false_value_n"], 0,
            "heldout.overall.false_value_n")
    return {"metrics": metrics, "phases": dict(phases),
            "states": dict(states), "format": format_block}


def replay_evidence(*, root: Path = ROOT, source_root: Path | None = None,
                    first_dir: Path | None = None,
                    restart_dir: Path | None = None,
                    summary_path: Path | None = None,
                    config_path: Path | None = None,
                    prepared_root: Path | None = None,
                    roles_root: Path | None = None) -> dict:
    """Run the complete replay and return a passed receipt.

    Raises :class:`ReplayError` on the first mismatch.  The exception carries
    the evidence accumulated before the failure.
    """
    root = Path(root).resolve()
    source_root = Path(source_root or root).resolve()
    first_dir = Path(first_dir or root / FIRST_DIR).resolve()
    restart_dir = Path(restart_dir or root / RESTART_DIR).resolve()
    summary_path = Path(summary_path or root / SUMMARY_PATH).resolve()
    config_path = Path(config_path or source_root / CONFIG_PATH).resolve()
    prepared_root = Path(prepared_root or source_root / PREPARED_PATH).resolve()
    roles_root = Path(roles_root or source_root / ROLES_PATH).resolve()

    first_path = first_dir / "report.json"
    restart_path = restart_dir / "report.json"
    evidence: dict[str, Any] = {}
    checks: list[str] = []

    def done(name: str) -> None:
        checks.append(name)

    try:
        first = _read_json(first_path, "report.first")
        restart = _read_json(restart_path, "report.restart")
        summary = _read_json(summary_path, "summary")
        config = _read_json(config_path, "config")

        evidence["inputs"] = {
            "root": _relative(root, root),
            "source_root": _relative(source_root, root),
            "first": _relative(first_dir, root),
            "restart": _relative(restart_dir, root),
            "summary": _relative(summary_path, root),
            "config": _relative(config_path, root),
            "prepared": _relative(prepared_root, root),
            "roles": _relative(roles_root, root),
            "first_sha256": sha256_file(first_path),
            "restart_sha256": sha256_file(restart_path),
            "summary_sha256": sha256_file(summary_path),
            "config_sha256": sha256_file(config_path),
        }
        evidence["checks"] = checks
        done("inputs.read")

        # -- Report identity, pins and classification ---------------------
        _expect(first.get("format"), REPORT_FORMAT, "report.first.format")
        _expect(restart.get("format"), REPORT_FORMAT,
                "report.restart.format")
        _expect(first.get("job"), "first", "report.first.job")
        _expect(restart.get("job"), "restart", "report.restart.job")
        done("report.formats")

        for label, report in (("first", first), ("restart", restart)):
            validate_classification(report)
            _expect(dict(report.get("roles") or {}),
                    dict(config.get("roles") or {}),
                    f"report.{label}.roles")
            _expect(list(report.get("limitations") or []),
                    list(config.get("limitations") or []),
                    f"report.{label}.limitations")
            validate_memory_disabled(report, config)
            validate_zero_updates(report, config)
        done("report.classification")
        done("report.memory_disabled")
        done("report.zero_updates")

        first_protocol = first["protocol"]
        restart_protocol = restart["protocol"]
        _expect(first_protocol.get("config_sha256"),
                sha256_file(config_path), "protocol.config_sha256")
        _expect(restart_protocol.get("config_sha256"),
                sha256_file(config_path), "protocol.config_sha256")
        expected_rule = sha_json({
            "bank": config["bank"],
            "reader": config["reader"],
            "endpoint": config["endpoint"],
            "generation": config["generation"],
            "jobs": config["jobs"],
        })
        for label, protocol in (("first", first_protocol),
                                ("restart", restart_protocol)):
            _expect(protocol.get("decision_rule_sha256"), expected_rule,
                    f"protocol.{label}.decision_rule_sha256")
            _expect(dict(protocol.get("thresholds") or {}),
                    dict(config["endpoint"]),
                    f"protocol.{label}.thresholds")
            _expect(dict(protocol.get("expected_dev_counts") or {}),
                    dict(config["endpoint"]["expected_dev_counts"]),
                    f"protocol.{label}.expected_dev_counts")
        _expect(first_protocol.get("implementation_sha256"),
                restart_protocol.get("implementation_sha256"),
                "protocol.implementation_sha256")
        done("report.protocol_pins")

        _expect(list(first["gating"].get("order") or []),
                GATING_ORDER["first"], "report.first.gating.order")
        _expect(list(restart["gating"].get("order") or []),
                GATING_ORDER["restart"], "report.restart.gating.order")
        _expect(first["gating"].get("stopped_before_heldout"), True,
                "report.first.stopped_before_heldout")
        _expect(first["gating"].get("stopped_before_model"), False,
                "report.first.stopped_before_model")
        _expect(first["gating"].get("stopped_before_restart"), False,
                "report.first.stopped_before_restart")
        for key in ("stopped_before_heldout", "stopped_before_model",
                    "stopped_before_restart"):
            _expect(restart["gating"].get(key), False,
                    f"report.restart.{key}")
        _expect(dict(first.get("role_label_access") or {}),
                {"opened": ["dev"], "heldout_opened": False},
                "report.first.role_label_access")
        _expect(dict(restart.get("role_label_access") or {}),
                {"opened": ["dev", "heldout", "train"],
                 "heldout_opened": True},
                "report.restart.role_label_access")
        for label, report, expected_outcome in (
                ("first", first, FIRST_OUTCOME),
                ("restart", restart, RESTART_OUTCOME)):
            outcome = dict(report.get("outcome") or {})
            for key, value in expected_outcome.items():
                _expect(outcome.get(key), value,
                        f"report.{label}.outcome.{key}")
            _expect_true(isinstance(outcome.get("claim"), str)
                         and outcome["claim"],
                         f"report.{label}.outcome.claim",
                         "claim is not a non-empty string")
        done("report.outcomes")

        # -- Raw outputs and artifact accounting --------------------------
        first_raw = validate_raw_manifest(first, first_dir, config)
        restart_raw = validate_raw_manifest(restart, restart_dir, config)
        done("raw.manifests")

        first_artifact_bytes = sum(
            path.stat().st_size for path in first_dir.iterdir()
            if path.is_file())
        restart_artifact_bytes = sum(
            path.stat().st_size for path in restart_dir.iterdir()
            if path.is_file())
        _expect(first.get("artifact_bytes"), first_artifact_bytes,
                "report.first.artifact_bytes")
        _expect(restart.get("artifact_bytes"), restart_artifact_bytes,
                "report.restart.artifact_bytes")
        evidence["artifact_bytes"] = {
            "first": first_artifact_bytes,
            "restart": restart_artifact_bytes,
        }
        done("raw.artifact_bytes")

        # -- Prepared inputs, roles and decisions -------------------------
        module = _lazy_imports()
        prepared = module["load_prepared"](config, prepared_root)
        role_manifest = module["verify_role_manifest"](roles_root, config)
        prepared_block = first_protocol.get("prepared") or {}
        _expect(prepared_block.get("hashes"), prepared["hashes"],
                "protocol.prepared.hashes")
        _expect(prepared_block.get("manifest"), prepared["manifest"],
                "protocol.prepared.manifest")
        _expect(prepared_block.get("episodes"), len(prepared["episodes"]),
                "protocol.prepared.episodes")
        _expect(_resolve(str(prepared_block.get("root") or ""),
                         source_root).resolve(),
                Path(prepared["root"]).resolve(),
                "protocol.prepared.root")
        _expect(dict(prepared_block.get("role_labels") or {}),
                role_manifest, "protocol.prepared.role_labels")
        done("prepared.pins")

        # Only the development labels are opened.  No heldout label read.
        dev_labels = module["load_role_labels"](roles_root, "dev", config)
        episodes = prepared["episodes"]
        revocations = prepared["revocations"]
        query_texts = module["query_text_register"](episodes)
        frozen = {}
        for split in ("dev", "heldout", "train"):
            rows = module["build_readout_split_decisions"](
                episodes, revocations, config, split)
            frozen[split] = module["freeze_decisions"](rows, split=split)

        expected_decision_files = {
            "dev": [first_dir / "decisions_dev.jsonl",
                    restart_dir / "decisions_dev.jsonl"],
            "heldout": [restart_dir / "decisions_heldout.jsonl"],
            "train": [restart_dir / "decisions_train.jsonl"],
        }
        for split, paths in expected_decision_files.items():
            for path in paths:
                _expect_true(path.is_file(),
                             f"decisions.{split}.file",
                             f"missing raw decision file: {path}")
                _expect(sha256_file(path), frozen[split].sha256,
                        f"decisions.{split}.sha256")
                _expect(path.read_bytes(), frozen[split].data,
                        f"decisions.{split}.bytes")

        all_decision_rows = tuple(
            row for split in ("dev", "heldout", "train")
            for row in frozen[split].decisions)
        identity_check = validate_decision_identities(
            all_decision_rows, "decisions.all")
        _expect(identity_check["rows"], 456, "decisions.total_rows")
        _expect(identity_check["unique_ids"], 456, "decisions.unique_ids")
        _expect(identity_check["unique_keys"], 456, "decisions.unique_keys")
        evidence["decision_rows"] = identity_check["rows"]
        evidence["decision_hashes"] = {
            split: frozen[split].sha256 for split in frozen}
        done("decisions.rebuilt")

        # -- Report count blocks ------------------------------------------
        dev_phase_counts = Counter(row["phase"]
                                   for row in frozen["dev"].decisions)
        _expect(dict(first["counts"].get("by_phase") or {}),
                {phase: dev_phase_counts[phase] for phase in
                 ("first_process", "restart", "revocation")},
                "report.first.counts.by_phase")
        _expect(first["counts"].get("decisions"), len(frozen["dev"].decisions),
                "report.first.counts.decisions")
        _expect(dict(first["counts"].get("decision_keys") or {}),
                {"rows": len(frozen["dev"].decisions),
                 "unique_keys": len(frozen["dev"].decisions),
                 "unique": True},
                "report.first.counts.decision_keys")

        _expect(restart["counts"].get("decisions"), 456,
                "report.restart.counts.decisions")
        _expect(dict(restart["counts"].get("by_split") or {}),
                {split: len(frozen[split].decisions)
                 for split in ("dev", "heldout", "train")},
                "report.restart.counts.by_split")
        _expect(dict(restart["counts"].get("decision_keys") or {}),
                {"rows": 456, "unique_keys": 456, "unique": True},
                "report.restart.counts.decision_keys")
        done("reports.counts")

        # -- Development gates --------------------------------------------
        literal_metrics = module["score_literal_values"](
            frozen["dev"], dev_labels)["splits"]["dev"]
        literal_gate = module["evaluate_literal_gate"](literal_metrics, config)
        _expect(dict(first["literal_metrics"]["dev"]), literal_metrics,
                "report.first.literal_metrics.dev")
        _expect(dict(first["gating"]["literal_dev"]), literal_gate,
                "report.first.gating.literal_dev")
        _expect(literal_gate.get("passed"), True,
                "literal_dev.passed")
        evidence["literal_gate"] = literal_gate
        evidence["literal_metrics"] = literal_metrics
        done("replay.literal_dev")

        dev_counts = module["decision_counts"](frozen["dev"].decisions,
                                               dev_labels)
        module["validate_decision_counts"](dev_counts, config)
        _expect(dict(first["counts"].get("dev_denominators") or {}),
                dev_counts, "report.first.counts.dev_denominators")
        evidence["dev_denominators"] = dev_counts
        done("replay.dev_denominators")

        def generation_rows(path: Path, check: str) -> tuple[dict, ...]:
            return _read_jsonl(path, check)

        first_rows = generation_rows(
            first_dir / "generations_dev_first_process.jsonl",
            "raw.first.first_process")
        restart_rows = generation_rows(
            restart_dir / "generations_dev_restart.jsonl",
            "raw.restart.restart")
        revocation_rows = generation_rows(
            restart_dir / "generations_dev_revocation.jsonl",
            "raw.restart.revocation")
        heldout_rows = generation_rows(
            restart_dir / "generations_heldout.jsonl",
            "raw.restart.heldout")

        phase_rows = {
            "first_process": first_rows,
            "restart": restart_rows,
            "revocation": revocation_rows,
        }
        expected_phase_counts = dict(dev_phase_counts)
        _expect({phase: len(rows) for phase, rows in phase_rows.items()},
                expected_phase_counts, "generations.dev_phase_counts")

        phase_metrics = {}
        phase_gates = {}
        for phase, rows in phase_rows.items():
            decisions = [row for row in frozen["dev"].decisions
                         if row["phase"] == phase]
            module["validate_generation_rows"](rows, decisions, query_texts)
            metrics = module["score_phase"](
                rows, dev_labels, frozen["dev"].decisions, phase)
            gate = module["evaluate_phase_gate"](metrics, config)
            report_block = first if phase == "first_process" else restart
            _expect(dict(report_block["gating"][phase]["metrics"]), metrics,
                    f"report.{report_block['job']}.gating.{phase}.metrics")
            _expect(dict(report_block["gating"][phase]["gate"]), gate,
                    f"report.{report_block['job']}.gating.{phase}.gate")
            _expect(gate.get("passed"), True, f"gate.{phase}.passed")
            phase_metrics[phase] = metrics
            phase_gates[phase] = gate
        evidence["first_process"] = {
            "metrics": phase_metrics["first_process"],
            "gate": phase_gates["first_process"],
        }
        evidence["restart"] = {
            "metrics": phase_metrics["restart"],
            "gate": phase_gates["restart"],
        }
        evidence["revocation"] = {
            "metrics": phase_metrics["revocation"],
            "gate": phase_gates["revocation"],
        }
        done("replay.dev_phases")

        _expect(dict(restart["gating"]["first_process"]["metrics"]),
                phase_metrics["first_process"],
                "report.restart.gating.first_process.metrics")
        _expect(dict(restart["gating"]["first_process"]["gate"]),
                phase_gates["first_process"],
                "report.restart.gating.first_process.gate")
        _expect(restart["gating"]["first_process"].get("status"),
                "evaluated_prior",
                "report.restart.gating.first_process.status")
        done("restart.first_process_receipt")

        _expect(first["counts"].get("generations"),
                {"first_process": len(first_rows)},
                "report.first.counts.generations")
        restart_dev_rows = len(restart_rows) + len(revocation_rows)
        _expect(restart["counts"].get("generations"),
                {"dev": restart_dev_rows,
                 "heldout": len(heldout_rows),
                 "train": 0},
                "report.restart.counts.generations")
        done("reports.generation_counts")

        dev_generation_rows = (len(first_rows) + len(restart_rows)
                               + len(revocation_rows))
        _expect(dev_generation_rows, 152, "replay.dev_generation_rows")
        _expect(len(heldout_rows), 152, "replay.heldout_generation_rows")
        evidence["dev_generation_rows"] = dev_generation_rows
        evidence["heldout_generation_rows"] = len(heldout_rows)
        evidence["generation_counts"] = {
            "first_process": len(first_rows),
            "restart": len(restart_rows),
            "revocation": len(revocation_rows),
            "dev": dev_generation_rows,
            "heldout": len(heldout_rows),
        }
        done("replay.generation_counts")

        # -- First receipt identity ---------------------------------------
        first_receipt = restart["gating"].get("first_receipt") or {}
        _expect(first_receipt.get("status"), "verified",
                "restart.first_receipt.status")
        _expect(first_receipt.get("first_process_gate_passed"), True,
                "restart.first_receipt.first_process_gate_passed")
        _expect(first_receipt.get("generations"), len(first_rows),
                "restart.first_receipt.generations")
        _expect(first_receipt.get("report_sha256"),
                sha256_file(first_path), "restart.first_receipt.report_sha256")
        _expect(first_receipt.get("decision_sha256"),
                frozen["dev"].sha256, "restart.first_receipt.decision_sha256")
        recorded_first = Path(str(first_receipt.get("report_path")))
        if recorded_first.exists():
            _expect(recorded_first.resolve(), first_path,
                    "restart.first_receipt.report_path")
        else:
            _expect(recorded_first.name, "report.json",
                    "restart.first_receipt.report_path")
        done("restart.first_receipt")

        # -- Combined state gate and restart accounting -------------------
        combined_dev_rows = first_rows + restart_rows
        state_gate = module["evaluate_state_gates"](
            combined_dev_rows, dev_labels, frozen["dev"].decisions, config)
        _expect(dict(restart["gating"]["state_denominators"]["gate"]),
                state_gate, "report.restart.gating.state_denominators.gate")
        _expect(state_gate.get("passed"), True, "state_denominators.passed")
        evidence["state_gate"] = state_gate
        done("replay.state_denominators")

        baseline = _validate_baseline(restart, config, source_root)
        first_baseline = _validate_baseline(first, config, source_root)
        _expect(first_baseline.get("first_process"),
                baseline.get("first_process"),
                "baseline.first_restart_match")
        baseline_restart = baseline.get("restart") or {}
        encoded = baseline_restart.get("encoded") or {}
        accounting = module["restart_query_accounting"](episodes, "dev")
        accounting["old_observations_encoded"] = int(
            encoded.get("old_observations_reencoded", -1))
        accounting["baseline_restart_new_queries"] = int(
            encoded.get("restart_new_queries", -1))
        accounting["old_first_process_queries_regenerated"] = sum(
            1 for row in restart_rows + revocation_rows
            if row["phase"] == "first_process")
        accounting_gate = module["evaluate_restart_accounting"](
            accounting, config)
        _expect(dict(restart["gating"]["restart_accounting"]["metrics"]),
                accounting,
                "report.restart.gating.restart_accounting.metrics")
        _expect(dict(restart["gating"]["restart_accounting"]["gate"]),
                accounting_gate,
                "report.restart.gating.restart_accounting.gate")
        _expect(accounting_gate.get("passed"), True,
                "restart_accounting.passed")
        evidence["accounting"] = accounting
        evidence["accounting_gate"] = accounting_gate
        done("replay.restart_accounting")

        # -- Heldout counts, structure and label-free format --------------
        module["validate_generation_rows"](
            heldout_rows, frozen["heldout"].decisions, query_texts)
        heldout = _validate_heldout_structure(
            restart, heldout_rows, frozen["heldout"].decisions,
            raw_rows_sha=sha256_file(
                restart_dir / "generations_heldout.jsonl"),
            module=module)
        evidence["heldout"] = heldout
        done("replay.heldout_structure")

        _expect(restart["gating"]["train"].get("decisions"),
                len(frozen["train"].decisions),
                "report.restart.gating.train.decisions")
        _expect(restart["gating"]["train"].get("status"),
                "descriptive_only", "report.restart.gating.train.status")
        _expect(restart["gating"]["train"].get("role"),
                "descriptive_only", "report.restart.gating.train.role")
        _expect(restart["gating"]["train"].get("model_generation"),
                "not_run_descriptive_only",
                "report.restart.gating.train.model_generation")
        _expect(len(frozen["train"].decisions), 152,
                "replay.train_decisions")
        done("replay.train")

        # -- Resource usage and all-row format facts ----------------------
        job_rows = {"first": first_rows,
                    "restart": restart_rows + revocation_rows + heldout_rows}
        usage = {}
        for label, report in (("first", first), ("restart", restart)):
            usage[label] = check_resource_usage(
                report, job_rows[label], config)
        _expect(usage["first"]["generation_rows"], len(first_rows),
                "resource.first.rows")
        _expect(usage["restart"]["generation_rows"],
                len(restart_rows) + len(revocation_rows) + len(heldout_rows),
                "resource.restart.rows")
        evidence["resource_usage"] = usage
        evidence["resource_totals"] = {
            key: usage["first"][key] + usage["restart"][key]
            for key in ZERO_RESOURCE_KEYS
        }
        for key in ZERO_RESOURCE_KEYS:
            _expect(evidence["resource_totals"][key], 0,
                    f"resource.total.{key}")

        all_rows = first_rows + restart_rows + revocation_rows + heldout_rows
        format_block = module["_format_block"](all_rows)
        evidence["generation_format"] = format_block
        _expect(format_block["cap_hit_n"], 0,
                "generation_format.cap_hit_n")
        _expect(format_block["single_line_rate"], 1.0,
                "generation_format.single_line_rate")
        _expect(format_block["ended_by_eos_rate"], 1.0,
                "generation_format.ended_by_eos_rate")
        evidence["control_false_values"] = sum(
            block["control"]["false_value_n"]
            for block in phase_metrics.values()) + \
            heldout["metrics"]["overall"]["false_value_n"]
        _expect(evidence["control_false_values"], 0,
                "generation_format.control_false_values")

        profiles = [first.get("profile") or {}, restart.get("profile") or {}]
        for profile in profiles:
            _expect(profile.get("errors"), [], "profile.errors")
            _expect_true(isinstance(profile.get("sample_count"), int)
                         and profile["sample_count"] > 0,
                         "profile.sample_count",
                         "sample count is not positive")
        peak_driver = max(
            int((profile.get("memory") or {}).get(
                "mps_driver_peak_sampled_bytes", -1))
            for profile in profiles)
        _expect(peak_driver, int(summary["resources"]["peak_driver_bytes"]),
                "profile.peak_driver_bytes")
        evidence["profile"] = {"peak_driver_bytes": peak_driver}
        done("resources.usage")

        guards = {
            "first": _load_guard(first_dir),
            "restart": _load_guard(restart_dir),
        }
        evidence["guards"] = guards
        if guards["first"] is not None and guards["restart"] is not None:
            for label, guard, profile in zip(
                    ("first", "restart"),
                    (guards["first"], guards["restart"]), profiles):
                _expect(guard["guard"].get("peak_driver_bytes"),
                        profile.get("memory", {}).get(
                            "mps_driver_peak_sampled_bytes"),
                        f"guards.{label}.peak_driver_bytes")
            done("resources.guards")

        # -- Summary ------------------------------------------------------
        first_relative = _relative(first_dir, root)
        restart_relative = _relative(restart_dir, root)
        not_replayed = validate_summary(
            summary, {"first": first, "restart": restart}, evidence,
            root=root, first_relative=first_relative,
            restart_relative=restart_relative)
        done("summary.claims")

        not_replayed.append(
            "heldout label-dependent correctness (labels were not reopened; "
            "counts, arithmetic and format were replayed)")
        receipt = {
            "format": RECEIPT_FORMAT,
            "passed": True,
            "native_execution": False,
            "heldout_labels_read": False,
            "heldout_rescored": False,
            "checks": checks,
            "inputs": evidence["inputs"],
            "evidence": {key: value for key, value in evidence.items()
                         if key not in ("inputs", "checks")},
            "not_independently_replayed": not_replayed,
        }
        return receipt
    except ReplayError as error:
        error.evidence = dict(evidence)
        raise
    except (KeyError, TypeError, IndexError) as error:
        wrapped = ReplayError(
            "evidence.schema",
            f"missing or invalid evidence field: {error!r}")
        wrapped.evidence = dict(evidence)
        raise wrapped from error


def _failure_receipt(error: ReplayError) -> dict:
    evidence = dict(error.evidence)
    return {
        "format": RECEIPT_FORMAT,
        "passed": False,
        "native_execution": False,
        "heldout_labels_read": False,
        "heldout_rescored": False,
        "check": error.check,
        "detail": error.detail,
        "checks": list(evidence.pop("checks", [])),
        "inputs": evidence.pop("inputs", {}),
        "evidence": evidence,
    }


def _guard_receipt_path(receipt: Path, first_dir: Path,
                        restart_dir: Path, summary_path: Path) -> Path:
    resolved = Path(receipt).resolve()
    for artifact in (first_dir, restart_dir):
        if resolved == artifact.resolve() or resolved.is_relative_to(
                artifact.resolve()):
            raise ReplayError(
                "receipt.path",
                f"receipt may not be written inside raw output directory: "
                f"{artifact}")
    if resolved == summary_path.resolve():
        raise ReplayError("receipt.path",
                          "receipt may not overwrite the summary")
    return resolved


def _write_receipt(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True,
                               allow_nan=False) + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Deterministic, fail-closed H1 readout evidence replay")
    parser.add_argument("--root", type=Path, default=ROOT,
                        help="artifact root (default: repository root)")
    parser.add_argument("--source-root", type=Path, default=None,
                        help="root for pinned prepared/config/baseline paths")
    parser.add_argument("--first", type=Path, default=None,
                        help=f"first02 directory (default: {FIRST_DIR})")
    parser.add_argument("--restart", type=Path, default=None,
                        help=f"restart02 directory (default: {RESTART_DIR})")
    parser.add_argument("--summary", type=Path, default=None,
                        help=f"H1 summary (default: {SUMMARY_PATH})")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--prepared", type=Path, default=None)
    parser.add_argument("--roles", type=Path, default=None)
    parser.add_argument("--receipt", type=Path, required=True,
                        help="fresh receipt path (deterministic JSON)")
    args = parser.parse_args(argv)

    root = args.root.resolve()
    first_dir = (args.first or root / FIRST_DIR).resolve()
    restart_dir = (args.restart or root / RESTART_DIR).resolve()
    summary_path = (args.summary or root / SUMMARY_PATH).resolve()
    try:
        receipt_path = _guard_receipt_path(
            args.receipt, first_dir, restart_dir, summary_path)
    except ReplayError as error:
        print(json.dumps({
            "receipt": None,
            "passed": False,
            "check": error.check,
            "detail": error.detail,
        }, sort_keys=True))
        return 2
    try:
        receipt = replay_evidence(
            root=root, source_root=args.source_root,
            first_dir=first_dir, restart_dir=restart_dir,
            summary_path=summary_path, config_path=args.config,
            prepared_root=args.prepared, roles_root=args.roles)
        _write_receipt(receipt_path, receipt)
        print(json.dumps({
            "receipt": str(receipt_path),
            "passed": True,
            "dev_generation_rows": receipt["evidence"]["dev_generation_rows"],
            "heldout_generation_rows":
                receipt["evidence"]["heldout_generation_rows"],
            "decision_rows": receipt["evidence"]["decision_rows"],
        }, sort_keys=True))
        return 0
    except ReplayError as error:
        _write_receipt(receipt_path, _failure_receipt(error))
        print(json.dumps({
            "receipt": str(receipt_path),
            "passed": False,
            "check": error.check,
            "detail": error.detail,
        }, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

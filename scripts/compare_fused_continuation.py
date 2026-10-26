#!/usr/bin/env python3
"""Compare uninterrupted A/A variation with uninterrupted-versus-resumed A/B.

All model execution stays outside this CPU tool. Exact checks are never relaxed:
``exact_status`` remains failed when parameter hashes or logits bytes differ.
A separately predeclared ``aa-envelope-v1`` policy may describe whether an A/B
delta is bracketed by measured uninterrupted variation, but it never changes the
exact result or a historical receipt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from replica_lab.training_state import compare_training_fingerprints  # noqa: E402
from replica_lab.continuation_fixture import (  # noqa: E402
    accounting_gates,
    policy_digest,
    require_fresh_output,
    validate_fixture_protocol,
)


UPDATE_FIELDS = (
    "step", "loss", "lr_factor", "learning_rates",
    "gradient_norm_before_clipping", "processed_tokens", "supervised_tokens",
)
NORM_PATHS = tuple(
    f"{section}.{metric}"
    for section in ("parameter_norms", "gradient_norms_pre_clip",
                    "optimizer_moment_norms")
    for metric in ("l1_norm", "l2_norm", "max_abs"))
RESUME_ONLY_IMPLEMENTATION_PATHS = frozenset({"src/replica_lab/training_state.py"})
AA_ENVELOPE_POLICY = {
    "name": "aa-envelope-v1",
    "declared": True,
    "rule": ("an A/B absolute delta is within the envelope only when it is less "
             "than or equal to the largest pairwise A/A delta for the same "
             "update and metric from independent uninterrupted arms"),
    "requires": [
        "at least two uninterrupted arms with identical protocol and implementation",
        "no differing implementation path outside resume-only code",
        "update 1 exactly equal across all arms",
        "first/resume boundary fingerprints exactly equal when present",
    ],
    "applies_to": ["scalar loss", "gradient norm", "output loss",
                   "aggregate and per-tensor norm summaries"],
    "excludes": ["exact parameter value hashes", "exact logits bytes"],
    "does_not_alter": ["exact_status", "historical receipts", "probe behavior"],
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def protocol_binding(protocol_path, *, root=None):
    """Bind a protocol file to its hash, tensor count and declared policy.

    Fixture protocols additionally require the exact boundary and
    direct-target-device accounting gates. A production protocol keeps the
    historical defaults and cannot acquire fixture gates by accident.
    """
    path = Path(protocol_path)
    protocol = load(path)
    protocol_sha = sha(path)
    if protocol.get("production_scope") is not False:
        return {
            "format": "replica.fused_continuation_binding.v1",
            "production_scope": True,
            "protocol_sha256": protocol_sha,
            "expected_tensor_count": protocol.get("expected_tensor_count"),
            "expected_production_scope": True,
            "require_boundary_fingerprint": False,
            "require_direct_restore_accounting": False,
            "aa_envelope_policy_sha256": None,
            "claim_boundary": None,
            "fixture": None,
        }
    fixture = validate_fixture_protocol(protocol, protocol_path=path, root=root)
    digest = fixture["protocol_policy_sha256"]
    if digest != policy_digest(AA_ENVELOPE_POLICY):
        raise ValueError(
            "Fixture protocol AA envelope policy differs from this comparator")
    return {
        "format": "replica.fused_continuation_fixture.binding.v1",
        "production_scope": False,
        "protocol_sha256": protocol_sha,
        "expected_tensor_count": fixture["tensor_count"],
        "expected_production_scope": False,
        "require_boundary_fingerprint": True,
        "require_direct_restore_accounting": True,
        "aa_envelope_policy_sha256": digest,
        "claim_boundary": fixture["claim_boundary"],
        "fixture": {
            "config_sha256": fixture["config_sha256"],
            "tensor_count": fixture["tensor_count"],
            "parameter_count": fixture["parameter_count"],
            "init_seed": fixture["init_seed"],
            "tokenizer_lock_sha256": fixture["tokenizer_lock_sha256"],
        },
    }


def binding_gates(binding):
    """Translate a validated protocol binding into ``compare_reports`` gates."""
    if binding.get("production_scope") is False:
        return {
            "expected_tensor_count": binding["expected_tensor_count"],
            "expected_production_scope": False,
            "expected_protocol_sha256": binding["protocol_sha256"],
            "expected_policy_sha256": binding["aa_envelope_policy_sha256"],
            "require_boundary_fingerprint": True,
            "require_direct_restore_accounting": True,
            "minimum_uninterrupted_arms": 2,
        }
    gates = {
        "require_boundary_fingerprint": False,
        "require_direct_restore_accounting": False,
    }
    if binding.get("expected_tensor_count") is not None:
        gates["expected_tensor_count"] = binding["expected_tensor_count"]
    return gates


def _nested(value, path):
    current = value
    for part in path.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def _finite(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(float(value)))


def _scalar_delta(left, right):
    if not (_finite(left) and _finite(right)):
        return None
    return abs(float(left) - float(right))


def _update_map(report):
    return {int(update["step"]): update for update in report.get("updates", [])
            if isinstance(update, dict) and "step" in update}


def _update_metric_delta(left, right, path):
    return _scalar_delta(_nested(left, path), _nested(right, path))


def _fingerprint_entries(fingerprint):
    """Flatten parameter, buffer and optimizer-state tensor summaries."""
    entries = {}
    for name, entry in ((fingerprint or {}).get("parameters") or {}).items():
        entries[f"parameters.{name}"] = entry
    for name, entry in (((fingerprint or {}).get("buffers") or {})
                        .get("buffers") or {}).items():
        entries[f"buffers.{name}"] = entry
    optimizer = (fingerprint or {}).get("optimizer") or {}
    for name, state in (optimizer.get("state") or {}).items():
        for key, entry in state.items():
            if isinstance(entry, dict) and "l2_norm" in entry:
                entries[f"optimizer.{name}.{key}"] = entry
    return entries


def _fingerprint_pair_delta(left, right):
    left_entries = _fingerprint_entries(left)
    right_entries = _fingerprint_entries(right)
    result = {
        "tensor_paths": len(set(left_entries) | set(right_entries)),
        "missing_paths": sorted(set(left_entries) - set(right_entries))[:16],
        "extra_paths": sorted(set(right_entries) - set(left_entries))[:16],
        "hash_differences": 0,
        "metadata_differences": 0,
        "max_l2_norm_delta": 0.0,
        "max_relative_l2_delta": 0.0,
        "worst_paths": [],
    }
    worst = []
    for path in sorted(set(left_entries) & set(right_entries)):
        left_entry, right_entry = left_entries[path], right_entries[path]
        if (left_entry.get("value_sha256") is not None
                and right_entry.get("value_sha256") is not None
                and left_entry.get("value_sha256") != right_entry.get("value_sha256")):
            result["hash_differences"] += 1
        if any(left_entry.get(field) != right_entry.get(field)
               for field in ("dtype", "device", "shape", "storage_bytes",
                             "storage_offset", "stride", "is_contiguous")):
            result["metadata_differences"] += 1
        left_l2, right_l2 = left_entry.get("l2_norm"), right_entry.get("l2_norm")
        if _finite(left_l2) and _finite(right_l2):
            delta = abs(float(left_l2) - float(right_l2))
            relative = delta / abs(float(left_l2)) if float(left_l2) else (
                0.0 if delta == 0.0 else None)
            result["max_l2_norm_delta"] = max(result["max_l2_norm_delta"], delta)
            if relative is not None:
                result["max_relative_l2_delta"] = max(
                    result["max_relative_l2_delta"], relative)
            worst.append((delta, path, relative))
    result["worst_paths"] = [
        {"path": path, "l2_norm_delta": delta, "relative_l2_delta": relative}
        for delta, path, relative in sorted(worst, reverse=True)[:8]]
    return result


def _fingerprint_envelope(reports):
    pairs = []
    for index, left in enumerate(reports):
        for right in reports[index + 1:]:
            left_fingerprint = left.get("final_fingerprint")
            right_fingerprint = right.get("final_fingerprint")
            if left_fingerprint is None or right_fingerprint is None:
                continue
            pairs.append(_fingerprint_pair_delta(left_fingerprint, right_fingerprint))
    if not pairs:
        return {"available": False, "reason": "final fingerprints absent"}
    return {
        "available": True,
        "pair_count": len(pairs),
        "max_l2_norm_delta": max(pair["max_l2_norm_delta"] for pair in pairs),
        "max_relative_l2_delta": max(pair["max_relative_l2_delta"] for pair in pairs),
        "max_hash_differences": max(pair["hash_differences"] for pair in pairs),
        "pairs": pairs,
    }


def _implementation_differences(reports):
    baseline = reports[0].get("implementation_sha256") or {}
    paths = set()
    for report in reports:
        implementation = report.get("implementation_sha256") or {}
        for path in set(baseline) | set(implementation):
            if baseline.get(path) != implementation.get(path):
                paths.add(path)
    return sorted(paths)


def _execution_implementation_identical(reports):
    return not (set(_implementation_differences(reports))
                - RESUME_ONLY_IMPLEMENTATION_PATHS)


def _aa_variation(reports):
    result = {
        "available": len(reports) >= 2,
        "arm_count": len(reports),
        "pair_count": 0,
        "protocol_identical": len({report.get("protocol_sha256")
                                   for report in reports}) == 1,
        "implementation_identical": not _implementation_differences(reports),
        "execution_implementation_identical": _execution_implementation_identical(reports),
        "implementation_differences": _implementation_differences(reports),
        "implementation_note": (
            "training_state.py contains the resume-materialization path; the "
            "continuous arms do not execute that path, so equal non-resume code "
            "is required for A/A variation."),
        "updates": {},
        "output_loss": None,
        "earliest_divergence": None,
        "fingerprints": _fingerprint_envelope(reports),
    }
    if len(reports) < 2:
        result["reason"] = "at least two uninterrupted arms are required"
        return result
    if not result["protocol_identical"] or not result["execution_implementation_identical"]:
        result["available"] = False
        result["reason"] = ("uninterrupted arms differ in protocol or "
                            "non-resume execution code")
        return result
    maps = [_update_map(report) for report in reports]
    step_numbers = sorted(set().union(*[set(mapping) for mapping in maps]))
    earliest = None
    for step in step_numbers:
        fields = {}
        for path in ("loss", "gradient_norm_before_clipping", *NORM_PATHS):
            values = []
            for mapping in maps:
                update = mapping.get(step)
                value = _nested(update, path) if update is not None else None
                if _finite(value):
                    values.append(float(value))
            if len(values) < 2:
                continue
            deltas = [abs(left - right) for index, left in enumerate(values)
                      for right in values[index + 1:]]
            maximum = max(deltas)
            fields[path] = {
                "min": min(values),
                "max": max(values),
                "pairwise_max_abs_delta": maximum,
            }
            if maximum > 0.0 and earliest is None:
                earliest = {"step": step, "field": path,
                            "pairwise_max_abs_delta": maximum,
                            "values": values}
        result["updates"][str(step)] = fields
    result["pair_count"] = len(reports) * (len(reports) - 1) // 2
    output_losses = [report.get("output_fingerprint", {}).get("loss")
                     for report in reports]
    output_losses = [float(value) for value in output_losses if _finite(value)]
    if len(output_losses) >= 2:
        deltas = [abs(left - right) for index, left in enumerate(output_losses)
                  for right in output_losses[index + 1:]]
        result["output_loss"] = {
            "min": min(output_losses),
            "max": max(output_losses),
            "pairwise_max_abs_delta": max(deltas),
        }
    hash_counts = []
    for index, left in enumerate(reports):
        for right in reports[index + 1:]:
            left_hashes = left.get("parameter_hashes_after") or {}
            right_hashes = right.get("parameter_hashes_after") or {}
            hash_counts.append(sum(
                1 for name in set(left_hashes) | set(right_hashes)
                if left_hashes.get(name) != right_hashes.get(name)))
    if hash_counts:
        result["parameter_hash_differences"] = {
            "pairwise_counts": hash_counts,
            "min": min(hash_counts),
            "max": max(hash_counts),
            "tensor_count": len(reports[0].get("parameter_hashes_after") or {}),
        }
    logits_hashes = [((report.get("output_fingerprint") or {}).get("logits_sha256"))
                     for report in reports]
    logits_hashes = [value for value in logits_hashes if value is not None]
    if logits_hashes:
        result["unique_output_logits_sha256"] = len(set(logits_hashes))
    result["earliest_divergence"] = earliest
    return result


def _ab_variation(reference, resume, aa):
    reference_updates = _update_map(reference)
    resume_updates = _update_map(resume)
    result = {
        "reference_execution_mode": reference.get("execution_mode"),
        "updates": {},
        "output_loss": None,
        "parameter_hash_differences": None,
        "aa_parameter_hash_differences": None,
        "exact_logits_byte_match": reference.get("output_fingerprint") == resume.get("output_fingerprint"),
        "earliest_divergence": None,
        "within_aa_envelope": None,
        "exceeding_fields": [],
    }
    step_numbers = sorted(set(reference_updates) & set(resume_updates))
    for step in step_numbers:
        fields = {}
        for path in ("loss", "gradient_norm_before_clipping", *NORM_PATHS):
            delta = _update_metric_delta(reference_updates[step], resume_updates[step], path)
            if delta is None:
                continue
            envelope = (aa.get("updates", {}).get(str(step), {})
                        .get(path, {}).get("pairwise_max_abs_delta"))
            within = envelope is not None and delta <= envelope
            fields[path] = {
                "delta": delta,
                "aa_pairwise_max_abs_delta": envelope,
                "within_aa_envelope": within,
            }
            if delta > 0.0 and result["earliest_divergence"] is None:
                result["earliest_divergence"] = {
                    "step": step, "field": path, "delta": delta,
                    "aa_pairwise_max_abs_delta": envelope, "within_aa_envelope": within,
                }
            if envelope is not None and not within:
                result["exceeding_fields"].append(f"update_{step}.{path}")
        result["updates"][str(step)] = fields
    output_delta = _scalar_delta(reference.get("output_fingerprint", {}).get("loss"),
                                 resume.get("output_fingerprint", {}).get("loss"))
    output_envelope = ((aa.get("output_loss") or {}).get("pairwise_max_abs_delta"))
    if output_delta is not None:
        result["output_loss"] = {
            "delta": output_delta,
            "aa_pairwise_max_abs_delta": output_envelope,
            "within_aa_envelope": output_envelope is not None and output_delta <= output_envelope,
        }
        if not result["output_loss"]["within_aa_envelope"]:
            result["exceeding_fields"].append("output_fingerprint.loss")
    reference_hashes = reference.get("parameter_hashes_after") or {}
    resume_hashes = resume.get("parameter_hashes_after") or {}
    result["parameter_hash_differences"] = sum(
        1 for name in set(reference_hashes) | set(resume_hashes)
        if reference_hashes.get(name) != resume_hashes.get(name))
    if aa.get("fingerprints", {}).get("max_hash_differences") is not None:
        result["aa_parameter_hash_differences"] = aa["fingerprints"]["max_hash_differences"]
    elif (aa.get("parameter_hash_differences") or {}).get("max") is not None:
        result["aa_parameter_hash_differences"] = aa["parameter_hash_differences"]["max"]
    result["within_aa_envelope"] = not result["exceeding_fields"]
    return result


def _exact_update_differences(reference, actual):
    differences = []
    left, right = _update_map(reference), _update_map(actual)
    if sorted(left) != sorted(right):
        differences.append("update_count")
        return differences
    for step in sorted(left):
        for field in UPDATE_FIELDS:
            if left[step].get(field) != right[step].get(field):
                differences.append(f"update_{step}.{field}")
    return differences


def _guard_summary(name, guard):
    policy = guard.get("guard", {}).get("policy", {})
    return {
        "arm": name,
        "status": guard.get("status"),
        "stop_reason": guard.get("guard", {}).get("stop_reason"),
        "peak_driver_bytes": guard.get("guard", {}).get("peak_driver_bytes"),
        "peak_swap_growth_bytes": guard.get("guard", {}).get("peak_swap_growth_bytes"),
        "minimum_observed_free_percent": guard.get("guard", {}).get(
            "minimum_observed_free_percent"),
        "driver_limit_bytes": policy.get("driver_limit_bytes"),
        "swap_growth_limit_bytes": policy.get("swap_growth_limit_bytes"),
        "minimum_free_percent": policy.get("minimum_free_percent"),
        "elapsed_seconds": guard.get("elapsed_seconds"),
    }


def _guard_checks(name, guard, checks, failures):
    policy = guard.get("guard", {}).get("policy", {})
    checks[f"{name}_guard_completed"] = (
        guard.get("status") == "completed_within_declared_sampled_limits")
    checks[f"{name}_guard_not_stopped"] = (
        guard.get("guard", {}).get("stop_reason") is None)
    driver = guard.get("guard", {}).get("peak_driver_bytes")
    swap = guard.get("guard", {}).get("peak_swap_growth_bytes")
    free = guard.get("guard", {}).get("minimum_observed_free_percent")
    checks[f"{name}_driver_within_limit"] = (
        driver is not None and driver <= policy.get("driver_limit_bytes", -1))
    checks[f"{name}_swap_within_limit"] = (
        swap is not None and swap <= policy.get("swap_growth_limit_bytes", -1))
    checks[f"{name}_free_within_limit"] = (
        free is not None and free >= policy.get("minimum_free_percent", 101))
    for key in (f"{name}_guard_completed", f"{name}_guard_not_stopped",
                f"{name}_driver_within_limit", f"{name}_swap_within_limit",
                f"{name}_free_within_limit"):
        if not checks[key]:
            failures.append(key)


def _accounting_summary(resume, resume_guard):
    accounting = resume.get("resume_accounting") or {}
    optimizer = accounting.get("optimizer") or {}
    staged = optimizer.get("staged") or {}
    restored = optimizer.get("restored") or {}
    migration = optimizer.get("migration") or {}
    model = accounting.get("model") or {}
    guard = resume_guard.get("guard", {})
    strategy = optimizer.get("restore_strategy")
    if strategy is None:
        # Reports written before direct restore either loaded optimizer state
        # directly on CPU or staged it on CPU and then migrated it explicitly.
        if migration:
            strategy = "cpu_staged_manual_migration"
        elif staged:
            strategy = "cpu_staged"
    return {
        "available": bool(accounting),
        "path": accounting.get("path"),
        "full_cpu_model_copy": accounting.get("full_cpu_model_copy"),
        "model_bytes": (model.get("loaded") or {}).get("bytes"),
        "temporary_checkpoint_tensor_bytes": model.get(
            "temporary_checkpoint_tensor_bytes"),
        "checkpoint_tensors_adopted_by_model": model.get(
            "checkpoint_tensors_adopted_by_model"),
        "optimizer_restore_strategy": strategy,
        "optimizer_restore_device": optimizer.get("target_device",
                                                   optimizer.get("staged_device")),
        "staged_cpu_optimizer_bytes": staged.get("bytes"),
        "staged_cpu_optimizer_tensors": staged.get("tensor_count"),
        "staged_cpu_optimizer_bytes_by_dtype": staged.get("bytes_by_dtype"),
        "restored_optimizer_bytes": restored.get("bytes"),
        "restored_optimizer_tensors": restored.get("tensor_count"),
        "restored_optimizer_bytes_by_dtype": restored.get("bytes_by_dtype"),
        "restored_optimizer_device": optimizer.get("restored_device"),
        "optimizer_migration_applied": bool(migration),
        "migrated_tensor_count": migration.get("migrated_tensor_count"),
        "migrated_bytes": migration.get("migrated_bytes"),
        "max_single_tensor_bytes": migration.get("max_single_tensor_bytes"),
        "migration_peak_driver_bytes": migration.get("peak_driver_bytes"),
        "swap_used_bytes_start": accounting.get("swap_used_bytes_start"),
        "swap_used_bytes_end": accounting.get("swap_used_bytes_end"),
        "swap_growth_bytes": accounting.get("swap_growth_bytes"),
        "guard_swap_growth_bytes": guard.get("peak_swap_growth_bytes"),
        "guard_stop_reason": guard.get("stop_reason"),
        "resident_estimate": accounting.get("resident_estimate"),
        "note": ("Restored optimizer bytes are device-resident state; direct "
                 "restore stages no additional CPU optimizer copy. Staged CPU "
                 "bytes from historical reports and device model bytes overlap "
                 "in unified memory; the guard reading is the stop-policy "
                 "evidence."),
    }


def compare_reports(uninterrupted, first, resume, *,
                    uninterrupted_guards=(), first_guard=None, resume_guard=None,
                    policy="none", reference_index=0, evidence=None,
                    expected_tensor_count=381,
                    expected_production_scope=None,
                    expected_protocol_sha256=None,
                    expected_policy_sha256=None,
                    require_boundary_fingerprint=False,
                    require_direct_restore_accounting=False,
                    minimum_uninterrupted_arms=1,
                    claim_boundary=None):
    uninterrupted = list(uninterrupted)
    if not uninterrupted:
        raise ValueError("At least one uninterrupted report is required")
    if not 0 <= reference_index < len(uninterrupted):
        raise ValueError("reference_index is outside the uninterrupted arms")
    reference = uninterrupted[reference_index]
    checks = {}
    failures = []

    def check(name, value):
        checks[name] = bool(value)
        if not value:
            failures.append(name)

    def check_report_identity(name, report):
        if expected_production_scope is not None:
            check(f"{name}_production_scope",
                  report.get("production_scope") is expected_production_scope)
        if expected_policy_sha256 is not None:
            check(f"{name}_policy_hash_bound",
                  report.get("protocol_policy_sha256") == expected_policy_sha256)

    if len(uninterrupted) < minimum_uninterrupted_arms:
        checks["uninterrupted_arm_count_sufficient"] = False
        failures.append("uninterrupted_arm_count_sufficient")
    else:
        checks["uninterrupted_arm_count_sufficient"] = True

    for index, report in enumerate(uninterrupted):
        check(f"uninterrupted_{index}_completed", report.get("status") == "completed")
        check(f"uninterrupted_{index}_execution_mode",
              report.get("execution_mode") == "continuous")
        check(f"uninterrupted_{index}_changed_tensors",
              report.get("changed_tensors") == expected_tensor_count)
        check(f"uninterrupted_{index}_not_exported",
              report.get("model_exported") is False)
        check_report_identity(f"uninterrupted_{index}", report)
    for name, report, mode in (("interrupted_first", first, "first"),
                               ("interrupted_resume", resume, "resume")):
        check(f"{name}_completed", report.get("status") == "completed")
        check(f"{name}_execution_mode", report.get("execution_mode") == mode)
        check(f"{name}_changed_tensors",
              report.get("changed_tensors") == expected_tensor_count)
        check(f"{name}_not_exported", report.get("model_exported") is False)
        check_report_identity(name, report)

    all_reports = uninterrupted + [first, resume]
    check("protocol_identical",
          len({report.get("protocol_sha256") for report in all_reports}) == 1)
    if expected_protocol_sha256 is not None:
        check("protocol_hash_matches_declared",
              all(report.get("protocol_sha256") == expected_protocol_sha256
                  for report in all_reports))
    check("implementation_identical",
          len({json.dumps(report.get("implementation_sha256"), sort_keys=True)
               for report in all_reports}) == 1)
    initial = uninterrupted[0].get("parameter_hashes_before")
    check("initial_weights_identical",
          all(report.get("parameter_hashes_before") == initial
              for report in uninterrupted + [first]))
    check("resume_start_matches_first",
          first.get("parameter_hashes_after") == resume.get("parameter_hashes_before"))
    check("resume_checkpoint_receipt_validated",
          resume.get("checkpoint_reload_validated") is True)
    check("resume_report_hash_validated",
          resume.get("resume_start_matches_first_process") is True)
    check("first_checkpoint_only",
          first.get("checkpoint") is not None
          and all(report.get("checkpoint") is None for report in uninterrupted)
          and resume.get("checkpoint") is None)
    check("first_step_one",
          first.get("steps") == 1 and first.get("start_step") == 0
          and first.get("optimizer_state_step_values") == [1.0])
    check("uninterrupted_two_steps",
          all(report.get("steps") == 2 and report.get("start_step") == 0
              and report.get("optimizer_state_step_values") == [2.0]
              for report in uninterrupted))
    check("resume_second_step",
          resume.get("steps") == 2 and resume.get("start_step") == 1
          and resume.get("optimizer_state_step_values") == [2.0])

    final_differences = sorted(
        name for name in set(reference.get("parameter_hashes_after") or {})
        | set(resume.get("parameter_hashes_after") or {})
        if (reference.get("parameter_hashes_after") or {}).get(name)
        != (resume.get("parameter_hashes_after") or {}).get(name))
    check("final_parameter_hashes_identical", not final_differences)
    update_differences = _exact_update_differences(reference, resume)
    check("update_losses_rates_and_counts_identical", not update_differences)
    check("final_logits_and_loss_identical",
          reference.get("output_fingerprint") == resume.get("output_fingerprint"))

    boundary = None
    reference_boundary = first.get("boundary_fingerprint")
    resume_boundary = resume.get("boundary_fingerprint")
    if reference_boundary is not None and resume_boundary is not None:
        if require_boundary_fingerprint:
            check("boundary_fingerprint_present", True)
        boundary = compare_training_fingerprints(reference_boundary, resume_boundary)
        boundary["status"] = "matched" if boundary["matched"] else "mismatched"
        check("boundary_fingerprint_identical", boundary["matched"])
    else:
        boundary = {"status": "unavailable",
                    "reason": "first or resume report has no boundary fingerprint"}
        if require_boundary_fingerprint:
            check("boundary_fingerprint_present", False)
        else:
            checks["boundary_fingerprint_identical"] = None

    guard_summaries = []
    if len(uninterrupted_guards) != len(uninterrupted):
        failures.append("uninterrupted_guard_count_mismatch")
    for index, report in enumerate(uninterrupted):
        if index < len(uninterrupted_guards):
            name = f"uninterrupted_{index}"
            guard = uninterrupted_guards[index]
            _guard_checks(name, guard, checks, failures)
            guard_summaries.append(_guard_summary(name, guard))
    if first_guard is not None:
        _guard_checks("interrupted_first", first_guard, checks, failures)
        guard_summaries.append(_guard_summary("interrupted_first", first_guard))
    else:
        failures.append("interrupted_first_guard_missing")
    if resume_guard is not None:
        _guard_checks("interrupted_resume", resume_guard, checks, failures)
        guard_summaries.append(_guard_summary("interrupted_resume", resume_guard))
    else:
        failures.append("interrupted_resume_guard_missing")

    aa = _aa_variation(uninterrupted)
    ab = _ab_variation(reference, resume, aa)
    if reference.get("final_fingerprint") and resume.get("final_fingerprint"):
        final_pair = _fingerprint_pair_delta(
            reference["final_fingerprint"], resume["final_fingerprint"])
        ab["final_fingerprint"] = {"available": True, **final_pair}
        envelope = aa.get("fingerprints") or {}
        if envelope.get("available"):
            if (final_pair["max_l2_norm_delta"] > envelope["max_l2_norm_delta"]
                    or final_pair["metadata_differences"] > 0
                    or final_pair["missing_paths"] or final_pair["extra_paths"]):
                ab["within_aa_envelope"] = False
                ab["exceeding_fields"].append("final_fingerprint")
    else:
        ab["final_fingerprint"] = {
            "available": False,
            "reason": "final fingerprint absent from reference or resume report"}
    if boundary.get("matched") is False:
        ab["within_aa_envelope"] = False
        ab["exceeding_fields"].append("boundary_fingerprint")
    if aa.get("available") and aa.get("earliest_divergence") is None:
        update_one_exact = all(
            all(_scalar_delta(_nested(_update_map(report).get(1), path),
                              _nested(_update_map(reference).get(1), path)) == 0.0
                for path in ("loss", "gradient_norm_before_clipping"))
            for report in uninterrupted)
        if not update_one_exact:
            ab["within_aa_envelope"] = False
            ab["exceeding_fields"].append("uninterrupted_update_1_loss")

    accounting = _accounting_summary(resume, resume_guard or {})
    if require_direct_restore_accounting:
        gates = accounting_gates(resume.get("resume_accounting") or {})["gates"]
        for name, satisfied in gates.items():
            check(f"resume_accounting_{name}", satisfied)

    exact_status = "passed" if not failures else "failed"
    if not aa.get("available"):
        variation_status = "unqualified"
    elif not ab.get("within_aa_envelope", False):
        variation_status = "divergent"
    else:
        variation_status = "within_aa_envelope"
    policy_applied = policy == "aa-envelope-v1"
    if exact_status == "passed":
        status = "passed"
    elif policy_applied and variation_status == "within_aa_envelope":
        status = "failed_exact_within_declared_variation"
    else:
        status = "failed"

    return {
        "format": "replica.fused_adamw_continuation_comparison.receipt.v2",
        "status": status,
        "exact_status": exact_status,
        "variation_status": variation_status,
        "production_scope": expected_production_scope,
        "claim_boundary": claim_boundary,
        "scope_binding": {
            "expected_tensor_count": expected_tensor_count,
            "expected_production_scope": expected_production_scope,
            "expected_protocol_sha256": expected_protocol_sha256,
            "expected_policy_sha256": expected_policy_sha256,
            "rulebook_policy_sha256": policy_digest(AA_ENVELOPE_POLICY),
            "require_boundary_fingerprint": bool(require_boundary_fingerprint),
            "require_direct_restore_accounting": bool(
                require_direct_restore_accounting),
            "minimum_uninterrupted_arms": minimum_uninterrupted_arms,
        },
        "policy": {
            "name": policy,
            "applied": policy_applied,
            "definition": AA_ENVELOPE_POLICY if policy_applied else None,
            "definition_sha256": policy_digest(AA_ENVELOPE_POLICY),
            "bound_sha256": expected_policy_sha256,
            "does_not_relax_exact_status": True,
        },
        "checks": checks,
        "failures": failures,
        "final_parameter_hash_difference_count": len(final_differences),
        "final_parameter_hash_differences": final_differences,
        "update_differences": update_differences,
        "continuous_output_fingerprint": reference.get("output_fingerprint"),
        "resumed_output_fingerprint": resume.get("output_fingerprint"),
        "aa_variation": aa,
        "ab_variation": ab,
        "earliest_divergence": {
            "aa": aa.get("earliest_divergence"),
            "ab": ab.get("earliest_divergence"),
            "boundary": boundary.get("earliest_divergence"),
        },
        "boundary_fingerprint_comparison": boundary,
        "resume_accounting": accounting,
        "guard_summary": guard_summaries,
        "evidence": evidence or {},
        "scope": ("CPU comparison of raw native reports; exact checks remain "
                  "authoritative and no model/quality claim is made"),
        "historical_retry_policy": {
            "stopped": True,
            "reason": ("comparison09 already showed A/B deltas inside A/A "
                       "variation while exact hashes differed; repeating the "
                       "same native comparison is uninformative"),
            "preserved_failures": [
                "comparison01/05/09 exact hash, update-2 loss and logits mismatch",
                "comparison02/04/06/07/08/10/11 resume swap-growth stop",
                "comparison03 checkpoint tensor device comparison rejection",
            ],
        },
    }


def compare(continuous_paths, first_path, resume_path, *,
            continuous_guard_paths=(), first_guard_path=None,
            resume_guard_path=None, policy="none", reference_index=0, **gates):
    uninterrupted = [load(path) for path in continuous_paths]
    guards = [load(path) for path in continuous_guard_paths]
    evidence = {}
    for index, path in enumerate(continuous_paths):
        evidence[f"uninterrupted_{index}"] = {"path": str(path), "sha256": sha(path)}
    for index, path in enumerate(continuous_guard_paths):
        evidence[f"uninterrupted_{index}_guard"] = {
            "path": str(path), "sha256": sha(path)}
    evidence["interrupted_first"] = {"path": str(first_path), "sha256": sha(first_path)}
    evidence["interrupted_resume"] = {"path": str(resume_path), "sha256": sha(resume_path)}
    if first_guard_path is not None:
        evidence["interrupted_first_guard"] = {
            "path": str(first_guard_path), "sha256": sha(first_guard_path)}
    if resume_guard_path is not None:
        evidence["interrupted_resume_guard"] = {
            "path": str(resume_guard_path), "sha256": sha(resume_guard_path)}
    return compare_reports(
        uninterrupted, load(first_path), load(resume_path),
        uninterrupted_guards=guards,
        first_guard=(load(first_guard_path) if first_guard_path is not None else None),
        resume_guard=(load(resume_guard_path) if resume_guard_path is not None else None),
        policy=policy, reference_index=reference_index, evidence=evidence, **gates)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uninterrupted", type=Path, action="append", default=[],
                        help="Uninterrupted run_report.json; repeat for A/A arms")
    parser.add_argument("--continuous", type=Path, action="append", default=[],
                        help="Alias for --uninterrupted")
    parser.add_argument("--uninterrupted-guard", type=Path, action="append", default=[])
    parser.add_argument("--continuous-guard", type=Path, action="append", default=[],
                        help="Alias for --uninterrupted-guard")
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--resume", type=Path, required=True)
    parser.add_argument("--first-guard", type=Path, required=True)
    parser.add_argument("--resume-guard", type=Path, required=True)
    parser.add_argument("--reference-index", type=int, default=0)
    parser.add_argument("--variation-policy", choices=("none", "aa-envelope-v1"),
                        default="none")
    parser.add_argument("--protocol", type=Path,
                        help="Pinned protocol; derives tensor count, gates and policy hash")
    parser.add_argument("--expected-tensor-count", type=int)
    parser.add_argument("--require-boundary-fingerprint", action="store_true")
    parser.add_argument("--require-direct-restore-accounting", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    uninterrupted = args.uninterrupted + args.continuous
    uninterrupted_guards = args.uninterrupted_guard + args.continuous_guard
    if not uninterrupted:
        parser.error("At least one --uninterrupted report is required")
    try:
        require_fresh_output(args.output, what="comparison receipt")
    except FileExistsError as error:
        parser.error(str(error))
    gates = {
        "require_boundary_fingerprint": args.require_boundary_fingerprint,
        "require_direct_restore_accounting": args.require_direct_restore_accounting,
    }
    if args.expected_tensor_count is not None:
        gates["expected_tensor_count"] = args.expected_tensor_count
    policy = args.variation_policy
    binding = None
    if args.protocol is not None:
        binding = protocol_binding(args.protocol, root=ROOT)
        gates.update(binding_gates(binding))
        if binding.get("claim_boundary"):
            gates["claim_boundary"] = binding["claim_boundary"]
        if binding.get("production_scope") is False:
            if args.variation_policy != "aa-envelope-v1":
                parser.error(
                    "Fixture protocol requires --variation-policy aa-envelope-v1")
            policy = "aa-envelope-v1"
    result = compare(
        uninterrupted, args.first, args.resume,
        continuous_guard_paths=uninterrupted_guards,
        first_guard_path=args.first_guard,
        resume_guard_path=args.resume_guard,
        policy=policy,
        reference_index=args.reference_index, **gates)
    if binding is not None:
        result["evidence"]["protocol"] = {
            "path": str(args.protocol), "sha256": binding["protocol_sha256"]}
        result["protocol_binding"] = binding
    if args.output.exists():
        # A concurrent writer must not be overwritten.
        parser.error("Comparison receipt path already exists")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"],
                      "exact_status": result["exact_status"],
                      "variation_status": result["variation_status"],
                      "production_scope": result.get("production_scope"),
                      "failures": result["failures"],
                      "output": str(args.output)}, sort_keys=True))
    return 0 if result["exact_status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

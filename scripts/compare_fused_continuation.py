#!/usr/bin/env python3
"""Independently compare matched fused-AdamW continuation reports on CPU."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def compare(continuous_path, first_path, resume_path,
            continuous_guard_path, first_guard_path, resume_guard_path):
    continuous = load(continuous_path)
    first = load(first_path)
    resume = load(resume_path)
    guards = [load(path) for path in (continuous_guard_path, first_guard_path, resume_guard_path)]
    checks = {}
    failures = []

    def check(name, value):
        checks[name] = bool(value)
        if not value:
            failures.append(name)

    for name, report, mode in (("continuous", continuous, "continuous"),
                               ("interrupted_first", first, "first"),
                               ("interrupted_resume", resume, "resume")):
        check(f"{name}_completed", report.get("status") == "completed")
        check(f"{name}_execution_mode", report.get("execution_mode") == mode)
        check(f"{name}_changed_tensors", report.get("changed_tensors") == 381)
        check(f"{name}_not_exported", report.get("model_exported") is False)

    check("protocol_identical", len({r["protocol_sha256"] for r in (continuous, first, resume)}) == 1)
    check("implementation_identical", len({json.dumps(r["implementation_sha256"], sort_keys=True)
                                           for r in (continuous, first, resume)}) == 1)
    check("initial_weights_identical", continuous["parameter_hashes_before"] == first["parameter_hashes_before"])
    check("resume_start_matches_first", first["parameter_hashes_after"] == resume["parameter_hashes_before"])
    check("resume_checkpoint_receipt_validated", resume.get("checkpoint_reload_validated") is True)
    check("resume_report_hash_validated", resume.get("resume_start_matches_first_process") is True)
    check("first_checkpoint_only", first.get("checkpoint") is not None
          and continuous.get("checkpoint") is None and resume.get("checkpoint") is None)
    check("first_step_one", first.get("steps") == 1 and first.get("start_step") == 0
          and first.get("optimizer_state_step_values") == [1.0])
    check("continuous_two_steps", continuous.get("steps") == 2 and continuous.get("start_step") == 0
          and continuous.get("optimizer_state_step_values") == [2.0])
    check("resume_second_step", resume.get("steps") == 2 and resume.get("start_step") == 1
          and resume.get("optimizer_state_step_values") == [2.0])

    final_diffs = sorted(k for k in set(continuous["parameter_hashes_after"]) | set(resume["parameter_hashes_after"])
                         if continuous["parameter_hashes_after"].get(k) != resume["parameter_hashes_after"].get(k))
    check("final_parameter_hashes_identical", not final_diffs)

    update_fields = ("step", "loss", "lr_factor", "learning_rates", "gradient_norm_before_clipping",
                     "processed_tokens", "supervised_tokens")
    update_diffs = []
    if len(continuous.get("updates", [])) != 2 or len(resume.get("updates", [])) != 2:
        update_diffs.append("update_count")
    else:
        for index, (left, right) in enumerate(zip(continuous["updates"], resume["updates"]), 1):
            for field in update_fields:
                if left.get(field) != right.get(field):
                    update_diffs.append(f"update_{index}.{field}")
    check("update_losses_rates_and_counts_identical", not update_diffs)

    output_equal = continuous.get("output_fingerprint") == resume.get("output_fingerprint")
    check("final_logits_and_loss_identical", output_equal)

    guard_summaries = []
    for name, guard in zip(("continuous", "interrupted_first", "interrupted_resume"), guards):
        policy = guard["guard"]["policy"]
        check(f"{name}_guard_completed", guard.get("status") == "completed_within_declared_sampled_limits")
        check(f"{name}_guard_not_stopped", guard["guard"].get("stop_reason") is None)
        check(f"{name}_driver_within_limit", guard["guard"]["peak_driver_bytes"] <= policy["driver_limit_bytes"])
        check(f"{name}_swap_within_limit", guard["guard"]["peak_swap_growth_bytes"] <= policy["swap_growth_limit_bytes"])
        check(f"{name}_free_within_limit", guard["guard"]["minimum_observed_free_percent"] >= policy["minimum_free_percent"])
        guard_summaries.append({"arm": name, "peak_driver_bytes": guard["guard"]["peak_driver_bytes"],
                                "peak_swap_growth_bytes": guard["guard"]["peak_swap_growth_bytes"],
                                "minimum_observed_free_percent": guard["guard"]["minimum_observed_free_percent"],
                                "elapsed_seconds": guard["elapsed_seconds"]})

    evidence = {}
    for name, path in (("continuous", continuous_path), ("interrupted_first", first_path),
                       ("interrupted_resume", resume_path), ("continuous_guard", continuous_guard_path),
                       ("interrupted_first_guard", first_guard_path), ("interrupted_resume_guard", resume_guard_path)):
        evidence[name] = {"path": str(path), "sha256": sha(path)}
    return {
        "format": "replica.fused_adamw_continuation_comparison.receipt.v1",
        "status": "passed" if not failures else "failed",
        "checks": checks,
        "failures": failures,
        "final_parameter_hash_difference_count": len(final_diffs),
        "final_parameter_hash_differences": final_diffs,
        "update_differences": update_diffs,
        "continuous_output_fingerprint": continuous.get("output_fingerprint"),
        "resumed_output_fingerprint": resume.get("output_fingerprint"),
        "guard_summary": guard_summaries,
        "evidence": evidence,
        "scope": "CPU comparison of raw native reports; no model execution or quality claim",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("continuous", "first", "resume", "continuous-guard", "first-guard", "resume-guard"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a fresh comparison receipt path")
    result = compare(args.continuous, args.first, args.resume,
                     args.continuous_guard, args.first_guard, args.resume_guard)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "failures": result["failures"],
                      "output": str(args.output)}, sort_keys=True))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

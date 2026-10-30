#!/usr/bin/env python3
"""Bounded CPU measured-total-cost pilot for the fixed v12 density arms.

The pilot uses actual monotonic wall measurements of one durable admission/read
plus zero, one or four whole-pass training updates on independently rebuilt
random-fixture paths.  It fits measured fixed and marginal cost, applies the
frozen whole-pass selection rule from the preregistered protocol, then reruns
each arm at its selected exposure and reports whether it falls inside the
declared measured-cost tolerance.

No scored label is read, no budget is selected from answer quality, no model is
downloaded, and no native arm is launched.  The tiny-fixture timings are
engineering evidence, not Spark latency or physical-memory qualification.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import tempfile
import time

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from profile_intrinsic_memory_density_v12 import (  # noqa: E402
    ARMS, build_path, spans,
)
from replica_lab.contextual_spark_fixture import ids_and_mask, labels_from_ids
from replica_lab.intrinsic_memory_density_receipts_v12 import write_new_json  # noqa: E402
from replica_lab.intrinsic_memory_density_v12 import (  # noqa: E402
    DensityProducerBoundary, cost_snapshot, ledger_delta,
    require_timing_synchronization,
)
from replica_lab.intrinsic_memory_density_protocols_v12 import (  # noqa: E402
    canonical_digest, cost_tolerance, fit_fixed_marginal, load_protocols,
    measured_cost_target, select_updates,
)
from replica_lab.tiered_memory import (  # noqa: E402
    MemoryScope, TierCeilings, TieredMemoryStore,
)

FORMAT = "replica.intrinsic_memory.density.cost_pilot.v12"


def scope():
    return MemoryScope("density_cost_pilot", "main", "engineering", 1000)


def run_pass(arm, updates, *, profile=False):
    """Measure one admission/read plus ``updates`` training passes."""
    path = build_path(arm)
    device = next(path.encoder.parameters()).device
    profiler = None
    try:
        with tempfile.TemporaryDirectory(prefix="density-cost-pilot-") as temp:
            store = TieredMemoryStore(
                Path(temp), profile_id="P1", enforce_volume_free=False,
                ceilings=TierCeilings(
                    hot_bytes=32 * 1024**2, warm_bytes=32 * 1024**2,
                    cold_bytes=32 * 1024**2, identity_reserved_bytes=64 * 1024,
                    replay_reserved_bytes=256 * 1024, minimum_volume_free_bytes=0,
                    workspace_quota_bytes=1 * 1024**2, artifact_budget_bytes=4 * 1024**2,
                    frequent_reserve_fraction=0.5, frequent_min_accesses=2))
            try:
                boundary = DensityProducerBoundary(
                    path, comparison="fixed_per_site",
                    expected_groups=len(ARMS[arm]))
                profiler = (boundary.profiler(expected_groups=len(ARMS[arm]))
                            if profile else None)
                ids, mask = ids_and_mask(4)
                before = cost_snapshot(path)
                synchronized_before = require_timing_synchronization(device)
                started = time.perf_counter_ns()
                path.admit(store, spans()[0], scope=scope())
                batch = path.read(store, "Badge", scope=scope())
                for _ in range(updates):
                    path.optimizer.zero_grad(set_to_none=True)
                    live = path.training_from_stored(batch, scope=scope())
                    if profiler is None:
                        logits = path.forward(ids, mask, live, scope=scope())
                    else:
                        logits = profiler.forward(path, ids, mask, live, scope=scope())
                    loss = F.cross_entropy(
                        logits.flatten(0, 1), labels_from_ids(ids).flatten())
                    loss.backward()
                    if profiler is not None:
                        profiler.record_backward(calls=1,
                                                 processed_positions=int(ids.numel()))
                    path.optimizer.step()
                synchronized_after = require_timing_synchronization(device)
                elapsed = time.perf_counter_ns() - started
                after = cost_snapshot(path)
                result = {
                    "updates": updates,
                    "elapsed_ns": elapsed,
                    "timing_synchronization": {
                        "device": str(device),
                        "before": synchronized_before,
                        "after": synchronized_after,
                        "boundary": "synchronize_before_and_after_each_timed_pass",
                        "mps_fallback_setting": os.environ.get(
                            "PYTORCH_ENABLE_MPS_FALLBACK"),
                    },
                    "cost_delta": ledger_delta(before, after),
                }
                if profiler is not None:
                    result["profiler"] = profiler.summary()
                return result
            finally:
                if profiler is not None:
                    profiler.close()
                store.close()
    finally:
        path.close()


def arm_pilot(arm, update_counts, repeat):
    measurements = {}
    runs = {}
    for updates in update_counts:
        values = [run_pass(arm, updates)["elapsed_ns"] for _ in range(repeat)]
        runs[str(updates)] = values
        measurements[str(updates)] = statistics.median(values)
    fit = fit_fixed_marginal(measurements)
    diagnostic = run_pass(arm, 1, profile=True)
    component = diagnostic["cost_delta"]
    return {
        "arm": arm,
        "sites": list(ARMS[arm]),
        "update_counts": list(update_counts),
        "repeat": repeat,
        "pilot_runs_ns": runs,
        "median_total_cost_ns": measurements,
        "fit": fit,
        "component_measurement": {
            "pilot_cost_ns": diagnostic["elapsed_ns"],
            "source": component["source"],
            "memory": component["memory"],
            "reader": {
                "invocations": diagnostic["profiler"]["total_site_invocations"],
                "physical_groups": diagnostic["profiler"]["physical_groups"],
                "group_wall_ns": diagnostic["profiler"]["timing"]["group_wall_ns_total"],
                "dispatch_wall_ns": diagnostic["profiler"]["timing"]["dispatch_wall_ns_total"],
                "activation_forward_bytes": diagnostic["profiler"]["activation_bytes"],
            },
            "training": {
                "updates": 1,
                "optimizer_steps": 1,
                "backward_work": diagnostic["profiler"]["backward_work"],
                "wall_ns": diagnostic["elapsed_ns"],
            },
            "timing_synchronization": diagnostic["timing_synchronization"],
            "post_update_snapshot_copies": component["memory"]["snapshot_copies"],
            "cache_bytes": diagnostic["profiler"]["cache_bytes"],
            "wall_ns": diagnostic["elapsed_ns"],
        },
        "pilot_cost_ns": sum(sum(values) for values in runs.values())
                          + diagnostic["elapsed_ns"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "reports/intrinsic_memory_density_v12/measurements/cost_pilot_summary.json")
    args = parser.parse_args()
    torch.set_num_threads(1)
    protocols = load_protocols(ROOT / "configs")
    total_protocol = protocols["preregistered_measured_total_cost"]
    total_sha256 = canonical_digest(total_protocol)
    pilot_config = total_protocol["pilot"]
    selection_config = total_protocol["selection"]
    update_counts = tuple(pilot_config["runs_per_arm_update_counts"])
    repeat = pilot_config["repeat"]
    common_updates = selection_config["declared_common_updates"]
    updates_cap = selection_config["declared_updates_cap"]
    selection_epsilon = selection_config["epsilon"]
    relative_tolerance = selection_config["tolerance"]["relative"]
    arms = {arm: arm_pilot(arm, update_counts, repeat) for arm in ARMS}
    target_fit = arms["two"]["fit"]
    target_ns = measured_cost_target(target_fit, common_updates)
    selected = {}
    for arm, result in arms.items():
        updates = select_updates(result["fit"]["fixed_cost"],
                                 result["fit"]["marginal_cost"], target_ns,
                                 updates_cap=updates_cap,
                                 selection_epsilon=selection_epsilon)
        measured = [run_pass(arm, updates)["elapsed_ns"] for _ in range(repeat)]
        median = statistics.median(measured)
        tolerance = cost_tolerance(
            target_ns, result["fit"]["marginal_cost"],
            relative=relative_tolerance)
        difference = median - target_ns
        selected[arm] = {
            "selected_updates": updates,
            "measured_runs_ns": measured,
            "measured_total_cost_ns": median,
            "target_ns": target_ns,
            "tolerance_ns": tolerance,
            "difference_ns": difference,
            "matched_within_declared_tolerance": abs(difference) <= tolerance,
            "matched_within_relative_2_percent": abs(difference) <= target_ns * relative_tolerance,
            "measurement_cost_ns": sum(measured),
        }
        arms[arm]["selection"] = selected[arm]
        arms[arm]["total_charged_cost_ns"] = (
            arms[arm]["pilot_cost_ns"] + selected[arm]["measurement_cost_ns"])
    control_arm, control_updates = "all", updates_cap
    control_runs = [run_pass(control_arm, control_updates)["elapsed_ns"]
                    for _ in range(repeat)]
    control_median = statistics.median(control_runs)
    control_tolerance = cost_tolerance(
        target_ns, arms[control_arm]["fit"]["marginal_cost"],
        relative=relative_tolerance)
    control_difference = control_median - target_ns
    control_out = abs(control_difference) > control_tolerance
    arithmetic_candidate = target_ns + control_tolerance + 1.0
    out_of_tolerance_control = {
        "measured_high_exposure": {
            "arm": control_arm,
            "updates": control_updates,
            "measured_runs_ns": control_runs,
            "measured_total_cost_ns": control_median,
            "target_ns": target_ns,
            "tolerance_ns": control_tolerance,
            "difference_ns": control_difference,
            "outside_declared_tolerance": control_out,
        },
        "arithmetic_control": {
            "candidate_cost_ns": arithmetic_candidate,
            "target_ns": target_ns,
            "tolerance_ns": control_tolerance,
            "outside_declared_tolerance": (
                abs(arithmetic_candidate - target_ns) > control_tolerance),
        },
        "deliberate_out_of_tolerance_observed": control_out,
        "control_cost_ns": sum(control_runs),
        "charged_to_arm": control_arm,
    }
    arms[control_arm]["control_cost_ns"] = sum(control_runs)
    arms[control_arm]["total_charged_cost_ns"] += sum(control_runs)
    report = {
        "format": FORMAT,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "CPU_random_36_block_fixture_engineering_only",
        "native_execution": False,
        "native_weights_loaded": False,
        "downloads": False,
        "scored_labels_read": False,
        "P_updates": 0,
        "promotion": False,
        "python": sys.version,
        "torch": torch.__version__,
        "platform": platform.platform(),
        "cost_metric": "measured_monotonic_wall_nanoseconds",
        "timing_synchronization": {
            "rule": pilot_config["timing_rule"],
            "observed": arms["two"]["component_measurement"]["timing_synchronization"],
        },
        "repeat": repeat,
        "aggregate": pilot_config["aggregate"],
        "declared_common_updates": common_updates,
        "declared_updates_cap": updates_cap,
        "tolerance_rule": (
            f"max(target*{relative_tolerance}, one_measured_marginal_update_cost_arm)"),
        "target_arm": "two",
        "target_cost_ns": target_ns,
        "total_cost_protocol_sha256": total_sha256,
        "selection_rule": selection_config["rule"],
        "selected_updates": {arm: row["selected_updates"] for arm, row in selected.items()},
        "matched_arms": [arm for arm, row in selected.items()
                         if row["matched_within_declared_tolerance"]],
        "unmatched_arms": [arm for arm, row in selected.items()
                           if not row["matched_within_declared_tolerance"]],
        "arms": arms,
        "out_of_tolerance_control": out_of_tolerance_control,
        "total_charged_cost_ns": sum(
            row["total_charged_cost_ns"] for row in arms.values()),
        "protocol_digests": {name: canonical_digest(protocol)
                             for name, protocol in protocols.items()},
        "claim_limits": [
            "Tiny CPU fixture measurements are not native Spark latency or physical memory.",
            "Selection uses measured cost only; no scored labels were opened.",
            "An arm outside tolerance is unmatched evidence, not a causal density arm.",
            "Different whole-pass exposure is an explicit confound.",
        ],
    }
    write_new_json(args.output, report)
    print(json.dumps({key: report[key] for key in (
        "format", "scope", "declared_common_updates", "declared_updates_cap",
        "target_cost_ns", "selected_updates", "matched_arms", "unmatched_arms")},
        indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

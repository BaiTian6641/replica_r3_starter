#!/usr/bin/env python3
"""Integrated 2/9/36 CPU engineering diagnostic, never a native capability gate."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
import torch.nn.functional as F

from replica_lab.architecture.intrinsic_memory_layout_v4 import spark_layout
from replica_lab.contextual_memory import canonical_spans, tensor_digest
from replica_lab.contextual_memory_layout_v4 import ContextualMemoryLayoutV4
from replica_lab.contextual_spark_fixture import build_contextual_adapter, ids_and_mask, labels_from_ids
from replica_lab.profiling import _ru_maxrss_bytes
from replica_lab.tiered_memory import MemoryScope, TierCeilings, TieredMemoryStore

ARMS = ("two", "full_attention_nine", "all")
SCOPE = MemoryScope("fixture", "main", "assistant", as_of=100)


def make(arm):
    torch.manual_seed(17)
    adapter, *_ = build_contextual_adapter(seed=17, layers=36, spine_sites=(3, 7),
                                           spine_slots=8, memory_size=256)
    path = ContextualMemoryLayoutV4(adapter, spark_layout(arm, hidden_size=8),
                    source_identity="cpu_contextual_fixture_seed17", initialization_seed=17)
    adapter.extension.spine = None
    return path


def open_store(directory):
    return TieredMemoryStore(directory, ceilings=TierCeilings(
        hot_bytes=32*1024**2, warm_bytes=32*1024**2, cold_bytes=32*1024**2,
        identity_reserved_bytes=64*1024, replay_reserved_bytes=256*1024,
        workspace_quota_bytes=1024**2, artifact_budget_bytes=1024**2,
        minimum_volume_free_bytes=0), enforce_volume_free=False)


def span():
    return canonical_spans("Crimson badge means ready.", record_id="rule", source_id="fixture",
                           source_epoch=1, available_at=10)[0]


def step(path, batch):
    ids, mask = ids_and_mask(4)
    live = path.training_from_stored(batch, scope=SCOPE)
    path.optimizer.zero_grad(set_to_none=True)
    logits = path.forward(ids, mask, live, scope=SCOPE)
    loss = F.cross_entropy(logits.float().flatten(0, 1), labels_from_ids(ids).flatten())
    loss.backward()
    groups = {group["group_name"]: sum(float(p.grad.detach().abs().sum())
              for p in group["params"] if p.grad is not None) for group in path.optimizer.param_groups}
    readers = {site: float(port.query_projection.weight.grad.abs().sum())
               for site, port in path.reader.ports.items()}
    finite = all(torch.isfinite(p.grad).all() for group in path.optimizer.param_groups
                 for p in group["params"] if p.grad is not None)
    path.optimizer.step()
    return {"loss": float(loss.detach()), "gradient_l1": groups,
            "reader_query_gradient_l1": readers, "finite_gradients": bool(finite)}


def canonical_state(value):
    if torch.is_tensor(value):
        return {"dtype": str(value.dtype), "shape": list(value.shape), "sha256": tensor_digest(value)}
    if isinstance(value, dict):
        return {str(key): canonical_state(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [canonical_state(item) for item in value]
    return value


def state_identity(path):
    payload = {"weights": canonical_state(path._tensors()),
               "optimizer": canonical_state(path.optimizer.state_dict())}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def restart(directory, arm):
    start = time.monotonic()
    path = make(arm)
    path.load(directory / "extension")
    store = open_store(directory / "evaluation_store")
    batch = path.read(store, "badge", scope=SCOPE)
    ids, mask = ids_and_mask(4)
    with torch.no_grad():
        logits = path.forward(ids, mask, batch, scope=SCOPE)
    result = {"logits_sha256": tensor_digest(logits), "continuation": step(path, batch),
              "state_after_update_sha256": state_identity(path),
              "frozen_base_integrity": path.adapter.base_unchanged(),
              "cost": path.cost, "wall_seconds": time.monotonic()-start,
              "process_peak_rss_bytes": _ru_maxrss_bytes(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)}
    store.close()
    (directory / "restart.json").write_text(json.dumps(result, indent=2) + "\n")


def arm_diagnostic(output, arm):
    start = time.monotonic()
    directory = output / arm
    directory.mkdir()
    path = make(arm)
    ids, mask = ids_and_mask(4)
    training = open_store(directory / "training_store")
    assert path.admit(training, span(), scope=SCOPE).outcome.ok
    batch = path.read(training, "badge", scope=SCOPE)
    baseline = path.forward(ids, mask, None, scope=SCOPE)
    parity = torch.equal(baseline, path.forward(ids, mask, batch, scope=SCOPE))
    steps = [step(path, batch), step(path, batch)]
    training.close()
    # Fitting changes the producer release. A separate, explicitly charged
    # admission serializes that release for inference; no silent slot refresh.
    evaluation = open_store(directory / "evaluation_store")
    assert path.admit(evaluation, span(), scope=SCOPE).outcome.ok
    current = path.read(evaluation, "badge", scope=SCOPE)
    addresses = []
    handles = [port.register_forward_pre_hook(lambda module, args: addresses.append(args[1].data_ptr()))
               for port in path.reader.ports.values()]
    before = dict(path.cost)
    try:
        with torch.no_grad():
            logits = path.forward(ids, mask, current, scope=SCOPE)
    finally:
        for handle in handles:
            handle.remove()
    forward_cost = {key: path.cost[key]-value for key, value in before.items()}
    path.save(directory / "extension")
    expected_logits = tensor_digest(logits)
    continuation = step(path, current)
    expected_state = state_identity(path)
    evaluation.close()
    command = [sys.executable, str(Path(__file__).resolve()), "--output", str(directory),
               "--restart", "--arm", arm]
    child = subprocess.run(command, cwd=ROOT, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
                           capture_output=True, text=True, timeout=60)
    (directory / "child.stdout").write_text(child.stdout)
    (directory / "child.stderr").write_text(child.stderr)
    if child.returncode:
        raise RuntimeError(child.stderr)
    fresh = json.loads((directory / "restart.json").read_text())
    report = {"arm": arm, "sites": list(path.spec.sites), "zero_init_parity": parity,
              "fit_updates": 2, "fit": steps, "reference_continuation_updates": 1,
              "fresh_process_continuation_updates": 1, "reference_continuation": continuation,
              "producer_gradient_after_output_opens": steps[1]["gradient_l1"]["producer"] > 0,
              "all_active_query_gradients_positive": all(v > 0 for v in steps[1]["reader_query_gradient_l1"].values()),
              "one_optimizer_owner": bool(path.assert_optimizer_ownership()),
              "one_snapshot_storage_for_all_sites": len(set(addresses)) == 1,
              "forward_cost_delta": forward_cost, "cost": path.cost,
              "fresh_process_exact_logits": fresh["logits_sha256"] == expected_logits,
              "fresh_process_exact_weights_and_optimizer_continuation": fresh["state_after_update_sha256"] == expected_state,
              "frozen_base_integrity": path.adapter.base_unchanged() and fresh["frozen_base_integrity"],
              "artifact_bytes": sum(file.stat().st_size for file in directory.rglob("*") if file.is_file()),
              "wall_seconds": time.monotonic()-start,
              "parent_cumulative_peak_rss_bytes": _ru_maxrss_bytes(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
              "child_process_peak_rss_bytes": fresh["process_peak_rss_bytes"]}
    report["passed"] = all(report[key] for key in (
        "zero_init_parity", "producer_gradient_after_output_opens", "all_active_query_gradients_positive",
        "one_optimizer_owner", "one_snapshot_storage_for_all_sites", "fresh_process_exact_logits",
        "fresh_process_exact_weights_and_optimizer_continuation", "frozen_base_integrity")) and all(
            forward_cost[key] == 0 for key in ("retrieval_calls", "admission_calls", "correction_calls"))
    (directory / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--arm", choices=ARMS)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.restart:
        if args.arm is None:
            parser.error("restart requires --arm")
        restart(args.output.resolve(), args.arm)
        return
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    reports = [arm_diagnostic(output, arm) for arm in (ARMS if args.arm is None else (args.arm,))]
    result = {"format": "replica.intrinsic_memory.contextual_training_cpu_diagnostics.v4",
              "evidence_class": "CPU_engineering_not_native_competence_or_useful_recall",
              "native_execution": False, "persistent_P_updates": 0, "external_spine_present": False,
              "arms": reports, "passed": all(arm["passed"] for arm in reports),
              "wall_seconds": time.monotonic()-start,
              "artifact_bytes": sum(file.stat().st_size for file in output.rglob("*") if file.is_file()),
              "protocol_sha256": hashlib.sha256((ROOT / "configs/contextual_memory_layer_training_protocol_v4.json").read_bytes()).hexdigest()}
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"passed": result["passed"], "arms": len(reports), "wall_seconds": result["wall_seconds"],
                      "artifact_bytes": result["artifact_bytes"]}))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""CPU-only density engineering profile for the reviewed v12 producer.

This command builds the real v12 producer, feature provider and tiered store on
the random 36-block CPU fixture.  It records per-site invocations, physical group
callbacks, parameter/optimizer ownership, store costs and a fresh-process
checkpoint continuation for every fixed per-site arm.  It never loads Spark
weights, downloads data, launches native work or claims competence/latency.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from replica_lab.architecture.intrinsic_memory_layout_v4 import spark_layout
from replica_lab.contextual_memory import canonical_spans
from replica_lab.contextual_memory_v12 import ContextualMemoryV12
from replica_lab.contextual_source_features_v12 import (
    FrozenSourceFeatures, SourceFeatureSpec, adapter_contract_digest,
    tokenizer_contract_digest,
)
from replica_lab.contextual_spark_fixture import (
    build_contextual_adapter, ids_and_mask, labels_from_ids,
)
from replica_lab.intrinsic_memory_density_protocols_v12 import (
    build_protocols, canonical_digest, FILENAMES,
)
from replica_lab.intrinsic_memory_density_receipts_v12 import write_new_json
from replica_lab.intrinsic_memory_density_v12 import (
    DensityProducerBoundary, SiteInvocationProfiler, cost_snapshot, ledger_delta,
)
from replica_lab.intrinsic_memory_matched_reader_v12 import (
    MatchedContextualMemoryV12, matched_v12_layout,
)
from replica_lab.tiered_memory import MemoryScope, TierCeilings, TieredMemoryStore

ARMS = {"two": (3, 7),
        "full_attention_nine": tuple(range(3, 36, 4)),
        "all": tuple(range(36))}
FORMAT = "replica.intrinsic_memory.density.cpu_engineering.v12"


class FixtureTokenizer:
    def __call__(self, text, *, add_special_tokens=False, truncation=False):
        return {"input_ids": [3 + (ord(character) % 29) for character in text]}

    def identity(self):
        return {"format": "replica.cpu_density_v12.synthetic_tokenizer.v1",
                "vocab_size": 32, "special_tokens": [0, 1, 2]}


class FixtureAccounting:
    def __init__(self):
        self.counter = 0

    def reserve(self, request):
        self.counter += 1
        return self.counter

    def complete(self, token, *, status="completed", actual=None, detail=None):
        if status not in ("completed", "failed"):
            raise ValueError("invalid fixture accounting status")


def scope():
    return MemoryScope("density_engineering", "main", "engineering", 1000)


def spans(text="Badge means red.", *, name="r1", available=5):
    return canonical_spans(text, record_id=name, source_id="source",
                           source_epoch=1, available_at=available)


def open_store(directory):
    return TieredMemoryStore(
        Path(directory), profile_id="P1", enforce_volume_free=False,
        ceilings=TierCeilings(
            hot_bytes=32 * 1024**2, warm_bytes=32 * 1024**2,
            cold_bytes=32 * 1024**2, identity_reserved_bytes=64 * 1024,
            replay_reserved_bytes=256 * 1024, minimum_volume_free_bytes=0,
            workspace_quota_bytes=1 * 1024**2, artifact_budget_bytes=4 * 1024**2,
            frequent_reserve_fraction=0.5, frequent_min_accesses=2))


def build_adapter_and_provider():
    adapter, *_ = build_contextual_adapter(
        seed=17, layers=36, spine_sites=(3, 7), spine_slots=8, memory_size=256)
    adapter.base.eval()
    for parameter in adapter.base.parameters():
        parameter.requires_grad_(False)
    tokenizer = FixtureTokenizer()
    spec = SourceFeatureSpec(
        original_parameters_sha256=hashlib.sha256(b"cpu_density_v12_original").hexdigest(),
        source_execution_sha256=adapter_contract_digest(adapter),
        tokenizer_sha256=tokenizer_contract_digest(tokenizer),
        hidden_size=adapter.config.geometry.hidden_size,
        max_tokens=64, cache_bytes=1024 * 1024, cache_entries=8)
    provider = FrozenSourceFeatures(adapter, tokenizer, spec,
                                    accounting=FixtureAccounting())
    return adapter, provider


def build_path(arm):
    adapter, provider = build_adapter_and_provider()
    path = ContextualMemoryV12(
        adapter, spark_layout(arm, hidden_size=adapter.config.geometry.hidden_size),
        source_identity="cpu_density_v12_fixture", feature_provider=provider,
        lr=0.001, initialization_seed=17,
        execution_identity="CPU_density_v12_engineering")
    path.adapter.extension.spine = None
    return path


def build_matched_path(arm):
    adapter, provider = build_adapter_and_provider()
    path = MatchedContextualMemoryV12(
        adapter, matched_v12_layout(arm, hidden_size=adapter.config.geometry.hidden_size),
        source_identity="cpu_density_v12_fixture", feature_provider=provider,
        lr=0.001, initialization_seed=17,
        execution_identity="CPU_density_v12_engineering")
    path.adapter.extension.spine = None
    return path


def training_batch(path):
    return path.training_values(spans(), scope=scope())


def step(path, batch=None):
    ids, mask = ids_and_mask(4)
    path.optimizer.zero_grad(set_to_none=True)
    live = (training_batch(path) if batch is None
            else path.training_from_stored(batch, scope=scope()))
    logits = path.forward(ids, mask, live, scope=scope())
    F.cross_entropy(logits.flatten(0, 1), labels_from_ids(ids).flatten()).backward()
    path.optimizer.step()


def forward(path, batch):
    ids, mask = ids_and_mask(4)
    return path.forward(ids, mask, batch, scope=scope())


def _digest_tensor(value):
    payload = value.detach().cpu().float().contiguous().numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


def post_update_profile(path, expected_groups, pre_digest):
    """One profiled forward after an update, to distinguish post-update arms."""
    step(path)
    profiler = SiteInvocationProfiler(path, expected_groups=expected_groups)
    try:
        batch = training_batch(path)
        ids, mask = ids_and_mask(4)
        before = cost_snapshot(path)
        wall = time.perf_counter_ns()
        logits = profiler.forward(path, ids, mask, batch, scope=scope(),
                                  forward_id="post-update")
        elapsed = time.perf_counter_ns() - wall
        after = cost_snapshot(path)
        summary = profiler.summary()
        digest = _digest_tensor(logits)
        return {
            "forward_wall_ns": elapsed,
            "single_forward_tiny_fixture_only": True,
            "logits_sha256": digest,
            "logits_changed_from_pre_update": digest != pre_digest,
            "reader_invocations": summary["total_site_invocations"],
            "physical_reader_groups": summary["physical_groups"],
            "cost_delta": ledger_delta(before, after),
            "profiler": {key: value for key, value in summary.items()
                         if key != "events"},
        }
    finally:
        profiler.close()


def no_update_profile(path, expected_groups, pre_digest):
    """Control: a repeat forward without an update must reproduce its digest."""
    profiler = SiteInvocationProfiler(path, expected_groups=expected_groups)
    try:
        batch = training_batch(path)
        ids, mask = ids_and_mask(4)
        before = cost_snapshot(path)
        logits = profiler.forward(path, ids, mask, batch, scope=scope(),
                                  forward_id="no-update-control")
        after = cost_snapshot(path)
        summary = profiler.summary()
        digest = _digest_tensor(logits)
        return {
            "logits_sha256": digest,
            "unchanged_from_pre_update": digest == pre_digest,
            "reader_invocations": summary["total_site_invocations"],
            "physical_reader_groups": summary["physical_groups"],
            "cost_delta": ledger_delta(before, after),
        }
    finally:
        profiler.close()


def profile_arm(arm, *, include_store=False):
    path = build_path(arm)
    profiler = None
    result = None
    try:
        boundary = DensityProducerBoundary(path, comparison="fixed_per_site")
        profiler = boundary.profiler(expected_groups=len(ARMS[arm]))
        batch = training_batch(path)
        ids, mask = ids_and_mask(4)
        before = cost_snapshot(path)
        wall = time.perf_counter_ns()
        logits = profiler.forward(path, ids, mask, batch, scope=scope())
        elapsed = time.perf_counter_ns() - wall
        after = cost_snapshot(path)
        summary = profiler.summary()
        digest = _digest_tensor(logits)
        result = {
            "arm": arm,
            "sites": list(ARMS[arm]),
            "physical_reader_groups": summary["physical_groups"],
            "callbacks_per_physical_group": summary["callbacks_per_physical_group"],
            "reader_invocations": summary["total_site_invocations"],
            "site_invocations": summary["site_invocations"],
            "forward_wall_ns": elapsed,
            "single_forward_tiny_fixture_only": True,
            "logits_sha256": digest,
            "parameter_audit": boundary.audit,
            "feature_contract": boundary.identity()["feature_contract"],
            "producer_release": boundary.identity()["producer_release"],
            "cost_delta": ledger_delta(before, after),
            "profiler": {key: value for key, value in summary.items()
                         if key != "events"},
        }
        if include_store:
            with tempfile.TemporaryDirectory(prefix="density-v12-store-") as temp:
                store = open_store(temp)
                try:
                    original = spans()[0]
                    admitted = path.admit(store, original, scope=scope())
                    batch = path.read(store, "Badge", scope=scope())
                    before = cost_snapshot(path)
                    profiler.forward(path, ids, mask, batch, scope=scope(),
                                     forward_id="durable-1")
                    after = cost_snapshot(path)
                    result["store"] = {
                        "admission_ok": bool(admitted.outcome.ok),
                        "source_span_count": len(batch.spans),
                        "cost_delta": ledger_delta(before, after),
                    }
                finally:
                    store.close()
        return result
    finally:
        if profiler is not None:
            profiler.close()
        try:
            if result is not None:
                result["no_update_control"] = no_update_profile(
                    path, len(ARMS[arm]), digest)
                result["post_update"] = post_update_profile(
                    path, len(ARMS[arm]), digest)
        finally:
            path.close()


def profile_matched_arm(arm):
    path = build_matched_path(arm)
    profiler = None
    result = None
    try:
        boundary = DensityProducerBoundary(path, comparison="matched_unique_parameters")
        profiler = boundary.profiler(expected_groups=2)
        batch = training_batch(path)
        ids, mask = ids_and_mask(4)
        before = cost_snapshot(path)
        wall = time.perf_counter_ns()
        logits = profiler.forward(path, ids, mask, batch, scope=scope())
        elapsed = time.perf_counter_ns() - wall
        after = cost_snapshot(path)
        summary = profiler.summary()
        digest = _digest_tensor(logits)
        result = {
            "arm": arm,
            "sites": list(ARMS[arm]),
            "physical_reader_groups": summary["physical_groups"],
            "callbacks_per_physical_group": summary["callbacks_per_physical_group"],
            "reader_invocations": summary["total_site_invocations"],
            "site_invocations": summary["site_invocations"],
            "forward_wall_ns": elapsed,
            "single_forward_tiny_fixture_only": True,
            "logits_sha256": digest,
            "parameter_audit": boundary.audit,
            "feature_contract": boundary.identity()["feature_contract"],
            "producer_release": boundary.identity()["producer_release"],
            "cost_delta": ledger_delta(before, after),
            "profiler": {key: value for key, value in summary.items()
                         if key != "events"},
        }
        return result
    finally:
        if profiler is not None:
            profiler.close()
        try:
            if result is not None:
                result["no_update_control"] = no_update_profile(path, 2, digest)
                result["post_update"] = post_update_profile(path, 2, digest)
        finally:
            path.close()


def continuation(arm, directory, *, matched=False):
    """Save, continue, then compare with a fresh process restoring the checkpoint."""
    directory = Path(directory)
    checkpoint, output = directory / "extension", directory / "continued.pt"
    path = build_matched_path(arm) if matched else build_path(arm)
    try:
        step(path)
        batch = training_batch(path)
        path.save(checkpoint)
        expected_logits = forward(path, batch).detach()
        step(path)
        expected = {"weights": {key: value.clone() for key, value in path._tensors().items()},
                    "optimizer": path.optimizer.state_dict()}
    finally:
        path.close()
    builder = "build_matched_path" if matched else "build_path"
    child = (
        "import sys, torch\n"
        "import profile_intrinsic_memory_density_v12 as m\n"
        "torch.set_num_threads(1)\n"
        f"p=m.{builder}(sys.argv[1]); p.load(sys.argv[2])\n"
        "b=m.training_batch(p); logits=m.forward(p,b).detach(); m.step(p)\n"
        "assert p.adapter.base_unchanged(); p.assert_optimizer_ownership()\n"
        "torch.save({'logits':logits,'weights':p._tensors(),"
        "'optimizer':p.optimizer.state_dict()},sys.argv[3])\n"
        "p.close()\n")
    result = subprocess.run(
        [sys.executable, "-c", child, arm, str(checkpoint), str(output)],
        cwd=ROOT,
        env={**os.environ,
             "PYTHONPATH": str(ROOT / "src") + os.pathsep + str(ROOT / "scripts"),
             "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True, text=True, timeout=900)
    if result.returncode != 0:
        raise RuntimeError(result.stderr)
    produced = torch.load(output, weights_only=True)
    exact_logits = torch.equal(produced["logits"], expected_logits)
    exact_weights = all(torch.equal(produced["weights"][key], value)
                        for key, value in expected["weights"].items())
    exact_optimizer = _tree_equal(produced["optimizer"], expected["optimizer"])
    return {"arm": arm, "exact_logits": exact_logits,
            "exact_weights": exact_weights, "exact_optimizer_moments": exact_optimizer,
            "actual_update_after_restore": True,
            "fresh_process": True}


def _matched_summary_arms():
    path = ROOT / "reports/intrinsic_memory_matched_reader_v1/clean/summary.json"
    if not path.is_file():
        return {}
    summary = json.loads(path.read_text())
    keys = ("arm", "sites", "unique_reader_groups", "unique_reader_parameters",
            "all_trainable_parameters", "optimizer_owned_tensors",
            "reader_invocations", "slot_storage_addresses", "original_unchanged")
    return {row["arm"]: {key: row[key] for key in keys if key in row}
            for row in summary.get("arms", [])}


def _tree_equal(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict):
        return (isinstance(right, dict) and set(left) == set(right)
                and all(_tree_equal(left[key], right[key]) for key in left))
    if isinstance(left, (list, tuple)):
        return (isinstance(right, (list, tuple)) and len(left) == len(right)
                and all(_tree_equal(a, b) for a, b in zip(left, right)))
    return left == right


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "reports/intrinsic_memory_density_v12/repair01/cpu_engineering_summary.json")
    parser.add_argument("--skip-continuation", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    protocols = build_protocols()
    report = {
        "format": FORMAT,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "CPU_random_36_block_fixture_engineering_only",
        "native_execution": False,
        "native_weights_loaded": False,
        "downloads": False,
        "P_updates": 0,
        "promotion": False,
        "competence_claim": False,
        "python": sys.version,
        "torch": torch.__version__,
        "platform": platform.platform(),
        "protocol_digests": {name: canonical_digest(value)
                             for name, value in protocols.items()},
        "protocol_files": {name: FILENAMES[name] for name in protocols},
        "arms": {},
        "matched_arms": {},
        "matched_reader_mechanics": {
            "implementation": "replica.intrinsic_memory.matched_reader.v12",
            "unique_reader_parameters_hidden_8": 272418,
            "unique_reader_groups": 2,
            "native_unique_reader_parameters_hidden_2560": 2895874,
            "v12_matched_wrapper": "implemented_separately_versioned",
            "existing_v5_cpu_summary": "reports/intrinsic_memory_matched_reader_v1/clean/summary.json",
            "existing_v5_cpu_arms": _matched_summary_arms(),
            "verification_command": (
                "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q "
                "tests/test_intrinsic_memory_density_v12_matched.py "
                "tests/test_intrinsic_memory_matched_reader_v1.py"),
        },
    }
    for arm in ARMS:
        report["arms"][arm] = profile_arm(arm, include_store=(arm == "two"))
        report["matched_arms"][arm] = profile_matched_arm(arm)
    if not args.skip_continuation:
        with tempfile.TemporaryDirectory(prefix="density-v12-continuation-") as temp:
            report["fresh_process_continuation"] = {
                arm: continuation(arm, Path(temp) / arm) for arm in ARMS}
            report["matched_fresh_process_continuation"] = {
                arm: continuation(arm, Path(temp) / ("matched-" + arm), matched=True)
                for arm in ARMS}
    write_new_json(args.output, report)
    print(json.dumps({key: report[key] for key in
                      ("format", "scope", "protocol_digests")
                      if key in report} | (
                          {"fresh_process_continuation":
                           report["fresh_process_continuation"]}
                          if "fresh_process_continuation" in report else {}),
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

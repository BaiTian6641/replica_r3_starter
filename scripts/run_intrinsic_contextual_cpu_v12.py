#!/usr/bin/env python3
"""Integrated 36-block contextual-v12 CPU workflow; synthetic engineering only.

Exercises the real v12 frozen-source producer, contextual compressor, shared
reader snapshot, durable store, cold tickets, producer release and complete
extension checkpoint over a tiny random BF16 backbone and an explicit synthetic
tokenizer. No pretrained checkpoint, download, native model load, GPU job or
capability claim is involved. ``--child`` is the fresh-process continuation.

Exact verification commands are recorded in the output; this script does not run
pytest and does not produce native readiness evidence.
"""
import argparse
from dataclasses import asdict
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

import torch  # noqa: E402

from replica_lab.contextual_memory import ContextualMemoryError  # noqa: E402
from replica_lab.contextual_memory_v5 import PendingContextualRead  # noqa: E402
from replica_lab.contextual_release_v12 import ProducerReleaseCorpus  # noqa: E402
from replica_lab.contextual_source_features_v12 import WorkLedgerSourceAccounting  # noqa: E402
from replica_lab.contextual_v12_fixture import (  # noqa: E402
    ENGINEERING_LABEL, build_v12_stack, canonical_span, canonical_spans,
    logits_digest, model_state_digest, open_v12_store, optimizer_digest,
    parameter_digest, scope, spans_for)
from replica_lab.intrinsic_memory_budget_v5 import COUNTERS, WorkLedger  # noqa: E402

FORMAT = "replica.intrinsic_memory.contextual_cpu_workflow.v12"
ARM_SITES = {"two": (3, 7), "full_attention_nine": tuple(range(3, 36, 4)),
             "all": tuple(range(36))}
ARMS = tuple(ARM_SITES)
STAGES = ("mechanics", "restart")
SCOPES = {
    "primary": "user_a",
    "correction": "user_correction",
    "revocation": "user_revocation",
}
IDS = torch.tensor([[3, 5, 7, 9, 11, 13]], dtype=torch.long)
LABELS = IDS.clone()
LABELS[:, 0] = -100
SPAN_TEXTS = ("the bright badge is clearly red", "the bright badge is clearly blue")


def log(path, message):
    line = f"{time.time():.6f} {message}\n"
    with path.open("a") as stream:
        stream.write(line)
    print(message, flush=True)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def plan():
    return {stage: {**{key: 20000 for key in COUNTERS},
                    "artifact_bytes": 64 * 1024**2, "wall_seconds": 300}
            for stage in STAGES}


def code_hashes():
    files = [
        "src/replica_lab/contextual_source_features_v12.py",
        "src/replica_lab/contextual_feature_identity_v12.py",
        "src/replica_lab/contextual_memory_layout_v12.py",
        "src/replica_lab/contextual_memory_v12.py",
        "src/replica_lab/contextual_release_v12.py",
        "src/replica_lab/contextual_v12_fixture.py",
        "scripts/run_intrinsic_contextual_cpu_v12.py",
        "tests/v12_fixtures.py",
        "tests/test_contextual_source_features_v12.py",
        "tests/test_contextual_memory_layout_v12.py",
        "tests/test_contextual_memory_v12.py",
        "tests/test_contextual_v12_integration.py",
    ]
    return {name: {"bytes": (ROOT / name).stat().st_size, "sha256": sha(ROOT / name)}
            for name in files}


def inventory(root):
    entries = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            entries.append({"path": str(path.relative_to(root)),
                            "bytes": path.stat().st_size,
                            "sha256": sha(path)})
    return entries


def train_step(path, adapter, spans, sc):
    path.optimizer.zero_grad(set_to_none=True)
    batch = path.training_values(spans, scope=sc)
    logits = path.forward(IDS, torch.ones_like(IDS), batch, scope=sc)
    loss = adapter.causal_loss(logits, IDS, LABELS)
    loss.backward()
    path.optimizer.step()
    return batch, logits, loss


def gradient_report(path):
    def summarize(module):
        names = {name: ("none" if parameter.grad is None
                        else "zero" if not bool(torch.count_nonzero(parameter.grad))
                        else "nonzero")
                 for name, parameter in module.named_parameters()}
        return {"nonzero": sorted(name for name, state in names.items()
                                  if state == "nonzero"),
                "zero": sorted(name for name, state in names.items() if state == "zero"),
                "missing": sorted(name for name, state in names.items() if state == "none")}
    return {"producer": summarize(path.encoder), "order": summarize(path.order),
            "reader": {site: summarize(port) for site, port in path.reader.ports.items()}}


def expected(path, sc, spans):
    with torch.no_grad():
        return logits_digest(path.forward(
            IDS, torch.ones_like(IDS), path.training_values(spans, scope=sc), scope=sc))


def build(arm, accounting):
    adapter, layout, provider, path = build_v12_stack(
        seed=17, layers=36, sites=ARM_SITES[arm], slots=8, lr=1e-3,
        initialization_seed=17, cache_entries=16, accounting=accounting,
        source_identity=f"cpu_v12_random_bf16_{arm}", execution_identity="CPU_v12_workflow")
    return adapter, layout, provider, path


def run_arm(arm, root):
    root.mkdir()
    run_log = root / "run.log"
    log(run_log, f"{arm}: start; synthetic={ENGINEERING_LABEL}")
    ledger = WorkLedger.create(root / "ledger.json", identity=f"cpu_v12_{arm}",
                               plan=plan(), deadline=time.monotonic() + 900)
    ledger.enter("mechanics")
    accounting = WorkLedgerSourceAccounting(ledger, "mechanics")
    adapter, layout, provider, path = build(arm, accounting)
    original = parameter_digest(adapter.base)
    declared = path.assert_optimizer_ownership()
    sc = scope(SCOPES["primary"])
    spans = spans_for(list(SPAN_TEXTS))
    ids, labels = IDS, LABELS

    log(run_log, f"{arm}: owners={sorted(declared)} sites={list(layout.sites)}")
    with torch.no_grad():
        batch = path.training_values(spans, scope=sc)
    disabled_before = dict(path.cost)
    provider_before = dict(provider.cost)
    with torch.no_grad():
        disabled = path.forward(ids, torch.ones_like(ids), batch, scope=sc, enabled=False)
        empty = path.forward(ids, torch.ones_like(ids), batch, scope=sc, active_sites=())
        reference = adapter.forward_text(ids, torch.ones_like(ids), use_projector=False,
                                         use_ports=False, use_spine=False)
    disabled_cost = {key: path.cost[key] - disabled_before[key] for key in path.cost}
    provider_disabled = {key: provider.cost[key] - provider_before[key]
                         for key in provider.cost}
    parity = {
        "enabled_false_equals_base": bool(torch.equal(disabled, reference)),
        "empty_active_sites_equals_base": bool(torch.equal(empty, reference)),
        "snapshot_copies_while_disabled": disabled_cost["snapshot_copies"],
        "producer_calls_while_disabled": disabled_cost["producer_calls"],
        "source_forwards_while_disabled": provider_disabled["forwards"],
    }

    # Step one: zero upstream gradients from the zero output projections.
    path.optimizer.zero_grad(set_to_none=True)
    batch1 = path.training_values(spans, scope=sc)
    logits1 = path.forward(ids, torch.ones_like(ids), batch1, scope=sc)
    loss1 = adapter.causal_loss(logits1, ids, labels)
    loss1.backward()
    step1 = gradient_report(path)
    path.optimizer.step()
    producer_step1_zero = not step1["producer"]["nonzero"]
    order_step1_zero = not step1["order"]["nonzero"]
    early_site = str(layout.sites[0])
    output_projection_live = (step1["reader"][early_site]["nonzero"]
                              == ["output_projection.weight"])

    # Step two: the live producer/order/reader path now receives gradients.
    path.optimizer.zero_grad(set_to_none=True)
    batch2 = path.training_values(spans, scope=sc)
    batch2.slots.retain_grad()
    logits2 = path.forward(ids, torch.ones_like(ids), batch2, scope=sc)
    loss2 = adapter.causal_loss(logits2, ids, labels)
    loss2.backward()
    step2 = gradient_report(path)
    trainable = [name for name, parameter in path.encoder.named_parameters()]
    producer_live = all(name in step2["producer"]["nonzero"] for name in
                        ("input_projection.weight", "slot_queries", "output.1.weight"))
    order_live = step2["order"]["nonzero"] == ["weight"]
    ordered_slots_live = bool(batch2.slots.grad is not None
                              and torch.count_nonzero(batch2.slots.grad))
    every_reader_live = all(
        step2["reader"][site]["missing"] == [] and step2["reader"][site]["nonzero"]
        for site in step2["reader"])
    early_reader_live = bool(step2["reader"][early_site]["nonzero"])
    path.optimizer.step()
    log(run_log, f"{arm}: gradients live="
                 f"{producer_live and order_live and ordered_slots_live and every_reader_live}")
    # Additional updates make the permutation control distinguishable in BF16.
    for _ in range(6):
        train_step(path, adapter, spans, sc)

    # One snapshot for all active readers on one unchanged batch.
    probe = path.training_values(spans, scope=sc)
    before = dict(path.cost)
    with torch.no_grad():
        path.forward(ids, torch.ones_like(ids), probe, scope=sc)
    snapshot_cost = {key: path.cost[key] - before[key] for key in path.cost}
    with torch.no_grad():
        context = path.snapshot(ids, probe, scope=sc)
    single = path.training_values((spans[0],), scope=sc)
    with torch.no_grad():
        finite = path.snapshot(ids, single, scope=sc, padding_value=0.0)
        huge = path.snapshot(ids, single, scope=sc, padding_value=1e6)
        finite_logits = adapter.forward_text(ids, torch.ones_like(ids), use_projector=False,
                                             use_ports=False, use_spine=False,
                                             intrinsic_memory=finite)
        huge_logits = adapter.forward_text(ids, torch.ones_like(ids), use_projector=False,
                                           use_ports=False, use_spine=False,
                                           intrinsic_memory=huge)
        early_only = path.snapshot(ids, probe, scope=sc, active_sites=(layout.sites[0],))
    masks = {
        "fixed_slot_count": int(finite.values.shape[2]),
        "masked_padding_slots": int((~finite.availability).sum()),
        "padded_value_does_not_change_logits": bool(torch.equal(finite_logits, huge_logits)),
        "snapshot_is_a_copy": bool(context.values.data_ptr() != probe.slots.data_ptr()),
        "early_site_mask": list(early_only.spec.sites),
    }
    one_snapshot = {
        "active_sites": len(context.spec.sites),
        "snapshot_copies": snapshot_cost["snapshot_copies"],
        "retrievals": snapshot_cost["retrieval_calls"],
        "admissions": snapshot_cost["admission_calls"],
        "value_slots": snapshot_cost["value_slots_read"],
        "budget_slots": layout.slots,
    }

    # Synthetic controls: correct, wrong, shuffled, zero, absent.
    correct = path.training_values(spans, scope=sc)
    wrong = path.training_values(spans_for(["the badge is green", "the badge is black"]), scope=sc)
    shuffled = path.training_values(tuple(reversed(spans)), scope=sc)
    from replica_lab.contextual_feature_identity_v12 import ValueBatch
    zero = ValueBatch(torch.zeros_like(correct.slots), correct.spans, correct.scope_key,
                      correct.producer_version, training=True)
    with torch.no_grad():
        controls = {name: logits_digest(path.forward(ids, torch.ones_like(ids), batch, scope=sc))
                    for name, batch in (("correct", correct), ("wrong", wrong),
                                        ("shuffled", shuffled), ("zero", zero),
                                        ("absent", None))}
    controls_distinct = len(set(controls.values())) == 5
    shuffled_slots_differ = not torch.equal(correct.slots, shuffled.slots)
    log(run_log, f"{arm}: controls distinct={controls_distinct} "
                 f"shuffled_slots_differ={shuffled_slots_differ}")

    # Warm durable read versus live training values.
    store = open_v12_store(root, name="store")
    primary = canonical_span("the badge is red", record_id="badge")[0]
    path.admit(store, primary, scope=sc)
    durable = path.read(store, "badge", scope=sc)
    with torch.no_grad():
        live_digest = expected(path, sc, (primary,))
        durable_digest = logits_digest(path.forward(
            ids, torch.ones_like(ids), durable, scope=sc))
    durable_parity = {"exact": live_digest == durable_digest}

    # Producer release publication and a release-bound read.
    corpus = ProducerReleaseCorpus(store, sc)
    published = corpus.rebuild(path)
    path.release_corpus = corpus
    release_read = path.read(store, "badge", scope=sc, wait_seconds=1)
    released_batch = path.read(store, "badge", scope=sc, wait_seconds=1)
    release = {"generation": published["generation"],
               "records": len(published["records"]),
               "read_spans": len(released_batch.spans),
               "bound_generation": released_batch.pins[0]["producer_generation"],
               "matches_live_slots": bool(torch.equal(release_read.slots,
                                                      released_batch.slots))}
    path.release_corpus = None

    # Correction and revocation fail closed and invalidate only their source.
    correction_scope = scope(SCOPES["correction"])
    correction_store = open_v12_store(root, name="correction_store")
    path.admit(correction_store, canonical_span("badge record red", record_id="corrected",
                                                available_at=1)[0], scope=correction_scope)
    correction_batch = path.read(correction_store, "badge", scope=correction_scope)
    correction_span = canonical_span("badge record blue", record_id="correction",
                                     available_at=2)[0]
    correction_span = correction_span.__class__(
        **{**asdict(correction_span), "source_epoch": 2,
           "supersedes": "corrected.span0"})
    path.correct(correction_store, "corrected.span0", correction_span, scope=correction_scope)
    corrected_read = path.read(correction_store, "badge", scope=correction_scope)
    correction = {"read_after_correction": len(corrected_read.spans),
                  "corrected_record_evicted": all(
                      name != "corrected.span0" for name, _ in provider._cache.values())}
    revocation_scope = scope(SCOPES["revocation"])
    revocation_store = open_v12_store(root, name="revocation_store")
    path.admit(revocation_store, canonical_span("revoked record", record_id="revoked")[0],
               scope=revocation_scope)
    revoked_batch = path.read(revocation_store, "revoked", scope=revocation_scope)
    revocation_store.revoke_record("revoked.span0", reason="workflow_revocation",
                                   scope=revocation_scope)
    refused = False
    try:
        path.validate(revoked_batch, revocation_scope)
    except ContextualMemoryError:
        refused = True
    revocation = {"validate_refused_after_revocation": refused}

    # Cold pending ticket for fresh-process recovery.
    cold_scope = scope("user_cold")
    cold_store = open_v12_store(root, name="cold_store")
    for index in range(2):
        path.admit(cold_store, canonical_span(f"cold marker {index}",
                                              record_id=f"cold{index}")[0],
                   scope=cold_scope, placement="cold")
    pending = path.read(cold_store, "cold marker", scope=cold_scope)
    assert isinstance(pending, PendingContextualRead)
    (root / "pending.json").write_text(json.dumps(asdict(pending)))
    (root / "cold_scope.json").write_text(json.dumps(asdict(cold_scope)))
    cold_store.close()

    # Complete checkpoint, then an uninterrupted third update for comparison.
    path.save(root / "checkpoint")
    pre_update = expected(path, sc, spans)
    loss3 = train_step(path, adapter, spans, sc)[2]
    parent = {"pre_logits": pre_update, "loss": float(loss3.detach()),
              "state": model_state_digest(path), "moments": optimizer_digest(path.optimizer),
              "scopes": {"primary": asdict(sc)},
              "span_texts": list(SPAN_TEXTS)}
    (root / "expected.json").write_text(json.dumps(parent))
    contract_refusal = {}
    _, _, _, other = build_v12_stack(seed=17, layers=36, sites=ARM_SITES[arm], slots=8,
                                     lr=1e-3, initialization_seed=17, cache_entries=16,
                                     tokenizer_mode="words", source_identity="other_contract",
                                     execution_identity="CPU_v12_workflow")
    try:
        other.load(root / "checkpoint")
        contract_refusal["different_feature_contract_refused"] = False
    except ContextualMemoryError:
        contract_refusal["different_feature_contract_refused"] = True
    original_unchanged = adapter.base_unchanged() and parameter_digest(adapter.base) == original
    ledger.leave("mechanics")
    ledger_snapshot = ledger.snapshot()
    path.close()
    store.close()
    correction_store.close()
    revocation_store.close()

    child = subprocess.run(
        [sys.executable, __file__, "--child", arm, "--output", str(root)],
        cwd=ROOT, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True, text=True, timeout=600)
    (root / "child.stdout").write_text(child.stdout)
    (root / "child.stderr").write_text(child.stderr)
    if child.returncode:
        raise RuntimeError(child.stderr)
    restart = json.loads((root / "restart.json").read_text())
    return {
        "arm": arm, "block_count": 36, "sites": list(layout.sites),
        "reader_count": len(layout.sites), "owner_groups": sorted(declared),
        "updates_before_continuation": 8, "continuation_update_index": 9,
        "gradient_probe_updates": [1, 2],
        "parameters": {name: len(names) for name, names in declared.items()},
        "original_parameters_sha256": original, "original_unchanged": original_unchanged,
        "gradients": {
            "step1_zero_upstream": {"producer_zero": producer_step1_zero,
                                    "order_zero": order_step1_zero,
                                    "output_projection_live": output_projection_live},
            "step2_live": {"producer": producer_live, "order": order_live,
                           "ordered_slots": ordered_slots_live,
                           "every_reader": every_reader_live,
                           "early_reader": early_reader_live,
                           "producer_tensor_count": len(trainable)},
        },
        "disabled": parity, "one_snapshot": one_snapshot, "masks": masks, "controls": controls,
        "controls_distinct": controls_distinct, "shuffled_slots_differ": shuffled_slots_differ,
        "durable_parity": durable_parity, "release": release,
        "correction": correction, "revocation": revocation,
        "contract_refusal": contract_refusal,
        "restart": restart,
        "ledger": {"stages": list(ledger_snapshot["stages"]),
                   "reserved": ledger_snapshot["reserved"],
                   "events": len(ledger_snapshot["events"])},
        "release_corpus_cost": corpus.cost,
        "producer_cost": dict(provider.cost),
        "path_cost": dict(path.cost),
        "artifact_bytes": sum(p.stat().st_size for p in root.rglob("*") if p.is_file()),
        "synthetic_only": True, "native_admissible": False,
    }


def child(arm, root):
    run_log = root / "child.log"
    log(run_log, f"{arm}: child start")
    torch.set_num_threads(1)
    ledger = WorkLedger(root / "ledger.json", identity=f"cpu_v12_{arm}")
    ledger.enter("restart")
    accounting = WorkLedgerSourceAccounting(ledger, "restart")
    adapter, layout, provider, path = build(arm, accounting)
    original = parameter_digest(adapter.base)
    expected_document = json.loads((root / "expected.json").read_text())
    path.load(root / "checkpoint")
    sc = scope(SCOPES["primary"])
    spans = spans_for(expected_document["span_texts"])
    with torch.no_grad():
        pre_logits = logits_digest(path.forward(
            IDS, torch.ones_like(IDS), path.training_values(spans, scope=sc), scope=sc))
    cold_scope = type(sc)(**json.loads((root / "cold_scope.json").read_text()))
    pending = PendingContextualRead(**json.loads((root / "pending.json").read_text()))
    cold_store = open_v12_store(root, name="cold_store")
    recovered = path.recover_pending(cold_store, pending.ticket_id, scope=cold_scope)
    cold_batch = path.resume(cold_store, recovered, scope=cold_scope, wait_seconds=1)
    cold_store.close()
    loss = train_step(path, adapter, spans, sc)[2]
    document = {
        "fresh_process": True,
        "pre_logits_exact": pre_logits == expected_document["pre_logits"],
        "loss_exact": float(loss.detach()) == expected_document["loss"],
        "state_matches_uninterrupted": model_state_digest(path) == expected_document["state"],
        "optimizer_moments_match": optimizer_digest(path.optimizer)
            == expected_document["moments"],
        "cold_pending_resumed": len(cold_batch.spans) == 2,
        "cold_completed_files": cold_batch.read_costs["cold_completed_files"],
        "original_unchanged": adapter.base_unchanged() and parameter_digest(adapter.base) == original,
        "owner_groups": sorted(path.assert_optimizer_ownership()),
        "actual_continuation_updates": 1,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "provider_cost": dict(provider.cost),
        "ledger_reserved": ledger.snapshot()["reserved"],
    }
    ledger.leave("restart")
    path.close()
    (root / "restart.json").write_text(json.dumps(document))
    log(run_log, f"{arm}: child complete state_exact="
                 f"{document['state_matches_uninterrupted']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--child", choices=ARMS)
    arguments = parser.parse_args()
    torch.set_num_threads(1)
    if arguments.child:
        child(arguments.child, arguments.output)
        return
    if arguments.output.exists():
        raise SystemExit("output directory must be fresh")
    arguments.output.mkdir(parents=True)
    started = time.monotonic()
    hashes = code_hashes()
    results = [run_arm(arm, arguments.output / arm) for arm in ARMS]
    result = {
        "format": FORMAT,
        "engineering_label": ENGINEERING_LABEL,
        "synthetic_only": True,
        "native_admissible": False,
        "native_claims": {"native_model_loaded": False, "downloads": False,
                          "native_competence": False, "useful_recall": False,
                          "semantic_recall": False, "persistent_learning": False,
                          "promotion": False, "rsi": False},
        "code_sha256": hashes,
        "tests": {
            "run_by_this_workflow": False,
            "command": ("python -m pytest -q tests/test_contextual_source_features_v12.py "
                        "tests/test_contextual_memory_layout_v12.py "
                        "tests/test_contextual_memory_v12.py "
                        "tests/test_contextual_v12_integration.py"),
            "files": [name for name in hashes if name.startswith("tests/test_")],
        },
        "arms": results,
        "resources": {
            "parent_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "wall_seconds": time.monotonic() - started,
            "torch_threads": torch.get_num_threads(),
            "device": "cpu", "native_gpu_used": False,
        },
        "artifact_inventory": inventory(arguments.output),
        "limitations": [
            "random tiny BF16 backbone and synthetic tokenizer; not Spark",
            "no pretrained checkpoint, no download, no native model load",
            "engineering influence on random weights is not semantic recall",
            "no persistent-P learning, promotion, RSI or capability claim",
            "full-original hashes are operator-declared here and must be verified natively",
            "pytest is the declared command and was not executed by this workflow",
        ],
    }
    (arguments.output / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"arms": [arm["arm"] for arm in results],
                      "native_admissible": False,
                      "artifact_inventory": len(result["artifact_inventory"])}))


if __name__ == "__main__":
    main()

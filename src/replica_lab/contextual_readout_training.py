"""Bounded readout experiment, before persistent P learning is interpretable.

Calibration, readout-fit and readout-development have distinct data identities.
Development labels are read only after the fixed updates and encoder freeze.
No task reward, generated explanation or development answer enters the encoder.
"""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import resource
import time

import torch
import torch.nn.functional as F

from .baseline_competence import sha, score, validate_receipt, load_spec
from .profiling import _ru_maxrss_bytes
import sys
from .contextual_memory import ContextualMemoryPath, canonical_spans, tensor_digest
from .tiered_memory import MemoryScope, TierCeilings, TieredMemoryStore

ARMS = ("contextual", "selected_exact", "wrong", "shuffled", "absent")


def load_readout_spec(protocol):
    binding = protocol.config["contextual_readout"]
    path = protocol.root / binding["path"]
    if sha(path) != binding["sha256"]:
        raise ValueError("contextual readout protocol hash differs")
    spec = json.loads(path.read_text())
    for name, metadata in spec["files"].items():
        if sha(protocol.root / metadata["path"]) != metadata["sha256"]:
            raise ValueError(f"readout data hash differs: {name}")
    if spec["arms"] != list(ARMS) or spec["steps"] != 32 or spec["active_P_experts"] != 1:
        raise ValueError("undeclared contextual work or controls")
    return spec


def spans_for(row):
    return canonical_spans(row["text"], record_id=row["case_id"], source_id=row["world_id"],
                           source_epoch=1, available_at=1)


def prompt_for(row):
    return "Reply with only the answer word. If unsupported reply unknown.\nQuestion: " + row["question"] + "\nAnswer:"


def response_step(path, backend, row, *, context_tokens):
    scope = MemoryScope("readout", "main", "assistant", as_of=10)
    batch = path.training_values(spans_for(row), scope=scope)
    ids, mask, labels = backend.encode_training(prompt_for(row), row["answer"], context_tokens)
    path.optimizer.zero_grad(set_to_none=True)
    logits = path.forward(ids, mask, batch, scope=scope)
    loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), labels.reshape(-1))
    loss.backward()
    norms = {}
    for group in path.optimizer.param_groups:
        norms[group["name"]] = 0.0
        for p in group["params"]:
            if p.grad is not None:
                if not torch.isfinite(p.grad).all(): raise ValueError("nonfinite readout gradient")
                norms[group["name"]] += float(p.grad.detach().abs().sum())
    before = path.encoder.release()
    path.optimizer.step()
    return {"loss": float(loss.detach()), "gradient_l1": norms,
            "producer_changed": before != path.encoder.release(), "positions": int(ids.shape[1])}


def generate(path, backend, row, batch, *, scope, context_tokens, max_new_tokens, exact=False):
    prompt = prompt_for(row)
    if exact:
        prompt = "Evidence:\n" + path.exact_fallback(batch, scope=scope, tokenizer=backend.encode_text) + "\n" + prompt
    ids, mask = backend.encode_prompt(prompt, context_tokens)
    generated, positions = [], 0
    started = time.monotonic()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = path.forward(ids, mask, None if exact else batch, scope=scope)
            value = int(logits[0,-1].argmax())
            generated.append(value); positions += int(ids.shape[1])
            if value in backend.stop_token_ids: break
            ids = torch.cat((ids, ids.new_tensor([[value]])), dim=1)
            mask = torch.cat((mask, mask.new_ones((1,1))), dim=1)
    text = backend.decode([i for i in generated if i not in backend.stop_token_ids]).strip()
    return {"decoded": text, "token_ids": generated, "generated_tokens": len(generated),
            "processed_positions": positions, "latency_seconds": time.monotonic()-started}


def run_readout(protocol, backend, output, ledger, competence, *, meter=None):
    calibration, _ = load_spec(protocol)
    validate_receipt(protocol, competence, context_tokens=ledger.context_tokens,
        max_new_tokens=min(ledger.max_new_tokens, calibration["max_new_tokens"]))
    if backend.native_admissible and competence.get("native_execution") is not True:
        raise ValueError("native readout requires native competence evidence")
    spec = load_readout_spec(protocol)
    output = Path(output)
    target = output / "contextual_readout"
    target.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    fit = json.loads((protocol.root / spec["files"]["fit"]["path"]).read_text())["cases"]
    path = ContextualMemoryPath(backend.adapter, lr=spec["learning_rate"])
    initial = path.encoder.release()
    traces = []
    for index in range(spec["steps"]):
        row = fit[index % len(fit)]
        receipt = response_step(path, backend, row, context_tokens=ledger.context_tokens)
        traces.append({"step": index+1, "case_id": row["case_id"], **receipt})
        ledger.charge("readout_steps")
        for name in ("training_forwards", "forward_calls", "backward_calls", "optimizer_steps"):
            ledger.charge(name)
        ledger.charge("processed_text_positions", receipt["positions"])
        ledger.charge("backward_processed_positions", receipt["positions"])
        (target / "fit.json").write_text(json.dumps(traces, indent=2)+"\n")
        if meter is not None: ledger.set_measured("artifact_bytes", meter.observe())
    if not (traces[0]["gradient_l1"]["reader"] > 0
            and any(t["gradient_l1"]["producer"] > 0 for t in traces[1:])
            and initial != path.encoder.release() and backend.adapter.base_unchanged()):
        raise ValueError("readout numerical qualification failed")
    manifest = path.save(target / "extension")
    ledger.charge("checkpoint_writes")
    if meter is not None: ledger.set_measured("artifact_bytes", meter.observe())
    # Persist a small reload canary before destroying the optimizer owner.
    scope = MemoryScope("readout", "main", "assistant", as_of=10)
    ids, mask, _ = backend.encode_training(prompt_for(fit[0]), fit[0]["answer"], ledger.context_tokens)
    with torch.no_grad():
        logits = path.forward(ids, mask, path.training_values(spans_for(fit[0]), scope=scope), scope=scope)
    ledger.charge("forward_calls")
    ledger.charge("processed_text_positions", int(ids.shape[1]))
    canary = {"logits_sha256": tensor_digest(logits), "case_id": fit[0]["case_id"],
              "context_tokens": ledger.context_tokens}
    (target / "reload_canary.json").write_text(json.dumps(canary, indent=2)+"\n")
    path.freeze_readout()
    # This is the only development-label read in the readout trainer.
    dev = json.loads((protocol.root / spec["files"]["development"]["path"]).read_text())["cases"]
    labels = json.loads((protocol.root / spec["files"]["development_labels"]["path"]).read_text())["labels"]
    if set(labels) != {r["case_id"] for r in dev} or {r["world_id"] for r in dev} & {r["world_id"] for r in fit}:
        raise ValueError("readout development labels/world split differs")
    trajectories = []
    limits = TierCeilings(minimum_volume_free_bytes=0)
    for index, row in enumerate(dev):
        store = TieredMemoryStore(target / "stores" / row["case_id"], ceilings=limits)
        span = spans_for(row)[0]
        path.admit(store, span, scope=scope)
        batch = path.read(store, row["question"], scope=scope)
        # Controls explicitly replace values; never claim current evidence for
        # these untrusted experimental tensors or serve them to an application.
        with torch.no_grad():
            wrong = path.training_values(spans_for(dev[(index+1)%len(dev)]), scope=scope)
        shuffled = replace(batch, slots=batch.slots.flip(0), pins=(), store=None,
                           training=True, producer_version="labelled_shuffled_control")
        for arm in ARMS:
            selected = {"contextual": batch, "selected_exact": batch, "wrong": wrong,
                        "shuffled": shuffled, "absent": None}[arm]
            result = generate(path, backend, row, selected, scope=scope,
                context_tokens=ledger.context_tokens, max_new_tokens=ledger.max_new_tokens,
                exact=arm=="selected_exact")
            correct = score(labels[row["case_id"]], result["decoded"])
            trajectories.append({"arm": arm, "case_id": row["case_id"], "world_id": row["world_id"],
                "correct": correct, "false_recall": labels[row["case_id"]]=="unknown" and not correct,
                **result})
            ledger.charge("readout_cases")
            ledger.charge("generation_requests")
            ledger.charge("verification_calls")
            for name in ("forward_calls", "autoregressive_forwards", "generated_tokens"):
                ledger.charge(name, result["generated_tokens"])
            ledger.charge("processed_text_positions", result["processed_positions"])
        store.close()
        (target / "development.json").write_text(json.dumps(trajectories, indent=2)+"\n")
        if meter is not None: ledger.set_measured("artifact_bytes", meter.observe())
    counts = {a:sum(r["correct"] for r in trajectories if r["arm"]==a) for a in ARMS}
    false = sum(r["false_recall"] for r in trajectories if r["arm"]=="contextual")
    criteria = spec["pass_criteria"]
    useful = (counts["contextual"] >= criteria["correct_min"] and false==0
              and counts["contextual"] >= counts["selected_exact"]-criteria["exact_tolerance_cases"]
              and counts["contextual"] > max(counts["wrong"], counts["absent"], counts["shuffled"]))
    report = {"format": "replica.contextual_readout.result.v2", "status": "evaluated",
        "protocol_sha256": protocol.sha256, "readout_protocol_sha256": protocol.config["contextual_readout"]["sha256"],
        "source_qualification": "preflight_validated", "numerical_qualification": True,
        "baseline_competence": competence["passed"], "learning_utility": useful,
        "promotion": False, "P_learning_allowed": False,
        "next_gate": "fresh_process_readout_continuation_then_P_protocol" if useful else "useful_contextual_readout_required",
        "counts": counts, "cases_per_arm": len(dev), "false_recall": false,
        "producer_version": path.encoder.release(), "extension": manifest,
        "frozen_base_integrity": backend.adapter.base_unchanged(), "readout_frozen": True,
        "work": ledger.report(), "memory_cost": path.cost, "wall_seconds": time.monotonic()-start,
        "process_max_rss_bytes": _ru_maxrss_bytes(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, sys.platform),
        "physical_GPU_residency": "not_inferred_from_directory_or_logical_tier",
        "native_execution": backend.native_admissible, "sealed_final_opened": False,
        "persistent_learning_measured": False, "retention_measured": False,
        "future_reasoning_cost_reduction_measured": False}
    (target / "report.json").write_text(json.dumps(report, indent=2)+"\n")
    return report


def restart_readout(protocol, backend, output, *, ledger=None, meter=None):
    target = Path(output) / "contextual_readout"
    spec = load_readout_spec(protocol)
    validate_receipt(protocol, json.loads((Path(output)/"competence/receipt.json").read_text()))
    report = json.loads((target / "report.json").read_text())
    if report["protocol_sha256"] != protocol.sha256:
        raise ValueError("readout restart protocol changed")
    path = ContextualMemoryPath(backend.adapter, lr=spec["learning_rate"])
    path.load(target / "extension")
    canary = json.loads((target / "reload_canary.json").read_text())
    row = json.loads((protocol.root / spec["files"]["fit"]["path"]).read_text())["cases"][0]
    scope = MemoryScope("readout", "main", "assistant", as_of=10)
    ids,mask,_ = backend.encode_training(prompt_for(row), row["answer"], canary["context_tokens"])
    with torch.no_grad():
        logits = path.forward(ids,mask,path.training_values(spans_for(row),scope=scope),scope=scope)
    if tensor_digest(logits) != canary["logits_sha256"]:
        raise ValueError("fresh-process readout logits differ")
    update = response_step(path,backend,row,context_tokens=canary["context_tokens"])
    if ledger is not None:
        for name in ("restarts", "reloads", "readout_steps", "training_forwards",
                     "backward_calls", "optimizer_steps"):
            ledger.charge(name)
        ledger.charge("forward_calls", 2)
        ledger.charge("processed_text_positions", int(ids.shape[1]) + update["positions"])
        ledger.charge("backward_processed_positions", update["positions"])
    result = {"format":"replica.contextual_readout.restart.v2", "protocol_sha256":protocol.sha256,
              "exact_logits":True, "actual_continuation_update":update,
              "base_unchanged":backend.adapter.base_unchanged(), "promotion":False}
    if ledger is not None: result["work"] = ledger.report()
    (target / "restart.json").write_text(json.dumps(result,indent=2)+"\n")
    if meter is not None: meter.observe()
    return result

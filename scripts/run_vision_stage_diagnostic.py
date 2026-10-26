#!/usr/bin/env python3
"""Bounded stage-wise fused-projector diagnostic with fresh-process reload."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


sys.path.insert(0, str(ROOT / "scripts"))
def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def _write_jsonl(path, rows):
    Path(path).write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in rows))


def _verify_prepared(protocol_path):
    from replica_lab.vision_curriculum import verify_implementation_pins
    protocol = json.loads(Path(protocol_path).read_text())
    root = Path(protocol["root"])
    data_root = root / protocol["data"]["root"]
    manifest_path = data_root / "manifest.json"
    if _sha(manifest_path) != protocol["data"]["manifest"]["sha256"]:
        raise ValueError("Prepared manifest bytes differ")
    manifest = json.loads(manifest_path.read_text())
    if manifest["seals"] != protocol["data"]["seals"] or manifest["counts"] != protocol["data"]["counts"]:
        raise ValueError("Prepared seals or counts differ")
    for relative, pin in manifest["files"].items():
        path = data_root / relative
        if not path.is_file() or _sha(path) != pin["sha256"]:
            raise ValueError(f"Prepared payload differs: {relative}")
    implementation = verify_implementation_pins(protocol, ROOT)
    return protocol, manifest, implementation


def _batches(rows, updates, batch, seed, tag):
    from replica_lab.vision_curriculum import _cycle_batches
    return _cycle_batches(rows, seed=seed, updates=updates, batch=batch, tag=tag)


def _projector_snapshot(adapter):
    return {name: parameter.detach().cpu().clone()
            for name, parameter in adapter.vision_projector.named_parameters()}


def _delta_norm(before, after):
    import torch
    return float(sum(float((after[name] - before[name]).square().sum())
                     for name in before) ** 0.5)


def run_updates(adapter, session, tokenizer, service, protocol, scope, batches,
                optimizer, trace_path, *, label, start_time):
    import torch
    from run_vision_curriculum import (
        _example_tensors, curriculum_prompt, observe, task_graph)
    train = protocol["train"]
    records = []
    for update_index, rows in enumerate(batches, 1):
        if time.monotonic() - start_time > train["max_seconds"]:
            raise RuntimeError("Stage exceeded its wall budget")
        optimizer.zero_grad(set_to_none=True)
        before = _projector_snapshot(adapter)
        losses, gradient_names = [], set()
        output_rms, output_max = [], []
        processed = supervised = 0
        for row in rows:
            visual = observe(
                service, image_path=row["image_path"], image_sha256=row["image_sha256"],
                case_id=row["case_id"], scope=scope, ledger_epoch=session.ledger.epoch,
                spec=service.spec, source_id=protocol["vision_source"]["source_id"])
            with torch.no_grad():
                projected = adapter.project_visual(visual)
                output_rms.append(float(projected.square().mean().sqrt()))
                output_max.append(float(projected.abs().max()))
            prompt_tensor, input_ids, labels = _example_tensors(
                session, tokenizer, row,
                maximum_prompt=protocol["limits"]["max_prompt_tokens"],
                maximum_sequence=protocol["limits"]["max_sequence_tokens"])
            prompt_text = curriculum_prompt(row["question"])
            with session.training_request(
                    prompt_ids=prompt_tensor,
                    prompt_mask=torch.ones_like(prompt_tensor, dtype=torch.bool),
                    query=prompt_text, scope=scope,
                    task=task_graph(row["case_id"], prompt_text), visual=visual,
                    enabled_ports=(), core_impl="none", enabled_branches=("memory",)) as request:
                result = session.forward(
                    request, input_ids,
                    attention_mask=torch.ones_like(input_ids, dtype=torch.bool), labels=labels)
                loss = result.loss
                if loss is None or not bool(torch.isfinite(loss)) or not loss.requires_grad:
                    raise RuntimeError("Training loss is nonfinite or detached")
                session.backward(request, loss / len(rows))
            losses.append(float(loss.detach()))
            processed += int(input_ids.numel())
            supervised += int((labels[:, 1:] != -100).sum())
            for name, parameter in adapter.named_parameters():
                if parameter.grad is not None:
                    gradient_names.add(name)
        nonprojector = sorted(name for name in gradient_names
                              if not name.startswith("vision_projector."))
        if nonprojector:
            raise RuntimeError(f"Non-projector gradients: {nonprojector}")
        grad_norm = float(torch.nn.utils.clip_grad_norm_(
            adapter.vision_projector.parameters(), train["max_grad_norm"],
            error_if_nonfinite=True).detach().cpu())
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        after = _projector_snapshot(adapter)
        record = {
            "stage": label, "update": update_index,
            "case_ids": [row["case_id"] for row in rows],
            "sources": [row["source"] for row in rows],
            "loss": sum(losses) / len(losses), "losses": losses,
            "grad_norm": grad_norm,
            "projector_update_norm": _delta_norm(before, after),
            "projector_output_rms": sum(output_rms) / len(output_rms),
            "projector_output_max": max(output_max),
            "processed_tokens": processed, "supervised_tokens": supervised,
            "response_only_labels": True,
            "nonprojector_gradient_names": nonprojector,
            "base_stamp_unchanged": (
                adapter.parameter_stamp(original_only=True) == adapter._initial_base_stamp),
        }
        if not record["base_stamp_unchanged"]:
            raise RuntimeError("Training changed the sealed base")
        records.append(record)
        _write_jsonl(trace_path, records)
    return records


def evaluate_stage(adapter, session, tokenizer, service, protocol, scope,
                   cases, labels, caption_cases, caption_labels, output, *,
                   stage, seed_offset, guard, start_time):
    from replica_lab.vision_curriculum import normalise_answer, score_caption, score_procedural
    from run_vision_curriculum import caption_evaluation, evaluate_generations
    generations = evaluate_generations(
        adapter, session, tokenizer, service, protocol, scope, cases,
        output / f"{stage}_generations.jsonl", seed_offset=seed_offset,
        guard=guard, start_time=start_time)
    procedural = score_procedural(
        cases, generations, controls=protocol["evaluation"]["controls"],
        seed=protocol["seed"],
        bootstrap_resamples=protocol["evaluation"]["bootstrap"]["resamples"],
        bootstrap_confidence=protocol["evaluation"]["bootstrap"]["confidence"],
        gates=protocol["evaluation"]["gates"])
    caption_generations = caption_evaluation(
        adapter, session, tokenizer, service, protocol, scope, caption_cases,
        output / f"{stage}_caption_generations.jsonl", guard=guard,
        start_time=start_time)
    caption = score_caption(caption_labels, caption_generations)
    distribution = {
        "correct": dict(Counter(normalise_answer(row["generated_text"])
                                for row in generations if row["control"] == "correct")),
        "wrong": dict(Counter(normalise_answer(row["generated_text"])
                              for row in generations if row["control"] == "wrong")),
        "noimage": dict(Counter(normalise_answer(row["generated_text"])
                                for row in generations if row["control"] == "noimage")),
    }
    result = {"stage": stage, "procedural": procedural, "caption": caption,
              "answer_distribution": distribution}
    (output / f"{stage}_score.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def _forgetting(stage_a, stage_b, retention_a, retention_b, evaluation):
    correct_a = stage_a["procedural"]["controls"]["correct"]
    correct_b = stage_b["procedural"]["controls"]["correct"]
    caption_a = stage_a["caption"]
    caption_b = stage_b["caption"]
    paired_a = stage_a["procedural"]["paired"]
    paired_b = stage_b["procedural"]["paired"]
    unknown_a = stage_a["procedural"]["unknown"]
    unknown_b = stage_b["procedural"]["unknown"]
    alerts = {
        "caption_f1_drop": (caption_a["token_f1"] - caption_b["token_f1"]
                            > evaluation["max_caption_f1_forgetting_drop"]),
        "procedural_accuracy_drop": (correct_a["accuracy"] - correct_b["accuracy"]
                                     > evaluation["max_procedural_accuracy_forgetting_drop"]),
        "text_retention_drop": (retention_a["score"]["accuracy"] - retention_b["score"]["accuracy"]
                                > evaluation["max_text_retention_drop"]),
    }
    per_task = {}
    for task in sorted(set(correct_a["per_task"]) | set(correct_b["per_task"])):
        a = correct_a["per_task"].get(task, {}).get("accuracy")
        b = correct_b["per_task"].get(task, {}).get("accuracy")
        per_task[task] = {"stage_a": a, "stage_b": b,
                          "delta": None if a is None or b is None else b - a}
    return {
        "caption_token_f1_delta": caption_b["token_f1"] - caption_a["token_f1"],
        "caption_response_loss_delta": (caption_b["teacher_forced_response_loss"]
                                        - caption_a["teacher_forced_response_loss"]),
        "procedural_accuracy_delta": correct_b["accuracy"] - correct_a["accuracy"],
        "unknown_accuracy_delta": (unknown_b["abstention_accuracy"]
                                   - unknown_a["abstention_accuracy"]),
        "margins": {name: paired_b[name]["correct_margin"] - paired_a[name]["correct_margin"]
                    for name in ("wrong", "shuffled", "noimage")},
        "per_task": per_task,
        "alerts": alerts,
        "caption_behavior_destroyed": alerts["caption_f1_drop"],
    }


def _audit_receipt(protocol, run_output, *, create):
    path = ROOT / "reports/vision_stage_diagnostic_20261026" / (
        f"audit_{protocol['data']['seals']['audit_case_ids'][:16]}.reserve.json")
    if create:
        if path.exists():
            raise RuntimeError("Fresh audit namespace has already been consumed")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "protocol_sha256": _sha(run_output / "protocol.json"),
            "run": str(run_output), "consumed": True,
        }, indent=2) + "\n")
    elif not path.exists():
        raise RuntimeError("Reload lacks the original audit reservation")
    return str(path)


def execute_train(args):
    import torch
    from replica_lab.architecture.loading import exclusive_mps_worker, load_local_mps
    from replica_lab.optimizers import build_optimizer
    from replica_lab.profiling import RunProfiler
    from replica_lab.vision_curriculum import score_caption
    from run_vision_curriculum import (
        Guard, backward_only_probe, evaluate_retention, export_extension,
        feature_service, freeze_projector_only, initialize_projector,
        make_session, protocol_scope, verify_source_and_spec)
    protocol_path = Path(args.protocol).resolve()
    protocol, manifest, implementation = _verify_prepared(protocol_path)
    data_root = Path(protocol["root"]) / protocol["data"]["root"]
    output = Path(args.output)
    guard = Guard(protocol)
    scope = protocol_scope(protocol)
    started = time.monotonic()
    (output / "protocol.json").write_text(protocol_path.read_text())
    with exclusive_mps_worker(ROOT):
        profiler = RunProfiler("mps", output)
        profiler.start()
        try:
            source, spec, _ = verify_source_and_spec(protocol, "mps")
            adapter, tokenizer, lock = load_local_mps(
                args.model_path, args.model_lock, vision=spec)
            guard.sample(index=0, event="model_loaded")
            freeze_projector_only(adapter)
            adapter._initial_base_stamp = adapter.parameter_stamp(original_only=True)
            init_hash = initialize_projector(adapter.vision_projector,
                                             protocol["train"]["init_seed"])
            if init_hash in set(protocol["train"]["forbidden_parent_weights_sha256"]):
                raise RuntimeError("Fresh initialization matches rejected parent")
            session = make_session(adapter, tokenizer, output)
            service = feature_service(protocol, "mps")
            guard.sample(index=0, event="feature_service")
            train_a = _read_jsonl(data_root / "training/stage_a.jsonl")
            train_b = _read_jsonl(data_root / "training/stage_b.jsonl")
            for row in train_a + train_b:
                row["image_path"] = data_root / row["image"]
            dev = _read_jsonl(data_root / "evaluation/dev_cases.jsonl")
            dev_labels = _read_jsonl(data_root / "evaluation/dev_labels.jsonl")
            caption_dev = _read_jsonl(data_root / "evaluation/caption_dev_cases.jsonl")
            caption_dev_labels = _read_jsonl(data_root / "evaluation/caption_dev_labels.jsonl")
            for rows in (dev, caption_dev):
                for row in rows:
                    row["image_path"] = data_root / row["image"]
            batches_a = _batches(train_a, protocol["schedule"]["stage_a"]["updates"],
                                 protocol["schedule"]["stage_a"]["batch"],
                                 protocol["train"]["seed"], "stage_a")
            batches_b = _batches(train_b, protocol["schedule"]["stage_b"]["updates"],
                                 protocol["schedule"]["stage_b"]["batch"],
                                 protocol["train"]["seed"], "stage_b")
            optimizer = build_optimizer(
                adapter, "adamw_fused", lr_base=0.0, lr_new=protocol["train"]["lr"],
                weight_decay=protocol["train"]["weight_decay"])
            probe_row = dict(dev[0], response=dev[0]["label"], image_path=dev[0]["image_path"])
            backward = backward_only_probe(
                adapter, session, tokenizer, service, protocol, scope, probe_row)
            initialize_projector(adapter.vision_projector, protocol["train"]["init_seed"])
            adapter._initial_base_stamp = adapter.parameter_stamp(original_only=True)
            retention_pre = evaluate_retention(
                adapter, session, tokenizer, protocol, output / "retention_pre.jsonl",
                label="before_updates", guard=guard, start_time=started)
            trace_a = run_updates(
                adapter, session, tokenizer, service, protocol, scope, batches_a,
                optimizer, output / "stage_a_updates.jsonl", label="stage_a", start_time=started)
            stage_a_extension = export_extension(
                adapter, output / "stage_a/extension", protocol, label="stage_a")
            retention_a = evaluate_retention(
                adapter, session, tokenizer, protocol, output / "retention_stage_a.jsonl",
                label="after_stage_a", guard=guard, start_time=started)
            dev_a = evaluate_stage(
                adapter, session, tokenizer, service, protocol, scope,
                dev, dev_labels, caption_dev, caption_dev_labels, output,
                stage="stage_a_dev", seed_offset=protocol["seed"] + 17,
                guard=guard, start_time=started)
            trace_b = run_updates(
                adapter, session, tokenizer, service, protocol, scope, batches_b,
                optimizer, output / "stage_b_updates.jsonl", label="stage_b", start_time=started)
            stage_b_extension = export_extension(
                adapter, output / "stage_b/extension", protocol, label="stage_b")
            retention_b = evaluate_retention(
                adapter, session, tokenizer, protocol, output / "retention_stage_b.jsonl",
                label="after_stage_b", guard=guard, start_time=started)
            dev_b = evaluate_stage(
                adapter, session, tokenizer, service, protocol, scope,
                dev, dev_labels, caption_dev, caption_dev_labels, output,
                stage="stage_b_dev", seed_offset=protocol["seed"] + 17,
                guard=guard, start_time=started)
            audit_receipt = _audit_receipt(protocol, output, create=True)
            audit = _read_jsonl(data_root / "evaluation/audit_cases.jsonl")
            audit_labels = _read_jsonl(data_root / "evaluation/audit_labels.jsonl")
            caption_audit = _read_jsonl(data_root / "evaluation/caption_audit_cases.jsonl")
            caption_audit_labels = _read_jsonl(data_root / "evaluation/caption_audit_labels.jsonl")
            for rows in (audit, caption_audit):
                for row in rows:
                    row["image_path"] = data_root / row["image"]
            audit_b = evaluate_stage(
                adapter, session, tokenizer, service, protocol, scope,
                audit, audit_labels, caption_audit, caption_audit_labels, output,
                stage="audit_after_stage_b", seed_offset=protocol["seed"] + 17,
                guard=guard, start_time=started)
            optimizer_audit = optimizer.audit()
            report = {
                "format": "replica.vision_stage_diagnostic.run.v1",
                "status": "completed", "disposition": "pending_reload",
                "protocol_sha256": _sha(protocol_path),
                "implementation_sha256": implementation,
                "data_manifest_sha256": protocol["data"]["manifest"]["sha256"],
                "data_seals": protocol["data"]["seals"],
                "source_revision": lock["revision"],
                "device": "mps", "cpu_fallback_enabled": False,
                "initialization": {"seed": protocol["train"]["init_seed"],
                                   "projector_init_sha256": init_hash,
                                   "warm_start": None},
                "trainable_scope": "vision_projector.",
                "trainable_tensors": 6,
                "optimizer": optimizer_audit,
                "backward_only_probe": backward,
                "stage_a": {"updates": trace_a, "extension": stage_a_extension,
                            "retention": retention_a, "development": dev_a},
                "stage_b": {"updates": trace_b, "extension": stage_b_extension,
                            "retention": retention_b, "development": dev_b,
                            "audit": audit_b},
                "retention_before": retention_pre,
                "forgetting": _forgetting(dev_a, dev_b, retention_a, retention_b,
                                          protocol["evaluation"]),
                "audit_receipt": audit_receipt,
                "total_updates": len(trace_a) + len(trace_b),
                "total_examples": sum(len(row["case_ids"]) for row in trace_a + trace_b),
                "base_saved": False, "promotion": False,
                "profile": profiler.stop(),
            }
            (output / "run_report.json").write_text(json.dumps(report, indent=2) + "\n")
            (output / "protocol.json").write_text(protocol_path.read_text())
            print(json.dumps({"status": report["status"],
                              "updates": report["total_updates"],
                              "disposition": report["disposition"],
                              "report": str(output / "run_report.json")}))
            return report
        except BaseException as error:
            failure = {"status": "failed", "error_type": type(error).__name__,
                       "error": str(error), "traceback": traceback.format_exc(),
                       "profile": profiler.stop()}
            (output / "failure_report.json").write_text(json.dumps(failure, indent=2) + "\n")
            raise


def execute_reload(args):
    from replica_lab.architecture.loading import exclusive_mps_worker, load_local_mps
    from replica_lab.profiling import RunProfiler
    from run_vision_curriculum import (
        Guard, evaluate_retention, feature_service, freeze_projector_only,
        make_session, protocol_scope, verify_source_and_spec)
    protocol_path = Path(args.protocol).resolve()
    protocol, manifest, implementation = _verify_prepared(protocol_path)
    run = Path(args.run)
    prior = json.loads((run / "run_report.json").read_text())
    if prior["protocol_sha256"] != _sha(protocol_path):
        raise ValueError("Reload protocol differs")
    output = Path(args.output)
    guard = Guard(protocol)
    scope = protocol_scope(protocol)
    with exclusive_mps_worker(ROOT):
        profiler = RunProfiler("mps", output)
        profiler.start()
        try:
            _, spec, _ = verify_source_and_spec(protocol, "mps")
            adapter, tokenizer, lock = load_local_mps(
                args.model_path, args.model_lock, vision=spec)
            freeze_projector_only(adapter)
            adapter.load_extension(run / "stage_b/extension")
            session = make_session(adapter, tokenizer, output)
            service = feature_service(protocol, "mps")
            audit = _read_jsonl(Path(protocol["root"]) / protocol["data"]["root"] / "evaluation/audit_cases.jsonl")
            audit_labels = _read_jsonl(Path(protocol["root"]) / protocol["data"]["root"] / "evaluation/audit_labels.jsonl")
            caption_audit = _read_jsonl(Path(protocol["root"]) / protocol["data"]["root"] / "evaluation/caption_audit_cases.jsonl")
            caption_audit_labels = _read_jsonl(Path(protocol["root"]) / protocol["data"]["root"] / "evaluation/caption_audit_labels.jsonl")
            for rows in (audit, caption_audit):
                for row in rows:
                    row["image_path"] = Path(protocol["root"]) / protocol["data"]["root"] / row["image"]
            started = time.monotonic()
            audit_receipt = _audit_receipt(protocol, run, create=False)
            reloaded = evaluate_stage(
                adapter, session, tokenizer, service, protocol, scope,
                audit, audit_labels, caption_audit, caption_audit_labels, output,
                stage="audit_after_stage_b", seed_offset=protocol["seed"] + 17,
                guard=guard, start_time=started)
            retention = evaluate_retention(
                adapter, session, tokenizer, protocol, output / "retention_stage_b.jsonl",
                label="after_stage_b", guard=guard, start_time=started)
            comparisons = {}
            for name in ("audit_after_stage_b_generations.jsonl",
                         "audit_after_stage_b_caption_generations.jsonl",
                         "retention_stage_b.jsonl"):
                comparisons[name] = _sha(run / name) == _sha(output / name)
            if not all(comparisons.values()):
                raise RuntimeError(f"Fresh-process outputs differ: {comparisons}")
            report = {
                "format": "replica.vision_stage_diagnostic.reload.v1",
                "status": "completed", "zero_updates": True,
                "protocol_sha256": _sha(protocol_path),
                "run_report_sha256": _sha(run / "run_report.json"),
                "extension_weights_sha256": prior["stage_b"]["extension"]["weights_sha256"],
                "comparisons": comparisons, "bitwise_equal": True,
                "audit_receipt": audit_receipt,
                "disposition": prior["disposition"],
                "reloaded_audit": reloaded,
                "retention": retention,
                "profile": profiler.stop(),
            }
            (output / "reload_report.json").write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps({"status": "completed", "bitwise_equal": True,
                              "report": str(output / "reload_report.json")}))
            return report
        except BaseException as error:
            failure = {"status": "failed", "error_type": type(error).__name__,
                       "error": str(error), "traceback": traceback.format_exc(),
                       "profile": profiler.stop()}
            (output / "failure_report.json").write_text(json.dumps(failure, indent=2) + "\n")
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("train", "reload"))
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--device", choices=("mps",), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-lock", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("REPLICA_CAPACITY_GUARD") != "1":
        parser.error("Use the capacity guard")
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0":
        parser.error("CPU fallback must be disabled")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Use fresh output")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.mode == "train":
        execute_train(args)
    else:
        if args.run is None:
            parser.error("reload requires --run")
        execute_reload(args)


if __name__ == "__main__":
    main()

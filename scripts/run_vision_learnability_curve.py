#!/usr/bin/env python3
"""Training-only bounded vision learning curve.

One native model load, one fresh seed-29 projector, frozen backbone/encoder and
native fused AdamW (lr 3e-4, weight decay 0.01) produce five readouts:

* prefix 0 after the two backward-only probes and before any optimizer step;
* prefixes 1, 2, 4 and 7 after the corresponding cumulative update.

Every readout scores all 26 training rows against the four matched controls
(104 records, 98 available, 6 declared placeholders), computes one greedy
hypothesis and prefill analysis per available cell, and computes teacher-forced
loss for both answers and captions through the same path.  Prefix 7 also reports
the 22 train-fit gates as endpoint context only.

The curve decision is T1/T2/T3 only.  No status authorizes development,
held-out, V0, promotion or retry.  One exclusive MPS worker, the capacity guard
and the profiler are mandatory.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from replica_lab import vision_learnability_curve as curve  # noqa: E402
from replica_lab import vision_learnability_diagnostic as learn  # noqa: E402
from replica_lab.vision_collapse_diagnostic import (  # noqa: E402
    CONTROL_NAMES, EXPECTED_STOP_TOKENS, control_plan, fit_coverage,
    require_fit_coverage, resolve_stop_tokens)
from replica_lab.vision_curriculum import (  # noqa: E402
    PROJECTOR_PARAMETER_COUNT, PROJECTOR_TENSORS)
import run_vision_balanced_alignment as balanced  # noqa: E402
import run_vision_learnability_diagnostic as base  # noqa: E402


_sha = base._sha
_read_json = base._read_json
_read_jsonl = base._read_jsonl
_write_json = base._write_json
_write_jsonl = base._write_jsonl
_now = base._now
_check_wall = base._check_wall
LoadAccounting = base.LoadAccounting
_accounted_load = base._accounted_load
_snapshot_new_parameters = base._snapshot_new_parameters
_changed = base._changed


def verify_pins(config, root=ROOT):
    """Reuse the historical pin verifier for the curve declaration."""
    learn.require(config.get("mode") == curve.MODE,
                  "The curve runner requires a curve config")
    pins = base.verify_pins(config, root)
    learn.require(pins["pins"]["mode"] == curve.MODE,
                  "The pinned curve config was mislabelled")
    return pins


def _readout_directory(output, prefix):
    return Path(output) / "readouts" / f"prefix_{int(prefix):03d}"


def _curve_readout(
        adapter, session, tokenizer, service, protocol, scope, rows, plan,
        labels, stops, *, prefix, output, guard, started, limits, init_hash,
        optimizer):
    """Score one prefix without writing weights, optimizer state or base."""
    import torch
    from run_vision_curriculum import (
        curriculum_prompt, generate_row, observe, projector_state_sha256)
    from run_vision_balanced_alignment import _teacher_forced_loss

    rows_by_case = {str(row["case_id"]): row for row in rows}
    caption_labels = {str(row["case_id"]): row for row in labels}
    readout_dir = _readout_directory(output, prefix)
    readout_dir.mkdir(parents=True, exist_ok=True)

    before_state = curve.optimizer_state_report(optimizer)
    before_digest = projector_state_sha256(adapter.vision_projector)
    before_base = adapter.parameter_stamp(original_only=True)
    before_snapshot = _snapshot_new_parameters(adapter)
    answer_rows = []
    caption_rows = []
    normalised = []
    counts = {
        "records": 0,
        "available": 0,
        "placeholders": 0,
        "generations": 0,
        "prefill_forwards": 0,
        "teacher_forced_cells": 0,
        "scoring_forwards": 0,
    }
    adapter.eval()
    try:
        for position, row in enumerate(rows, 1):
            _check_wall(started, limits["max_seconds"], "curve readout")
            if guard is not None and position % 4 == 0:
                guard.sample(index=position, event=f"curve_readout_{prefix}")
            case_id = str(row["case_id"])
            plan_row = plan[case_id]
            for control in CONTROL_NAMES:
                counts["records"] += 1
                image_case = {
                    "correct": plan_row["correct_case_id"],
                    "sibling": plan_row["sibling_case_id"],
                    "shuffled": plan_row["shuffled_case_id"],
                    "noimage": None,
                }[control]
                if control == "sibling" and not plan_row["sibling_available"]:
                    counts["placeholders"] += 1
                    normalised.append({
                        "case_id": case_id,
                        "control": control,
                        "kind": row["kind"],
                        "available": False,
                        "generated_ids": [],
                        "generated_text": None,
                        "prefill_argmax_matches_first_token": False,
                        "hypothesis_count": 1,
                        "stop_tokens": [],
                    })
                    continue
                visual = None
                if image_case is not None:
                    image_row = rows_by_case[str(image_case)]
                    visual = observe(
                        service, image_path=image_row["image_path"],
                        image_sha256=image_row["image_sha256"],
                        case_id=case_id, scope=scope,
                        ledger_epoch=session.ledger.epoch,
                        spec=service.spec,
                        source_id=protocol["vision_source"]["source_id"])
                maximum_new = (int(limits["caption_max_new_tokens"])
                               if row["kind"] == "caption"
                               else int(limits["max_new_tokens"]))
                generated = generate_row(
                    session, tokenizer, control=control, case=row,
                    question=curriculum_prompt(row["question"]),
                    visual=visual, scope=scope,
                    maximum_prompt=limits["max_prompt_tokens"],
                    maximum_new=maximum_new,
                    image_case_id=image_case,
                    image_sha256=(None if image_case is None
                                  else rows_by_case[str(image_case)]
                                  ["image_sha256"]))
                prefill = base._prefill_analysis(
                    session, tokenizer, row, visual, scope, limits=limits)
                prefill_matches = bool(generated["generated_ids"]) and (
                    prefill["argmax_token"]
                    == int(generated["generated_ids"][0]))
                if row["kind"] == "answer":
                    target_text = row["response"]
                else:
                    target_text = caption_labels[case_id]["answer"]
                loss = _teacher_forced_loss(
                    session, tokenizer, protocol, scope, visual, row,
                    row["question"], target_text)
                record = {
                    "format": curve.READOUT_FORMAT + ".generation",
                    **generated,
                    "case_id": case_id,
                    "kind": row["kind"],
                    "available": True,
                    "hypothesis_count": 1,
                    "prefill": prefill,
                    "prefill_argmax_matches_first_token": prefill_matches,
                    "teacher_forced_loss": loss,
                    "target_text": target_text,
                }
                if row["kind"] == "caption":
                    record["target_visible_objects"] = [
                        list(item)
                        for item in caption_labels[case_id]["visible_objects"]]
                    caption_rows.append(record)
                else:
                    answer_rows.append(record)
                normalised.append({
                    "case_id": case_id,
                    "control": control,
                    "kind": row["kind"],
                    "available": True,
                    "generated_ids": list(generated["generated_ids"]),
                    "generated_text": generated["generated_text"],
                    "prefill_argmax_matches_first_token": prefill_matches,
                    "prefill": prefill,
                    "teacher_forced_loss": loss,
                    "target_text": target_text,
                    "target_visible_objects": record.get(
                        "target_visible_objects"),
                    "hypothesis_count": 1,
                    "stop_tokens": list(generated["stop_tokens"]),
                })
                counts["available"] += 1
                counts["generations"] += 1
                counts["prefill_forwards"] += 1
                counts["teacher_forced_cells"] += 1
                counts["scoring_forwards"] += 3
                learn.require(
                    counts["scoring_forwards"]
                    <= int(limits["max_scoring_forwards"]),
                    "The scoring-forward cap was reached")
                learn.require(
                    counts["generations"]
                    <= learn.EXPECTED_GENERATIONS,
                    "The scoring-generation cap was reached")
    finally:
        adapter.train()
    after_state = curve.optimizer_state_report(optimizer)
    after_digest = projector_state_sha256(adapter.vision_projector)
    after_base = adapter.parameter_stamp(original_only=True)
    weight_changes = _changed(before_snapshot,
                              _snapshot_new_parameters(adapter))
    generated_report = curve.generation_hypothesis_report(
        normalised,
        answer_max_new_tokens=int(limits["max_new_tokens"]),
        caption_max_new_tokens=int(limits["caption_max_new_tokens"]))
    learn.require(generated_report["ok"],
                  f"Generation evidence failed: "
                  f"{generated_report['violations']}")
    scores = learn.score_train_fit(
        records=normalised, plan=plan, rows_by_case=rows_by_case,
        caption_labels=caption_labels)
    metrics = curve.readout_metrics(normalised, scores)
    aggregates = curve.teacher_forced_aggregates(normalised)
    gates = None
    if int(prefix) == 7:
        gates = {
            **learn.train_fit_gate_report(
                answer=scores["answer"], caption=scores["caption"],
                gates=learn.GATE_THRESHOLDS),
            "endpoint_context_only": True,
            "curve_decision_input": False,
        }
    readout = {
        "format": curve.READOUT_FORMAT,
        "mode": curve.MODE,
        "prefix": int(prefix),
        "optimizer_steps": int(after_state["max_step"]),
        "optimizer_steps_before": int(before_state["max_step"]),
        "optimizer_state_before": before_state,
        "optimizer_state_after": after_state,
        "optimizer_state_unchanged": (
            before_state["sha256"] == after_state["sha256"]
            and before_state["max_step"] == after_state["max_step"]
            and before_state["state_entries"] == after_state["state_entries"]),
        "optimizer_state_empty": bool(after_state["empty"]),
        "projector_state_sha256_before": before_digest,
        "projector_state_sha256_after": after_digest,
        "projector_state_sha256": after_digest,
        "projector_unchanged_by_readout": before_digest == after_digest,
        "projector_digest_matches_initialization": after_digest == init_hash,
        "weight_change_names": weight_changes,
        "readout_weight_writes": 0,
        "base_stamp_before": before_base,
        "base_stamp_after": after_base,
        "base_stamp_unchanged": before_base == after_base,
        "adapter_mode_after": "train" if adapter.training else "eval",
        "adapter_train_restored": bool(adapter.training),
        "counts": counts,
        "generation": generated_report,
        "scores": scores,
        "teacher_forced_aggregates": aggregates,
        "metrics": metrics,
        "gates": gates,
        "files": {
            "answer_generations": str(
                readout_dir / "train_answer_generations.jsonl"),
            "caption_generations": str(
                readout_dir / "train_caption_generations.jsonl"),
            "records": str(readout_dir / "records.jsonl"),
            "answer_scores": str(readout_dir / "scores/train_fit_answer.json"),
            "caption_scores": str(
                readout_dir / "scores/train_fit_caption.json"),
            "gates": (None if gates is None
                      else str(readout_dir / "gates/train_fit.json")),
        },
        "scope": ("one readout at this cumulative prefix; scoring never "
                  "writes weights, optimizer state or the sealed base"),
    }
    _write_jsonl(readout_dir / "train_answer_generations.jsonl", answer_rows)
    _write_jsonl(readout_dir / "train_caption_generations.jsonl",
                 caption_rows)
    _write_jsonl(readout_dir / "records.jsonl", normalised)
    _write_json(readout_dir / "scores/train_fit_answer.json", {
        "format": curve.READOUT_FORMAT + ".answer_score",
        **scores["answer"],
    })
    _write_json(readout_dir / "scores/train_fit_caption.json", {
        "format": curve.READOUT_FORMAT + ".caption_score",
        **scores["caption"],
    })
    if gates is not None:
        _write_json(readout_dir / "gates/train_fit.json", gates)
    _write_json(readout_dir / "readout.json", readout)
    return readout


def _failure_report(output, exc, *, stop_report=None, partial=None):
    payload = {
        "format": curve.FAILURE_FORMAT,
        "generated_at_utc": _now(),
        "mode": curve.MODE,
        "status": "failed",
        "error_type": type(exc).__name__,
        "error": str(exc),
        "traceback": traceback.format_exc(),
        "partial": partial or {},
        "stop_conditions": stop_report,
        "claims": curve.claims_block(),
    }
    try:
        _write_json(Path(output) / "failure_report.json", payload)
    except Exception:  # noqa: BLE001 - never mask the original failure
        pass


def execute_curve(args, config, pins):
    import torch
    from replica_lab.architecture.loading import (
        exclusive_mps_worker, load_local_mps)
    from replica_lab.optimizers import NativeFusedAdamW
    from replica_lab.optimizers import build_optimizer as build_native_optimizer
    from replica_lab.profiling import RunProfiler
    from replica_lab.vision_balanced_alignment import (
        scorer_contract_preflight_v2)
    from run_vision_curriculum import (
        Guard, feature_service, freeze_projector_only, initialize_projector,
        make_session, protocol_scope, projector_state_sha256,
        verify_source_and_spec)

    protocol = pins["protocol"]
    output = Path(args.output)
    limits = config["limits"]
    started = time.monotonic()
    guard = None
    accounting = LoadAccounting()
    trace_path = output / "training_trace.jsonl"
    trace: list[dict] = []
    readouts: list[dict] = []
    update_backward_calls = 0
    optimizer_steps = 0
    init_hash = None
    readout_weight_writes = 0
    _write_json(output / "config.json", config)
    (output / "protocol.json").write_bytes(pins["protocol_path"].read_bytes())
    accessed: list[str] = []
    try:
        learn.require(Path(args.model_path).resolve()
                      == (ROOT / config["source"]["model_path"]).resolve(),
                      "CLI model path differs from the pinned source")
        learn.require(Path(args.model_lock).resolve()
                      == (ROOT / config["source"]["model_lock"]).resolve(),
                      "CLI model lock differs from the pinned lock")
        precondition_path = ROOT / config["precondition"]["report"]
        learn.require(precondition_path.is_file(),
                      "The preserved ablation receipt is missing")
        precondition_receipt = _read_json(precondition_path)
        precondition = learn.ablation_precondition_report(
            precondition_receipt,
            required_decision=config["precondition"]["required_decision"],
            required_mode=config["precondition"]["required_mode"],
            config_sha256=config["precondition"]["config_sha256"])
        learn.require(precondition["ok"],
                      f"The curve is refused: {precondition['violations']}")
        order = learn.train_only_access_order()
        order.access("train")
        training = balanced._load_training_v2(protocol)
        accessed.extend([
            str(pins["prepared_root"] / "training/alignment.jsonl"),
            str(pins["prepared_root"] / config["schedule"]["source"]),
            str(pins["prepared_root"] / config["schedule"]["caption_labels"]),
        ])
        accessed.extend(str(row["image_path"])
                        for row in training["training"])
        schedule_report = learn.prefix_schedule_report(
            training["batches"],
            prefix_updates=int(config["schedule"]["prefix_updates"]),
            batch_size=int(config["schedule"]["batch_size"]),
            answer_per_update=int(config["schedule"]["answer_per_update"]),
            caption_per_update=int(config["schedule"]["caption_per_update"]),
            expected_rows=int(config["schedule"]["unique_rows"]),
            expected_answer_rows=int(
                config["schedule"]["expected_answer_rows"]),
            expected_caption_rows=int(
                config["schedule"]["expected_caption_rows"]),
            expected_repeated_rows=int(
                config["schedule"]["expected_repeated_rows"]))
        learn.require_prefix_schedule(schedule_report)
        prefix = training["batches"][
            :int(config["schedule"]["prefix_updates"])]
        learn.require(len(prefix) == curve.EXPECTED_UPDATES,
                      "The prefix does not contain seven updates")
        for batch in prefix:
            learn.require(len(batch) == 4, "A prefix batch is not four rows")
        plan = control_plan(training["training"])
        guard = Guard(protocol)
        scope = protocol_scope(protocol)
        with exclusive_mps_worker(ROOT):
            profiler = RunProfiler("mps", output)
            profiler.start()
            profiler_stopped = False
            try:
                _, spec, _ = verify_source_and_spec(protocol, "mps")
                adapter, tokenizer, _ = _accounted_load(
                    lambda: load_local_mps(
                        args.model_path, args.model_lock, vision=spec,
                        profiler=profiler), accounting, "model")
                guard.sample(index=0, event="model_loaded")
                profiler.mark("model_loaded")
                stops, stop_source = resolve_stop_tokens(
                    getattr(adapter.base.config, "eos_token_id", None),
                    getattr(tokenizer, "eos_token_id", None))
                learn.require(tuple(stops) == EXPECTED_STOP_TOKENS,
                              f"Model stop set {stops} differs")
                learn.require(int(tokenizer.eos_token_id)
                              == int(config["target_eos_token"]),
                              "Tokenizer EOS differs from the declared target")
                adapter.eval()
                freeze_projector_only(adapter)
                adapter._initial_base_stamp = adapter.parameter_stamp(
                    original_only=True)
                init_hash = initialize_projector(
                    adapter.vision_projector,
                    int(config["initialization"]["init_seed"]))
                learn.require(init_hash not in set(
                    config["initialization"]
                    ["forbidden_parent_weights_sha256"]),
                    "Fresh initialization matches the rejected parent")
                session = make_session(adapter, tokenizer, output)
                service = feature_service(protocol, "mps")
                guard.sample(index=0, event="feature_service")
                scorer_contract = scorer_contract_preflight_v2(
                    protocol["evaluation"])
                _write_json(output / "scores/scorer_contracts.json",
                            scorer_contract)
                optimizer = build_native_optimizer(
                    adapter, "adamw_fused", lr_base=0.0,
                    lr_new=float(config["optimizer"]["lr"]),
                    weight_decay=float(config["optimizer"]["weight_decay"]))
                learn.require(isinstance(optimizer, NativeFusedAdamW),
                              "The curve requires NativeFusedAdamW")
                learn.require(len(optimizer.param_groups) == 1
                              and len(optimizer.param_groups[0]["params"]) == 6
                              and sum(parameter.numel() for parameter in
                                      optimizer.param_groups[0]["params"])
                              == PROJECTOR_PARAMETER_COUNT,
                              "Optimizer ownership differs")
                profiler.mark("optimizer_ready")

                # Two backward-only probes, no optimizer step and no readout
                # before this point.
                probe_rows = {}
                for batch in prefix:
                    for row in batch:
                        probe_rows.setdefault(str(row["kind"]), row)
                probes = []
                probe_snapshot = _snapshot_new_parameters(adapter)
                for role in learn.EXPECTED_PROBE_ROLES:
                    probe = base._backward_probe(
                        adapter, session, tokenizer, service, protocol,
                        scope, probe_rows[role], role=role, stops=stops,
                        limits=limits)
                    probe["optimizer_state_empty"] = len(optimizer.state) == 0
                    probes.append(probe)
                learn.require(not _changed(
                    probe_snapshot, _snapshot_new_parameters(adapter)),
                    "A backward probe changed the weights")
                learn.require(len(optimizer.state) == 0,
                              "A backward probe produced optimizer state")
                probe_report = learn.validate_backward_probes(probes)
                learn.require(probe_report["ok"],
                              f"Backward probes failed: "
                              f"{probe_report['violations']}")
                _write_json(output / "backward_probes.json", probes)
                profiler.mark("probes_complete")

                # Readout 0: after probes, before any optimizer step.
                readout = _curve_readout(
                    adapter, session, tokenizer, service, protocol, scope,
                    training["training"], plan,
                    training["train_caption_labels"], stops, prefix=0,
                    output=output, guard=guard, started=started, limits=limits,
                    init_hash=init_hash, optimizer=optimizer)
                learn.require(readout["optimizer_state_empty"] is True
                              and readout["optimizer_steps"] == 0,
                              "Readout 0 preceded an empty optimizer state")
                learn.require(
                    readout["projector_digest_matches_initialization"] is True,
                    "Readout 0 projector differs from initialization")
                readouts.append(readout)
                order.advance("train_fit")
                order.access("train_fit")

                adapter.train()
                for index, batch in enumerate(prefix, 1):
                    _check_wall(started, limits["max_seconds"],
                                "gradient update")
                    record, calls = base._gradient_update(
                        adapter, session, tokenizer, service, protocol,
                        scope, batch, optimizer, limits, stops, index=index,
                        guard=guard,
                        source_id=protocol["vision_source"]["source_id"])
                    learn.require(not set(
                        record["changed_projector_tensors"])
                        - set(PROJECTOR_TENSORS),
                        "An update changed a non-projector tensor")
                    trace.append(record)
                    update_backward_calls += int(calls)
                    optimizer_steps += 1
                    learn.require(update_backward_calls
                                  <= int(limits["max_update_backward_calls"]),
                                  "The update backward-call cap was reached")
                    _write_jsonl(trace_path, trace)
                    if index in curve.READOUT_UPDATES:
                        order.access("train_fit")
                        readout = _curve_readout(
                            adapter, session, tokenizer, service, protocol,
                            scope, training["training"], plan,
                            training["train_caption_labels"], stops,
                            prefix=index, output=output, guard=guard,
                            started=started, limits=limits,
                            init_hash=init_hash, optimizer=optimizer)
                        readouts.append(readout)
                        readout_weight_writes += int(
                            readout["readout_weight_writes"])
                trace_report = learn.validate_gradient_trace(
                    trace, limits=limits)
                learn.require(trace_report["ok"],
                              f"Gradient trace failed: "
                              f"{trace_report['violations']}")
                learn.require(len(trace) == curve.EXPECTED_UPDATES,
                              "The curve did not complete seven updates")
                learn.require(len(readouts) == curve.EXPECTED_READOUTS,
                              "The curve did not complete five readouts")
                learn.require(len(prefix) == curve.EXPECTED_UPDATES,
                              "The exact seven-update schedule was truncated")
                retained = [check for record in trace
                            for check in record["fit_contract_rows"]]
                coverage = fit_coverage(
                    retained,
                    [str(row["example_id"]) for row in training["training"]])
                require_fit_coverage(coverage, expected_kinds={
                    "answer": protocol["data"]["counts"]["train_answers"],
                    "caption": protocol["data"]["counts"]["train_captions"],
                })
                learn.require(coverage["checked_rows"] == learn.EXPECTED_ROWS,
                              "Fit coverage did not check every unique row")
                learn.require(sorted(coverage["repeated_example_ids"])
                              == sorted(schedule_report[
                                  "repeated_example_ids"]),
                              "Fit coverage repeats differ from the schedule")
                optimizer_audit = optimizer.audit()
                learn.require(optimizer_audit["state_tensor_bytes"] > 0
                              and len(optimizer_audit["groups"]) == 1
                              and optimizer_audit["groups"][0]["tensors"] == 6
                              and optimizer_audit["groups"][0]["parameters"]
                              == PROJECTOR_PARAMETER_COUNT,
                              "Optimizer state/ownership audit differs")
                _write_json(output / "fit_contract_coverage.json", coverage)
                access = learn.training_only_access_report(
                    order.snapshot(), accessed)
                learn.require(access["ok"],
                              f"Access evidence failed: "
                              f"{access['violations']}")
                final = readouts[-1]
                engineering = curve.curve_engineering_report(
                    trace=trace, probes=probes, limits=limits,
                    schedule_report=schedule_report,
                    generation_report=final["generation"],
                    fit_coverage=coverage, access_report=access,
                    precondition=precondition, readouts=readouts,
                    init_hash=init_hash,
                    updates_completed=len(trace),
                    optimizer_steps=optimizer_steps,
                    update_backward_calls=update_backward_calls,
                    probe_backward_calls=learn.EXPECTED_PROBE_BACKWARD_CALLS,
                    readout_weight_writes=readout_weight_writes,
                    expected_kinds={
                        "answer": protocol["data"]["counts"]["train_answers"],
                        "caption": protocol["data"]["counts"]["train_captions"]},
                    decision_contract=config["decision"])
                trajectory = engineering["trajectory"]
                decision = curve.curve_decision(
                    engineering_report=engineering,
                    trajectory_report=trajectory)
                learn.require(accounting.counts()
                              == {"model_loads": 1, "extension_loads": 0},
                              f"Load counts differ: {accounting.counts()}")
                learn.require(not any(
                    path.name == "extensions.safetensors"
                    for path in output.rglob("*")),
                    "The curve wrote an extension")
                profiler.mark("report_ready")
                summary = profiler.stop()
                profiler_stopped = True
                data_bytes = base._data_bytes(output)
                stop_report = curve.curve_stop_conditions_report(
                    wall_seconds=time.monotonic() - started,
                    wall_limit_seconds=limits["max_seconds"],
                    artifact_bytes=data_bytes,
                    artifact_limit_bytes=limits["artifact_bytes"],
                    updates=len(trace), optimizer_steps=optimizer_steps,
                    update_backward_calls=update_backward_calls,
                    probe_backward_calls=learn.EXPECTED_PROBE_BACKWARD_CALLS,
                    model_loads=accounting.counts()["model_loads"],
                    extension_loads=accounting.counts()["extension_loads"],
                    extension_writes=0, checkpoint_writes=0, downloads=0,
                    cpu_fallback=False, sealed_evaluation_access=0,
                    guard_stop_reason=guard.report().get("stop_reason"),
                    readouts=len(readouts),
                    scoring_generations=sum(
                        int(item["counts"]["generations"])
                        for item in readouts),
                    teacher_forced_cells=sum(
                        int(item["counts"]["teacher_forced_cells"])
                        for item in readouts),
                    scoring_forwards=sum(
                        int(item["counts"]["scoring_forwards"])
                        for item in readouts),
                    readout_weight_writes=readout_weight_writes)
                learn.require(stop_report["mode"] == curve.MODE,
                              "The curve stop receipt was labelled incorrectly")
                curve.require_curve_stop_conditions(stop_report)
                if engineering["ok"] is not True:
                    learn.require(False,
                                  "Curve engineering invariants failed: "
                                  f"{engineering['violations']}")
                report = {
                    "format": curve.REPORT_FORMAT,
                    "version": curve.VERSION,
                    "mode": curve.MODE,
                    "status": decision["status"],
                    "output": str(output),
                    "config_sha256": _sha(Path(args.config)),
                    "declared_at_utc": config["declared_at_utc"],
                    "generated_at_utc": _now(),
                    "command": [sys.executable, *sys.argv],
                    "pins": pins["pins"],
                    "policy_declared": config["policy"],
                    "limits": limits,
                    "ablation_precondition": precondition,
                    "generation_stop_set": {
                        "stop_tokens": list(stops),
                        "source": stop_source,
                        "expected": list(EXPECTED_STOP_TOKENS),
                        "target_eos_token": int(config["target_eos_token"]),
                    },
                    "initialization": {
                        "init_seed": int(
                            config["initialization"]["init_seed"]),
                        "projector_init_sha256": init_hash,
                        "warm_start": None,
                        "forbidden_parent_checked": True,
                    },
                    "trainable_scope": "vision_projector.",
                    "trainable_tensors": len(PROJECTOR_TENSORS),
                    "trainable_parameters": PROJECTOR_PARAMETER_COUNT,
                    "optimizer": optimizer_audit,
                    "native_fused_adamw": True,
                    "backward_only_probes": probes,
                    "training": {
                        "updates": trace,
                        "total_updates": len(trace),
                        "total_examples": sum(len(row["example_ids"])
                                              for row in trace),
                        "schedule": schedule_report,
                        "fit_contract": coverage,
                        "update_backward_calls": update_backward_calls,
                        "optimizer_steps": optimizer_steps,
                    },
                    "readout_schedule": {
                        "prefixes": list(curve.PREFIXES),
                        "readouts": len(readouts),
                        "zero_prefix": "after_probes_before_updates",
                        "after_updates": list(curve.READOUT_UPDATES),
                        "exact_seven_update_required": True,
                        "truncation_at_update_4_forbidden": True,
                        "access_advance_after":
                            config["access"]["advance_after"],
                        "readout_weight_writes": readout_weight_writes,
                    },
                    "readouts": readouts,
                    "curve_trajectory": trajectory,
                    "engineering": engineering,
                    "generation_evidence": final["generation"],
                    "prefix7_endpoint_context": {
                        "scores": final["scores"],
                        "gates": final["gates"],
                        "endpoint_context_only": True,
                        "curve_decision_input": False,
                        "note": ("the 22 train-fit gates are endpoint context "
                                 "only and do not decide the curve"),
                    },
                    "access": access,
                    "decision": decision,
                    "resource": {
                        "guard": guard.report(),
                        "profile": summary,
                    },
                    "stop_conditions": stop_report,
                    "claims": curve.claims_block(),
                    "limitations": config["limitations"],
                }
                base._write_report_fixed_point(
                    output, report, limits, data_bytes)
                _write_json(ROOT / config["report_root"] / "summary.json", {
                    "format": curve.REPORT_FORMAT + ".summary",
                    "mode": curve.MODE,
                    "status": decision["status"],
                    "decision": decision,
                    "readouts": [
                        {
                            "prefix": item["prefix"],
                            "answer_loss": item[
                                "teacher_forced_aggregates"]["answer"][
                                    "correct"]["mean"],
                            "caption_loss": item[
                                "teacher_forced_aggregates"]["caption"][
                                    "correct"]["mean"],
                            "projector_state_sha256":
                                item["projector_state_sha256"],
                        }
                        for item in readouts
                    ],
                    "trajectory": {
                        "t1": trajectory["t1"],
                        "t2": trajectory["t2"],
                        "t3": trajectory["t3"],
                    },
                    "report": str(output / "report.json"),
                })
                return report
            except BaseException:
                if not profiler_stopped:
                    profiler.stop()
                raise
    except BaseException as exc:
        stop_report = curve.curve_stop_conditions_report(
            wall_seconds=time.monotonic() - started,
            wall_limit_seconds=limits["max_seconds"],
            artifact_bytes=base._data_bytes(args.output),
            artifact_limit_bytes=limits["artifact_bytes"],
            updates=0 if not trace_path.exists() else len(
                _read_jsonl(trace_path)),
            optimizer_steps=0 if not trace_path.exists() else len(
                _read_jsonl(trace_path)),
            update_backward_calls=0 if not trace_path.exists() else
            sum(int(row.get("backward_calls") or 0)
                for row in _read_jsonl(trace_path)),
            probe_backward_calls=0,
            model_loads=accounting.counts()["model_loads"],
            extension_loads=accounting.counts()["extension_loads"],
            extension_writes=0, checkpoint_writes=0, downloads=0,
            cpu_fallback=False, sealed_evaluation_access=0,
            guard_stop_reason=(guard.report().get("stop_reason")
                               if guard is not None else None),
            readouts=len(readouts),
            scoring_generations=sum(
                int(item["counts"]["generations"]) for item in readouts),
            teacher_forced_cells=sum(
                int(item["counts"]["teacher_forced_cells"])
                for item in readouts),
            scoring_forwards=sum(
                int(item["counts"]["scoring_forwards"])
                for item in readouts),
            readout_weight_writes=readout_weight_writes)
        _failure_report(output, exc, stop_report=stop_report,
                        partial={"loads": accounting.counts(),
                                 "trace_rows": (len(_read_jsonl(trace_path))
                                                if trace_path.exists() else 0),
                                 "readouts": len(readouts)})
        raise


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=(curve.MODE,), default=curve.MODE)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("mps",), default="mps")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-lock", type=Path, required=True)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if os.environ.get("REPLICA_CAPACITY_GUARD") != "1":
        parser.error("Use the capacity guard")
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0":
        parser.error("CPU fallback must be disabled")
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE",
                 "HF_DATASETS_OFFLINE"):
        if os.environ.get(name) != "1":
            parser.error(f"{name}=1 is required")
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config = curve.load_config(config_path)
    args.config = config_path
    if config["mode"] != args.mode:
        parser.error("--mode differs from the declared config mode")
    if args.device != config["policy"]["device"]:
        parser.error("The declared curve device must be mps")
    learn.require(args.output not in (ROOT, Path("/")),
                  "Refusing an unsafe output path")
    if args.output.exists() and (not args.output.is_dir()
                                 or any(args.output.iterdir())):
        parser.error("Use fresh output")
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    try:
        pins = verify_pins(config)
        execute_curve(args, config, pins)
    except BaseException as exc:
        if not (args.output / "failure_report.json").exists():
            stop_report = curve.curve_stop_conditions_report(
                wall_seconds=time.monotonic() - started,
                wall_limit_seconds=config["limits"]["max_seconds"],
                artifact_bytes=base._data_bytes(args.output),
                artifact_limit_bytes=config["limits"]["artifact_bytes"],
                updates=0, optimizer_steps=0, update_backward_calls=0,
                probe_backward_calls=0, model_loads=0,
                extension_loads=0, extension_writes=0,
                checkpoint_writes=0, downloads=0, cpu_fallback=False,
                sealed_evaluation_access=0, guard_stop_reason=None)
            _failure_report(args.output, exc, stop_report=stop_report)
        raise


if __name__ == "__main__":
    main()

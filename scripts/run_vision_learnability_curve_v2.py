#!/usr/bin/env python3
"""Measured accounting replay and the single predeclared step-1 experiment.

Two commands are exposed:

``replay-accounting``
    CPU-only reconstruction over preserved v1 receipts.  It writes (or checks)
    the accounting correction evidence and never loads a model.

``run``
    Fail-closed native execution.  Only the step-1 protocol is executable, and
    only after an external authorization receipt binds the exact config
    SHA-256 and a committed protocol identity.  The v2 curve config is a
    declaration artifact for a later gate; it is not executable here.

No command reads development, held-out or semantic-audit payloads.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from replica_lab import vision_forward_accounting as fwd  # noqa: E402
from replica_lab import vision_learnability_curve as curve_v1  # noqa: E402
from replica_lab import vision_learnability_curve_v2 as curve_v2  # noqa: E402
from replica_lab import vision_learnability_diagnostic as learn  # noqa: E402
from replica_lab.vision_collapse_diagnostic import (  # noqa: E402
    CONTROL_NAMES, EXPECTED_STOP_TOKENS, control_plan, resolve_stop_tokens)
from replica_lab.vision_curriculum import PROJECTOR_TENSORS  # noqa: E402

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
    return base.verify_pins(config, root)


# ---------------------------------------------------------------------------
# CPU-only accounting replay
# ---------------------------------------------------------------------------


def replay_accounting(args) -> dict:
    run = args.run
    summary = args.summary
    config = args.config
    step1 = args.step1_config
    evidence = curve_v2.build_historical_correction(
        root=ROOT, run=run, summary=summary, v1_config=config,
        step1_config=step1,
        replay_command=(
            "python scripts/run_vision_learnability_curve_v2.py "
            "replay-accounting --run runs/vision_learnability_curve_v1_"
            "20261026_attempt04 --summary reports/vision_learnability_curve_"
            "v1_20261026/summary.json --config configs/vision_learnability_"
            "curve_v1.json --step1-config configs/vision_learnability_step1_"
            "reproduction_v1.json"))
    curve_v2.validate_historical_correction(evidence)
    destination = Path(args.output)
    if args.check is not None:
        expected = _read_json(args.check)
        if expected != evidence:
            raise RuntimeError(
                "Accounting replay differs from the preserved correction")
        print(json.dumps({
            "status": "replay_matches",
            "path": str(args.check),
            "v1_verdict": evidence["v1_verdict"],
            "scoring_passes": evidence["correction"]["reconstruction"]
            ["totals"]["scoring_decoder_passes"],
            "grand_total_events": evidence["correction"]["grand_total_events"],
        }, indent=2))
        return evidence
    _write_json(destination, evidence)
    print(json.dumps({
        "status": "accounting_correction_written",
        "path": str(destination),
        "v1_verdict": evidence["v1_verdict"],
        "scoring_passes": evidence["correction"]["reconstruction"]["totals"]
        ["scoring_decoder_passes"],
        "grand_total_events": evidence["correction"]["grand_total_events"],
    }, indent=2))
    return evidence


def _binding(root: Path, path: Path, role: str) -> dict[str, Any]:
    resolved = path.resolve()
    try:
        relative = str(resolved.relative_to(root.resolve()))
    except ValueError:
        relative = str(path)
    return {
        "role": role,
        "path": relative,
        "sha256": _sha(path),
    }


def replay_step1(args) -> dict:
    """Recompute the corrected step-1 receipt without native execution."""
    config = curve_v2.load_config(args.config)
    learn.require(config["mode"] == curve_v2.STEP1_MODE,
                  "replay-step1 requires the curve_step1 config")
    run = Path(args.run)
    learn.require(run.is_dir(), f"Attempt run is missing: {run}")
    report_path = run / "report.json"
    accounting_path = run / "accounting.json"
    trace_path = run / "training_trace.jsonl"
    mask_path = run / "response_mask_audit.json"
    for path in (report_path, accounting_path, trace_path, mask_path):
        learn.require(path.is_file(), f"Attempt artifact is missing: {path}")
    readout_paths = [run / "readouts" / label / "readout.json"
                     for label in ("prefix_0", "prefix_0_null", "prefix_1")]
    for path in readout_paths:
        learn.require(path.is_file(), f"Attempt readout is missing: {path}")
    report = _read_json(report_path)
    accounting = _read_json(accounting_path)
    readouts = [_read_json(path) for path in readout_paths]
    trace = _read_jsonl(trace_path)
    learn.require(len(trace) == 1, "Exactly one attempt update is required")
    command = (
        "python scripts/run_vision_learnability_curve_v2.py replay-step1 "
        f"--run {args.run} --config {args.config}")
    bindings = {
        "report": _binding(ROOT, report_path, "original_report"),
        "executed_config": _binding(ROOT, run / "config.json",
                                    "executed_config"),
        "validator_config": _binding(ROOT, Path(args.config),
                                     "validator_config"),
        "accounting": _binding(ROOT, accounting_path, "accounting"),
        "training_trace": _binding(ROOT, trace_path, "update_trace"),
        "response_mask_audit": _binding(ROOT, mask_path, "response_mask"),
    }
    for index, (label, path) in enumerate(zip(
            ("prefix_0", "prefix_0_null", "prefix_1"), readout_paths)):
        bindings[f"readout_{index}:{label}"] = _binding(
            ROOT, path, "readout")
    receipt = curve_v2.build_derived_step1_receipt(
        config=config, readouts=readouts, accounting=accounting,
        update=trace[0], original_report=report,
        source_bindings=bindings, replay_command=command)
    curve_v2.validate_derived_step1_receipt(receipt, root=ROOT)
    destination = Path(args.output)
    if args.check is not None:
        expected = _read_json(args.check)
        if expected != receipt:
            raise RuntimeError(
                "Derived step-1 replay differs from the preserved receipt")
        print(json.dumps({
            "status": "replay_matches",
            "path": str(args.check),
            "corrected_status": receipt["corrected_status"],
            "original_status": receipt["original_status"],
        }, indent=2))
        return receipt
    _write_json(destination, receipt)
    print(json.dumps({
        "status": "derived_receipt_written",
        "path": str(destination),
        "corrected_status": receipt["corrected_status"],
        "original_status": receipt["original_status"],
    }, indent=2))
    return receipt



# ---------------------------------------------------------------------------
# Authorization
# ---------------------------------------------------------------------------


def _load_authorization(config, config_path, explicit_path=None) -> dict:
    authorization = config["authorization"]
    path = Path(explicit_path or authorization["receipt_path"])
    learn.require(path.is_file(),
                  f"Native execution is not authorized; receipt missing: {path}")
    receipt = _read_json(path)
    return curve_v2.validate_authorization_receipt(
        receipt, config_sha256=_sha(Path(config_path)), config=config)


# ---------------------------------------------------------------------------
# Measured readout
# ---------------------------------------------------------------------------


def _readout_directory(output, label):
    return Path(output) / "readouts" / str(label)


def _generation_digest(records) -> str:
    payload = [
        [record.get("case_id"), record.get("control"),
         list(record.get("generated_ids") or [])]
        for record in records if record.get("available") is not False
    ]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()
                          ).hexdigest()


def _measured_prefill_analysis(session, tokenizer, row, visual, scope,
                               target_text, limits):
    """First-token logits against the exact scored target, not another field."""
    import torch
    from run_vision_curriculum import curriculum_prompt, prompt_ids, task_graph
    from replica_lab.vision_collapse_diagnostic import first_token_analysis

    prompt_text = curriculum_prompt(row["question"])
    prompt_tensor = prompt_ids(
        tokenizer, prompt_text, session.device, limits["max_prompt_tokens"])
    response = tokenizer(target_text, add_special_tokens=False)["input_ids"]
    learn.require(bool(response), "Readout target tokenizes to nothing")
    request = session.prepare(
        prompt_ids=prompt_tensor,
        prompt_mask=torch.ones_like(prompt_tensor, dtype=torch.bool),
        query=prompt_text, scope=scope,
        task=task_graph(row["case_id"], prompt_text),
        training=False, visual=visual, enabled_ports=(), core_impl="none",
        enabled_branches=("memory",))
    try:
        with torch.no_grad():
            result = session.forward(
                request, prompt_tensor,
                attention_mask=torch.ones_like(prompt_tensor,
                                               dtype=torch.bool))
        learn.require(result.logits is not None,
                      "Prefill forward returned no logits")
        learn.require(bool(torch.isfinite(result.logits).all()),
                      "Prefill logits are nonfinite")
        analysis = first_token_analysis(
            result.logits[0, -1], int(response[0]))
    finally:
        request.release()
    return {
        "gold_token": int(response[0]),
        "gold_logit": analysis["gold_logit"],
        "gold_rank": analysis["gold_rank"],
        "argmax_token": analysis["argmax_token"],
        "argmax_is_gold": analysis["argmax_is_gold"],
        "entropy": analysis["entropy"],
        "vocabulary": analysis["vocabulary"],
    }


def _measured_readout(
        adapter, session, tokenizer, service, protocol, scope, rows, plan,
        labels, stops, *, prefix, label, expected_prefixes,
        prefix_position, output, guard, started, limits,
        optimizer, accounting):
    import torch
    from run_vision_curriculum import (
        curriculum_prompt, generate_row, observe, projector_state_sha256)
    from run_vision_balanced_alignment import _teacher_forced_loss

    rows_by_case = {str(row["case_id"]): row for row in rows}
    caption_labels = {str(row["case_id"]): row for row in labels}
    readout_dir = _readout_directory(output, label)
    readout_dir.mkdir(parents=True, exist_ok=True)

    before_state = curve_v1.optimizer_state_report(optimizer)
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
        "generated_tokens": 0,
        "logical_scoring_cells": 0,
        "logical_scoring_operations": 0,
    }
    eos_only = 0
    whitespace_only = 0
    length_by_kind: dict[str, int] = {"answer": 0, "caption": 0}
    accounting.begin_readout(prefix)
    adapter.eval()
    try:
        for position, row in enumerate(rows, 1):
            _check_wall(started, limits["max_seconds"], "step-1 readout")
            if guard is not None and position % 4 == 0:
                guard.sample(index=position, event=f"step1_readout_{label}")
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
                        "prefix": prefix,
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
                if row["kind"] == "answer":
                    target_text = row["response"]
                else:
                    target_text = caption_labels[case_id]["answer"]
                target_tokens = tokenizer(
                    target_text, add_special_tokens=False)["input_ids"]
                learn.require(bool(target_tokens),
                              "Readout target tokenizes to nothing")
                target_first_token = int(target_tokens[0])
                prefill = _measured_prefill_analysis(
                    session, tokenizer, row, visual, scope, target_text,
                    limits)
                prefill_matches = bool(generated["generated_ids"]) and (
                    prefill["argmax_token"]
                    == int(generated["generated_ids"][0]))
                loss = _teacher_forced_loss(
                    session, tokenizer, protocol, scope, visual, row,
                    row["question"], target_text)
                record = {
                    "format": curve_v2.STEP1_READOUT_FORMAT + ".generation",
                    **generated,
                    "prefix": prefix,
                    "case_id": case_id,
                    "kind": row["kind"],
                    "available": True,
                    "hypothesis_count": 1,
                    "prefill": prefill,
                    "prefill_argmax_matches_first_token": prefill_matches,
                    "teacher_forced_loss": loss,
                    "target_text": target_text,
                    "target_first_token": target_first_token,
                }
                if row["kind"] == "caption":
                    record["target_visible_objects"] = [
                        list(item)
                        for item in caption_labels[case_id]["visible_objects"]]
                    caption_rows.append(record)
                else:
                    answer_rows.append(record)
                ids = list(generated["generated_ids"])
                stops_for_record = list(generated["stop_tokens"])
                if fwd.eos_only_ids(ids, stops_for_record):
                    eos_only += 1
                elif fwd.whitespace_only_text(generated["generated_text"]):
                    whitespace_only += 1
                length_by_kind[row["kind"]] = (
                    length_by_kind.get(row["kind"], 0) + len(ids))
                normalised.append({
                    "prefix": prefix,
                    "case_id": case_id,
                    "control": control,
                    "kind": row["kind"],
                    "available": True,
                    "generated_ids": ids,
                    "generated_text": generated["generated_text"],
                    "prefill_argmax_matches_first_token": prefill_matches,
                    "prefill": prefill,
                    "teacher_forced_loss": loss,
                    "target_text": target_text,
                    "target_first_token": target_first_token,
                    "target_visible_objects": record.get(
                        "target_visible_objects"),
                    "hypothesis_count": 1,
                    "stop_tokens": stops_for_record,
                })
                counts["available"] += 1
                counts["generations"] += 1
                counts["prefill_forwards"] += 1
                counts["teacher_forced_cells"] += 1
                counts["generated_tokens"] += len(ids)
                counts["logical_scoring_cells"] += 1
                counts["logical_scoring_operations"] += 3
                accounting.record_logical(cells=1, operations=3)
    finally:
        adapter.train()
    measured = accounting.end_readout()
    after_state = curve_v1.optimizer_state_report(optimizer)
    after_digest = projector_state_sha256(adapter.vision_projector)
    after_base = adapter.parameter_stamp(original_only=True)
    weight_changes = _changed(before_snapshot,
                              _snapshot_new_parameters(adapter))
    generated_report = curve_v1.generation_hypothesis_report(
        normalised,
        answer_max_new_tokens=int(limits["max_new_tokens"]),
        caption_max_new_tokens=int(limits["caption_max_new_tokens"]))
    learn.require(generated_report["ok"],
                  f"Generation evidence failed: "
                  f"{generated_report['violations']}")
    response_audit = curve_v2.response_audit(
        normalised,
        rows_by_case=rows_by_case,
        caption_labels=caption_labels)
    learn.require(response_audit["ok"],
                  f"Response audit failed: {response_audit['violations']}")
    scores = learn.score_train_fit(
        records=normalised, plan=plan, rows_by_case=rows_by_case,
        caption_labels=caption_labels)
    metrics = curve_v1.readout_metrics(normalised, scores)
    aggregates = curve_v1.teacher_forced_aggregates(normalised)
    teacher_forced = {
        role: ((aggregates.get(role) or {}).get("correct") or {}).get("mean")
        for role in ("answer", "caption")
    }
    generation_summary = {
        "generated_tokens": counts["generated_tokens"],
        "lengths_by_kind": length_by_kind,
        "generation_cells": counts["generations"],
        "prefill_calls": int(measured["counters"]
                             ["generation_prefill_calls"]),
        "decode_calls": int(measured["counters"]
                            ["generation_decode_calls"]),
        "eos_only": eos_only,
        "whitespace_only": whitespace_only,
        "generation_digest": _generation_digest(normalised),
    }
    readout = {
        "format": curve_v2.STEP1_READOUT_FORMAT,
        "mode": curve_v2.STEP1_MODE,
        "prefix": int(prefix),
        "label": label,
        "optimizer_steps": int(after_state["max_step"]),
        "optimizer_state_before": before_state,
        "optimizer_state_after": after_state,
        "optimizer_state_unchanged": (
            before_state["sha256"] == after_state["sha256"]
            and before_state["max_step"] == after_state["max_step"]),
        "projector_state_sha256": after_digest,
        "projector_unchanged_by_readout": before_digest == after_digest,
        "weight_change_names": weight_changes,
        "readout_weight_writes": 0,
        "base_stamp_unchanged": before_base == after_base,
        "adapter_train_restored": bool(adapter.training),
        "counts": counts,
        "measured": measured,
        "generation_summary": generation_summary,
        "generation": generated_report,
        "response_audit": response_audit,
        "scores": scores,
        "teacher_forced": teacher_forced,
        "teacher_forced_aggregates": aggregates,
        "metrics": metrics,
        "gates": None,
        "files": {
            "answer_generations": str(
                readout_dir / "train_answer_generations.jsonl"),
            "caption_generations": str(
                readout_dir / "train_caption_generations.jsonl"),
            "records": str(readout_dir / "records.jsonl"),
        },
    }
    validation = curve_v2.validate_measured_readout(
        readout, available_per_readout=98,
        prefixes=expected_prefixes, prefix_position=prefix_position,
        records=normalised,
        max_new_tokens_by_kind={"answer": 8, "caption": 32},
        endpoint=False)
    learn.require(validation["ok"],
                  f"Measured readout failed: {validation['violations']}")
    readout["validation"] = validation
    _write_jsonl(readout_dir / "train_answer_generations.jsonl", answer_rows)
    _write_jsonl(readout_dir / "train_caption_generations.jsonl",
                 caption_rows)
    _write_jsonl(readout_dir / "records.jsonl", normalised)
    _write_json(readout_dir / "readout.json", readout)
    return readout


# ---------------------------------------------------------------------------
# Step-1 execution
# ---------------------------------------------------------------------------


def _anchor_deltas(readout, expected):
    values = readout.get("teacher_forced") or {}
    return {
        role: {
            "observed": values.get(role),
            "expected": expected[role],
            "absolute_delta": (abs(float(values[role]) - float(expected[role]))
                               if values.get(role) is not None else None),
        }
        for role in ("answer", "caption")
    }


def execute_step1(args, config, pins) -> dict:
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
    learn.require(not output.exists(),
                  f"Fresh output required; path exists: {output}")
    output.mkdir(parents=True)
    limits = config["limits"]
    started = time.monotonic()
    guard = None
    accounting_loads = LoadAccounting()
    trace: list[dict] = []
    readouts: list[dict] = []
    init_hash = None
    optimizer_steps = 0
    update_backward_calls = 0
    _write_json(output / "config.json", config)
    (output / "protocol.json").write_bytes(pins["protocol_path"].read_bytes())
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
                      f"Step-1 is refused: {precondition['violations']}")
        training = balanced._load_training_v2(protocol)
        schedule_report = learn.prefix_schedule_report(
            training["batches"],
            prefix_updates=7,
            batch_size=4, answer_per_update=2, caption_per_update=2,
            expected_rows=26, expected_answer_rows=14,
            expected_caption_rows=12, expected_repeated_rows=2)
        learn.require_prefix_schedule(schedule_report)
        batch = list(training["batches"][0])
        learn.require(len(batch) == 4 and len(training["batches"]) >= 1,
                      "The step-1 update is not four rows")
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
                        profiler=profiler), accounting_loads, "model")
                guard.sample(index=0, event="model_loaded")
                profiler.mark("model_loaded")
                stops, stop_source = resolve_stop_tokens(
                    getattr(adapter.base.config, "eos_token_id", None),
                    getattr(tokenizer, "eos_token_id", None))
                learn.require(tuple(stops) == EXPECTED_STOP_TOKENS,
                              f"Model stop set {stops} differs")
                learn.require(int(tokenizer.eos_token_id) == 1,
                              "Tokenizer EOS differs from the declared target")
                adapter.eval()
                freeze_projector_only(adapter)
                adapter._initial_base_stamp = adapter.parameter_stamp(
                    original_only=True)
                init_hash = initialize_projector(
                    adapter.vision_projector,
                    int(config["initialization"]["init_seed"]))
                learn.require(init_hash == curve_v2.STEP1_INIT_SHA256,
                              "Fresh seed-29 projector differs from the anchor")
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
                              "Step-1 requires NativeFusedAdamW")
                learn.require(len(optimizer.param_groups) == 1
                              and len(optimizer.param_groups[0]["params"]) == 6,
                              "Optimizer ownership differs")
                budget = fwd.ForwardBudget(config["accounting"])
                accounting = fwd.ForwardAccounting(budget=budget)
                with accounting.wrap_native(adapter), \
                        accounting.wrap_feature_service(service), \
                        accounting.measured_session(session):
                    readout_a = _measured_readout(
                        adapter, session, tokenizer, service, protocol, scope,
                        training["training"], plan,
                        training["train_caption_labels"], stops,
                        prefix=0, label="prefix_0",
                        expected_prefixes=[0, 0, 1], prefix_position=0,
                        output=output, guard=guard, started=started,
                        limits=limits,
                        optimizer=optimizer, accounting=accounting)
                    readout_b = _measured_readout(
                        adapter, session, tokenizer, service, protocol, scope,
                        training["training"], plan,
                        training["train_caption_labels"], stops,
                        prefix=0, label="prefix_0_null",
                        expected_prefixes=[0, 0, 1], prefix_position=1,
                        output=output, guard=guard, started=started,
                        limits=limits,
                        optimizer=optimizer, accounting=accounting)
                    readouts.extend([readout_a, readout_b])
                    nondeterminism = curve_v2.readout_nondeterminism_report(
                        readout_a, readout_b,
                        tolerance=float(config["nondeterminism_stop"]
                                        ["teacher_forced_abs_tolerance"]))
                    if not nondeterminism["ok"]:
                        accounting_totals = accounting.historical_totals()
                        total_report = curve_v2.validate_step1_accounting(
                            readouts, accounting_totals)
                        decision = curve_v2.step1_decision(
                            prefix0=readout_a, prefix0_null=readout_b,
                            prefix1=readout_a, update={},
                            accounting=total_report,
                            nondeterminism=nondeterminism)
                        _write_json(output / "accounting.json",
                                    accounting_totals)
                        profiler_stopped = True
                        return _finish_step1(
                            output, config, args.config, pins, decision,
                            readouts, trace, total_report,
                            accounting_totals, nondeterminism, profiler,
                            guard, accounting_loads, started, stop_source,
                            init_hash)
                    _check_wall(started, limits["max_seconds"],
                                "step-1 update")
                    with accounting.forced_phase(fwd.PHASE_TRAINING):
                        with accounting.counted_optimizer_step(optimizer):
                            update, calls = base._gradient_update(
                                adapter, session, tokenizer, service, protocol,
                                scope, batch, optimizer, limits, stops,
                                index=1, guard=guard,
                                source_id=protocol["vision_source"]["source_id"])
                    learn.require(calls == 4 and update["backward_calls"] == 4,
                                  "The single update did not make four backward calls")
                    learn.require(not set(update["changed_projector_tensors"])
                                  - set(PROJECTOR_TENSORS),
                                  "The update changed a non-projector tensor")
                    update["optimizer_steps"] = 1
                    optimizer_steps = 1
                    update_backward_calls = int(calls)
                    trace.append(update)
                    _write_jsonl(output / "training_trace.jsonl", trace)
                    mask_audit = curve_v2.validate_response_masks(
                        update.get("fit_contract_rows") or [])
                    learn.require(mask_audit["ok"],
                                  f"Update mask audit failed: "
                                  f"{mask_audit['violations']}")
                    readout_c = _measured_readout(
                        adapter, session, tokenizer, service, protocol, scope,
                        training["training"], plan,
                        training["train_caption_labels"], stops,
                        prefix=1, label="prefix_1",
                        expected_prefixes=[0, 0, 1], prefix_position=2,
                        output=output, guard=guard, started=started,
                        limits=limits,
                        optimizer=optimizer, accounting=accounting)
                    readouts.append(readout_c)
                    accounting_totals = accounting.historical_totals()
                    total_report = curve_v2.validate_step1_accounting(
                        readouts, accounting_totals,
                        expected=config["expected_measured_work"])
                decision = curve_v2.step1_decision(
                    prefix0=readout_a, prefix0_null=readout_b,
                    prefix1=readout_c, update=update,
                    accounting=total_report, nondeterminism=nondeterminism)
                _write_json(output / "accounting.json", accounting_totals)
                _write_json(output / "response_mask_audit.json", mask_audit)
                profiler_stopped = True
                return _finish_step1(
                    output, config, args.config, pins, decision, readouts,
                    trace, total_report, accounting_totals,
                    nondeterminism, profiler, guard, accounting_loads,
                    started, stop_source, init_hash)
            finally:
                if not profiler_stopped:
                    try:
                        profiler.stop()
                    except Exception:  # noqa: BLE001
                        pass
    except Exception as exc:
        failure = {
            "format": curve_v2.STEP1_REPORT_FORMAT + ".failure",
            "generated_at_utc": _now(),
            "status": curve_v2.STEP1_STATUS_ENGINEERING,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "curve_v1_verdict": curve_v2.V1_VERDICT,
            "curve_verdict_preserved": True,
            "dev_accessed": False,
            "heldout_accessed": False,
            "v0_run": False,
            "promotion": False,
            "retry_authorized": False,
        }
        try:
            _write_json(output / "failure_report.json", failure)
        except Exception:  # noqa: BLE001
            pass
        raise


def _finish_step1(output, config, config_path, pins, decision, readouts, trace,
                  total_report, accounting_totals, nondeterminism,
                  profiler, guard, accounting_loads, started, stop_source,
                  init_hash):
    profiler.mark("report_ready")
    profiler_summary = profiler.stop()
    data_bytes = base._data_bytes(output)
    report = {
        "format": curve_v2.STEP1_REPORT_FORMAT,
        "version": curve_v2.VERSION,
        "mode": curve_v2.STEP1_MODE,
        "name": curve_v2.STEP1_NAME,
        "status": decision["status"],
        "declared_at_utc": config["declared_at_utc"],
        "generated_at_utc": _now(),
        "command": [sys.executable, *sys.argv],
        "config_sha256": _sha(Path(config_path)),
        "pins": pins["pins"],
        "initialization": {
            "init_seed": 29,
            "projector_init_sha256": init_hash,
            "expected_sha256": curve_v2.STEP1_INIT_SHA256,
            "matches_anchor": init_hash == curve_v2.STEP1_INIT_SHA256,
        },
        "optimizer": {
            "name": "NativeFusedAdamW",
            "lr": config["optimizer"]["lr"],
            "weight_decay": config["optimizer"]["weight_decay"],
            "tensors": 6,
            "parameters": 11542016,
            "optimizer_steps": 1 if trace else 0,
        },
        "generation_stop_set": {
            "stop_tokens": list(EXPECTED_STOP_TOKENS),
            "source": stop_source,
        },
        "readouts": readouts,
        "readout_sequence": [readout["label"] for readout in readouts],
        "training": {"updates": trace, "total_updates": len(trace),
                     "update_backward_calls": 4 if trace else 0,
                     "optimizer_steps": 1 if trace else 0},
        "accounting": accounting_totals,
        "measured_work": total_report,
        "nondeterminism": nondeterminism,
        "anchors_as_hypotheses": {
            "declared": curve_v2.STEP1_ANCHORS,
            "prefix0_deltas": _anchor_deltas(
                readouts[0], curve_v2.STEP1_ANCHORS[
                    "prefix0_teacher_forced_loss"]),
            "prefix1_deltas": (_anchor_deltas(
                readouts[-1], curve_v2.STEP1_ANCHORS[
                    "prefix1_teacher_forced_loss"]) if len(readouts) == 3
                else None),
            "used_as_pass_criteria": False,
        },
        "decision": decision,
        "curve_v1_verdict": curve_v2.V1_VERDICT,
        "curve_verdict_preserved": True,
        "dev_authorized": False,
        "dev_accessed": False,
        "heldout_authorized": False,
        "heldout_accessed": False,
        "v0_authorized": False,
        "v0_run": False,
        "promotion": False,
        "retry_authorized": False,
        "load_counts": accounting_loads.counts(),
        "artifact_bytes": data_bytes,
        "artifact_limit_bytes": config["limits"]["artifact_bytes"],
        "wall_seconds": time.monotonic() - started,
        "wall_limit_seconds": config["limits"]["max_seconds"],
        "guard": guard.report(),
        "profiler": profiler_summary,
        "claims": config["claims"],
        "limitations": config["limitations"],
    }
    _write_json(Path(output) / "report.json", report)
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser():
    parser = argparse.ArgumentParser(
        description="Measured vision accounting replay and step-1 protocol")
    sub = parser.add_subparsers(dest="command", required=True)
    replay = sub.add_parser("replay-accounting",
                            help="CPU-only historical reconstruction")
    replay.add_argument("--run", default=curve_v2.V1_RUN)
    replay.add_argument("--summary", default=curve_v2.V1_SUMMARY)
    replay.add_argument("--config", default=curve_v2.V1_CONFIG)
    replay.add_argument(
        "--step1-config",
        default="configs/vision_learnability_step1_reproduction_v1.json")
    replay.add_argument(
        "--output",
        default="reports/remediation_20261027/"
                "vision_learnability_curve_v1_accounting_correction.json")
    replay.add_argument("--check", default=None)
    replay_step = sub.add_parser(
        "replay-step1", help="CPU-only derived correction of a preserved run")
    replay_step.add_argument(
        "--run",
        default="runs/vision_learnability_step1_reproduction_v1_20261027_"
                "attempt03")
    replay_step.add_argument(
        "--config",
        default="configs/vision_learnability_step1_reproduction_v1.json")
    replay_step.add_argument(
        "--output",
        default="reports/remediation_20261027/"
                "step1_attempt03_derived_validator_correction.json")
    replay_step.add_argument("--check", default=None)
    run = sub.add_parser("run", help="fail-closed native step-1 execution")
    run.add_argument("--config", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--model-path", required=True)
    run.add_argument("--model-lock", required=True)
    run.add_argument("--authorization", default=None)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "replay-accounting":
        replay_accounting(args)
        return 0
    if args.command == "replay-step1":
        replay_step1(args)
        return 0
    config = curve_v2.load_config(args.config)
    if config["mode"] != curve_v2.STEP1_MODE:
        raise learn.LearnabilityDiagnosticError(
            "The v2 curve config is a declaration artifact; only "
            "curve_step1 is executable at this gate")
    _load_authorization(config, args.config, args.authorization)
    pins = verify_pins(config)
    execute_step1(args, config, pins)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

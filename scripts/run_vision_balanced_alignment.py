#!/usr/bin/env python3
"""Bounded balanced projector alignment diagnostic with fresh reload.

Modes:
* ``train``: verify sealed data, initialize the existing projector, run the
  pinned balanced fused-AdamW schedule, export a strict extension, score dev,
  then score the sealed held-out counterfactual role once.
* ``reload``: fresh process, zero updates, strict extension reload, re-run the
  held-out and retention evaluations and require bitwise equal raw outputs.

No base weights are saved. No backbone tensor is trained. The preserved stage
rejection is not rerun or replaced.
"""
from __future__ import annotations

import argparse
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

from replica_lab.vision_balanced_alignment import (  # noqa: E402
    compare_reload_artifacts, diagnostic_gate_report, load_prepared,
    score_caption_controls, score_counterfactual_answers,
)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("".join(json.dumps(row, allow_nan=False) + "\n"
                                  for row in rows))


def _write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _data_root(protocol):
    return Path(protocol["root"]) / protocol["data"]["root"]


def _bind_images(rows, data_root):
    for row in rows:
        if "image" in row:
            row["image_path"] = data_root / row["image"]
        if "counterfactual_image" in row:
            row["counterfactual_image_path"] = data_root / row[
                "counterfactual_image"]
        for control in ("wrong", "shuffled", "counterfactual"):
            if isinstance(row.get(control), dict) and "image" in row[control]:
                row[control]["image_path"] = data_root / row[control]["image"]


def _merge_labels(cases, labels):
    by_id = {row["case_id"]: row for row in labels}
    merged = []
    for case in cases:
        label = by_id[case["case_id"]]
        merged.append({**case, "label": label["answer"],
                       "counterfactual_answer": label.get(
                           "counterfactual_answer"),
                       "visible_objects": label.get("visible_objects"),
                       "counterfactual_visible_objects": label.get(
                           "counterfactual_visible_objects")})
    return merged


def _load_data(protocol):
    from replica_lab.vision_balanced_alignment import _read_jsonl as read_jsonl
    data_root = _data_root(protocol)
    training = read_jsonl(data_root / "training/alignment.jsonl")
    batches = read_jsonl(
        data_root / protocol["schedule"]["batches"]["path"])
    _require(len(batches) == protocol["schedule"]["total_updates"],
             "Pinned batch count differs")
    examples = {row["example_id"]: row for row in training}
    _require(len(examples) == len(training),
             "Training example ids repeat")
    require_response_only = protocol["train"]["response_only_labels"]
    _require(require_response_only is True,
             "Response-only labels are not declared")
    scheduled = []
    used: set[str] = set()
    for batch in batches:
        ids = batch["example_ids"]
        _require(len(ids) == protocol["schedule"]["alignment"]["batch"],
                 "Batch size differs")
        rows = []
        for example_id in ids:
            _require(example_id in examples,
                     f"Batch references an unknown example: {example_id}")
            rows.append(examples[example_id])
            used.add(example_id)
        scheduled.append(rows)
    _require(used == set(examples),
             "Pinned schedule does not cover all training examples")

    dev_answer_cases = read_jsonl(
        data_root / "evaluation/dev_answer_cases.jsonl")
    dev_answer_labels = read_jsonl(
        data_root / "evaluation/dev_answer_labels.jsonl")
    dev_caption_cases = read_jsonl(
        data_root / "evaluation/dev_caption_cases.jsonl")
    dev_caption_labels = read_jsonl(
        data_root / "evaluation/dev_caption_labels.jsonl")
    dev_answer_controls = read_jsonl(
        data_root / "evaluation/dev_answer_controls.jsonl")
    dev_caption_controls = read_jsonl(
        data_root / "evaluation/dev_caption_controls.jsonl")

    for rows, labels in (
            (dev_answer_cases, dev_answer_labels),
            (dev_caption_cases, dev_caption_labels)):
        _bind_images(rows, data_root)
        _require({row["case_id"] for row in rows}
                 == {row["case_id"] for row in labels},
                 "Development cases and labels differ")
        _require(len({row["case_id"] for row in rows}) == len(rows),
                 "Development case ids repeat")
    _bind_images(dev_answer_controls, data_root)
    _bind_images(dev_caption_controls, data_root)
    for controls, cases in (
            (dev_answer_controls, dev_answer_cases),
            (dev_caption_controls, dev_caption_cases)):
        _require({row["case_id"] for row in controls}
                 == {row["case_id"] for row in cases},
                 "Development control map coverage differs")

    for row in training:
        _require(isinstance(row.get("response"), str) and row["response"],
                 "Training response is missing")
        _require("label" not in row and "answer" not in row,
                 "Training rows must not carry evaluation labels")
        row["image_path"] = data_root / row["image"]

    train_caption_labels = read_jsonl(
        data_root / "training/train_caption_labels.jsonl")
    train_answer_cases = []
    for row in training:
        if row.get("kind") == "answer":
            train_answer_cases.append({
                **row, "label": row["response"], "scene_sha256": None})
    train_caption_cases = []
    for row in training:
        if row.get("kind") == "caption":
            train_caption_cases.append(dict(row))
    caption_labels = {row["case_id"]: row for row in train_caption_labels}
    _require({row["case_id"] for row in train_caption_cases}
             == set(caption_labels),
             "Train caption probe labels differ")
    return {
        "training": training, "batches": scheduled,
        "dev_answer_cases": dev_answer_cases,
        "dev_answer_labels": dev_answer_labels,
        "dev_answer_scored": _merge_labels(dev_answer_cases,
                                           dev_answer_labels),
        "dev_caption_cases": dev_caption_cases,
        "dev_caption_labels": dev_caption_labels,
        "dev_answer_controls": {row["case_id"]: row
                                for row in dev_answer_controls},
        "dev_caption_controls": {row["case_id"]: row
                                 for row in dev_caption_controls},
        "train_answer_cases": train_answer_cases,
        "train_caption_cases": train_caption_cases,
        "train_caption_labels": train_caption_labels,
    }


def _load_heldout(protocol):
    """Read sealed held-out cases only after the extension is exported."""
    from replica_lab.vision_balanced_alignment import _read_jsonl as read_jsonl
    data_root = _data_root(protocol)
    cases = read_jsonl(
        data_root / "evaluation/heldout_answer_cases.jsonl")
    labels = read_jsonl(
        data_root / "evaluation/heldout_answer_labels.jsonl")
    caption_cases = read_jsonl(
        data_root / "evaluation/heldout_caption_cases.jsonl")
    caption_labels = read_jsonl(
        data_root / "evaluation/heldout_caption_labels.jsonl")
    answer_controls = read_jsonl(
        data_root / "evaluation/heldout_answer_controls.jsonl")
    caption_controls = read_jsonl(
        data_root / "evaluation/heldout_caption_controls.jsonl")
    for rows, label_rows in ((cases, labels),
                             (caption_cases, caption_labels)):
        _bind_images(rows, data_root)
        _require({row["case_id"] for row in rows}
                 == {row["case_id"] for row in label_rows},
                 "Held-out cases and labels differ")
        _require(len({row["case_id"] for row in rows}) == len(rows),
                 "Held-out case ids repeat")
    _bind_images(answer_controls, data_root)
    _bind_images(caption_controls, data_root)
    for controls, case_rows in ((answer_controls, cases),
                                (caption_controls, caption_cases)):
        _require({row["case_id"] for row in controls}
                 == {row["case_id"] for row in case_rows},
                 "Held-out control map coverage differs")
    return {
        "heldout_answer_cases": cases,
        "heldout_answer_labels": labels,
        "heldout_answer_scored": _merge_labels(cases, labels),
        "heldout_caption_cases": caption_cases,
        "heldout_caption_labels": caption_labels,
        "heldout_answer_controls": {row["case_id"]: row
                                    for row in answer_controls},
        "heldout_caption_controls": {row["case_id"]: row
                                     for row in caption_controls},
    }


def _projector_snapshot(adapter):
    return {name: parameter.detach().cpu().clone()
            for name, parameter in adapter.vision_projector.named_parameters()}


def _delta_norm(before, after):
    import torch
    return float(sum(float((after[name] - before[name]).square().sum())
                     for name in before) ** 0.5)


def run_updates(adapter, session, tokenizer, service, protocol, scope, batches,
                optimizer, trace_path, *, start_time):
    import torch
    from run_vision_curriculum import (
        _example_tensors, curriculum_prompt, observe, task_graph)
    train = protocol["train"]
    limits = protocol["limits"]
    records = []
    for update_index, rows in enumerate(batches, 1):
        if time.monotonic() - start_time > train["max_seconds"]:
            raise RuntimeError("Alignment exceeded its wall budget")
        optimizer.zero_grad(set_to_none=True)
        before = _projector_snapshot(adapter)
        losses = []
        gradient_names: set[str] = set()
        output_rms, output_max = [], []
        processed = supervised = 0
        roles = {"caption": 0, "answer": 0}
        for row in rows:
            visual = observe(
                service, image_path=row["image_path"],
                image_sha256=row["image_sha256"], case_id=row["case_id"],
                scope=scope, ledger_epoch=session.ledger.epoch,
                spec=service.spec,
                source_id=protocol["vision_source"]["source_id"])
            with torch.no_grad():
                projected = adapter.project_visual(visual)
                output_rms.append(float(projected.square().mean().sqrt()))
                output_max.append(float(projected.abs().max()))
            prompt_tensor, input_ids, labels = _example_tensors(
                session, tokenizer, row,
                maximum_prompt=limits["max_prompt_tokens"],
                maximum_sequence=limits["max_sequence_tokens"])
            prompt_text = curriculum_prompt(row["question"])
            with session.training_request(
                    prompt_ids=prompt_tensor,
                    prompt_mask=torch.ones_like(prompt_tensor,
                                                dtype=torch.bool),
                    query=prompt_text, scope=scope,
                    task=task_graph(row["case_id"], prompt_text),
                    visual=visual, enabled_ports=(), core_impl="none",
                    enabled_branches=("memory",)) as request:
                result = session.forward(
                    request, input_ids,
                    attention_mask=torch.ones_like(input_ids,
                                                   dtype=torch.bool),
                    labels=labels)
                loss = result.loss
                if loss is None or not bool(torch.isfinite(loss)) \
                        or not loss.requires_grad:
                    raise RuntimeError("Training loss is nonfinite or detached")
                session.backward(request, loss / len(rows))
            losses.append(float(loss.detach()))
            processed += int(input_ids.numel())
            supervised += int((labels[:, 1:] != -100).sum())
            roles[row.get("kind", "unknown")] = roles.get(
                row.get("kind", "unknown"), 0) + 1
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
            "update": update_index,
            "example_ids": [row["example_id"] for row in rows],
            "case_ids": [row["case_id"] for row in rows],
            "roles": roles,
            "loss": sum(losses) / len(losses),
            "losses": losses,
            "grad_norm": grad_norm,
            "projector_update_norm": _delta_norm(before, after),
            "projector_output_rms": sum(output_rms) / len(output_rms),
            "projector_output_max": max(output_max),
            "processed_tokens": processed,
            "supervised_tokens": supervised,
            "response_only_labels": True,
            "nonprojector_gradient_names": nonprojector,
            "base_stamp_unchanged": (
                adapter.parameter_stamp(original_only=True)
                == adapter._initial_base_stamp),
        }
        if not record["base_stamp_unchanged"]:
            raise RuntimeError("Training changed the sealed base")
        records.append(record)
        _write_jsonl(trace_path, records)
    return records


def _control_image(case, control, control_map):
    if control == "correct":
        return {**case, "case_id": case["case_id"]}
    if control == "noimage":
        return None
    entry = control_map[case["case_id"]][control]
    image = entry["image_path"]
    return {**entry, "image_path": image}


def evaluate_answer_controls(adapter, session, tokenizer, service, protocol,
                             scope, cases, controls, control_map,
                             output_path, *, guard, start_time):
    from run_vision_curriculum import (
        curriculum_prompt, generate_row, observe)
    maximum_prompt = protocol["limits"]["max_prompt_tokens"]
    maximum_new = int(protocol["evaluation"]["max_new_tokens"])
    rows = []
    adapter.eval()
    for position, case in enumerate(cases, 1):
        if time.monotonic() - start_time > protocol["limits"]["max_seconds"]:
            raise RuntimeError("Answer evaluation exceeded its wall budget")
        if guard is not None and position % 10 == 0:
            guard.sample(index=position, event="answer_evaluation")
        question = curriculum_prompt(case["question"])
        for control in controls:
            image_case = _control_image(case, control, control_map)
            visual = None
            if image_case is not None:
                visual = observe(
                    service, image_path=image_case["image_path"],
                    image_sha256=image_case["image_sha256"],
                    case_id=case["case_id"], scope=scope,
                    ledger_epoch=session.ledger.epoch,
                    spec=service.spec,
                    source_id=protocol["vision_source"]["source_id"])
            row = generate_row(
                session, tokenizer, control=control, case=case,
                question=question, visual=visual, scope=scope,
                maximum_prompt=maximum_prompt, maximum_new=maximum_new,
                image_case_id=(None if image_case is None
                               else image_case["case_id"]),
                image_sha256=(case["image_sha256"] if image_case is None
                              else image_case["image_sha256"]))
            row["split_role"] = "answer"
            row["counterfactual_same_image"] = bool(
                control == "counterfactual"
                and control_map[case["case_id"]].get(
                    "counterfactual_same_image", False))
            rows.append(row)
            _write_jsonl(output_path, rows)
    return rows


def _teacher_forced_loss(session, tokenizer, protocol, scope, visual, case,
                         question, target_text):
    import torch
    from run_vision_curriculum import (
        curriculum_prompt, prompt_ids, task_graph)
    maximum_prompt = protocol["limits"]["max_prompt_tokens"]
    maximum_sequence = protocol["limits"]["max_sequence_tokens"]
    prompt_tensor = prompt_ids(tokenizer, curriculum_prompt(question),
                               session.device, maximum_prompt)
    response = tokenizer(target_text, add_special_tokens=False)["input_ids"]
    _require(bool(response), "Caption target tokenizes to nothing")
    _require(tokenizer.eos_token_id is not None,
             "Tokenizer requires an explicit EOS token")
    response = list(response) + [int(tokenizer.eos_token_id)]
    input_ids = torch.cat(
        (prompt_tensor,
         torch.tensor([response], dtype=torch.long,
                      device=prompt_tensor.device)), dim=1)
    _require(input_ids.shape[1] <= maximum_sequence,
             "Caption target exceeds the sequence budget")
    labels = torch.tensor(
        [[-100] * prompt_tensor.shape[1] + response],
        dtype=torch.long, device=prompt_tensor.device)
    request = session.prepare(
        prompt_ids=prompt_tensor,
        prompt_mask=torch.ones_like(prompt_tensor, dtype=torch.bool),
        query=curriculum_prompt(question), scope=scope,
        task=task_graph(case["case_id"], question), training=False,
        visual=visual, enabled_ports=(), core_impl="none",
        enabled_branches=("memory",))
    try:
        with torch.no_grad():
            result = session.forward(
                request, input_ids,
                attention_mask=torch.ones_like(input_ids, dtype=torch.bool),
                labels=labels)
    finally:
        request.release()
    loss = float(result.loss.detach().cpu())
    _require(loss == loss and loss >= 0, "Caption loss is invalid")
    return loss


def evaluate_caption_controls(adapter, session, tokenizer, service, protocol,
                              scope, cases, labels, controls, control_map,
                              output_path, *, guard, start_time):
    from run_vision_curriculum import (
        curriculum_prompt, generate_row, observe)
    maximum_prompt = protocol["limits"]["max_prompt_tokens"]
    maximum_new = int(protocol["evaluation"]["caption_max_new_tokens"])
    labels_by_id = {row["case_id"]: row for row in labels}
    rows = []
    adapter.eval()
    for position, case in enumerate(cases, 1):
        if time.monotonic() - start_time > protocol["limits"]["max_seconds"]:
            raise RuntimeError("Caption evaluation exceeded its wall budget")
        if guard is not None and position % 2 == 0:
            guard.sample(index=position, event="caption_evaluation")
        label = labels_by_id[case["case_id"]]
        question = case["question"]
        for control in controls:
            if control == "correct":
                image_case = case
                target_text = label["answer"]
                target_objects = label["visible_objects"]
            elif control == "counterfactual":
                entry = control_map[case["case_id"]]["counterfactual"]
                image_case = {**entry, "image_path": entry["image_path"]}
                target_text = label["counterfactual_answer"]
                target_objects = label["counterfactual_visible_objects"]
            elif control == "shuffled":
                entry = control_map[case["case_id"]]["shuffled"]
                image_case = {**entry, "image_path": entry["image_path"]}
                target_text = entry["caption"]
                target_objects = entry["visible_objects"]
            elif control == "noimage":
                image_case = None
                target_text = label["answer"]
                target_objects = label["visible_objects"]
            else:
                raise RuntimeError(f"Undeclared caption control: {control}")
            visual = None
            if image_case is not None:
                visual = observe(
                    service, image_path=image_case["image_path"],
                    image_sha256=image_case["image_sha256"],
                    case_id=case["case_id"], scope=scope,
                    ledger_epoch=session.ledger.epoch,
                    spec=service.spec,
                    source_id=protocol["vision_source"]["source_id"])
            generated = generate_row(
                session, tokenizer, control=control, case=case,
                question=curriculum_prompt(question), visual=visual,
                scope=scope, maximum_prompt=maximum_prompt,
                maximum_new=maximum_new,
                image_case_id=(None if image_case is None
                               else image_case["case_id"]),
                image_sha256=(None if image_case is None
                              else image_case["image_sha256"]))
            loss = _teacher_forced_loss(
                session, tokenizer, protocol, scope, visual, case, question,
                target_text)
            row = {
                **generated,
                "split_role": "caption",
                "target_case_id": (None if image_case is None
                                   else image_case["case_id"]),
                "target_text": target_text,
                "target_visible_objects": [list(item)
                                           for item in target_objects],
                "response_loss": loss,
                "scope": "teacher_forced_target_response",
            }
            rows.append(row)
            _write_jsonl(output_path, rows)
    return rows


def _score_bundle(protocol, output, *, kind, answer_rows=None,
                  answer_cases=None, answer_controls=None,
                  caption_rows=None, caption_labels=None,
                  caption_controls=None, seed_offset=0):
    evaluation = protocol["evaluation"]
    bootstrap = evaluation["bootstrap"]
    result = {}
    if answer_rows is not None:
        result["answer"] = score_counterfactual_answers(
            answer_cases, answer_rows, controls=answer_controls,
            seed=int(protocol["seed"]) + seed_offset,
            bootstrap=bootstrap,
            gates=(evaluation["gates"]
                   if answer_controls == evaluation["dev_answer_controls"]
                   else None))
    if caption_rows is not None:
        result["caption"] = score_caption_controls(
            caption_labels, caption_rows, controls=caption_controls,
            seed=int(protocol["seed"]) + seed_offset + 101,
            bootstrap=bootstrap)
    _write_json(output / f"scores/{kind}.json", result)
    return result


def _audit_receipt(protocol, run_output, *, create):
    path = ROOT / "reports/vision_balanced_alignment_20261026" / (
        f"audit_{protocol['data']['seals']['heldout_answer_case_ids'][:16]}"
        ".reserve.json")
    if create:
        if path.exists():
            raise RuntimeError("Fresh held-out namespace was already consumed")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "protocol_sha256": _sha(Path(run_output) / "protocol.json"),
            "run": str(run_output), "consumed": True,
        }, indent=2) + "\n")
    elif not path.exists():
        raise RuntimeError("Reload lacks the original held-out reservation")
    return str(path)


def execute_train(args):
    import torch
    from replica_lab.architecture.loading import (
        exclusive_mps_worker, load_local_mps)
    from replica_lab.optimizers import build_optimizer
    from replica_lab.profiling import RunProfiler
    from run_vision_curriculum import (
        Guard, backward_only_probe, evaluate_retention, export_extension,
        feature_service, freeze_projector_only, initialize_projector,
        make_session, protocol_scope, verify_source_and_spec)
    protocol_path = Path(args.protocol).resolve()
    protocol, manifest, implementation, data_root = load_prepared(
        protocol_path, ROOT)
    output = Path(args.output)
    guard = Guard(protocol)
    scope = protocol_scope(protocol)
    started = time.monotonic()
    (output / "protocol.json").write_text(protocol_path.read_text())
    data = _load_data(protocol)
    with exclusive_mps_worker(ROOT):
        profiler = RunProfiler("mps", output)
        profiler.start()
        try:
            _, spec, _ = verify_source_and_spec(protocol, "mps")
            adapter, tokenizer, lock = load_local_mps(
                args.model_path, args.model_lock, vision=spec)
            guard.sample(index=0, event="model_loaded")
            freeze_projector_only(adapter)
            adapter._initial_base_stamp = adapter.parameter_stamp(
                original_only=True)
            init_hash = initialize_projector(adapter.vision_projector,
                                             protocol["train"]["init_seed"])
            if init_hash in set(
                    protocol["train"]["forbidden_parent_weights_sha256"]):
                raise RuntimeError("Fresh initialization matches the rejected parent")
            session = make_session(adapter, tokenizer, output)
            service = feature_service(protocol, "mps")
            guard.sample(index=0, event="feature_service")
            optimizer = build_optimizer(
                adapter, "adamw_fused", lr_base=0.0,
                lr_new=protocol["train"]["lr"],
                weight_decay=protocol["train"]["weight_decay"])
            probe_case = dict(data["dev_answer_scored"][0])
            probe_case["response"] = probe_case["label"]
            backward = backward_only_probe(
                adapter, session, tokenizer, service, protocol, scope,
                probe_case)
            initialize_projector(adapter.vision_projector,
                                 protocol["train"]["init_seed"])
            adapter._initial_base_stamp = adapter.parameter_stamp(
                original_only=True)
            retention_pre = evaluate_retention(
                adapter, session, tokenizer, protocol,
                output / "retention_before.jsonl",
                label="before_updates", guard=guard, start_time=started)
            adapter.train()
            trace = run_updates(
                adapter, session, tokenizer, service, protocol, scope,
                data["batches"], optimizer, output / "training_trace.jsonl",
                start_time=started)
            extension = export_extension(
                adapter, output / "alignment/extension", protocol,
                label="alignment")
            retention_after = evaluate_retention(
                adapter, session, tokenizer, protocol,
                output / "retention_after.jsonl",
                label="after_updates", guard=guard, start_time=started)
            train_answer_rows = evaluate_answer_controls(
                adapter, session, tokenizer, service, protocol, scope,
                data["train_answer_cases"], ["correct"], {},
                output / "train_fit_answer_generations.jsonl",
                guard=guard, start_time=started)
            train_caption_rows = evaluate_caption_controls(
                adapter, session, tokenizer, service, protocol, scope,
                data["train_caption_cases"], data["train_caption_labels"],
                ["correct"], {},
                output / "train_fit_caption_generations.jsonl",
                guard=guard, start_time=started)
            dev_answer_rows = evaluate_answer_controls(
                adapter, session, tokenizer, service, protocol, scope,
                data["dev_answer_cases"],
                protocol["evaluation"]["dev_answer_controls"],
                data["dev_answer_controls"],
                output / "dev_answer_generations.jsonl",
                guard=guard, start_time=started)
            dev_caption_rows = evaluate_caption_controls(
                adapter, session, tokenizer, service, protocol, scope,
                data["dev_caption_cases"], data["dev_caption_labels"],
                protocol["evaluation"]["caption_controls"],
                data["dev_caption_controls"],
                output / "dev_caption_generations.jsonl",
                guard=guard, start_time=started)
            audit_receipt = _audit_receipt(protocol, output, create=True)
            heldout = _load_heldout(protocol)
            heldout_answer_rows = evaluate_answer_controls(
                adapter, session, tokenizer, service, protocol, scope,
                heldout["heldout_answer_cases"],
                protocol["evaluation"]["heldout_answer_controls"],
                heldout["heldout_answer_controls"],
                output / "heldout_answer_generations.jsonl",
                guard=guard, start_time=started)
            heldout_caption_rows = evaluate_caption_controls(
                adapter, session, tokenizer, service, protocol, scope,
                heldout["heldout_caption_cases"],
                heldout["heldout_caption_labels"],
                protocol["evaluation"]["caption_controls"],
                heldout["heldout_caption_controls"],
                output / "heldout_caption_generations.jsonl",
                guard=guard, start_time=started)

            train_answer = _score_bundle(
                protocol, output, kind="train_fit_answer",
                answer_rows=train_answer_rows,
                answer_cases=[{**row, "label": row["response"]}
                              for row in data["train_answer_cases"]],
                answer_controls=["correct"], seed_offset=11)
            train_caption = _score_bundle(
                protocol, output, kind="train_fit_caption",
                caption_rows=train_caption_rows,
                caption_labels=data["train_caption_labels"],
                caption_controls=["correct"], seed_offset=13)
            dev_answer = _score_bundle(
                protocol, output, kind="dev_answer",
                answer_rows=dev_answer_rows,
                answer_cases=data["dev_answer_scored"],
                answer_controls=protocol["evaluation"]["dev_answer_controls"],
                seed_offset=17)
            dev_caption = _score_bundle(
                protocol, output, kind="dev_caption",
                caption_rows=dev_caption_rows,
                caption_labels=data["dev_caption_labels"],
                caption_controls=protocol["evaluation"]["caption_controls"],
                seed_offset=19)
            heldout_answer = _score_bundle(
                protocol, output, kind="heldout_answer",
                answer_rows=heldout_answer_rows,
                answer_cases=heldout["heldout_answer_scored"],
                answer_controls=protocol["evaluation"][
                    "heldout_answer_controls"],
                seed_offset=23)
            heldout_caption = _score_bundle(
                protocol, output, kind="heldout_caption",
                caption_rows=heldout_caption_rows,
                caption_labels=heldout["heldout_caption_labels"],
                caption_controls=protocol["evaluation"]["caption_controls"],
                seed_offset=29)
            dev_gates = diagnostic_gate_report(
                label="dev", answer=dev_answer["answer"],
                caption=dev_caption["caption"],
                evaluation=protocol["evaluation"])
            heldout_gates = diagnostic_gate_report(
                label="heldout", answer=heldout_answer["answer"],
                caption=heldout_caption["caption"],
                evaluation=protocol["evaluation"])
            retention = {
                "before": retention_pre["score"],
                "after": retention_after["score"],
                "accuracy_delta": (
                    retention_after["score"]["accuracy"]
                    - retention_pre["score"]["accuracy"]),
            }
            _write_json(output / "scores/retention.json", retention)
            overfit = {
                "train_correct_accuracy": train_answer["answer"]["controls"]
                ["correct"]["accuracy"],
                "heldout_correct_accuracy":
                    heldout_answer["answer"]["controls"]["correct"][
                        "accuracy"],
                "answer_accuracy_gap": (
                    train_answer["answer"]["controls"]["correct"]["accuracy"]
                    - heldout_answer["answer"]["controls"]["correct"][
                        "accuracy"]),
                "train_caption_token_f1": train_caption["caption"]["controls"]
                ["correct"]["token_f1"],
                "heldout_caption_token_f1":
                    heldout_caption["caption"]["controls"]["correct"][
                        "token_f1"],
                "caption_token_f1_gap": (
                    train_caption["caption"]["controls"]["correct"]["token_f1"]
                    - heldout_caption["caption"]["controls"]["correct"][
                        "token_f1"]),
                "train_caption_semantic_f1": train_caption["caption"]
                ["controls"]["correct"]["semantic_f1"],
                "heldout_caption_semantic_f1": heldout_caption["caption"]
                ["controls"]["correct"]["semantic_f1"],
                "caption_semantic_f1_gap": (
                    train_caption["caption"]["controls"]["correct"]
                    ["semantic_f1"]
                    - heldout_caption["caption"]["controls"]["correct"]
                    ["semantic_f1"]),
                "scope": ("trained-set fit minus sealed held-out fit only; "
                          "no semantic-quality or promotion claim"),
            }
            _write_json(output / "scores/overfit_summary.json", overfit)
            report = {
                "format": "replica.vision_balanced_alignment.run.v1",
                "status": "completed_pending_reload",
                "disposition": "diagnostic_only_no_promotion",
                "protocol_sha256": _sha(protocol_path),
                "implementation_sha256": implementation,
                "data_manifest_sha256":
                    protocol["data"]["manifest"]["sha256"],
                "data_seals": protocol["data"]["seals"],
                "source_revision": lock["revision"],
                "device": "mps",
                "cpu_fallback_enabled": False,
                "preserved_stage_result": protocol["prior_consumed"],
                "initialization": {
                    "seed": protocol["train"]["init_seed"],
                    "projector_init_sha256": init_hash,
                    "warm_start": None,
                },
                "trainable_scope": "vision_projector.",
                "trainable_tensors": 6,
                "optimizer": optimizer.audit(),
                "backward_only_probe": backward,
                "training": {
                    "updates": trace,
                    "total_updates": len(trace),
                    "total_examples": sum(len(row["example_ids"])
                                          for row in trace),
                    "schedule": protocol["schedule"],
                },
                "extension": extension,
                "retention": retention,
                "train_fit": {
                    "answer": train_answer["answer"],
                    "caption": train_caption["caption"],
                },
                "dev": {
                    "answer": dev_answer["answer"],
                    "caption": dev_caption["caption"],
                    "gates": dev_gates,
                },
                "heldout": {
                    "answer": heldout_answer["answer"],
                    "caption": heldout_caption["caption"],
                    "gates": heldout_gates,
                    "audit_receipt": audit_receipt,
                },
                "overfit_summary": overfit,
                "raw_outputs": [
                    "training_trace.jsonl",
                    "train_fit_answer_generations.jsonl",
                    "train_fit_caption_generations.jsonl",
                    "dev_answer_generations.jsonl",
                    "dev_caption_generations.jsonl",
                    "heldout_answer_generations.jsonl",
                    "heldout_caption_generations.jsonl",
                    "retention_before.jsonl",
                    "retention_after.jsonl",
                ],
                "base_saved": False,
                "promotion": False,
                "profile": profiler.stop(),
            }
            _write_json(output / "run_report.json", report)
            print(json.dumps({
                "status": report["status"],
                "updates": len(trace),
                "heldout_gate_failures": heldout_gates["gate_failures"],
                "report": str(output / "run_report.json"),
            }))
            return report
        except BaseException as error:
            failure = {
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "partial_outputs": sorted(
                    path.name for path in output.glob("*.jsonl")),
                "profile": profiler.stop(),
            }
            _write_json(output / "failure_report.json", failure)
            raise


def execute_reload(args):
    from replica_lab.architecture.loading import (
        exclusive_mps_worker, load_local_mps)
    from replica_lab.profiling import RunProfiler
    from run_vision_curriculum import (
        Guard, evaluate_retention, feature_service, freeze_projector_only,
        make_session, protocol_scope, verify_source_and_spec)
    protocol_path = Path(args.protocol).resolve()
    protocol, manifest, implementation, data_root = load_prepared(
        protocol_path, ROOT)
    run = Path(args.run)
    prior = json.loads((run / "run_report.json").read_text())
    if prior["protocol_sha256"] != _sha(protocol_path):
        raise RuntimeError("Reload protocol differs")
    if prior["disposition"] != "diagnostic_only_no_promotion":
        raise RuntimeError("Reload refuses a non-diagnostic disposition")
    output = Path(args.output)
    guard = Guard(protocol)
    scope = protocol_scope(protocol)
    started = time.monotonic()
    (output / "protocol.json").write_text(protocol_path.read_text())
    with exclusive_mps_worker(ROOT):
        profiler = RunProfiler("mps", output)
        profiler.start()
        try:
            _, spec, _ = verify_source_and_spec(protocol, "mps")
            adapter, tokenizer, lock = load_local_mps(
                args.model_path, args.model_lock, vision=spec)
            guard.sample(index=0, event="model_loaded")
            freeze_projector_only(adapter)
            adapter.load_extension(run / "alignment/extension")
            session = make_session(adapter, tokenizer, output)
            service = feature_service(protocol, "mps")
            guard.sample(index=0, event="feature_service")
            audit_receipt = _audit_receipt(protocol, run, create=False)
            heldout = _load_heldout(protocol)
            evaluate_answer_controls(
                adapter, session, tokenizer, service, protocol, scope,
                heldout["heldout_answer_cases"],
                protocol["evaluation"]["heldout_answer_controls"],
                heldout["heldout_answer_controls"],
                output / "heldout_answer_generations.jsonl",
                guard=guard, start_time=started)
            evaluate_caption_controls(
                adapter, session, tokenizer, service, protocol, scope,
                heldout["heldout_caption_cases"],
                heldout["heldout_caption_labels"],
                protocol["evaluation"]["caption_controls"],
                heldout["heldout_caption_controls"],
                output / "heldout_caption_generations.jsonl",
                guard=guard, start_time=started)
            retention = evaluate_retention(
                adapter, session, tokenizer, protocol,
                output / "retention_after.jsonl",
                label="after_updates", guard=guard, start_time=started)
            names = (
                "heldout_answer_generations.jsonl",
                "heldout_caption_generations.jsonl",
                "retention_after.jsonl",
            )
            comparisons = compare_reload_artifacts(run, output, names)
            if not all(comparisons.values()):
                raise RuntimeError(
                    f"Fresh-process outputs differ: {comparisons}")
            report = {
                "format": "replica.vision_balanced_alignment.reload.v1",
                "status": "completed",
                "zero_updates": True,
                "protocol_sha256": _sha(protocol_path),
                "run_report_sha256": _sha(run / "run_report.json"),
                "extension_weights_sha256":
                    prior["extension"]["weights_sha256"],
                "comparisons": comparisons,
                "bitwise_equal": True,
                "retention": retention["score"],
                "audit_receipt": audit_receipt,
                "disposition": prior["disposition"],
                "promotion": False,
                "profile": profiler.stop(),
            }
            _write_json(output / "reload_report.json", report)
            print(json.dumps({
                "status": "completed",
                "bitwise_equal": True,
                "report": str(output / "reload_report.json"),
            }))
            return report
        except BaseException as error:
            failure = {
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "partial_outputs": sorted(
                    path.name for path in output.glob("*.jsonl")),
                "profile": profiler.stop(),
            }
            _write_json(output / "failure_report.json", failure)
            raise


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", nargs="?", choices=("train", "reload"))
    parser.add_argument("--mode", dest="mode_flag",
                        choices=("train", "reload"))
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--device", choices=("mps",), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-lock", type=Path, required=True)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.mode is not None and args.mode_flag is not None \
            and args.mode != args.mode_flag:
        parser.error("Positional mode and --mode disagree")
    args.mode = args.mode_flag or args.mode
    if args.mode is None:
        parser.error("A train or reload mode is required")
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

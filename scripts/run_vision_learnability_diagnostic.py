#!/usr/bin/env python3
"""Training-only MiniCPM5 vision-learnability diagnostic.

Two predeclared modes, one native invocation each:

* ``--mode ablation`` replaces every valid projected patch by that image's
  mean (paired with the unmodified train04 extension) and decides
  ``spatial_contribution_observed`` / ``spatial_residual_not_established`` /
  ``baseline_mismatch``.  Zero optimizer steps, zero backward passes, zero
  gradients and zero writes.
* ``--mode gradient`` is refused unless the preserved ablation receipt records
  ``spatial_contribution_observed``.  It trains only the six projector tensors
  on the smallest full-coverage seven-update prefix with native fused AdamW,
  two backward-only probes, exactly 30 backward calls and train-only scoring.

Neither mode reads development, held-out or semantic-audit payloads, downloads
anything, enables CPU fallback or runs V0.  One exclusive MPS worker and the
existing capacity guard/profiler are mandatory.
"""
from __future__ import annotations

import argparse
from collections import Counter
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

from replica_lab import vision_learnability_diagnostic as learn  # noqa: E402
from replica_lab.vision_collapse_diagnostic import (  # noqa: E402
    CONTROL_NAMES, EXPECTED_STOP_TOKENS, comparison_cells, control_plan,
    first_token_analysis, fit_coverage, pair_groups, require_fit_coverage,
    require_stop_conditions, resolve_stop_tokens)
from replica_lab.vision_curriculum import (  # noqa: E402
    PROJECTOR_PARAMETER_COUNT, PROJECTOR_TENSORS, VisionCurriculumError,
    verify_implementation_pins)
import run_vision_balanced_alignment as balanced  # noqa: E402
import run_vision_collapse_diagnostic as collapse  # noqa: E402
import run_vision_curriculum as curriculum  # noqa: E402


_sha = collapse._sha
_read_json = collapse._read_json
_read_jsonl = collapse._read_jsonl
_write_json = collapse._write_json
_write_jsonl = collapse._write_jsonl
_now = collapse._now
_check_wall = collapse._check_wall
LoadAccounting = collapse.LoadAccounting
_accounted_load = collapse._accounted_load
_snapshot_new_parameters = collapse._snapshot_new_parameters
_changed = collapse._changed
_artifact_manifest = collapse._artifact_manifest
ARTIFACT_SCOPE = collapse.ARTIFACT_SCOPE
MANIFEST_SCOPE = collapse.MANIFEST_SCOPE


# ---------------------------------------------------------------------------
# Pins
# ---------------------------------------------------------------------------


def verify_pins(config, root=ROOT):
    """Verify prepared/source/extension/baseline bytes before any model load."""
    root = Path(root)
    mode = config["mode"]
    prepared = config["prepared"]
    prepared_root = root / prepared["root"]
    require = learn.require
    require(prepared_root.is_dir(), "The prepared data root is missing")
    protocol_path = root / prepared["protocol"]
    manifest_path = root / prepared["manifest"]
    training_path = root / prepared["training"]
    for path, label in ((protocol_path, "prepared protocol"),
                        (manifest_path, "prepared manifest"),
                        (training_path, "training alignment")):
        require(path.is_file(), f"The {label} is missing: {path}")
    protocol_sha = _sha(protocol_path)
    require(protocol_sha == prepared["protocol_sha256"],
            "Prepared protocol bytes differ from the pin")
    manifest_sha = _sha(manifest_path)
    require(manifest_sha == prepared["manifest_sha256"],
            "Prepared manifest bytes differ from the pin")
    training_sha = _sha(training_path)
    require(training_sha == prepared["training_sha256"],
            "Training alignment bytes differ from the pin")
    protocol = _read_json(protocol_path)
    manifest = _read_json(manifest_path)
    require(protocol["data"]["root"] == prepared["root"],
            "Prepared protocol data root differs")
    require(protocol["data"]["manifest"]["sha256"] == manifest_sha,
            "Prepared protocol manifest pin differs")
    require(protocol["data"]["seals"]["train_alignment_payload"] == training_sha,
            "Prepared alignment seal differs")
    counts = protocol["data"]["counts"]
    require(counts["train_examples"] == prepared["training_examples"]
            and counts["train_answers"] == prepared["answer_examples"]
            and counts["train_captions"] == prepared["caption_examples"],
            "Prepared training counts differ")
    require(manifest["counts"] == counts, "Prepared manifest counts differ")
    fit = protocol["fit_contract"]
    contract = config["prompt_contract"]
    require(fit["question_template"] == contract["question_template"]
            == protocol["evaluation"]["question_template"],
            "The question template pins differ")
    require(fit["caption_template"] == contract["caption_template"],
            "The caption template pins differ")
    require(fit["loss"] == contract["loss"]
            and fit["response_only_labels"] is True,
            "The prepared fit loss/label contract differs")
    limits = config["limits"]
    for name in ("max_prompt_tokens", "max_sequence_tokens",
                 "max_new_tokens", "caption_max_new_tokens"):
        require(int(protocol["limits"][name]) == int(limits[name]),
                f"The declared {name} differs from the prepared protocol")
    resource = protocol["resource_policy"]
    require(resource["device"] == config["policy"]["device"] == "mps",
            "The declared device differs")
    require(resource["cpu_fallback"] is False
            and resource["downloads"] == 0
            and resource["workers"] == 1
            and resource["exclusive_mps_worker"] is True,
            "The prepared resource policy differs")
    lock_path = root / config["source"]["model_lock"]
    require(lock_path.is_file(), "The model lock is missing")
    lock_sha = _sha(lock_path)
    require(lock_sha == config["source"]["model_lock_sha256"],
            "Model lock bytes differ from the pin")
    lock = _read_json(lock_path)
    require(lock["revision"] == config["source"]["model_revision"],
            "Model lock revision differs")
    require(protocol["model"]["revision"] == config["source"]["model_revision"],
            "Prepared protocol model revision differs")
    require(protocol["model"]["lock_sha256"] == lock_sha,
            "Prepared protocol model lock pin differs")
    model_path = root / config["source"]["model_path"]
    require(model_path.is_dir(), "The sealed model path is missing")
    require(_sha(model_path / "config.json") == lock["config_sha256"],
            "Local model config differs from the lock")
    require(bool(list(model_path.glob("*.safetensors"))),
            "The sealed model snapshot has no safetensors weights")
    vision_manifest = root / config["source"]["vision_source_manifest"]
    require(vision_manifest.is_file(), "The vision source manifest is missing")
    vision_manifest_sha = _sha(vision_manifest)
    require(vision_manifest_sha
            == config["source"]["vision_source_manifest_sha256"],
            "Vision source manifest bytes differ from the pin")
    require(protocol["vision_source"]["manifest"]["sha256"]
            == vision_manifest_sha,
            "Prepared protocol vision manifest pin differs")

    extension_path = None
    if mode == "ablation":
        extension_path = root / config["extension"]["path"]
        weights_path = extension_path / "extensions.safetensors"
        receipt_path = extension_path / "manifest.json"
        require(weights_path.is_file() and receipt_path.is_file(),
                "The pinned train04 extension is incomplete")
        receipt_sha = _sha(receipt_path)
        require(receipt_sha == config["extension"]["manifest_sha256"],
                "Extension receipt bytes differ from the pin")
        weights_sha = _sha(weights_path)
        require(weights_sha == config["extension"]["weights_sha256"],
                "Extension weights differ from the pin")
        require(weights_path.stat().st_size
                == config["extension"]["weights_bytes"],
                "Extension weight bytes differ from the pin")
        receipt = _read_json(receipt_path)
        require(receipt["format"] == config["extension"]["format"]
                and receipt["regime"] == config["extension"]["regime"]
                and receipt["base_saved"] is False,
                "Extension format/regime/base policy differs")
        require(receipt["weights_sha256"] == weights_sha
                and receipt["manifest_digest"]
                == config["extension"]["manifest_digest"],
                "Extension receipt hashes differ")
        require(receipt["manifest"]["vision"] == protocol["vision"],
                "Extension vision pins differ from the prepared protocol")
        require(receipt["manifest"]["source_revision"]
                == config["source"]["model_revision"],
                "Extension source revision differs")
        parent = config["parent_attempt"]
        for field, path in (("report_sha256", parent["report"]),
                            ("rows_sha256", parent["rows"]),
                            ("cells_sha256", parent["cells"]),
                            ("features_sha256", parent["features"])):
            require(_sha(root / path) == parent[field],
                    f"Baseline {field} differs from the pin")
        baseline_report = _read_json(root / parent["report"])
        require(baseline_report["format"] == collapse.REPORT_FORMAT,
                "Baseline report format differs")
        stage = baseline_report["stages"][parent["baseline_stage"]]
        require(stage["teacher_forced_cells"] == parent["teacher_forced_cells"]
                and stage["comparison_cells"] == parent["comparison_cells"]
                and stage["evidence_records"] == parent["records"],
                "Baseline report stage counts differ")
    else:
        schedule = config["schedule"]
        batches_path = root / prepared["root"] / schedule["source"]
        labels_path = root / prepared["root"] / schedule["caption_labels"]
        require(batches_path.is_file() and labels_path.is_file(),
                "The pinned training batches/caption labels are missing")
        require(_sha(batches_path) == schedule["source_sha256"],
                "Training batch bytes differ from the pin")
        require(_sha(labels_path) == schedule["caption_labels_sha256"],
                "Train caption label bytes differ from the pin")
        require(protocol["schedule"]["batches"]["sha256"]
                == schedule["source_sha256"]
                and protocol["schedule"]["total_updates"] == 24,
                "Prepared schedule pins differ")
        require(protocol["train"]["init_seed"]
                == config["initialization"]["init_seed"],
                "The fresh init seed differs from the prepared protocol")
        require(config["precondition"]["config_sha256"]
                == _sha(root / config["precondition"]["config"]),
                "The precondition ablation config bytes differ from the pin")

    migration_config = config["implementation_pins"]
    migrations = dict(migration_config["migrations"])
    prepared_pins = dict(protocol.get("implementation_sha256") or {})
    require(prepared_pins, "The prepared protocol lacks implementation pins")
    require(set(migrations) <= set(prepared_pins),
            "An implementation migration names an unpinned file")
    unchanged = {"implementation_sha256": {
        relative: expected for relative, expected in prepared_pins.items()
        if relative not in migrations}}
    try:
        actual = verify_implementation_pins(unchanged, root)
    except VisionCurriculumError as error:
        raise learn.LearnabilityDiagnosticError(str(error)) from error
    for relative, migration in migrations.items():
        expected = prepared_pins[relative]
        require(migration["parent_sha256"] == expected,
                f"Implementation migration parent differs: {relative}")
        try:
            migrated = verify_implementation_pins(
                {"implementation_sha256": {
                    relative: migration["current_sha256"]}}, root)
        except VisionCurriculumError as error:
            raise learn.LearnabilityDiagnosticError(
                f"Migrated implementation bytes differ: {relative}") from error
        actual.update(migrated)
    implementation = {}
    for relative in sorted(prepared_pins):
        expected = prepared_pins[relative]
        implementation[relative] = {
            "pinned_sha256": expected,
            "actual_sha256": actual[relative],
            "matches_prepared_pin": actual[relative] == expected,
            "migrated": relative in migrations,
            "migration": migrations.get(relative),
        }
    new_files = {}
    for relative, expected in config["implementation_pins"]["new_files"].items():
        actual_sha = _sha(root / relative)
        require(actual_sha == expected,
                f"New implementation bytes differ: {relative}")
        new_files[relative] = {"pinned_sha256": expected,
                               "actual_sha256": actual_sha,
                               "enforced": True}
    pins = {
        "mode": mode,
        "prepared": {
            "root": prepared["root"],
            "protocol_sha256": protocol_sha,
            "manifest_sha256": manifest_sha,
            "training_sha256": training_sha,
            "training_examples": counts["train_examples"],
            "answer_examples": counts["train_answers"],
            "caption_examples": counts["train_captions"],
        },
        "source": {
            "model_path": config["source"]["model_path"],
            "model_lock_sha256": lock_sha,
            "model_revision": lock["revision"],
            "model_config_sha256": lock["config_sha256"],
            "vision_source_manifest_sha256": vision_manifest_sha,
            "vision_digest": protocol["vision_digest"],
        },
        "implementation": {
            "enforced": True,
            "mode": migration_config["mode"],
            "reason": migration_config["reason"],
            "files": implementation,
            "new_files": new_files,
        },
    }
    if mode == "ablation":
        pins["extension"] = {
            "path": config["extension"]["path"],
            "manifest_sha256": config["extension"]["manifest_sha256"],
            "weights_sha256": config["extension"]["weights_sha256"],
            "weights_bytes": config["extension"]["weights_bytes"],
            "manifest_digest": config["extension"]["manifest_digest"],
            "format": config["extension"]["format"],
            "base_saved": False,
        }
    else:
        pins["schedule"] = {
            "source_sha256": config["schedule"]["source_sha256"],
            "caption_labels_sha256":
                config["schedule"]["caption_labels_sha256"],
        }
    return {
        "protocol": protocol,
        "manifest": manifest,
        "pins": pins,
        "protocol_path": protocol_path,
        "prepared_root": prepared_root,
        "training_path": training_path,
        "extension_path": extension_path,
    }


# ---------------------------------------------------------------------------
# Shared execution helpers
# ---------------------------------------------------------------------------


def _prepare_visuals(service, protocol, session, rows, plan, scope, spec):
    """Observe every needed (image, target-task) pair exactly once."""
    rows_by_case = {str(row["case_id"]): row for row in rows}
    groups = pair_groups(rows)
    visual_pairs = set()
    for row in rows:
        case_id = str(row["case_id"])
        plan_row = plan[case_id]
        visual_pairs.add((case_id, case_id))
        if plan_row["sibling_available"]:
            visual_pairs.add((str(plan_row["sibling_case_id"]), case_id))
        visual_pairs.add((str(plan_row["shuffled_case_id"]), case_id))
    for group in [*groups["answer"], *groups["caption"]]:
        for cell_spec in comparison_cells(group, plan):
            visual_pairs.add((str(cell_spec["image_case_id"]),
                              str(cell_spec["target_case_id"])))
    visuals = {}
    accessed = []
    for image_case, target_case in sorted(visual_pairs):
        image_row = rows_by_case[image_case]
        visual = service.observe(
            str(image_row["image_path"]),
            source_id=protocol["vision_source"]["source_id"],
            task_id=target_case, scope=scope, observed_at=0,
            source_epoch=session.ledger.epoch)
        learn.require(visual.producer == spec.digest,
                      "Observed visual producer differs")
        learn.require(visual.task_id == target_case,
                      "Observed visual task binding differs")
        learn.require(visual.image_sha256 == image_row["image_sha256"],
                      "Observed image hash differs from its row")
        learn.require(bool(visual.features.isfinite().all()),
                      "Visual features are nonfinite")
        visuals[(image_case, target_case)] = visual
        accessed.append(str(image_row["image_path"]))
    encoder_means = {}
    import torch
    for row in rows:
        case_id = str(row["case_id"])
        visual = visuals[(case_id, case_id)]
        with torch.no_grad():
            valid = visual.valid[0]
            mean_patch = visual.features[0][valid].mean(dim=0)
            encoder_means[case_id] = [float(value)
                                      for value in mean_patch.tolist()]
    return visuals, encoder_means, accessed


def _stage_summary(payload):
    return {
        "kind": payload["kind"],
        "updates": 0,
        "rows": payload["rows"],
        "rows_with_all_four_controls": payload["rows_with_all_four_controls"],
        "computed_control_records": payload["computed_control_records"],
        "unavailable_controls": payload["unavailable_controls"],
        "evidence_records": len(payload["records"]),
        "teacher_forced_cells": payload["teacher_forced_cells"],
        "comparison_cells": len(payload["cells"]),
        "control_summary": payload["control_summary"],
        "text_ceiling": payload["text_ceiling"],
        "feature_diagnostics": payload["feature_diagnostics"],
        "image_stats": payload["image_stats"],
        "fit_coverage": payload["fit_coverage"],
        "projector_state_sha256": payload["projector_state_sha256"],
        "duration_seconds": payload["duration_seconds"],
    }


def _normalised_generation_rows(records):
    rows = []
    for record in records:
        generation = record.get("generation") or {}
        rows.append({
            "case_id": str(record["case_id"]),
            "control": str(record["control"]),
            "kind": str(record["kind"]),
            "available": record.get("available") is not False,
            "generated_ids": list(record.get("generated_ids")
                                  or generation.get("generated_ids") or []),
            "generated_text": record.get("generated_text",
                                         generation.get("generated_text")),
            "prefill_argmax_matches_first_token": bool(
                record.get("prefill_argmax_matches_first_token",
                           generation.get(
                               "prefill_argmax_matches_first_token"))),
            "hypothesis_count": 1,
            "stop_tokens": list(generation.get("stop_tokens") or
                                record.get("stop_tokens") or []),
        })
    return rows


def _write_report_fixed_point(output, report, limits, data_bytes):
    output = Path(output)
    previous = None
    payload = ""
    for _ in range(8):
        payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
        size = len(payload.encode())
        if previous == size:
            break
        previous = size
        total = data_bytes + size
        report["artifact_bytes_estimate_including_report"] = total
        report["artifact_accounting"] = {
            "data_bytes_before_report": data_bytes,
            "data_plus_report_bytes": total,
            "estimate_excludes_manifest": True,
            "manifest_total_bytes_scope":
                "listed files excluding manifest.json",
            "manifest_final_total_scope":
                "total_bytes plus the exact manifest.json size",
            "final_artifact_bytes_source": "manifest.json",
            "scope": ARTIFACT_SCOPE,
        }
        report["stop_conditions"] = {
            **report["stop_conditions"],
            "artifact_bytes_estimate_including_report": total,
            "artifact_bytes_estimate_excludes_manifest": True,
            "artifact_bytes_scope": ARTIFACT_SCOPE,
        }
    else:
        raise learn.LearnabilityDiagnosticError(
            "Report byte total did not stabilize")
    (output / "report.json").write_text(payload)
    total = sum(path.stat().st_size for path in output.rglob("*")
                if path.is_file())
    learn.require(total == report["artifact_bytes_estimate_including_report"],
                  f"Data-plus-report bytes differ: recorded "
                  f"{report['artifact_bytes_estimate_including_report']} actual {total}")
    learn.require(total <= int(limits["artifact_bytes"]),
                  "Artifact bytes exceed the declared ceiling")
    manifest, manifest_payload = _artifact_manifest(
        output, limit_bytes=limits["artifact_bytes"])
    learn.require(manifest["total_bytes"] == total,
                  "Manifest listed total differs")
    (output / "manifest.json").write_text(manifest_payload)
    final_bytes = sum(path.stat().st_size for path in output.rglob("*")
                      if path.is_file())
    learn.require(final_bytes == manifest["final_artifact_bytes"],
                  "Final artifact bytes differ from manifest")
    learn.require(final_bytes <= int(limits["artifact_bytes"]),
                  "Final artifact bytes exceed the ceiling")
    return report


def _data_bytes(output):
    return sum(path.stat().st_size for path in Path(output).rglob("*")
               if path.is_file())


def _failure_report(output, exc, *, mode, stop_report=None, partial=None):
    payload = {
        "format": learn.REPORT_FORMAT + ".failure",
        "generated_at_utc": _now(),
        "mode": mode,
        "status": "failed",
        "error_type": type(exc).__name__,
        "error": str(exc),
        "traceback": traceback.format_exc(),
        "partial": partial or {},
        "stop_conditions": stop_report,
        "claims": learn.claims_block(),
    }
    try:
        _write_json(Path(output) / "failure_report.json", payload)
    except Exception:  # noqa: BLE001 - never mask the original failure
        pass


# ---------------------------------------------------------------------------
# Ablation mode
# ---------------------------------------------------------------------------


def execute_ablation(args, config, pins):
    import torch
    from replica_lab.architecture.loading import (
        exclusive_mps_worker, load_local_mps)
    from replica_lab.profiling import RunProfiler
    from replica_lab.vision_qualification import validate_extension_inventory
    from run_vision_curriculum import (
        Guard, feature_service, freeze_projector_only, make_session,
        protocol_scope, projector_state_sha256, verify_source_and_spec)
    from run_vision_collapse_diagnostic import load_training_rows

    protocol = pins["protocol"]
    output = Path(args.output)
    limits = config["limits"]
    started = time.monotonic()
    guard = None
    accounting = LoadAccounting()
    _write_json(output / "config.json", config)
    (output / "protocol.json").write_bytes(pins["protocol_path"].read_bytes())
    try:
        learn.require(Path(args.model_path).resolve()
                      == (ROOT / config["source"]["model_path"]).resolve(),
                      "CLI model path differs from the pinned source")
        learn.require(Path(args.model_lock).resolve()
                      == (ROOT / config["source"]["model_lock"]).resolve(),
                      "CLI model lock differs from the pinned lock")
        rows, layout = load_training_rows(config, pins)
        plan = control_plan(rows)
        order = learn.train_only_access_order()
        order.access("train")
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
                session = make_session(adapter, tokenizer, output)
                service = feature_service(protocol, "mps")
                guard.sample(index=0, event="feature_service")
                profiler.mark("feature_service")
                visuals, encoder_means, accessed = _prepare_visuals(
                    service, protocol, session, rows, plan, scope, spec)
                guard.sample(index=0, event="encoder_features")
                profiler.mark("encoder_features", images=len(rows),
                              bound_visuals=len(visuals))
                for row in rows:
                    accessed.append(str(row["image_path"]))
                accessed.append(str(pins["training_path"]))

                receipt = validate_extension_inventory(
                    pins["extension_path"],
                    expected_names=[name for name, _ in
                                    adapter.new_named_parameters()],
                    vision_pins=protocol["vision"])
                learn.require(receipt["weights_sha256"]
                              == config["extension"]["weights_sha256"],
                              "Validated extension weights differ")
                _accounted_load(
                    lambda: adapter.load_extension(pins["extension_path"]),
                    accounting, "extension")
                extension_snapshot = _snapshot_new_parameters(adapter)
                extension_projector_sha = projector_state_sha256(
                    adapter.vision_projector)
                base_stamp = adapter.parameter_stamp(original_only=True)

                full_stage = collapse.StageEvaluator(
                    config["stages"][0], adapter, session, tokenizer,
                    protocol, scope, rows, plan, visuals, encoder_means,
                    stops, limits, guard, profiler, started).run()
                full_stage["projector_state_sha256"] = extension_projector_sha
                learn.require(not _changed(
                    extension_snapshot,
                    _snapshot_new_parameters(adapter)),
                    "The full stage changed extension tensors")
                learn.require(projector_state_sha256(adapter.vision_projector)
                              == extension_projector_sha,
                              "The full stage changed the projector")
                learn.require(not any(parameter.grad is not None
                                      for parameter in adapter.parameters()),
                              "The full stage produced gradients")
                profiler.mark("full_patch_complete",
                              rows=len(full_stage["records"]))

                tolerance = float(config["ablation"]["tolerance"])
                with learn.mean_only_projector(
                        adapter, tolerance=tolerance) as wrapper:
                    mean_stage = collapse.StageEvaluator(
                        config["stages"][1], adapter, session, tokenizer,
                        protocol, scope, rows, plan, visuals, encoder_means,
                        stops, limits, guard, profiler, started).run()
                wrapper.assert_restored()
                receipts = wrapper.summary()
                learn.require(receipts["installed"] is False,
                              "The ablation wrapper was not restored")
                mean_stage["projector_state_sha256"] = extension_projector_sha
                learn.require(projector_state_sha256(adapter.vision_projector)
                              == extension_projector_sha,
                              "The mean stage changed the projector")
                learn.require(not _changed(
                    extension_snapshot,
                    _snapshot_new_parameters(adapter)),
                    "The mean stage changed extension tensors")
                learn.require(adapter.parameter_stamp(original_only=True)
                              == base_stamp,
                              "The ablation changed the sealed base")
                learn.require(not any(parameter.grad is not None
                                      for parameter in adapter.parameters()),
                              "The mean stage produced gradients")
                profiler.mark("mean_only_complete",
                              rows=len(mean_stage["records"]))

                learn.require(full_stage["fit_rows"] == mean_stage["fit_rows"],
                              "Fit contract rows differ between stages")
                learn.require(full_stage["fit_coverage"]
                              == mean_stage["fit_coverage"],
                              "Fit coverage differs between stages")
                learn.require(full_stage["text_embedding"]
                              == mean_stage["text_embedding"],
                              "Text embeddings differ between stages")
                image_invariants = learn.verify_mean_only_image_stats(
                    full_stage["image_stats"], mean_stage["image_stats"],
                    tolerance=tolerance,
                    expected_valid_tokens=int(
                        config["expectations"]["valid_patches_per_image"]))
                preservation = learn.verify_mean_preservation(
                    full_stage["feature_diagnostics"],
                    mean_stage["feature_diagnostics"], tolerance=tolerance)
                receipt_invariants = learn.verify_ablation_receipts(
                    receipts,
                    expected_image_shas=[row["image_sha256"] for row in rows],
                    expected_valid_tokens=int(
                        config["expectations"]["valid_patches_per_image"]),
                    tolerance=tolerance)
                full_normalized = _normalised_generation_rows(full_stage["records"])
                full_generations = learn.generation_hypothesis_report(
                    full_normalized,
                    answer_max_new_tokens=int(limits["max_new_tokens"]),
                    caption_max_new_tokens=int(
                        limits["caption_max_new_tokens"]))
                mean_normalized = _normalised_generation_rows(mean_stage["records"])
                mean_generations = learn.generation_hypothesis_report(
                    mean_normalized,
                    answer_max_new_tokens=int(limits["max_new_tokens"]),
                    caption_max_new_tokens=int(
                        limits["caption_max_new_tokens"]))
                violations = []
                for name, report in (
                        ("receipts", receipt_invariants),
                        ("image_stats", image_invariants),
                        ("mean_preservation", preservation),
                        ("full_generations", full_generations),
                        ("mean_generations", mean_generations)):
                    if report.get("ok") is not True:
                        violations.append(f"{name}:{report.get('violations')}")
                if not any(record["prefill_argmax_matches_first_token"]
                           for record in full_normalized
                           if record.get("available")):
                    violations.append("full_stage_prefill")
                learn.require(not violations,
                              "Ablation invariants failed: "
                              + "; ".join(violations))
                profiler.mark("ablation_invariants_complete")

                parent = config["parent_attempt"]
                baseline_rows = _read_jsonl(ROOT / parent["rows"])
                baseline_cells = _read_jsonl(ROOT / parent["cells"])
                learn.require(_sha(ROOT / parent["rows"])
                              == parent["rows_sha256"]
                              and _sha(ROOT / parent["cells"])
                              == parent["cells_sha256"],
                              "Baseline rows/cells changed after pinning")
                baseline = learn.baseline_identity_report(
                    full_stage=full_stage,
                    baseline_rows=baseline_rows,
                    baseline_cells=baseline_cells,
                    tolerances=config["comparison"]["tolerances"],
                    expected=config["comparison"]["identity"])
                paired = learn.paired_ablation_report(
                    full_records=full_stage["records"],
                    mean_records=mean_stage["records"],
                    thresholds=config["decision"]["observed_if"])
                decision = learn.ablation_decision(
                    baseline_match=baseline["match"], paired=paired,
                    decision=config["decision"])
                ablation_invariants = {
                    "ok": not violations,
                    "violations": violations,
                    "receipts": receipt_invariants["ok"],
                    "image_stats": image_invariants["ok"],
                    "mean_preservation": preservation["ok"],
                    "full_generations": full_generations["ok"],
                    "mean_generations": mean_generations["ok"],
                    "fit_rows_identical":
                        full_stage["fit_rows"] == mean_stage["fit_rows"],
                    "fit_coverage_identical":
                        full_stage["fit_coverage"]
                        == mean_stage["fit_coverage"],
                    "text_embedding_identical":
                        full_stage["text_embedding"]
                        == mean_stage["text_embedding"],
                    "projector_state_unchanged":
                        projector_state_sha256(adapter.vision_projector)
                        == extension_projector_sha,
                    "extension_tensors_unchanged": not _changed(
                        extension_snapshot,
                        _snapshot_new_parameters(adapter)),
                    "base_parameter_stamp_unchanged":
                        adapter.parameter_stamp(original_only=True)
                        == base_stamp,
                    "no_gradients": not any(
                        parameter.grad is not None
                        for parameter in adapter.parameters()),
                }
                stage_payloads = {learn.FULL_STAGE: full_stage,
                                  learn.MEAN_STAGE: mean_stage}
                rows_all = (full_stage["records"] + mean_stage["records"])
                cells_all = full_stage["cells"] + mean_stage["cells"]
                _write_jsonl(output / "rows.jsonl", rows_all)
                _write_jsonl(output / "cells.jsonl", cells_all)
                _write_jsonl(output / "fit_contract_rows.jsonl",
                             full_stage["fit_rows"])
                _write_json(output / "fit_contract_coverage.json",
                            full_stage["fit_coverage"])
                features = {
                    "encoder_mean_patch": encoder_means,
                    "stages": {
                        name: {
                            "image_stats": payload["image_stats"],
                            "text_embedding": payload["text_embedding"],
                            "summary": payload["feature_diagnostics"],
                        }
                        for name, payload in stage_payloads.items()
                    },
                    "ablation": {
                        "mode": config["ablation"]["mode"],
                        "tolerance": tolerance,
                        "receipts": receipts,
                        "image_stats_invariants": image_invariants,
                        "mean_preservation": preservation,
                        "receipt_invariants": receipt_invariants,
                    },
                }
                _write_json(output / "features.json", features)
                _write_json(output / "paired_ablation.json", paired)
                access = learn.training_only_access_report(
                    order.snapshot(), accessed)
                learn.require(access["ok"],
                              f"Sealed access observed: {access['violations']}")
                learn.require(accounting.counts()
                              == {"model_loads": 1, "extension_loads": 1},
                              f"Load counts differ: {accounting.counts()}")
                profiler.mark("report_ready")
                summary = profiler.stop()
                profiler_stopped = True
                data_bytes = _data_bytes(output)
                stop_report = collapse._stop_report(
                    started=started, limits=limits, guard=guard,
                    accounting=accounting, artifact_bytes=data_bytes)
                require_stop_conditions(stop_report)
                learn.require(stop_report["no_gradients"]
                              and stop_report["no_optimizer_steps"]
                              and stop_report["no_training_updates"]
                              and stop_report["no_extension_writes"],
                              "Zero-gradient policy was violated")
                status = decision["decision"]
                report = {
                    "format": learn.REPORT_FORMAT,
                    "version": learn.VERSION,
                    "mode": "ablation",
                    "status": status,
                    "output": str(output),
                    "config_sha256": _sha(Path(args.config)),
                    "declared_at_utc": config["declared_at_utc"],
                    "generated_at_utc": _now(),
                    "command": [sys.executable, *sys.argv],
                    "pins": pins["pins"],
                    "policy_declared": config["policy"],
                    "limits": limits,
                    "generation_stop_set": {
                        "stop_tokens": list(stops),
                        "source": stop_source,
                        "expected": list(EXPECTED_STOP_TOKENS),
                        "target_eos_token": int(config["target_eos_token"]),
                    },
                    "pair_layout": layout,
                    "controls": {
                        "names": list(CONTROL_NAMES),
                        "rows": len(rows),
                        "rows_with_all_four_controls": sum(
                            1 for entry in plan.values()
                            if entry["sibling_available"]),
                        "sibling_unavailable_case_ids": sorted(
                            case_id for case_id, entry in plan.items()
                            if not entry["sibling_available"]),
                        "per_row": plan,
                    },
                    "stages": {name: _stage_summary(payload)
                               for name, payload in stage_payloads.items()},
                    "ablation": {
                        "mode": config["ablation"]["mode"],
                        "tolerance": tolerance,
                        "valid_only": True,
                        "projected_space": "projected",
                        "global_dataset_mean": False,
                        "preserve_mean": True,
                        "expected_spatial_variance": 0.0,
                        "apply_to": config["ablation"]["apply_to"],
                        "receipts": receipts,
                        "image_stats_invariants": image_invariants,
                        "mean_preservation": preservation,
                        "receipt_invariants": receipt_invariants,
                    },
                    "ablation_invariants": ablation_invariants,
                    "baseline_identity": baseline,
                    "paired_ablation": paired,
                    "decision": decision,
                    "generation_evidence": {
                        "full": full_generations,
                        "mean_only": mean_generations,
                    },
                    "access": access,
                    "fit_contract": {
                        "coverage": full_stage["fit_coverage"],
                        "expected_kinds": {
                            "answer": learn.EXPECTED_ANSWER_ROWS,
                            "caption": learn.EXPECTED_CAPTION_ROWS,
                        },
                        "rows_retained": len(full_stage["fit_rows"]),
                        "contract": protocol["fit_contract"],
                    },
                    "resource": {
                        "guard": guard.report(),
                        "profile": summary,
                    },
                    "stop_conditions": stop_report,
                    "claims": learn.claims_block(),
                    "limitations": config["limitations"],
                }
                _write_report_fixed_point(output, report, limits, data_bytes)
                _write_json(ROOT / config["report_root"] / "summary.json", {
                    "format": learn.REPORT_FORMAT + ".summary",
                    "mode": "ablation",
                    "status": status,
                    "decision": decision,
                    "baseline_match": baseline["match"],
                    "ablation_invariants_ok": ablation_invariants["ok"],
                    "report": str(output / "report.json"),
                })
                return report
            except BaseException:
                if not profiler_stopped:
                    profiler.stop()
                raise
    except BaseException as exc:
        stop_report = collapse._stop_report(
            started=started, limits=limits, guard=guard,
            accounting=accounting, artifact_bytes=_data_bytes(args.output))
        _failure_report(output, exc, mode="ablation",
                        stop_report=stop_report,
                        partial={"loads": accounting.counts(),
                                 "evidence": {
                                     name: locals().get(name)
                                     for name in (
                                         "receipt_invariants", "image_invariants",
                                         "preservation", "full_generations",
                                         "mean_generations", "baseline",
                                         "paired", "decision",
                                         "full_normalized", "mean_normalized")}})
        raise


# ---------------------------------------------------------------------------
# Gradient mode
# ---------------------------------------------------------------------------


def _update_norm(before, after):
    import torch
    total = 0.0
    for name in PROJECTOR_TENSORS:
        total += float((after[name] - before[name]).square().sum())
    return math.sqrt(total)


def _backward_probe(adapter, session, tokenizer, service, protocol, scope,
                    row, *, role, stops, limits):
    import torch
    adapter.train()
    adapter.zero_grad(set_to_none=True)
    before = _snapshot_new_parameters(adapter)
    before_sha = curriculum.projector_state_sha256(adapter.vision_projector)
    visual = curriculum.observe(
        service, image_path=row["image_path"],
        image_sha256=row["image_sha256"], case_id=row["case_id"],
        scope=scope, ledger_epoch=session.ledger.epoch, spec=service.spec,
        source_id=protocol["vision_source"]["source_id"])
    prompt_tensor, input_ids, labels = curriculum._example_tensors(
        session, tokenizer, row,
        maximum_prompt=limits["max_prompt_tokens"],
        maximum_sequence=limits["max_sequence_tokens"])
    prompt_text = curriculum.curriculum_prompt(row["question"])
    with session.training_request(
            prompt_ids=prompt_tensor,
            prompt_mask=torch.ones_like(prompt_tensor, dtype=torch.bool),
            query=prompt_text, scope=scope,
            task=curriculum.task_graph(row["case_id"], prompt_text),
            visual=visual, enabled_ports=(), core_impl="none",
            enabled_branches=("memory",)) as request:
        result = session.forward(
            request, input_ids,
            attention_mask=torch.ones_like(input_ids, dtype=torch.bool),
            labels=labels)
        loss = result.loss
        learn.require(loss is not None and bool(torch.isfinite(loss)),
                      "Backward probe loss is invalid")
        session.backward(request, loss)
    audit = adapter.gradient_audit()
    after = _snapshot_new_parameters(adapter)
    after_sha = curriculum.projector_state_sha256(adapter.vision_projector)
    changed = _changed(before, after)
    record = {
        "role": role,
        "case_id": str(row["case_id"]),
        "example_id": str(row["example_id"]),
        "backward_only": True,
        "probe_applied_update": False,
        "optimizer_steps": 0,
        "weight_change_names": changed,
        "projector_state_sha256_before": before_sha,
        "projector_state_sha256_after": after_sha,
        "loss": float(loss.detach().cpu()),
        "projector_gradient_names": sorted(audit["new"]["present"]),
        "projector_gradients_nonzero_names": sorted(audit["new"]["nonzero"]),
        "projector_gradients_finite": audit["new"]["nonfinite"] == [],
        "nonprojector_gradient_names": sorted(
            audit["original"]["present"]),
        "backward_calls": 1,
        "fit_contract": balanced._verify_fit_tensor_contract(
            tokenizer, row, prompt_tensor, input_ids, labels, protocol,
            stop_tokens=stops),
    }
    adapter.zero_grad(set_to_none=True)
    return record


def _gradient_update(adapter, session, tokenizer, service, protocol, scope,
                     batch, optimizer, limits, stops, *, index, guard,
                     source_id):
    import torch
    optimizer.zero_grad(set_to_none=True)
    before = _snapshot_new_parameters(adapter)
    losses: list[float] = []
    role_losses = {"answer": [], "caption": []}
    role_supervised = {"answer": 0, "caption": 0}
    role_processed = {"answer": 0, "caption": 0}
    fit_checks = []
    backward_calls = 0
    for row in batch:
        kind = str(row["kind"])
        learn.require(kind in role_losses, f"Unknown update role: {kind}")
        visual = curriculum.observe(
            service, image_path=row["image_path"],
            image_sha256=row["image_sha256"], case_id=row["case_id"],
            scope=scope, ledger_epoch=session.ledger.epoch,
            spec=service.spec, source_id=source_id)
        prompt_tensor, input_ids, labels = curriculum._example_tensors(
            session, tokenizer, row,
            maximum_prompt=limits["max_prompt_tokens"],
            maximum_sequence=limits["max_sequence_tokens"])
        fit_check = balanced._verify_fit_tensor_contract(
            tokenizer, row, prompt_tensor, input_ids, labels, protocol,
            stop_tokens=stops)
        learn.require(fit_check is not None,
                      "The fit contract report is missing")
        fit_checks.append(fit_check)
        prompt_text = curriculum.curriculum_prompt(row["question"])
        with session.training_request(
                prompt_ids=prompt_tensor,
                prompt_mask=torch.ones_like(prompt_tensor, dtype=torch.bool),
                query=prompt_text, scope=scope,
                task=curriculum.task_graph(row["case_id"], prompt_text),
                visual=visual, enabled_ports=(), core_impl="none",
                enabled_branches=("memory",)) as request:
            result = session.forward(
                request, input_ids,
                attention_mask=torch.ones_like(input_ids, dtype=torch.bool),
                labels=labels)
            loss = result.loss
            learn.require(loss is not None and bool(torch.isfinite(loss)),
                          "Training loss is unavailable or nonfinite")
            learn.require(loss.requires_grad,
                          "Training loss is detached from autograd")
            session.backward(request, loss / len(batch))
        backward_calls += 1
        value = float(loss.detach().cpu())
        losses.append(value)
        role_losses[kind].append(value)
        role_supervised[kind] += int((labels[:, 1:] != -100).sum())
        role_processed[kind] += int(input_ids.numel())
    audit = adapter.gradient_audit()
    learn.require(audit["original"]["present"] == []
                  and audit["original"]["nonzero"] == []
                  and audit["original"]["nonfinite"] == [],
                  "Non-projector gradients were produced")
    learn.require(sorted(audit["new"]["present"])
                  == sorted(PROJECTOR_TENSORS)
                  and sorted(audit["new"]["nonzero"])
                  == sorted(PROJECTOR_TENSORS)
                  and audit["new"]["nonfinite"] == [],
                  "Projector gradient ownership/finiteness failed")
    grad_norm = float(torch.nn.utils.clip_grad_norm_(
        adapter.vision_projector.parameters(),
        protocol["train"]["max_grad_norm"],
        error_if_nonfinite=True).detach().cpu())
    learn.require(math.isfinite(grad_norm), "Gradient norm is nonfinite")
    learn.require(adapter.parameter_stamp(original_only=True)
                  == adapter._initial_base_stamp,
                  "An update begins with a changed sealed base")
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    after = _snapshot_new_parameters(adapter)
    changed = _changed(before, after)
    learn.require(changed and set(changed) <= set(PROJECTOR_TENSORS),
                  f"Update changed undeclared tensors: {changed}")
    record = {
        "format": learn.UPDATE_FORMAT,
        "label": "seed29_prefix",
        "update": index,
        "roles": {"answer": len(role_losses["answer"]),
                  "caption": len(role_losses["caption"])},
        "example_ids": [str(row["example_id"]) for row in batch],
        "case_ids": [str(row["case_id"]) for row in batch],
        "losses": losses,
        "loss": sum(losses) / len(losses),
        "role_losses": role_losses,
        "role_loss_mean": {
            role: (sum(values) / len(values) if values else None)
            for role, values in role_losses.items()},
        "role_supervised_tokens": role_supervised,
        "role_processed_tokens": role_processed,
        "gradient_audit": audit,
        "grad_norm": grad_norm,
        "backward_calls": backward_calls,
        "projector_update_norm": _update_norm(before, after),
        "changed_projector_tensors": changed,
        "nonprojector_gradient_names": [],
        "base_stamp_unchanged": (
            adapter.parameter_stamp(original_only=True)
            == adapter._initial_base_stamp),
        "response_only_labels": True,
        "fit_contract_checked": True,
        "fit_contract_rows": fit_checks,
        "fit_contract": {
            "rows": len(fit_checks),
            "kinds": dict(sorted(
                Counter(check["kind"]
                        for check in fit_checks).items())),
            "all_valid": all(check["contract_valid"] for check in fit_checks),
            "supervised_label_tokens": sum(
                check["supervised_label_tokens"] for check in fit_checks),
            "loss_tokens": sum(check["loss_tokens"] for check in fit_checks),
            "label_shift": 1,
            "stop_tokens": list(stops),
        },
    }
    learn.require(record["base_stamp_unchanged"],
                  "An update changed the sealed base")
    learn.require_update_role_mix(record)
    if guard is not None:
        guard.sample(index=index, event="gradient_update")
    return record, backward_calls


def _prefill_analysis(session, tokenizer, row, visual, scope, *, limits):
    import torch
    prompt_text = curriculum.curriculum_prompt(row["question"])
    prompt_tensor = curriculum.prompt_ids(
        tokenizer, prompt_text, session.device, limits["max_prompt_tokens"])
    response = tokenizer(row["response"], add_special_tokens=False)["input_ids"]
    learn.require(bool(response), "A training response tokenizes to nothing")
    request = session.prepare(
        prompt_ids=prompt_tensor,
        prompt_mask=torch.ones_like(prompt_tensor, dtype=torch.bool),
        query=prompt_text, scope=scope,
        task=curriculum.task_graph(row["case_id"], prompt_text),
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
        analysis = first_token_analysis(result.logits[0, -1], int(response[0]))
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


def _evaluate_train_controls(adapter, session, tokenizer, service, protocol,
                             scope, rows, plan, stops, labels, *, guard,
                             started, limits, output):
    import torch
    from run_vision_curriculum import (
        curriculum_prompt, generate_row, observe)
    from run_vision_balanced_alignment import _teacher_forced_loss
    caption_labels = {row["case_id"]: row for row in labels}
    rows_by_case = {str(row["case_id"]): row for row in rows}
    answer_rows = []
    caption_rows = []
    normalised = []
    adapter.eval()
    for position, row in enumerate(rows, 1):
        _check_wall(started, limits["max_seconds"], "train-fit scoring")
        if guard is not None and position % 4 == 0:
            guard.sample(index=position, event="train_fit_scoring")
        case_id = str(row["case_id"])
        plan_row = plan[case_id]
        for control in CONTROL_NAMES:
            image_case = {
                "correct": plan_row["correct_case_id"],
                "sibling": plan_row["sibling_case_id"],
                "shuffled": plan_row["shuffled_case_id"],
                "noimage": None,
            }[control]
            if control == "sibling" and not plan_row["sibling_available"]:
                normalised.append({
                    "case_id": case_id, "control": control,
                    "kind": row["kind"], "available": False,
                    "generated_ids": [], "generated_text": None,
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
                    ledger_epoch=session.ledger.epoch, spec=service.spec,
                    source_id=protocol["vision_source"]["source_id"])
            maximum_new = (int(limits["caption_max_new_tokens"])
                           if row["kind"] == "caption"
                           else int(limits["max_new_tokens"]))
            generated = generate_row(
                session, tokenizer, control=control, case=row,
                question=curriculum_prompt(row["question"]), visual=visual,
                scope=scope, maximum_prompt=limits["max_prompt_tokens"],
                maximum_new=maximum_new,
                image_case_id=image_case,
                image_sha256=(None if image_case is None
                              else rows_by_case[str(image_case)]
                              ["image_sha256"]))
            prefill = _prefill_analysis(
                session, tokenizer, row, visual, scope, limits=limits)
            prefill_matches = bool(generated["generated_ids"]) and (
                prefill["argmax_token"]
                == int(generated["generated_ids"][0]))
            record = {
                "format": learn.REPORT_FORMAT + ".generation",
                **generated,
                "case_id": case_id,
                "kind": row["kind"],
                "available": True,
                "hypothesis_count": 1,
                "prefill": prefill,
                "prefill_argmax_matches_first_token": prefill_matches,
            }
            if row["kind"] == "answer":
                record["target_text"] = row["response"]
                answer_rows.append(record)
            else:
                target = caption_labels[case_id]
                loss = _teacher_forced_loss(
                    session, tokenizer, protocol, scope, visual, row,
                    row["question"], target["answer"])
                record["teacher_forced_loss"] = loss
                record["target_text"] = target["answer"]
                record["target_visible_objects"] = [
                    list(item) for item in target["visible_objects"]]
                caption_rows.append(record)
            normalised.append({
                "case_id": case_id,
                "control": control,
                "kind": row["kind"],
                "available": True,
                "generated_ids": list(generated["generated_ids"]),
                "generated_text": generated["generated_text"],
                "prefill_argmax_matches_first_token":
                    record["prefill_argmax_matches_first_token"],
                "prefill": record.get("prefill"),
                "teacher_forced_loss": record.get("teacher_forced_loss"),
                "target_text": record.get("target_text"),
                "hypothesis_count": 1,
                "stop_tokens": list(generated["stop_tokens"]),
            })
    _write_jsonl(Path(output) / "train_answer_generations.jsonl",
                 answer_rows)
    _write_jsonl(Path(output) / "train_caption_generations.jsonl",
                 caption_rows)
    return {
        "answer": answer_rows,
        "caption": caption_rows,
        "normalised": normalised,
        "available": sum(1 for row in normalised if row["available"]),
        "unavailable": sum(1 for row in normalised if not row["available"]),
    }


def execute_gradient(args, config, pins):
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
                      f"The gradient diagnostic is refused: "
                      f"{precondition['violations']}")
        order = learn.train_only_access_order()
        order.access("train")
        training = balanced._load_training_v2(protocol)
        accessed.extend([
            str(pins["prepared_root"] / "training/alignment.jsonl"),
            str(pins["prepared_root"] / config["schedule"]["source"]),
            str(pins["prepared_root"] / config["schedule"]["caption_labels"]),
        ])
        accessed.extend(str(row["image_path"]) for row in training["training"])
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
        prefix = training["batches"][:int(config["schedule"]["prefix_updates"])]
        learn.require(len(prefix) == learn.EXPECTED_UPDATES,
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
                              "The gradient arm requires NativeFusedAdamW")
                learn.require(len(optimizer.param_groups) == 1
                              and len(optimizer.param_groups[0]["params"]) == 6
                              and sum(parameter.numel() for parameter in
                                      optimizer.param_groups[0]["params"])
                              == PROJECTOR_PARAMETER_COUNT,
                              "Optimizer ownership differs")
                profiler.mark("optimizer_ready")

                # Two backward-only probes (one answer, one caption), no step.
                probe_rows = {}
                for batch in prefix:
                    for row in batch:
                        probe_rows.setdefault(str(row["kind"]), row)
                probes = []
                probe_snapshot = _snapshot_new_parameters(adapter)
                for role in learn.EXPECTED_PROBE_ROLES:
                    probe = _backward_probe(
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

                adapter.train()
                trace = []
                update_backward_calls = 0
                optimizer_steps = 0
                for index, batch in enumerate(prefix, 1):
                    _check_wall(started, limits["max_seconds"],
                                "gradient update")
                    record, calls = _gradient_update(
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
                trace_report = learn.validate_gradient_trace(
                    trace, limits=limits)
                learn.require(trace_report["ok"],
                              f"Gradient trace failed: "
                              f"{trace_report['violations']}")
                learn.require(len(trace) == learn.EXPECTED_UPDATES,
                              "The gradient arm did not complete seven updates")
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
                profiler.mark("updates_complete")
                order.advance("train_fit")
                order.access("train_fit")

                evaluation = _evaluate_train_controls(
                    adapter, session, tokenizer, service, protocol, scope,
                    training["training"], plan, stops,
                    training["train_caption_labels"], guard=guard,
                    started=started, limits=limits, output=output)
                generation_report = learn.generation_hypothesis_report(
                    evaluation["normalised"],
                    answer_max_new_tokens=int(limits["max_new_tokens"]),
                    caption_max_new_tokens=int(
                        limits["caption_max_new_tokens"]))
                learn.require(generation_report["ok"],
                              f"Generation evidence failed: "
                              f"{generation_report['violations']}")
                rows_by_case = {str(row["case_id"]): row
                                for row in training["training"]}
                caption_labels = {row["case_id"]: row for row in
                                  training["train_caption_labels"]}
                scores = learn.score_train_fit(
                    records=evaluation["normalised"], plan=plan,
                    rows_by_case=rows_by_case,
                    caption_labels=caption_labels)
                gates = learn.train_fit_gate_report(
                    answer=scores["answer"], caption=scores["caption"],
                    gates=config["gates"])
                _write_json(output / "scores/train_fit_answer.json", {
                    "format": learn.REPORT_FORMAT + ".answer_score",
                    **scores["answer"],
                })
                _write_json(output / "scores/train_fit_caption.json", {
                    "format": learn.REPORT_FORMAT + ".caption_score",
                    **scores["caption"],
                })
                _write_json(output / "gates/train_fit.json", gates)
                access = learn.training_only_access_report(
                    order.snapshot(), accessed)
                engineering = learn.gradient_engineering_report(
                    trace=trace, probes=probes, limits=limits,
                    schedule_report=schedule_report,
                    generation_report=generation_report,
                    fit_coverage=coverage, access_report=access,
                    precondition=precondition,
                    updates_completed=len(trace),
                    optimizer_steps=optimizer_steps,
                    update_backward_calls=update_backward_calls,
                    probe_backward_calls=learn.EXPECTED_PROBE_BACKWARD_CALLS,
                    expected_kinds={
                        "answer": protocol["data"]["counts"]["train_answers"],
                        "caption": protocol["data"]["counts"][
                            "train_captions"]})
                decision = learn.gradient_decision(
                    precondition_report=precondition,
                    engineering_report=engineering,
                    gate_report=gates)
                learn.require(accounting.counts()
                              == {"model_loads": 1, "extension_loads": 0},
                              f"Load counts differ: {accounting.counts()}")
                learn.require(not any(
                    path.name == "extensions.safetensors"
                    for path in output.rglob("*")),
                    "The gradient arm wrote an extension")
                profiler.mark("report_ready")
                summary = profiler.stop()
                profiler_stopped = True
                data_bytes = _data_bytes(output)
                stop_report = learn.gradient_stop_conditions_report(
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
                    guard_stop_reason=guard.report().get("stop_reason"))
                learn.require_gradient_stop_conditions(stop_report)
                if engineering["ok"] is not True:
                    learn.require(False,
                                  "Gradient engineering invariants failed: "
                                  f"{engineering['violations']}")
                report = {
                    "format": learn.REPORT_FORMAT,
                    "version": learn.VERSION,
                    "mode": "gradient",
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
                    "engineering": engineering,
                    "generation_evidence": generation_report,
                    "train_fit": {
                        "answer": scores["answer"],
                        "caption": scores["caption"],
                    },
                    "gates": gates,
                    "access": access,
                    "decision": decision,
                    "resource": {
                        "guard": guard.report(),
                        "profile": summary,
                    },
                    "stop_conditions": stop_report,
                    "claims": learn.claims_block(),
                    "limitations": config["limitations"],
                }
                _write_report_fixed_point(output, report, limits, data_bytes)
                _write_json(ROOT / config["report_root"] / "summary.json", {
                    "format": learn.REPORT_FORMAT + ".summary",
                    "mode": "gradient",
                    "status": decision["status"],
                    "decision": decision,
                    "train_fit_gates": {
                        "answer_passed":
                            gates["answer_gates_passed"],
                        "caption_passed":
                            gates["caption_gates_passed"],
                        "failures": gates["gate_failures"],
                    },
                    "report": str(output / "report.json"),
                })
                return report
            except BaseException:
                if not profiler_stopped:
                    profiler.stop()
                raise
    except BaseException as exc:
        stop_report = learn.gradient_stop_conditions_report(
            wall_seconds=time.monotonic() - started,
            wall_limit_seconds=limits["max_seconds"],
            artifact_bytes=_data_bytes(args.output),
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
                               if guard is not None else None))
        _failure_report(output, exc, mode="gradient",
                        stop_report=stop_report,
                        partial={"loads": accounting.counts(),
                                 "trace_rows": (len(_read_jsonl(trace_path))
                                                if trace_path.exists() else 0)})
        raise


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=learn.MODES, required=True)
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
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        if os.environ.get(name) != "1":
            parser.error(f"{name}=1 is required")
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config = learn.load_config(config_path)
    args.config = config_path
    if config["mode"] != args.mode:
        parser.error("--mode differs from the declared config mode")
    if args.device != config["policy"]["device"]:
        parser.error("The declared diagnostic device must be mps")
    learn.require(args.output not in (ROOT, Path("/")),
                  "Refusing an unsafe output path")
    if args.output.exists() and (not args.output.is_dir()
                                 or any(args.output.iterdir())):
        parser.error("Use fresh output")
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    try:
        pins = verify_pins(config)
        if config["mode"] == "ablation":
            execute_ablation(args, config, pins)
        else:
            execute_gradient(args, config, pins)
    except BaseException as exc:
        if not (args.output / "failure_report.json").exists():
            if config["mode"] == "ablation":
                stop_report = collapse._stop_report(
                    started=started, limits=config["limits"], guard=None,
                    accounting=LoadAccounting(),
                    artifact_bytes=_data_bytes(args.output))
            else:
                stop_report = learn.gradient_stop_conditions_report(
                    wall_seconds=time.monotonic() - started,
                    wall_limit_seconds=config["limits"]["max_seconds"],
                    artifact_bytes=_data_bytes(args.output),
                    artifact_limit_bytes=config["limits"]["artifact_bytes"],
                    updates=0, optimizer_steps=0, update_backward_calls=0,
                    probe_backward_calls=0, model_loads=0,
                    extension_loads=0, extension_writes=0,
                    checkpoint_writes=0, downloads=0, cpu_fallback=False,
                    sealed_evaluation_access=0, guard_stop_reason=None)
            _failure_report(args.output, exc, mode=config["mode"],
                            stop_report=stop_report)
        raise


if __name__ == "__main__":
    main()

"""CPU-only deterministic tests for protocol, admission, scorer and lineage logic.

No native model is loaded, no GPU work is scheduled, and no quantity-quality
claim is made.  The tests use copies of the owned procedural data and tiny CPU
tensors only.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
import shutil
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from replica_lab import vision_qualification as vq  # noqa: E402
from replica_lab.optimizers import ReplicaOptimizer  # noqa: E402


DATA = ROOT / "data" / "vision_preparation_v1"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def copy_stage(tmp_path, stage):
    shutil.copytree(DATA / stage, tmp_path / stage)
    return tmp_path


def stage_pins(root, stage):
    result = {}
    for name, filename in (("manifest", "vision_manifest.json"),
                           ("model_rows", "model_rows.jsonl"),
                           ("verifier_cases", "verifier_cases.jsonl"),
                           ("verifier_scenes", "verifier_scenes.jsonl")):
        path = root / stage / filename
        result[name] = {"path": f"{stage}/{filename}", "sha256": file_sha(path)}
    return result


def vision_spec():
    return {
        "encoder_repo": "google/siglip2-base-patch16-224",
        "encoder_revision": "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2",
        "encoder_source_digest": "612923381c76ec5a9bed335d1c48827e3f2e506ac31b044b63b2031fadee6a0b",
        "processor_digest": "a" * 64,
        "encoder_width": 768,
        "image_size": 224,
        "patch_size": 16,
        "max_visual_tokens": 196,
        "projector_hidden": 2048,
        "feature_version": "siglip_patch_last_hidden.v1",
        "compute_dtype": "float32",
    }


def base_protocol(root, stage):
    pins = stage_pins(root, stage)
    protocol = {
        "format": vq.PROTOCOL_FORMAT,
        "template": False,
        "stage": stage,
        "root": str(root),
        "model": {"revision": "f97400052a43d642bbc6e9975e2397e3ae6a6b52",
                  "lock_sha256": "b" * 64},
        "vision": vision_spec(),
        "vision_digest": vq.fingerprint(vision_spec()),
        "vision_source": {
            "path": "source",
            "manifest": {"path": "source_manifest.json", "sha256": "c" * 64},
            "source_id": "replica_lab:procedural_scene_png_v1",
            "scope": {"subject_id": "vision-qualification", "branch_id": "primary",
                      "purpose": "interface_alignment_qualification", "as_of": 0,
                      "mode": "assistant_for_source", "snapshot_at": None},
        },
        "data": {
            "root": ".",
            "stage": stage,
            "manifest": pins["manifest"],
            "model_rows": pins["model_rows"],
            "verifier_cases": pins["verifier_cases"],
            "verifier_scenes": pins["verifier_scenes"],
        },
        "limits": {
            "max_prompt_tokens": 256, "max_new_tokens": 8, "max_rows": 4096,
            "max_seconds": 3600, "max_updates": 8, "max_repeats": 1,
            "max_eval_cases": 64, "max_export_bytes": 4294967296,
            "min_free_disk_bytes": 5368709120, "max_cache_bytes": 8388608,
        },
        "implementation_sha256": {
            relative: vq.sha256_file(ROOT / relative)
            for relative in vq.EXECUTABLE_IMPLEMENTATION_FILES
        },
    }
    protocol["v0"] = {
        "probe": {"case_id": "v0-f004-root-color",
                  "wrong_image_case_id": "v0-f009-root-color", "split": "audit"},
        "parity": {"prompt": "The quick brown fox crosses the quiet road.", "atol": 0.0},
        "backward": {"prompt": "Describe the visual interface probe.",
                     "target_text": "probe", "max_response_tokens": 8},
        "generation": {"max_new_tokens": 8, "seed": 17},
        "distinct_logit_min": 1e-6,
        "require_restart": True,
        "restart": {"atol": 0.0, "rtol": 0.0},
        "update": {"enabled": True, "optimizer": "sgd", "lr": 0.001},
    }
    protocol["v1"] = {
        "run_id": "owned-v1-fixture",
        "question_template": vq.VISUAL_QUESTION_TEMPLATE,
        "prerequisites": {
            "v0_protocol_sha256": "d" * 64,
            "v0_score_report": {"path": "score.json", "sha256": "e" * 64},
            "require_accepted": True,
            "data_admission": {"path": "admission.json", "sha256": "f" * 64},
        },
        "initial_checkpoint": {"path": "extension", "manifest_sha256": "1" * 64,
                               "weights_sha256": "2" * 64},
        "train": {"seed": 17, "updates": 4, "repeats": 1, "grad_accum": 1,
                  "lr_new": 1e-4, "momentum": 0.95, "ns_steps": 5,
                  "block_size": 2048, "weight_decay": 0.01,
                  "max_seconds": 600, "optimizer": "muon8bit",
                  "schedule_version": vq.VISION_TRAINING_SCHEDULE,
                  "repeat_policy": "once", "max_grad_norm": 1.0},
        "eval": {"splits": ["dev"], "max_cases": 4, "seed": 17,
                 "max_new_tokens": 8,
                 "required_controls": list(vq.CONTROLS)},
        "controls": {"wrong_source": "different_family",
                     "shuffled_source": "deterministic_derangement",
                     "text_context_template":
                         "External text context:\n{context}\n\nQuestion: {question}"},
        "quality": {
            "gates": {
                "min_correct_accuracy": 0.5,
                "min_correct_margin_wrong": 0.1,
                "min_correct_margin_shuffled": 0.1,
                "min_correct_margin_noimage": 0.1,
                "min_correct_margin_captiononly": 0.0,
                "min_correct_margin_ocrtextonly": 0.0,
                "min_unique_families": 2,
                "max_unknown_rate": 0.5,
                "min_text_retention_accuracy": 0.5,
                "max_text_retention_drop": 0.2,
            },
            "retention_baseline": "initial_checkpoint_same_prompts.v1",
        },
        "retention": {
            "prompts": {"path": "retention_prompts.jsonl", "sha256": "3" * 64},
            "labels": {"path": "retention_labels.jsonl", "sha256": "4" * 64},
            "splits": ["dev"], "max_cases": 2,
        },
    }
    return protocol


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def write_jsonl(path, rows):
    Path(path).write_text("".join(json.dumps(row) + "\n" for row in rows))


def make_v1_admission_fixture(tmp_path, protocol, dataset):
    """Create a named admission, text contexts, retention and accepted V0 report."""
    eval_cases = vq.select_eval_cases(dataset, protocol["v1"]["eval"])
    train_cases = [row for row in dataset.model_rows
                   if dataset.case(row["case_id"])["split"] == "train"][:4]
    training_rows = []
    for row in train_cases:
        case = dataset.case(row["case_id"])
        answer = dataset.label(row["case_id"])
        if vq.leakage_hits(row["question"], answer):
            continue
        asset = dataset.assets[case["asset_id"]]
        training_rows.append({
            "case_id": row["case_id"], "prompt": vq.visual_question(protocol, row["question"]),
            "response": answer, "image": asset["path"],
            "image_sha256": asset["sha256"], "family": case["group_id"],
            "group_id": case["group_id"], "split": case["split"]})
    assert len(training_rows) >= 2
    contexts = []
    for row in eval_cases:
        for kind in ("caption_only", "ocr_text_only"):
            text = "An independent textual scene description."
            contexts.append({"case_id": row["case_id"], "kind": kind,
                             "text": text, "text_sha256": vq.sha256_text(text),
                             "source_id": "fixture:independent_text",
                             "information_bound": "independent"})
    retention_prompts = [
        {"case_id": "ret-1", "prompt": "What is two plus two?",
         "family": "ret-family-1", "group_id": "ret-family-1", "split": "dev"},
        {"case_id": "ret-2", "prompt": "Name the color of a clear daytime sky.",
         "family": "ret-family-2", "group_id": "ret-family-2", "split": "dev"},
    ]
    retention_labels = [{"case_id": "ret-1", "answer": "4"},
                        {"case_id": "ret-2", "answer": "blue"}]
    context_path = tmp_path / "text_contexts.jsonl"
    training_path = tmp_path / "training_rows.jsonl"
    prompts_path = tmp_path / "retention_prompts.jsonl"
    labels_path = tmp_path / "retention_labels.jsonl"
    write_jsonl(context_path, contexts)
    write_jsonl(training_path, training_rows)
    write_jsonl(prompts_path, retention_prompts)
    write_jsonl(labels_path, retention_labels)
    protocol["v1"]["retention"]["prompts"]["sha256"] = file_sha(prompts_path)
    protocol["v1"]["retention"]["labels"]["sha256"] = file_sha(labels_path)
    splits = {}
    for split in ("train", "dev"):
        rows = [dict(row) for row in dataset.rows_for_split(split)]
        splits[split] = {
            "model_rows_sha256": vq.sha256_text(
                "".join(canonical(row) + "\n" for row in rows)),
            "model_row_count": len(rows),
            "family_count": len({dataset.case(row["case_id"])["group_id"]
                                 for row in rows}),
            "row_ids_sha256": vq.sha256_json(sorted(row["case_id"] for row in rows)),
        }
    admission = {
        "format": vq.ADMISSION_FORMAT,
        "stage": "v1",
        "admitted": True,
        "run_id": protocol["v1"]["run_id"],
        "eligible_rows": sum(item["model_row_count"] for item in splits.values()),
        "admitted_training_rows": len(training_rows),
        "data": {key: protocol["data"][key]["sha256"]
                 for key in ("manifest", "model_rows", "verifier_cases",
                             "verifier_scenes")},
        "splits": splits,
        "training_rows": {
            "path": str(training_path), "sha256": file_sha(training_path),
            "format": vq.TRAINING_ROWS_FORMAT,
            "rows_sha256": vq.rows_sha256(training_rows),
            "count": len(training_rows),
        },
        "text_contexts": {
            "path": str(context_path), "sha256": file_sha(context_path),
            "format": vq.TEXT_CONTEXTS_FORMAT,
            "rows_sha256": vq.rows_sha256(contexts),
        },
    }
    admission_path = tmp_path / "admission.json"
    write_json(admission_path, admission)
    run_report_path = tmp_path / "v0_run_report.json"
    write_json(run_report_path, {"status": "completed", "vision_spec": protocol["vision"]})
    score_report = {
        "format": vq.SCORE_FORMAT, "stage": "v0", "accepted": True,
        "engineering_pass": True, "protocol_sha256": "d" * 64,
        "run_report_path": str(run_report_path),
        "run_report_sha256": file_sha(run_report_path),
        "vision_spec_digest": protocol["vision_digest"],
        "vision_source_manifest_sha256": protocol["vision_source"]["manifest"]["sha256"],
        "model_revision": protocol["model"]["revision"],
        "model_lock_sha256": protocol["model"]["lock_sha256"],
        "extension": {"manifest_sha256": protocol["v1"]["initial_checkpoint"]["manifest_sha256"],
                      "weights_sha256": protocol["v1"]["initial_checkpoint"]["weights_sha256"]},
    }
    score_path = tmp_path / "score.json"
    write_json(score_path, score_report)
    protocol["v1"]["prerequisites"]["v0_score_report"] = {
        "path": str(score_path), "sha256": file_sha(score_path)}
    protocol["v1"]["prerequisites"]["data_admission"] = {
        "path": str(admission_path), "sha256": file_sha(admission_path)}
    return admission, training_rows, contexts, retention_prompts, retention_labels


# ---------------------------------------------------------------------------
# Protocol/template


def test_prospective_template_cannot_admit_a_native_run(tmp_path):
    copy_stage(tmp_path, "v0")
    protocol = base_protocol(tmp_path, "v0")
    protocol["template"] = True
    with pytest.raises(vq.QualificationError, match="Prospective template"):
        vq.validate_protocol(protocol)


def test_executable_protocol_rejects_missing_pins_and_unbounded_budget(tmp_path):
    copy_stage(tmp_path, "v0")
    protocol = base_protocol(tmp_path, "v0")
    vq.validate_protocol(protocol)
    protocol["implementation_sha256"].pop(vq.EXECUTABLE_IMPLEMENTATION_FILES[0])
    with pytest.raises(vq.QualificationError, match="must pin implementation"):
        vq.validate_protocol(protocol)
    protocol = base_protocol(tmp_path, "v0")
    protocol["v1"]["train"]["updates"] = 9  # history mutation irrelevant for v0
    vq.validate_protocol(protocol)
    protocol = base_protocol(tmp_path, "v0")
    protocol["limits"]["max_updates"] = 0
    with pytest.raises(vq.QualificationError, match="max_updates"):
        vq.validate_protocol(protocol)


def test_implementation_pin_tamper_is_rejected():
    protocol = {"root": str(ROOT),
                "implementation_sha256": {
                    relative: vq.sha256_file(ROOT / relative)
                    for relative in vq.EXECUTABLE_IMPLEMENTATION_FILES}}
    assert vq.verify_implementation_pins(protocol, ROOT)
    protocol["implementation_sha256"][vq.EXECUTABLE_IMPLEMENTATION_FILES[0]] = "0" * 64
    with pytest.raises(vq.QualificationError, match="Implementation bytes changed"):
        vq.verify_implementation_pins(protocol, ROOT)


# ---------------------------------------------------------------------------
# Data isolation and tamper rejection


def test_actual_v0_dataset_reconstructs_labels_and_rejects_tampering(tmp_path):
    copy_stage(tmp_path, "v0")
    protocol = base_protocol(tmp_path, "v0")
    dataset = vq.load_verified_dataset(protocol)
    assert len(dataset.assets) == 160
    assert len(dataset.model_rows) == 640
    case_id = "v0-f004-root-color"
    assert dataset.label(case_id) == dataset.case(case_id)["answer"]
    assert vq.reconstruct_label(dataset.case(case_id),
                                dataset.scenes[dataset.case(case_id)["asset_id"]]["scene"])

    # A gold/answer field in a model row is rejected even when the file hash matches.
    rows_path = tmp_path / "v0" / "model_rows.jsonl"
    rows = [json.loads(line) for line in rows_path.read_text().splitlines() if line]
    rows[0]["answer"] = "blue"
    write_jsonl(rows_path, rows)
    protocol["data"]["model_rows"]["sha256"] = file_sha(rows_path)
    with pytest.raises(vq.QualificationError, match="model row fields differ"):
        vq.load_verified_dataset(protocol)

    # Restore rows and tamper a verifier answer.
    shutil.copyfile(DATA / "v0" / "model_rows.jsonl", rows_path)
    protocol["data"]["model_rows"]["sha256"] = file_sha(rows_path)
    cases_path = tmp_path / "v0" / "verifier_cases.jsonl"
    cases = [json.loads(line) for line in cases_path.read_text().splitlines() if line]
    cases[0]["answer"] = "red"
    write_jsonl(cases_path, cases)
    protocol["data"]["verifier_cases"]["sha256"] = file_sha(cases_path)
    with pytest.raises(vq.QualificationError, match="Verifier answer differs"):
        vq.load_verified_dataset(protocol)

    # Restore cases and tamper an image byte.
    shutil.copyfile(DATA / "v0" / "verifier_cases.jsonl", cases_path)
    protocol["data"]["verifier_cases"]["sha256"] = file_sha(cases_path)
    asset = next((tmp_path / "v0" / "assets").glob("*.png"))
    asset.write_bytes(asset.read_bytes() + b"X")
    with pytest.raises(vq.QualificationError, match="Asset byte size differs|Asset bytes changed"):
        vq.load_verified_dataset(protocol)

    # Restore the asset and tamper scene truth.
    shutil.copyfile(DATA / "v0" / "assets" / asset.name, asset)
    scenes_path = tmp_path / "v0" / "verifier_scenes.jsonl"
    scenes = [json.loads(line) for line in scenes_path.read_text().splitlines() if line]
    scenes[0]["scene"]["objects"][0]["color"] = (
        "red" if scenes[0]["scene"]["objects"][0]["color"] != "red" else "blue")
    write_jsonl(scenes_path, scenes)
    protocol["data"]["verifier_scenes"]["sha256"] = file_sha(scenes_path)
    with pytest.raises(vq.QualificationError, match="Scene truth changed|Asset/scene hash"):
        vq.load_verified_dataset(protocol)


def test_wrong_and_shuffled_controls_are_deterministic_and_distinct(tmp_path):
    copy_stage(tmp_path, "v1")
    protocol = base_protocol(tmp_path, "v1")
    dataset = vq.load_verified_dataset(protocol)
    case_ids = [row["case_id"] for row in vq.select_eval_cases(
        dataset, protocol["v1"]["eval"])]
    first = vq.assign_controls(dataset, case_ids)
    second = vq.assign_controls(dataset, case_ids)
    assert first == second
    for case_id in case_ids:
        own = dataset.case(case_id)
        wrong = dataset.case(first["wrong"][case_id])
        shuffled = dataset.case(first["shuffled"][case_id])
        assert wrong["group_id"] != own["group_id"]
        assert wrong["asset_id"] != own["asset_id"]
        assert shuffled["asset_id"] != own["asset_id"]


# ---------------------------------------------------------------------------
# Admission and hard prerequisites


def test_v1_admission_and_accepted_v0_are_hard_prerequisites(tmp_path):
    copy_stage(tmp_path, "v1")
    protocol = base_protocol(tmp_path, "v1")
    dataset = vq.load_verified_dataset(protocol)
    admission, training_rows, contexts, prompts, labels = make_v1_admission_fixture(
        tmp_path, protocol, dataset)
    vq.validate_protocol(protocol)
    report, digest = vq.verify_v0_prerequisite(protocol)
    assert report["accepted"] is True
    verified = vq.verify_admission(
        protocol, dataset, train_splits=["train"], eval_splits=["dev"])
    assert len(verified["_training_rows"]) == len(training_rows)
    assert len(verified["_text_contexts"]) == len(contexts)
    assert len(verified["_retention_prompts"]) == 2

    # Unaccepted or stale V0 evidence blocks V1 before any native work.
    score_path = Path(protocol["v1"]["prerequisites"]["v0_score_report"]["path"])
    score = json.loads(score_path.read_text())
    score["accepted"] = False
    write_json(score_path, score)
    protocol["v1"]["prerequisites"]["v0_score_report"]["sha256"] = file_sha(score_path)
    with pytest.raises(vq.QualificationError, match="not independently accepted"):
        vq.verify_v0_prerequisite(protocol)

    score["accepted"] = True
    write_json(score_path, score)
    protocol["v1"]["prerequisites"]["v0_score_report"]["sha256"] = file_sha(score_path)
    admission_path = Path(protocol["v1"]["prerequisites"]["data_admission"]["path"])
    admission["admitted"] = False
    write_json(admission_path, admission)
    protocol["v1"]["prerequisites"]["data_admission"]["sha256"] = file_sha(admission_path)
    with pytest.raises(vq.QualificationError, match="must explicitly admit"):
        vq.verify_admission(protocol, dataset, train_splits=["train"],
                            eval_splits=["dev"])

    # A required text control cannot silently disappear.
    admission["admitted"] = True
    context_path = Path(admission["text_contexts"]["path"])
    write_jsonl(context_path, contexts[:-1])
    admission["text_contexts"]["sha256"] = file_sha(context_path)
    admission["text_contexts"]["rows_sha256"] = vq.rows_sha256(contexts[:-1])
    write_json(admission_path, admission)
    protocol["v1"]["prerequisites"]["data_admission"]["sha256"] = file_sha(admission_path)
    with pytest.raises(vq.QualificationError, match="lacks pinned text"):
        vq.verify_admission(protocol, dataset, train_splits=["train"],
                            eval_splits=["dev"])



def test_v1_fresh_initialization_allows_only_connector_width_change(tmp_path):
    copy_stage(tmp_path, "v1")
    protocol = base_protocol(tmp_path, "v1")
    dataset = vq.load_verified_dataset(protocol)
    make_v1_admission_fixture(tmp_path, protocol, dataset)
    protocol["v1"]["initial_checkpoint"] = None
    protocol["vision"]["projector_hidden"] = 4096
    protocol["vision_digest"] = vq.fingerprint(protocol["vision"])
    vq.validate_protocol(protocol)
    report, _ = vq.verify_v0_prerequisite(protocol)
    assert report["accepted"] is True

    protocol["vision"]["encoder_source_digest"] = "e" * 64
    protocol["vision_digest"] = vq.fingerprint(protocol["vision"])
    with pytest.raises(vq.QualificationError, match="another vision encoder"):
        vq.verify_v0_prerequisite(protocol)


def test_training_schedule_and_budget_are_deterministic_and_bounded(tmp_path):
    copy_stage(tmp_path, "v1")
    protocol = base_protocol(tmp_path, "v1")
    dataset = vq.load_verified_dataset(protocol)
    rows = [dict(row) for row in dataset.rows_for_split("train")][:8]
    schedule = vq.training_schedule(rows, protocol["v1"]["train"])
    assert schedule == vq.training_schedule(rows, protocol["v1"]["train"])
    assert len(schedule) == protocol["v1"]["train"]["updates"]
    assert all(len(step) == 1 for step in schedule)
    assert {row["case_id"] for step in schedule for row in step} <= {
        row["case_id"] for row in rows}
    plan = {"updates": 4, "evaluation_cases": 4, "generation_tokens": 8,
            "training_rows": 8, "max_seconds": 10}
    vq.validate_budget(plan, protocol["limits"])
    plan["updates"] = 99
    with pytest.raises(vq.QualificationError, match="max_updates"):
        vq.validate_budget(plan, protocol["limits"])
    exhausted = {**protocol["v1"]["train"], "seed": 1, "updates": 2}
    with pytest.raises(vq.QualificationError):
        vq.training_schedule(rows[:1], exhausted)


# ---------------------------------------------------------------------------
# Independent scoring and quality separation


def test_scorer_reconstructs_answers_counts_unknowns_and_families(tmp_path):
    copy_stage(tmp_path, "v0")
    protocol = base_protocol(tmp_path, "v0")
    dataset = vq.load_verified_dataset(protocol)
    case_ids = ["v0-f004-root-color", "v0-f009-root-color"]
    controls = list(vq.CONTROLS)
    generations = []
    for control in controls:
        for case_id in case_ids:
            expected = dataset.label(case_id)
            generated = (expected if control == "correct"
                         else "unknown" if control == "noimage"
                         else "wrong")
            generations.append({"control": control, "case_id": case_id,
                                "generated_text": generated})
    scores = vq.score_controls(dataset, generations, controls=controls)
    assert scores["controls"]["correct"]["accuracy"] == 1.0
    assert scores["controls"]["correct"]["unknown"] == 0
    assert scores["controls"]["correct"]["unique_families"] == 2
    assert scores["controls"]["noimage"]["unknown_rate"] == 1.0
    assert scores["paired"]["wrong"]["correct_margin"] == 1.0
    rows = scores["rows"]
    assert all("expected" in row and "generated" in row for row in rows)
    assert all(row["family"] in {"v0-family-004", "v0-family-009"} for row in rows)


def test_retention_policy_separates_baseline_capability_from_nonregression(tmp_path):
    copy_stage(tmp_path, "v0")
    protocol = base_protocol(tmp_path, "v0")
    dataset = vq.load_verified_dataset(protocol)
    case_ids = ["v0-f004-root-color", "v0-f009-root-color"]
    controls = list(vq.CONTROLS)
    generations = []
    for control in controls:
        for case_id in case_ids:
            generated = dataset.label(case_id) if control == "correct" else "wrong"
            generations.append({"control": control, "case_id": case_id,
                                "generated_text": generated})
    scores = vq.score_controls(dataset, generations, controls=controls)
    gates = dict(protocol["v1"]["quality"]["gates"])
    gates["min_correct_accuracy"] = 0.5
    gates["min_correct_margin_wrong"] = 0.0
    gates["min_correct_margin_shuffled"] = 0.0
    gates["min_correct_margin_noimage"] = 0.0
    gates["min_correct_margin_captiononly"] = 0.0
    gates["min_correct_margin_ocrtextonly"] = 0.0
    gates["min_unique_families"] = 2
    gates["max_unknown_rate"] = 1.0
    gates["min_text_retention_accuracy"] = 0.75
    gates["max_text_retention_drop"] = 0.0
    retention = {"accuracy": 22 / 32, "cases": 32, "correct": 22, "unknown": 0,
                 "unknown_rate": 0.0, "unique_families": 32, "rows": []}
    # The projector-only arm reproduces text exactly, but the frozen baseline is
    # itself below the declared capability floor.
    ineligible = vq.apply_quality_gates(
        scores, retention, gates, retention_baseline_accuracy=22 / 32,
        retention_policy="component_nonregression.v1", text_tokens_unchanged=True)
    assert ineligible["quality_pass"] is False
    assert "baseline_capability_ineligible" in ineligible["failures"]
    assert "min_text_retention_accuracy" not in ineligible["failures"]
    assert ineligible["values"]["baseline_eligible"] is False
    assert ineligible["values"]["text_tokens_unchanged"] is True

    eligible = vq.apply_quality_gates(
        scores, retention, gates, retention_baseline_accuracy=0.9,
        retention_policy="component_nonregression.v1", text_tokens_unchanged=True)
    assert eligible["quality_pass"] is False
    assert eligible["values"]["baseline_eligible"] is True
    assert "baseline_capability_ineligible" not in eligible["failures"]
    assert "max_text_retention_drop" in eligible["failures"]

    passing_retention = dict(retention, accuracy=0.9, correct=29)
    passed = vq.apply_quality_gates(
        scores, passing_retention, gates, retention_baseline_accuracy=0.9,
        retention_policy="component_nonregression.v1", text_tokens_unchanged=True)
    assert passed["quality_pass"] is True

    changed_tokens = vq.apply_quality_gates(
        scores, retention, gates, retention_baseline_accuracy=0.9,
        retention_policy="component_nonregression.v1", text_tokens_unchanged=False)
    assert "text_tokens_changed" in changed_tokens["failures"]

    # Legacy protocols keep the original combined gate.
    legacy = vq.apply_quality_gates(
        scores, retention, gates, retention_baseline_accuracy=0.9)
    assert "min_text_retention_accuracy" in legacy["failures"]
    assert "baseline_capability_ineligible" not in legacy["failures"]


def test_quality_can_reject_while_engineering_passes(tmp_path):
    copy_stage(tmp_path, "v0")
    protocol = base_protocol(tmp_path, "v0")
    dataset = vq.load_verified_dataset(protocol)
    case_ids = ["v0-f004-root-color", "v0-f009-root-color"]
    controls = list(vq.CONTROLS)
    generations = []
    for control in controls:
        for case_id in case_ids:
            generated = dataset.label(case_id) if control == "correct" and case_id == case_ids[0] else "wrong"
            generations.append({"control": control, "case_id": case_id,
                                "generated_text": generated})
    scores = vq.score_controls(dataset, generations, controls=controls)
    retention = {"accuracy": 0.5, "cases": 2, "correct": 1, "unknown": 0,
                 "unknown_rate": 0.0, "unique_families": 2, "rows": []}
    gates = dict(protocol["v1"]["quality"]["gates"])
    gates["min_correct_accuracy"] = 0.99
    quality = vq.apply_quality_gates(scores, retention, gates,
                                    retention_baseline_accuracy=1.0)
    assert quality["quality_pass"] is False
    assert "min_correct_accuracy" in quality["failures"]
    engineering = vq.summarize_engineering({"training": True, "export": True})
    assert engineering == {"engineering_pass": True,
                           "engineering_failures": [],
                           "checks": {"export": True, "training": True}}
    gates["min_correct_accuracy"] = 0.5
    gates["max_text_retention_drop"] = 0.7
    gates["min_text_retention_accuracy"] = 0.3
    accepted_quality = vq.apply_quality_gates(
        scores, retention, gates, retention_baseline_accuracy=1.0)
    assert accepted_quality["quality_pass"] is True


# ---------------------------------------------------------------------------
# Optimizer ownership and audited dtypes


def test_projector_optimizer_routes_2d_to_muon_and_1d_to_fp32_adamw():
    projector = torch.nn.Sequential(
        torch.nn.LayerNorm(4), torch.nn.Linear(4, 3), torch.nn.GELU(),
        torch.nn.Linear(3, 2))
    for parameter in projector.parameters():
        parameter.requires_grad_(True)
    train = {"lr_new": 1e-3, "weight_decay": 0.01, "momentum": 0.95,
             "ns_steps": 5, "block_size": 4}
    groups = vq.projector_optimizer_groups(projector, train,
                                           prefix="vision_projector.")
    route = vq.optimizer_route(groups)
    assert {item["algorithm"] for item in route} == {"muon8bit", "adamw"}
    assert all((len(item["shape"]) == 2) == (item["algorithm"] == "muon8bit")
               for item in route)
    optimizer = ReplicaOptimizer(groups, "muon8bit")
    x = torch.randn(2, 4)
    target = torch.randn(2, 2)
    loss = (projector(x) - target).pow(2).mean()
    loss.backward()
    optimizer.step()
    audit = optimizer.audit()
    vq.validate_optimizer_route(route, audit)
    assert set(audit["state_bytes_by_dtype"]) == {"torch.int8", "torch.float32"}
    tampered = [dict(item) for item in route]
    tampered[0]["algorithm"], tampered[-1]["algorithm"] = (
        tampered[-1]["algorithm"], tampered[0]["algorithm"])
    with pytest.raises(vq.QualificationError, match="wrong algorithm|route.*Muon8bit"):
        vq.validate_optimizer_route(tampered, audit)


# ---------------------------------------------------------------------------
# Extension inventory/lineage and raw-trace verification


def make_extension(path, *, projector_weight, other_weight, vision):
    from safetensors.torch import save_file
    path.mkdir(parents=True, exist_ok=True)
    tensors = {
        "record_writer.weight": torch.tensor([other_weight], dtype=torch.float32),
        "vision_projector.0.weight": torch.tensor([[projector_weight]],
                                                   dtype=torch.float32),
    }
    save_file(tensors, str(path / "extensions.safetensors"))
    manifest = {"training_scope": "visual_prefix_text", "vision": dict(vision),
                "format_version": 1}
    receipt = {
        "format": "replica.minicpm5.extensions.v4",
        "manifest": manifest,
        "manifest_digest": vq.fingerprint(manifest),
        "regime": "modules_only",
        "weights_sha256": file_sha(path / "extensions.safetensors"),
        "parameters": {},
        "base_saved": False,
        "notes": "test",
        "components": {
            "record_writer": ["record_writer.weight"],
            "vision_projector": ["vision_projector.0.weight"],
        },
    }
    write_json(path / "manifest.json", receipt)
    return receipt


def test_extension_inventory_and_only_projector_lineage(tmp_path):
    vision = vision_spec()
    initial = tmp_path / "initial"
    final = tmp_path / "final"
    make_extension(initial, projector_weight=0.0, other_weight=1.0, vision=vision)
    make_extension(final, projector_weight=2.0, other_weight=1.0, vision=vision)
    receipt = vq.validate_extension_inventory(
        final, expected_names=["record_writer.weight",
                               "vision_projector.0.weight"],
        vision_pins=vision)
    assert receipt["base_saved"] is False
    lineage = vq.compare_extension_tensors(initial, final)
    assert lineage["changed"] == ["vision_projector.0.weight"]
    assert lineage["frozen"] == ["record_writer.weight"]
    make_extension(final, projector_weight=2.0, other_weight=3.0, vision=vision)
    with pytest.raises(vq.QualificationError, match="Non-projector extension tensors changed"):
        vq.compare_extension_tensors(initial, final)
    with pytest.raises(vq.QualificationError, match="inventory differs"):
        vq.validate_extension_inventory(
            final, expected_names=["vision_projector.0.weight"],
            vision_pins=vision)


def test_generation_logits_and_restart_compare_reject_tampering(tmp_path):
    from safetensors.torch import save_file
    logits = torch.zeros(2, 5)
    logits[0, 1] = 5.0
    logits[1, 3] = 5.0
    logits_path = tmp_path / "logits.safetensors"
    save_file({"logits": logits}, str(logits_path))
    trace = {"prompt_ids": [10, 11],
             "prompt_sha256": vq.sha256_json([10, 11]),
             "generated_ids": [1, 3],
             "logits_sha256": file_sha(logits_path)}
    verified = vq.validate_generation_trace(
        trace, logits_path, protocol_prompt_ids=[10, 11], max_new_tokens=4)
    assert verified["generated_ids"] == [1, 3]
    bad = dict(trace, generated_ids=[1, 2])
    with pytest.raises(vq.QualificationError, match="do not match"):
        vq.validate_generation_trace(
            bad, logits_path, protocol_prompt_ids=[10, 11], max_new_tokens=4)
    with pytest.raises(vq.QualificationError, match="prompt tokens differ"):
        vq.validate_generation_trace(
            trace, logits_path, protocol_prompt_ids=[10, 12], max_new_tokens=4)

    restart_path = tmp_path / "restart.safetensors"
    save_file({"logits": logits.clone()}, str(restart_path))
    prior = {"generated_ids": [1, 3], "prompt_ids": [10, 11],
             "logits_path": str(logits_path)}
    restart = {"generated_ids": [1, 3], "prompt_ids": [10, 11],
               "logits_path": str(restart_path)}
    comparison = vq.compare_restart_traces(prior, restart, atol=0.0, rtol=0.0)
    assert comparison["tokens_equal"] and comparison["logits_equal"]
    changed = logits.clone()
    changed[0, 1] = 0.5
    save_file({"logits": changed}, str(restart_path))
    with pytest.raises(vq.QualificationError, match="differ beyond tolerance"):
        vq.compare_restart_traces(prior, restart, atol=0.0, rtol=0.0)
    with pytest.raises(vq.QualificationError, match="generation tokens differ"):
        vq.compare_restart_traces(
            prior, {"generated_ids": [1, 2], "prompt_ids": [10, 11],
                    "logits_path": str(restart_path)}, atol=0.0, rtol=0.0)


def test_prompt_leakage_and_unknown_normalisation():
    assert vq.leakage_hits("What shape is the blue object?", "blue")
    assert not vq.leakage_hits("What shape is the blue object?", "circle")
    assert vq.answer_correct(" Blue. ", "blue")
    assert not vq.answer_correct("dark blue", "blue")
    assert vq.is_unknown("I don't know")
    assert vq.is_unknown("")
    assert not vq.is_unknown("blue")
    assert vq.normalise_answer("  YES. ") == "yes"


@pytest.mark.parametrize("mutation", ["label", "prompt", "named_run", "count", "legacy_format"])
def test_named_admission_rejects_rehashed_semantic_substitutions(tmp_path, mutation):
    copy_stage(tmp_path, "v1")
    protocol = base_protocol(tmp_path, "v1")
    dataset = vq.load_verified_dataset(protocol)
    admission, rows, _, _, _ = make_v1_admission_fixture(tmp_path, protocol, dataset)
    if mutation in ("label", "prompt"):
        rows[0]["response" if mutation == "label" else "prompt"] = "unverified substitution"
        path = Path(admission["training_rows"]["path"])
        write_jsonl(path, rows)
        admission["training_rows"].update(sha256=file_sha(path), rows_sha256=vq.rows_sha256(rows))
    elif mutation == "named_run":
        admission["run_id"] = "foreign-run"
    elif mutation == "count":
        admission["admitted_training_rows"] = admission["eligible_rows"]
    else:
        admission["format"] = "replica.vision_training_admission.v1"
    path = Path(protocol["v1"]["prerequisites"]["data_admission"]["path"])
    write_json(path, admission)
    protocol["v1"]["prerequisites"]["data_admission"]["sha256"] = file_sha(path)
    with pytest.raises(vq.QualificationError):
        vq.verify_admission(protocol, dataset, train_splits=["train"], eval_splits=["dev"])


def test_v1_cannot_substitute_an_unqualified_initial_projector(tmp_path):
    copy_stage(tmp_path, "v1")
    protocol = base_protocol(tmp_path, "v1")
    dataset = vq.load_verified_dataset(protocol)
    make_v1_admission_fixture(tmp_path, protocol, dataset)
    protocol["v1"]["initial_checkpoint"]["weights_sha256"] = "0" * 64
    with pytest.raises(vq.QualificationError):
        vq.verify_v0_prerequisite(protocol)


def test_protected_retention_inputs_do_not_echo_verifier_answers():
    from replica_lab.vision_admission import retention_cases
    prompts, labels = retention_cases()
    by_id = {row["case_id"]: row["answer"] for row in labels}
    assert all(not vq.leakage_hits(row["prompt"], by_id[row["case_id"]]) for row in prompts)
    assert set(by_id) == {row["case_id"] for row in prompts}

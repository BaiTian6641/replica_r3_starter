#!/usr/bin/env python3
"""Independent CPU scorer for V0/V1 vision-qualification raw artifacts.

This script never loads the native model and never runs the runner.  It reloads
the protocol/data/applicable admission by hash, re-tokenizes every declared
prompt, decodes raw token IDs with the sealed local CPU tokenizer, reconstructs
every verifier label from verifier-only scene truth, recomputes greedy-token
consistency from saved logits, checks extension lineage, and applies predeclared
numeric gates.  Engineering and quality remain separate so a useful engineering
path can be established while perception quality is honestly rejected.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import traceback

for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
    os.environ[key] = "1"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from replica_lab.vision_qualification import (  # noqa: E402
    CONTROLS, QualificationError, TEXT_CONTROL_KIND, answer_correct,
    apply_quality_gates, assign_controls, build_text_context_prompt,
    compare_extension_tensors, ensure_fresh_directory, load_verified_dataset,
    load_extension_receipt, normalise_answer, require, require_completed_run,
    require_same_implementation, resolve_project_path, score_controls,
    score_retention, select_eval_cases, select_probe, sha256_file, sha256_json,
    summarize_engineering, training_schedule, validate_extension_inventory,
    validate_generation_trace, validate_optimizer_route,
    validate_protocol, validate_restart_report, verify_admission,
    verify_implementation_pins, verify_v0_prerequisite,
)


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def read_jsonl(path):
    rows = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        require(isinstance(value, dict), f"{path}:{line_number} is not an object")
        rows.append(value)
    require(bool(rows), f"{path} is empty")
    return rows


def artifact_bindings(run, report, protocol):
    artifacts = report.get("artifacts")
    require(isinstance(artifacts, dict) and bool(artifacts),
            "Run report lacks an artifact hash inventory")
    for relative, digest in artifacts.items():
        path = Path(run) / relative
        require(path.is_file(), f"Reported artifact is missing: {relative}")
        require(sha256_file(path) == digest,
                f"Reported artifact bytes changed: {relative}")
    require({"profile_samples.jsonl", "profile_summary.json"} <= set(artifacts),
            "Native profile files are not hash-bound")
    profile = json.loads((Path(run) / "profile_summary.json").read_text())
    require(profile == report.get("profile"), "Native profile summary differs from the report")
    require(profile["device"] == "mps"
            and report.get("environment", {}).get("PYTORCH_ENABLE_MPS_FALLBACK") == "0",
            "Native execution or disabled CPU fallback was not established")
    samples = [row for row in read_jsonl(Path(run) / "profile_samples.jsonl")
               if row.get("type") == "sample"]
    require(bool(samples) and len(samples) == profile["sample_count"]
            and all(type(row.get("mps_driver_bytes")) is int
                    and row["mps_driver_bytes"] >= 0 for row in samples),
            "Native driver sampling is incomplete or malformed")
    peak = max(row["mps_driver_bytes"] for row in samples)
    require(peak == profile["memory"]["mps_driver_peak_sampled_bytes"]
            and 0 < peak <= 53_687_091_200,
            "Sampled native peak differs from raw samples or exceeds 50 GiB")
    elapsed = profile["elapsed_seconds"]
    require(type(elapsed) in (int, float) and 0 < elapsed <= protocol["limits"]["max_seconds"],
            "Native run exceeded its declared wall-time acceptance ceiling")
    return artifacts


def verify_source_pins(report, protocol):
    manifest = resolve_project_path(
        protocol, protocol["vision_source"]["manifest"]["path"])
    require(manifest.is_file(), "Pinned vision source manifest is missing")
    digest = sha256_file(manifest)
    require(digest == protocol["vision_source"]["manifest"]["sha256"],
            "Vision source manifest bytes changed")
    require(report.get("vision_source_manifest_sha256") == digest,
            "Run used a different vision source manifest")
    require(report.get("vision_spec") == dict(protocol["vision"]),
            "Run vision spec differs from the protocol")
    require(report.get("vision_spec_digest") == protocol["vision_digest"],
            "Run vision spec digest differs")
    return digest


def load_tokenizer(args, protocol):
    from transformers import AutoTokenizer
    from replica_lab.model_adapter import validate_model_lock
    require(sha256_file(args.model_lock) == protocol["model"]["lock_sha256"],
            "Model lock bytes differ from the protocol pin")
    lock = json.loads(args.model_lock.read_text(encoding="utf-8"))
    require(lock.get("revision") == protocol["model"]["revision"],
            "Model revision differs from the protocol pin")
    validate_model_lock(args.model_path, lock)
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model_path), local_files_only=True, trust_remote_code=False)
    require(tokenizer.is_fast, "Scorer requires the sealed fast tokenizer")
    require(tokenizer.eos_token_id is not None,
            "Tokenizer must declare an explicit EOS token")
    return tokenizer


def expected_prompt_ids(tokenizer, text):
    from replica_lab.architecture.loading import encode_prompt
    ids = encode_prompt(tokenizer, text, "cpu")
    return [int(value) for value in ids[0].tolist()]


def decode_ids(tokenizer, ids):
    return tokenizer.decode([int(value) for value in ids], skip_special_tokens=True,
                            clean_up_tokenization_spaces=False).strip()


def forbidden_row_fields(row, name):
    forbidden = {"answer", "label", "gold", "expected", "correct", "score",
                 "check", "scene", "scene_sha256"}
    present = sorted(forbidden & set(row))
    require(not present, f"{name} contains evaluator-label/cache fields: {present}")


def verify_parity(run, report, tokenizer, protocol):
    parity = report.get("text_parity")
    require(isinstance(parity, dict), "V0 report lacks text parity evidence")
    expected = expected_prompt_ids(tokenizer, protocol["v0"]["parity"]["prompt"])
    require(parity["prompt_ids"] == expected,
            "Text parity prompt tokens differ from independent tokenization")
    require(parity["prompt_sha256"] == sha256_json(expected),
            "Text parity prompt hash differs")
    require(sha256_file(Path(run) / parity["base_logits_path"])
            == parity["base_logits_sha256"],
            "Parity base logits bytes changed")
    require(sha256_file(Path(run) / parity["adapter_logits_path"])
            == parity["adapter_logits_sha256"],
            "Parity adapter logits bytes changed")
    from safetensors.torch import load_file
    import torch
    base = load_file(str(Path(run) / parity["base_logits_path"]))["logits"].float()
    adapter = load_file(str(Path(run) / parity["adapter_logits_path"]))["logits"].float()
    require(tuple(base.shape) == tuple(adapter.shape),
            "Text parity logits shapes differ")
    require(list(base.shape) == parity["shape"], "Text parity shape receipt differs")
    maximum = float((base - adapter).abs().max().item())
    require(maximum <= float(parity["atol"]),
            f"Text parity failed independently: {maximum} > {parity['atol']}")
    require(abs(maximum - float(parity["maximum_absolute_difference"])) <= 1e-6,
            "Text parity maximum-difference receipt differs")
    return {"maximum_absolute_difference": maximum,
            "atol": parity["atol"], "prompt_tokens": len(expected)}


def verify_probes(run, report, tokenizer, protocol, dataset):
    rows = read_jsonl(Path(run) / "probes" / "v0_generations.jsonl")
    require(report.get("probes", {}).get("rows") == rows,
            "V0 report probes differ from the raw trace file")
    probe = protocol["v0"]["probe"]
    select_probe(dataset, protocol["v0"])
    case_id = probe["case_id"]
    wrong_case_id = probe["wrong_image_case_id"]
    expected_prompt = expected_prompt_ids(
        tokenizer, dataset.row(case_id)["question"])
    require([row.get("condition") for row in rows] == ["correct", "wrong", "noimage"],
            "V0 probes lack the exact ordered correct/wrong/noimage inventory")
    by_condition = {row["condition"]: row for row in rows}
    max_new = protocol["v0"]["generation"]["max_new_tokens"]
    for condition, row in by_condition.items():
        forbidden_row_fields(row, f"V0 {condition} trace")
        require(row["prompt_ids"] == expected_prompt,
                "V0 probe prompt differs from the independently tokenized question")
        require(row["prompt_sha256"] == sha256_json(expected_prompt),
                "V0 probe prompt hash differs")
        validate_generation_trace(
            row, Path(run) / row["logits_file"],
            protocol_prompt_ids=expected_prompt, max_new_tokens=max_new)
    expected_images = {"correct": case_id, "wrong": wrong_case_id,
                       "noimage": None}
    for condition, image_case in expected_images.items():
        row = by_condition[condition]
        require(row["image_case_id"] == image_case,
                f"V0 {condition} image source differs from the protocol")
        expected_sha = (None if image_case is None
                        else dataset.asset_for_case(image_case)["sha256"])
        require(row["image_sha256"] == expected_sha,
                f"V0 {condition} image hash differs from the verified asset")
        if image_case is None:
            require(row["visual_cache_key"] is None
                    and row["visual_image_sha256"] is None,
                    "No-image probe carries a visual payload")
        else:
            require(row["visual_cache_key"] is not None
                    and row["visual_image_sha256"] == expected_sha,
                    "V0 visual payload hash differs from the verified image")
    require(by_condition["correct"]["visual_cache_key"]
            != by_condition["wrong"]["visual_cache_key"],
            "Correct/wrong visual cache keys did not differ")
    from safetensors.torch import load_file
    import torch
    logits = {condition: load_file(str(Path(run) / row["logits_file"]))["logits"].float()
              for condition, row in by_condition.items()}
    differences = {
        "correct_vs_wrong": float((logits["correct"][0] - logits["wrong"][0]).abs().max().item()),
        "correct_vs_noimage": float((logits["correct"][0] - logits["noimage"][0]).abs().max().item()),
    }
    require(all(value >= float(protocol["v0"]["distinct_logit_min"])
                for value in differences.values()),
            "V0 correct/wrong/no-image logits are not independently distinct")
    decoded = {condition: decode_ids(tokenizer, row["generated_ids"])
               for condition, row in by_condition.items()}
    return {"conditions": sorted(by_condition), "prompt_tokens": len(expected_prompt),
            "logit_max_differences": differences,
            "pinned_payloads_differ": True,
            "logit_use_passed": True,
            "decoded_diagnostic_only": decoded}


def verify_gradient(report, tokenizer, protocol, dataset):
    gradient = report.get("gradient")
    require(isinstance(gradient, dict), "V0 report lacks gradient evidence")
    backward = protocol["v0"]["backward"]
    prompt = expected_prompt_ids(tokenizer, backward["prompt"])
    response = tokenizer(backward["target_text"],
                         add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
    full = prompt + response
    require(gradient["prompt_ids"] == prompt,
            "Gradient prompt tokens differ from independent tokenization")
    require(gradient["response_ids"] == response,
            "Gradient response tokens differ")
    require(gradient["labels"] == [-100] * len(prompt) + response,
            "Gradient labels are not response-only")
    require(gradient["input_ids_sha256"] == sha256_json(full),
            "Gradient input IDs receipt differs")
    require(gradient["response_only"] is True, "Response-only flag is false")
    require(gradient["text_aligned_logits"] is True,
            "Projector training did not report text-aligned logits")
    require(isinstance(gradient["loss"], (int, float))
            and gradient["loss"] == gradient["loss"]
            and abs(gradient["loss"]) != float("inf"),
            "Projector loss is missing/nonfinite")
    projector = sorted(gradient["projector_parameter_names"])
    require(bool(projector) and all(name.startswith("vision_projector.") for name in projector),
            "Projector parameter inventory is empty or outside the projector")
    require(sorted(gradient["projector_gradients_present"]) == projector,
            "Not every projector tensor reported a gradient")
    require(gradient["projector_gradients_finite"] is True
            and gradient["projector_gradients_nonzero"] is True,
            "Projector gradients were not finite/nonzero")
    require(not gradient["nonprojector_gradient_names"],
            "Non-projector tensors received gradients")
    require(gradient["decoder_input_gradient_finite"] is True
            and gradient["decoder_input_gradient_nonzero"] is True,
            "Decoder-to-projector input gradient was not retained")
    require(gradient["base_parameter_stamp_unchanged"] is True,
            "Backward/update changed the sealed base")
    update = protocol["v0"]["update"]
    if update["enabled"]:
        require(bool(gradient["update_changed"]),
                "Declared V0 update changed no projector tensor")
        require(set(gradient["update_changed"]) <= set(projector),
                "V0 update changed a non-projector tensor")
    else:
        require(not gradient["update_changed"],
                "V0 update changed tensors while disabled")
    require(gradient["visual_payload"]["visual_cache_key"] is not None,
            "Gradient probe lacks its pinned visual payload")
    return {"projector_parameters": projector,
            "loss": gradient["loss"],
            "response_only": gradient["response_only"],
            "projector_gradients": bool(projector
                                         and gradient["projector_gradients_finite"]
                                         and gradient["projector_gradients_nonzero"]),
            "decoder_input_gradient": bool(
                gradient["decoder_input_gradient_finite"]
                and gradient["decoder_input_gradient_nonzero"]),
            "nonprojector_gradients_absent": not gradient["nonprojector_gradient_names"],
            "update_ownership": (
                not update["enabled"]
                or (bool(gradient["update_changed"])
                    and set(gradient["update_changed"]) <= set(projector))),
            "update_changed": gradient["update_changed"]}


def verify_frozen(report, protocol):
    frozen = report.get("frozen")
    require(isinstance(frozen, dict), "V0 report lacks frozen-component evidence")
    require(frozen["base_parameter_stamp_unchanged"] is True,
            "Base parameter stamp changed")
    require(frozen["base_trainable_parameters"] == 0,
            "Base exposes trainable parameters")
    require(frozen["nonprojector_trainable"] == [],
            "Non-projector tensors are trainable")
    require(frozen["sealed_original_identity_matches"] is True,
            "Sealed original identity differs")
    require(frozen["encoder_spec_digest"] == protocol["vision_digest"],
            "Encoder spec digest differs")
    source_manifest = json.loads(resolve_project_path(
        protocol, protocol["vision_source"]["manifest"]["path"]).read_text())
    inventory = source_manifest["tensor_inventory"]
    for key in ("encoder_audit_before", "encoder_audit_after"):
        audit = frozen[key]
        for field in ("bytes", "hits", "misses", "invalidations"):
            require(type(audit.get(field)) is int and audit[field] >= 0,
                    f"Encoder cache audit field is invalid: {key}.{field}")
        encoder = audit.get("encoder")
        require(isinstance(encoder, dict), "Live frozen encoder audit is missing")
        require(encoder["tensor_count"] == inventory["vision_tensor_count"]
                and encoder["parameter_elements"] == inventory["vision_parameter_elements"],
                "Loaded encoder inventory differs from the pinned source")
        require(encoder["trainable_names"] == [] and encoder["gradient_names"] == []
                and encoder["parameter_stamp_unchanged"] is True,
                "Encoder was trainable, received gradients, or changed")
        require(encoder["dtypes"] == ["torch.float32"]
                and bool(encoder["devices"])
                and set(encoder["devices"]) <= {"mps", "mps:0"},
                "Frozen encoder did not run in declared native FP32")
    require(bool(frozen["vision_projector_trainable"]),
            "Projector trainable receipt is empty")
    return {"base_stamp_sha256": frozen["base_parameter_stamp_sha256"],
            "projector_trainable": frozen["vision_projector_trainable"],
            "encoder_spec_digest": frozen["encoder_spec_digest"],
            "encoder": frozen["encoder_audit_after"]["encoder"]}


def verify_extension(run, report, protocol, *, relative="extension"):
    extension = report.get(relative) or report.get("extension")
    require(isinstance(extension, dict), "Report lacks extension evidence")
    expected = extension["tensor_names"]
    receipt = validate_extension_inventory(
        Path(run) / relative, expected_names=expected,
        vision_pins=protocol["vision"])
    require(receipt["weights_sha256"] == extension["weights_sha256"],
            "Extension weight receipt differs")
    require(receipt["manifest_digest"] == extension["manifest_digest"],
            "Extension manifest digest differs")
    require(receipt["format"] == "replica.minicpm5.extensions.v4",
            "Extension is not strict v4")
    require(receipt["base_saved"] is False, "Extension saved base weights")
    require(receipt["manifest"]["vision"] == dict(protocol["vision"]),
            "Extension manifest vision pins differ")
    component_names = {name for names in receipt["components"].values()
                       for name in names}
    require(component_names == set(receipt["tensor_names"]),
            "Extension receipt components differ from tensor inventory")
    require(any(name.startswith("vision_projector.")
                for name in receipt["tensor_names"]),
            "Extension lacks projector tensors")
    require(not any(name.startswith("base.")
                    for name in receipt["tensor_names"]),
            "Extension contains base tensors")
    return receipt


def verify_restart(args, run, report, run_report_sha, tokenizer, protocol,
                   dataset, probes):
    require(protocol["v0"]["require_restart"] is True,
            "Protocol did not require restart but scorer was asked to check it")
    require(args.restart is not None,
            "Fresh-process restart evidence is required for V0 acceptance")
    restart_report, restart_sha = validate_restart_report(
        args.restart, protocol, sha256_file(Path(args.protocol)),
        run_report_sha)
    artifacts = artifact_bindings(args.restart, restart_report, protocol)
    require(restart_report.get("implementation_sha256")
            == protocol["implementation_sha256"],
            "Restart implementation pins differ from the protocol")
    extension = report["extension"]
    require(restart_report["prior_extension_weights_sha256"]
            == extension["weights_sha256"],
            "Restart compared a different extension")
    require(restart_report.get("loaded_extension", {}).get("weights_sha256")
            == extension["weights_sha256"],
            "Restart loaded the wrong extension")
    conditions = {item["condition"]: item for item in restart_report["conditions"]}
    primary = {row["condition"]: row for row in probes["rows"]}
    require(set(conditions) == set(primary),
            "Restart conditions differ from primary probes")
    from safetensors.torch import load_file
    import torch
    tolerance = protocol["v0"]["restart"]
    for condition, item in conditions.items():
        row = primary[condition]
        require(row["generated_ids"] == item["restart_generated_ids"]
                == item["prior_generated_ids"],
                f"Restart tokens differ: {condition}")
        primary_logits = load_file(str(Path(run) / row["logits_file"]))["logits"].float()
        prior_logits = load_file(
            str(Path(args.restart) / "prior_logits" / f"{condition}.safetensors")
        )["logits"].float()
        restart_logits = load_file(
            str(Path(args.restart) / "restart_logits" / f"{condition}.safetensors")
        )["logits"].float()
        require(sha256_file(Path(args.restart) / "prior_logits" /
                            f"{condition}.safetensors")
                == row["logits_sha256"],
                f"Restart prior logits are not the primary logits: {condition}")
        require(bool(torch.allclose(primary_logits, restart_logits,
                                    atol=float(tolerance["atol"]),
                                    rtol=float(tolerance["rtol"]))),
                f"Restart logits differ beyond tolerance: {condition}")
        require(bool(torch.allclose(prior_logits, restart_logits,
                                    atol=float(tolerance["atol"]),
                                    rtol=float(tolerance["rtol"]))),
                f"Restart copied logits differ: {condition}")
    require(restart_report["zero_updates"] is True,
            "Restart performed an optimizer update")
    require(all(restart_report.get("checks", {}).values()),
            "Restart self-checks failed")
    return {"restart_report_sha256": restart_sha,
            "conditions": sorted(conditions),
            "zero_updates": True}


def score_v0(args, protocol, protocol_sha, dataset, tokenizer, run, report,
             run_report_sha):
    require_same_implementation(report, protocol)
    artifacts = artifact_bindings(run, report, protocol)
    require(report.get("data_audit") == dict(dataset.pins),
            "Run data pins differ from independently verified data")
    verify_source_pins(report, protocol)
    checks = {"implementation_match": True, "data_verified": True}
    parity = verify_parity(run, report, tokenizer, protocol)
    probes = verify_probes(run, report, tokenizer, protocol, dataset)
    gradient_file = json.loads(
        (Path(run) / "probes" / "gradient.json").read_text(encoding="utf-8"))
    require(gradient_file == report.get("gradient"),
            "V0 gradient file differs from the report")
    frozen_file = json.loads(
        (Path(run) / "probes" / "frozen.json").read_text(encoding="utf-8"))
    require(frozen_file == report.get("frozen"),
            "V0 frozen file differs from the report")
    gradient = verify_gradient(report, tokenizer, protocol, dataset)
    frozen = verify_frozen(report, protocol)
    extension = verify_extension(run, report, protocol)
    projector_full = set(gradient["projector_parameters"])
    require(projector_full <= set(extension["tensor_names"]),
            "Gradient projector inventory differs from the extension")
    ledger_path = Path(run) / "ledger.sqlite"
    from replica_lab.vision_qualification import ledger_record_count
    ledger_count = ledger_record_count(ledger_path)
    checks.update({
        "text_only_native_parity": parity["maximum_absolute_difference"]
            <= float(protocol["v0"]["parity"]["atol"]),
        "base_frozen": frozen["base_stamp_sha256"]
            == report["frozen"]["base_parameter_stamp_sha256"],
        "encoder_reference_frozen": frozen["encoder_spec_digest"]
            == protocol["vision_digest"],
        "projector_only_trainable": all(
            name.startswith("vision_projector.")
            for name in report.get("trainable_receipt", [])),
        "response_only_labels": gradient["response_only"],
        "projector_gradients": gradient["projector_gradients"],
        "decoder_input_gradient": gradient["decoder_input_gradient"],
        "nonprojector_gradients_absent": gradient["nonprojector_gradients_absent"],
        "projector_update_ownership": gradient["update_ownership"],
        "correct_wrong_noimage_pinned": probes["pinned_payloads_differ"],
        "correct_wrong_noimage_logit_use": probes["logit_use_passed"],
        "bounded_generation": True,
        "strict_extension_export": extension["format"]
            == "replica.minicpm5.extensions.v4",
        "extension_base_absent": not any(
            name.startswith("base.") for name in extension["tensor_names"]),
        "empty_ledger": ledger_count == 0,
        "runner_self_checks": all(bool(value)
                                  for value in report.get("checks", {}).values()),
    })
    checks["bounded_generation"] = all(
        len(row["generated_ids"])
        <= protocol["v0"]["generation"]["max_new_tokens"]
        for row in report["probes"]["rows"])
    restart = None
    if protocol["v0"]["require_restart"]:
        if args.restart is None:
            checks["fresh_process_restart"] = False
        else:
            restart = verify_restart(args, run, report, run_report_sha,
                                     tokenizer, protocol, dataset, probes)
            checks["fresh_process_restart"] = True
    else:
        checks["fresh_process_restart"] = True
    engineering = summarize_engineering(checks)
    score = {
        "format": "replica.vision_qualification_score.v1",
        "stage": "v0",
        "status": "completed",
        "accepted": engineering["engineering_pass"],
        "engineering_pass": engineering["engineering_pass"],
        "engineering_failures": engineering["engineering_failures"],
        "quality_pass": None,
        "quality_claim": "V0 is an interface qualification; no broad-perception claim",
        "blocks_dependents": ([] if engineering["engineering_pass"]
                              else ["V1", "V2", "V3_learned_visual_memory"]),
        "protocol_sha256": protocol_sha,
        "run_report_path": str(Path(run).resolve()),
        "run_report_sha256": run_report_sha,
        "checks": engineering["checks"],
        "parity": parity,
        "probes": {"conditions": probes["conditions"],
                   "logit_max_differences": probes["logit_max_differences"],
                   "decoded_diagnostic_only": probes["decoded_diagnostic_only"]},
        "gradient": gradient,
        "frozen": frozen,
        "extension": {"weights_sha256": extension["weights_sha256"],
                      "manifest_sha256": sha256_file(
                          Path(run) / "extension" / "manifest.json"),
                      "format": extension["format"],
                      "tensor_count": len(extension["tensor_names"])},
        "restart": restart,
        "artifacts": artifacts,
        "scored_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    return score


def verify_training(run, report, dataset, admission, protocol, tokenizer):
    training = report.get("training")
    require(isinstance(training, dict), "V1 report lacks training evidence")
    trace_rows = read_jsonl(Path(run) / "v1" / "training_trace.jsonl")
    require(trace_rows == training["trace"],
            "Training trace file differs from the report")
    require(training["completed_updates"] == protocol["v1"]["train"]["updates"],
            "Declared projector updates did not complete")
    require(training["budget_stop"] is None, "V1 stopped on a training budget")
    require(len(trace_rows) == protocol["v1"]["train"]["updates"],
            "Training trace row count differs from the declared updates")
    schedule = training_schedule(admission["_training_rows"],
                                 protocol["v1"]["train"])
    trainable_file = json.loads(
        (Path(run) / "v1" / "trainable_receipt.json").read_text(encoding="utf-8"))
    route_file = json.loads(
        (Path(run) / "v1" / "optimizer_route.json").read_text(encoding="utf-8"))
    audit_file = json.loads(
        (Path(run) / "v1" / "optimizer_audit.json").read_text(encoding="utf-8"))
    require(trainable_file["trainable"] == training["trainable_receipt"],
            "Trainable receipt file differs from the report")
    require(route_file == training["optimizer_route"],
            "Optimizer route file differs from the report")
    require(audit_file == training["optimizer_audit"],
            "Optimizer audit file differs from the report")
    require(training["trainable_receipt"],
            "V1 trainable receipt is empty")
    require(all(name.startswith("vision_projector.")
                for name in training["trainable_receipt"]),
            "V1 trainable scope is not projector-only")
    route_names = {item["name"] for item in route_file}
    require(route_names == set(training["trainable_receipt"]),
            "Optimizer route differs from the trainable receipt")
    validate_optimizer_route(route_file, audit_file)
    require(training["nonprojector_tensors_unchanged"] is True,
            "V1 changed a non-projector tensor")
    require(training["base_parameter_stamp_unchanged"] is True,
            "V1 changed the sealed base identity")
    require(training["optimizer_audit"]["optimizer"] == "muon8bit",
            "V1 optimizer name differs")
    for index, row in enumerate(trace_rows):
        require(row["update"] == index + 1, "Training trace updates are not sequential")
        _subset = ("case_ids", "losses", "loss", "grad_norm", "response_only_labels",
                   "decoder_autograd", "text_aligned_logits",
                   "projector_gradient_names", "nonprojector_gradient_names",
                   "base_parameter_stamp_unchanged")
        require(all(key in row for key in _subset),
                "Training trace row lacks declared evidence")
        expected_cases = [item["case_id"] for item in schedule[index]]
        require(row["case_ids"] == expected_cases,
                "Training schedule/case order differs from the declared seed")
        require(len(row["losses"]) == len(expected_cases),
                "Training loss count differs from gradient accumulation")
        require(all(isinstance(value, (int, float)) and value == value
                    and abs(value) != float("inf") for value in row["losses"]),
                "Training losses are missing/nonfinite")
        require(row["response_only_labels"] is True,
                "Training labels were not response-only")
        require(row["decoder_autograd"] is True,
                "Decoder autograd was disabled during projector training")
        require(row["text_aligned_logits"] is True,
                "Training logits were not text-aligned")
        require(row["nonprojector_gradient_names"] == [],
                "Non-projector gradients occurred during training")
        require(set(row["projector_gradient_names"]) == route_names,
                "Projector gradient inventory differs from the optimizer route")
        require(row["base_parameter_stamp_unchanged"] is True,
                "A training update changed the sealed base")
        require(isinstance(row["grad_norm"], (int, float))
                and row["grad_norm"] >= 0 and row["grad_norm"] == row["grad_norm"],
                "Training gradient norm is invalid")
    require(trainable_file["nonprojector_gradients"] == [],
            "Trainable receipt reports non-projector gradients")
    return {"trace": trace_rows, "trainable": trainable_file["trainable"],
            "route": route_file, "audit": audit_file,
            "updates": len(trace_rows)}


def verify_v1_extension(run, report, protocol, training):
    checkpoint = protocol["v1"]["initial_checkpoint"]
    initial = resolve_project_path(protocol, checkpoint["path"])
    require(sha256_file(initial / "manifest.json")
            == checkpoint["manifest_sha256"],
            "Initial checkpoint receipt changed")
    require(sha256_file(initial / "extensions.safetensors")
            == checkpoint["weights_sha256"],
            "Initial checkpoint weights changed")
    initial_receipt = load_extension_receipt(initial,
                                             vision_pins=protocol["vision"])
    require(initial_receipt["weights_sha256"] == checkpoint["weights_sha256"],
            "Initial checkpoint receipt differs")
    final = verify_extension(run, {"final_extension": report["final_extension"]},
                             protocol, relative="final_extension")
    lineage = compare_extension_tensors(initial, Path(run) / "final_extension")
    require(lineage == report["extension_lineage"],
            "Extension lineage differs from the independent comparison")
    require(lineage["changed"], "Training changed no projector tensor")
    require(all(name.startswith("vision_projector.")
                for name in lineage["changed"]),
            "Non-projector extension tensors changed")
    require(bool(set(training["trainable"]) & set(lineage["changed"])),
            "No declared trainable projector tensor changed")
    return {"initial": initial_receipt, "final": final, "lineage": lineage}


def verify_v1_evaluation(run, report, protocol, dataset, admission, tokenizer):
    evaluation = report.get("evaluation")
    require(isinstance(evaluation, dict), "V1 report lacks evaluation evidence")
    rows = read_jsonl(Path(run) / "v1" / "eval_generations.jsonl")
    require(sha256_file(Path(run) / "v1" / "eval_generations.jsonl")
            == evaluation["eval_generations_sha256"],
            "Evaluation generations changed")
    cases = select_eval_cases(dataset, protocol["v1"]["eval"])
    case_ids = [row["case_id"] for row in cases]
    require(evaluation["case_ids"] == case_ids,
            "Reported evaluation cases differ from the declared selection")
    require(evaluation["cases"] == len(case_ids),
            "Reported evaluation case count differs")
    assignment = assign_controls(dataset, case_ids)
    require(evaluation["assignment"] == assignment,
            "Control assignment differs from deterministic reconstruction")
    controls = protocol["v1"]["eval"]["required_controls"]
    require(set(evaluation["controls"]) == set(controls),
            "Evaluation control inventory differs")
    contexts = {(row["case_id"], row["kind"]): row
                for row in admission.get("_text_contexts", ())}
    template = protocol["v1"]["controls"]["text_context_template"]
    maximum = protocol["limits"]["max_prompt_tokens"]
    max_new = protocol["v1"]["eval"]["max_new_tokens"]
    scored = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        forbidden_row_fields(row, "V1 evaluation row")
        control, case_id = row["control"], row["case_id"]
        require(control in controls, f"Undeclared evaluation control: {control}")
        require(case_id in case_ids, f"Undeclared evaluation case: {case_id}")
        require((control, case_id) not in seen,
                f"Duplicate evaluation row: {control}/{case_id}")
        seen.add((control, case_id))
        case = dataset.case(case_id)
        require((row["family"], row["group_id"], row["split"])
                == (case["group_id"], case["group_id"], case["split"]),
                f"Evaluation row family/split differs: {control}/{case_id}")
        if control in ("captiononly", "ocrtextonly"):
            kind = TEXT_CONTROL_KIND[control]
            context_row = contexts[(case_id, kind)]
            text = build_text_context_prompt(
                dataset.row(case_id)["question"], context_row["text"], template)
            expected_image = None
            require(row["text_context_kind"] == kind,
                    "Text-control kind differs")
            require(row["text_sha256"] == context_row["text_sha256"],
                    "Text-control context hash differs")
            require(row["image_sha256"] is None and row["image_case_id"] is None,
                    "Text-only control carries an image")
        else:
            image_case = (case_id if control == "correct"
                          else assignment[control][case_id])
            if control == "noimage":
                image_case = None
            text = dataset.row(case_id)["question"]
            expected_image = (None if image_case is None
                              else dataset.asset_for_case(image_case)["sha256"])
            require(row["image_case_id"] == image_case,
                    f"Evaluation image source differs: {control}/{case_id}")
            require(row["image_sha256"] == expected_image,
                    f"Evaluation image hash differs: {control}/{case_id}")
            require(row["text_context_kind"] is None,
                    "Visual control carries a text context")
        expected = expected_prompt_ids(tokenizer, text)
        require(row["prompt_ids"] == expected,
                f"Evaluation prompt differs from independent tokenization: {control}/{case_id}")
        require(row["prompt_sha256"] == sha256_json(expected),
                "Evaluation prompt hash differs")
        require(len(expected) <= maximum,
                "Evaluation prompt exceeds the declared budget")
        require(1 <= len(row["generated_ids"]) <= max_new,
                "Evaluation generation exceeds the declared budget")
        scored.append({"control": control, "case_id": case_id,
                       "generated_text": decode_ids(tokenizer, row["generated_ids"])})
    expected_pairs = {(control, case_id) for control in controls
                      for case_id in case_ids}
    require(seen == expected_pairs,
            "Evaluation rows do not cover every control/case")
    require(evaluation["complete"] is True, "Evaluation did not complete")
    scores = score_controls(dataset, scored, controls=controls)
    retention = verify_v1_retention(run, report, protocol, admission, tokenizer)
    quality = apply_quality_gates(
        scores, retention, protocol["v1"]["quality"]["gates"],
        retention_baseline_accuracy=protocol["v1"]["quality"][
            "retention_baseline_accuracy"])
    return {"scores": scores, "retention": retention, "quality": quality,
            "eval_generations_sha256": evaluation["eval_generations_sha256"],
            "case_ids": case_ids, "controls": controls}


def verify_v1_retention(run, report, protocol, admission, tokenizer):
    retention = report.get("retention")
    require(isinstance(retention, dict), "V1 report lacks retention evidence")
    rows = read_jsonl(Path(run) / "v1" / "retention_generations.jsonl")
    require(sha256_file(Path(run) / "v1" / "retention_generations.jsonl")
            == retention["retention_generations_sha256"],
            "Retention generations changed")
    prompts = admission["_retention_prompts"][
        :protocol["v1"]["retention"]["max_cases"]]
    require(len(rows) == len(prompts), "Retention generation count differs")
    maximum = protocol["limits"]["max_prompt_tokens"]
    max_new = protocol["v1"]["eval"]["max_new_tokens"]
    scored = []
    for row, prompt in zip(rows, prompts):
        require(row["case_id"] == prompt["case_id"],
                "Retention generation order/case differs")
        expected = expected_prompt_ids(tokenizer, prompt["prompt"])
        require(row["prompt_ids"] == expected,
                "Retention prompt differs from independent tokenization")
        require(row["prompt_sha256"] == sha256_json(expected),
                "Retention prompt hash differs")
        require(len(expected) <= maximum, "Retention prompt exceeds budget")
        require(1 <= len(row["generated_ids"]) <= max_new,
                "Retention generation exceeds budget")
        scored.append({"case_id": row["case_id"],
                       "generated_text": decode_ids(tokenizer, row["generated_ids"])})
    require(retention["complete"] is True, "Retention evaluation did not complete")
    return score_retention(admission["_retention_prompts"],
                           admission["_retention_labels"], scored)


def score_v1(args, protocol, protocol_sha, dataset, tokenizer, run, report,
             run_report_sha):
    require_same_implementation(report, protocol)
    artifacts = artifact_bindings(run, report, protocol)
    require(report.get("data_audit") == dict(dataset.pins),
            "V1 data pins differ from independently verified data")
    verify_source_pins(report, protocol)
    v0_report, v0_digest = verify_v0_prerequisite(protocol)
    require(report.get("v0_score_report_sha256")
            == protocol["v1"]["prerequisites"]["v0_score_report"]["sha256"],
            "V1 run scored a different V0 report")
    train_splits = sorted({dataset.case(row["case_id"])["split"]
                           for row in dataset.model_rows
                           if dataset.case(row["case_id"])["split"] == "train"})
    require(bool(train_splits), "No training split is present")
    admission = verify_admission(
        protocol, dataset, train_splits=train_splits,
        eval_splits=protocol["v1"]["eval"]["splits"])
    training = verify_training(run, report, dataset, admission, protocol, tokenizer)
    lineage = verify_v1_extension(run, report, protocol, training)
    evaluation = verify_v1_evaluation(run, report, protocol, dataset,
                                      admission, tokenizer)
    costs = report.get("costs", {})
    checks = {
        "v0_prerequisite_accepted": v0_report["accepted"] is True,
        "v0_report_hash": report.get("v0_score_report_sha256") == v0_digest,
        "named_admission_verified": True,
        "projector_only_trainable": True,
        "nonprojector_tensors_frozen": True,
        "zero_nonprojector_gradients": True,
        "response_only_labels": all(row["response_only_labels"]
                                    for row in training["trace"]),
        "decoder_autograd_retained": all(row["decoder_autograd"]
                                         for row in training["trace"]),
        "declared_updates_completed": (
            training["updates"] == protocol["v1"]["train"]["updates"]),
        "budget_stop_absent": report["training"]["budget_stop"] is None,
        "optimizer_audit": report["training"]["optimizer_audit"].get(
            "optimizer") == "muon8bit",
        "strict_extension_export": lineage["final"]["format"]
            == "replica.minicpm5.extensions.v4",
        "only_projector_weights_changed": all(
            name.startswith("vision_projector.")
            for name in lineage["lineage"]["changed"]),
        "evaluation_complete": evaluation["scores"]["controls"].get(
            "correct", {}).get("available") is True,
        "evaluation_controls_complete": True,
        "retention_complete": report["retention"]["complete"] is True,
        "runner_self_checks": all(bool(value)
                                  for value in report.get("checks", {}).values()),
        "budget_counts": (
            costs.get("optimizer_updates", -1)
            <= protocol["limits"]["max_updates"]
            and costs.get("evaluation_generated_tokens", -1)
            <= (protocol["v1"]["eval"]["max_cases"]
                * protocol["v1"]["eval"]["max_new_tokens"])
            * len(protocol["v1"]["eval"]["required_controls"])),
    }
    engineering = summarize_engineering(checks)
    quality = evaluation["quality"]
    accepted = engineering["engineering_pass"] and quality["quality_pass"]
    score = {
        "format": "replica.vision_qualification_score.v1",
        "stage": "v1",
        "status": "completed",
        "accepted": accepted,
        "engineering_pass": engineering["engineering_pass"],
        "engineering_failures": engineering["engineering_failures"],
        "quality_pass": quality["quality_pass"],
        "quality_failures": quality["failures"],
        "quality_values": quality["values"],
        "quality_gates": quality["gates"],
        "quality_claim": ("Bounded procedural/projector alignment pilot only; "
                          "not broad perception or deployment evidence"),
        "protocol_sha256": protocol_sha,
        "run_report_path": str(Path(run).resolve()),
        "run_report_sha256": run_report_sha,
        "v0_score_report_sha256": v0_digest,
        "admission_sha256": protocol["v1"]["prerequisites"]["data_admission"]["sha256"],
        "checks": engineering["checks"],
        "training": {"updates": training["updates"],
                     "trainable": training["trainable"],
                     "optimizer_route": training["route"],
                     "optimizer_audit": training["audit"]},
        "metrics": {
            "controls": evaluation["scores"]["controls"],
            "paired": evaluation["scores"]["paired"],
            "retention": {key: value for key, value in evaluation["retention"].items()
                          if key != "rows"},
            "evaluation_cases": evaluation["case_ids"],
            "controls_evaluated": evaluation["controls"],
        },
        "extension_lineage": lineage["lineage"],
        "extension": {"weights_sha256": lineage["final"]["weights_sha256"],
                      "manifest_sha256": sha256_file(
                          Path(run) / "final_extension" / "manifest.json"),
                      "format": lineage["final"]["format"],
                      "tensor_count": len(lineage["final"]["tensor_names"])},
        "artifacts": artifacts,
        "blocks_dependents": ([] if accepted else ["V2", "V3_learned_visual_memory"]),
        "scored_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    return score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-lock", type=Path, required=True)
    parser.add_argument("--restart", "--restart-run", dest="restart", type=Path,
                        help="Fresh-process V0 restart directory scored with its primary")
    args = parser.parse_args()
    ensure_fresh_directory(args.output)
    protocol_sha = sha256_file(args.protocol)
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    validate_protocol(protocol)
    require(args.run.is_dir(), "Run directory is missing")
    verify_implementation_pins(protocol, ROOT)
    dataset = load_verified_dataset(protocol)
    report, run_report_sha = require_completed_run(args.run, protocol,
                                                   protocol_sha)
    tokenizer = load_tokenizer(args, protocol)
    try:
        if protocol["stage"] == "v0":
            score = score_v0(args, protocol, protocol_sha, dataset, tokenizer,
                             args.run, report, run_report_sha)
        else:
            score = score_v1(args, protocol, protocol_sha, dataset, tokenizer,
                             args.run, report, run_report_sha)
        score["accepted"] = bool(score["accepted"])
        dump(args.output / "score_report.json", score)
        print(json.dumps({"accepted": score["accepted"],
                          "stage": score["stage"],
                          "engineering_pass": score["engineering_pass"],
                          "quality_pass": score.get("quality_pass"),
                          "report": str(args.output / "score_report.json")}))
    except BaseException as error:
        failure = {
            "format": "replica.vision_qualification_score_failure.v1",
            "status": "failed",
            "stage": protocol["stage"],
            "protocol_sha256": protocol_sha,
            "run_report_path": str(Path(args.run).resolve()),
            "run_report_sha256": run_report_sha,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "scored_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        dump(args.output / "score_failure.json", failure)
        raise


if __name__ == "__main__":
    main()

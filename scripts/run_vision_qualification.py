#!/usr/bin/env python3
"""Run the preregistered V0 interface or gated V1 projector qualification.

The parent launches this from the guarded native entry point with the vision
overlay interpreter.  This script never downloads, never selects an answer for
the model, never imports the historical runtime pilot, and never runs native
work unless the protocol/admission gates pass.  V0 uses the empty-memory visual
prefix with all other injections disabled; V1 additionally requires an accepted
independent V0 score and named data-admission hashes.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
    os.environ[key] = "1"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from replica_lab.vision_qualification import (  # noqa: E402
    QualificationError, assign_controls, build_text_context_prompt,
    compare_extension_tensors, compare_restart_traces, ensure_fresh_directory,
    implementation_hashes, leakage_hits, ledger_record_count,
    load_verified_dataset, optimizer_route, projector_optimizer_groups,
    require, resolve_project_path, rows_sha256, select_eval_cases, select_probe,
    sha256_bytes, sha256_file, sha256_json, sha256_text, training_schedule,
    validate_budget, validate_extension_inventory, validate_generation_trace,
    validate_protocol, verify_admission, verify_implementation_pins,
    verify_v0_prerequisite, visual_question,
)


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_jsonl(path, rows):
    Path(path).write_text("".join(json.dumps(row, allow_nan=False) + "\n"
                                  for row in rows))


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def artifact_hashes(output, paths):
    result = {}
    for relative in paths:
        path = Path(output) / relative
        require(path.is_file(), f"Required artifact is missing: {relative}")
        result[relative] = sha256_file(path)
    return result


def protocol_scope(protocol):
    from replica_lab.memory_system.ledger import EligibilityScope
    raw = protocol["vision_source"]["scope"]
    return EligibilityScope(raw["subject_id"], raw["branch_id"], raw["purpose"],
                            raw["as_of"], raw["mode"], raw.get("snapshot_at"))


def task_graph(task_id, text):
    from replica_lab.architecture.contracts import TaskGraph
    return TaskGraph(task_id=task_id, goal=text)


def prompt_ids(tokenizer, text, device, maximum):
    from replica_lab.architecture.loading import encode_prompt
    ids = encode_prompt(tokenizer, text, device)
    require(ids.shape[1] <= maximum,
            f"Prompt exceeds the declared token budget: {ids.shape[1]} > {maximum}")
    return ids


def empty_ledger(output):
    from replica_lab.memory_system.ledger import RecordLedger
    return RecordLedger(Path(output) / "ledger.sqlite")


def make_session(adapter, tokenizer, ledger):
    from replica_lab.memory_system.session import MemorySession
    return MemorySession(adapter, tokenizer, ledger)


def feature_service(protocol, device):
    from replica_lab.architecture.vision import VisionFeatureService
    source = resolve_project_path(protocol, protocol["vision_source"]["path"])
    manifest = resolve_project_path(protocol,
                                    protocol["vision_source"]["manifest"]["path"])
    limit = int(protocol["limits"].get("max_cache_bytes", 8 * 1024 * 1024))
    return VisionFeatureService(source, manifest, device=device,
                                max_cache_bytes=limit,
                                projector_hidden=protocol["vision"]["projector_hidden"])


def verify_source_and_spec(protocol, device):
    from replica_lab.architecture.contracts import VisionSpec
    from replica_lab.architecture.vision import load_vision_source, vision_spec_from_source
    source_path = resolve_project_path(protocol, protocol["vision_source"]["path"])
    manifest_path = resolve_project_path(
        protocol, protocol["vision_source"]["manifest"]["path"])
    require(source_path.exists(), "Pinned vision source checkpoint is missing")
    require(manifest_path.is_file(), "Pinned vision source manifest is missing")
    require(sha256_file(manifest_path)
            == protocol["vision_source"]["manifest"]["sha256"],
            "Vision source manifest bytes changed")
    source = load_vision_source(source_path)
    spec = vision_spec_from_source(source,
                                   projector_hidden=protocol["vision"]["projector_hidden"])
    spec.validate()
    fields = ("encoder_repo", "encoder_revision", "encoder_source_digest",
              "processor_digest", "encoder_width", "image_size", "patch_size",
              "max_visual_tokens", "projector_hidden", "feature_version",
              "compute_dtype")
    declared = VisionSpec(**{key: protocol["vision"][key] for key in fields})
    declared.validate()
    require(asdict(spec) == asdict(declared),
            "VisionFeatureService spec differs from the declared protocol")
    require(spec.digest == protocol["vision_digest"],
            "Vision spec digest differs from the protocol")
    return source, spec, manifest_path


def verify_encoder_frozen(service):
    """Inspect both the live frozen encoder and the feature-cache counters."""
    spec = service.spec
    spec.validate()
    audit = service.cache_audit()
    require(isinstance(audit, dict), "Vision service cache audit is unavailable")
    for key in ("bytes", "hits", "misses", "invalidations"):
        require(type(audit.get(key)) is int and audit[key] >= 0,
                f"Vision cache audit field is invalid: {key}")
    audit["encoder"] = service.frozen_audit()
    return audit


def observe(service, dataset, *, image_case_id, task_id, source_id, scope,
            ledger_epoch, spec):
    asset = dataset.asset_for_case(image_case_id)
    path = dataset.image_path(image_case_id)
    visual = service.observe(str(path), source_id=source_id, task_id=task_id,
                             scope=scope, observed_at=0,
                             source_epoch=ledger_epoch)
    require(visual.producer == spec.digest,
            "Observed visual producer differs from the pinned spec")
    require(visual.image_sha256 == asset["sha256"],
            "Observed visual image hash differs from the verified asset")
    require(visual.task_id == task_id and visual.subject_id == scope.subject_id
            and visual.branch_id == scope.branch_id
            and visual.purpose == scope.purpose,
            "Observed visual ownership differs from the request scope")
    require(visual.features.requires_grad is False,
            "Inference-boundary visual features must remain detached")
    require(bool(visual.features.isfinite().all()),
            "Observed visual features are nonfinite")
    require(tuple(visual.features.shape) == (1, spec.max_visual_tokens,
                                             spec.encoder_width),
            "Observed visual tensor shape differs from the pinned layout")
    require(tuple(visual.grid) == (spec.image_size // spec.patch_size,
                                   spec.image_size // spec.patch_size),
            "Observed visual grid differs from the pinned layout")
    visual.assert_immutable()
    return visual


def visual_payload(visual):
    if visual is None:
        return {"visual_cache_key": None, "visual_producer": None,
                "visual_encoder_producer": None, "visual_image_sha256": None,
                "visual_grid": None, "features_sha256": None}
    from safetensors.torch import save_file
    digest = hashlib.sha256()
    digest.update(str(tuple(visual.features.shape)).encode())
    digest.update(str(visual.features.dtype).encode())
    digest.update(visual.features.detach().cpu().contiguous().numpy().tobytes())
    return {"visual_cache_key": visual.cache_key,
            "visual_producer": visual.producer,
            "visual_encoder_producer": visual.encoder_producer,
            "visual_image_sha256": visual.image_sha256,
            "visual_grid": list(visual.grid),
            "features_sha256": digest.hexdigest()}


def save_logits(path, logits):
    from safetensors.torch import save_file
    save_file({"logits": logits.detach().cpu().contiguous()}, str(path))
    return sha256_file(path)


def build_request(session, tokenizer, *, text, visual, task_id, scope, device,
                  maximum):
    import torch
    ids = prompt_ids(tokenizer, text, device, maximum)
    mask = torch.ones_like(ids, dtype=torch.bool)
    request = session.prepare(prompt_ids=ids, prompt_mask=mask, query=text,
                              scope=scope, task=task_graph(task_id, text),
                              training=False, visual=visual, enabled_ports=(),
                              core_impl="none", enabled_branches=("memory",))
    return ids, mask, request


def generate_capture(session, request, *, max_new_tokens, logits_path,
                     eos_token_ids):
    """Consume a request, saving the greedy tokens and teacher-forced logits."""
    import torch
    try:
        prompt_len = int(request.frame.sense.prompt_ids.shape[1])
        ids = session.generate(request, max_new_tokens=max_new_tokens)
        generated = [int(value) for value in ids[0, prompt_len:].tolist()]
        require(bool(generated), "Generation returned no tokens")
        require(len(generated) <= max_new_tokens,
                "Generation exceeded its declared budget")
        mask = torch.ones_like(ids, dtype=torch.bool)
        with torch.no_grad():
            full = session.forward(request, ids, attention_mask=mask).logits
        require(full.shape[0] == 1 and full.shape[1] == ids.shape[1],
                "Text-aligned logits do not match the text token count")
        step_logits = full[0, prompt_len - 1:prompt_len - 1 + len(generated)]
        require(step_logits.shape[0] == len(generated),
                "Teacher-forced logits do not cover the generated tokens")
        require(bool(torch.isfinite(step_logits).all()),
                "Generated logits are nonfinite")
        require([int(index) for index in step_logits.argmax(dim=-1).tolist()] == generated,
                "Production generation disagrees with independently recomputed logits")
        digest = save_logits(logits_path, step_logits)
        return {"generated_ids": generated,
                "ended_by_eos": generated[-1] in eos_token_ids,
                "logits_sha256": digest, "logits_path": logits_path}
    finally:
        request.release()


def text_only_parity(session, adapter, tokenizer, protocol, scope, output):
    import torch
    maximum = protocol["limits"]["max_prompt_tokens"]
    text = protocol["v0"]["parity"]["prompt"]
    ids = prompt_ids(tokenizer, text, "mps", maximum)
    mask = torch.ones_like(ids, dtype=torch.bool)
    with torch.no_grad():
        request = session.prepare(prompt_ids=ids, prompt_mask=mask, query=text,
                                  scope=scope, task=task_graph("v0-parity", text),
                                  training=False, visual=None, enabled_ports=(),
                                  core_impl="none", enabled_branches=("memory",))
        try:
            require(request.frame.sense.visual is None,
                    "Text-only parity unexpectedly carries a visual input")
            adapter_logits = session.forward(request, ids,
                                             attention_mask=mask).logits.detach()
            base_logits = adapter.base(input_ids=ids, attention_mask=mask,
                                       use_cache=False).logits.detach()
        finally:
            request.release()
    base_path = Path(output) / "probes" / "parity_base.safetensors"
    adapter_path = Path(output) / "probes" / "parity_adapter.safetensors"
    base_sha = save_logits(base_path, base_logits)
    adapter_sha = save_logits(adapter_path, adapter_logits)
    require(tuple(base_logits.shape) == tuple(adapter_logits.shape),
            "Text-only parity shapes differ")
    maximum_difference = float((base_logits.float()
                                - adapter_logits.float()).abs().max().item())
    atol = float(protocol["v0"]["parity"]["atol"])
    passed = maximum_difference <= atol
    require(passed, f"Text-only native parity failed: {maximum_difference} > {atol}")
    return {"prompt": text, "prompt_ids": [int(value) for value in ids[0].tolist()],
            "prompt_sha256": sha256_json([int(value) for value in ids[0].tolist()]),
            "base_logits_path": "probes/parity_base.safetensors",
            "adapter_logits_path": "probes/parity_adapter.safetensors",
            "base_logits_sha256": base_sha, "adapter_logits_sha256": adapter_sha,
            "shape": list(base_logits.shape),
            "maximum_absolute_difference": maximum_difference,
            "atol": atol, "passed": passed}


def projector_gradient_probe(adapter, session, tokenizer, protocol, service,
                             dataset, scope, ledger_epoch, output):
    import torch
    probe = protocol["v0"]["probe"]
    backward = protocol["v0"]["backward"]
    case_id = probe["case_id"]
    visual = observe(service, dataset, image_case_id=case_id,
                     task_id="v0-backward",
                     source_id=protocol["vision_source"]["source_id"],
                     scope=scope, ledger_epoch=ledger_epoch,
                     spec=service.spec)
    maximum = protocol["limits"]["max_prompt_tokens"]
    device = adapter.base.get_input_embeddings().weight.device
    prompt = prompt_ids(tokenizer, backward["prompt"], device, maximum)
    response = tokenizer(backward["target_text"], add_special_tokens=False)["input_ids"]
    require(bool(response), "Declared backward target tokenizes to nothing")
    eos = tokenizer.eos_token_id
    require(eos is not None, "Tokenizer requires an explicit EOS token")
    response = list(response) + [int(eos)]
    require(len(response) <= backward["max_response_tokens"],
            "Declared backward response exceeds its token budget")
    input_ids = torch.cat((prompt, torch.tensor([response], dtype=torch.long,
                                                device=prompt.device)), dim=1)
    require(input_ids.shape[1] <= maximum,
            "Backward example exceeds the declared sequence budget")
    labels = torch.tensor([[-100] * prompt.shape[1] + response],
                          dtype=torch.long, device=prompt.device)
    require(not bool((labels[:, :prompt.shape[1]] != -100).any()),
            "Prompt labels must be masked")

    captured = {}
    def capture_gradient(gradient):
        captured["projector_output_gradient"] = gradient.detach()
    def hook(module, inputs, result):
        if torch.is_tensor(result) and result.requires_grad:
            result.register_hook(capture_gradient)
    handle = adapter.vision_projector.register_forward_hook(hook)
    adapter.train()
    projector_names = {"vision_projector." + name: parameter for name, parameter
                       in adapter.vision_projector.named_parameters()}
    base_stamp_before = adapter.parameter_stamp(original_only=True)
    update = protocol["v0"]["update"]
    before = {name: parameter.detach().cpu().clone()
              for name, parameter in projector_names.items()}
    nonprojector_before = {name: parameter.detach().cpu().clone()
                           for name, parameter in adapter.named_parameters()
                           if not name.startswith(("vision_projector.", "base."))}
    changed, unchanged = [], []
    try:
        with session.training_request(
                prompt_ids=prompt, prompt_mask=torch.ones_like(prompt, dtype=torch.bool),
                query=backward["prompt"], scope=scope,
                task=task_graph("v0-backward", backward["prompt"]),
                visual=visual, enabled_ports=(),
                core_impl="none", enabled_branches=("memory",)) as request:
            require(request.frame.sense.visual is visual,
                    "Training request lost its pinned visual payload")
            bad = labels.clone()
            bad[0, 0] = input_ids[0, 0]
            rejected = False
            try:
                session.forward(request, input_ids, attention_mask=torch.ones_like(
                    input_ids, dtype=torch.bool), labels=bad)
            except ValueError:
                rejected = True
            require(rejected, "Prompt-position supervision was not rejected")
            output_value = session.forward(
                request, input_ids,
                attention_mask=torch.ones_like(input_ids, dtype=torch.bool),
                labels=labels)
            require(output_value.logits.shape[1] == input_ids.shape[1],
                    "Vision logits must remain text-aligned")
            loss = output_value.loss
            require(loss is not None and bool(torch.isfinite(loss)),
                    "Projector training loss is unavailable or nonfinite")
            require(loss.requires_grad, "Projector training loss requires_grad is false")
            session.backward(request, loss)
            projector_grads = {name: parameter.grad
                               for name, parameter in projector_names.items()}
            other_grads = {name: parameter.grad for name, parameter
                           in adapter.named_parameters()
                           if not name.startswith("vision_projector.")}
            input_gradient = captured.get("projector_output_gradient")
            require(input_gradient is not None
                    and bool(torch.isfinite(input_gradient).all())
                    and bool(input_gradient.ne(0).any()),
                    "Decoder-to-projector input gradient is missing/nonfinite/zero")
            require(all(gradient is not None for gradient in projector_grads.values()),
                    "A projector parameter received no gradient")
            require(all(bool(torch.isfinite(gradient).all())
                        for gradient in projector_grads.values()),
                    "Projector gradients are nonfinite")
            require(any(bool(gradient.ne(0).any())
                        for gradient in projector_grads.values()),
                    "All projector gradients are zero")
            require(not any(gradient is not None for gradient in other_grads.values()),
                    "Non-projector tensors received gradients")
            response_only = (not bool((labels[:, :prompt.shape[1]] != -100).any())
                             and bool((labels[:, prompt.shape[1]:] != -100).all()))
    finally:
        handle.remove()
    # The training lease verifies the unchanged frame at exit; perform the
    # declared one-step ownership update only after that lease has closed.
    if update["enabled"]:
        optimizer = torch.optim.SGD(projector_names.values(), lr=float(update["lr"]))
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        for name, parameter in projector_names.items():
            (changed if not bool(torch.equal(before[name],
                                             parameter.detach().cpu()))
             else unchanged).append(name)
        require(bool(changed), "Declared V0 update changed no projector tensor")
        require(not any(not torch.equal(nonprojector_before[name],
                                        parameter.detach().cpu())
                        for name, parameter in adapter.named_parameters()
                        if not name.startswith(("vision_projector.", "base."))),
                "V0 update changed a non-projector tensor")
    else:
        for name, parameter in projector_names.items():
            (unchanged if torch.equal(before[name], parameter.detach().cpu())
             else changed).append(name)
    base_stamp_after = adapter.parameter_stamp(original_only=True)
    require(base_stamp_before == base_stamp_after,
            "V0 backward/update changed the sealed native parameters")
    adapter.eval()
    return {
        "prompt_ids": [int(value) for value in prompt[0].tolist()],
        "response_ids": response,
        "input_ids_sha256": sha256_json([int(value) for value in input_ids[0].tolist()]),
        "labels": [int(value) for value in labels[0].tolist()],
        "response_only": response_only,
        "loss": float(loss.detach().cpu()),
        "text_aligned_logits": True,
        "projector_parameter_names": sorted(projector_names),
        "projector_gradients_present": sorted(projector_grads),
        "projector_gradients_finite": True,
        "projector_gradients_nonzero": any(
            bool(gradient.ne(0).any()) for gradient in projector_grads.values()),
        "nonprojector_gradient_names": sorted(
            name for name, gradient in other_grads.items() if gradient is not None),
        "decoder_input_gradient_finite": True,
        "decoder_input_gradient_nonzero": True,
        "update_enabled": update["enabled"],
        "update_changed": sorted(changed),
        "update_unchanged": sorted(unchanged),
        "base_parameter_stamp_sha256": sha256_json(
            repr(base_stamp_before)),
        "base_parameter_stamp_unchanged": True,
        "visual_payload": visual_payload(visual),
    }


def freeze_for_v0(adapter):
    trainable = []
    for name, parameter in adapter.named_parameters():
        parameter.requires_grad_(name.startswith("vision_projector."))
        if parameter.requires_grad:
            trainable.append(name)
    require(trainable and all(name.startswith("vision_projector.")
                              for name in trainable),
            "V0 trainable scope must be projector-only")
    require(not any(parameter.requires_grad for name, parameter
                    in adapter.named_parameters()
                    if not name.startswith("vision_projector.")),
            "V0 froze an unexpected tensor as trainable")
    return trainable


def evaluate_probes(session, tokenizer, adapter, protocol, service, dataset,
                    scope, ledger_epoch, output):
    import torch
    probe = protocol["v0"]["probe"]
    select_probe(dataset, protocol["v0"])
    case_id = probe["case_id"]
    wrong_case_id = probe["wrong_image_case_id"]
    row = dataset.row(case_id)
    question = row["question"]
    require(not leakage_hits(question, dataset.label(case_id)),
            "Probe question leaks its verifier answer")
    maximum = protocol["limits"]["max_prompt_tokens"]
    generation_limit = protocol["v0"]["generation"]["max_new_tokens"]
    eos = {tokenizer.eos_token_id}
    traces = []
    logits_dir = Path(output) / "probes" / "logits"
    logits_dir.mkdir(parents=True, exist_ok=True)
    for condition, image_case, task_id in (
            ("correct", case_id, case_id),
            ("wrong", wrong_case_id, case_id),
            ("noimage", None, case_id)):
        visual = (None if image_case is None else observe(
            service, dataset, image_case_id=image_case, task_id=task_id,
            source_id=protocol["vision_source"]["source_id"], scope=scope,
            ledger_epoch=ledger_epoch, spec=service.spec))
        ids, mask, request = build_request(
            session, tokenizer, text=question, visual=visual, task_id=task_id,
            scope=scope, device="mps", maximum=maximum)
        logits_path = logits_dir / f"{condition}.safetensors"
        result = generate_capture(session, request, max_new_tokens=generation_limit,
                                  logits_path=logits_path, eos_token_ids=eos)
        trace = {
            "stage": "v0", "condition": condition, "case_id": case_id,
            "task_id": task_id, "prompt_ids": [int(value) for value in ids[0].tolist()],
            "prompt_sha256": sha256_json([int(value) for value in ids[0].tolist()]),
            "generated_ids": result["generated_ids"],
            "ended_by_eos": result["ended_by_eos"],
            "prompt_tokens": int(ids.shape[1]),
            "generated_tokens": len(result["generated_ids"]),
            "max_new_tokens": generation_limit,
            "image_case_id": image_case,
            "image_sha256": (None if image_case is None
                             else dataset.asset_for_case(image_case)["sha256"]),
            "logits_file": str(Path("probes") / "logits" / f"{condition}.safetensors"),
            "logits_sha256": result["logits_sha256"],
            "question_sha256": sha256_text(question),
        }
        trace.update(visual_payload(visual))
        validate_generation_trace(
            trace, Path(output) / trace["logits_file"],
            protocol_prompt_ids=trace["prompt_ids"],
            max_new_tokens=generation_limit)
        traces.append(trace)
        write_jsonl(Path(output) / "probes" / "v0_generations.jsonl", traces)
    by_condition = {trace["condition"]: trace for trace in traces}
    require(by_condition["correct"]["visual_cache_key"]
            != by_condition["wrong"]["visual_cache_key"],
            "Correct and wrong visual payloads share a cache key")
    require(by_condition["noimage"]["visual_cache_key"] is None,
            "No-image control unexpectedly carries a visual payload")
    logits = {}
    for condition, trace in by_condition.items():
        from safetensors.torch import load_file
        logits[condition] = load_file(str(Path(output) / trace["logits_file"]))["logits"]
    distinct = protocol["v0"]["distinct_logit_min"]
    differences = {
        "correct_vs_wrong": float((logits["correct"][0].float()
                                   - logits["wrong"][0].float()).abs().max().item()),
        "correct_vs_noimage": float((logits["correct"][0].float()
                                     - logits["noimage"][0].float()).abs().max().item()),
    }
    require(all(value >= distinct for value in differences.values()),
            "Pinned correct/wrong/no-image logits are not observably distinct")
    return {"rows": traces, "logit_max_differences": differences,
            "distinct_logit_min": distinct,
            "pinned_payloads_differ": True}


def write_extension(adapter, path, protocol, *, label):
    receipt = adapter.save_extension(path)
    exported_bytes = sum((Path(path) / name).stat().st_size
                         for name in ("manifest.json", "extensions.safetensors"))
    require(exported_bytes <= protocol["limits"]["max_export_bytes"],
            "Extension exceeds the declared export byte ceiling")
    expected = [name for name, _ in adapter.new_named_parameters()]
    validated = validate_extension_inventory(
        path, expected_names=expected, vision_pins=protocol["vision"])
    receipt_path = Path(path) / "manifest.json"
    result = {"label": label, "path": str(Path(path).relative_to(ROOT))
              if Path(path).is_relative_to(ROOT) else str(path),
              "receipt_sha256": sha256_file(receipt_path),
              "weights_sha256": validated["weights_sha256"],
              "manifest_digest": validated["manifest_digest"],
              "tensor_names": validated["tensor_names"],
              "format": validated["format"],
              "base_saved": validated["base_saved"],
              "components": validated["components"]}
    return result


def execute_v0(args, protocol, protocol_sha, dataset, source, spec, manifest_path):
    import torch
    from replica_lab.architecture.loading import load_local_mps
    require(torch.backends.mps.is_available(), "MPS is unavailable; no CPU fallback")
    torch.manual_seed(protocol["v0"]["generation"]["seed"])
    adapter, tokenizer, lock = load_local_mps(
        args.model_path, args.model_lock, profiler=args.profiler, vision=spec)
    require(adapter.manifest.vision is not None
            and adapter.manifest.vision.digest == spec.digest,
            "Loaded manifest does not carry the pinned vision spec")
    require(adapter.manifest.training_scope == "visual_prefix_text",
            "Vision loader must declare visual_prefix_text scope")
    require(adapter.vision_projector is not None,
            "Loaded adapter lacks the trainable vision projector")
    trainable = freeze_for_v0(adapter)
    base_stamp = adapter.parameter_stamp(original_only=True)
    scope = protocol_scope(protocol)
    ledger = empty_ledger(args.output)
    session = make_session(adapter, tokenizer, ledger)
    service = feature_service(protocol, args.device)
    encoder_audit_before = verify_encoder_frozen(service)
    output = Path(args.output)
    (output / "probes").mkdir(parents=True, exist_ok=True)

    parity = text_only_parity(session, adapter, tokenizer, protocol, scope, output)
    gradient = projector_gradient_probe(adapter, session, tokenizer, protocol,
                                        service, dataset, scope, ledger.epoch, output)
    dump(output / "probes" / "gradient.json", gradient)
    # Restart compares the exported projector, so all captured outputs must
    # follow the optional interface-only update rather than precede it.
    probes = evaluate_probes(session, tokenizer, adapter, protocol, service,
                             dataset, scope, ledger.epoch, output)
    write_jsonl(output / "probes" / "v0_generations.jsonl", probes["rows"])
    encoder_audit_after = verify_encoder_frozen(service)
    base_stamp_after = adapter.parameter_stamp(original_only=True)
    frozen = {
        "base_parameter_stamp_sha256": sha256_json(repr(base_stamp)),
        "base_parameter_stamp_unchanged": base_stamp == base_stamp_after,
        "base_trainable_parameters": sum(
            parameter.requires_grad for parameter in adapter.base.parameters()),
        "vision_projector_trainable": trainable,
        "nonprojector_trainable": sorted(
            name for name, parameter in adapter.named_parameters()
            if parameter.requires_grad and not name.startswith("vision_projector.")),
        "encoder_audit_before": encoder_audit_before,
        "encoder_audit_after": encoder_audit_after,
        "encoder_spec_digest": spec.digest,
        "encoder_source_manifest_sha256": sha256_file(manifest_path),
        "sealed_original_identity_matches": (
            adapter._original_identity == adapter.parameter_stamp(original_only=True)),
    }
    require(frozen["base_parameter_stamp_unchanged"], "Base parameters changed")
    require(frozen["base_trainable_parameters"] == 0, "Base has trainable parameters")
    require(frozen["sealed_original_identity_matches"],
            "Sealed original identity no longer matches")
    dump(output / "probes" / "frozen.json", frozen)

    extension = write_extension(adapter, output / "extension", protocol,
                                label="v0_extension")
    ledger_count = ledger_record_count(Path(args.output) / "ledger.sqlite")
    require(ledger_count == 0, "V0 must use an empty source ledger")
    checks = {
        "text_only_native_parity": parity["passed"],
        "base_frozen": frozen["base_parameter_stamp_unchanged"],
        "encoder_frozen_reference": frozen["encoder_spec_digest"] == spec.digest,
        "projector_only_trainable": not frozen["nonprojector_trainable"],
        "response_only_labels": gradient["response_only"],
        "projector_gradient_present_finite": (
            bool(gradient["projector_gradients_present"])
            and gradient["projector_gradients_finite"]
            and gradient["projector_gradients_nonzero"]),
        "decoder_input_gradient_finite": (
            gradient["decoder_input_gradient_finite"]
            and gradient["decoder_input_gradient_nonzero"]),
        "projector_update_ownership": (
            not gradient["update_enabled"]
            or (bool(gradient["update_changed"])
                and all(name.startswith("vision_projector.")
                        for name in gradient["update_changed"]))),
        "nonprojector_gradients_absent": not gradient["nonprojector_gradient_names"],
        "correct_wrong_noimage_pinned": probes["pinned_payloads_differ"],
        "correct_wrong_noimage_logit_use": all(
            value >= probes["distinct_logit_min"]
            for value in probes["logit_max_differences"].values()),
        "bounded_generation": all(
            len(row["generated_ids"]) <= protocol["v0"]["generation"]["max_new_tokens"]
            for row in probes["rows"]),
        "strict_extension_export": extension["format"]
            == "replica.minicpm5.extensions.v4",
        "empty_ledger": ledger_count == 0,
    }
    report = {
        "format": "replica.vision_qualification_run.v1",
        "stage": "v0",
        "protocol_sha256": protocol_sha,
        "implementation_sha256": verify_implementation_pins(protocol, ROOT),
        "model_lock_sha256": sha256_file(args.model_lock),
        "model_revision": lock["revision"],
        "data_audit": dataset.pins,
        "vision_spec": dict(protocol["vision"]),
        "vision_spec_digest": spec.digest,
        "vision_source_manifest_sha256": sha256_file(manifest_path),
        "text_parity": parity,
        "probes": {"rows": probes["rows"],
                   "logit_max_differences": probes["logit_max_differences"],
                   "distinct_logit_min": probes["distinct_logit_min"]},
        "gradient": gradient,
        "frozen": frozen,
        "trainable_receipt": trainable,
        "extension": extension,
        "checks": checks,
        "artifacts": artifact_hashes(output, [
            "probes/v0_generations.jsonl",
            "probes/gradient.json",
            "probes/frozen.json",
            "probes/parity_base.safetensors",
            "probes/parity_adapter.safetensors",
            *[f"probes/logits/{condition}.safetensors"
              for condition in ("correct", "wrong", "noimage")],
            "extension/manifest.json",
            "extension/extensions.safetensors",
        ]),
        "costs": {
            "generated_tokens": sum(row["generated_tokens"] for row in probes["rows"]),
            "prompt_tokens": sum(row["prompt_tokens"] for row in probes["rows"]),
            "optimizer_updates": 1 if gradient["update_enabled"] else 0,
            "vision_cache": encoder_audit_after,
            "downloads": 0,
        },
    }
    ledger.close()
    return report


def execute_restart(args, protocol, protocol_sha, dataset, source, spec,
                    manifest_path):
    import torch
    from replica_lab.architecture.loading import load_local_mps
    from safetensors.torch import load_file
    prior = Path(args.restart_run)
    prior_report, prior_sha = require_completed_run(
        prior, protocol, protocol_sha)
    require(prior_report.get("stage") == "v0", "Restart source is not a V0 run")
    require(prior_report.get("checks", {}).get("strict_extension_export") is True,
            "Restart source did not export a strict extension")
    prior_extension = prior / "extension"
    require(prior_extension.is_dir(), "Restart source extension is missing")
    prior_receipt = json.loads((prior_extension / "manifest.json").read_text())
    require(sha256_file(prior_extension / "extensions.safetensors")
            == prior_receipt["weights_sha256"],
            "Restart source extension bytes changed")
    rows = [json.loads(line) for line in
            (prior / "probes" / "v0_generations.jsonl").read_text().splitlines()
            if line.strip()]
    prior_by_condition = {row["condition"]: row for row in rows}
    require(set(prior_by_condition) == {"correct", "wrong", "noimage"},
            "Restart source probes are incomplete")

    torch.manual_seed(protocol["v0"]["generation"]["seed"])
    adapter, tokenizer, lock = load_local_mps(
        args.model_path, args.model_lock, profiler=args.profiler, vision=spec)
    require(adapter.manifest.vision.digest == spec.digest,
            "Restart manifest vision differs")
    adapter.load_extension(prior_extension)
    for _, parameter in adapter.named_parameters():
        parameter.requires_grad_(False)
    adapter.eval()
    extension_before = sha256_file(prior_extension / "extensions.safetensors")
    scope = protocol_scope(protocol)
    ledger = empty_ledger(args.output)
    session = make_session(adapter, tokenizer, ledger)
    service = feature_service(protocol, args.device)
    output = Path(args.output)
    (output / "prior_logits").mkdir(parents=True, exist_ok=True)
    (output / "restart_logits").mkdir(parents=True, exist_ok=True)
    probe = protocol["v0"]["probe"]
    select_probe(dataset, protocol["v0"])
    case_id = probe["case_id"]
    wrong_case_id = probe["wrong_image_case_id"]
    question = dataset.row(case_id)["question"]
    maximum = protocol["limits"]["max_prompt_tokens"]
    generation_limit = protocol["v0"]["generation"]["max_new_tokens"]
    tolerance = protocol["v0"]["restart"]
    conditions = []
    new_rows = []
    for condition, image_case, task_id in (
            ("correct", case_id, case_id),
            ("wrong", wrong_case_id, case_id),
            ("noimage", None, case_id)):
        prior_row = prior_by_condition[condition]
        prior_logits_path = prior / prior_row["logits_file"]
        require(sha256_file(prior_logits_path) == prior_row["logits_sha256"],
                "Prior logits changed before restart comparison")
        copied = output / "prior_logits" / f"{condition}.safetensors"
        shutil.copyfile(prior_logits_path, copied)
        visual = (None if image_case is None else observe(
            service, dataset, image_case_id=image_case, task_id=task_id,
            source_id=protocol["vision_source"]["source_id"], scope=scope,
            ledger_epoch=ledger.epoch, spec=spec))
        ids, mask, request = build_request(
            session, tokenizer, text=question, visual=visual, task_id=task_id,
            scope=scope, device="mps", maximum=maximum)
        require([int(value) for value in ids[0].tolist()] == prior_row["prompt_ids"],
                "Restart prompt tokens differ")
        logits_path = output / "restart_logits" / f"{condition}.safetensors"
        generated = generate_capture(session, request,
                                     max_new_tokens=generation_limit,
                                     logits_path=logits_path,
                                     eos_token_ids={tokenizer.eos_token_id})
        new_row = {
            "condition": condition, "case_id": case_id, "prompt_ids": prior_row["prompt_ids"],
            "prompt_sha256": prior_row["prompt_sha256"],
            "generated_ids": generated["generated_ids"],
            "logits_sha256": generated["logits_sha256"],
            "logits_path": str(Path("restart_logits") / f"{condition}.safetensors"),
            "image_case_id": image_case,
            "visual_cache_key": (None if visual is None else visual.cache_key),
        }
        comparison = compare_restart_traces(
            {**prior_row, "logits_path": str(copied)},
            {**new_row, "logits_path": str(logits_path)},
            atol=float(tolerance["atol"]), rtol=float(tolerance["rtol"]))
        require(prior_row["visual_cache_key"] == new_row["visual_cache_key"],
                "Restart visual cache key differs")
        conditions.append({
            "condition": condition,
            "prior_generated_ids": prior_row["generated_ids"],
            "restart_generated_ids": generated["generated_ids"],
            "tokens_equal": comparison["tokens_equal"],
            "prior_logits_sha256": prior_row["logits_sha256"],
            "restart_logits_sha256": generated["logits_sha256"],
            "logits_equal": comparison["logits_equal"],
            "maximum_absolute_difference": comparison["maximum_absolute_difference"],
            "tolerance": comparison["tolerance"],
            "prompt_sha256": prior_row["prompt_sha256"],
        })
        new_rows.append(new_row)
        write_jsonl(output / "restart_generations.jsonl", new_rows)
    write_jsonl(output / "restart_generations.jsonl", new_rows)
    require(ledger_record_count(Path(args.output) / "ledger.sqlite") == 0,
            "Restart ledger is not empty")
    require(sha256_file(prior_extension / "extensions.safetensors")
            == extension_before, "Restart changed exported extension bytes")
    base_stamp = adapter.parameter_stamp(original_only=True)
    require(adapter._original_identity == base_stamp,
            "Restart changed the sealed native identity")
    report = {
        "format": "replica.vision_qualification_restart.v1",
        "stage": "v0",
        "status": "completed",
        "protocol_sha256": protocol_sha,
        "prior_run_report_path": str(prior),
        "prior_run_report_sha256": prior_sha,
        "prior_extension_weights_sha256": prior_receipt["weights_sha256"],
        "prior_extension_manifest_sha256": sha256_file(
            prior_extension / "manifest.json"),
        "loaded_extension": {
            "path_sha256": sha256_text(str(prior_extension.resolve())),
            "weights_sha256": prior_receipt["weights_sha256"],
            "manifest_sha256": sha256_file(prior_extension / "manifest.json"),
            "tensor_names": sorted(
                name for names in prior_receipt["components"].values()
                for name in names),
        },
        "zero_updates": True,
        "implementation_sha256": verify_implementation_pins(protocol, ROOT),
        "costs": {"zero_updates": True,
                  "generated_tokens": sum(len(row["generated_ids"])
                                          for row in new_rows),
                  "vision_cache": service.cache_audit(),
                  "downloads": 0},
        "conditions": conditions,
        "restart_generations_path": "restart_generations.jsonl",
        "restart_generations_sha256": sha256_file(
            output / "restart_generations.jsonl"),
        "artifacts": artifact_hashes(output, [
            "restart_generations.jsonl",
            *[f"prior_logits/{condition}.safetensors"
              for condition in ("correct", "wrong", "noimage")],
            *[f"restart_logits/{condition}.safetensors"
              for condition in ("correct", "wrong", "noimage")],
        ]),
        "checks": {"tokens_equal": all(c["tokens_equal"] for c in conditions),
                   "logits_equal": all(c["logits_equal"] for c in conditions),
                   "zero_updates": True,
                   "extension_unchanged": True,
                   "ledger_empty": True},
    }
    ledger.close()
    return report


def execute_v1_restart(args, protocol, protocol_sha, dataset, source, spec,
                       manifest_path):
    """Reload a completed V1 extension in a fresh process and re-evaluate bitwise."""
    import torch
    from replica_lab.architecture.loading import load_local_mps
    prior = Path(args.restart_run)
    prior_report, prior_sha = require_completed_run(prior, protocol, protocol_sha)
    require(prior_report.get("stage") == "v1", "Restart source is not a V1 run")
    require(prior_report.get("checks", {}).get("strict_extension_export") is True,
            "Restart source did not export a strict extension")
    prior_extension = prior / "final_extension"
    require(prior_extension.is_dir(), "Restart source extension is missing")
    prior_receipt = json.loads((prior_extension / "manifest.json").read_text())
    require(sha256_file(prior_extension / "extensions.safetensors")
            == prior_receipt["weights_sha256"],
            "Restart source extension bytes changed")

    adapter, tokenizer, lock = load_local_mps(
        args.model_path, args.model_lock, profiler=args.profiler, vision=spec)
    adapter.load_extension(prior_extension)
    for _, parameter in adapter.named_parameters():
        parameter.requires_grad_(False)
    adapter.eval()
    scope = protocol_scope(protocol)
    ledger = empty_ledger(args.output)
    session = make_session(adapter, tokenizer, ledger)
    service = feature_service(protocol, args.device)
    output = Path(args.output)
    train_splits = [dataset.case(row["case_id"])["split"] for row in dataset.model_rows
                    if dataset.case(row["case_id"])["split"] == "train"]
    train_splits = sorted(set(train_splits))
    admission = verify_admission(
        protocol, dataset, train_splits=train_splits,
        eval_splits=protocol["v1"]["eval"]["splits"])
    evaluation = evaluate_v1(adapter, session, tokenizer, protocol, service,
                             dataset, admission, scope, output)
    retention = evaluate_retention(session, tokenizer, protocol, admission,
                                   scope, output)
    prior_eval = [json.loads(line) for line in
                  (prior / "v1" / "eval_generations.jsonl").read_text().splitlines()
                  if line.strip()]
    new_eval = [json.loads(line) for line in
                (output / "v1" / "eval_generations.jsonl").read_text().splitlines()
                if line.strip()]
    require(prior_eval == new_eval, "Fresh-process V1 evaluation generations differ")
    prior_ret = [json.loads(line) for line in
                 (prior / "v1" / "retention_generations.jsonl").read_text().splitlines()
                 if line.strip()]
    new_ret = [json.loads(line) for line in
               (output / "v1" / "retention_generations.jsonl").read_text().splitlines()
               if line.strip()]
    require(prior_ret == new_ret, "Fresh-process V1 retention generations differ")
    require(adapter._original_identity == adapter.parameter_stamp(original_only=True),
            "Restart changed the sealed native identity")
    require(ledger_record_count(output / "ledger.sqlite") == 0,
            "Restart ledger is not empty")
    return {
        "format": "replica.vision_qualification_restart.v1",
        "stage": "v1", "status": "completed",
        "protocol_sha256": protocol_sha,
        "prior_run_report_path": str(prior),
        "prior_run_report_sha256": prior_sha,
        "prior_extension_weights_sha256": prior_receipt["weights_sha256"],
        "loaded_extension": {
            "weights_sha256": prior_receipt["weights_sha256"],
            "manifest_sha256": sha256_file(prior_extension / "manifest.json"),
            "tensor_names": sorted(
                name for names in prior_receipt["components"].values()
                for name in names),
        },
        "zero_updates": True,
        "implementation_sha256": verify_implementation_pins(protocol, ROOT),
        "costs": {
            "zero_updates": True,
            "generated_tokens": sum(row["generated_tokens"] for row in new_eval),
            "vision_cache": service.cache_audit(),
            "downloads": 0,
        },
        "checks": {
            "tokens_equal": prior_eval == new_eval,
            "retention_tokens_equal": prior_ret == new_ret,
            "zero_updates": True,
            "extension_unchanged": True,
            "ledger_empty": True,
        },
        "artifacts": artifact_hashes(output, [
            "v1/eval_generations.jsonl",
            "v1/retention_generations.jsonl",
        ]),
    }



def freeze_for_v1(adapter):
    adapter.set_regime("modules_only")
    trainable = []
    for name, parameter in adapter.named_parameters():
        parameter.requires_grad_(name.startswith("vision_projector."))
        if parameter.requires_grad:
            trainable.append(name)
    require(trainable and all(name.startswith("vision_projector.")
                              for name in trainable),
            "V1 trainable scope must be projector-only")
    require(adapter.vision_projector is not None,
            "V1 requires the vision projector")
    require(not any(parameter.requires_grad for name, parameter
                    in adapter.named_parameters()
                    if not name.startswith("vision_projector.")),
            "V1 froze an unexpected trainable tensor")
    return trainable


def train_projector(adapter, session, tokenizer, protocol, service, dataset,
                    admission, scope, output):
    import torch
    from replica_lab.optimizers import ReplicaOptimizer
    train = protocol["v1"]["train"]
    limits = protocol["limits"]
    adapter.train()
    torch.manual_seed(train["seed"])
    rows = admission["_training_rows"]
    schedule = training_schedule(rows, train)
    groups = projector_optimizer_groups(
        adapter.vision_projector, train, prefix="vision_projector.")
    optimizer_name = train.get("optimizer", "muon8bit")
    require(optimizer_name in ("muon8bit", "sgd"), "Unsupported projector optimizer")
    if optimizer_name == "sgd":
        optimizer = torch.optim.SGD(
            [parameter for group in groups for parameter in group["params"]],
            lr=float(train["lr_new"]))
    else:
        optimizer = ReplicaOptimizer(groups, "muon8bit")
    route = optimizer_route(groups)
    trainable = freeze_for_v1(adapter)
    require({item["name"] for item in route} == set(trainable),
            "Optimizer route and trainable receipt differ")
    base_stamp = adapter.parameter_stamp(original_only=True)
    nonprojector_before = {
        name: parameter.detach().cpu().clone()
        for name, parameter in adapter.named_parameters()
        if not name.startswith(("vision_projector.", "base."))}
    trace = []
    all_examples = []
    completed = 0
    budget_stop = None
    started = time.monotonic()
    for update_index, update_rows in enumerate(schedule[:train["updates"]], 1):
        elapsed = time.monotonic() - started
        if elapsed > train["max_seconds"]:
            budget_stop = "training_time_budget"
            break
        optimizer.zero_grad(set_to_none=True)
        losses = []
        tokens = {"prompt": 0, "response": 0}
        examples = []
        grad_names = set()
        for row in update_rows:
            visual = observe(service, dataset, image_case_id=row["case_id"],
                             task_id=row["case_id"],
                             source_id=protocol["vision_source"]["source_id"],
                             scope=scope, ledger_epoch=session.ledger.epoch,
                             spec=service.spec)
            device = adapter.base.get_input_embeddings().weight.device
            prompt = prompt_ids(tokenizer, row["prompt"], device,
                                limits["max_prompt_tokens"])
            response = tokenizer(row["response"], add_special_tokens=False)["input_ids"]
            require(bool(response), "Training response tokenizes to nothing")
            require(tokenizer.eos_token_id is not None,
                    "Tokenizer requires an explicit EOS token")
            response = list(response) + [int(tokenizer.eos_token_id)]
            input_ids = torch.cat(
                (prompt, torch.tensor([response], dtype=torch.long,
                                      device=prompt.device)), dim=1)
            require(input_ids.shape[1] <= limits["max_prompt_tokens"],
                    "Training example exceeds the declared sequence budget")
            labels = torch.tensor([[-100] * prompt.shape[1] + response],
                                  dtype=torch.long, device=prompt.device)
            require(not bool((labels[:, :prompt.shape[1]] != -100).any()),
                    "Training prompt labels are not masked")
            with session.training_request(
                    prompt_ids=prompt,
                    prompt_mask=torch.ones_like(prompt, dtype=torch.bool),
                    query=row["prompt"], scope=scope,
                    task=task_graph(row["case_id"], row["prompt"]),
                    visual=visual, enabled_ports=(),
                    core_impl="none", enabled_branches=("memory",)) as request:
                require(request.frame.sense.visual is visual,
                        "Training request lost its pinned visual payload")
                result = session.forward(
                    request, input_ids,
                    attention_mask=torch.ones_like(input_ids, dtype=torch.bool),
                    labels=labels)
                require(result.logits.shape[1] == input_ids.shape[1],
                        "Projector training logits are not text-aligned")
                loss = result.loss
                require(loss is not None and bool(torch.isfinite(loss)),
                        "Projector training loss is unavailable or nonfinite")
                require(loss.requires_grad,
                        "Projector training loss is detached from autograd")
                session.backward(request, loss / len(update_rows))
                losses.append(float(loss.detach().cpu()))
                example = {
                    "update": update_index, "case_id": row["case_id"],
                    "backward_called": True, "loss": losses[-1],
                    "prompt_tokens": int(prompt.shape[1]), "response_tokens": len(response),
                    "input_ids_sha256": sha256_json(input_ids[0].tolist()),
                    "labels_sha256": sha256_json(labels[0].tolist()),
                }
                examples.append(example)
                all_examples.append(example)
                write_jsonl(Path(output) / "v1" / "training_examples.jsonl", all_examples)
            tokens["prompt"] += int(prompt.shape[1])
            tokens["response"] += len(response)
            for name, parameter in adapter.named_parameters():
                if parameter.grad is not None:
                    grad_names.add(name)
        nonprojector_grads = sorted(
            name for name in grad_names if not name.startswith("vision_projector."))
        require(not nonprojector_grads,
                f"V1 non-projector tensors received gradients: {nonprojector_grads}")
        projector_names = {"vision_projector." + name for name, _ in adapter.vision_projector.named_parameters()}
        require(grad_names == projector_names,
                "V1 projector gradient inventory is incomplete")
        for name, parameter in adapter.vision_projector.named_parameters():
            require(parameter.grad is not None, f"V1 projector gradient missing: {name}")
        norm = float(torch.nn.utils.clip_grad_norm_(
            adapter.vision_projector.parameters(), train["max_grad_norm"],
            error_if_nonfinite=True).detach().cpu())
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        completed = update_index
        trace.append({
            "update": update_index,
            "seed": train["seed"],
            "repeat": ((update_index - 1) * train["grad_accum"]) // max(len(rows), 1) + 1,
            "case_ids": [row["case_id"] for row in update_rows],
            "losses": losses,
            "loss": sum(losses) / len(losses),
            "grad_norm": norm,
            "backward_calls": len(examples),
            "examples": examples,
            "max_grad_norm": train["max_grad_norm"],
            "response_only_labels": True,
            "decoder_autograd": True,
            "text_aligned_logits": True,
            "projector_gradient_names": sorted(grad_names),
            "nonprojector_gradient_names": nonprojector_grads,
            "prompt_tokens": tokens["prompt"],
            "response_tokens": tokens["response"],
            "wall_seconds": time.monotonic() - started,
            "base_parameter_stamp_unchanged": (
                adapter.parameter_stamp(original_only=True) == base_stamp),
        })
        require(trace[-1]["base_parameter_stamp_unchanged"],
                "V1 update changed sealed native parameters")
        write_jsonl(Path(output) / "v1" / "training_trace.jsonl", trace)
    base_unchanged = adapter.parameter_stamp(original_only=True) == base_stamp
    nonprojector_unchanged = all(
        torch.equal(value, parameter.detach().cpu())
        for name, value in nonprojector_before.items()
        for parameter in [dict(adapter.named_parameters())[name]])
    require(base_unchanged, "V1 training changed the sealed native identity")
    require(nonprojector_unchanged, "V1 training changed a non-projector tensor")
    audit = (optimizer.audit() if optimizer_name == "muon8bit" else {
        "optimizer": "sgd", "lr": float(train["lr_new"]),
        "parameter_dtype": "float32",
        "groups": [{"group_name": "projector_sgd",
                    "param_names": [name for group in groups
                                    for name in group["param_names"]]}]})
    adapter.eval()
    return {
        "trace": trace,
        "completed_updates": completed,
        "declared_updates": train["updates"],
        "budget_stop": budget_stop,
        "trainable_receipt": trainable,
        "optimizer_route": route,
        "optimizer_audit": audit,
        "base_parameter_stamp_unchanged": base_unchanged,
        "nonprojector_tensors_unchanged": nonprojector_unchanged,
        "wall_seconds": time.monotonic() - started,
        "response_only_labels": True,
        "decoder_autograd": True,
        "zero_nonprojector_gradients": True,
    }


def generation_row(session, tokenizer, *, control, case_id, dataset, question,
                   visual, scope, maximum, generation_limit, text_context=None,
                   text_context_kind=None, image_case_id=None):
    ids, mask, request = build_request(
        session, tokenizer, text=question, visual=visual, task_id=case_id,
        scope=scope, device=session.device, maximum=maximum)
    try:
        token_ids = session.generate(request, max_new_tokens=generation_limit)
        prompt_len = int(request.frame.sense.prompt_ids.shape[1])
        generated = [int(value) for value in token_ids[0, prompt_len:].tolist()]
    finally:
        request.release()
    require(1 <= len(generated) <= generation_limit,
            "Evaluation generation exceeded its budget")
    case = dataset.case(case_id)
    return {
        "control": control, "case_id": case_id,
        "family": case["group_id"], "group_id": case["group_id"],
        "split": case["split"],
        "prompt_ids": [int(value) for value in ids[0].tolist()],
        "prompt_sha256": sha256_json([int(value) for value in ids[0].tolist()]),
        "generated_ids": generated,
        "prompt_tokens": int(ids.shape[1]),
        "generated_tokens": len(generated),
        "ended_by_eos": generated[-1] == tokenizer.eos_token_id,
        "max_new_tokens": generation_limit,
        "image_sha256": (None if image_case_id is None
                         else dataset.asset_for_case(image_case_id)["sha256"]),
        "image_case_id": image_case_id,
        "text_context_kind": text_context_kind,
        "text_sha256": (None if text_context is None
                        else sha256_text(text_context)),
        "text_context_information": (None if text_context_kind is None
                                      else "pinned_admission_context"),
    }


def evaluate_v1(adapter, session, tokenizer, protocol, service, dataset,
                admission, scope, output):
    evaluation = protocol["v1"]["eval"]
    controls = evaluation["required_controls"]
    maximum = protocol["limits"]["max_prompt_tokens"]
    generation_limit = evaluation["max_new_tokens"]
    cases = select_eval_cases(dataset, evaluation)
    case_ids = [row["case_id"] for row in cases]
    assignment = assign_controls(dataset, case_ids)
    contexts = {(row["case_id"], row["kind"]): row
                for row in admission.get("_text_contexts", ())}
    template = protocol["v1"]["controls"]["text_context_template"]
    rows = []
    started = time.monotonic()
    deadline = protocol["limits"]["max_seconds"]
    complete = True
    for case_id in case_ids:
        question = visual_question(protocol, dataset.row(case_id)["question"])
        require(not leakage_hits(question, dataset.label(case_id)),
                "Evaluation question leaks its verifier answer")
        visual_specs = {
            "correct": (case_id, case_id),
            "wrong": (assignment["wrong"][case_id], case_id),
            "shuffled": (assignment["shuffled"][case_id], case_id),
            "noimage": (None, case_id),
        }
        for control in ("correct", "wrong", "shuffled", "noimage"):
            if control not in controls:
                continue
            if time.monotonic() - started > deadline:
                complete = False
                break
            image_case_id, task_id = visual_specs[control]
            visual = (None if image_case_id is None else observe(
                service, dataset, image_case_id=image_case_id, task_id=task_id,
                source_id=protocol["vision_source"]["source_id"], scope=scope,
                ledger_epoch=session.ledger.epoch, spec=service.spec))
            rows.append(generation_row(
                session, tokenizer, control=control, case_id=case_id,
                dataset=dataset, question=question, visual=visual, scope=scope,
                maximum=maximum, generation_limit=generation_limit,
                image_case_id=image_case_id))
            write_jsonl(Path(output) / "v1" / "eval_generations.jsonl", rows)
        if not complete:
            break
        for control in ("captiononly", "ocrtextonly"):
            if control not in controls:
                continue
            kind = {"captiononly": "caption_only",
                    "ocrtextonly": "ocr_text_only"}[control]
            context = contexts[(case_id, kind)]["text"]
            text = build_text_context_prompt(question, context, template)
            if time.monotonic() - started > deadline:
                complete = False
                break
            rows.append(generation_row(
                session, tokenizer, control=control, case_id=case_id,
                dataset=dataset, question=text, visual=None, scope=scope,
                maximum=maximum, generation_limit=generation_limit,
                text_context=context, text_context_kind=kind,
                image_case_id=None))
            write_jsonl(Path(output) / "v1" / "eval_generations.jsonl", rows)
        if not complete:
            break
    write_jsonl(Path(output) / "v1" / "eval_generations.jsonl", rows)
    expected = {"correct", "wrong", "shuffled", "noimage"} & set(controls)
    if complete:
        for control in expected:
            count = sum(row["control"] == control for row in rows)
            require(count == len(case_ids),
                    f"Control {control} did not cover every evaluation case")
        for control in ("captiononly", "ocrtextonly"):
            if control in controls:
                count = sum(row["control"] == control for row in rows)
                require(count == len(case_ids),
                        f"Text control {control} did not cover every case")
    return {"rows": rows, "complete": complete,
            "cases": len(case_ids), "controls": list(controls),
            "assignment": assignment,
            "case_ids": case_ids,
            "wall_seconds": time.monotonic() - started,
            "eval_generations_sha256": sha256_file(
                Path(output) / "v1" / "eval_generations.jsonl")}


def evaluate_retention(session, tokenizer, protocol, admission, scope, output, *, baseline=False):
    retention = protocol["v1"]["retention"]
    maximum = protocol["limits"]["max_prompt_tokens"]
    generation_limit = protocol["v1"]["eval"]["max_new_tokens"]
    prompts = admission["_retention_prompts"][:retention["max_cases"]]
    filename = "baseline_retention_generations.jsonl" if baseline else "retention_generations.jsonl"
    path = Path(output) / "v1" / filename
    rows = []
    started = time.monotonic()
    complete = True
    for row in prompts:
        if time.monotonic() - started > protocol["limits"]["max_seconds"]:
            complete = False
            break
        ids, mask, request = build_request(
            session, tokenizer, text=row["prompt"], visual=None,
            task_id=row["case_id"], scope=scope, device=session.device, maximum=maximum)
        try:
            token_ids = session.generate(request, max_new_tokens=generation_limit)
            prompt_len = int(request.frame.sense.prompt_ids.shape[1])
            generated = [int(value) for value in token_ids[0, prompt_len:].tolist()]
        finally:
            request.release()
        rows.append({
            "case_id": row["case_id"], "family": row["family"],
            "group_id": row["group_id"], "split": row["split"],
            "prompt_ids": [int(value) for value in ids[0].tolist()],
            "prompt_sha256": sha256_json([int(value) for value in ids[0].tolist()]),
            "generated_ids": generated, "prompt_tokens": int(ids.shape[1]),
            "generated_tokens": len(generated),
            "ended_by_eos": generated[-1] == tokenizer.eos_token_id,
        })
        write_jsonl(path, rows)
    write_jsonl(path, rows)
    return {"rows": rows, "complete": complete and len(rows) == len(prompts),
            "cases": len(rows), "generations_path": str(Path("v1") / filename),
            "retention_generations_sha256": sha256_file(path),
            "wall_seconds": time.monotonic() - started}


def execute_v1(args, protocol, protocol_sha, dataset, source, spec, manifest_path):
    import torch
    from replica_lab.architecture.loading import load_local_mps
    stage_protocol = protocol
    prerequisites = stage_protocol["v1"]["prerequisites"]
    # Gate before native loading: accepted independent V0 and named admission.
    verify_v0_prerequisite(stage_protocol)
    train_splits = [dataset.case(row["case_id"])["split"] for row in dataset.model_rows
                    if dataset.case(row["case_id"])["split"] == "train"]
    train_splits = sorted(set(train_splits))
    require(bool(train_splits), "Admitted V1 data lacks a training split")
    admission = verify_admission(
        stage_protocol, dataset, train_splits=train_splits,
        eval_splits=stage_protocol["v1"]["eval"]["splits"])
    budget_plan = {
        "updates": stage_protocol["v1"]["train"]["updates"],
        "evaluation_cases": stage_protocol["v1"]["eval"]["max_cases"],
        "generation_tokens": stage_protocol["v1"]["eval"]["max_new_tokens"],
        "training_rows": len(admission["_training_rows"]),
        "max_seconds": stage_protocol["v1"]["train"]["max_seconds"],
    }
    validate_budget(budget_plan, stage_protocol["limits"])
    training_schedule(admission["_training_rows"], stage_protocol["v1"]["train"])
    require(torch.backends.mps.is_available(), "MPS is unavailable; no CPU fallback")
    torch.manual_seed(stage_protocol["v1"]["train"]["seed"])
    adapter, tokenizer, lock = load_local_mps(
        args.model_path, args.model_lock, profiler=args.profiler, vision=spec)
    require(adapter.manifest.vision.digest == spec.digest,
            "V1 loaded manifest vision differs")
    checkpoint = stage_protocol["v1"].get("initial_checkpoint")
    if checkpoint is None:
        initial_path = Path(args.output) / "initial_extension"
        initial_receipt = write_extension(adapter, initial_path, stage_protocol,
                                          label="v1_initial_fresh")
    else:
        initial_path = resolve_project_path(stage_protocol, checkpoint["path"])
        require(sha256_file(initial_path / "manifest.json")
                == checkpoint["manifest_sha256"],
                "V1 initial checkpoint manifest differs from its pin")
        require(sha256_file(initial_path / "extensions.safetensors")
                == checkpoint["weights_sha256"],
                "V1 initial checkpoint weights differ from their pin")
        initial_receipt = json.loads((initial_path / "manifest.json").read_text())
        require(initial_receipt["format"] == "replica.minicpm5.extensions.v4",
                "V1 initial checkpoint is not a strict v4 vision extension")
        adapter.load_extension(initial_path)
    require(adapter.manifest.vision.digest == spec.digest,
            "V1 reloaded extension vision differs")
    output = Path(args.output)
    (output / "v1").mkdir(parents=True, exist_ok=True)
    scope = protocol_scope(stage_protocol)
    ledger = empty_ledger(args.output)
    session = make_session(adapter, tokenizer, ledger)
    service = feature_service(stage_protocol, args.device)
    encoder_audit_before = verify_encoder_frozen(service)
    baseline_retention = evaluate_retention(
        session, tokenizer, stage_protocol, admission, scope, output, baseline=True)
    require(baseline_retention["complete"], "Initial text baseline did not complete")
    training = train_projector(adapter, session, tokenizer, stage_protocol,
                               service, dataset, admission, scope, output)
    write_jsonl(output / "v1" / "training_trace.jsonl", training["trace"])
    dump(output / "v1" / "trainable_receipt.json",
         {"trainable": training["trainable_receipt"],
          "nonprojector_gradients": []})
    dump(output / "v1" / "optimizer_route.json", training["optimizer_route"])
    dump(output / "v1" / "optimizer_audit.json", training["optimizer_audit"])
    final_extension = write_extension(adapter, output / "final_extension",
                                      stage_protocol, label="v1_final_extension")
    # Strict reload before any evaluation; fixed updates were already completed.
    adapter.load_extension(output / "final_extension")
    lineage = compare_extension_tensors(initial_path, output / "final_extension")
    dump(output / "v1" / "extension_lineage.json", lineage)
    evaluation = evaluate_v1(adapter, session, tokenizer, stage_protocol, service,
                             dataset, admission, scope, output)
    retention = evaluate_retention(session, tokenizer, stage_protocol, admission,
                                   scope, output)
    encoder_audit_after = verify_encoder_frozen(service)
    require(ledger_record_count(Path(args.output) / "ledger.sqlite") == 0,
            "V1 must use an empty source ledger")
    require(training["completed_updates"] == training["declared_updates"],
            "V1 training did not complete its declared updates")
    require(evaluation["complete"], "V1 evaluation did not complete its declared bound")
    require(retention["complete"], "V1 text retention did not complete")
    checks = {
        "v0_prerequisite_accepted": True,
        "named_admission_verified": True,
        "projector_only_trainable": all(
            name.startswith("vision_projector.")
            for name in training["trainable_receipt"]),
        "nonprojector_tensors_frozen": training["nonprojector_tensors_unchanged"],
        "zero_nonprojector_gradients": training["zero_nonprojector_gradients"],
        "response_only_labels": training["response_only_labels"],
        "decoder_autograd_retained": training["decoder_autograd"],
        "declared_updates_completed": (
            training["completed_updates"] == training["declared_updates"]),
        "budget_stop_absent": training["budget_stop"] is None,
        "strict_extension_export": final_extension["format"]
            == "replica.minicpm5.extensions.v4",
        "only_projector_weights_changed": bool(lineage["changed"])
            and all(name.startswith("vision_projector.")
                    for name in lineage["changed"]),
        "evaluation_complete": evaluation["complete"],
        "retention_complete": retention["complete"],
        "baseline_retention_complete": baseline_retention["complete"],
        "text_only_tokens_unchanged": (
            [(row["case_id"], row["generated_ids"]) for row in baseline_retention["rows"]]
            == [(row["case_id"], row["generated_ids"]) for row in retention["rows"]]),
        "empty_ledger": True,
    }
    report = {
        "format": "replica.vision_qualification_run.v1",
        "stage": "v1",
        "protocol_sha256": protocol_sha,
        "implementation_sha256": verify_implementation_pins(stage_protocol, ROOT),
        "model_lock_sha256": sha256_file(args.model_lock),
        "model_revision": lock["revision"],
        "data_audit": dataset.pins,
        "admission_sha256": prerequisites["data_admission"]["sha256"],
        "v0_score_report_sha256": prerequisites["v0_score_report"]["sha256"],
        "vision_spec": dict(stage_protocol["vision"]),
        "vision_spec_digest": spec.digest,
        "vision_source_manifest_sha256": sha256_file(manifest_path),
        "training": training,
        "evaluation": {key: value for key, value in evaluation.items()
                       if key != "rows"},
        "retention": {key: value for key, value in retention.items()
                      if key != "rows"},
        "baseline_retention": {key: value for key, value in baseline_retention.items()
                               if key != "rows"},
        "initial_checkpoint": checkpoint,
        "final_extension": final_extension,
        "extension_lineage": lineage,
        "trainable_receipt": training["trainable_receipt"],
        "optimizer_route": training["optimizer_route"],
        "optimizer_audit": training["optimizer_audit"],
        "checks": checks,
        "artifacts": artifact_hashes(output, [
            "v1/training_trace.jsonl",
            "v1/training_examples.jsonl",
            "v1/baseline_retention_generations.jsonl",
            "v1/trainable_receipt.json",
            "v1/optimizer_route.json",
            "v1/optimizer_audit.json",
            "v1/eval_generations.jsonl",
            "v1/retention_generations.jsonl",
            "v1/extension_lineage.json",
            "final_extension/manifest.json",
            "final_extension/extensions.safetensors",
        ]),
        "costs": {
            "optimizer_updates": training["completed_updates"],
            "training_wall_seconds": training["wall_seconds"],
            "evaluation_wall_seconds": evaluation["wall_seconds"],
            "retention_wall_seconds": retention["wall_seconds"],
            "baseline_retention_wall_seconds": baseline_retention["wall_seconds"],
            "training_backward_calls": sum(row["backward_calls"] for row in training["trace"]),
            "evaluation_prompt_tokens": sum(row["prompt_tokens"]
                                            for row in evaluation["rows"]),
            "evaluation_generated_tokens": sum(row["generated_tokens"]
                                               for row in evaluation["rows"]),
            "retention_generated_tokens": sum(row["generated_tokens"]
                                              for row in retention["rows"]),
            "vision_cache_before": encoder_audit_before,
            "vision_cache_after": encoder_audit_after,
            "downloads": 0,
        },
    }
    ledger.close()
    return report


def require_completed_run(run, protocol, protocol_sha):
    from replica_lab.vision_qualification import require_completed_run as helper
    return helper(run, protocol, protocol_sha)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("mps",), default="mps")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-lock", type=Path, required=True)
    parser.add_argument("--restart-run", type=Path,
                        help="Fresh-process V0 parity against this completed run")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.device != "mps":
        parser.error("Only the declared MPS device is supported")
    if not args.protocol.is_file():
        parser.error("Protocol file is missing")
    protocol_sha = sha256_file(args.protocol)
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    validate_protocol(protocol)
    output = ensure_fresh_directory(args.output)
    free = shutil.disk_usage(ROOT).free
    if free < protocol["limits"]["min_free_disk_bytes"]:
        parser.error("Insufficient free disk headroom for the declared run")
    require(sha256_file(args.model_lock) == protocol["model"]["lock_sha256"],
            "Model lock bytes differ from the protocol pin")
    lock = json.loads(args.model_lock.read_text())
    require(lock.get("revision") == protocol["model"]["revision"],
            "Model revision differs from the protocol pin")
    implementation = verify_implementation_pins(protocol, ROOT)
    dataset = load_verified_dataset(protocol)
    source, spec, manifest_path = verify_source_and_spec(protocol, args.device)

    from replica_lab.profiling import RunProfiler
    from replica_lab.architecture.loading import exclusive_mps_worker
    report = {
        "format": "replica.vision_qualification_run.v1",
        "stage": protocol["stage"],
        "protocol_sha256": protocol_sha,
        "protocol": protocol,
        "arguments": {key: str(value) for key, value in vars(args).items()},
        "implementation_sha256": implementation,
        "started_at_utc": utc_now(),
        "packages": {name: importlib.metadata.version(name)
                     for name in ("torch", "transformers", "safetensors")},
        "environment": {key: os.environ[key] for key in
                        ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE",
                         "HF_DATASETS_OFFLINE", "PYTORCH_ENABLE_MPS_FALLBACK")},
        "free_disk_before_bytes": free,
        "downloads": 0,
    }
    profiler = RunProfiler(args.device, output, sample_interval=0.25)
    args.profiler = profiler
    restart = args.restart_run is not None
    try:
        with exclusive_mps_worker(ROOT):
            profiler.start()
            native_started = time.monotonic()
            if restart:
                result = (execute_v1_restart(args, protocol, protocol_sha, dataset,
                                             source, spec, manifest_path)
                          if protocol["stage"] == "v1"
                          else execute_restart(args, protocol, protocol_sha, dataset,
                                               source, spec, manifest_path))
            elif protocol["stage"] == "v0":
                result = execute_v0(args, protocol, protocol_sha, dataset,
                                    source, spec, manifest_path)
            else:
                result = execute_v1(args, protocol, protocol_sha, dataset,
                                    source, spec, manifest_path)
            report.update(result)
            require(time.monotonic() - native_started <= protocol["limits"]["max_seconds"],
                    "Native run exceeded its declared wall-time acceptance ceiling")
            report["status"] = "completed"
    except BaseException as error:
        report.update(status="failed", error_type=type(error).__name__,
                      error=str(error), traceback=traceback.format_exc(),
                      blocks_dependents=(["V2", "V3_learned_visual_memory"]
                                         if protocol["stage"] == "v1" else ["V1"]))
        try:
            dump(output / "failure_report.json", report)
        finally:
            raise
    finally:
        profile = profiler.stop()
        report["profile"] = profile
        memory = (profile or {}).get("memory") or {}
        report["costs"] = {**report.get("costs", {}),
                           "profiled_wall_seconds": (profile or {}).get("elapsed_seconds"),
                           "sampled_peak_driver_bytes": memory.get("mps_driver_peak_sampled_bytes"),
                           "sampled_peak_rss_bytes": memory.get("rss_peak_sampled_bytes")}
        report.setdefault("artifacts", {}).update(artifact_hashes(
            output, ["profile_samples.jsonl", "profile_summary.json"]))
        report["free_disk_after_bytes"] = shutil.disk_usage(ROOT).free
        report["artifact_bytes"] = sum(path.stat().st_size
                                       for path in output.rglob("*")
                                       if path.is_file())
        report["ended_at_utc"] = utc_now()
        if report.get("status") == "failed":
            dump(output / "failure_report.json", report)
        elif report.get("status") == "completed":
            dump(output / ("restart_report.json" if restart else "run_report.json"), report)
    print(json.dumps({"status": "completed", "stage": protocol["stage"],
                      "report": str(output / ("restart_report.json" if restart else "run_report.json")),
                      "checks": report.get("checks")}))


if __name__ == "__main__":
    main()

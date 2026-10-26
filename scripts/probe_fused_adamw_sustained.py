#!/usr/bin/env python3
"""Guarded native fused-AdamW resume smoke and sustained full-scope probe."""
from __future__ import annotations

import argparse
import hashlib
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
PROTOCOL_PATH = ROOT / "configs/fused_adamw_sustained_probe_v1.json"


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _protocol_path(args):
    value = getattr(args, "protocol", None)
    if value is None:
        return PROTOCOL_PATH
    path = Path(value)
    return path if path.is_absolute() else ROOT / path

def _synchronize(torch):
    torch.mps.synchronize()


def _summary(records):
    def training_seconds(row):
        return row.get("training_seconds", row["total_seconds"])

    seconds = sum(training_seconds(row) for row in records)
    processed = sum(row["processed_tokens"] for row in records)
    supervised = sum(row["supervised_tokens"] for row in records)
    phases = ("forward_backward_seconds", "gradient_audit_seconds",
              "gradient_clip_seconds", "optimizer_seconds")
    return {
        "updates": len(records),
        "update_indices": [row["step"] for row in records],
        "seconds": seconds,
        "updates_per_second": len(records) / seconds if seconds else None,
        "processed_tokens": processed,
        "supervised_tokens": supervised,
        "processed_tokens_per_second": processed / seconds if seconds else None,
        "supervised_tokens_per_second": supervised / seconds if seconds else None,
        "instrumentation_seconds": sum(row.get("instrumentation_seconds", 0.0)
                                       for row in records),
        "mean_seconds_by_phase": {
            name: sum(row[name] for row in records) / len(records) if records else None
            for name in phases
        },
    }


def _output_fingerprint(model, encoded, torch):
    """Hash one fixed final forward without retaining autograd history."""
    model.eval()
    batch = {key: value.to("mps") for key, value in encoded[0].items()}
    try:
        with torch.no_grad():
            result = model(**batch)
            logits = result.logits.detach().cpu().contiguous()
            digest = hashlib.sha256(logits.numpy().tobytes(order="C")).hexdigest()
            loss = float(result.loss.detach()) if result.loss is not None else None
        return {"example_index": 0, "logits_shape": list(logits.shape),
                "logits_dtype": str(logits.dtype), "logits_sha256": digest,
                "loss": loss, "source": "fixed encoded train row 0 after final update"}
    finally:
        model.release_context()


def _require_full_gradient_audit(audit, expected_tensors):
    original = audit["original"]
    if (original["nonzero_gradient_tensors"] != expected_tensors
            or original["missing_gradient_names"]):
        raise RuntimeError("Full-scope gradient audit failed")


def _require_boundary_match(comparison, label):
    if not comparison.get("matched"):
        raise RuntimeError(
            f"{label} boundary fingerprint mismatch: "
            f"{json.dumps(comparison.get('earliest_divergence'), sort_keys=True)}")


def _boundary_state(step, plan_sha, update_reports, base_rates):
    return {
        "data_state": {"next_step": step, "batch_plan_sha256": plan_sha,
                       "update_reports": update_reports},
        "schedule_state": {"completed_steps": step, "base_rates": base_rates},
    }


def execute(args, profiler):
    if os.environ.get("REPLICA_CAPACITY_GUARD") != "1":
        raise RuntimeError("Use the capacity guard")
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0":
        raise RuntimeError("Explicit CPU fallback=0 is required")
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        if os.environ.get(key) != "1":
            raise RuntimeError(f"{key}=1 is required")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from replica_lab import continuation_fixture
    from replica_lab.checkpointing import save_training_checkpoint, validate_training_checkpoint
    from replica_lab.model_adapter import LlamaMemoryAdapter, validate_model_lock
    from replica_lab.optimizers import build_optimizer
    from replica_lab.training_state import (
        capture_training_fingerprint,
        compare_training_fingerprints,
        load_runtime_state,
        summarize_gradient_norms,
        summarize_optimizer_norms,
        summarize_parameter_norms,
    )
    from replica_lab.training_tools import apply_learning_rates, response_statistics
    from profile_full_training import assert_full_scope, parameter_hashes
    from train_text import encode_row, load_rows

    protocol_path = _protocol_path(args)
    protocol = json.loads(protocol_path.read_text())
    numeric = protocol.get("numeric_policy", {})
    if numeric.get("torch_use_deterministic_algorithms") is True:
        torch.use_deterministic_algorithms(True)
    if numeric.get("mps_fast_math") == "disabled":
        if os.environ.get("PYTORCH_MPS_FAST_MATH") != "0":
            raise RuntimeError("PYTORCH_MPS_FAST_MATH=0 is required by the protocol")
    mode = args.mode
    continuous = bool(getattr(args, "continuous", False))
    source = protocol["source"]
    scope = protocol["scope"]
    optim = protocol["optimizer"]
    fixture = (continuation_fixture.validate_fixture_protocol(
        protocol, protocol_path=protocol_path, root=ROOT)
        if continuation_fixture.is_fixture_protocol(protocol) else None)
    data_path = ROOT / source["data"]
    if _sha(data_path) != source["data_sha256"]:
        raise ValueError("Sealed diagnostic data bytes differ")
    if fixture is None:
        model_path = ROOT / source["model_path"]
        lock_path = ROOT / source["model_lock"]
        if _sha(lock_path) != source["model_lock_sha256"]:
            raise ValueError("Sealed model lock bytes differ")
        lock = json.loads(lock_path.read_text())
        validate_model_lock(model_path, lock)
        if lock["revision"] != source["revision"]:
            raise ValueError("Use the original sealed MiniCPM5 source")
        tokenizer_source = model_path
        source_revision = lock["revision"]
        production_scope = True
    else:
        model_path = None
        lock = {"fixture_config_sha256": fixture["config_sha256"],
                "tokenizer_lock_sha256": fixture["tokenizer_lock_sha256"]}
        tokenizer_source = ROOT / fixture["tokenizer_path"]
        source_revision = fixture["tokenizer_lock_sha256"]
        production_scope = False
    implementation = {name: _sha(ROOT / name) for name in protocol["implementation_paths"]}
    total_steps = int(protocol[mode]["total_updates"])
    comparison = protocol.get("continuation_comparison", {})
    if continuous and (mode != "smoke" or not comparison.get("enabled")):
        raise ValueError("Continuous comparison requires the declared smoke comparison protocol")
    if continuous and (args.resume_from is not None or args.resume_report is not None):
        raise ValueError("Continuous comparison cannot resume a checkpoint")
    end_step = (total_steps if continuous else
                1 if mode == "smoke" and args.resume_from is None else total_steps)
    if mode == "sustained" and (args.resume_from is not None or args.resume_report is not None):
        raise ValueError("Sustained mode is a fresh uninterrupted probe")
    if mode == "smoke" and ((args.resume_from is None) != (args.resume_report is None)):
        raise ValueError("Smoke resume requires both checkpoint and first-process report")

    random.seed(optim["seed"])
    torch.manual_seed(optim["seed"])
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source, local_files_only=True, trust_remote_code=False)
    rows = load_rows(data_path, "train")
    if len(rows) != source["examples"] or len({row["response"] for row in rows}) != source["distinct_responses"]:
        raise ValueError("Diagnostic example inventory differs")
    encoded = [encode_row(row, tokenizer, scope["tokenization_max_length"]) for row in rows]
    if {batch["input_ids"].shape[1] for batch in encoded} != {scope["sequence_length"]}:
        raise ValueError("Diagnostic sequence length differs")
    batch_plan = [list(range(len(encoded))) for _ in range(total_steps)]
    plan_sha = hashlib.sha256(json.dumps(batch_plan).encode()).hexdigest()
    contract = {
        "format_version": 1,
        "source_lock_sha256": (fixture["config_sha256"] if fixture is not None
                               else source["model_lock_sha256"]),
        "data_sha256": source["data_sha256"],
        "batch_plan_sha256": plan_sha,
        "settings": {
            "mode": mode, "steps": total_steps, "scope": scope, "optimizer": optim,
        },
        "packages": {name: importlib.metadata.version(name)
                     for name in ("torch", "transformers", "safetensors")},
        "implementation_sha256": implementation,
    }
    if fixture is not None:
        contract["fixture_config_sha256"] = fixture["config_sha256"]
        contract["tokenizer_lock_sha256"] = fixture["tokenizer_lock_sha256"]
    resume_receipt = (validate_training_checkpoint(args.resume_from, contract)
                      if args.resume_from else None)
    resume_report = (json.loads(Path(args.resume_report).read_text())
                     if args.resume_report else None)
    if resume_report is not None:
        if (resume_report.get("protocol_sha256") != _sha(protocol_path)
                or resume_report.get("mode") != "smoke"
                or resume_report.get("steps") != 1):
            raise ValueError("Resume report is not the matching first smoke process")

    base_path = (Path(args.resume_from) / "base" if args.resume_from
                 else model_path)
    resume_accounting = {}
    boundary_fingerprint = None
    boundary_comparison = None
    if args.resume_from:
        from replica_lab.training_state import materialize_resume_model
        base, model, optimizer, restored, base_rates = materialize_resume_model(
            args.resume_from / "base",
            adapter_factory=lambda base: LlamaMemoryAdapter(base, "full"),
            optimizer_factory=lambda model: build_optimizer(
                model, "adamw_fused", lr_base=optim["lr"], lr_new=optim["lr"],
                weight_decay=optim["weight_decay"]),
            runtime_path=args.resume_from / "runtime.pt",
            run_contract=contract, device="mps", accounting=resume_accounting)
    else:
        if fixture is not None:
            base = continuation_fixture.build_fixture_model(
                fixture["config"], seed=fixture["init_seed"])
        else:
            base = AutoModelForCausalLM.from_pretrained(
                base_path, local_files_only=True, trust_remote_code=False,
                dtype=torch.float32, attn_implementation="eager", use_safetensors=True)
        model = LlamaMemoryAdapter(base, "full").to("mps").train()
        optimizer = build_optimizer(
            model, "adamw_fused", lr_base=optim["lr"], lr_new=optim["lr"],
            weight_decay=optim["weight_decay"])
        base_rates = [group["lr"] for group in optimizer.param_groups]
    model.train()
    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if fixture is None:
        assert_full_scope(model, optimizer, scope["trainable_parameters"],
                          scope["trainable_tensors"])
    else:
        continuation_fixture.assert_fixture_scope(model, optimizer, fixture)
    for group in optimizer.param_groups:
        if group["betas"] != tuple(optim["betas"]) or group["eps"] != optim["eps"]:
            raise ValueError("Native optimizer hyperparameters differ")
    before = parameter_hashes(model)
    if resume_report is not None and before != resume_report["parameter_hashes_after"]:
        raise ValueError("Resume model weights differ from first-process final weights")

    start_step = 0
    update_reports = []
    if args.resume_from:
        start_step = restored["step"]
        if start_step != 1 or start_step != resume_receipt["completed_steps"]:
            raise ValueError("Smoke resume must continue exactly update 1")
        if restored["schedule_state"] != {"completed_steps": 1, "base_rates": base_rates}:
            raise ValueError("Resume scheduler state differs")
        state = restored["data_state"]
        if state.get("next_step") != 1 or state.get("batch_plan_sha256") != plan_sha:
            raise ValueError("Resume data cursor differs")
        update_reports = state.get("update_reports", [])
        if len(update_reports) != 1:
            raise ValueError("Resume update history is incomplete")
        boundary_state = _boundary_state(1, plan_sha, update_reports, base_rates)
        boundary_fingerprint = capture_training_fingerprint(
            model, optimizer, schedule_state=boundary_state["schedule_state"],
            data_state=boundary_state["data_state"])
        reference_boundary = (resume_report or {}).get("boundary_fingerprint")
        if reference_boundary is None:
            if fixture is not None:
                raise RuntimeError(
                    "Fixture resume requires a first-process boundary fingerprint")
            boundary_comparison = {
                "format": "replica.training_fingerprint_comparison.v1",
                "status": "unavailable",
                "reason": "first-process report has no boundary fingerprint",
            }
        else:
            boundary_comparison = compare_training_fingerprints(
                reference_boundary, boundary_fingerprint)
            boundary_comparison["status"] = (
                "matched" if boundary_comparison["matched"] else "mismatched")
            _require_boundary_match(boundary_comparison, "resume")
        profiler.mark("runtime_restored", completed_steps=1)

    profiler.mark("model_optimizer_loaded", start_step=start_step)
    started = time.monotonic()
    process_records = []
    checkpoint_record = None
    for step in range(start_step, end_step):
        optimizer.zero_grad(set_to_none=True)
        lr_factor = apply_learning_rates(
            optimizer, base_rates, step, total_steps, optim["warmup_updates"],
            schedule="constant", min_lr_ratio=1.0)
        _synchronize(torch)
        update_started = time.monotonic()
        supervised = sum(int((batch["labels"][:, 1:] != -100).sum()) for batch in encoded)
        loss_sum = 0.0
        for batch in encoded:
            count = int((batch["labels"][:, 1:] != -100).sum())
            batch = {key: value.to("mps") for key, value in batch.items()}
            result = model(**batch)
            loss = result.loss * (count / supervised)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("Nonfinite training loss")
            response_statistics(result.logits, batch["labels"])
            loss.backward()
            loss_sum += float(loss.detach())
            model.release_context()
        _synchronize(torch)
        backward_done = time.monotonic()
        gradient_instrumentation_started = time.monotonic()
        gradient_norms = summarize_gradient_norms(model)
        _synchronize(torch)
        backward_at = time.monotonic()
        gradient_instrumentation_seconds = (
            backward_at - gradient_instrumentation_started)
        gradient_audit = None
        if step < start_step + 2:
            gradient_audit = model.gradient_audit()
            _require_full_gradient_audit(gradient_audit, scope["trainable_tensors"])
        _synchronize(torch)
        audit_at = time.monotonic()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), scope["max_grad_norm"], error_if_nonfinite=True)
        _synchronize(torch)
        clip_at = time.monotonic()
        optimizer.step()
        _synchronize(torch)
        ended = time.monotonic()
        post_step_instrumentation_started = time.monotonic()
        parameter_norms = summarize_parameter_norms(model)
        optimizer_moment_norms = summarize_optimizer_norms(optimizer, model)
        post_step_instrumentation_seconds = (
            time.monotonic() - post_step_instrumentation_started)
        instrumentation_seconds = (
            gradient_instrumentation_seconds + post_step_instrumentation_seconds)
        record = {
            "step": step + 1,
            "lr_factor": lr_factor,
            "learning_rates": [group["lr"] for group in optimizer.param_groups],
            "loss": loss_sum,
            "forward_backward_seconds": backward_done - update_started,
            "gradient_audit_seconds": audit_at - backward_at,
            "gradient_clip_seconds": clip_at - audit_at,
            "optimizer_seconds": ended - clip_at,
            "total_seconds": ended - update_started,
            "training_seconds": (ended - update_started) - gradient_instrumentation_seconds,
            "instrumentation_seconds": instrumentation_seconds,
            "gradient_norm_instrumentation_seconds": gradient_instrumentation_seconds,
            "post_step_instrumentation_seconds": post_step_instrumentation_seconds,
            "processed_tokens": sum(batch["input_ids"].numel() for batch in encoded),
            "supervised_tokens": supervised,
            "gradient_norm_before_clipping": float(grad_norm),
            "gradient_norms_pre_clip": gradient_norms,
            "parameter_norms": parameter_norms,
            "optimizer_moment_norms": optimizer_moment_norms,
            "gradient_audit": gradient_audit,
        }
        update_reports.append(record)
        process_records.append(record)
        optimizer.zero_grad(set_to_none=True)
        print(json.dumps({"mode": mode, "step": step + 1, "loss": loss_sum,
                          "seconds": record["total_seconds"]}), flush=True)
        if mode == "smoke" and not continuous and args.resume_from is None and step + 1 == 1:
            profiler.mark("resume_checkpoint_start", step=1)
            checkpoint_path = args.output / "resume_checkpoint"
            boundary_state = _boundary_state(1, plan_sha, update_reports, base_rates)
            receipt = save_training_checkpoint(
                model, tokenizer, optimizer, checkpoint_path, step=1,
                data_state=boundary_state["data_state"],
                schedule_state=boundary_state["schedule_state"],
                run_contract=contract,
                manifest={"model_lock": lock, "run_contract": contract})
            boundary_fingerprint = capture_training_fingerprint(
                model, optimizer, schedule_state=boundary_state["schedule_state"],
                data_state=boundary_state["data_state"])
            checkpoint_record = {
                **receipt,
                "bytes": sum(path.stat().st_size for path in checkpoint_path.rglob("*")
                             if path.is_file()),
                "final_model_export": False,
            }
            profiler.mark("resume_checkpoint_complete", step=1)
        max_seconds = protocol[mode].get("max_seconds", protocol[mode].get("max_seconds_per_process"))
        if time.monotonic() - started > max_seconds:
            raise RuntimeError("Probe exceeded its declared wall budget")

    final_schedule_state = {"completed_steps": end_step, "base_rates": base_rates}
    final_data_state = {"next_step": end_step, "batch_plan_sha256": plan_sha,
                        "update_reports": update_reports}
    if (boundary_fingerprint is not None and not continuous
            and args.resume_from is None and end_step == 1):
        final_fingerprint = boundary_fingerprint
    else:
        final_fingerprint = capture_training_fingerprint(
            model, optimizer, schedule_state=final_schedule_state,
            data_state=final_data_state)
    output_receipt = (_output_fingerprint(model, encoded, torch)
                      if comparison.get("enabled") and mode == "smoke" else None)
    after = parameter_hashes(model)
    changed = [name for name in before if before[name] != after[name]]
    if len(changed) != scope["trainable_tensors"]:
        raise RuntimeError("Not every original parameter changed")
    optimizer_audit = optimizer.audit()
    step_values = sorted({float(state["step"]) for state in optimizer.state.values()})
    if step_values != [float(end_step)]:
        raise RuntimeError("Native optimizer state step differs from completed updates")
    optimizer.zero_grad(set_to_none=True)
    profiler.mark("changed_tensor_hashes_verified", changed_tensors=len(changed))
    process_seconds = sum(row.get("training_seconds", row["total_seconds"])
                          for row in process_records)
    return {
        "format": "replica.fused_adamw_sustained_probe.run.v1",
        "status": "completed",
        "mode": mode,
        "execution_mode": ("continuous" if continuous else
                           "resume" if args.resume_from else "first"),
        "protocol_sha256": _sha(protocol_path),
        "implementation_sha256": implementation,
        "source_revision": source_revision,
        "source_kind": "fixture" if fixture is not None else "sealed_minicpm5",
        "production_scope": production_scope,
        "protocol_policy_sha256": (
            fixture["protocol_policy_sha256"] if fixture is not None else None),
        "expected_tensor_count": scope["trainable_tensors"],
        "fixture": (None if fixture is None else {
            "format": fixture["format"],
            "config_sha256": fixture["config_sha256"],
            "init_seed": fixture["init_seed"],
            "tensor_count": fixture["tensor_count"],
            "parameter_count": fixture["parameter_count"],
            "tokenizer_path": fixture["tokenizer_path"],
            "tokenizer_lock_sha256": fixture["tokenizer_lock_sha256"],
            "claim_boundary": fixture["claim_boundary"],
        }),
        "claim_boundary": (fixture["claim_boundary"] if fixture is not None
                           else protocol.get("claim_boundary")),
        "data_sha256": source["data_sha256"],
        "device": "mps",
        "cpu_fallback_enabled": False,
        "parameter_dtype": "float32",
        "compute_dtype": "float32",
        "trainable_tensors": scope["trainable_tensors"],
        "trainable_parameters": scope["trainable_parameters"],
        "effective_batch": scope["effective_batch"],
        "physical_microbatch": scope["physical_microbatch"],
        "sequence_length": scope["sequence_length"],
        "activation_checkpointing": True,
        "attention": "eager",
        "optimizer": optim,
        "optimizer_audit": optimizer_audit,
        "optimizer_state_step_values": step_values,
        "steps": end_step,
        "start_step": start_step,
        "updates": update_reports,
        "process_updates": process_records,
        "process_training_seconds": process_seconds,
        "instrumentation": {
            "format": "replica.probe_instrumentation.v1",
            "per_update": [
                "parameter_norms", "gradient_norms_pre_clip",
                "optimizer_moment_norms", "training_seconds",
            ],
            "boundary": "replica.training_fingerprint.v1",
            "gradient_norm_source": "direct pre-clipping gradient tensors",
            "fingerprint_tensor_scope": "value bytes, dtype, device, shape, storage, norms",
            "performance_excludes_instrumentation": True,
        },
        "boundary_fingerprint": boundary_fingerprint,
        "boundary_comparison": boundary_comparison,
        "final_fingerprint": final_fingerprint,
        "resume_accounting": resume_accounting or None,
        "performance": {
            "process": _summary(process_records),
            "warmup": _summary(process_records[:2]),
            "steady": _summary(process_records[2:]),
        },
        "changed_tensors": len(changed),
        "parameter_hashes_before": before,
        "parameter_hashes_after": after,
        "output_fingerprint": output_receipt,
        "checkpoint": checkpoint_record,
        "checkpoint_reload_validated": resume_receipt is not None,
        "resume_start_matches_first_process": resume_report is not None,
        "model_exported": False,
        "selected_checkpoint_changed": False,
        "limitations": protocol["limitations"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "sustained"), required=True)
    parser.add_argument("--device", choices=("mps",), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL_PATH)
    parser.add_argument("--continuous", action="store_true",
                        help="Run the declared uninterrupted two-update comparison arm")
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--resume-report", type=Path)
    args = parser.parse_args()
    from replica_lab import continuation_fixture
    try:
        continuation_fixture.require_fresh_output(args.output, what="probe output")
    except FileExistsError as error:
        parser.error(str(error))
    args.output.mkdir(parents=True, exist_ok=True)
    from replica_lab.architecture.loading import exclusive_mps_worker
    from replica_lab.profiling import RunProfiler
    with exclusive_mps_worker(ROOT):
        profiler = RunProfiler("mps", args.output)
        profiler.start()
        try:
            report = execute(args, profiler)
        except BaseException as error:
            failure = {"status": "failed",
                       "failure_class": continuation_fixture.classify_failure(error),
                       "error_type": type(error).__name__,
                       "error": str(error), "traceback": traceback.format_exc(),
                       "selected_checkpoint_changed": False,
                       "profile": profiler.stop()}
            try:
                failure_protocol_path = _protocol_path(args)
                failure_protocol = json.loads(failure_protocol_path.read_text())
                failure["protocol_sha256"] = _sha(failure_protocol_path)
                if continuation_fixture.is_fixture_protocol(failure_protocol):
                    failure["production_scope"] = False
                    failure["claim_boundary"] = failure_protocol["fixture"][
                        "claim_boundary"]
                    failure["failure_classes"] = list(
                        continuation_fixture.FAILURE_CLASSES)
            except (OSError, ValueError, KeyError, json.JSONDecodeError):
                pass
            (args.output / "failure_report.json").write_text(
                json.dumps(failure, indent=2) + "\n")
            raise
        report["profile"] = profiler.stop()
        (args.output / "run_report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"mode": args.mode, "steps": report["steps"],
                          "changed_tensors": report["changed_tensors"],
                          "report": str(args.output / "run_report.json")}))


if __name__ == "__main__":
    main()

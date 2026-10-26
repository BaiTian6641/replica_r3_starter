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


def _synchronize(torch):
    torch.mps.synchronize()


def _summary(records):
    seconds = sum(row["total_seconds"] for row in records)
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
        "mean_seconds_by_phase": {
            name: sum(row[name] for row in records) / len(records) if records else None
            for name in phases
        },
    }


def _require_full_gradient_audit(audit, expected_tensors):
    original = audit["original"]
    if (original["nonzero_gradient_tensors"] != expected_tensors
            or original["missing_gradient_names"]):
        raise RuntimeError("Full-scope gradient audit failed")


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
    from replica_lab.checkpointing import save_training_checkpoint, validate_training_checkpoint
    from replica_lab.model_adapter import LlamaMemoryAdapter, validate_model_lock
    from replica_lab.optimizers import build_optimizer
    from replica_lab.training_state import load_runtime_state
    from replica_lab.training_tools import apply_learning_rates, response_statistics
    from profile_full_training import assert_full_scope, parameter_hashes
    from train_text import encode_row, load_rows

    protocol = json.loads(PROTOCOL_PATH.read_text())
    mode = args.mode
    source = protocol["source"]
    scope = protocol["scope"]
    optim = protocol["optimizer"]
    model_path = ROOT / source["model_path"]
    lock_path = ROOT / source["model_lock"]
    data_path = ROOT / source["data"]
    if _sha(lock_path) != source["model_lock_sha256"] or _sha(data_path) != source["data_sha256"]:
        raise ValueError("Sealed model lock or diagnostic data bytes differ")
    lock = json.loads(lock_path.read_text())
    validate_model_lock(model_path, lock)
    if lock["revision"] != source["revision"]:
        raise ValueError("Use the original sealed MiniCPM5 source")
    implementation = {name: _sha(ROOT / name) for name in protocol["implementation_paths"]}
    total_steps = int(protocol[mode]["total_updates"])
    end_step = 1 if mode == "smoke" and args.resume_from is None else total_steps
    if mode == "sustained" and (args.resume_from is not None or args.resume_report is not None):
        raise ValueError("Sustained mode is a fresh uninterrupted probe")
    if mode == "smoke" and ((args.resume_from is None) != (args.resume_report is None)):
        raise ValueError("Smoke resume requires both checkpoint and first-process report")

    random.seed(optim["seed"])
    torch.manual_seed(optim["seed"])
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=False)
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
        "source_lock_sha256": source["model_lock_sha256"],
        "data_sha256": source["data_sha256"],
        "batch_plan_sha256": plan_sha,
        "settings": {
            "mode": mode, "steps": total_steps, "scope": scope, "optimizer": optim,
        },
        "packages": {name: importlib.metadata.version(name)
                     for name in ("torch", "transformers", "safetensors")},
        "implementation_sha256": implementation,
    }
    resume_receipt = (validate_training_checkpoint(args.resume_from, contract)
                      if args.resume_from else None)
    resume_report = (json.loads(Path(args.resume_report).read_text())
                     if args.resume_report else None)
    if resume_report is not None:
        if (resume_report.get("protocol_sha256") != _sha(PROTOCOL_PATH)
                or resume_report.get("mode") != "smoke"
                or resume_report.get("steps") != 1):
            raise ValueError("Resume report is not the matching first smoke process")

    base_path = Path(args.resume_from) / "base" if args.resume_from else model_path
    base = AutoModelForCausalLM.from_pretrained(
        base_path, local_files_only=True, trust_remote_code=False,
        dtype=torch.float32, attn_implementation="eager", use_safetensors=True)
    model = LlamaMemoryAdapter(base, "full").to("mps").train()
    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    optimizer = build_optimizer(
        model, "adamw_fused", lr_base=optim["lr"], lr_new=optim["lr"],
        weight_decay=optim["weight_decay"])
    assert_full_scope(model, optimizer, scope["trainable_parameters"],
                      scope["trainable_tensors"])
    for group in optimizer.param_groups:
        if group["betas"] != tuple(optim["betas"]) or group["eps"] != optim["eps"]:
            raise ValueError("Native optimizer hyperparameters differ")
    base_rates = [group["lr"] for group in optimizer.param_groups]
    before = parameter_hashes(model)
    if resume_report is not None and before != resume_report["parameter_hashes_after"]:
        raise ValueError("Resume model weights differ from first-process final weights")

    start_step = 0
    update_reports = []
    if args.resume_from:
        model.to("cpu")
        gc.collect()
        torch.mps.empty_cache()
        restored = load_runtime_state(
            Path(args.resume_from) / "runtime.pt", optimizer, run_contract=contract)
        model.to("mps")
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
        backward_at = time.monotonic()
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
        record = {
            "step": step + 1,
            "lr_factor": lr_factor,
            "learning_rates": [group["lr"] for group in optimizer.param_groups],
            "loss": loss_sum,
            "forward_backward_seconds": backward_at - update_started,
            "gradient_audit_seconds": audit_at - backward_at,
            "gradient_clip_seconds": clip_at - audit_at,
            "optimizer_seconds": ended - clip_at,
            "total_seconds": ended - update_started,
            "processed_tokens": sum(batch["input_ids"].numel() for batch in encoded),
            "supervised_tokens": supervised,
            "gradient_norm_before_clipping": float(grad_norm),
            "gradient_audit": gradient_audit,
        }
        update_reports.append(record)
        process_records.append(record)
        optimizer.zero_grad(set_to_none=True)
        print(json.dumps({"mode": mode, "step": step + 1, "loss": loss_sum,
                          "seconds": record["total_seconds"]}), flush=True)
        if mode == "smoke" and args.resume_from is None and step + 1 == 1:
            profiler.mark("resume_checkpoint_start", step=1)
            checkpoint_path = args.output / "resume_checkpoint"
            receipt = save_training_checkpoint(
                model, tokenizer, optimizer, checkpoint_path, step=1,
                data_state={"next_step": 1, "batch_plan_sha256": plan_sha,
                            "update_reports": update_reports},
                schedule_state={"completed_steps": 1, "base_rates": base_rates},
                run_contract=contract,
                manifest={"model_lock": lock, "run_contract": contract})
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
    process_seconds = sum(row["total_seconds"] for row in process_records)
    return {
        "format": "replica.fused_adamw_sustained_probe.run.v1",
        "status": "completed",
        "mode": mode,
        "protocol_sha256": _sha(PROTOCOL_PATH),
        "implementation_sha256": implementation,
        "source_revision": lock["revision"],
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
        "performance": {
            "process": _summary(process_records),
            "warmup": _summary(process_records[:2]),
            "steady": _summary(process_records[2:]),
        },
        "changed_tensors": len(changed),
        "parameter_hashes_before": before,
        "parameter_hashes_after": after,
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
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--resume-report", type=Path)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Use fresh output")
    args.output.mkdir(parents=True, exist_ok=True)
    from replica_lab.architecture.loading import exclusive_mps_worker
    from replica_lab.profiling import RunProfiler
    with exclusive_mps_worker(ROOT):
        profiler = RunProfiler("mps", args.output)
        profiler.start()
        try:
            report = execute(args, profiler)
        except BaseException as error:
            failure = {"status": "failed", "error_type": type(error).__name__,
                       "error": str(error), "traceback": traceback.format_exc(),
                       "selected_checkpoint_changed": False,
                       "profile": profiler.stop()}
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

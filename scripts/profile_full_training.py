#!/usr/bin/env python3
"""Bounded full-scope MPS throughput probe; discard candidate weights.

Run only through run_capacity_preflight.py. No download, model export, selected
artifact change, dataset admission or quality comparison is performed here.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def collate_full_batches(batches, pad_token_id):
    """Right pad independent causal sequences, preserving response-only labels."""
    import torch
    width = max(batch["input_ids"].shape[1] for batch in batches)
    ids = torch.full((len(batches), width), pad_token_id, dtype=torch.long)
    mask = torch.zeros_like(ids)
    labels = torch.full_like(ids, -100)
    for index, batch in enumerate(batches):
        size = batch["input_ids"].shape[1]
        ids[index, :size] = batch["input_ids"][0]
        mask[index, :size] = batch["attention_mask"][0]
        labels[index, :size] = batch["labels"][0]
    return {"input_ids": ids, "attention_mask": mask, "labels": labels}


def assert_full_scope(model, optimizer, expected_parameters=2516756480,
                      expected_tensors=381):
    parameters = list(model.parameters())
    if (len(parameters) != expected_tensors
            or sum(parameter.numel() for parameter in parameters) != expected_parameters
            or not all(parameter.requires_grad for parameter in parameters)):
        raise ValueError("Full original parameter scope is incomplete")
    owned = [id(p) for group in optimizer.param_groups for p in group["params"]]
    if len(owned) != len(set(owned)) or set(owned) != {id(p) for p in parameters}:
        raise ValueError("Optimizer does not own every original tensor exactly once")


def parameter_hashes(model):
    # One tensor at a time, outside the timing window; no full parameter backup.
    return {name: hashlib.sha256(memoryview(parameter.detach().cpu().numpy()).cast("B")).hexdigest()
            for name, parameter in model.named_parameters()}


def execute(args, profiler):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from replica_lab.model_adapter import LlamaMemoryAdapter, validate_model_lock
    from replica_lab.optimizers import build_optimizer
    from replica_lab.training_tools import response_statistics
    from train_text import encode_row, load_rows

    if os.environ.get("REPLICA_CAPACITY_GUARD") != "1":
        raise RuntimeError("Use the capacity guard")
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0":
        raise RuntimeError("Explicit CPU fallback=0 is required")
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        if os.environ.get(key) != "1":
            raise RuntimeError(f"{key}=1 is required")
    lock_path = ROOT / "configs/minicpm5_sealed_lock.local.json"
    model_path = ROOT / "checkpoints/minicpm5_source"
    lock = json.loads(lock_path.read_text())
    validate_model_lock(model_path, lock)
    if lock["revision"] != "f97400052a43d642bbc6e9975e2397e3ae6a6b52":
        raise ValueError("Use the original sealed MiniCPM5 source")
    random.seed(17)
    torch.manual_seed(17)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True,
                                              trust_remote_code=False)
    data_path = ROOT / "data/diagnostics/tiny_overfit_mask1/train.jsonl"
    rows = load_rows(data_path, "train")
    if len(rows) != 8 or len({row["response"] for row in rows}) != 8:
        raise ValueError("Expected eight exposed balanced diagnostic rows")
    encoded = [encode_row(row, tokenizer, 256) for row in rows]
    if {batch["input_ids"].shape[1] for batch in encoded} != {55}:
        raise ValueError("Expected unchanged 55-token diagnostic")
    base = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True,
        trust_remote_code=False, dtype=torch.float32,
        attn_implementation=args.attention, use_safetensors=True)
    model = LlamaMemoryAdapter(base, "full").to("mps").train()
    if args.checkpointing:
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    parameters = list(model.parameters())
    if args.optimizer == "muon8bit":
        optimizer = build_optimizer(model, "muon8bit", lr_base=1e-4, lr_new=1e-4,
                                    weight_decay=0.0)
    else:
        optimizer = torch.optim.AdamW(parameters, lr=1e-4, betas=(0.9, 0.999),
                                      eps=1e-8, weight_decay=0.0, fused=True)
    assert_full_scope(model, optimizer)
    profiler.mark("model_optimizer_loaded")
    before = parameter_hashes(model)
    profiler.mark("initial_parameter_hashes_complete")
    updates = []
    for update in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        for group in optimizer.param_groups:
            group["lr"] = 1e-4 * min(update / 2, 1.0)
        profiler.mark("update_start", update=update)
        started = time.monotonic()
        loss_sum = 0.0
        group = [encoded[index] for index in range(8)]
        supervised = sum(int((batch["labels"][:, 1:] != -100).sum()) for batch in group)
        for start in range(0, 8, args.microbatch):
            batch = collate_full_batches(group[start:start + args.microbatch], tokenizer.pad_token_id)
            batch = {key: value.to("mps") for key, value in batch.items()}
            count = int((batch["labels"][:, 1:] != -100).sum())
            result = model(**batch)
            loss = result.loss * (count / supervised)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("Nonfinite training loss")
            # Match the reference's response/EOS diagnostics and synchronization.
            response_statistics(result.logits, batch["labels"])
            loss.backward()
            loss_sum += float(loss.detach())
            model.release_context()
        profiler.mark("backward_complete", update=update)
        backward_at = time.monotonic()
        audit = model.gradient_audit()
        if audit["original"]["nonzero_gradient_tensors"] != 381:
            raise RuntimeError("Full-scope backward did not reach all original tensors")
        audit_at = time.monotonic()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
        profiler.mark("gradient_checks_complete", update=update)
        checks_at = time.monotonic()
        optimizer.step()
        profiler.mark("optimizer_step_complete", update=update)
        ended = time.monotonic()
        record = {"update": update, "forward_backward_seconds": backward_at - started,
                  "gradient_checks_seconds": checks_at - backward_at,
                  "gradient_audit_seconds": audit_at - backward_at,
                  "gradient_clip_and_mark_seconds": checks_at - audit_at,
                  "optimizer_seconds": ended - checks_at,
                  "total_seconds": ended - started, "loss": loss_sum,
                  "original_nonzero_gradient_tensors": 381,
                  "processed_tokens": 440, "supervised_tokens": supervised}
        updates.append(record)
        print(json.dumps(record), flush=True)
        if ended - args.started_monotonic > 180:
            raise RuntimeError("Probe exceeded the declared 180-second budget")
    optimizer.zero_grad(set_to_none=True)
    after = parameter_hashes(model)
    changed = [name for name in before if before[name] != after[name]]
    if len(changed) != 381:
        raise RuntimeError("Not all 381 original parameter tensors changed")
    profiler.mark("changed_tensor_hashes_verified", changed_tensors=len(changed))
    total = sum(row["total_seconds"] for row in updates)
    return {"format": "replica.full_training_throughput_probe.v1", "status": "completed",
            "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "protocol_sha256": hashlib.sha256((ROOT / "configs/full_training_throughput_probe_v1.json").read_bytes()).hexdigest(),
            "implementation_sha256": {
                name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                for name in ("scripts/profile_full_training.py", "scripts/train_text.py",
                             "src/replica_lab/model_adapter.py", "src/replica_lab/optimizers.py",
                             "src/replica_lab/training_tools.py", "src/replica_lab/profiling.py")},
            "source_revision": lock["revision"], "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
            "optimizer": args.optimizer, "attention": args.attention,
            "checkpointing": args.checkpointing, "microbatch": args.microbatch,
            "effective_batch": 8, "sequence_length": 55, "device": "mps",
            "parameter_dtype": "float32", "compute_dtype": "float32",
            "torch_version": torch.__version__,
            "mps_environment": {key: os.environ.get(key) for key in (
                "PYTORCH_MPS_FAST_MATH", "PYTORCH_MPS_PREFER_METAL", "PYTORCH_ENABLE_MPS_FALLBACK")},
            "trainable_parameters": 2516756480, "trainable_tensors": 381,
            "changed_tensors": len(changed), "parameter_hashes_before": before,
            "parameter_hashes_after": after, "updates": updates,
            "training_seconds": total, "optimizer_updates_per_second": len(updates) / total,
            "processed_tokens_per_second": len(updates) * 440 / total,
            "model_exported": False, "selected_checkpoint_changed": False,
            "limitations": ["Three-update throughput diagnostic on an exposed 55-token fixture.",
                            "Not a convergence, long-context or quality-equivalence result.",
                            "AdamW changes the optimizer algorithm, not parameter scope.",
                            "Gradient audits run every update; the reference audits its first two."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("mps",), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--optimizer", choices=("muon8bit", "adamw-fused"), required=True)
    parser.add_argument("--steps", type=int, default=3, choices=(3,))
    parser.add_argument("--microbatch", type=int, default=1, choices=(1, 2, 4, 8))
    parser.add_argument("--attention", choices=("eager", "sdpa"), default="eager")
    parser.add_argument("--checkpointing", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    protocol = json.loads((ROOT / "configs/full_training_throughput_probe_v1.json").read_text())
    if args.steps != protocol["scope"]["updates_per_arm"] or not any(
            all(getattr(args, key) == arm[key]
                for key in ("optimizer", "microbatch", "attention", "checkpointing"))
            for arm in protocol["arms"]):
        parser.error("Use a declared three-update protocol arm")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Use fresh output")
    # Own the existing native-worker lock without importing a native adapter.
    fd = os.open(ROOT / "runs/.D2_mps_worker.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        from replica_lab.profiling import RunProfiler
        args.output.mkdir(parents=True, exist_ok=True)
        args.started_monotonic = time.monotonic()
        with RunProfiler("mps", args.output) as profiler:
            try:
                report = execute(args, profiler)
            except BaseException as error:
                (args.output / "failure_report.json").write_text(json.dumps({
                    "error_type": type(error).__name__, "error": str(error),
                    "traceback": traceback.format_exc(), "selected_checkpoint_changed": False,
                }, indent=2) + "\n")
                raise
        report["profile"] = profiler.stop()
        (args.output / "run_report.json").write_text(json.dumps(report, indent=2) + "\n")
    finally:
        os.close(fd)


if __name__ == "__main__":
    main()

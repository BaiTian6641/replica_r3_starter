#!/usr/bin/env python3
"""Local, single-device MiniCPM5-2B text-training reference.

Requires an independently acquired/pinned native checkpoint and verified JSONL.
No downloads, inference device maps, automatic offload, VLM support, autonomous
promotion, hidden benchmark access or GPU-fit guarantees are implemented here.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import gc
import hashlib
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


def load_rows(path: Path, split: str):
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("split") != split or row.get("verified") is not True:
            raise ValueError(f"{path.name}:{line_number}: requires verified=true and split={split}")
        for key in ("id", "group_id", "prompt", "response"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f"{path.name}:{line_number}: missing/non-string {key}")
        for key in ("memory_texts", "source_ids"):
            if not isinstance(row.get(key), list) or any(not isinstance(x, str) for x in row[key]):
                raise ValueError(f"{path.name}:{line_number}: {key} must be a string list")
        if row["memory_texts"] and not row["source_ids"]:
            raise ValueError("Memory records must retain source IDs")
        rows.append(row)
    if not rows:
        raise ValueError(f"{path} is empty")
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError(f"Duplicate example IDs in {path}")
    return rows


def encode_row(row, tokenizer, max_length):
    """Prompt/answer tokenization is separate and the final EOS is supervised.

    Deliberately reject overlong rows instead of silently removing evidence.
    Every arm sees the identical memory text in the native chat prompt.
    """
    import torch
    memory = "\n".join(row["memory_texts"])
    content = ("Observed source records:\n" + (memory or "[none]")
               + "\n\nTask:\n" + row["prompt"])
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False)
    prompt = tokenizer.encode(rendered, add_special_tokens=False)
    response = tokenizer.encode(row["response"], add_special_tokens=False)
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer must define an explicit EOS token")
    response.append(tokenizer.eos_token_id)
    if not prompt or len(prompt) + len(response) > max_length:
        raise ValueError(f"Row {row['id']} exceeds sequence budget; revise data/budget explicitly")
    support = tokenizer.encode(memory, add_special_tokens=False) if memory else []
    batch = {
        "input_ids": torch.tensor([prompt + response], dtype=torch.long),
        "attention_mask": torch.ones(1, len(prompt) + len(response), dtype=torch.long),
        "labels": torch.tensor([[-100] * len(prompt) + response], dtype=torch.long),
        "prompt_ids": torch.tensor([prompt], dtype=torch.long),
        "prompt_mask": torch.ones(1, len(prompt), dtype=torch.bool),
        "memory_ids": torch.tensor([support], dtype=torch.long),
        "memory_mask": torch.ones(1, len(support), dtype=torch.bool),
    }
    return batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-lock", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--regime", choices=("baseline", "full", "modules_only", "full_augmented"), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="mps" if sys.platform == "darwin" else "cuda")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32",
                        help="Compute dtype; resident parameters remain FP32")
    parser.add_argument("--optimizer", choices=("muon8bit", "adamw"), default="muon8bit")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--lr-base", type=float, default=1e-5)
    parser.add_argument("--lr-new", type=float, default=1e-4)
    parser.add_argument("--lr-muon-base", type=float, default=1e-3,
                        help="Separate provisional Muon LR; retune before quality comparisons")
    parser.add_argument("--lr-muon-new", type=float, default=1e-3)
    parser.add_argument("--momentum", type=float, default=0.95)
    parser.add_argument("--ns-steps", type=int, default=5)
    parser.add_argument("--block-size", type=int, default=2048)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--memory-size", type=int, default=64)
    parser.add_argument("--workspace-size", type=int, default=64)
    parser.add_argument("--insertion-layer", type=int, default=-1)
    parser.add_argument("--checkpointing", action="store_true")
    parser.add_argument("--verify-reload", action="store_true",
                        help="Reload the exported model after releasing training state and compare logits")
    args = parser.parse_args()
    if args.steps < 0 or args.grad_accum < 1 or args.max_length < 8:
        parser.error("Invalid step, accumulation or sequence budget")
    if args.regime == "baseline" and args.steps:
        parser.error("baseline is evaluation-only; specify --steps 0")
    if args.regime != "baseline" and args.steps == 0:
        parser.error("Trainable regimes require at least one update")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Use a new empty output directory to preserve earlier runs")
    if args.device == "mps" and os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
        parser.error("Unset PYTORCH_ENABLE_MPS_FALLBACK to keep MPS provenance explicit")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from replica_lab.model_adapter import LlamaMemoryAdapter, validate_model_lock
    from replica_lab.contracts import audit_splits
    from replica_lab.optimizers import build_optimizer
    from replica_lab.profiling import RunProfiler

    args.output.mkdir(parents=True, exist_ok=True)
    profiler = RunProfiler(args.device, args.output)
    profiler.start()
    try:
        report = execute(args, profiler)
    except BaseException as exc:
        failure = {"status": "failed", "error_type": type(exc).__name__,
                   "error": str(exc), "traceback": traceback.format_exc(),
                   "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                   "profile": profiler.stop()}
        (args.output / "failure_report.json").write_text(json.dumps(failure, indent=2) + "\n")
        raise
    else:
        report["profile"] = profiler.stop()
        (args.output / "run_report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"report": str(args.output / "run_report.json"),
                          "dev_response_nll": report["dev_response_nll"]}))


def execute(args, profiler):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from replica_lab.model_adapter import LlamaMemoryAdapter, validate_model_lock
    from replica_lab.contracts import audit_splits
    from replica_lab.optimizers import build_optimizer

    lock = json.loads(args.model_lock.read_text())
    validate_model_lock(args.model_path, lock)
    train_rows, dev_rows = load_rows(args.data, "train"), load_rows(args.dev, "dev")
    audit_splits(train_rows + dev_rows)
    if {r["group_id"] for r in train_rows} & {r["group_id"] for r in dev_rows}:
        raise ValueError("Train/dev source-family group leakage")
    if {r["id"] for r in train_rows} & {r["id"] for r in dev_rows}:
        raise ValueError("Train/dev duplicate IDs")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; no silent backend substitution")
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable")
    if args.dtype == "bfloat16":
        if args.device == "mps":
            raise ValueError("This reference supports MPS FP32 only; validate BF16 separately")
        if args.device == "cuda":
            if not torch.cuda.is_bf16_supported(including_emulation=False):
                raise RuntimeError("Native CUDA BF16 not supported")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True,
                                               trust_remote_code=False)
    encoded_train = [encode_row(row, tokenizer, args.max_length) for row in train_rows]
    encoded_dev = [encode_row(row, tokenizer, args.max_length) for row in dev_rows]
    actual_lengths = [int(b["input_ids"].shape[1]) for b in encoded_train + encoded_dev]
    profiler.mark("data_encoded", train_examples=len(train_rows), dev_examples=len(dev_rows),
                  minimum_sequence_length=min(actual_lengths), maximum_sequence_length=max(actual_lengths))
    base = AutoModelForCausalLM.from_pretrained(
        args.model_path, local_files_only=True, trust_remote_code=False,
        torch_dtype=torch.float32, attn_implementation="eager", use_safetensors=True)
    model = LlamaMemoryAdapter(base, args.regime, memory_size=args.memory_size,
                               workspace_size=args.workspace_size,
                               insertion_layer=args.insertion_layer).to(args.device)
    profiler.mark("model_loaded", parameters=sum(p.numel() for p in model.parameters()))
    if args.checkpointing:
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    audit = model.parameter_audit()
    expected = 2516756480
    if audit["original"]["total"] != expected:
        raise RuntimeError(f"Unexpected unique parameter count: {audit['original']['total']}")
    def context():
        return (torch.autocast(args.device, dtype=torch.bfloat16)
                if args.dtype == "bfloat16" else nullcontext())
    def device_batch(batch):
        return {key: value.to(args.device) for key, value in batch.items()}
    optimizer = build_optimizer(
        model, args.optimizer, lr_base=args.lr_base, lr_new=args.lr_new,
        lr_muon_base=args.lr_muon_base, lr_muon_new=args.lr_muon_new,
        momentum=args.momentum, ns_steps=args.ns_steps,
        block_size=args.block_size, weight_decay=args.weight_decay)
    first_gradient_audit = None
    update_reports = []
    processed_tokens = 0
    supervised_tokens = 0
    started = time.monotonic()
    model.train()
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    order = list(range(len(encoded_train)))
    random.shuffle(order)
    cursor = 0
    losses = []
    for step in range(args.steps):
        profiler.mark("update_start", step=step + 1)
        update_started = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        selected = []
        selected_ids = []
        for _ in range(args.grad_accum):
            if cursor == len(order):
                random.shuffle(order)
                cursor = 0
            selected.append(encoded_train[order[cursor]])
            selected_ids.append(train_rows[order[cursor]]["id"])
            cursor += 1
        counts = [int((b["labels"][:, 1:] != -100).sum()) for b in selected]
        total = sum(counts)
        accumulated = 0.0
        for batch, tokens in zip(selected, counts):
            try:
                with context():
                    output = model(**device_batch(batch))
                    loss = output.loss * (tokens / total)
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite loss; aborting this candidate")
                loss.backward()
                accumulated += float(loss.detach())
                profiler.mark("backward_complete", step=step + 1, tokens=tokens)
            finally:
                model.release_context()
        if step == 0:
            first_gradient_audit = model.gradient_audit()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        profiler.mark("optimizer_step_complete", step=step + 1)
        update_elapsed = time.monotonic() - update_started
        step_tokens = sum(int(batch["input_ids"].numel()) for batch in selected)
        processed_tokens += step_tokens
        supervised_tokens += total
        update_reports.append({"step": step + 1, "example_ids": selected_ids,
                               "processed_tokens": step_tokens, "supervised_tokens": total,
                               "train_response_nll_before_update": accumulated,
                               "elapsed_seconds": update_elapsed,
                               "processed_tokens_per_second": step_tokens / update_elapsed})
        losses.append(accumulated)
        print(json.dumps({"step": step + 1, "train_nll": accumulated}), flush=True)
    training_elapsed = sum(x["elapsed_seconds"] for x in update_reports)
    optimizer_audit = optimizer.audit() if optimizer is not None else None
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
    model.eval()
    profiler.mark("evaluation_start")
    evaluation_started = time.monotonic()
    dev_sum = 0.0
    dev_tokens = 0
    reload_expected = None
    with torch.no_grad():
        for batch in encoded_dev:
            count = int((batch["labels"][:, 1:] != -100).sum())
            try:
                with context():
                    result = model(**device_batch(batch))
                dev_sum += float(result.loss) * count
                dev_tokens += count
                if args.verify_reload and reload_expected is None:
                    reload_expected = result.logits.detach().cpu()
            finally:
                model.release_context()
    if args.device == "cuda":
        torch.cuda.synchronize()
    elif args.device == "mps":
        torch.mps.synchronize()
    profiler.mark("evaluation_complete", dev_response_nll=dev_sum / dev_tokens)
    report = {"status": "completed_local_baseline" if args.regime == "baseline" else "completed_local_training_run", "model_lock": lock,
              "parameters": audit, "parameter_dtype": "float32", "compute_dtype": args.dtype,
              "device": args.device, "steps": args.steps, "seed": args.seed,
              "grad_accum": args.grad_accum, "max_length": args.max_length,
              "checkpointing": args.checkpointing,
              "optimizer": optimizer_audit,
              "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "actual_sequence_lengths": {"minimum": min(actual_lengths), "maximum": max(actual_lengths)},
              "train_updates": update_reports,
              "processed_train_tokens": processed_tokens, "supervised_train_tokens": supervised_tokens,
              "dev_supervised_tokens": dev_tokens,
              "training_update_seconds": training_elapsed,
              "evaluation_seconds": time.monotonic() - evaluation_started,
              "processed_train_tokens_per_second": processed_tokens / training_elapsed if training_elapsed else None,
              "cpu_fallback_enabled": False,
              "train_examples": len(train_rows), "dev_examples": len(dev_rows),
              "data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
              "dev_sha256": hashlib.sha256(args.dev.read_bytes()).hexdigest(),
              "dev_response_nll": dev_sum / dev_tokens,
              "first_update_gradient_audit": first_gradient_audit,
              "elapsed_seconds": time.monotonic() - started,
              "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated() if args.device == "cuda" else None,
              "mps_current_allocated_bytes": torch.mps.current_allocated_memory() if args.device == "mps" else None,
              "mps_driver_allocated_bytes": torch.mps.driver_allocated_memory() if args.device == "mps" else None,
              "packages": {name: importlib.metadata.version(name)
                           for name in ("torch", "transformers", "safetensors")},
              "limitations": ["Single device, no optimizer offload or distributed sharding",
                              "Dev NLL is not held-out task success or self-improvement evidence",
                              "No autoregressive cached generation integration",
                              "Weight export only; optimizer/RNG resume checkpoint is not implemented",
                              "Muon8bit is a portable project experiment, not a validated paper reproduction",
                              "This runner is text-only; it does not train a VLM"]}
    if args.regime != "baseline":
        # Release optimizer/graph/gradient buffers before serializing FP32 weights.
        del optimizer
        if args.steps:
            del output, loss
        del result
        gc.collect()
        if args.device == "mps":
            torch.mps.empty_cache()
        profiler.mark("export_start")
        model.save_pretrained(args.output / "checkpoint", manifest=report)
        tokenizer.save_pretrained(args.output / "checkpoint" / "base")
        profiler.mark("export_complete")
        report["checkpoint_exported"] = True
        if args.verify_reload:
            del model, base
            gc.collect()
            if args.device == "mps":
                torch.mps.empty_cache()
            elif args.device == "cuda":
                torch.cuda.empty_cache()
            profiler.mark("reload_start")
            reloaded_base = AutoModelForCausalLM.from_pretrained(
                args.output / "checkpoint" / "base", local_files_only=True,
                trust_remote_code=False, torch_dtype=torch.float32,
                attn_implementation="eager", use_safetensors=True)
            reloaded = LlamaMemoryAdapter(reloaded_base, args.regime,
                                         memory_size=args.memory_size, workspace_size=args.workspace_size,
                                         insertion_layer=args.insertion_layer)
            if reloaded.augmented:
                reloaded.load_sidecars(args.output / "checkpoint" / "sidecars.safetensors")
            reloaded = reloaded.to(args.device).eval()
            with torch.no_grad():
                try:
                    with context():
                        actual = reloaded(**device_batch(encoded_dev[0])).logits.detach().cpu()
                finally:
                    reloaded.release_context()
            torch.testing.assert_close(actual, reload_expected, rtol=1e-5, atol=1e-6)
            report["reload_verification"] = {
                "status": "passed", "dev_example_id": dev_rows[0]["id"],
                "compared_logit_elements": actual.numel(), "rtol": 1e-5, "atol": 1e-6,
                "maximum_absolute_logit_difference": float((actual - reload_expected).abs().max())}
            profiler.mark("reload_verified")
    else:
        report["checkpoint_exported"] = False
    return report


if __name__ == "__main__":
    main()

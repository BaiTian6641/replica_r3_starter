#!/usr/bin/env python3
"""Qualify the existing sidecar adapter on one sealed native MiniCPM5 example.

No optimizer updates or augmented generation are performed. One model is kept
resident at a time, including export/reload. Invoke after other full-size jobs
exit; the report does not establish architecture quality or augmented decoding.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from replica_lab.model_adapter import LlamaMemoryAdapter, validate_model_lock
from replica_lab.profiling import RunProfiler
from train_text import encode_row, load_rows


NATIVE_PARAMETERS = 2_516_756_480
SETTINGS = {"memory_size": 64, "workspace_size": 64, "slots": 4,
            "workspace_steps": 2, "insertion_layer": -1, "heads": 4}
IMPLEMENTATION_FILES = ("scripts/qualify_native_adapter.py", "scripts/train_text.py",
                        "src/replica_lab/model_adapter.py", "src/replica_lab/neural.py",
                        "src/replica_lab/profiling.py")


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def initial_provenance(args) -> dict:
    """Retain reproducible inputs even if validation/loading subsequently fails."""
    result = {"status": "started", "device": args.device,
              "parameter_and_compute_dtype": "float32", "adapter_settings": SETTINGS,
              "qualification_scope": "Single-batch native D1/D2 forward/backward/export; no optimizer updates",
              "arguments": {name: str(value) if isinstance(value, Path) else value
                            for name, value in vars(args).items()},
              "implementation_sha256": {name: file_digest(ROOT / name)
                                         for name in IMPLEMENTATION_FILES}}
    for label, path in (("source_lock", args.model_lock), ("data", args.data)):
        try:
            result[label + "_receipt"] = {"path": str(path.resolve()), "sha256": file_digest(path)}
        except (OSError, ValueError) as error:
            result[label + "_receipt"] = {"path": str(path), "error": str(error)}
    try:
        claimed = json.loads(args.model_lock.read_text())
        if not isinstance(claimed, dict):
            raise ValueError("Source lock must be a JSON object")
        result["claimed_source_lock"] = {key: claimed.get(key) for key in
            ("repo_id", "revision", "config_sha256", "snapshot_files_sha256")}
    except (OSError, ValueError, TypeError) as error:
        result["claimed_source_lock_error"] = str(error)
    result["source_seal_validated"] = False
    return result


def check_native_inventory(base) -> dict:
    """Reject a substituted/tiny model in the native CLI qualification path."""
    inventory = {"parameters": sum(parameter.numel() for parameter in base.parameters()),
                 "hidden_size": base.config.hidden_size,
                 "layers": len(base.model.layers), "vocab_size": base.config.vocab_size}
    if inventory != {"parameters": NATIVE_PARAMETERS, "hidden_size": 2048,
                     "layers": 42, "vocab_size": 130560}:
        raise ValueError(f"Unexpected native MiniCPM5 inventory: {inventory}")
    return inventory


def validate_causal_batch(batch: dict) -> None:
    """Require the encoder's exact observed-prefix/answer boundary."""
    required = {"input_ids", "attention_mask", "labels", "prompt_ids", "prompt_mask",
                "memory_ids", "memory_mask"}
    if set(batch) != required:
        raise ValueError("Qualification requires the complete encoded causal batch")
    inputs, labels, prompt = batch["input_ids"], batch["labels"], batch["prompt_ids"]
    if inputs.ndim != 2 or inputs.shape[0] != 1 or labels.shape != inputs.shape:
        raise ValueError("Qualification requires one labelled sequence")
    if prompt.ndim != 2:
        raise ValueError("Prompt IDs must have shape [batch, positions]")
    boundary = prompt.shape[1]
    if (not 0 < boundary < inputs.shape[1]
            or not torch.equal(inputs[:, :boundary], prompt)
            or not bool((labels[:, :boundary] == -100).all())
            or not bool((labels[:, boundary:] != -100).all())
            or not torch.equal(labels[:, boundary:], inputs[:, boundary:])):
        raise ValueError("Future supervised targets must be outside the fixed prompt prefix")
    if (batch["prompt_mask"].shape != prompt.shape
            or not bool(batch["prompt_mask"].all())
            or batch["memory_ids"].ndim != 2
            or batch["memory_mask"].shape != batch["memory_ids"].shape):
        raise ValueError("Invalid prompt or observed-memory masks")


def native_logits(base, batch: dict) -> torch.Tensor:
    base.eval()
    with torch.inference_mode():
        return base(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                    use_cache=False, return_dict=True).logits.detach().cpu()


def adapter_forward(model, batch: dict) -> tuple[torch.Tensor, tuple]:
    """Capture small context evidence while always releasing its batch owner."""
    model.eval()
    try:
        with torch.inference_mode():
            output = model(**batch)
            context = tuple(value.detach().cpu().clone() for value in model._context)
            return output.logits.detach().cpu(), context
    finally:
        model.release_context()


def assert_null_parity(model, batch: dict, expected: torch.Tensor, *,
                       rtol: float = 0.0, atol: float = 0.0) -> dict:
    actual, _ = adapter_forward(model, batch)
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    return {"status": "passed", "compared_logits": actual.numel(),
            "rtol": rtol, "atol": atol,
            "maximum_absolute_difference": (actual - expected).abs().max().item()}


def _probe_projection(model) -> None:
    weight = model.workspace.reader.output_projection.weight
    with torch.no_grad():
        value = torch.arange(weight.numel(), dtype=torch.float32, device="cpu")
        weight.copy_((value.sin() * 0.02).reshape_as(weight).to(weight.device))


def qualify_causality(model, batch: dict) -> dict:
    _, first = adapter_forward(model, batch)
    boundary = batch["prompt_ids"].shape[1]
    altered = dict(batch)
    altered["input_ids"] = batch["input_ids"].clone()
    altered["input_ids"][:, boundary:] = (
        altered["input_ids"][:, boundary:] + 1) % model.base.config.vocab_size
    _, second = adapter_forward(model, altered)
    for previous, current in zip(first, second):
        torch.testing.assert_close(previous, current, rtol=0, atol=0)

    contaminated = {**batch, "prompt_ids": batch["input_ids"],
                    "prompt_mask": torch.ones_like(batch["input_ids"], dtype=torch.bool)}
    try:
        model(**contaminated)
    except ValueError as error:
        if "supervised target" not in str(error):
            raise
    else:
        raise AssertionError("Adapter accepted supervised answer tokens in its workspace")
    finally:
        model.release_context()

    # A writer change must rebuild source features instead of reusing a cache.
    projection = model.support_writer.input_projection.weight
    saved = projection.detach().clone()
    try:
        if bool(batch["memory_mask"].any()):
            with torch.no_grad():
                projection[0, 0].add_(0.05)
            _, refreshed = adapter_forward(model, batch)
            if torch.equal(first[0], refreshed[0]):
                raise AssertionError("Observed-memory slots were stale after a writer change")
            refreshed_status = "passed"
        else:
            refreshed_status = "not_tested_empty_memory"
    finally:
        with torch.no_grad():
            projection.copy_(saved)
    return {"target_suffix_context_invariance": "passed",
            "target_in_workspace_rejected": True,
            "source_features_rebuilt_after_writer_change": refreshed_status,
            "context_released": model._context is None}


def qualify_checkpoint_backward(model, batch: dict) -> dict:
    model.base.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    calls = {"decoder": 0, "support_writer": 0, "workspace_encoder": 0}

    def count(name):
        def hook(module, inputs):
            calls[name] += 1
        return hook

    # Native LlamaDecoderLayer owns checkpointing inside __call__. Its outer
    # sidecar wrapper runs once, while the original block recomputes on backward.
    original_layer = model.base.model.layers[model.insertion_layer].original
    handles = [original_layer.register_forward_pre_hook(count("decoder")),
               model.support_writer.register_forward_pre_hook(count("support_writer")),
               model.workspace.encoder.register_forward_pre_hook(count("workspace_encoder"))]
    model.zero_grad(set_to_none=True)
    try:
        output = model(**batch)
        context = model._context
        versions = tuple(value._version for value in context)
        if not bool(torch.isfinite(output.loss)):
            raise AssertionError("Nonfinite qualification loss")
        output.loss.backward()
        if model._context is not context or tuple(value._version for value in context) != versions:
            raise AssertionError("Checkpoint recomputation mutated or replaced its context")
        if calls["decoder"] < 2 or calls["support_writer"] != 1 or calls["workspace_encoder"] != 1:
            raise AssertionError(f"Unexpected recomputation/context construction counts: {calls}")
        gradients = model.gradient_audit()
        nonfinite = [name for name, parameter in model.named_parameters()
                     if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all())]
        if nonfinite:
            raise AssertionError(f"Nonfinite qualification gradients: {nonfinite}")
        if model.regime == "modules_only" and any(
                parameter.grad is not None for parameter in model._original_parameters.values()):
            raise AssertionError("D1 produced original-parameter gradients")
        if model.regime == "full_augmented" and gradients["original"]["missing_gradient_names"]:
            raise AssertionError("D2 omitted original-parameter gradients")
        for projection in (model.memory_reader.output_projection.weight,
                           model.workspace.reader.output_projection.weight):
            if projection.grad is None or not bool(torch.isfinite(projection.grad).all()):
                raise AssertionError("Missing/nonfinite branch output-projection gradient")
            if not bool(torch.count_nonzero(projection.grad)):
                raise AssertionError("Null branch output projection received no learning signal")
        return {"status": "passed", "loss": output.loss.detach().cpu().item(),
                "context_identity_and_versions_unchanged": True,
                "checkpoint_boundary": "Native original decoder block; residual sidecars outside recomputation",
                "forward_calls": calls, "gradient_audit": gradients,
                "nonfinite_gradient_names": nonfinite,
                "note": "One backward only; zero-initialized branches can have zero upstream gradients"}
    finally:
        for handle in handles:
            handle.remove()
        model.release_context()
        model.zero_grad(set_to_none=True)
        model.eval()


def qualify_adapter(model, batch: dict, expected: torch.Tensor, *,
                    backward: bool = True, rtol: float = 0.0, atol: float = 0.0) -> dict:
    validate_causal_batch(batch)
    model.assert_scope()
    for projection in (model.memory_reader.output_projection.weight,
                       model.workspace.reader.output_projection.weight):
        if bool(torch.count_nonzero(projection)):
            raise AssertionError("Null-branch qualification requires exactly zero output projections")
    scope = model.parameter_audit()
    groups = model.optimizer_groups(1e-4, 1e-4)
    result = {"scope": scope,
              "optimizer_scope": [{"group_name": group["group_name"],
                                   "parameters": sum(p.numel() for p in group["params"]),
                                   "tensors": len(group["params"])} for group in groups],
              "null_parity": assert_null_parity(model, batch, expected, rtol=rtol, atol=atol),
              "causality": qualify_causality(model, batch)}
    try:
        _probe_projection(model)
        perturbed, _ = adapter_forward(model, batch)
        difference = (perturbed - expected).abs().max().item()
        if difference <= 0 or not math.isfinite(difference):
            raise AssertionError("Nonzero workspace branch did not change native logits finitely")
        result["nonzero_branch_sensitivity"] = {"status": "passed", "maximum_difference": difference}
    finally:
        with torch.no_grad():
            model.workspace.reader.output_projection.weight.zero_()
    result["checkpoint_backward"] = (qualify_checkpoint_backward(model, batch) if backward
                                     else {"status": "not_run", "reason": "--skip-backward"})
    if model._context is not None:
        raise AssertionError("Qualification leaked its batch context")
    return result


def _release_device(device: str) -> None:
    gc.collect()
    if device == "mps":
        torch.mps.synchronize()
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def execute(args, profiler: RunProfiler, progress: dict | None = None) -> dict:
    lock = json.loads(args.model_lock.read_text())
    validate_model_lock(args.model_path, lock)
    if progress is not None:
        progress.update(source_seal_validated=True, source_revision=lock["revision"],
                        source_config_sha256=lock["config_sha256"])
    profiler.mark("source_seal_validated")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True,
                                               trust_remote_code=False)
    template_hash = hashlib.sha256(tokenizer.get_chat_template().encode()).hexdigest()
    if progress is not None:
        progress["effective_chat_template_sha256"] = template_hash
    row = load_rows(args.data, args.split)[0]
    batch = {key: value.to(args.device) for key, value in encode_row(
        row, tokenizer, args.max_length).items()}
    validate_causal_batch(batch)
    torch.manual_seed(args.seed)
    base = AutoModelForCausalLM.from_pretrained(
        args.model_path, local_files_only=True, trust_remote_code=False,
        torch_dtype=torch.float32, attn_implementation="eager", use_safetensors=True).to(args.device)
    inventory = check_native_inventory(base)
    if ({p.dtype for p in base.parameters()} != {torch.float32}
            or {p.device.type for p in base.parameters()} != {args.device}):
        raise ValueError("All native parameters must be FP32 on the requested device")
    profiler.mark("native_loaded", **inventory)
    reference = native_logits(base, batch)
    profiler.mark("native_reference_complete")
    arms = []
    for regime in ("modules_only", "full_augmented"):
        model = LlamaMemoryAdapter(base, regime, **SETTINGS).to(args.device)
        profiler.mark("arm_start", regime=regime)
        arm = qualify_adapter(model, batch, reference, backward=not args.skip_backward,
                              rtol=args.rtol, atol=args.atol)
        arms.append(arm)
        profiler.mark("arm_complete", regime=regime)
        if regime == "modules_only":
            # Unwrap before constructing D2 around this same single resident base.
            base.model.layers[model.insertion_layer] = base.model.layers[model.insertion_layer].original
            del model
    # A nonzero serialization probe makes the sidecar reload test observable.
    _probe_projection(model)
    expected_export, _ = adapter_forward(model, batch)
    export = args.output / "adapter_export"
    model.save_pretrained(export, manifest={"purpose": "S2.1 qualification; no optimizer update",
                                           "source_lock": lock, "serialization_probe": True})
    tokenizer.save_pretrained(export / "base")
    profiler.mark("native_export_complete")
    del model, base
    _release_device(args.device)
    profiler.mark("original_model_released")
    loaded_base = AutoModelForCausalLM.from_pretrained(
        export / "base", local_files_only=True, trust_remote_code=False,
        torch_dtype=torch.float32, attn_implementation="eager", use_safetensors=True)
    loaded = LlamaMemoryAdapter(loaded_base, "full_augmented", **SETTINGS)
    loaded.load_sidecars(export / "sidecars.safetensors")
    loaded = loaded.to(args.device).eval()
    actual_export, _ = adapter_forward(loaded, batch)
    torch.testing.assert_close(actual_export, expected_export, rtol=args.rtol, atol=args.atol)
    profiler.mark("export_reload_complete")
    report = {"status": "passed_native_adapter_forward_qualification",
              "source_revision": lock["revision"], "native_inventory": inventory,
              "source_lock_sha256": hashlib.sha256(args.model_lock.read_bytes()).hexdigest(),
              "source_snapshot_files_sha256": lock["snapshot_files_sha256"],
              "source_seal_validated": True,
              "source_config_sha256": lock["config_sha256"], "device": args.device,
              "effective_chat_template_sha256": template_hash,
              "template_settings": {"enable_thinking": False, "add_generation_prompt": True},
              "parameter_and_compute_dtype": "float32", "cpu_fallback_enabled": False,
              "adapter_settings": SETTINGS, "seed": args.seed,
              "packages": {name: importlib.metadata.version(name)
                           for name in ("torch", "transformers", "safetensors")},
              "implementation_sha256": {name: file_digest(ROOT / name)
                                         for name in IMPLEMENTATION_FILES},
              "example": {"id": row["id"], "source_ids": row["source_ids"],
                          "group_id": row["group_id"],
                          "sequence_tokens": batch["input_ids"].shape[1],
                          "prompt_tokens": batch["prompt_ids"].shape[1],
                          "observed_memory_tokens": batch["memory_ids"].shape[1],
                          "supervised_tokens": int((batch["labels"] != -100).sum()),
                          "prompt_token_ids_sha256": hashlib.sha256(json.dumps(
                              batch["prompt_ids"].cpu().tolist()).encode()).hexdigest(),
                          "memory_token_ids_sha256": hashlib.sha256(json.dumps(
                              batch["memory_ids"].cpu().tolist()).encode()).hexdigest(),
                          "data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest()},
              "arms": arms,
              "export_reload": {"status": "passed", "regime": "full_augmented",
                                "nonzero_sidecar_probe": True,
                                "compared_logits": actual_export.numel(),
                                "maximum_absolute_difference": (
                                    actual_export - expected_export).abs().max().item()},
              "optimizer_updates": 0, "generation": {"status": "not_implemented",
                "reason": "Augmented context reconstruction/caching is a separate S2.1 gate"},
              "limitations": ["One verified fixture; no architecture quality or learning result",
                "No augmented generation or D1/D2 training-convergence qualification",
                "Nonzero serialization probe is diagnostic, not a trained checkpoint",
                "Sampled MPS peaks can miss transients; RSS and driver counters overlap"]}
    del loaded, loaded_base
    _release_device(args.device)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-lock", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=ROOT / "data/fixtures/dev.jsonl")
    parser.add_argument("--split", choices=("train", "dev"), default="dev")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="mps")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--skip-backward", action="store_true")
    parser.add_argument("--rtol", type=float, default=0.0)
    parser.add_argument("--atol", type=float, default=0.0)
    args = parser.parse_args()
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        parser.error("Use a new empty output directory")
    if (args.max_length < 8 or min(args.rtol, args.atol) < 0
            or not all(math.isfinite(value) for value in (args.rtol, args.atol))):
        parser.error("Invalid sequence budget or tolerance")
    if args.device == "mps" and os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
        parser.error("Unset PYTORCH_ENABLE_MPS_FALLBACK; backend substitution is forbidden")
    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS unavailable; no backend substitution")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable; no backend substitution")
    args.output.mkdir(parents=True, exist_ok=True)
    profiler = RunProfiler(args.device, args.output, sample_interval=0.25).start()
    report = initial_provenance(args)
    try:
        report.update(execute(args, profiler, progress=report))
    except BaseException as error:
        report.update(status="failed_native_adapter_qualification", error=str(error),
                      error_type=type(error).__name__, traceback=traceback.format_exc())
        raise
    finally:
        report["profile"] = profiler.stop()
        (args.output / "qualification_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "report": str(args.output / "qualification_report.json")}))


if __name__ == "__main__":
    main()

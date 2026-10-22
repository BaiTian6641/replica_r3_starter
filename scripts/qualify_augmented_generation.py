#!/usr/bin/env python3
"""Qualify uncached prompt-fixed native D1/D2 generation without training.

One sealed native backbone is resident at a time. Greedy null-path answers and
next-token logits must match the original unwrapped native decoder. A forced
multi-token probe checks one-time context construction; nonzero sidecar probes
and export/reload check observable branch and serialization behavior. These are
interface checks on one fixture, not quality, learning or KV-cache qualification.
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
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from replica_lab.generation import generate_fixed_context
from replica_lab.model_adapter import LlamaMemoryAdapter, validate_model_lock
from replica_lab.profiling import RunProfiler
from qualify_native_adapter import (NATIVE_PARAMETERS, SETTINGS, check_native_inventory,
                                    file_digest, validate_causal_batch)
from train_text import encode_row, load_rows

IMPLEMENTATION_FILES = (
    "scripts/qualify_augmented_generation.py", "scripts/qualify_native_adapter.py",
    "scripts/train_text.py", "src/replica_lab/generation.py",
    "src/replica_lab/model_adapter.py", "src/replica_lab/neural.py",
    "src/replica_lab/profiling.py")


def initial_provenance(args):
    report = {"status": "started", "device": args.device,
              "parameter_and_compute_dtype": "float32", "adapter_settings": SETTINGS,
              "qualification_scope": "One-fixture uncached native D1/D2 generation; zero optimizer updates",
              "arguments": {key: str(value) if isinstance(value, Path) else value
                            for key, value in vars(args).items()},
              "implementation_sha256": {name: file_digest(ROOT / name) for name in IMPLEMENTATION_FILES},
              "source_seal_validated": False}
    for label, path in (("source_lock", args.model_lock), ("data", args.data)):
        try:
            report[label + "_receipt"] = {"path": str(path.resolve()), "sha256": file_digest(path)}
        except OSError as error:
            report[label + "_receipt"] = {"path": str(path), "error": str(error)}
    return report


def _release(device):
    gc.collect()
    if device == "mps":
        torch.mps.synchronize()
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def _capture_logits(base):
    logits = []

    def capture(module, inputs, output):
        last = output.logits[:, -1, :].detach().cpu().clone()
        if not bool(torch.isfinite(last).all()):
            raise AssertionError("Nonfinite next-token logits during generation")
        logits.append(last)
    return logits, base.register_forward_hook(capture)


def native_reference(base, batch, cap, eos):
    """Capture the unwrapped native HF route with caching explicitly disabled."""
    config = GenerationConfig.from_model_config(base.config)
    config.do_sample = False
    config.num_beams = 1
    config.use_cache = False
    config.max_new_tokens = cap
    config.eos_token_id = eos
    steps, handle = _capture_logits(base)
    try:
        with torch.inference_mode():
            sequence = base.generate(input_ids=batch["prompt_ids"],
                                     attention_mask=batch["prompt_mask"],
                                     generation_config=config, logits_to_keep=0)
        return {"sequence": sequence.detach().cpu(), "step_logits": steps}
    finally:
        handle.remove()


def observed_generation(model, batch, cap, eos):
    """Audit public generation inputs and one writer/workspace build per call."""
    names = ("prompt_ids", "prompt_mask", "memory_ids", "memory_mask")
    before = {name: batch[name].clone() for name in names}
    modes = [module.training for module in model.modules()]
    calls = {"support_writer": 0, "workspace_encoder": 0}

    def count(name):
        def hook(module, inputs):
            calls[name] += 1
        return hook

    steps, handle = _capture_logits(model.base)
    handles = [handle,
               model.support_writer.register_forward_pre_hook(count("support_writer")),
               model.workspace.encoder.register_forward_pre_hook(count("workspace_encoder"))]
    try:
        sequence = generate_fixed_context(model, **{name: batch[name] for name in names},
                                          max_new_tokens=cap, eos_token_id=eos).detach().cpu()
        if calls != {"support_writer": 1, "workspace_encoder": 1}:
            raise AssertionError(f"Decision context was not constructed exactly once: {calls}")
        if model._context is not None:
            raise AssertionError("Generation leaked its owned context")
        if [module.training for module in model.modules()] != modes:
            raise AssertionError("Generation changed caller-owned module modes")
        for name in names:
            torch.testing.assert_close(batch[name], before[name], rtol=0, atol=0)
        if not torch.equal(sequence[:, :batch["prompt_ids"].shape[1]], before["prompt_ids"].cpu()):
            raise AssertionError("Generation changed the original prompt prefix")
        if len(steps) != sequence.shape[1] - batch["prompt_ids"].shape[1]:
            raise AssertionError("Expected one uncached native forward per emitted token")
        return {"sequence": sequence, "step_logits": steps,
                "context_build_calls": calls, "inputs_unchanged": True,
                "module_modes_restored": True, "context_released": True}
    finally:
        for item in handles:
            item.remove()


def assert_generation_parity(actual, expected, *, rtol=0.0, atol=0.0):
    if not torch.equal(actual["sequence"], expected["sequence"]):
        raise AssertionError("Generated tokens differ from the declared reference")
    if len(actual["step_logits"]) != len(expected["step_logits"]) or not actual["step_logits"]:
        raise AssertionError("Generation step-logit coverage differs")
    maximum = 0.0
    for current, reference in zip(actual["step_logits"], expected["step_logits"]):
        torch.testing.assert_close(current, reference, rtol=rtol, atol=atol)
        maximum = max(maximum, (current - reference).abs().max().item())
    return {"status": "passed", "generated_tokens_exact": True,
            "compared_decode_steps": len(actual["step_logits"]),
            "compared_next_token_logits": sum(item.numel() for item in actual["step_logits"]),
            "maximum_absolute_logit_difference": maximum, "rtol": rtol, "atol": atol}


def _probe_projection(weight):
    with torch.no_grad():
        values = torch.arange(weight.numel(), device="cpu", dtype=torch.float32)
        weight.copy_((values.sin() * 0.02).reshape_as(weight).to(weight.device))


def qualify_sensitivity(model, batch, reference):
    results = {}
    for name, weight in (("memory", model.memory_reader.output_projection.weight),
                         ("workspace", model.workspace.reader.output_projection.weight)):
        try:
            _probe_projection(weight)
            actual = observed_generation(model, batch, 1, model.base.config.eos_token_id)
            difference = (actual["step_logits"][0] - reference["step_logits"][0]).abs().max().item()
            if not math.isfinite(difference) or difference <= 0:
                raise AssertionError(f"Nonzero {name} projection did not change first-token logits")
            results[name] = {"status": "passed", "maximum_first_token_logit_difference": difference,
                             "probe": "sin(arange) * 0.02, diagnostic not trained"}
        finally:
            with torch.no_grad():
                weight.zero_()
    return results


def _completion(generation, tokenizer, prompt_length, eos):
    tokens = generation["sequence"][0, prompt_length:].tolist()
    stops = set(eos if isinstance(eos, list) else [eos]) - {None}
    ended = bool(tokens and tokens[-1] in stops)
    return {"generated_token_ids": tokens, "generated_tokens": len(tokens),
            "ended_by_eos": ended,
            "raw_completion": tokenizer.decode(tokens, skip_special_tokens=False),
            "completion": tokenizer.decode(tokens[:-1] if ended else tokens, skip_special_tokens=False)}


def execute(args, profiler, progress=None, *, expected_original_parameters=NATIVE_PARAMETERS):
    """Internal count override is for CPU fixture tests; the CLI never exposes it."""
    base = model = loaded_base = loaded = None
    try:
        lock = json.loads(args.model_lock.read_text())
        validate_model_lock(args.model_path, lock)
        if progress is not None:
            progress.update(source_seal_validated=True, source_revision=lock["revision"])
        profiler.mark("source_seal_validated")
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True,
                                                   trust_remote_code=False)
        row = load_rows(args.data, args.split)[0]
        batch = {key: value.to(args.device) for key, value in
                 encode_row(row, tokenizer, args.max_length).items()}
        validate_causal_batch(batch)
        torch.manual_seed(args.seed)
        base = AutoModelForCausalLM.from_pretrained(
            args.model_path, local_files_only=True, trust_remote_code=False,
            torch_dtype=torch.float32, attn_implementation="eager", use_safetensors=True).to(args.device).eval()
        if expected_original_parameters == NATIVE_PARAMETERS:
            inventory = check_native_inventory(base)
        else:
            inventory = {"parameters": sum(parameter.numel() for parameter in base.parameters()),
                         "hidden_size": base.config.hidden_size, "layers": len(base.model.layers),
                         "vocab_size": base.config.vocab_size}
            if inventory["parameters"] != expected_original_parameters:
                raise ValueError("Unexpected fixture original parameter count")
        if ({parameter.dtype for parameter in base.parameters()} != {torch.float32}
                or {parameter.device.type for parameter in base.parameters()} != {args.device}):
            raise ValueError("All native parameters must be FP32 on the requested backend")
        profiler.mark("native_loaded", **inventory)
        eos = base.config.eos_token_id
        reference = native_reference(base, batch, args.max_new_tokens, eos)
        forced_cap = min(3, args.max_new_tokens)
        forced_reference = native_reference(base, batch, forced_cap, [])
        if len(forced_reference["step_logits"]) != forced_cap:
            raise AssertionError("Native no-EOS reference failed the multi-token budget")
        profiler.mark("unwrapped_native_references_complete")
        arms = []
        for regime in ("modules_only", "full_augmented"):
            model = LlamaMemoryAdapter(base, regime, **SETTINGS).to(args.device).eval()
            model.assert_scope()
            actual = observed_generation(model, batch, args.max_new_tokens, eos)
            forced = observed_generation(model, batch, forced_cap, [])
            arms.append({"regime": regime, "scope": model.parameter_audit(),
                         "native_uncached_parity": assert_generation_parity(
                             actual, reference, rtol=args.rtol, atol=args.atol),
                         "forced_multitoken_parity": assert_generation_parity(
                             forced, forced_reference, rtol=args.rtol, atol=args.atol),
                         "context_audit": {key: forced[key] for key in
                             ("context_build_calls", "inputs_unchanged", "module_modes_restored", "context_released")},
                         "completion": _completion(actual, tokenizer, batch["prompt_ids"].shape[1], eos)})
            profiler.mark("null_generation_qualified", regime=regime)
            if regime == "modules_only":
                base.model.layers[model.insertion_layer] = base.model.layers[model.insertion_layer].original
                model = None
        sensitivity = qualify_sensitivity(model, batch, reference)
        _probe_projection(model.memory_reader.output_projection.weight)
        _probe_projection(model.workspace.reader.output_projection.weight)
        before_export = observed_generation(model, batch, args.max_new_tokens, eos)
        profiler.mark("nonzero_generation_qualified")
        export = args.output / "adapter_export"
        model.save_pretrained(export, manifest={"purpose": "S2.1 uncached generation qualification",
                                               "model_lock": lock, "serialization_probe": True,
                                               "optimizer_updates": 0})
        tokenizer.save_pretrained(export / "base")
        export_receipt = {path.relative_to(export).as_posix(): file_digest(path)
                          for path in sorted(export.rglob("*")) if path.is_file()}
        profiler.mark("native_export_complete")
        model = base = None
        _release(args.device)
        profiler.mark("original_backbone_released")
        loaded_base = AutoModelForCausalLM.from_pretrained(
            export / "base", local_files_only=True, trust_remote_code=False,
            torch_dtype=torch.float32, attn_implementation="eager", use_safetensors=True)
        loaded = LlamaMemoryAdapter(loaded_base, "full_augmented", **SETTINGS)
        loaded.load_sidecars(export / "sidecars.safetensors")
        loaded = loaded.to(args.device).eval()
        after_export = observed_generation(loaded, batch, args.max_new_tokens, eos)
        reload_parity = assert_generation_parity(after_export, before_export,
                                                rtol=args.rtol, atol=args.atol)
        profiler.mark("export_reload_generation_qualified")
        return {"status": ("passed_native_augmented_generation_qualification"
                           if expected_original_parameters == NATIVE_PARAMETERS
                           else "passed_tiny_augmented_generation_qualification"),
                "source_seal_validated": True, "source_revision": lock["revision"],
                "source_config_sha256": lock["config_sha256"],
                "source_lock_sha256": file_digest(args.model_lock),
                "source_snapshot_files_sha256": lock.get("snapshot_files_sha256", {}),
                "effective_chat_template_sha256": hashlib.sha256(tokenizer.get_chat_template().encode()).hexdigest(),
                "template_settings": {"enable_thinking": False, "add_generation_prompt": True},
                "native_inventory": inventory, "device": args.device,
                "parameter_and_compute_dtype": "float32", "cpu_fallback_enabled": False,
                "adapter_settings": SETTINGS, "seed": args.seed,
                "example": {"id": row["id"], "group_id": row["group_id"], "source_ids": row["source_ids"],
                            "reference_answer": row["response"], "sequence_tokens": batch["input_ids"].shape[1],
                            "prompt_tokens": batch["prompt_ids"].shape[1],
                            "memory_tokens": batch["memory_ids"].shape[1],
                            "prompt_ids_sha256": hashlib.sha256(json.dumps(batch["prompt_ids"].cpu().tolist()).encode()).hexdigest(),
                            "memory_ids_sha256": hashlib.sha256(json.dumps(batch["memory_ids"].cpu().tolist()).encode()).hexdigest()},
                "decoding": {"do_sample": False, "num_beams": 1, "use_cache": False,
                             "max_new_tokens": args.max_new_tokens, "eos_token_id": eos,
                             "forced_multitoken_cap": forced_cap, "forced_multitoken_eos": []},
                "native_reference": _completion(reference, tokenizer, batch["prompt_ids"].shape[1], eos),
                "arms": arms, "nonzero_sidecar_sensitivity": sensitivity,
                "export_reload": {**reload_parity, "nonzero_sidecars": True,
                                  "files_sha256": export_receipt,
                                  "completion": _completion(after_export, tokenizer, batch["prompt_ids"].shape[1], eos)},
                "optimizer_updates": 0, "packages": {name: importlib.metadata.version(name)
                    for name in ("torch", "transformers", "safetensors")},
                "implementation_sha256": {name: file_digest(ROOT / name) for name in IMPLEMENTATION_FILES},
                "limitations": ["One fixture; interface parity does not establish task quality or learning",
                    "Prefix is recomputed per token; KV-cache optimization is not implemented or qualified",
                    "Nonzero sidecars are deterministic serialization probes, not trained artifacts",
                    "No optimizer update, D1 overfit, D2 capacity or frozen-memory benefit claim",
                    "Sampled memory peaks can miss transients; MPS and RSS counters overlap"]}
    finally:
        if model is not None:
            model.release_context()
        if loaded is not None:
            loaded.release_context()
        model = base = loaded = loaded_base = None
        _release(args.device)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-lock", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=ROOT / "data/diagnostics/capacity_256/dev.jsonl")
    parser.add_argument("--split", choices=("train", "dev"), default="dev")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="mps")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--rtol", type=float, default=0)
    parser.add_argument("--atol", type=float, default=0)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        parser.error("Use a new empty output directory")
    if (args.max_length < 8 or args.max_new_tokens < 2 or min(args.rtol, args.atol) < 0
            or not all(math.isfinite(value) for value in (args.rtol, args.atol))):
        parser.error("Require max-length >=8, at least two new tokens and finite nonnegative tolerances")
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
        report.update(status="failed_native_augmented_generation_qualification", error=str(error),
                      error_type=type(error).__name__, traceback=traceback.format_exc())
        raise
    finally:
        report["profile"] = profiler.stop()
        (args.output / "qualification_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "report": str(args.output / "qualification_report.json")}))


if __name__ == "__main__":
    main()

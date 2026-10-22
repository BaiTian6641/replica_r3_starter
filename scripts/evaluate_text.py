#!/usr/bin/env python3
"""Offline native MiniCPM5 greedy generation on verified development fixtures.

This small functional preflight measures fixture task completion, not general
model quality. The training-only memory adapter is deliberately bypassed; only
the original sealed parent or a plain D0 full-tuning export can be evaluated.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from train_text import encode_row, load_rows
from replica_lab.evaluation import score_answer, strict_integer


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def exported_inventory(directory: Path, source_config: dict, lock: dict) -> dict:
    """Accept a native plain-full export and preserve its local byte provenance."""
    if not directory.is_dir() or not list(directory.glob("*.safetensors")):
        raise ValueError("--exported-base requires a local native safetensors export")
    manifest_path = directory.parent / "adapter_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("regime") != "full" or manifest.get("training_scope") != "text_only":
        raise ValueError("Native export evaluation supports plain D0 full tuning only")
    parent = manifest.get("run", {}).get("model_lock", {})
    if any(parent.get(key) != lock.get(key) for key in ("repo_id", "revision", "config_sha256")):
        raise ValueError("Export parent does not match the sealed source model lock")
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    for name in ("architectures", "model_type", "hidden_size", "intermediate_size",
                 "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
                 "vocab_size", "tie_word_embeddings"):
        if config.get(name) != source_config.get(name):
            raise ValueError(f"Export changes native architecture field {name}")
    if config.get("auto_map"):
        raise ValueError("Remote-code exports are unsupported")
    relevant = sorted(p for p in directory.rglob("*")
                      if p.is_file() and p.suffix in {".json", ".safetensors", ".model", ".jinja"})
    return {"base_path": str(directory.resolve()),
            "adapter_manifest_sha256": file_sha256(manifest_path),
            "files_sha256": {p.relative_to(directory).as_posix(): file_sha256(p) for p in relevant}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True,
                        help="Original sealed native source, including its tokenizer")
    parser.add_argument("--model-lock", type=Path, required=True)
    parser.add_argument("--exported-base", type=Path,
                        help="Optional checkpoint/base from a plain D0 full-tuning run")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="mps")
    parser.add_argument("--max-examples", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=256,
                        help="Same full prompt-plus-reference budget used by the trainer")
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    if min(args.max_examples, args.max_new_tokens) < 1 or args.max_length < 8:
        parser.error("Example/token budgets must be positive and max-length must be at least 8")
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        parser.error("Use a new empty output directory to preserve earlier runs")
    if args.device == "mps" and os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
        parser.error("Unset PYTORCH_ENABLE_MPS_FALLBACK; CPU fallback is not permitted")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
    from replica_lab.model_adapter import validate_model_lock
    from replica_lab.profiling import RunProfiler

    args.output.mkdir(parents=True, exist_ok=True)
    report = {"status": "started", "device": args.device,
              "parameter_dtype": "float32", "compute_dtype": "float32",
              "model_path": str(args.model_path.resolve()),
              "model_lock_path": str(args.model_lock.resolve()),
              "tokenizer_path": str(args.model_path.resolve()),
              "data_path": str(args.data.resolve()), "data_sha256": file_sha256(args.data),
              "seed": args.seed, "max_examples": args.max_examples,
              "max_new_tokens": args.max_new_tokens, "max_length": args.max_length,
              "decoding": {"do_sample": False, "num_beams": 1, "use_cache": True,
                           "enable_thinking": False}, "examples": [],
              "limitations": ["Small synthetic XOR development-fixture preflight, not general quality",
                              "Only the first declared development examples are evaluated",
                              "Native cached generation covers the parent and plain D0 export only",
                              "Memory peaks are sampled; profiler records interval and counter semantics"]}
    profiler = RunProfiler(args.device, args.output, sample_interval=0.25)
    profiler.start()
    started = time.monotonic()
    try:
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; no backend substitution")
        if args.device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but unavailable; no backend substitution")
        lock = json.loads(args.model_lock.read_text(encoding="utf-8"))
        source_config = validate_model_lock(args.model_path, lock)
        report["model_lock"] = lock
        model_path = args.model_path
        if args.exported_base is not None:
            report["export_provenance"] = exported_inventory(args.exported_base, source_config, lock)
            model_path = args.exported_base
        profiler.mark("snapshot_validated")
        all_rows = load_rows(args.data, "dev")
        rows = all_rows[:args.max_examples]
        report["available_dev_examples"] = len(all_rows)
        torch.manual_seed(args.seed)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True,
                                                   trust_remote_code=False)
        template = tokenizer.get_chat_template()
        report["effective_chat_template_sha256"] = hashlib.sha256(template.encode()).hexdigest()
        encoded = [encode_row(row, tokenizer, args.max_length) for row in rows]
        base = AutoModelForCausalLM.from_pretrained(
            model_path, local_files_only=True, trust_remote_code=False,
            torch_dtype=torch.float32, attn_implementation="eager", use_safetensors=True)
        base.config.use_cache = True
        base.requires_grad_(False)
        base = base.to(args.device).eval()
        count = sum(parameter.numel() for parameter in base.parameters())
        if count != 2516756480:
            raise RuntimeError(f"Unexpected native MiniCPM5 parameter count: {count}")
        parameter_dtypes = {parameter.dtype for parameter in base.parameters()}
        parameter_devices = {parameter.device.type for parameter in base.parameters()}
        if parameter_dtypes != {torch.float32} or parameter_devices != {args.device}:
            raise RuntimeError("Native generation requires all parameters in FP32 on the requested device")
        report.update(parameter_count=count,
                      parameter_dtypes=sorted(str(dtype) for dtype in parameter_dtypes),
                      parameter_devices=sorted(parameter_devices), native_use_cache=base.config.use_cache)
        generation_config = GenerationConfig.from_model_config(base.config)
        generation_config.do_sample = False
        generation_config.num_beams = 1
        generation_config.max_new_tokens = args.max_new_tokens
        generation_config.use_cache = True
        report["generation_config"] = generation_config.to_dict()
        eos = generation_config.eos_token_id
        eos_ids = set(eos if isinstance(eos, list) else [eos])
        profiler.mark("model_loaded", parameter_count=count)
        with torch.inference_mode():
            for row, batch in zip(rows, encoded):
                input_ids = batch["prompt_ids"].to(args.device)
                attention_mask = batch["prompt_mask"].to(args.device)
                profiler.mark("generation_start", example_id=row["id"])
                example_started = time.monotonic()
                output = base.generate(input_ids=input_ids, attention_mask=attention_mask,
                                       generation_config=generation_config)
                if args.device == "mps":
                    torch.mps.synchronize()
                elif args.device == "cuda":
                    torch.cuda.synchronize()
                elapsed = time.monotonic() - example_started
                generated = output[0, input_ids.shape[1]:].cpu().tolist()
                ended_by_eos = bool(generated and generated[-1] in eos_ids)
                # Remove only an actual terminal EOS, preserving any other emitted
                # special tokens so malformed answers cannot pass strict scoring.
                answer_ids = generated[:-1] if ended_by_eos else generated
                completion = tokenizer.decode(answer_ids, skip_special_tokens=False)
                result = {"id": row["id"], "group_id": row["group_id"],
                          "source_ids": row["source_ids"], "expected": row["response"],
                          "prompt_tokens": input_ids.shape[1],
                          "prompt_token_ids_sha256": hashlib.sha256(
                              json.dumps(batch["prompt_ids"].tolist()).encode()).hexdigest(),
                          "generated_token_ids": generated,
                          "generated_tokens": len(generated), "ended_by_eos": ended_by_eos,
                          "hit_token_budget": not ended_by_eos and len(generated) >= args.max_new_tokens,
                          "raw_completion": tokenizer.decode(generated, skip_special_tokens=False),
                          "completion": completion, "elapsed_seconds": elapsed,
                          **score_answer(completion, row["response"])}
                report["examples"].append(result)
                with (args.output / "generations.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(result, ensure_ascii=False) + "\n")
                profiler.mark("generation_finished", example_id=row["id"],
                              generated_tokens=len(generated), task_correct=result["task_correct"])
                print(json.dumps({"example_id": row["id"], "task_correct": result["task_correct"],
                                  "generated_tokens": len(generated), "elapsed_seconds": elapsed}), flush=True)
                del output, input_ids, attention_mask
        examples = report["examples"]
        report.update(status="completed_native_generation_preflight", evaluated_examples=len(examples),
                      task_correct=sum(r["task_correct"] for r in examples),
                      fixture_task_accuracy=sum(r["task_correct"] for r in examples) / len(examples),
                      exact_text_accuracy=sum(r["exact_text_match"] for r in examples) / len(examples),
                      packages={name: importlib.metadata.version(name)
                                for name in ("torch", "transformers", "safetensors")})
    except BaseException as error:
        report.update(status="failed_native_generation_preflight", error_type=type(error).__name__,
                      error=str(error))
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        report["profile"] = profiler.stop()
        (args.output / "generation_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(args.output / "generation_report.json"),
                      "fixture_task_accuracy": report["fixture_task_accuracy"]}), flush=True)


if __name__ == "__main__":
    main()

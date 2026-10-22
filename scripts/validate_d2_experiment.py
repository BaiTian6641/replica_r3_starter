#!/usr/bin/env python3
"""Validate completed D2 receipts/runtime metadata and replay recorded guards.

Default: small provenance files, safetensor headers, memory-mapped optimizer
metadata and bounded sidecar continuation comparisons. No native weight/runtime
hashing, model construction, GPU calls or global RNG/state restoration.
Use --verify-payload-bytes explicitly for a full checkpoint byte audit.
Pass every completed chunk in a resume chain using repeated --run arguments.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
from safetensors import safe_open

from replica_lab.capacity_guard import CapacityGuard, CapacityPolicy
from replica_lab.checkpointing import file_digest, validate_training_checkpoint
from replica_lab.evaluation import score_answer
from replica_lab.curriculum import balanced_batches, batch_diagnostics
from replica_lab.training_tools import apply_learning_rates


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def object_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def verify_implementation(hashes, historical_code_ref=None):
    """Bind historical claims to actual local Git objects, never ignore drift."""
    require(isinstance(hashes, dict) and bool(hashes), "Missing implementation provenance")
    commit = None
    if historical_code_ref is not None:
        commit = subprocess.run(
            ["git", "rev-parse", "--verify", historical_code_ref + "^{commit}"],
            cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
    mismatches = []
    for name, expected in hashes.items():
        path = ROOT / name
        require(path.resolve().is_relative_to(ROOT), "Implementation path escapes repository")
        current = file_digest(path) if path.is_file() else None
        if current != expected:
            mismatches.append(name)
        if commit is None:
            require(current == expected, f"Bound implementation bytes changed: {name}; "
                    "supply the actual --historical-code-ref for historical evidence")
        else:
            archived = subprocess.run(["git", "show", f"{commit}:{name}"], cwd=ROOT,
                                      check=True, capture_output=True).stdout
            require(hashlib.sha256(archived).hexdigest() == expected,
                    f"Historical implementation bytes differ: {name}")
    return {"mode": "historical_git_objects" if commit else "current_workspace",
            "historical_commit": commit, "verified_files": len(hashes),
            "current_workspace_mismatches": sorted(mismatches)}


def inspect_checkpoint(checkpoint, contract, *, verify_payload_bytes=False):
    """Verify receipts without paging native weights/optimizer storage by default."""
    if verify_payload_bytes:
        receipt = validate_training_checkpoint(checkpoint, contract)
        return receipt, {"mode": "full_payload_bytes", "all_receipt_bytes_verified": True,
                         "unhashed_payloads": []}
    receipt = read(checkpoint / "checkpoint_receipt.json")
    require(receipt.get("format_version") == 1 and receipt.get("run_contract") == contract
            and receipt.get("contract_sha256") == object_digest(contract), "Checkpoint receipt contract differs")
    files = {p.relative_to(checkpoint).as_posix(): p for p in checkpoint.rglob("*")
             if p.is_file() and p.name != "checkpoint_receipt.json"}
    require(not any(p.is_symlink() for p in checkpoint.rglob("*")), "Checkpoint symlinks are unsupported")
    require(set(files) == set(receipt.get("files_sha256", {})), "Checkpoint receipt file inventory differs")
    required = {"runtime.pt", "adapter_manifest.json", "sidecars.safetensors", "base/config.json",
                "base/tokenizer.json", "base/tokenizer_config.json"}
    require(required.issubset(files), "Checkpoint lacks model/runtime/tokenizer artifacts")
    weights = {name for name in files if name.startswith("base/") and name.count("/") == 1
               and name.endswith(".safetensors")}
    require(bool(weights), "Checkpoint lacks native model weights")
    if "base/model.safetensors.index.json" in files:
        shards = {"base/" + p for p in read(files["base/model.safetensors.index.json"])["weight_map"].values()}
        require(shards == weights, "Native weight shard index differs")
    unhashed = []
    for name, path in files.items():
        require(path.stat().st_size > 0, "Empty checkpoint file")
        digest = receipt["files_sha256"][name]
        require(isinstance(digest, str) and len(digest) == 64
                and all(c in "0123456789abcdef" for c in digest), "Malformed payload receipt digest")
        if name == "runtime.pt" or name.endswith(".safetensors"):
            unhashed.append(name)
        else:
            require(path.stat().st_size < 32 * 1024**2, "Oversized provenance file")
            require(file_digest(path) == digest, f"Checkpoint small-file byte mismatch: {name}")
    return receipt, {"mode": "metadata_headers_and_small_files", "all_receipt_bytes_verified": False,
                     "unhashed_payloads": sorted(unhashed)}


def tensor_headers(paths):
    result = {}
    for path in sorted(paths):
        with safe_open(path, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                require(name not in result, "Duplicate tensor in checkpoint shards")
                value = handle.get_slice(name)
                require(value.get_dtype() == "F32", "D2 checkpoint requires FP32 tensor storage")
                result[name] = tuple(value.get_shape())
    require(bool(result), "Missing checkpoint tensor inventory")
    return result


def optimizer_inventory(optimizer, checkpoint, manifest, *, step, settings=None):
    """Match every optimizer tensor to an actual header without reading weights."""
    original = tensor_headers((checkpoint / "base").glob("*.safetensors"))
    added = tensor_headers([checkpoint / "sidecars.safetensors"])
    config = read(checkpoint / "base/config.json")
    insertion = manifest["adapter_settings"]["insertion_layer"]
    if insertion < 0:
        insertion += config["num_hidden_layers"]
    wrapped_prefix = f"model.layers.{insertion}."
    shapes = {"base." + name.replace(wrapped_prefix, wrapped_prefix + "original.", 1): shape
              for name, shape in original.items()}
    require(not (set(shapes) & set(added)), "Original and sidecar ownership overlaps")
    shapes.update(added)
    require(optimizer.get("replica_optimizer") == {"format_version": 1, "name": "muon8bit"},
            "Wrong optimizer implementation/format")
    groups = optimizer["param_groups"]
    require([(g["scope"], g["algorithm"]) for g in groups] == [
        ("original", "muon8bit"), ("original", "adamw"),
        ("new", "muon8bit"), ("new", "adamw")], "D2 must own all four optimizer groups")
    owned_names, owned_ids, counts, dtype_bytes = set(), set(), [], Counter()
    for group in groups:
        require(group["group_name"] == group["scope"] + "_" + group["algorithm"], "Optimizer group name differs")
        if settings:
            require(group["weight_decay"] == settings["weight_decay"], "Optimizer decay differs")
            if group["algorithm"] == "muon8bit":
                require(all(group[k] == settings[k] for k in ("momentum", "ns_steps", "block_size")),
                        "Muon hyperparameters differ")
            else:
                require(tuple(group["betas"]) == (0.9, 0.999) and group["eps"] == 1e-8,
                        "AdamW hyperparameters differ")
        require(len(group["params"]) == len(group["param_names"]), "Optimizer names/IDs differ")
        parameters = 0
        for identity, name in zip(group["params"], group["param_names"]):
            require(identity not in owned_ids and name not in owned_names, "Duplicated optimizer ownership")
            require(name in shapes, "Optimizer parameter missing from saved model")
            owned_ids.add(identity)
            owned_names.add(name)
            shape = shapes[name]
            parameters += math.prod(shape)
            expected_scope = "original" if name.startswith("base.") else "new"
            excluded = name in {"base.model.embed_tokens.weight", "base.lm_head.weight"} or name.endswith(".slot_queries")
            expected_algorithm = "muon8bit" if len(shape) == 2 and not excluded else "adamw"
            require(group["scope"] == expected_scope and group["algorithm"] == expected_algorithm,
                    "Optimizer scope or embedding/shape routing differs")
            state = optimizer["state"].get(identity)
            require(isinstance(state, dict) and bool(state), "Missing persistent optimizer state")
            if group["algorithm"] == "muon8bit":
                require(set(state) == {"momentum_q8", "momentum_scale"}, "Invalid Muon state keys")
                require(state["momentum_q8"].dtype == torch.int8
                        and tuple(state["momentum_q8"].shape) == shape, "Muon codes dtype/shape differ")
                blocks = (math.prod(shape) + group["block_size"] - 1) // group["block_size"]
                require(state["momentum_scale"].dtype == torch.float32
                        and tuple(state["momentum_scale"].shape) == (blocks,), "Muon scales dtype/shape differ")
            else:
                require(set(state) == {"step", "exp_avg", "exp_avg_sq"}
                        and state["step"] == step, "Adam state step/keys differ")
                require(all(state[key].dtype == torch.float32 and tuple(state[key].shape) == shape
                            for key in ("exp_avg", "exp_avg_sq")), "Adam moments dtype/shape differ")
            for value in state.values():
                if isinstance(value, torch.Tensor):
                    require(value.device.type == "cpu", "Runtime inspection must stay on CPU")
                    dtype_bytes[str(value.dtype)] += value.numel() * value.element_size()
        counts.append({"scope": group["scope"], "algorithm": group["algorithm"],
                       "tensors": len(group["params"]), "parameters": parameters})
    require(owned_names == set(shapes) and set(optimizer["state"]) == owned_ids,
            "Optimizer omits model tensors or contains unowned state")
    return {"original_tensors": len(original), "original_parameters": sum(math.prod(s) for s in original.values()),
            "new_tensors": len(added), "new_parameters": sum(math.prod(s) for s in added.values()),
            "groups": counts, "state_tensor_bytes": sum(dtype_bytes.values()),
            "state_bytes_by_dtype": dict(dtype_bytes)}


def replay_guard(directory, guard_path=None, *, require_native=True, historical_code_ref=None):
    guard_path = Path(guard_path) if guard_path else Path(str(directory) + "_guard.json")
    report = read(guard_path)
    require(Path(report["output"]).resolve() == directory.resolve(), "Guard output belongs to another run")
    stored = report["guard"]
    guard = CapacityGuard(CapacityPolicy(**stored["policy"]), stored["baseline"])
    if require_native:
        require(guard.policy == CapacityPolicy() and report["host_poll_seconds"] == 2
                and report["profile_poll_seconds"] == 0.25, "Native guard policy/cadence differs")
        command = report["command"]
        require(command.count("--output") == 1 and Path(command[command.index("--output") + 1]).resolve()
                == directory.resolve(), "Guard command output differs")
    provenance = verify_implementation(report["implementation_sha256"], historical_code_ref)
    host_path, profile_path = Path(report["host_samples"]), directory / "profile_samples.jsonl"
    host_events = [json.loads(line) for line in host_path.read_text().splitlines()]
    require(bool(host_events), "Missing host trace")
    for event in host_events:
        guard.observe_host(event)
    for line in profile_path.read_text().splitlines():
        guard.observe_profile(json.loads(line))
    require(guard.report() == stored, "Recorded guard summary differs from exact trace replay")
    require(report["status"] == "completed_within_declared_sampled_limits"
            and report["child_exit_code"] == 0 and guard.stop_reason is None,
            "Run failed or a recorded capacity limit was breached")
    require(guard.driver_samples > 0, "Missing accelerator telemetry")
    return {"path": str(guard_path), "sha256": file_digest(guard_path),
            "host_trace_sha256": file_digest(host_path), "profile_trace_sha256": file_digest(profile_path),
            "exact_replay": True, "implementation_provenance": provenance, **guard.report()}


def validate_generation(report, state, *, rows=None, eos_ids=None, required_answer_histogram=None):
    if required_answer_histogram is not None:
        require(rows is not None and Counter(row["response"] for row in rows) == required_answer_histogram,
                "Native training fixture must contain answers 0 through 7 exactly once")
    history = report["generation_history"]
    interval = report["run_contract"]["settings"]["generation_every"]
    expected_steps = list(range(interval, report["steps"] + 1, interval)) if interval else []
    require([h["step"] for h in history] == expected_steps, "Generation checks missing or reordered")
    consecutive, first_stop, compact = 0, None, []
    prompts = {}
    for check in history:
        examples = check["examples"]
        require(len(examples) == check["evaluated_examples"] == report["train_examples"],
                "Training generation does not cover declared examples")
        if rows is not None:
            require([(e["id"], e["expected"]) for e in examples]
                    == [(r["id"], r["response"]) for r in rows], "Generation identities/targets differ from data")
        for example in examples:
            ids = example["generated_token_ids"]
            require(len(ids) == example["generated_tokens"] and 0 < len(ids)
                    <= report["run_contract"]["settings"]["generation_max_new_tokens"], "Generation token budget differs")
            require(example["hit_token_budget"] == (len(ids) == report["run_contract"]["settings"]["generation_max_new_tokens"]
                    and not example["ended_by_eos"]), "Generation cap status differs")
            if eos_ids is not None:
                require(example["ended_by_eos"] == (ids[-1] in eos_ids)
                        and not any(token in eos_ids for token in ids[:-1]), "Generation EOS status differs")
            prompt = (example["prompt_tokens"], example["prompt_token_ids_sha256"])
            require(prompts.setdefault(example["id"], prompt) == prompt, "Generation original prompt changed across checks")
        scores = [score_answer(e["completion"], e["expected"]) for e in examples]
        require(all(all(example[key] == score[key] for key in score)
                    for example, score in zip(examples, scores)), "Stored answer score differs")
        correct = sum(score["task_correct"] for score in scores)
        histogram = dict(Counter(e["completion"] for e in examples))
        expected_classes = len({e["expected"] for e in examples})
        require(correct == check["task_correct"] and check["task_accuracy"] == correct / len(examples)
                and histogram == check["response_histogram"] and len(histogram) == check["distinct_responses"],
                "Generation summary differs from whole-answer scoring")
        require(check["decoding"]["use_cache"] is False
                and check["decoding"]["context"] == "fixed_original_prompt_and_support"
                and check["decoding"]["max_new_tokens"] == report["run_contract"]["settings"]["generation_max_new_tokens"],
                "Augmented generation contract differs")
        distinct_required = len(required_answer_histogram) if required_answer_histogram is not None else expected_classes
        perfect = correct == len(examples) and len(histogram) == distinct_required
        consecutive = consecutive + 1 if perfect else 0
        if consecutive >= 2 and first_stop is None:
            first_stop = check["step"]
        compact.append({"step": check["step"], "correct": correct, "examples": len(examples),
                        "distinct_responses": len(histogram),
                        "all_ended_by_eos": all(e["ended_by_eos"] for e in examples),
                        "capped_responses": sum(e["hit_token_budget"] for e in examples)})
    require(state["consecutive_perfect"] == consecutive, "Saved generation stop counter differs")
    stop_enabled = report["run_contract"]["settings"]["stop_on_train_perfect"]
    if stop_enabled and first_stop is not None:
        require(report["steps"] == first_stop and report["stop_reason"] == "two_consecutive_perfect_training_generation_checks",
                "Run continued past, or misreported, its two-perfect-check stop")
    else:
        expected_stop = "step_budget_completed" if report["steps"] == report["maximum_step_budget"] else "process_chunk_completed"
        require(report["stop_reason"] == expected_stop, "Budget/chunk stop reason differs")
    return {"checks": compact, "first_two_perfect_check_step": first_stop}


def validate_run(directory, *, require_native=True, verify_payload_bytes=False, historical_code_ref=None):
    directory = Path(directory)
    report_path = directory / "run_report.json"
    report = read(report_path)
    contract = report["run_contract"]
    settings = contract["settings"]
    step = report["steps"]
    require(report["status"] == "completed_local_training_run" and step > 0, "Training did not complete")
    require(settings["regime"] == report["parameters"]["regime"] == "full_augmented", "D2 scope required")
    require(settings["dtype"] == report["parameter_dtype"] == report["compute_dtype"] == "float32", "D2 FP32 contract differs")
    require(not report["cpu_fallback_enabled"], "CPU fallback is forbidden")
    require(all(report["arguments"].get(k) == v for k, v in settings.items()), "Arguments differ from run contract")
    require(report["maximum_step_budget"] == settings["steps"] and 0 <= report["start_step"] < step
            <= settings["steps"], "Completed step/budget interval differs")
    require(report["packages"] == contract["packages"], "Package provenance differs")
    if require_native:
        require(settings["device"] == report["device"] == "mps", "Native qualification requires MPS")
        require(settings["lr_base"] == settings["lr_muon_base"] == 1e-4
                and settings["lr_new"] == settings["lr_muon_new"] == 1e-3
                and settings["warmup_steps"] == 2 and settings["weight_decay"] == 0,
                "Native D2 protocol rates/warmup/decay differ")
        fixed = {"optimizer": "muon8bit", "grad_accum": 8, "momentum": 0.95, "ns_steps": 5,
                 "block_size": 2048, "seed": 17, "memory_size": 64, "workspace_size": 64,
                 "insertion_layer": -1, "checkpointing": True, "sampling": "answer_balanced",
                 "lr_schedule": "constant", "generation_max_new_tokens": 8}
        require(all(settings[k] == v for k, v in fixed.items()), "Native D2 scope/optimizer protocol differs")
        if settings["steps"] == 2:
            require(settings["max_length"] in (256, 512) and not settings["generation_every"]
                    and not settings["stop_on_train_perfect"] and report["arguments"]["checkpoint_every"] == 1,
                    "Capacity length/checkpoint/stop protocol differs")
            expected_length = settings["max_length"]
            require(report["train_examples"] == report["dev_examples"] == 8, "Capacity fixture count differs")
        else:
            require(settings["steps"] == 64 and settings["max_length"] == 256
                    and settings["generation_every"] == 4 and settings["stop_on_train_perfect"]
                    and report["arguments"]["checkpoint_every"] == 16, "Tiny-overfit protocol differs")
            expected_length = 55
            require(report["train_examples"] == 8 and report["dev_examples"] == 16, "Tiny fixture count differs")
        require(report["actual_sequence_lengths"] == {"minimum": expected_length, "maximum": expected_length},
                "Recorded actual sequence length differs from protocol")
        if report["start_step"] == 0:
            require(step == (1 if settings["steps"] == 2 else 4), "Fresh process did not stop at declared split")
    for field, argument in (("source_lock_sha256", "model_lock"), ("data_sha256", "data"), ("dev_sha256", "dev")):
        require(contract[field] == file_digest(Path(report["arguments"][argument])), "Bound input bytes changed")
    require(report["model_lock"] == read(report["arguments"]["model_lock"]), "Source lock metadata differs")
    if require_native:
        require(report["model_lock"]["repo_id"] == "openbmb/MiniCPM5-2B"
                and report["model_lock"]["revision"] == "f97400052a43d642bbc6e9975e2397e3ae6a6b52",
                "Native sealed source identity differs")
    provenance = verify_implementation(contract["implementation_sha256"], historical_code_ref)
    rows = [json.loads(line) for line in Path(report["arguments"]["data"]).read_text().splitlines()]
    batches = list(balanced_batches(rows, settings["grad_accum"], settings["steps"], settings["seed"]))
    require(hashlib.sha256(json.dumps(batches).encode()).hexdigest() == contract["batch_plan_sha256"],
            "Deterministic balanced batch plan differs")
    entries = [entry for entry in report["checkpoint_receipts"] if entry["step"] == step]
    require(len(entries) == 1, "Missing final completed-update checkpoint")
    checkpoint = Path(entries[0]["path"])
    require(checkpoint.resolve() == (directory / f"training_checkpoint_{step:04d}").resolve(), "Final checkpoint path differs")
    receipt, byte_scope = inspect_checkpoint(checkpoint, contract, verify_payload_bytes=verify_payload_bytes)
    receipt_digest = file_digest(checkpoint / "checkpoint_receipt.json")
    require(entries[0]["receipt_sha256"] == receipt_digest, "Receipt digest differs from run report")
    manifest = read(checkpoint / "adapter_manifest.json")
    require(manifest["regime"] == "full_augmented" and manifest["training_scope"] == "text_only", "Checkpoint scope differs")
    require(manifest["parameters"] == report["parameters"], "Checkpoint parameter audit differs from run")
    if require_native:
        require(manifest["adapter_settings"] == {"memory_size": 64, "workspace_size": 64, "slots": 4,
                "workspace_steps": 2, "insertion_layer": -1, "heads": 4}, "Native sidecar settings differ")
    runtime = torch.load(checkpoint / "runtime.pt", map_location="cpu", weights_only=True, mmap=True)
    metadata, optimizer = runtime["metadata"], runtime["optimizer"]
    state, schedule = metadata["data_state"], metadata["schedule_state"]
    require(step == metadata["step"] == state["next_step"] == schedule["completed_steps"], "Runtime cursor/schedule differs")
    require(metadata["run_contract"] == contract and metadata["contract_sha256"] == object_digest(contract), "Runtime contract differs")
    require(state["batch_plan_sha256"] == contract["batch_plan_sha256"], "Runtime batch-plan cursor differs")
    require(state["update_reports"] == report["train_updates"] and state["generation_history"] == report["generation_history"], "Saved histories differ")
    require({"python", "torch_cpu"}.issubset(runtime["rng"]), "Missing Python/CPU RNG state")
    random.Random().setstate(runtime["rng"]["python"])
    torch.Generator(device="cpu").set_state(runtime["rng"]["torch_cpu"])
    require(sorted(k.removeprefix("torch_") for k in runtime["rng"] if k != "python")
            == metadata["rng_devices"], "RNG device metadata differs")
    for name, value in runtime["rng"].items():
        if name != "python":
            require(isinstance(value, torch.Tensor) and value.device.type == "cpu"
                    and value.dtype == torch.uint8 and value.ndim == 1 and value.numel() > 0,
                    "Malformed saved RNG bytes")
    if require_native:
        require("torch_mps" in runtime["rng"] and "mps" in metadata["rng_devices"], "Missing MPS RNG state")
    inventory = optimizer_inventory(optimizer, checkpoint, manifest, step=step, settings=settings)
    if require_native:
        require((inventory["original_parameters"], inventory["original_tensors"], inventory["new_parameters"], inventory["new_tensors"])
                == (2516756480, 381, 895490, 48), "Native D2 parameter/header guard differs")
    for scope in ("original", "new"):
        summary = report["parameters"][scope]
        require(summary["total"] == summary["trainable"] == inventory[scope + "_parameters"]
                and summary["tensor_count"] == inventory[scope + "_tensors"], "Trainable scope differs from headers")
    audit = report["optimizer"]
    expected_audit = {"optimizer": "muon8bit", "implementation": "replica_portable_eager_v1",
                      "parameter_dtype": "float32", "muon_temporary_dtype": "float32",
                      "muon_nesterov": False, "muon_lr_adjustment": "match_rms_adamw",
                      **{key: settings[key] for key in ("momentum", "block_size", "ns_steps")}}
    require(all(audit.get(key) == value for key, value in expected_audit.items()), "Optimizer implementation audit differs")
    require(audit["state_tensor_bytes"] == inventory["state_tensor_bytes"]
            and audit["state_bytes_by_dtype"] == inventory["state_bytes_by_dtype"], "Optimizer byte audit differs")
    require(len(audit["groups"]) == len(inventory["groups"]) and all(all(record[key] == observed[key] for key in ("scope", "algorithm", "tensors", "parameters"))
                for record, observed in zip(audit["groups"], inventory["groups"])), "Optimizer group count audit differs")
    groups = optimizer["param_groups"]
    base_rates = [(settings["lr_muon_base"] if g["algorithm"] == "muon8bit" else settings["lr_base"])
                  if g["scope"] == "original" else
                  (settings["lr_muon_new"] if g["algorithm"] == "muon8bit" else settings["lr_new"]) for g in groups]
    require(schedule["base_rates"] == base_rates, "Original schedule rates differ")
    history = report["train_updates"]
    require([u["step"] for u in history] == list(range(1, step + 1)), "Update history has gaps")
    fake_optimizer = SimpleNamespace(param_groups=[{} for _ in groups])
    gradient_checks = []
    for update in history:
        factor = apply_learning_rates(fake_optimizer, base_rates, update["step"] - 1,
            settings["steps"], settings["warmup_steps"], settings["lr_schedule"], settings["min_lr_ratio"])
        require(update["learning_rates"] == [g["lr"] for g in fake_optimizer.param_groups]
                and update["lr_factor"] == factor, "Update learning rate differs")
        require(all(math.isfinite(update[key]) for key in ("answer_nll", "eos_nll", "train_response_nll_before_update")), "Nonfinite recorded loss")
        require(update["selection"] == batch_diagnostics(rows, batches[update["step"] - 1])
                and update["example_ids"] == update["selection"]["example_ids"], "Update deterministic target coverage differs")
        gradient = update.get("gradient_audit")
        if gradient is not None:
            for scope in ("original", "new"):
                require(gradient[scope]["eligible_tensors"] == gradient[scope]["gradient_tensors"] == inventory[scope + "_tensors"]
                        and not gradient[scope]["missing_gradient_names"], "Missing D2 gradients")
            require(gradient["original"]["nonzero_gradient_tensors"] == inventory["original_tensors"], "Original gradients did not cover full scope")
            expected_new = 2 if update["step"] == 1 else inventory["new_tensors"]
            require(gradient["new"]["nonzero_gradient_tensors"] == expected_new, "New branch gradients did not open")
            gradient_checks.append({"step": update["step"], "original_nonzero": inventory["original_tensors"], "new_nonzero": expected_new})
    require(all(history[i - 1].get("gradient_audit") is not None
                for i in range(report["start_step"] + 1, min(step, report["start_step"] + 2) + 1)),
            "Missing first/later gradient opening audits")
    require(report["first_update_gradient_audit"] == history[report["start_step"]]["gradient_audit"],
            "First process gradient audit differs")
    require([g["lr"] for g in groups] == [g["lr"] for g in fake_optimizer.param_groups], "Saved optimizer rates differ")
    require(report["processed_train_tokens"] == sum(u["processed_tokens"] for u in history)
            and report["supervised_train_tokens"] == sum(u["supervised_tokens"] for u in history), "Token totals differ")
    config = read(checkpoint / "base/config.json")
    eos_ids = config.get("eos_token_id")
    if isinstance(eos_ids, int): eos_ids = [eos_ids]
    generation = validate_generation(report, state, rows=rows, eos_ids=eos_ids,
        required_answer_histogram={str(value): 1 for value in range(8)} if require_native else None)
    require(report["reload_verification"]["status"] == "passed"
            and report["reload_verification"]["maximum_absolute_logit_difference"] == 0,
            "Export/reload did not match exactly")
    require(report["profile"]["errors"] == [], "Profiler errors recorded")
    phases = [event for event in (json.loads(line) for line in (directory / "profile_samples.jsonl").read_text().splitlines())
              if event.get("type") == "phase"]
    encoded = [event["fields"] for event in phases if event.get("phase") == "data_encoded"]
    require(len(encoded) == 1 and encoded[0]["minimum_sequence_length"] == report["actual_sequence_lengths"]["minimum"]
            and encoded[0]["maximum_sequence_length"] == report["actual_sequence_lengths"]["maximum"],
            "Raw encoded-length phase differs")
    restored = [event["fields"]["completed_steps"] for event in phases if event.get("phase") == "runtime_restored"]
    require(restored == ([report["start_step"]] if report["start_step"] else []), "Raw resume phase differs")
    guard = replay_guard(directory, require_native=require_native, historical_code_ref=historical_code_ref)
    result = {"status": "passed", "run": str(directory), "report_sha256": file_digest(report_path),
        "contract_sha256": metadata["contract_sha256"], "start_step": report["start_step"], "completed_steps": step,
        "maximum_step_budget": settings["steps"],
        "source_lock_sha256": contract["source_lock_sha256"],
        "checkpoint": str(checkpoint), "receipt_sha256": receipt_digest,
        "validated_checkpoint_scope": "final_completed_update_checkpoint_in_each_process",
        "checkpoint_logical_bytes": sum((checkpoint / name).stat().st_size for name in receipt["files_sha256"]),
        "checkpoint_byte_verification": byte_scope, "implementation_provenance": provenance,
        "checkpoint_weight_receipt_sha256": {name: digest for name, digest in receipt["files_sha256"].items()
                                             if name.endswith(".safetensors")},
        "scope_and_optimizer_headers": inventory, "current_rates": [g["lr"] for g in groups],
        "runtime": {"next_step": state["next_step"], "schedule_completed_steps": schedule["completed_steps"],
                    "base_rates": base_rates, "rng_devices": metadata["rng_devices"],
                    "update_history_length": len(history), "histories_equal_run_report": True},
        "gradient_checks": gradient_checks, "generation": generation, "stop_reason": report["stop_reason"],
        "reload_verification": report["reload_verification"], "guard": guard,
        "profile_memory": report["profile"]["memory"],
        "resume_from": report["arguments"].get("resume_from"),
        "history": history, "generation_history": report["generation_history"],
        "source_weight_values_checked": False,
        "source_weight_expectation": "D2 trains all original tensors; independent streaming changed-value audit required"}
    del runtime
    return result


def compare_sidecars(before, after):
    """Only the bounded new-module files are value-compared; never native weights."""
    before_path, after_path = Path(before["checkpoint"]) / "sidecars.safetensors", Path(after["checkpoint"]) / "sidecars.safetensors"
    require(before_path.stat().st_size < 16 * 1024**2 and after_path.stat().st_size < 16 * 1024**2,
            "Sidecar comparison exceeds bounded small-module scope")
    for item, path in ((before, before_path), (after, after_path)):
        require(file_digest(path) == item["checkpoint_weight_receipt_sha256"]["sidecars.safetensors"],
                "Sidecar bytes differ from receipt")
    deltas = {}
    with safe_open(before_path, framework="pt", device="cpu") as left, safe_open(after_path, framework="pt", device="cpu") as right:
        require(set(left.keys()) == set(right.keys()), "Sidecar continuation tensor inventory differs")
        for name in left.keys():
            a, b = left.get_tensor(name), right.get_tensor(name)
            require(a.shape == b.shape and bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all()),
                    "Nonfinite/incompatible sidecar values")
            deltas[name] = float((b - a).abs().max())
    changed = sum(value > 0 for value in deltas.values())
    require(changed == len(deltas), "Sidecar continuation left an upstream tensor unchanged")
    return {"before": str(before_path), "after": str(after_path), "tensor_count": len(deltas),
            "changed_tensor_count": changed, "maximum_absolute_delta_by_tensor": deltas,
            "both_files_receipt_hash_verified": True}


def bind_original_audits(paths, results):
    """Bind root's existing streaming native-value audit schema to saved receipts."""
    bound = []
    for path in paths:
        audit = read(path)
        base = Path(audit["exported_base"]).resolve()
        matches = [r for r in results if base in {(Path(r["checkpoint"]) / "base").resolve(),
                                                   (Path(r["run"]) / "checkpoint/base").resolve()}]
        require(len(matches) == 1, "Original-value audit does not identify one validated run")
        result = matches[0]
        require(audit["source_lock_sha256"] == result["source_lock_sha256"]
                and audit["source_normalized_dtype"] == "float32", "Original-value audit source differs")
        require(audit["changed_tensor_count"] == audit["tensor_count"] == result["scope_and_optimizer_headers"]["original_tensors"]
                and audit["parameter_count"] == result["scope_and_optimizer_headers"]["original_parameters"]
                and audit["every_original_tensor_changed"] and audit["exact_native_names_and_shapes_match"],
                "Original-value audit did not change the complete native scope")
        recorded = {"base/" + name: digest for name, digest in audit["exported_checkpoint_sha256"].items()}
        expected = {name: digest for name, digest in result["checkpoint_weight_receipt_sha256"].items() if name.startswith("base/")}
        require(recorded == expected, "Original-value audit bytes do not bind to checkpoint receipt")
        bound.append({"path": str(path), "sha256": file_digest(path), "run": result["run"],
                      "changed_tensor_count": audit["changed_tensor_count"], "receipt_digest_binding": True})
    return bound


def validate_experiment(directories, *, require_native=True, verify_payload_bytes=False,
                        historical_code_ref=None, original_audits=()):
    require(len({Path(p).resolve() for p in directories}) == len(directories), "Repeated run directory")
    results = [validate_run(directory, require_native=require_native, verify_payload_bytes=verify_payload_bytes,
                            historical_code_ref=historical_code_ref) for directory in directories]
    checkpoints = {str(Path(item["checkpoint"]).resolve()): item for item in results}
    sidecar_changes = []
    for item in results:
        if item["resume_from"] is not None:
            parent = checkpoints.get(str(Path(item["resume_from"]).resolve()))
            require(parent is not None, "Include the parent completed chunk in --run inputs")
            require(parent["contract_sha256"] == item["contract_sha256"]
                    and parent["completed_steps"] == item["start_step"], "Resume parent contract/cursor differs")
            require(parent["history"] == item["history"][:parent["completed_steps"]]
                    and parent["generation_history"] == item["generation_history"][:len(parent["generation_history"])],
                    "Resume changed completed update/generation history")
            sidecar_changes.append(compare_sidecars(parent, item))
        else:
            require(item["start_step"] == 0, "Fresh run has nonzero start step")
    require(len({r["contract_sha256"] for r in results if not r["resume_from"]})
            == sum(not r["resume_from"] for r in results), "Repeated fresh runs share one contract")
    roots = [r for r in results if not r["resume_from"]]
    require(bool(roots), "No fresh root in resume chain")
    if require_native:
        for root in roots:
            require(root["maximum_step_budget"] in (2, 64), "Unsupported native protocol budget")
            require(root["completed_steps"] == (1 if root["maximum_step_budget"] == 2 else 4),
                    "Native fresh root did not stop at required restart split")
    resumed_paths = [str(Path(r["resume_from"]).resolve()) for r in results if r["resume_from"]]
    require(len(set(resumed_paths)) == len(resumed_paths), "Resume chain forks from the same checkpoint")
    terminal = [r for r in results if str(Path(r["checkpoint"]).resolve()) not in resumed_paths]
    protocol_complete = all(r["stop_reason"] in {"step_budget_completed", "two_consecutive_perfect_training_generation_checks"}
                            for r in terminal)
    if require_native:
        protocol_complete = protocol_complete and all(str(Path(root["checkpoint"]).resolve()) in resumed_paths
                                                      for root in roots)
    audits = bind_original_audits(original_audits, results)
    for item in results:
        item.pop("history")
        item.pop("generation_history")
    return {"status": "passed", "validated_at_utc": datetime.now(timezone.utc).isoformat(),
            "validation_scope": "CPU receipt metadata/small files, safetensor headers, mmap runtime metadata, bounded sidecar values and exact recorded guard replay",
            "validator_sha256": file_digest(Path(__file__)), "runs": results,
            "sidecar_continuation": sidecar_changes, "bound_original_value_audits": audits,
            "declared_protocol_chains_complete": protocol_complete,
            "original_value_audit_status": "complete" if {a["run"] for a in audits} == {r["run"] for r in results} else "incomplete_external_audits",
            "full_payload_bytes_verified": verify_payload_bytes,
            "model_instantiated": False, "gpu_executed": False, "native_tensor_value_comparisons_performed": False,
            "limitations": ["Does not establish native uninterrupted-versus-resumed numerical equivalence",
                "Native original tensor values require separately byte-bound streaming audits; sidecars only are compared here",
                "Default metadata mode does not rehash native weights/runtime; use --verify-payload-bytes for full receipt verification",
                "Declared source/data metadata and local receipts are not publisher authentication",
                "Sampled guards can miss transients; host metrics are system-wide; MPS/RSS overlap",
                "Generation checks are training diagnostics, not independent generalization scores"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-payload-bytes", action="store_true",
                        help="Rehash every model/runtime checkpoint payload; expensive native I/O")
    parser.add_argument("--historical-code-ref", help="Verify all bound implementation files against this local Git commit")
    parser.add_argument("--original-audit", type=Path, action="append", default=[],
                        help="Bind an existing streaming native changed-value audit to a validated receipt")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new report path")
    report = {"status": "started", "requested_runs": [str(path) for path in args.run],
              "validation_options": {"verify_payload_bytes": args.verify_payload_bytes,
                                     "historical_code_ref": args.historical_code_ref,
                                     "original_audits": [str(path) for path in args.original_audit]}}
    try:
        report.update(validate_experiment(args.run, verify_payload_bytes=args.verify_payload_bytes,
            historical_code_ref=args.historical_code_ref, original_audits=args.original_audit))
    except BaseException as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "runs": len(report["runs"]), "report": str(args.output)}))


if __name__ == "__main__":
    main()

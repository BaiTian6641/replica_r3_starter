#!/usr/bin/env python3
"""Train, reload, independently score and finalize the staged curriculum.

Modes (each is one process; train/reload/finalize load the native model):

* ``train --arm {muon8bit,adamw_control}``: fresh seeded 4096-wide projector,
  symmetric dev-only LR probe, 128 staged updates, pre/post text retention,
  dev generations and the narrow external-caption diagnostic. No audit access.
* ``reload --run RUN``: fresh process, zero updates, strict extension reload,
  bitwise dev and retention generation equality.
* ``score --run RUN``: no model; independent scoring from artifacts only.
* ``finalize --run RUN --score-report SCORE``: once, after the exact dev report
  hash, score the sealed procedural audit and the untouched external final
  split. Repeat finalize is refused.

The native entry follows the declared one-worker exclusive MPS guard and the
CapacityGuard host/swap/driver thresholds. There is no CPU fallback, no model
download and no base-weight save.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
    os.environ[key] = "1"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
os.environ.setdefault("REPLICA_CAPACITY_GUARD", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from replica_lab.vision_curriculum import (  # noqa: E402
    ARMS, CONTROLS, PROJECTOR_PARAMETER_COUNT, PROJECTOR_TENSORS,
    RELOAD_REPORT_FORMAT, RUN_REPORT_FORMAT, SCORE_REPORT_FORMAT,
    UNKNOWN_ANSWER, VisionCurriculumError, _required_run_checks,
    answer_correct, assemble_score_report, assign_controls, audit_dataset,
    build_finalize_receipt, build_protocol, curriculum_prompt, dataset_pins,
    finalize_receipt_paths, is_unknown_answer, load_config,
    load_external_source, load_protocol, load_run_report, protocol_sha256,
    require_finalize_allowed, require_global_finalize_allowed,
    reserve_finalize, retention_policy, score_caption, score_procedural,
    sha256_bytes, sha256_file, validate_config, validate_finalize_receipt,
    validate_generation_rows, validate_lr_reports, validate_reload_report,
    validate_run_report, validate_score_report, verify_data_pins,
    verify_implementation_pins, verify_retention_pins,
    write_finalize_receipt,
)
from replica_lab.capacity_guard import (  # noqa: E402
    CapacityGuard, CapacityPolicy, parse_host_readings,
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VisionCurriculumError(message)


def dump(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("".join(json.dumps(row, allow_nan=False) + "\n"
                                  for row in rows))


def read_jsonl(path):
    return [json.loads(line) for line in
            Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def ensure_fresh_directory(path):
    path = Path(path)
    if path.exists():
        require(path.is_dir() and not any(path.iterdir()),
                f"Use a fresh output directory: {path}")
    else:
        path.mkdir(parents=True)
    return path


def environment_report():
    import importlib.metadata
    packages = {}
    for name in ("torch", "transformers", "safetensors"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return packages


def host_reading():
    def run(args):
        return subprocess.run(args, capture_output=True, text=True,
                              check=True, timeout=2).stdout
    return parse_host_readings(run(["/usr/sbin/sysctl", "-n", "vm.swapusage"]),
                               run(["/usr/bin/memory_pressure", "-Q"]))


class Guard:
    """One-worker guard: exclusive lock, host limits and the 50 GiB driver cap."""

    def __init__(self, protocol):
        policy = protocol["resource_policy"]
        self.capacity = CapacityGuard(
            CapacityPolicy(
                driver_limit_bytes=int(policy["driver_limit_bytes"]),
                swap_growth_limit_bytes=int(
                    policy["sustained_swap_growth_bytes"]),
                minimum_free_percent=float(
                    policy["minimum_host_free_percent"]),
                sustained_host_samples=int(policy["sustained_host_samples"])),
            host_reading())
        self.protocol = protocol
        self.events = []
        self.driver_peak = 0
        self.sample(index=0, event="start")

    def _driver_bytes(self):
        try:
            import torch
            if torch.backends.mps.is_available():
                return int(torch.mps.driver_allocated_memory())
        except (RuntimeError, AttributeError):
            return None
        return None

    def sample(self, *, index, event):
        reading = host_reading()
        stop = self.capacity.observe_host(reading)
        driver = self._driver_bytes()
        if driver is not None:
            self.driver_peak = max(self.driver_peak, driver)
            self.capacity.observe_profile({"mps_driver_bytes": driver})
        self.events.append({"index": index, "event": event,
                            "swap_used_bytes": reading["swap_used_bytes"],
                            "memory_free_percent": reading["memory_free_percent"],
                            "mps_driver_bytes": driver})
        stop = stop or self.capacity.stop_reason
        require(stop is None, f"Capacity guard stop: {stop}")

    def report(self):
        return {"policy": asdict(self.capacity.policy),
                "baseline": dict(self.capacity.baseline),
                "peak_driver_bytes": self.driver_peak,
                "samples": len(self.events),
                "events": self.events[-64:],
                "stop_reason": self.capacity.stop_reason,
                "workers": 1,
                "cpu_fallback": False}


# ---------------------------------------------------------------------------
# Protocol / data / model verification
# ---------------------------------------------------------------------------


def resolve_project_path(protocol, value):
    path = Path(protocol["root"]) / value
    return path if path.is_absolute() else ROOT / path


def load_protocol_and_data(args):
    protocol = load_protocol(args.protocol)
    protocol_sha = sha256_file(args.protocol)
    implementation = verify_implementation_pins(protocol, ROOT)
    data_root = resolve_project_path(protocol, protocol["data"]["root"])
    dataset = audit_dataset(data_root)
    verify_data_pins(dataset_pins(data_root), data_root)
    require(dataset.manifest["_manifest_sha256"]
            == protocol["data"]["manifest"]["sha256"],
            "Protocol manifest pin differs from prepared data")
    for name in ("train_seal_sha256", "dev_seal_sha256", "audit_seal_sha256"):
        key = {"train_seal_sha256": "train", "dev_seal_sha256": "dev",
               "audit_seal_sha256": "audit"}[name]
        require(dataset.seals[name] == protocol["data"]["seals"][key],
                f"Protocol {key} seal differs")
    require(dataset.seals["external_final_seal_sha256"]
            == protocol["data"]["seals"]["external_final"],
            "Protocol external final seal differs")
    require(dataset.seals["audit_scored"] is False
            and dataset.seals["external_final_consumed"] is False,
            "Final evidence is already marked as consumed")
    verify_retention_pins(protocol, ROOT)
    return protocol, protocol_sha, dataset, implementation


def protocol_scope(protocol):
    from replica_lab.memory_system.ledger import EligibilityScope
    raw = protocol["vision_source"]["scope"]
    return EligibilityScope(raw["subject_id"], raw["branch_id"],
                            raw["purpose"], raw["as_of"], raw["mode"],
                            raw.get("snapshot_at"))


def verify_source_and_spec(protocol, device):
    from replica_lab.architecture.contracts import VisionSpec
    from replica_lab.architecture.vision import (load_vision_source,
                                                vision_spec_from_source)
    source_path = resolve_project_path(protocol,
                                       protocol["vision_source"]["path"])
    manifest_path = resolve_project_path(
        protocol, protocol["vision_source"]["manifest"]["path"])
    require(source_path.exists(), "Pinned vision source is missing")
    require(manifest_path.is_file(), "Pinned vision manifest is missing")
    require(sha256_file(manifest_path)
            == protocol["vision_source"]["manifest"]["sha256"],
            "Vision manifest bytes changed")
    source = load_vision_source(source_path)
    spec = vision_spec_from_source(
        source, projector_hidden=protocol["vision"]["projector_hidden"])
    spec.validate()
    fields = ("encoder_repo", "encoder_revision", "encoder_source_digest",
              "processor_digest", "encoder_width", "image_size", "patch_size",
              "max_visual_tokens", "projector_hidden", "feature_version",
              "compute_dtype")
    declared = VisionSpec(**{key: protocol["vision"][key] for key in fields})
    declared.validate()
    require(dict(asdict(spec)) == dict(asdict(declared)),
            "Loaded vision spec differs from the protocol")
    require(spec.digest == protocol["vision_digest"],
            "Vision digest differs from the protocol")
    return source, spec, manifest_path


def feature_service(protocol, device):
    from replica_lab.architecture.vision import VisionFeatureService
    source_path = resolve_project_path(protocol,
                                       protocol["vision_source"]["path"])
    manifest_path = resolve_project_path(
        protocol, protocol["vision_source"]["manifest"]["path"])
    return VisionFeatureService(
        source_path, manifest_path, device=device,
        max_cache_bytes=int(protocol["limits"]["max_cache_bytes"]),
        projector_hidden=int(protocol["vision"]["projector_hidden"]))


def make_session(adapter, tokenizer, output):
    from replica_lab.memory_system.ledger import RecordLedger
    from replica_lab.memory_system.session import MemorySession
    ledger = RecordLedger(Path(output) / "ledger.sqlite")
    return MemorySession(adapter, tokenizer, ledger)


def task_graph(task_id, text):
    from replica_lab.architecture.contracts import TaskGraph
    return TaskGraph(task_id=task_id, goal=text)


def prompt_ids(tokenizer, text, device, maximum):
    from replica_lab.architecture.loading import encode_prompt
    ids = encode_prompt(tokenizer, text, device)
    require(ids.shape[1] <= maximum,
            f"Prompt exceeds the declared budget: {ids.shape[1]} > {maximum}")
    return ids


def observe(service, *, image_path, image_sha256, case_id, scope,
            ledger_epoch, spec, source_id):
    visual = service.observe(str(image_path), source_id=source_id,
                             task_id=case_id, scope=scope, observed_at=0,
                             source_epoch=ledger_epoch)
    require(visual.producer == spec.digest,
            "Observed visual producer differs from the spec")
    require(visual.image_sha256 == image_sha256,
            "Observed image hash differs from the prepared row")
    require(visual.features.requires_grad is False
            and bool(visual.features.isfinite().all()),
            "Visual features must be finite and detached")
    visual.assert_immutable()
    return visual


# ---------------------------------------------------------------------------
# Fresh initialization and arm optimizers
# ---------------------------------------------------------------------------


def initialize_projector(projector, seed):
    """Fresh seeded 4096-wide projector; never warm-started from V1 weights."""
    import torch
    from torch import nn
    torch.manual_seed(int(seed))
    for module in projector:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    return projector_state_sha256(projector)


def projector_state_sha256(projector):
    import torch
    digest = hashlib.sha256()
    state = projector.state_dict()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def freeze_projector_only(adapter):
    adapter.set_regime("modules_only")
    trainable = []
    frozen = []
    for name, parameter in adapter.named_parameters():
        allowed = name.startswith("vision_projector.")
        parameter.requires_grad_(allowed)
        (trainable if allowed else frozen).append(name)
    require(sorted(trainable) == sorted(PROJECTOR_TENSORS),
            "Trainable scope must be exactly the six projector tensors")
    require(adapter.vision_projector is not None,
            "The loaded adapter lacks the vision projector")
    require(sum(p.numel() for p in adapter.vision_projector.parameters())
            == PROJECTOR_PARAMETER_COUNT,
            "Projector parameter count differs from 11,542,016")
    require(not any(parameter.requires_grad
                    for name, parameter in adapter.named_parameters()
                    if not name.startswith("vision_projector.")),
            "A non-projector tensor remains trainable")
    return trainable, frozen


def build_optimizer(adapter, arm, selected_lr, config):
    from replica_lab.optimizers import ReplicaOptimizer
    from replica_lab.vision_qualification import projector_optimizer_groups
    projector = adapter.vision_projector
    if arm == "muon8bit":
        train = {"lr_new": float(selected_lr),
                 "momentum": float(config["arms"]["muon8bit"]["momentum"]),
                 "ns_steps": int(config["arms"]["muon8bit"]["ns_steps"]),
                 "block_size": int(config["arms"]["muon8bit"]["block_size"]),
                 "weight_decay": float(config["arms"]["muon8bit"]
                                       ["weight_decay"])}
        aux_lr = float(config["arms"]["muon8bit"]["aux_lr"])
        require(aux_lr == 1e-4,
                "The Muon auxiliary AdamW LR must stay fixed at the preregistered 1e-4")
        groups = projector_optimizer_groups(projector, train,
                                            prefix="vision_projector.")
        # Only the 2D matrix route is probe-selected. The FP32 AdamW auxiliary
        # groups keep the preregistered fixed LR.
        for group in groups:
            if group["algorithm"] == "adamw":
                group["lr"] = aux_lr
        optimizer = ReplicaOptimizer(groups, "muon8bit")
        for group in optimizer.param_groups:
            if group["algorithm"] == "adamw":
                group["lr"] = aux_lr
    elif arm == "adamw_control":
        named = dict(projector.named_parameters())
        groups = [{
            "params": [named[name.split(".", 1)[1]]
                       for name in PROJECTOR_TENSORS],
            "param_names": list(PROJECTOR_TENSORS),
            "scope": "vision_projector",
            "group_name": "vision_projector_adamw_control",
            "algorithm": "adamw",
            "lr": float(selected_lr),
            "weight_decay": float(config["arms"]["adamw_control"]
                                  ["weight_decay"]),
            "betas": (0.9, 0.999), "eps": float(config["arms"]["adamw_control"]
                                                ["eps"]),
        }]
        optimizer = ReplicaOptimizer(groups, "adamw")
    else:
        raise VisionCurriculumError(f"Unknown arm: {arm}")
    route = []
    for group in groups:
        for name, parameter in zip(group["param_names"], group["params"]):
            route.append({"name": name, "shape": list(parameter.shape),
                          "algorithm": group["algorithm"],
                          "group_name": group["group_name"]})
    require(len({item["name"] for item in route}) == len(route),
            "Optimizer route names repeat")
    require(sorted(item["name"] for item in route) == sorted(PROJECTOR_TENSORS),
            "Optimizer route does not cover the projector")
    if arm == "muon8bit":
        require({item["algorithm"] for item in route} == {"muon8bit", "adamw"},
                "Muon arm must route both algorithms")
        for item in route:
            expected = "muon8bit" if len(item["shape"]) == 2 else "adamw"
            require(item["algorithm"] == expected,
                    f"Wrong Muon route for {item['name']}")
    else:
        require({item["algorithm"] for item in route} == {"adamw"},
                "AdamW control must route every projector tensor to AdamW")
    state = optimizer.state_dict()
    state_dtypes = set()
    for entry in state.get("state", {}).values():
        for value in entry.values():
            tensor = getattr(value, "dtype", None)
            if tensor is not None:
                state_dtypes.add(str(tensor))
    audit = {
        "optimizer": "muon8bit" if arm == "muon8bit" else "adamw",
        "parameter_dtype": "float32",
        "state_dtype": ("float32" if arm == "adamw_control"
                        else "int8_momentum+float32_aux"),
        "state_dtypes": sorted(state_dtypes),
        "groups": [{"group_name": group["group_name"],
                    "algorithm": group["algorithm"],
                    "param_names": list(group["param_names"]),
                    "lr": float(group["lr"])}
                   for group in groups],
    }
    return optimizer, route, audit


# ---------------------------------------------------------------------------
# Batches and native training
# ---------------------------------------------------------------------------


def batch_lookup(dataset, schedule):
    examples = {row["case_id"]: row for row in dataset.training_examples()}
    require(len(examples) == len(dataset.training_examples()),
            "Training case ids repeat")
    batches = []
    for stage in schedule["stages"]:
        for batch in stage["batches"]:
            rows = []
            for case_id in batch:
                require(case_id in examples,
                        f"Schedule references a non-train case: {case_id}")
                rows.append(examples[case_id])
            batches.append(rows)
    probe_batches = []
    for batch in schedule["probe"]["batches"]:
        probe_batches.append([examples[case_id] for case_id in batch])
    return batches, probe_batches, examples


def _example_tensors(session, tokenizer, row, *, maximum_prompt,
                     maximum_sequence):
    import torch
    device = session.device
    prompt = curriculum_prompt(row["question"])
    prompt_tensor = prompt_ids(tokenizer, prompt, device, maximum_prompt)
    response = tokenizer(row["response"], add_special_tokens=False)["input_ids"]
    require(bool(response), "Training response tokenizes to nothing")
    require(tokenizer.eos_token_id is not None,
            "Tokenizer requires an explicit EOS token")
    response = list(response) + [int(tokenizer.eos_token_id)]
    input_ids = torch.cat((prompt_tensor,
                           torch.tensor([response], dtype=torch.long,
                                        device=prompt_tensor.device)), dim=1)
    require(input_ids.shape[1] <= maximum_sequence,
            "Training example exceeds the declared sequence budget")
    labels = torch.tensor([[-100] * prompt_tensor.shape[1] + response],
                          dtype=torch.long, device=prompt_tensor.device)
    require(not bool((labels[:, :prompt_tensor.shape[1]] != -100).any()),
            "Prompt labels are not masked")
    return prompt_tensor, input_ids, labels


def run_updates(adapter, session, tokenizer, service, protocol, scope, batches,
                optimizer, guard, trace_path, *, label, max_seconds,
                start_time):
    import torch
    limits = protocol["limits"]
    adapter.train()
    trace = []
    completed = 0
    total_examples = 0
    for update_index, rows in enumerate(batches, 1):
        if time.monotonic() - start_time > max_seconds:
            raise VisionCurriculumError("Training time budget exhausted")
        optimizer.zero_grad(set_to_none=True)
        losses = []
        gradient_names = set()
        for row in rows:
            visual = observe(
                service, image_path=row["image_path"],
                image_sha256=row["image_sha256"], case_id=row["case_id"],
                scope=scope, ledger_epoch=session.ledger.epoch,
                spec=service.spec, source_id=protocol["vision_source"]["source_id"])
            prompt_tensor, input_ids, labels = _example_tensors(
                session, tokenizer, row,
                maximum_prompt=limits["max_prompt_tokens"],
                maximum_sequence=limits["max_sequence_tokens"])
            prompt_text = curriculum_prompt(row["question"])
            with session.training_request(
                    prompt_ids=prompt_tensor,
                    prompt_mask=torch.ones_like(prompt_tensor, dtype=torch.bool),
                    query=prompt_text, scope=scope,
                    task=task_graph(row["case_id"], prompt_text),
                    visual=visual, enabled_ports=(),
                    core_impl="none", enabled_branches=("memory",)) as request:
                require(request.frame.sense.visual is visual,
                        "Training request lost its pinned visual payload")
                result = session.forward(
                    request, input_ids,
                    attention_mask=torch.ones_like(input_ids, dtype=torch.bool),
                    labels=labels)
                require(result.logits.shape[1] == input_ids.shape[1],
                        "Training logits are not text-aligned")
                loss = result.loss
                require(loss is not None and bool(torch.isfinite(loss)),
                        "Training loss is unavailable or nonfinite")
                require(loss.requires_grad,
                        "Training loss is detached from autograd")
                session.backward(request, loss / len(rows))
                losses.append(float(loss.detach().cpu()))
            for name, parameter in adapter.named_parameters():
                if parameter.grad is not None:
                    gradient_names.add(name)
        nonprojector = sorted(name for name in gradient_names
                              if not name.startswith("vision_projector."))
        require(not nonprojector,
                f"Non-projector tensors received gradients: {nonprojector}")
        require(gradient_names == set(PROJECTOR_TENSORS),
                "Projector gradient inventory is incomplete")
        norm = float(torch.nn.utils.clip_grad_norm_(
            adapter.vision_projector.parameters(),
            protocol["train"]["max_grad_norm"],
            error_if_nonfinite=True).detach().cpu())
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        completed = update_index
        total_examples += len(rows)
        entry = {
            "label": label,
            "update": update_index,
            "case_ids": [row["case_id"] for row in rows],
            "sources": [row["source"] for row in rows],
            "losses": losses,
            "loss": sum(losses) / len(losses),
            "grad_norm": norm,
            "examples": len(rows),
            "response_only_labels": True,
            "nonprojector_gradient_names": nonprojector,
            "projector_gradient_names": sorted(gradient_names),
            "base_stamp_unchanged": (
                adapter.parameter_stamp(original_only=True)
                == adapter._initial_base_stamp),
        }
        require(entry["base_stamp_unchanged"],
                "An update changed the sealed native base")
        trace.append(entry)
        write_jsonl(trace_path, trace)
        if update_index % 8 == 0:
            guard.sample(index=update_index, event="train")
    return {"trace": trace, "completed": completed,
            "examples": total_examples}


def backward_only_probe(adapter, session, tokenizer, service, protocol, scope,
                        example):
    """One forward/backward without any optimizer step, explicitly labelled."""
    import torch
    adapter.zero_grad(set_to_none=True)
    visual = observe(service, image_path=example["image_path"],
                     image_sha256=example["image_sha256"],
                     case_id=example["case_id"], scope=scope,
                     ledger_epoch=session.ledger.epoch, spec=service.spec,
                     source_id=protocol["vision_source"]["source_id"])
    prompt_tensor, input_ids, labels = _example_tensors(
        session, tokenizer, example,
        maximum_prompt=protocol["limits"]["max_prompt_tokens"],
        maximum_sequence=protocol["limits"]["max_sequence_tokens"])
    prompt_text = curriculum_prompt(example["question"])
    with session.training_request(
            prompt_ids=prompt_tensor,
            prompt_mask=torch.ones_like(prompt_tensor, dtype=torch.bool),
            query=prompt_text, scope=scope,
            task=task_graph(example["case_id"], prompt_text),
            visual=visual, enabled_ports=(), core_impl="none",
            enabled_branches=("memory",)) as request:
        result = session.forward(
            request, input_ids,
            attention_mask=torch.ones_like(input_ids, dtype=torch.bool),
            labels=labels)
        loss = result.loss
        require(loss is not None and bool(torch.isfinite(loss)),
                "Backward probe loss is invalid")
        session.backward(request, loss)
    present = sorted(name for name, parameter in adapter.named_parameters()
                     if parameter.grad is not None)
    nonprojector = [name for name in present
                    if not name.startswith("vision_projector.")]
    finite = all(bool(torch.isfinite(parameter.grad).all())
                 for name, parameter in adapter.named_parameters()
                 if parameter.grad is not None)
    nonzero = all(bool((parameter.grad != 0).any())
                  for name, parameter in adapter.named_parameters()
                  if parameter.grad is not None)
    adapter.zero_grad(set_to_none=True)
    require(not nonprojector,
            f"Backward probe produced non-projector gradients: {nonprojector}")
    require(sorted(present) == sorted(PROJECTOR_TENSORS) and finite,
            "Backward probe did not exercise every projector gradient")
    require(nonzero, "Backward probe produced only zero projector gradients")
    return {"backward_only_probe": True, "probe_applied_update": False,
            "probe_label": "backward_only_no_update",
            "loss": float(loss.detach().cpu()),
            "projector_gradient_names": present,
            "nonprojector_gradient_names": nonprojector,
            "projector_gradients_finite": finite,
            "projector_gradients_nonzero": nonzero}


def dev_loss(adapter, session, tokenizer, service, protocol, scope, cases):
    import torch
    total = 0.0
    losses = []
    adapter.eval()
    with torch.no_grad():
        for case in cases:
            visual = observe(service, image_path=case["image_path"],
                             image_sha256=case["image_sha256"],
                             case_id=case["case_id"], scope=scope,
                             ledger_epoch=session.ledger.epoch,
                             spec=service.spec,
                             source_id=protocol["vision_source"]["source_id"])
            example = {**case, "response": case["label"]}
            prompt_tensor, input_ids, labels = _example_tensors(
                session, tokenizer, example,
                maximum_prompt=protocol["limits"]["max_prompt_tokens"],
                maximum_sequence=protocol["limits"]["max_sequence_tokens"])
            prompt_text = curriculum_prompt(case["question"])
            with session.training_request(
                    prompt_ids=prompt_tensor,
                    prompt_mask=torch.ones_like(prompt_tensor, dtype=torch.bool),
                    query=prompt_text, scope=scope,
                    task=task_graph(case["case_id"], prompt_text),
                    visual=visual, enabled_ports=(), core_impl="none",
                    enabled_branches=("memory",)) as request:
                result = session.forward(
                    request, input_ids,
                    attention_mask=torch.ones_like(input_ids, dtype=torch.bool),
                    labels=labels)
            loss = float(result.loss.detach().cpu())
            require(math.isfinite(loss), "Dev response loss is nonfinite")
            losses.append(loss)
            total += loss
    adapter.train()
    return total / len(losses), losses


def select_learning_rate(adapter, session, tokenizer, service, protocol, scope,
                         config, arm, probe_batches, probe_cases, guard,
                         start_time, trace_dir):
    candidates = [float(item) for item in
                  config["lr_probe"]["candidates"][arm]]
    results = []
    for candidate in sorted(set(candidates)):
        projector = adapter.vision_projector
        init_hash = initialize_projector(
            projector, config["train"]["init_seed"])
        optimizer, route, audit = build_optimizer(adapter, arm, candidate,
                                                  config)
        adapter._initial_base_stamp = adapter.parameter_stamp(
            original_only=True)
        Path(trace_dir).mkdir(parents=True, exist_ok=True)
        run_updates(adapter, session, tokenizer, service, protocol, scope,
                    probe_batches, optimizer, guard,
                    Path(trace_dir) / f"probe_{candidate}.jsonl",
                    label=f"probe_{candidate}",
                    max_seconds=protocol["train"]["max_seconds"],
                    start_time=start_time)
        mean_loss, losses = dev_loss(adapter, session, tokenizer, service,
                                     protocol, scope, probe_cases)
        results.append({"lr": candidate, "dev_loss": mean_loss,
                        "dev_losses": losses, "init_sha256": init_hash,
                        "route": route, "optimizer_audit": audit})
        results[-1]["pass"] = True
    best = min(results, key=lambda item: (item["dev_loss"], item["lr"]))
    return {"arm": arm, "split": "procedural_dev",
            "measure": "mean_dev_response_loss",
            "tie_break": "lowest_lr",
            "candidates": sorted(set(candidates)),
            "results": results, "selected_lr": best["lr"],
            "selected_dev_loss": best["dev_loss"],
            "probe_case_ids": [case["case_id"] for case in probe_cases]}


# ---------------------------------------------------------------------------
# Generation and retention
# ---------------------------------------------------------------------------


def _generated_text(tokenizer, ids):
    return tokenizer.decode([int(value) for value in ids],
                            skip_special_tokens=True)


def generate_row(session, tokenizer, *, control, case, question, visual,
                 scope, maximum_prompt, maximum_new, image_case_id=None,
                 image_sha256=None):
    ids = prompt_ids(tokenizer, question, session.device, maximum_prompt)
    mask = __import__("torch").ones_like(ids, dtype=__import__("torch").bool)
    request = session.prepare(
        prompt_ids=ids, prompt_mask=mask, query=question, scope=scope,
        task=task_graph(case["case_id"], question), training=False,
        visual=visual, enabled_ports=(), core_impl="none",
        enabled_branches=("memory",))
    try:
        token_ids = session.generate(request, max_new_tokens=maximum_new)
        prompt_len = int(request.frame.sense.prompt_ids.shape[1])
        generated = [int(value) for value in
                     token_ids[0, prompt_len:].tolist()]
    finally:
        request.release()
    require(1 <= len(generated) <= maximum_new,
            "Generation exceeded its declared token budget")
    return {"control": control, "case_id": case["case_id"],
            "generated_ids": generated,
            "generated_text": _generated_text(tokenizer, generated),
            "prompt_sha256": sha256_bytes(
                json.dumps([int(v) for v in ids[0].tolist()]).encode()),
            "image_sha256": (None if image_case_id is None
                             else (case["image_sha256"]
                                   if image_sha256 is None else image_sha256)),
            "image_case_id": image_case_id,
            "ended_by_eos": generated[-1] == tokenizer.eos_token_id,
            "max_new_tokens": maximum_new}


def evaluate_generations(adapter, session, tokenizer, service, protocol,
                         scope, cases, output_path, *, seed_offset, guard=None,
                         start_time=None):
    from replica_lab.vision_curriculum import CONTROLS as DECLARED
    controls = list(protocol["evaluation"]["controls"])
    require(controls == list(DECLARED), "Control inventory differs")
    assignment = assign_controls(cases, seed_offset=seed_offset)
    case_by_id = {case["case_id"]: case for case in cases}
    maximum_prompt = protocol["limits"]["max_prompt_tokens"]
    maximum_new = int(protocol["evaluation"]["max_new_tokens"])
    require(maximum_new == 8,
            "Procedural evaluation must use the declared 8-token budget")
    rows = []
    adapter.eval()
    started = time.monotonic() if start_time is None else start_time
    for position, case in enumerate(cases, 1):
        require(time.monotonic() - started
                <= protocol["limits"]["max_seconds"],
                "Evaluation exceeded the declared wall-time ceiling")
        if guard is not None and position % 10 == 0:
            guard.sample(index=position, event="evaluate")
        question = curriculum_prompt(case["question"])
        specs = {
            "correct": (case["case_id"], case["case_id"]),
            "wrong": (assignment["wrong"][case["case_id"]], case["case_id"]),
            "shuffled": (assignment["shuffled"][case["case_id"]],
                         case["case_id"]),
            "noimage": (None, case["case_id"]),
        }
        for control in controls:
            image_case_id, _ = specs[control]
            if image_case_id is None:
                visual = None
            else:
                image_case = case_by_id[image_case_id]
                visual = observe(
                    service, image_path=image_case["image_path"],
                    image_sha256=image_case["image_sha256"],
                    case_id=case["case_id"], scope=scope,
                    ledger_epoch=session.ledger.epoch, spec=service.spec,
                    source_id=protocol["vision_source"]["source_id"])
            used_hash = (case["image_sha256"] if image_case_id is None
                         else case_by_id[image_case_id]["image_sha256"])
            rows.append(generate_row(
                session, tokenizer, control=control, case=case,
                question=question, visual=visual, scope=scope,
                maximum_prompt=maximum_prompt, maximum_new=maximum_new,
                image_case_id=image_case_id, image_sha256=used_hash))
            write_jsonl(output_path, rows)
    for control in controls:
        require(sum(row["control"] == control for row in rows) == len(cases),
                f"Control {control} is incomplete")
    return rows


def caption_evaluation(adapter, session, tokenizer, service, protocol, scope,
                       cases, output_path, *, guard=None, start_time=None):
    import torch
    maximum_prompt = protocol["limits"]["max_prompt_tokens"]
    maximum_new = int(protocol["evaluation"]["caption_max_new_tokens"])
    rows = []
    adapter.eval()
    started = time.monotonic() if start_time is None else start_time
    with torch.no_grad():
        for case in cases:
            require(time.monotonic() - started
                    <= protocol["limits"]["max_seconds"],
                    "Caption evaluation exceeded the wall-time ceiling")
            visual = observe(service, image_path=case["image_path"],
                             image_sha256=case["image_sha256"],
                             case_id=case["case_id"], scope=scope,
                             ledger_epoch=session.ledger.epoch,
                             spec=service.spec,
                             source_id=protocol["vision_source"]["source_id"])
            question = curriculum_prompt(case["question"])
            generated = generate_row(
                session, tokenizer, control="correct", case=case,
                question=question, visual=visual, scope=scope,
                maximum_prompt=maximum_prompt, maximum_new=maximum_new,
                image_case_id=case["case_id"])
            prompt_tensor = prompt_ids(tokenizer, question, session.device,
                                       maximum_prompt)
            response = tokenizer(case["label"], add_special_tokens=False)[
                "input_ids"] + [int(tokenizer.eos_token_id)]
            input_ids = torch.cat(
                (prompt_tensor, torch.tensor([response], dtype=torch.long,
                                             device=prompt_tensor.device)),
                dim=1)
            require(input_ids.shape[1] <= protocol["limits"]
                    ["max_sequence_tokens"],
                    "Caption response exceeds the sequence budget")
            labels = torch.tensor([[-100] * prompt_tensor.shape[1] + response],
                                  dtype=torch.long, device=prompt_tensor.device)
            request = session.prepare(
                prompt_ids=prompt_tensor,
                prompt_mask=torch.ones_like(prompt_tensor, dtype=torch.bool),
                query=question, scope=scope,
                task=task_graph(case["case_id"], question), training=False,
                visual=visual, enabled_ports=(), core_impl="none",
                enabled_branches=("memory",))
            try:
                with torch.no_grad():
                    result = session.forward(
                        request, input_ids,
                        attention_mask=torch.ones_like(input_ids,
                                                       dtype=torch.bool),
                        labels=labels)
            finally:
                request.release()
            generated["response_loss"] = float(result.loss.detach().cpu())
            rows.append(generated)
            write_jsonl(output_path, rows)
    return rows


def load_retention(protocol):
    prompts = []
    for row in read_jsonl(resolve_project_path(
            protocol, protocol["retention"]["prompts"]["path"])):
        prompts.append(row)
    labels = {row["case_id"]: row["answer"] for row in read_jsonl(
        resolve_project_path(protocol,
                             protocol["retention"]["labels"]["path"]))}
    require(len(prompts) == int(protocol["retention"]["max_cases"]),
            "Retention prompt count differs")
    require(set(labels) == {row["case_id"] for row in prompts},
            "Retention labels differ from prompts")
    return prompts, labels


def evaluate_retention(adapter, session, tokenizer, protocol, output_path,
                       *, label):
    from replica_lab.vision_qualification import score_retention
    prompts, labels = load_retention(protocol)
    maximum_prompt = protocol["limits"]["max_prompt_tokens"]
    maximum_new = int(protocol["evaluation"]["max_new_tokens"])
    require(maximum_new == 8,
            "Retention must use the declared 8-token budget")
    rows = []
    adapter.eval()
    for prompt in prompts:
        ids = prompt_ids(tokenizer, prompt["prompt"], session.device,
                         maximum_prompt)
        mask = __import__("torch").ones_like(ids, dtype=__import__("torch").bool)
        request = session.prepare(
            prompt_ids=ids, prompt_mask=mask, query=prompt["prompt"],
            scope=protocol_scope(protocol),
            task=task_graph(prompt["case_id"], prompt["prompt"]),
            training=False, visual=None, enabled_ports=(),
            core_impl="none", enabled_branches=("memory",))
        try:
            token_ids = session.generate(request, max_new_tokens=maximum_new)
            prompt_len = int(request.frame.sense.prompt_ids.shape[1])
            generated = [int(v) for v in token_ids[0, prompt_len:].tolist()]
        finally:
            request.release()
        rows.append({"case_id": prompt["case_id"],
                     "generated_ids": generated,
                     "generated_text": _generated_text(tokenizer, generated),
                     "max_new_tokens": maximum_new})
    _write_retention(output_path, rows)
    scored = score_retention(
        [{"case_id": row["case_id"], "group_id": row["family"],
          "family": row["family"], "split": row["split"],
          "prompt": row["prompt"]} for row in prompts],
        [{"case_id": case_id, "answer": answer}
         for case_id, answer in labels.items()],
        [{"case_id": row["case_id"],
          "generated_text": row["generated_text"]} for row in rows])
    return {"label": label, "rows": rows, "score": scored}


def _write_retention(path, rows):
    write_jsonl(path, rows)


# ---------------------------------------------------------------------------
# Extension export
# ---------------------------------------------------------------------------


def export_extension(adapter, path, protocol, *, label):
    from replica_lab.vision_qualification import validate_extension_inventory
    receipt = adapter.save_extension(path)
    exported_bytes = sum((Path(path) / name).stat().st_size
                         for name in ("manifest.json",
                                      "extensions.safetensors"))
    require(exported_bytes <= protocol["limits"]["max_export_bytes"],
            "Extension exceeds the declared export byte ceiling")
    expected = [name for name, _ in adapter.new_named_parameters()]
    validated = validate_extension_inventory(
        path, expected_names=expected, vision_pins=protocol["vision"])
    return {"label": label,
            "format": validated["format"],
            "receipt_sha256": sha256_file(Path(path) / "manifest.json"),
            "weights_sha256": validated["weights_sha256"],
            "manifest_digest": validated["manifest_digest"],
            "tensor_names": validated["tensor_names"],
            "base_saved": validated["base_saved"],
            "components": validated["components"],
            "bytes": exported_bytes}


def extension_tensor_snapshot(adapter):
    import torch
    return {name: parameter.detach().cpu().clone()
            for name, parameter in adapter.new_named_parameters()}


def changed_extension_tensors(before, after):
    names = sorted(set(before) | set(after))
    changed = [name for name in names
               if name not in before or name not in after
               or not __import__("torch").equal(before[name], after[name])]
    return changed


# ---------------------------------------------------------------------------
# Train mode
# ---------------------------------------------------------------------------


def execute_train(args, protocol, protocol_sha, dataset, implementation):
    import torch
    from replica_lab.architecture.loading import load_local_mps
    require(torch.backends.mps.is_available(), "MPS is unavailable; no fallback")
    config = load_config(args.config)
    validate_config(config)
    from replica_lab.contracts import fingerprint
    require(fingerprint(config) == protocol["config_sha256"],
            "Runner config differs from the preregistered protocol config")
    require(config["data_root"] == protocol["data"]["root"],
            "Runner data root differs from the protocol")
    guard = Guard(protocol)
    output = ensure_fresh_directory(args.output)
    free = shutil.disk_usage(ROOT).free
    require(free >= protocol["limits"]["min_free_disk_bytes"],
            "Insufficient free disk headroom")
    require(sha256_file(args.model_lock) == protocol["model"]["lock_sha256"],
            "Model lock bytes differ from the protocol pin")
    lock = json.loads(args.model_lock.read_text())
    require(lock.get("revision") == protocol["model"]["revision"],
            "Model revision differs from the protocol pin")
    source, spec, manifest_path = verify_source_and_spec(protocol, "mps")
    batches, probe_batches, examples = batch_lookup(dataset, dataset.schedule)
    dev_cases_all = dataset.generation_cases("dev")
    dev_by_id = {case["case_id"]: case for case in dev_cases_all}
    probe_cases = [dev_by_id[case_id] for case_id in
                   dataset.schedule["dev_probe_case_ids"]]
    scope = protocol_scope(protocol)
    started = time.monotonic()
    ledger = None
    from replica_lab.architecture.loading import exclusive_mps_worker
    with exclusive_mps_worker(ROOT):
        adapter, tokenizer, lock_loaded = load_local_mps(
            args.model_path, args.model_lock, vision=spec)
        require(adapter.manifest.vision is not None
                and adapter.manifest.vision.digest == spec.digest,
                "Loaded manifest lacks the pinned vision spec")
        require(adapter.vision_projector is not None,
                "Loaded adapter lacks the trainable projector")
        trainable, frozen = freeze_projector_only(adapter)
        adapter._initial_base_stamp = adapter.parameter_stamp(
            original_only=True)
        init_hash = initialize_projector(adapter.vision_projector,
                                         config["train"]["init_seed"])
        require(init_hash not in set(protocol["train"][
            "forbidden_parent_weights_sha256"]),
            "Fresh initialization matches a rejected parent weight hash")
        session = make_session(adapter, tokenizer, output)
        ledger = session.ledger
        service = feature_service(protocol, "mps")
        encoder_audit = service.frozen_audit()

        retention_before = evaluate_retention(
            adapter, session, tokenizer, protocol,
            output / "retention_pre.jsonl", label="before_updates")
        train_started = time.monotonic()
        probe_example = {**probe_cases[0],
                         "response": probe_cases[0]["label"]}
        probe = backward_only_probe(
            adapter, session, tokenizer, service, protocol, scope,
            probe_example)
        initialize_projector(adapter.vision_projector,
                             config["train"]["init_seed"])
        lr_selection = select_learning_rate(
            adapter, session, tokenizer, service, protocol, scope, config,
            args.arm, probe_batches, probe_cases, guard, train_started,
            output / "lr_probe")
        selected_lr = float(lr_selection["selected_lr"])
        init_hash = initialize_projector(adapter.vision_projector,
                                         config["train"]["init_seed"])
        require(init_hash == lr_selection["results"][0]["init_sha256"],
                "Final initialization differs from the probe initialization")
        adapter._initial_base_stamp = adapter.parameter_stamp(
            original_only=True)
        initial_tensors = extension_tensor_snapshot(adapter)
        optimizer, route, audit = build_optimizer(adapter, args.arm,
                                                  selected_lr, config)
        adapter.train()
        training = run_updates(
            adapter, session, tokenizer, service, protocol, scope, batches,
            optimizer, guard, output / "training_trace.jsonl",
            label="train",
            max_seconds=protocol["train"]["max_seconds"],
            start_time=train_started)
        adapter.eval()
        base_unchanged = (adapter.parameter_stamp(original_only=True)
                          == adapter._initial_base_stamp)
        require(base_unchanged, "Training changed the sealed native base")
        final_tensors = extension_tensor_snapshot(adapter)
        changed = changed_extension_tensors(initial_tensors, final_tensors)
        require(bool(changed),
                "Training changed no extension tensor")
        require(all(name.startswith("vision_projector.")
                    for name in changed),
                "Only projector extension tensors may change")
        extension = export_extension(adapter, output / "extension", protocol,
                                     label=f"{args.arm}_final")
        retention_after = evaluate_retention(
            adapter, session, tokenizer, protocol,
            output / "retention_post.jsonl", label="after_updates")
        dev_cases = dev_cases_all
        require(len(dev_cases) <= protocol["limits"]["max_eval_cases"],
                "Dev case count exceeds the evaluation limit")
        dev_response_loss, dev_losses = dev_loss(
            adapter, session, tokenizer, service, protocol, scope, dev_cases)
        dev_rows = evaluate_generations(
            adapter, session, tokenizer, service, protocol, scope, dev_cases,
            output / "dev_generations.jsonl",
            seed_offset=int(protocol["evaluation"]["shuffle_seed_offset"]),
            guard=guard, start_time=started)
        write_jsonl(output / "dev_cases.jsonl", dev_cases)
        caption_rows = caption_evaluation(
            adapter, session, tokenizer, service, protocol, scope,
            dataset.caption_cases(), output / "caption_generations.jsonl",
            guard=guard, start_time=started)
        write_jsonl(output / "caption_cases.jsonl",
                    dataset.caption_cases())
        write_jsonl(output / "caption_labels.jsonl",
                    [{"case_id": case["case_id"], "answer": case["label"]}
                     for case in dataset.caption_cases()])
        dump(output / "lr_probe.json", lr_selection)
        dump(output / "backward_probe.json", probe)
        driver_peak = guard.driver_peak
        token_parity = all(
            [row["generated_ids"] for row in retention_before["rows"]]
            == [row["generated_ids"] for row in retention_after["rows"]])
        retention_result = retention_policy(
            measured_correct=retention_after["score"]["correct"],
            measured_cases=int(protocol["retention"]["max_cases"]),
            baseline_correct=int(protocol["retention"]["baseline"]["correct"]),
            baseline_cases=int(protocol["retention"]["baseline"]["cases"]),
            token_parity=token_parity,
            gates=protocol["evaluation"]["gates"])
        baseline_matches = (
            retention_before["score"]["correct"] == 22
            and retention_before["score"]["cases"] == 32)
        checks = {name: True for name in _required_run_checks()}
        checks.update({
            "fresh_initialization": (
                init_hash == lr_selection["results"][0]["init_sha256"]),
            "not_warm_start": (
                init_hash not in set(protocol["train"][
                    "forbidden_parent_weights_sha256"])),
            "retention_baseline_matches_declared": baseline_matches,
            "backward_only_probe_labelled": (
                probe["probe_label"] == "backward_only_no_update"),
            "equal_compute_within_arm": (
                training["completed"] == protocol["schedule"]["updates"]
                and training["examples"] == protocol["schedule"]["examples"]),
        })
        require(all(value is True for value in checks.values()),
                f"Run checks failed: {sorted(name for name, value in checks.items() if value is not True)}")
        report = {
            "format": RUN_REPORT_FORMAT,
            "version": protocol["version"],
            "stage": "train",
            "arm": args.arm,
            "protocol_sha256": protocol_sha,
            "data_pins": dict(protocol["data"]["manifest"]),
            "implementation_sha256": implementation,
            "initialization": {
                "recipe": protocol["train"]["init_recipe"],
                "seed": int(protocol["train"]["init_seed"]),
                "projector_init_sha256": init_hash,
                "warm_start": None,
                "matches_forbidden_parent": False,
            },
            "trainable": {
                "names": sorted(trainable),
                "parameter_count": PROJECTOR_PARAMETER_COUNT,
                "frozen_names": frozen,
                "trainable_count": len(trainable),
            },
            "gradient_audit": probe,
            "optimizer_route": route,
            "optimizer_audit": {
                "optimizer": audit["optimizer"],
                "parameter_dtype": audit["parameter_dtype"],
                "state_dtype": ("int8_momentum+float32_aux"
                                if args.arm == "muon8bit" else "float32"),
                "groups": audit["groups"],
            },
            "schedule": {
                "schedule_sha256": protocol["schedule"]["schedule_sha256"],
                "updates": protocol["schedule"]["updates"],
                "examples": protocol["schedule"]["examples"],
                "completed_updates": training["completed"],
                "probe_updates_per_candidate":
                    protocol["schedule"]["probe"]["updates_per_candidate"],
            },
            "lr_selection": lr_selection,
            "training": {
                "updates": training["completed"],
                "examples": training["examples"],
                "completed": (training["completed"]
                              == protocol["schedule"]["updates"]),
                "response_only_labels": True,
                "nonprojector_gradients_absent": True,
                "base_parameter_stamp_unchanged": base_unchanged,
                "max_seconds": protocol["train"]["max_seconds"],
            },
            "retention": {
                "policy": "component_nonregression.v1",
                "baseline_correct": 22, "baseline_cases": 32,
                "baseline_accuracy": 22 / 32,
                "measured_correct": retention_after["score"]["correct"],
                "measured_cases": retention_after["score"]["cases"],
                "measured_accuracy": retention_after["score"]["accuracy"],
                "before_correct": retention_before["score"]["correct"],
                "before_accuracy": retention_before["score"]["accuracy"],
                "token_parity": token_parity,
                "baseline_matches_declared": baseline_matches,
                "baseline_eligible": retention_result["baseline_eligible"],
                "overall_baseline_eligible": False,
                "component_nonregression_pass":
                    retention_result["component_nonregression_pass"],
                "component_failures": retention_result["component_failures"],
                "retention_drop": retention_result["retention_drop"],
                "pre_sha256": sha256_file(output / "retention_pre.jsonl"),
                "post_sha256": sha256_file(output / "retention_post.jsonl"),
            },
            "dev": {
                "cases": len(dev_cases),
                "generations_sha256": sha256_file(
                    output / "dev_generations.jsonl"),
                "cases_sha256": sha256_file(output / "dev_cases.jsonl"),
                "seal_sha256": protocol["splits"]["dev"]["seal_sha256"],
                "dataset_dev_cases_sha256": fingerprint(dev_cases),
                "response_loss": dev_response_loss,
                "losses": dev_losses,
                "scored_splits": ["curriculum_dev", "diagnostic_dev",
                                  "external_development"],
                "caption_cases_sha256": sha256_file(
                    output / "caption_cases.jsonl"),
                "caption_labels_sha256": sha256_file(
                    output / "caption_labels.jsonl"),
                "caption_generations_sha256": sha256_file(
                    output / "caption_generations.jsonl"),
                "audit_touched": False,
            },
            "extension": extension,
            "artifacts": {
                "training_trace.jsonl": sha256_file(
                    output / "training_trace.jsonl"),
                "lr_probe.json": sha256_file(output / "lr_probe.json"),
                "backward_probe.json": sha256_file(
                    output / "backward_probe.json"),
                "dev_cases.jsonl": sha256_file(output / "dev_cases.jsonl"),
                "dev_generations.jsonl": sha256_file(
                    output / "dev_generations.jsonl"),
                "caption_cases.jsonl": sha256_file(
                    output / "caption_cases.jsonl"),
                "caption_labels.jsonl": sha256_file(
                    output / "caption_labels.jsonl"),
                "caption_generations.jsonl": sha256_file(
                    output / "caption_generations.jsonl"),
                "retention_pre.jsonl": sha256_file(
                    output / "retention_pre.jsonl"),
                "retention_post.jsonl": sha256_file(
                    output / "retention_post.jsonl"),
                "extension/manifest.json": extension["receipt_sha256"],
                "extension/extensions.safetensors":
                    extension["weights_sha256"],
            },
            "checks": checks,
            "audit_scored": False,
            "external_final_consumed": False,
            "costs": {
                "generated_tokens": sum(len(row["generated_ids"])
                                        for row in dev_rows + caption_rows),
                "updates": training["completed"],
                "probe_updates": (len(lr_selection["candidates"])
                                  * protocol["schedule"]["probe"]
                                  ["updates_per_candidate"]),
                "peak_driver_bytes": driver_peak,
                "workers": 1,
                "downloads": 0,
            },
            "claims": {
                "perception_quality": False,
                "semantic_quality": False,
                "overall_baseline_eligible": False,
                "scope": "staged_curriculum_engineering_and_learned_quality_only",
            },
            "implementation_runtime": {
                "packages": environment_report(),
                "device": "mps",
                "encoder_source_manifest_sha256": sha256_file(manifest_path),
                "encoder_frozen_audit": encoder_audit,
                "guard": guard.report(),
                "elapsed_seconds": time.monotonic() - started,
                "free_disk_before_bytes": free,
                "free_disk_after_bytes": shutil.disk_usage(ROOT).free,
                "started_at_utc": utc_now(),
            },
        }
        require(time.monotonic() - started
                <= protocol["limits"]["max_seconds"],
                "Training run exceeded the declared overall ceiling")
        # Drop the out-of-band hash before writing the report file.
        validate_run_report({**report, "_run_report_sha256": ""}, protocol)
        dump(output / "run_report.json", report)
        if ledger is not None:
            ledger.close()
        return report


# ---------------------------------------------------------------------------
# Reload mode
# ---------------------------------------------------------------------------


def execute_reload(args, protocol, protocol_sha, dataset):
    import torch
    from replica_lab.architecture.loading import (exclusive_mps_worker,
                                                load_local_mps)
    require(torch.backends.mps.is_available(), "MPS is unavailable; no fallback")
    run = Path(args.run)
    prior = load_run_report(run / "run_report.json")
    validate_run_report(prior, protocol)
    require(prior["extension"]["base_saved"] is False,
            "Prior extension saved the native base")
    require(prior["protocol_sha256"] == protocol_sha,
            "Prior run used another protocol")
    from replica_lab.contracts import canonical_json, fingerprint
    dataset_dev = dataset.generation_cases("dev")
    frozen_dev_cases = read_jsonl(run / "dev_cases.jsonl")
    require(canonical_json(frozen_dev_cases) == canonical_json(dataset_dev),
            "Frozen dev cases differ from the prepared dataset")
    require(prior["dev"]["dataset_dev_cases_sha256"] == fingerprint(dataset_dev),
            "Run report dev cases hash differs from the dataset")
    require(prior["dev"]["cases_sha256"]
            == sha256_file(run / "dev_cases.jsonl"),
            "Frozen dev cases artifact changed")
    require(prior["dev"]["caption_cases_sha256"]
            == sha256_file(run / "caption_cases.jsonl"),
            "Frozen caption cases artifact changed")
    require(prior["dev"]["caption_labels_sha256"]
            == sha256_file(run / "caption_labels.jsonl"),
            "Frozen caption labels artifact changed")
    require(prior["dev"]["caption_generations_sha256"]
            == sha256_file(run / "caption_generations.jsonl"),
            "Frozen caption generations artifact changed")
    require(prior["dev"]["generations_sha256"]
            == sha256_file(run / "dev_generations.jsonl"),
            "Frozen dev generations artifact changed")
    require(prior["retention"]["pre_sha256"]
            == sha256_file(run / "retention_pre.jsonl")
            and prior["retention"]["post_sha256"]
            == sha256_file(run / "retention_post.jsonl"),
            "Frozen retention artifacts changed")
    source, spec, manifest_path = verify_source_and_spec(protocol, "mps")
    output = ensure_fresh_directory(args.output)
    require(shutil.disk_usage(ROOT).free
            >= protocol["limits"]["min_free_disk_bytes"],
            "Insufficient free disk headroom")
    guard = Guard(protocol)
    scope = protocol_scope(protocol)
    started = time.monotonic()
    prior_dev = read_jsonl(run / "dev_generations.jsonl")
    prior_retention = read_jsonl(run / "retention_post.jsonl")
    with exclusive_mps_worker(ROOT):
        adapter, tokenizer, lock = load_local_mps(
            args.model_path, args.model_lock, vision=spec)
        freeze_projector_only(adapter)
        adapter._initial_base_stamp = adapter.parameter_stamp(
            original_only=True)
        adapter.load_extension(run / "extension")
        receipt = json.loads((run / "extension/manifest.json").read_text())
        require(receipt["weights_sha256"] == prior["extension"]["weights_sha256"],
                "Reloaded extension hash differs")
        session = make_session(adapter, tokenizer, output)
        service = feature_service(protocol, "mps")
        before_stamp = adapter.parameter_stamp(original_only=True)
        retention = evaluate_retention(
            adapter, session, tokenizer, protocol,
            output / "retention.jsonl", label="reload")
        dev_cases = read_jsonl(run / "dev_cases.jsonl")
        rows = evaluate_generations(
            adapter, session, tokenizer, service, protocol, scope, dev_cases,
            output / "dev_generations.jsonl",
            seed_offset=int(protocol["evaluation"]["shuffle_seed_offset"]),
            guard=guard, start_time=time.monotonic())
        write_jsonl(output / "dev_cases.jsonl", dev_cases)
        after_stamp = adapter.parameter_stamp(original_only=True)
        require(before_stamp == after_stamp,
                "Reload evaluation changed a parameter")
        require(prior["extension"]["weights_sha256"]
                == sha256_file(run / "extension/extensions.safetensors"),
                "Reloaded extension bytes changed")
        dev_equal = ([{k: row[k] for k in
                       ("control", "case_id", "generated_ids")}
                      for row in rows]
                     == [{k: row[k] for k in
                          ("control", "case_id", "generated_ids")}
                         for row in prior_dev])
        retention_equal = ([row["generated_ids"] for row in retention["rows"]]
                           == [row["generated_ids"]
                               for row in prior_retention])
        require(dev_equal, "Fresh-process dev generations differ")
        require(retention_equal,
                "Fresh-process retention generations differ")
        require(time.monotonic() - started
                <= protocol["limits"]["max_seconds"],
                "Reload exceeded the declared wall-time ceiling")
        require(sha256_file(output / "dev_cases.jsonl")
                == prior["dev"]["cases_sha256"],
                "Reload dev cases differ from the frozen run")
        require(sha256_file(output / "dev_generations.jsonl")
                == prior["dev"]["generations_sha256"],
                "Reload dev generations differ from the frozen run")
        require(sha256_file(output / "retention.jsonl")
                == prior["retention"]["post_sha256"],
                "Reload retention generations differ from the frozen run")
        report = {
            "format": RELOAD_REPORT_FORMAT,
            "version": protocol["version"],
            "run_report_sha256": prior["_run_report_sha256"],
            "protocol_sha256": protocol_sha,
            "arm": prior["arm"],
            "zero_updates": True,
            "extension_weights_sha256": receipt["weights_sha256"],
            "dev_generations_equal": dev_equal,
            "retention_generations_equal": retention_equal,
            "extension_unchanged": True,
            "dev_cases_sha256": sha256_file(output / "dev_cases.jsonl"),
            "dataset_dev_cases_sha256": fingerprint(dataset_dev),
            "dev_generations_sha256": sha256_file(
                output / "dev_generations.jsonl"),
            "retention_sha256": sha256_file(output / "retention.jsonl"),
            "checks": {"zero_updates": True, "extension_unchanged": True,
                       "dev_equal": True, "retention_equal": True,
                       "strict_export": True},
            "claims": {"bitwise_reload": True, "quality_claim": False},
            "guard": guard.report(),
            "packages": environment_report(),
        }
        validate_reload_report(report, prior, protocol)
        dump(output / "reload_report.json", report)
        session.ledger.close()
        return report


# ---------------------------------------------------------------------------
# Independent scoring
# ---------------------------------------------------------------------------


def execute_score(args, protocol, protocol_sha, dataset):
    run = Path(args.run)
    report_path = run / "run_report.json"
    prior = load_run_report(report_path)
    validate_run_report(prior, protocol)
    reload_path = run / "reload_report/reload_report.json"
    if not reload_path.is_file():
        reload_path = run / "reload_report.json"
    require(reload_path.is_file(),
            "A fresh-process reload report is required before scoring")
    reload_report = json.loads(reload_path.read_text())
    validate_reload_report(reload_report, prior, protocol)
    from replica_lab.contracts import canonical_json, fingerprint
    for name, expected in prior["artifacts"].items():
        path = run / name
        require(path.is_file(),
                f"Run artifact is missing: {name}")
        require(sha256_file(path) == expected,
                f"Run artifact hash changed: {name}")
    dev_cases = read_jsonl(run / "dev_cases.jsonl")
    generations = read_jsonl(run / "dev_generations.jsonl")
    caption_cases = read_jsonl(run / "caption_cases.jsonl")
    caption_labels = read_jsonl(run / "caption_labels.jsonl")
    caption_generations = read_jsonl(run / "caption_generations.jsonl")
    retention_post = read_jsonl(run / "retention_post.jsonl")
    retention_pre = read_jsonl(run / "retention_pre.jsonl")
    dataset_dev = dataset.generation_cases("dev")
    require(canonical_json(dev_cases) == canonical_json(dataset_dev),
            "Dev cases were substituted or truncated")
    require(prior["dev"]["dataset_dev_cases_sha256"] == fingerprint(dataset_dev),
            "Dev dataset hash differs from the frozen report")
    require(prior["dev"]["cases"] == len(dataset_dev),
            "Dev case count differs from the frozen report")
    dataset_caption = dataset.caption_cases()
    require(canonical_json(caption_cases) == canonical_json(dataset_caption),
            "Caption cases were substituted or truncated")
    expected_caption_labels = [{"case_id": case["case_id"],
                                "answer": case["label"]}
                               for case in dataset_caption]
    require(canonical_json(caption_labels)
            == canonical_json(expected_caption_labels),
            "Caption labels were substituted or truncated")
    expected_caption_ids = {case["case_id"] for case in dataset_caption}
    require({row["case_id"] for row in caption_generations}
            == expected_caption_ids
            and len(caption_generations) == len(dataset_caption),
            "Caption generations are incomplete or substituted")
    validate_generation_rows(generations, {case["case_id"]
                                           for case in dev_cases},
                             protocol["evaluation"]["controls"])
    score_dev = read_jsonl(run / "dev_generations.jsonl")
    require(len(score_dev) == len(dev_cases)
            * len(protocol["evaluation"]["controls"]),
            "Dev generation row count differs")
    from replica_lab.vision_qualification import score_retention
    prompts, labels = load_retention(protocol)
    retention = score_retention(
        [{"case_id": row["case_id"], "group_id": row["family"],
          "family": row["family"], "split": row["split"],
          "prompt": row["prompt"]} for row in prompts],
        [{"case_id": case_id, "answer": answer}
         for case_id, answer in labels.items()],
        [{"case_id": row["case_id"],
          "generated_text": row["generated_text"]}
         for row in retention_post])
    token_parity = ([row["generated_ids"] for row in retention_pre]
                    == [row["generated_ids"] for row in retention_post])
    score = assemble_score_report(
        protocol=protocol, run_report=prior, cases=dev_cases,
        generations=generations, caption_labels=caption_labels,
        caption_generations=caption_generations,
        retention={"measured_correct": retention["correct"],
                   "token_parity": token_parity})
    output = ensure_fresh_directory(args.output)
    dump(output / "score_report.json", score)
    scored = score_procedural(
        dev_cases, generations,
        controls=protocol["evaluation"]["controls"],
        seed=int(protocol["seed"]),
        bootstrap_resamples=int(
            protocol["evaluation"]["bootstrap"]["resamples"]),
        bootstrap_confidence=float(
            protocol["evaluation"]["bootstrap"]["confidence"]),
        gates=protocol["evaluation"]["gates"])
    write_jsonl(output / "procedural_rows.jsonl", scored["rows"])
    return score


# ---------------------------------------------------------------------------
# Finalize mode: audit and external final scored exactly once
# ---------------------------------------------------------------------------


def load_external_final(protocol):
    """Read the sealed final rows only for the one-shot final evaluation."""
    from replica_lab.contracts import canonical_json
    final = protocol["external"]["final_sealed"]
    manifest_path = resolve_project_path(protocol, final["path"])
    require(manifest_path.is_file(), "Sealed final manifest is missing")
    require(sha256_file(manifest_path) == final["sha256"],
            "Sealed final manifest bytes changed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(manifest.get("role") == "final_eval_untouched",
            "Sealed final role differs")
    corpus_path = resolve_project_path(
        protocol, protocol["external"]["corpus"]["path"])
    require(sha256_file(corpus_path)
            == protocol["external"]["corpus"]["sha256"],
            "External corpus bytes changed")
    corpus = {row["row_id"]: row for row in read_jsonl(corpus_path)}
    image_root = resolve_project_path(
        protocol, protocol["external"]["root"]) / "raw/images"
    rows = []
    for entry in manifest["rows"]:
        row = corpus[entry["row_id"]]
        require(sha256_bytes(canonical_json(row).encode("utf-8"))
                == entry["row_sha256"],
                "Sealed final row bytes changed")
        require(sha256_text_local(row["target"]["caption"])
                == entry["target_sha256"],
                "Sealed final caption changed")
        image = image_root / f"{row['root_asset']['upstream_id']}.jpg"
        require(image.is_file(), "Sealed final image is missing")
        require(sha256_file(image) == entry["image_sha256"],
                "Sealed final image bytes changed")
        rows.append({"case_id": entry["row_id"], "image_path": str(image),
                     "image_sha256": entry["image_sha256"],
                     "question": row["observation"]["prompt"],
                     "label": row["target"]["caption"]})
    require(len(rows) == int(protocol["splits"]["external_final"]["rows"]),
            "Sealed final row count differs")
    return rows


def sha256_text_local(value):
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def _arm_report_summary(report):
    return {
        "run_report_sha256": report["_run_report_sha256"],
        "extension_weights_sha256": report["extension"]["weights_sha256"],
        "projector_init_sha256":
            report["initialization"]["projector_init_sha256"],
        "schedule_sha256": report["schedule"]["schedule_sha256"],
        "updates": report["schedule"]["updates"],
        "examples": report["schedule"]["examples"],
        "probe_updates_per_candidate":
            report["schedule"]["probe_updates_per_candidate"],
        "training_max_seconds": report["training"]["max_seconds"],
        "dev_response_loss": float(report["dev"]["response_loss"]),
    }


def _select_arm(reports, rule):
    losses = {arm: float(report["dev"]["response_loss"])
              for arm, report in reports.items()}
    best = min(losses.values())
    selected = next(arm for arm in rule["order"] if losses[arm] == best)
    comparison = {
        "rule": {"measure": rule["measure"],
                 "tie_break": rule["tie_break"],
                 "both_arms_required": rule["both_arms_required"]},
        "order": list(rule["order"]),
        "reports": {arm: _arm_report_summary(reports[arm])
                    for arm in rule["order"]},
        "selected_arm": selected,
    }
    return selected, comparison


def execute_finalize(args, protocol, protocol_sha, dataset):
    import torch
    from replica_lab.architecture.loading import (exclusive_mps_worker,
                                                load_local_mps)
    require(torch.backends.mps.is_available(), "MPS is unavailable; no fallback")
    if args.peer_run is None or args.peer_score_report is None:
        raise VisionCurriculumError(
            "Finalize requires both arm runs and both score reports")
    run = Path(args.run)
    peer_run = Path(args.peer_run)
    primary = load_run_report(run / "run_report.json")
    peer = load_run_report(peer_run / "run_report.json")
    validate_run_report(primary, protocol)
    validate_run_report(peer, protocol)
    validate_lr_reports(primary, peer)
    reports = {primary["arm"]: primary, peer["arm"]: peer}
    require(set(reports) == set(ARMS),
            "Finalize requires exactly the two declared arms")
    primary_score = json.loads(Path(args.score_report).read_text())
    peer_score = json.loads(Path(args.peer_score_report).read_text())
    for name, score, report in (
            ("primary", primary_score, primary),
            ("peer", peer_score, peer)):
        validate_score_report(score)
        require(score["run_report_sha256"] == report["_run_report_sha256"],
                f"{name} score report belongs to another run")
        require(score["engineering"]["pass"] is True,
                f"{name} engineering checks must pass before finalize")
        require(score["eligibility"]["audit_scores_present"] is False,
                f"{name} score report already contains audit scores")
    scores = {primary["arm"]: primary_score, peer["arm"]: peer_score}
    rule = {"measure": protocol["schedule"]["arm_selection"]["measure"],
            "tie_break": protocol["schedule"]["arm_selection"]["tie_break"],
            "both_arms_required": True,
            "order": list(ARMS)}
    selected_arm, comparison = _select_arm(reports, rule)
    selected_run = run if primary["arm"] == selected_arm else peer_run
    selected = reports[selected_arm]
    selected_score = scores[selected_arm]
    dev_report_sha256 = selected["_run_report_sha256"]
    score_report_sha256 = sha256_file(Path(
        args.score_report if primary["arm"] == selected_arm
        else args.peer_score_report))
    output = ensure_fresh_directory(selected_run / "finalize")
    require(shutil.disk_usage(ROOT).free
            >= protocol["limits"]["min_free_disk_bytes"],
            "Insufficient free disk headroom")
    started = time.monotonic()
    source, spec, manifest_path = verify_source_and_spec(protocol, "mps")
    guard = Guard(protocol)
    scope = protocol_scope(protocol)
    # Reserve the global audit/final scoring for this protocol and both arms
    # BEFORE any model load or scoring. The reservation and receipt live outside
    # the data root, so every output directory and both arms share one gate.
    paths = require_global_finalize_allowed(
        protocol_sha, project_root=ROOT,
        data_manifest_sha256=protocol["data"]["manifest"]["sha256"])
    reservation = reserve_finalize(
        protocol, protocol_sha, project_root=ROOT,
        comparison=comparison, selected_arm=selected_arm,
        dev_report_sha256=dev_report_sha256,
        score_report_sha256=score_report_sha256)
    with exclusive_mps_worker(ROOT):
        adapter, tokenizer, lock = load_local_mps(
            args.model_path, args.model_lock, vision=spec)
        freeze_projector_only(adapter)
        adapter._initial_base_stamp = adapter.parameter_stamp(
            original_only=True)
        adapter.load_extension(selected_run / "extension")
        receipt = json.loads(
            (selected_run / "extension/manifest.json").read_text())
        require(receipt["weights_sha256"]
                == selected["extension"]["weights_sha256"],
                "Finalize loaded another extension")
        frozen_at = utc_now()
        selected_checkpoint = {
            "weights_sha256": receipt["weights_sha256"],
            "frozen_at_utc": frozen_at,
            "frozen_before_audit": True,
        }
        session = make_session(adapter, tokenizer, output)
        service = feature_service(protocol, "mps")
        before = adapter.parameter_stamp(original_only=True)
        audit_cases = dataset.generation_cases("audit")
        require(len(audit_cases) <= protocol["limits"]["max_eval_cases"],
                "Audit case count exceeds the limit")
        audit_rows = evaluate_generations(
            adapter, session, tokenizer, service, protocol, scope, audit_cases,
            output / "audit_generations.jsonl",
            seed_offset=int(protocol["evaluation"]["shuffle_seed_offset"]),
            guard=guard, start_time=time.monotonic())
        write_jsonl(output / "audit_cases.jsonl", audit_cases)
        audit_score = score_procedural(
            audit_cases, audit_rows,
            controls=protocol["evaluation"]["controls"],
            seed=int(protocol["seed"]),
            bootstrap_resamples=int(protocol["evaluation"]["bootstrap"]
                                   ["resamples"]),
            bootstrap_confidence=float(protocol["evaluation"]["bootstrap"]
                                       ["confidence"]),
            gates=protocol["evaluation"]["gates"])
        final_cases = load_external_final(protocol)
        write_jsonl(output / "external_final_cases.jsonl",
                    [{"case_id": row["case_id"],
                      "image_sha256": row["image_sha256"]}
                     for row in final_cases])
        final_rows = caption_evaluation(
            adapter, session, tokenizer, service, protocol, scope,
            [{"case_id": row["case_id"], "image_path": row["image_path"],
              "image_sha256": row["image_sha256"],
              "question": row["question"], "label": row["label"]}
             for row in final_cases],
            output / "external_final_generations.jsonl",
            guard=guard, start_time=time.monotonic())
        write_jsonl(output / "external_final_generations.jsonl", final_rows)
        final_caption = score_caption(
            [{"case_id": row["case_id"], "answer": row["label"]}
             for row in final_cases], final_rows)
        after = adapter.parameter_stamp(original_only=True)
        require(before == after, "Finalize changed a parameter")
        require(time.monotonic() - started
                <= protocol["limits"]["max_seconds"],
                "Finalize exceeded the declared wall-time ceiling")
        final_receipt = build_finalize_receipt(
            protocol=protocol, comparison=comparison,
            reservation=reservation, selected_arm=selected_arm,
            selected_checkpoint=selected_checkpoint,
            dev_report_sha256=dev_report_sha256,
            score_report_sha256=score_report_sha256,
            audit={"scored_once": True,
                   "seal_sha256": protocol["splits"]["audit"]["seal_sha256"],
                   "cases": len(audit_cases),
                   "cases_sha256": sha256_file(output / "audit_cases.jsonl"),
                   "generations_sha256": sha256_file(
                       output / "audit_generations.jsonl"),
                   "score": audit_score,
                   "learned_quality_pass":
                       audit_score["learned_quality_pass"],
                   "gate_failures": audit_score["gate_failures"]},
            external_final={"consumed_once": True,
                            "seal_sha256": protocol["splits"]
                            ["external_final"]["seal_sha256"],
                            "rows": len(final_cases),
                            "cases_sha256": sha256_file(
                                output / "external_final_cases.jsonl"),
                            "generations_sha256": sha256_file(
                                output / "external_final_generations.jsonl"),
                            "caption": final_caption})
        receipt_sha = write_finalize_receipt(paths["receipt"], final_receipt)
        require(sha256_file(paths["receipt"]) == receipt_sha,
                "Final receipt write differs")
        guard.sample(index=1, event="finalize_end")
        dump(output / "guard.json", guard.report())
        dump(output / "arm_comparison.json", comparison)
        session.ledger.close()
        return final_receipt


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("train", "reload", "score",
                                         "finalize"))
    parser.add_argument("--config", type=Path,
                        default=ROOT / "configs/vision_curriculum_v1.json")
    parser.add_argument("--protocol", type=Path,
                        default=ROOT / "data/vision_curriculum_20261025/"
                                         "protocol.json")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--arm", choices=ARMS, default=None)
    parser.add_argument("--run", type=Path, default=None)
    parser.add_argument("--peer-run", type=Path, default=None)
    parser.add_argument("--score-report", type=Path, default=None)
    parser.add_argument("--peer-score-report", type=Path, default=None)
    parser.add_argument("--device", choices=("mps",), default="mps")
    parser.add_argument("--model-path", type=Path,
                        default=ROOT / "checkpoints/minicpm5_source")
    parser.add_argument("--model-lock", type=Path,
                        default=ROOT / "configs/minicpm5_sealed_lock.local.json")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.device != "mps":
        parser.error("Only the declared MPS device is supported")
    if not args.protocol.is_file():
        parser.error("Protocol file is missing")
    if args.mode == "train":
        if args.arm is None or args.output is None:
            parser.error("train requires --arm and --output")
        require_fresh_cli_output(args.output)
    elif args.mode == "reload":
        if args.run is None or args.output is None:
            parser.error("reload requires --run and --output")
        ensure_fresh_directory(args.output)
    elif args.mode == "score":
        if args.run is None or args.output is None:
            parser.error("score requires --run and --output")
        ensure_fresh_directory(args.output)
    elif args.mode == "finalize":
        if (args.run is None or args.peer_run is None
                or args.score_report is None
                or args.peer_score_report is None):
            parser.error("finalize requires --run, --peer-run, "
                         "--score-report and --peer-score-report")
        args.output = args.output or (Path(args.run) / "finalize")
    protocol, protocol_sha, dataset, implementation = load_protocol_and_data(
        args)
    if args.mode == "train":
        result = execute_train(args, protocol, protocol_sha, dataset,
                               implementation)
        print(json.dumps({"status": "completed", "mode": "train",
                          "arm": args.arm,
                          "run_report": str(Path(args.output) /
                                             "run_report.json")}))
        return 0
    if args.mode == "reload":
        execute_reload(args, protocol, protocol_sha, dataset)
        print(json.dumps({"status": "completed", "mode": "reload"}))
        return 0
    if args.mode == "score":
        report = execute_score(args, protocol, protocol_sha, dataset)
        print(json.dumps({"status": "completed", "mode": "score",
                          "learned_quality_pass":
                              report["learned_quality"]["pass"],
                          "component_nonregression_pass":
                              report["eligibility"][
                                  "component_nonregression_pass"]}))
        return 0
    receipt = execute_finalize(args, protocol, protocol_sha, dataset)
    print(json.dumps({"status": "completed", "mode": "finalize",
                      "selected_arm": receipt["selected_arm"],
                      "audit_learned_quality_pass":
                          receipt["audit"]["learned_quality_pass"],
                      "receipt": receipt["reservation"]["path"].replace(
                          ".reservation.json", ".json")}))
    return 0


def require_fresh_cli_output(path):
    path = Path(path)
    require(not path.exists() or (path.is_dir() and not any(path.iterdir())),
            f"Use a fresh output directory: {path}")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except VisionCurriculumError as error:
        print(json.dumps({"status": "failed",
                          "error_type": type(error).__name__,
                          "error": str(error)}), file=sys.stderr)
        raise

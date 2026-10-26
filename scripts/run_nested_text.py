#!/usr/bin/env python3
"""One-clock nested-text MiniCPM5 runner (teach / P-only / resume).

This entry point is deliberately conservative:

* one native model/tokenizer load per process, one exclusive MPS worker,
  offline execution and no CPU fallback;
* a fresh output directory that is never reused;
* ``teaching`` runs only eligible teaching/correction events and stops;
* ``p-only-eval`` loads weights only (no controller/coordinator/moments/dedup)
  on development first, writes the development gate, and opens ``sealed_final``
  at most once and only after that gate passes;
* ``resume`` restores the full checkpoint and applies exactly one next event,
  comparing its state digest with an uninterrupted reference from the same
  checkpoint.

No run here establishes capability or utility.  A single seed is descriptive.
Launch through ``scripts/run_capacity_preflight.py --entry-point
run_nested_text.py`` after adding that entry point to the guard allowlist.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
# These must be set before torch/transformers import.  The capacity guard sets
# REPLICA_CAPACITY_GUARD=1; the native loader verifies all of them.
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
for _name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
    os.environ[_name] = "1"

import torch  # noqa: E402
from replica_lab.architecture.contracts import TaskGraph, fingerprint  # noqa: E402
from replica_lab.architecture.loading import (exclusive_mps_worker,  # noqa: E402
                                              load_local_mps)
from replica_lab.architecture.plastic import (fixed_projection,  # noqa: E402
                                              projection_digest)
from replica_lab.memory_system.ledger import EligibilityScope, RecordLedger  # noqa: E402
from replica_lab.memory_system.session import MemorySession  # noqa: E402
from replica_lab.nested_learning import (ExpertBank, ObservationCoordinator,  # noqa: E402
                                         Observer, PersonalityIdentity,
                                         PlasticLearningController)
from replica_lab.nested_text import (NestedTextLearner, P_ONLY_FORMAT,  # noqa: E402
                                     P_ONLY_KEYS, PlasticReadObserver,
                                     TextAccessOrder, TextProtocol,
                                     TextTeachingEvent, evaluate_dev_gate,
                                     prompt_route_key, validate_p_only_payload)

PROMPT_FORM = "replica.nested_text.run.v1"


class RunnerError(RuntimeError):
    """A runner contract or data-access gate failed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RunnerError(message)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


def peak_driver_bytes() -> int:
    if not torch.backends.mps.is_available():
        return 0
    counter = getattr(torch.mps, "driver_allocated_memory", None)
    return int(counter()) if counter is not None else 0


def _tensor_sha(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().to("cpu", torch.float32)
                          .contiguous().numpy().tobytes()).hexdigest()


def _fresh_output(path: Path) -> None:
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise RunnerError("output path must be fresh or empty; never reuse a run")
    path.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n"
                            for row in rows), encoding="utf-8")


def load_episode_manifest(directory: Path) -> dict:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    _require(manifest.get("format") == "replica.nested_text.episode_manifest.v1",
             "unsupported episode manifest")
    for name, digest in manifest["files_sha256"].items():
        path = directory / name
        _require(path.exists(), f"episode file missing: {name}")
        _require(_sha256_file(path) == digest,
                 f"episode file digest differs: {name}")
    return manifest


def load_rows(directory: Path, manifest: Mapping[str, Any], role: str) -> list[dict]:
    _require(role in manifest["role_files"], f"undeclared episode role {role!r}")
    path = directory / manifest["role_files"][role]
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        for key in ("id", "world_uid", "scope_uid", "session_id",
                    "concept_or_rule_family", "source_group", "split_role",
                    "teaching_or_query", "prompt", "expected_response",
                    "prompt_sha256", "response_sha256", "verified", "eligible",
                    "source", "feedback_type", "target_available_at",
                    "causal_cutoff"):
            _require(key in row, f"{path.name}:{line_number}: missing {key}")
        _require(row["prompt_sha256"] == hashlib.sha256(
            row["prompt"].encode()).hexdigest(),
            f"{path.name}:{line_number}: prompt digest differs")
        _require(row["response_sha256"] == hashlib.sha256(
            row["expected_response"].encode()).hexdigest(),
            f"{path.name}:{line_number}: response digest differs")
        _require(row["labels_used_for_routing"] is False
                 and row["world_used_for_routing"] is False
                 and row["answer_label_used_for_observation"] is False,
                 f"{path.name}:{line_number}: labels cannot enter routing/observations")
        _require("route_key" not in row,
                 f"{path.name}:{line_number}: stored route keys are forbidden")
        if row["eligible"]:
            _require(row["verified"] is True and row["split_role"] == "teaching"
                     and row["teaching_or_query"] == "teaching",
                     f"{path.name}:{line_number}: invalid eligible event")
        rows.append(row)
    _require(rows, f"{path.name} is empty")
    _require(len({row["id"] for row in rows}) == len(rows),
             f"duplicate episode ids in {path.name}")
    return rows


@dataclass(frozen=True)
class Episode:
    row: dict
    event: TextTeachingEvent
    prompt_ids: torch.Tensor
    prompt_mask: torch.Tensor
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    response_mask: torch.Tensor

    @property
    def expected(self) -> str:
        return self.row["expected_response"]

    @property
    def prompt(self) -> str:
        return self.row["prompt"]


def encode_prompt(tokenizer, text: str, *, max_tokens: int) -> tuple[torch.Tensor, torch.Tensor]:
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": text}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False)
    ids = tokenizer(rendered, add_special_tokens=False)["input_ids"]
    _require(0 < len(ids) <= max_tokens,
             f"prompt exceeds token budget {max_tokens}")
    value = torch.tensor([ids], dtype=torch.long)
    return value, torch.ones_like(value, dtype=torch.bool)


def build_episode(row: dict, tokenizer, protocol: TextProtocol, *,
                  max_prompt_tokens: int, max_response_tokens: int) -> Episode:
    prompt_ids, prompt_mask = encode_prompt(
        tokenizer, row["prompt"], max_tokens=max_prompt_tokens)
    response = tokenizer(row["expected_response"], add_special_tokens=False)["input_ids"]
    eos = tokenizer.eos_token_id
    _require(eos is not None, "tokenizer must declare EOS")
    response = list(response) + [int(eos)]
    _require(0 < len(response) <= max_response_tokens,
             f"row {row['id']} exceeds response budget {max_response_tokens}")
    response_tensor = torch.tensor([response], dtype=torch.long)
    input_ids = torch.cat((prompt_ids, response_tensor), dim=1)
    attention = torch.ones_like(input_ids, dtype=torch.bool)
    response_mask = torch.zeros_like(attention)
    response_mask[:, prompt_ids.shape[1]:] = True
    event = TextTeachingEvent(
        event_id=row["id"], world_uid=row["world_uid"], scope=row["scope_uid"],
        split_role=row["split_role"], teaching_or_query=row["teaching_or_query"],
        prompt_sha256=row["prompt_sha256"],
        response_sha256=row["response_sha256"],
        target_available_at=float(row["target_available_at"]),
        causal_cutoff=float(row["causal_cutoff"]),
        eligible=bool(row["eligible"]), source=row["source"],
        verified=bool(row["verified"]),
        route_key=tuple(float(item) for item in prompt_route_key(prompt_ids)),
        concept_family=row["concept_or_rule_family"],
        contamination_group=row.get("contamination_group", row["source_group"]),
        feedback_type=row["feedback_type"], supersedes=row.get("supersedes"))
    return Episode(row=row, event=event, prompt_ids=prompt_ids,
                   prompt_mask=prompt_mask, input_ids=input_ids,
                   attention_mask=attention, response_mask=response_mask)


def episodes_for_teaching(directory: Path, manifest: Mapping[str, Any],
                          tokenizer, protocol: TextProtocol, *,
                          max_prompt_tokens: int, max_response_tokens: int,
                          limit: int) -> list[Episode]:
    files = ("teaching", "correction") if protocol.world_uid == manifest["primary_world"] \
        else ("wrong_world_teaching",)
    rows = []
    for role in files:
        rows.extend(load_rows(directory, manifest, role))
    rows = [row for row in rows if row["eligible"]
            and row["world_uid"] == protocol.world_uid
            and row["scope_uid"] == protocol.scope]
    rows.sort(key=lambda row: (float(row["target_available_at"]), row["id"]))
    rows = rows[:limit]
    return [build_episode(row, tokenizer, protocol,
                          max_prompt_tokens=max_prompt_tokens,
                          max_response_tokens=max_response_tokens)
            for row in rows]


def episodes_for_role(directory: Path, manifest: Mapping[str, Any], role: str,
                      tokenizer, protocol: TextProtocol, *,
                      max_prompt_tokens: int, max_response_tokens: int,
                      limit: int | None = None) -> list[Episode]:
    rows = load_rows(directory, manifest, role)
    if limit is not None:
        rows = rows[:limit]
    return [build_episode(row, tokenizer, protocol,
                          max_prompt_tokens=max_prompt_tokens,
                          max_response_tokens=max_response_tokens)
            for row in rows]


def _tokenizer_digest(lock: Mapping[str, Any]) -> str:
    tokenizer_files = {name: digest for name, digest in
                       lock["snapshot_files_sha256"].items()
                       if name.startswith(("tokenizer", "special_tokens",
                                           "chat_template"))}
    _require(tokenizer_files, "lock lacks tokenizer files")
    return fingerprint(tokenizer_files)


def _parameter_binding(lock: Mapping[str, Any]) -> str:
    return _sha256_json(sorted(lock["snapshot_files_sha256"].items()))


def protocol_for_world(config: Mapping[str, Any], lock: Mapping[str, Any], *,
                       hidden_size: int, world_uid: str) -> TextProtocol:
    bank = config["bank"]
    features = config["features"]
    read = config["read_path"]
    projection = fixed_projection(int(hidden_size), int(bank["width"]),
                                  int(features["target_projection_seed"]))
    digest = projection_digest(projection)
    cfg_layers = config.get("runtime", {})
    _require(int(cfg_layers.get("model_loads_per_process", 1)) == 1,
             "exactly one model load per process is required")
    scope = f"{world_uid}/text"
    return TextProtocol(
        world_uid=world_uid, scope=scope,
        expert_count=int(bank["experts"]), width=int(bank["width"]),
        hidden_width=int(bank["hidden_width"]), hidden_size=int(hidden_size),
        feature_sites=tuple(int(site) for site in features["sites"]),
        target_projection_seed=int(features["target_projection_seed"]),
        target_projection_sha256=digest, read_gain=float(read["gain"]),
        read_path_sha256=digest,
        route_mode=config["routing"]["mode"], route_seed=int(config["routing"]["seed"]),
        theta_revision=str(config["theta"]["revision"]),
        theta_lock_sha256=_sha256_json(lock),
        parameter_stamp_sha256=_parameter_binding(lock),
        tokenizer_sha256=_tokenizer_digest(lock),
        steps_per_event=1, total_step_cap=int(config["transaction"]["total_step_cap"]),
        clock_period=int(config["transaction"]["clock_period"]),
        gradient_norm_max=float(config["transaction"]["gradient_norm_max"]))


def build_learning_owners(protocol: TextProtocol, config: Mapping[str, Any], *,
                          personality: PersonalityIdentity | None = None,
                          observer: Observer | None = None,
                          spine: Observer | None = None):
    personality = personality or PersonalityIdentity.default()
    bank = ExpertBank(
        expert_count=protocol.expert_count, width=protocol.width,
        hidden_width=protocol.hidden_width,
        seed=int(config["bank"]["seed"]),
        lr=float(config["bank"]["lr"]),
        betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0,
        scope=protocol.scope)
    coordinator = ObservationCoordinator(
        scope=protocol.scope, width=protocol.width,
        seed=int(config["bank"]["seed"]), personality=personality,
        observer=observer, spine=spine)
    controller = PlasticLearningController(
        bank, scope=protocol.scope, personality=personality,
        coordinator=coordinator,
        max_steps=int(config["transaction"]["max_steps"]),
        total_step_cap=int(config["transaction"]["total_step_cap"]),
        clock_period=int(config["transaction"]["clock_period"]),
        gradient_norm_max=float(config["transaction"]["gradient_norm_max"]))
    return bank, coordinator, controller


def build_read_only_learner(protocol: TextProtocol, config: Mapping[str, Any],
                            adapter, p_only: Mapping[str, Any] | None = None,
                            *, personality: PersonalityIdentity | None = None):
    bank = ExpertBank(
        expert_count=protocol.expert_count, width=protocol.width,
        hidden_width=protocol.hidden_width, seed=int(config["bank"]["seed"]),
        lr=float(config["bank"]["lr"]))
    learner = NestedTextLearner(adapter, bank=bank, protocol=protocol,
                                personality=personality)
    if p_only is not None:
        validate_p_only_payload(p_only, protocol=protocol,
                                route_permutation=learner.route.permutation)
        learner.load_p_only(p_only)
    return learner


def state_digest(learner: NestedTextLearner) -> str:
    controller = learner.controller
    _require(controller is not None, "state digest requires the controller")
    parameters = {
        f"{expert_id}.{name}": _tensor_sha(parameter)
        for expert_id in range(learner.bank.expert_count)
        for name, parameter in learner.bank.expert(expert_id).named_parameters()}
    moments = {}
    for expert_id in range(learner.bank.expert_count):
        for key, value in learner.bank.optimizer_state_for_expert(expert_id).items():
            if isinstance(value, torch.Tensor):
                moments[key] = _tensor_sha(value)
            elif isinstance(value, dict):
                moments[key] = {name: (_tensor_sha(item)
                                       if isinstance(item, torch.Tensor) else item)
                                for name, item in value.items()}
            else:
                moments[key] = value
    coordinator = controller.coordinator.checkpoint_payload()
    return _sha256_json({
        "parameters": parameters, "moments": moments,
        "expert_steps": list(learner.bank.expert_steps),
        "clock": controller.clock.to_dict(),
        "dedup": controller.dedup.event_ids,
        "revision": controller.revision,
        "fast_state": _tensor_sha(coordinator["fast_state"]),
        "transition_count": coordinator["transition_count"],
        "observation_ids": [item["event_id"] for item in coordinator["observations"]],
        "bucket": learner.route.mode,
    })


# ---------------------------------------------------------------------------
# P-only request and scoring
# ---------------------------------------------------------------------------


def empty_scope(protocol: TextProtocol) -> EligibilityScope:
    return EligibilityScope(subject_id=protocol.world_uid,
                            branch_id="nested-text",
                            purpose="nested_text_read",
                            as_of=10 ** 9,
                            mode="assistant_for_source")


def build_read_request(learner: NestedTextLearner, session: MemorySession,
                       episode: Episode, *, route_mode: str | None,
                       task_id: str):
    protocol = learner.protocol
    read = learner.plastic_read(episode.prompt_ids, episode.prompt_mask,
                                route_mode=route_mode)
    _require(learner.controller is None and learner.coordinator is None,
             "P-only requests cannot carry a learning controller/coordinator")
    task = TaskGraph(task_id=task_id, goal=episode.prompt, revision=0)
    device = session.device
    request = session.prepare(
        prompt_ids=episode.prompt_ids.to(device),
        prompt_mask=episode.prompt_mask.to(device),
        query=episode.prompt, scope=empty_scope(protocol), task=task,
        training=False, enabled_ports=(), core_impl="none",
        enabled_branches=("plastic",), plastic=read,
        identity=None, identity_enabled=False)
    frame = request.frame
    snapshot = session.ledger.snapshot(empty_scope(protocol))
    audit = {
        "memory_spans": len(frame.memory.spans),
        "memory_valid_any": bool(frame.memory.valid.any()),
        "selection_records": len(request.selection.records),
        "ledger_records": len(snapshot.records),
        "enabled_branches": list(frame.enabled_branches),
        "controller_constructed": learner.controller is not None,
        "coordinator_constructed": learner.coordinator is not None,
        "updates_enabled": False,
        "expert_id": frame.plastic.expert_id,
        "revision": frame.plastic.revision,
        "route_mode": frame.plastic.route_mode,
        "sources_present": bool(frame.memory.spans) or bool(request.selection.records),
    }
    return request, audit


def generate_text(session: MemorySession, request, *, max_new_tokens: int) -> str:
    ids = session.generate(request, max_new_tokens=max_new_tokens)
    prompt_length = request.frame.sense.prompt_ids.shape[1]
    generated = ids[0, prompt_length:].tolist()
    eos = session.tokenizer.eos_token_id
    if eos is not None and eos in generated:
        generated = generated[:generated.index(eos)]
    return session.tokenizer.decode(generated, skip_special_tokens=True).strip()


def normalize(value: str) -> str:
    return " ".join(value.lower().strip().split())


def score_episodes(learner: NestedTextLearner, session: MemorySession,
                   episodes: list[Episode], *, route_mode: str | None,
                   max_new_tokens: int, label: str,
                   latency: list[float] | None = None) -> dict:
    results = []
    correct = 0
    failures = 0
    for index, episode in enumerate(episodes):
        record = {"id": episode.row["id"], "label": label,
                  "expected": episode.expected,
                  "prompt_sha256": episode.row["prompt_sha256"]}
        try:
            request, audit = build_read_request(
                learner, session, episode, route_mode=route_mode,
                task_id=f"{label}:{episode.row['id']}")
            started = time.perf_counter()
            output = generate_text(session, request, max_new_tokens=max_new_tokens)
            if latency is not None:
                latency.append(time.perf_counter() - started)
            request.release()
            record.update({"output": output, "request_audit": audit})
            matched = normalize(output) == normalize(episode.expected)
            record["correct"] = matched
            correct += int(matched)
        except BaseException as error:  # raw failure is preserved, never retried
            failures += 1
            record.update({"error": f"{type(error).__name__}: {error}",
                           "traceback": traceback.format_exc()})
        results.append(record)
    return {"label": label, "count": len(results),
            "correct": correct, "failures": failures,
            "accuracy": (correct / len(results)) if results else 0.0,
            "results": results}


def false_recall(score: Mapping[str, Any], taught_answers: set[str]) -> float:
    total = score["count"] or 1
    false = sum(1 for item in score["results"]
                if item.get("output") and normalize(item["output"]) in
                {normalize(value) for value in taught_answers})
    return false / total


def p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))
    return float(ordered[index])


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------


def run_teach(args, config: dict) -> int:
    output: Path = args.output
    manifest = load_episode_manifest(args.episodes)
    world_uid = (manifest["primary_world"] if args.world_role == "primary"
                 else manifest["wrong_world"])
    with exclusive_mps_worker(ROOT):
        adapter, tokenizer, lock = load_local_mps(
            args.model_path, args.model_lock)
        protocol = protocol_for_world(
            config, lock, hidden_size=adapter.base.config.hidden_size,
            world_uid=world_uid)
        adapter.install_plastic(protocol.plastic_spec)
        _require(adapter.plastic_read_digest() == protocol.read_path_sha256,
                 "adapter read path differs from the protocol")
        personality = PersonalityIdentity.default()
        observer = Observer() if args.observer else None
        spine = Observer() if args.spine else None
        bank, coordinator, controller = build_learning_owners(
            protocol, config, personality=personality,
            observer=observer, spine=spine)
        learner = NestedTextLearner(
            adapter, bank=bank, protocol=protocol, controller=controller,
            coordinator=coordinator, personality=personality,
            tokenizer_sha256=protocol.tokenizer_sha256)
        order = TextAccessOrder()
        order.access("teaching")
        stamp_before = adapter.parameter_stamp()
        episodes = episodes_for_teaching(
            args.episodes, manifest, tokenizer, protocol,
            max_prompt_tokens=args.max_prompt_tokens,
            max_response_tokens=args.max_response_tokens,
            limit=args.max_events)
        _require(episodes, "no eligible teaching episodes")
        transactions, observations = [], []
        started = time.perf_counter()
        update_seconds = 0.0
        for episode in episodes:
            update_started = time.perf_counter()
            outcome = learner.learn_event(
                episode.event, prompt_ids=episode.prompt_ids,
                prompt_mask=episode.prompt_mask,
                input_ids=episode.input_ids,
                attention_mask=episode.attention_mask,
                response_mask=episode.response_mask)
            update_seconds += time.perf_counter() - update_started
            transactions.append(outcome.to_dict())
            observations.append({"event_id": outcome.event_id,
                                 "chunk": outcome.chunk.to_dict()})
        wall = time.perf_counter() - started
        stamp_after = adapter.parameter_stamp()
        _require(stamp_before == stamp_after,
                 "native theta parameter stamp changed during plastic learning")
        _write_jsonl(output / "transactions.jsonl", transactions)
        _write_jsonl(output / "observations.jsonl", observations)
        _write_json(output / "protocol.json", protocol.to_dict())
        checkpoint = learner.save_checkpoint(output / "full_checkpoint.pt")
        p_only_path = learner.save_p_only(output / "p_only" / "state.pt")
        p_only = torch.load(p_only_path, map_location="cpu", weights_only=True)
        _write_json(output / "p_only" / "manifest.json", {
            "format": P_ONLY_FORMAT,
            "bytes": p_only_path.stat().st_size,
            "sha256": _sha256_file(p_only_path),
            "keys": sorted(p_only),
            "forbidden_absent": not any(
                key in p_only for key in ("optimizer", "dedup", "clock",
                                          "revision", "coordinator",
                                          "observations", "moments", "sources")),
            "source_inputs": 0,
        })
        _write_json(output / "parameter_audit.json", {
            "theta_stamp_unchanged": stamp_before == stamp_after,
            "plastic_parameters": [
                f"expert_{index}.{name}"
                for index in range(bank.expert_count)
                for name, _ in bank.expert(index).named_parameters()],
            "plastic_parameter_count": sum(
                1 for expert in bank.experts for _ in expert.parameters()),
            "read_path_trainable_parameters": 0,
            "expert_steps": list(bank.expert_steps),
        })
        _write_json(output / "identity.json", {
            "personality_before": personality.to_dict(),
            "personality_after": learner.personality.to_dict(),
            "stable": personality == learner.personality,
            "observer_reports": len(observer.reports) if observer else 0,
            "spine_reports": len(spine.reports) if spine else 0,
        })
        _write_json(output / "access_order.json", order.snapshot())
        _write_json(output / "cost.json", {
            "events": len(episodes),
            "eligible_events": int(learner.accounting["eligible_events"]),
            "optimizer_steps": int(learner.accounting["optimizer_steps"]),
            "feature_vectors": int(learner.accounting["feature_vectors"]),
            "response_tokens": int(learner.accounting["response_tokens"]),
            "processed_positions": int(learner.accounting["response_tokens"]),
            "wall_seconds": wall,
            "update_seconds": update_seconds,
            "duty_cycle": (update_seconds / wall) if wall > 0 else None,
            "protocol_sha256": protocol.protocol_sha256,
            "peak_driver_bytes": peak_driver_bytes(),
            "checkpoint_bytes": checkpoint.stat().st_size,
            "p_only_bytes": p_only_path.stat().st_size,
            "model_loads": 1, "tokenizer_loads": 1,
            "teacher_requests": len(episodes),
        })
        _write_json(output / "run.json", {
            "format": PROMPT_FORM, "mode": "teach", "status": "completed",
            "world_uid": world_uid, "protocol_sha256": protocol.protocol_sha256,
            "episodes_manifest_sha256": _sha256_file(args.episodes / "manifest.json"),
            "checkpoint": str(checkpoint), "p_only": str(p_only_path),
            "descriptive_only": True, "promotion": False,
            "non_claims": ["no utility, capability or heldout claim",
                           "single seed is descriptive only"],
        })
    return 0


def run_p_only(args, config: dict) -> int:
    output: Path = args.output
    manifest = load_episode_manifest(args.episodes)
    primary_world = manifest["primary_world"]
    wrong_world = manifest["wrong_world"]
    state = torch.load(args.p_state, map_location="cpu", weights_only=True)
    _require(isinstance(state, Mapping) and set(state) == P_ONLY_KEYS,
             "P-only state must contain exactly the declared weights/binding keys")
    _write_json(output / "p_only_input.json", {
        "path": str(args.p_state), "sha256": _sha256_file(args.p_state),
        "keys": sorted(state),
        "forbidden_present": sorted(
            key for key in state if key in
            ("optimizer", "dedup", "clock", "revision", "coordinator",
             "observations", "moments", "sources")),
    })
    with exclusive_mps_worker(ROOT):
        adapter, tokenizer, lock = load_local_mps(
            args.model_path, args.model_lock)
        protocol = protocol_for_world(
            config, lock, hidden_size=adapter.base.config.hidden_size,
            world_uid=primary_world)
        adapter.install_plastic(protocol.plastic_spec)
        validate_p_only_payload(state, protocol=protocol)
        learner = build_read_only_learner(protocol, config, adapter, state)
        personality = PersonalityIdentity.default()
        ledger = RecordLedger(output / "p_only" / "ledger.sqlite")
        session = MemorySession(adapter, tokenizer, ledger)
        # Development first: no heldout exists for this process before the gate.
        order = TextAccessOrder()
        order.advance("development")
        order.access("development")
        dev_rows = {
            "development": episodes_for_role(
                args.episodes, manifest, "development", tokenizer, protocol,
                max_prompt_tokens=args.max_prompt_tokens,
                max_response_tokens=args.max_response_tokens, limit=args.max_dev),
            "correction": episodes_for_role(
                args.episodes, manifest, "correction", tokenizer, protocol,
                max_prompt_tokens=args.max_prompt_tokens,
                max_response_tokens=args.max_response_tokens, limit=args.max_dev),
            "unaffected": episodes_for_role(
                args.episodes, manifest, "unaffected", tokenizer, protocol,
                max_prompt_tokens=args.max_prompt_tokens,
                max_response_tokens=args.max_response_tokens, limit=args.max_dev),
            "unknown": episodes_for_role(
                args.episodes, manifest, "unknown", tokenizer, protocol,
                max_prompt_tokens=args.max_prompt_tokens,
                max_response_tokens=args.max_response_tokens, limit=args.max_dev),
            "revoked": episodes_for_role(
                args.episodes, manifest, "revoked", tokenizer, protocol,
                max_prompt_tokens=args.max_prompt_tokens,
                max_response_tokens=args.max_response_tokens, limit=args.max_dev),
            "legacy_retention": episodes_for_role(
                args.episodes, manifest, "legacy_retention", tokenizer, protocol,
                max_prompt_tokens=args.max_prompt_tokens,
                max_response_tokens=args.max_response_tokens, limit=args.max_dev),
            "wrong_world": episodes_for_role(
                args.episodes, manifest, "wrong_world", tokenizer, protocol,
                max_prompt_tokens=args.max_prompt_tokens,
                max_response_tokens=args.max_response_tokens, limit=args.max_dev),
        }
        preteach = build_read_only_learner(protocol, config, adapter, None)
        reset = build_read_only_learner(protocol, config, adapter, None)
        taught_answers = {row["expected_response"]
                          for role in ("development", "correction", "unaffected")
                          for row in (r.row for r in dev_rows[role])}
        latency: list[float] = []
        scores = {}
        preteach_scores = {}
        for role, rows in dev_rows.items():
            if not rows:
                continue
            scores[role] = score_episodes(
                learner, session, rows, route_mode=None,
                max_new_tokens=args.max_new_tokens,
                label=f"taught_{role}",
                latency=latency if role == "development" else None)
            preteach_scores[role] = score_episodes(
                preteach, session, rows, route_mode=None,
                max_new_tokens=args.max_new_tokens,
                label=f"preteach_{role}")
        shuffled = score_episodes(
            learner, session, dev_rows["development"], route_mode="shuffled",
            max_new_tokens=args.max_new_tokens, label="shuffled_development")
        frozen = preteach_scores["development"]
        reset_score = score_episodes(
            reset, session, dev_rows["development"], route_mode=None,
            max_new_tokens=args.max_new_tokens, label="reset_P_development")
        # Read-only observer parity: identical outputs with a tap attached;
        # no coordinator, spine, moment or update exists in this process.
        read_observer = PlasticReadObserver()
        parity_learner = build_read_only_learner(protocol, config, adapter, state)
        parity_learner.attach_read_observer(read_observer)
        parity_score = score_episodes(
            parity_learner, session, dev_rows["development"], route_mode=None,
            max_new_tokens=args.max_new_tokens, label="spine_absent_development")
        primary_outputs = [item.get("output")
                           for item in scores["development"]["results"]]
        parity_outputs = [item.get("output") for item in parity_score["results"]]
        parity_equal = (primary_outputs == parity_outputs
                        and len(read_observer.reports) == len(dev_rows["development"])
                        and all(report["readonly"]
                                for report in read_observer.reports))
        wrong_world = None
        wrong_on_primary = None
        if args.wrong_world_p is not None:
            wrong_protocol = protocol_for_world(
                config, lock, hidden_size=adapter.base.config.hidden_size,
                world_uid=wrong_world)
            wrong_state = torch.load(args.wrong_world_p, map_location="cpu",
                                     weights_only=True)
            validate_p_only_payload(wrong_state, protocol=wrong_protocol)
            wrong_learner = build_read_only_learner(
                wrong_protocol, config, adapter, wrong_state)
            wrong_world = score_episodes(
                wrong_learner, session, dev_rows["wrong_world"], route_mode=None,
                max_new_tokens=args.max_new_tokens,
                label="wrong_world_P")
            wrong_on_primary = score_episodes(
                wrong_learner, session, dev_rows["development"], route_mode=None,
                max_new_tokens=args.max_new_tokens,
                label="wrong_world_on_primary")
        audit_scores = list(scores.values()) + [parity_score]
        if wrong_world is not None:
            audit_scores.extend([wrong_world, wrong_on_primary])
        audits = [item.get("request_audit")
                  for score in audit_scores
                  for item in score["results"] if item.get("request_audit")]
        source_free = bool(audits) and all(
            audit["sources_present"] is False
            and audit["memory_spans"] == 0
            and audit["selection_records"] == 0
            and audit["ledger_records"] == 0
            and audit["controller_constructed"] is False
            and audit["coordinator_constructed"] is False
            and audit["updates_enabled"] is False
            for audit in audits)
        _write_json(output / "request_audit.json", {
            "requests_audited": len(audits),
            "source_free": source_free,
            "controller_constructed": False,
            "coordinator_constructed": False,
            "updates_enabled": False,
            "moments_loaded": False,
            "dedup_loaded": False,
            "clock_loaded": False,
            "ledger_records": audits[0]["ledger_records"] if audits else None,
            "source_text_present": not source_free,
        })
        _require(source_free, "P-only request constructed a source-bearing state")
        gates = config["gates"]["dev"]
        cost_limits = config["gates"]["cost"]
        new_accuracy = scores.get("development", {}).get("accuracy", 0.0)
        preteach_accuracy = preteach_scores.get("development", {}).get("accuracy", 0.0)
        gain = new_accuracy - preteach_accuracy
        unaffected = scores.get("unaffected", {}).get("accuracy", 0.0)
        correction = scores.get("correction", {}).get("accuracy", 0.0)
        retention_decline = max(
            0.0, preteach_scores.get("legacy_retention", {}).get("accuracy", 0.0)
            - scores.get("legacy_retention", {}).get("accuracy", 0.0))
        unknown_false = false_recall(scores.get("unknown", {}), taught_answers)
        revoked_false = false_recall(scores.get("revoked", {}), taught_answers)
        peak = peak_driver_bytes()
        teach_cost = None
        if args.teach_cost is not None:
            teach_cost = json.loads(Path(args.teach_cost).read_text())
            _require(teach_cost.get("protocol_sha256") == protocol.protocol_sha256,
                     "teach cost report belongs to another protocol")
        cost = {
            "peak_driver_bytes": peak,
            "warm_p95_overhead_ms": p95(latency[1:]) * 1000.0,
            "duty_cycle": (None if teach_cost is None
                           else teach_cost.get("duty_cycle")),
            "duty_cycle_source": (str(args.teach_cost)
                                  if args.teach_cost is not None else None),
            "p_only_bytes": args.p_state.stat().st_size,
            "model_loads": 1, "tokenizer_loads": 1,
            "generated_tokens": sum(
                len(tokenizer.encode(item.get("output", ""),
                                     add_special_tokens=False))
                for score in scores.values() for item in score["results"]),
            "teacher_requests": 0,
            "updates": 0,
        }
        metrics = {
            "new_instance_accuracy": new_accuracy,
            "preteach_accuracy": preteach_accuracy,
            "gain_over_preteach": gain,
            "correction_accuracy": correction,
            "unaffected_accuracy": unaffected,
            "unknown_false_recall": unknown_false,
            "revoked_false_recall": revoked_false,
            "retention_decline": retention_decline,
            "shuffled_accuracy": shuffled["accuracy"],
            "frozen_accuracy": frozen["accuracy"],
            "reset_P_accuracy": reset_score["accuracy"],
            "wrong_world_available": args.wrong_world_p is not None,
            "wrong_world_on_primary_accuracy": (
                None if wrong_on_primary is None
                else wrong_on_primary["accuracy"]),
            "wrong_world_own_accuracy": (
                None if wrong_world is None else wrong_world["accuracy"]),
            "spine_absent_parity": parity_equal,
        }
        failures = evaluate_dev_gate(metrics=metrics, cost=cost, gates=gates,
                                     cost_limits=cost_limits)
        report = {
            "label": "development", "descriptive_only": True, "promotion": False,
            "single_seed": True, "gate_failures": failures,
            "all_declared_gates_passed": not failures,
            "metrics": {**metrics, "cost": cost},
            "controls": {"preteach": preteach_scores.get("development", {}),
                         "reset_P": reset_score, "frozen": frozen,
                         "shuffled_route": shuffled,
                         "spine_absent": parity_score,
                         "wrong_world_P": wrong_world,
                         "wrong_world_on_primary": wrong_on_primary},
            "scores": scores,
        }
        _write_json(output / "gates" / "dev.json", report)
        _write_json(output / "cost.json", cost)
        _write_json(output / "gates" / "access_before_heldout.json",
                    order.snapshot())
        if failures:
            order.record_dev_gate(report)
            _write_json(output / "gates" / "access_final.json", order.snapshot())
            _write_json(output / "run.json", {
                "format": PROMPT_FORM, "mode": "p_only_eval",
                "status": "stopped_before_heldout", "gate_failures": failures,
                "heldout_opened": False})
            return 2
        if args.no_heldout:
            order.record_dev_gate(report)
            _write_json(output / "gates" / "access_final.json", order.snapshot())
            _write_json(output / "run.json", {
                "format": PROMPT_FORM, "mode": "p_only_eval",
                "status": "development_passed_heldout_closed",
                "heldout_opened": False, "descriptive_only": True})
            return 0
        order.record_dev_gate(report)
        order.open_heldout()
        heldout_rows = episodes_for_role(
            args.episodes, manifest, "sealed_final", tokenizer, protocol,
            max_prompt_tokens=args.max_prompt_tokens,
            max_response_tokens=args.max_response_tokens, limit=args.max_heldout)
        try:
            heldout = score_episodes(
                learner, session, heldout_rows, route_mode=None,
                max_new_tokens=args.max_new_tokens, label="heldout")
            heldout_pre = score_episodes(
                preteach, session, heldout_rows, route_mode=None,
                max_new_tokens=args.max_new_tokens, label="heldout_preteach")
        except BaseException:
            order.fail("heldout evaluation raised; no retry is permitted")
            _write_json(output / "gates" / "access_final.json", order.snapshot())
            raise
        _write_json(output / "p_only" / "heldout.json", {
            "taught": heldout, "preteach": heldout_pre,
            "descriptive_only": True, "single_seed": True,
            "retry": False})
        _write_json(output / "gates" / "heldout.json", {
            "heldout_opened": True, "retry": False,
            "taught_accuracy": heldout["accuracy"],
            "preteach_accuracy": heldout_pre["accuracy"],
            "failures": heldout["failures"],
            "descriptive_only": True, "single_seed": True})
        _write_json(output / "gates" / "access_final.json", order.snapshot())
        _write_json(output / "run.json", {
            "format": PROMPT_FORM, "mode": "p_only_eval",
            "status": "completed", "heldout_opened": True,
            "descriptive_only": True, "promotion": False,
            "non_claims": ["single-seed descriptive diagnostic only",
                           "no promotion or capability claim"]})
    return 0


def run_resume(args, config: dict) -> int:
    output: Path = args.output
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    _require(payload.get("format") == "replica.nested_text.checkpoint.v1",
             "resume requires a nested-text full checkpoint")
    protocol = TextProtocol.from_dict(payload["text"]["protocol"])
    manifest = load_episode_manifest(args.episodes)
    with exclusive_mps_worker(ROOT):
        adapter, tokenizer, lock = load_local_mps(args.model_path, args.model_lock)
        live_protocol = protocol_for_world(
            config, lock, hidden_size=adapter.base.config.hidden_size,
            world_uid=protocol.world_uid)
        _require(live_protocol == protocol,
                 "live release/config differs from the checkpoint binding")
        adapter.install_plastic(protocol.plastic_spec)
        _require(adapter.plastic_read_digest() == protocol.read_path_sha256,
                 "resume adapter read path differs")
        personality = PersonalityIdentity.from_dict(
            payload["text"]["personality"])

        def make_learner():
            bank, coordinator, controller = build_learning_owners(
                protocol, config, personality=personality)
            return NestedTextLearner(
                adapter, bank=bank, protocol=protocol, controller=controller,
                coordinator=coordinator, personality=personality,
                tokenizer_sha256=protocol.tokenizer_sha256)

        resumed = make_learner()
        resumed.load_checkpoint(payload)
        reference = make_learner()
        reference.load_checkpoint(payload)
        _require(state_digest(resumed) == state_digest(reference),
                 "two loads of one checkpoint disagree")
        rows = []
        for role in ("teaching", "correction"):
            rows.extend(load_rows(args.episodes, manifest, role))
        rows = [row for row in rows if row["eligible"]
                and row["world_uid"] == protocol.world_uid]
        rows.sort(key=lambda row: (float(row["target_available_at"]), row["id"]))
        next_row = None
        for row in rows:
            if row["id"] == args.next_event:
                next_row = row
                break
        _require(next_row is not None,
                 f"next event {args.next_event!r} is not an eligible teaching row")
        _require(args.next_event not in resumed.controller.dedup.event_ids,
                 "next event is already committed; choose an untaught event")
        episode = build_episode(next_row, tokenizer, protocol,
                                max_prompt_tokens=args.max_prompt_tokens,
                                max_response_tokens=args.max_response_tokens)
        before = state_digest(resumed)
        _require(before == state_digest(reference),
                 "reference and resumed checkpoints start differently")
        resumed_outcome = resumed.learn_event(
            episode.event, prompt_ids=episode.prompt_ids,
            prompt_mask=episode.prompt_mask, input_ids=episode.input_ids,
            attention_mask=episode.attention_mask,
            response_mask=episode.response_mask)
        reference_outcome = reference.learn_event(
            episode.event, prompt_ids=episode.prompt_ids,
            prompt_mask=episode.prompt_mask, input_ids=episode.input_ids,
            attention_mask=episode.attention_mask,
            response_mask=episode.response_mask)
        after_resumed = state_digest(resumed)
        after_reference = state_digest(reference)
        equal = after_resumed == after_reference
        if args.expected_state is not None:
            expected = json.loads(Path(args.expected_state).read_text())
            equal = equal and expected.get("after_state") == after_resumed
        _write_json(output / "resume.json", {
            "format": PROMPT_FORM, "mode": "resume",
            "next_event": args.next_event,
            "before_state": before,
            "after_state": after_resumed,
            "reference_after_state": after_reference,
            "restart_equal": equal,
            "resumed_outcome": resumed_outcome.to_dict(),
            "reference_outcome": reference_outcome.to_dict(),
            "transaction_retried": bool(resumed_outcome.transaction.retried),
            "descriptive_only": True,
        })
        _write_json(output / "run.json", {
            "format": PROMPT_FORM, "mode": "resume", "status": "completed",
            "restart_equal": equal})
    return 0 if equal else 3


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("teach", "p-only-eval", "resume"),
                        required=True)
    parser.add_argument("--protocol", type=Path,
                        default=ROOT / "configs/nested_text_nl1_v1.json")
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("mps",), default="mps")
    parser.add_argument("--world-role", choices=("primary", "wrong_world"),
                        default="primary")
    parser.add_argument("--max-events", type=int, default=32)
    parser.add_argument("--max-dev", type=int, default=32)
    parser.add_argument("--max-heldout", type=int, default=32)
    parser.add_argument("--max-prompt-tokens", type=int, default=256)
    parser.add_argument("--max-response-tokens", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--observer", action="store_true")
    parser.add_argument("--spine", action="store_true")
    parser.add_argument("--p-state", type=Path)
    parser.add_argument("--wrong-world-p", type=Path)
    parser.add_argument("--teach-cost", type=Path,
                        help="measured teach-mode cost.json for the duty-cycle gate")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--next-event")
    parser.add_argument("--expected-state", type=Path)
    parser.add_argument("--no-updates", action="store_true")
    parser.add_argument("--no-replay", action="store_true")
    parser.add_argument("--no-heldout", action="store_true")
    args = parser.parse_args()
    if args.device != "mps":
        parser.error("only the declared MPS device is supported")
    if args.mode == "p-only-eval" and args.p_state is None:
        parser.error("--p-state is required for p-only-eval")
    if args.mode == "resume" and (args.checkpoint is None or args.next_event is None):
        parser.error("--checkpoint and --next-event are required for resume")
    if args.mode == "teach" and args.no_updates:
        parser.error("teach mode must perform its declared update transaction")
    if args.mode != "p-only-eval" and (args.p_state is not None
                                       or args.wrong_world_p is not None
                                       or args.teach_cost is not None):
        parser.error("P-only state paths are only valid for p-only-eval")
    if args.mode == "p-only-eval" and args.checkpoint is not None:
        parser.error("a full checkpoint is not a P-only input")
    if args.max_events < 1 or args.max_new_tokens < 1:
        parser.error("positive event/generation budgets are required")
    try:
        _fresh_output(args.output)
        from replica_lab.profiling import RunProfiler
        with RunProfiler("mps", args.output) as profiler:
            config = json.loads(args.protocol.read_text(encoding="utf-8"))
            _require(config.get("format") == "replica.nested_text.nl1.v1",
                     "unsupported nested-text protocol")
            _require(config.get("one_clock_control_only") is True,
                     "the baseline must remain one-clock only")
            _require(config.get("replay_bytes") == 0,
                     "the one-clock baseline forbids replay bytes")
            if args.mode == "teach":
                return run_teach(args, config)
            if args.mode == "p-only-eval":
                return run_p_only(args, config)
            return run_resume(args, config)
    except BaseException as error:
        try:
            _write_json(args.output / "failure.json", {
                "status": "failed", "mode": args.mode,
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
                "raw_outputs_preserved": True})
        except BaseException:
            pass
        raise


if __name__ == "__main__":
    sys.exit(main())

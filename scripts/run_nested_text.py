#!/usr/bin/env python3
"""One-clock nested-text MiniCPM5 runner (init / teach / P-only / resume).

This entry point is deliberately conservative:

* one native model/tokenizer load per process, one exclusive MPS worker,
  offline execution and no CPU fallback;
* a fresh output directory that is never reused;
* ``init`` (v2 only) writes the null full checkpoint and P-only export with
  zero learning events/updates plus parameter/access/cost audits;
* ``teaching`` runs only eligible teaching/correction events and stops;
* ``p-only-eval`` loads weights only (no controller/coordinator/moments/dedup)
  at most once and only after that gate passes (v2 uses corrected gates and
  never retries held-out);
* ``resume`` restores the full checkpoint and applies exactly one next event,
  comparing its state digest with an uninterrupted reference from the same
  checkpoint.

V2 evaluation uses the declared 24-token budget, stop tokens ``[1, 130073]``
and the shared ``nested_text_scoring`` grammar, retains raw/stop metadata and
reports measured per-phase latency and work counts.  V1 behavior and report
schemas are preserved.

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
from contextlib import contextmanager
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
from replica_lab.nested_text_scoring import (SCORING_VERSION, STOP_TOKENS,  # noqa: E402
                                             finalize_generation, score_response)

PROMPT_FORM = "replica.nested_text.run.v1"
CONFIG_FORMAT_V1 = "replica.nested_text.nl1.v1"
CONFIG_FORMAT_V2 = "replica.nested_text.nl1.v2"
MANIFEST_FORMATS = frozenset({
    "replica.nested_text.episode_manifest.v1",
    "replica.nested_text.episode_manifest.v2",
})
LATENCY_FIELDS = ("plastic_read_seconds", "session_prepare_seconds",
                  "generation_finalize_seconds", "release_seconds",
                  "total_seconds")
WORK_COUNTERS = ("plastic_feature_forward_calls",
                 "plastic_feature_prompt_positions",
                 "session_sense_forward_calls",
                 "session_sense_prompt_positions",
                 "generation_prefill_forward_calls",
                 "generation_prefill_positions",
                 "generation_decode_forward_calls",
                 "decode_positions",
                 "generated_ids",
                 "backward_calls",
                 "replay_calls",
                 "response_supervised_tokens")


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
    _require(manifest.get("format") in MANIFEST_FORMATS,
             "unsupported episode manifest")
    for name, digest in manifest["files_sha256"].items():
        path = directory / name
        _require(path.exists(), f"episode file missing: {name}")
        _require(_sha256_file(path) == digest,
                 f"episode file digest differs: {name}")
    return manifest


def config_format(config: Mapping[str, Any]) -> str:
    value = config.get("format")
    _require(value in (CONFIG_FORMAT_V1, CONFIG_FORMAT_V2),
             "unsupported nested-text protocol")
    return str(value)


def is_v2(config: Mapping[str, Any]) -> bool:
    return config_format(config) == CONFIG_FORMAT_V2


def validate_run_mode(config: Mapping[str, Any], mode: str) -> None:
    """Dispatch v2 modes from the declaration and reject init for v1."""
    if config_format(config) == CONFIG_FORMAT_V1:
        _require(mode != "init",
                 "init mode belongs to the v2 protocol; v1 has no init mode")
        return
    allowed = config.get("allowed_native_modes")
    _require(isinstance(allowed, list) and allowed,
             "the v2 protocol must declare allowed_native_modes")
    _require(mode in allowed,
             f"mode {mode!r} is not admitted by the v2 protocol: {allowed}")


def validate_v2_config(config: Mapping[str, Any]) -> None:
    """Validate the additive v2 bindings the runner relies on."""
    _require(config_format(config) == CONFIG_FORMAT_V2,
             "v2 validation requires the v2 protocol")
    bank = config["bank"]
    routing = config["routing"]
    _require(int(bank["experts"]) == 1,
             "the v2 baseline declares exactly one shared expert")
    _require(routing.get("mode") == "fixed"
             and routing.get("constant_single_expert") is True
             and int(routing.get("expert_id", 0)) == 0,
             "the v2 baseline requires constant single-expert routing")
    for name in ("labels_used", "world_uid_used", "gold_record_ids_used",
                 "adaptive_router"):
        _require(routing.get(name) is False,
                 f"v2 routing cannot enable {name}")
    generation = config["generation"]
    _require(int(generation["max_new_tokens"]) >= 1,
             "v2 generation budget must be positive")
    _require(tuple(int(token) for token in generation["stop_tokens"])
             == tuple(STOP_TOKENS),
             "v2 stop tokens differ from the scoring contract")
    _require(generation.get("scoring_version") == SCORING_VERSION,
             "v2 generation/scoring versions differ")
    _require(generation.get("token_budget_credit") is False
             and generation.get("empty_generation_credit") is False,
             "v2 cannot credit empty or budget-truncated generations")
    workload = config["workload"]
    attempts = workload["attempts"]
    _require(sum(int(value) for value in attempts.values())
             == int(workload["scoring_attempts"]),
             "v2 workload attempts do not sum to the declared total")
    _require(int(workload["teaching_events_for_this_baseline"]) == 0,
             "this v2 baseline performs no teaching events")


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
        route_mode=config["routing"]["mode"],
        route_seed=int(config["routing"].get("seed", 0)),
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


def read_plastic(learner: NestedTextLearner, episode: Episode,
                 route_mode: str | None):
    return learner.plastic_read(episode.prompt_ids, episode.prompt_mask,
                                route_mode=route_mode)


def prepare_read_request(learner: NestedTextLearner, session: MemorySession,
                         episode: Episode, read, *, task_id: str):
    protocol = learner.protocol
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


def build_read_request(learner: NestedTextLearner, session: MemorySession,
                       episode: Episode, *, route_mode: str | None,
                       task_id: str):
    read = read_plastic(learner, episode, route_mode)
    return prepare_read_request(learner, session, episode, read,
                                task_id=task_id)


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
    return _percentile(values, 0.95)


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round(quantile * (len(ordered) - 1))))
    return float(ordered[index])


def p50(values: list[float]) -> float:
    return _percentile(values, 0.5)


def workload_duty_cycle(*, learning_update_seconds: float,
                        teaching_end_to_end_seconds: float,
                        evaluation_end_to_end_seconds: float) -> float:
    """Amortized duty over the declared teaching + evaluation workload.

    A missing or negative component fails closed; idle time is never added.
    """
    values = {"learning_update_seconds": learning_update_seconds,
              "teaching_end_to_end_seconds": teaching_end_to_end_seconds,
              "evaluation_end_to_end_seconds": evaluation_end_to_end_seconds}
    for name, value in values.items():
        _require(isinstance(value, (int, float)) and not isinstance(value, bool)
                 and float(value) >= 0.0,
                 f"duty-cycle component {name} is required and nonnegative")
    denominator = (float(teaching_end_to_end_seconds)
                   + float(evaluation_end_to_end_seconds))
    _require(denominator > 0.0, "duty-cycle denominator must be positive")
    return float(learning_update_seconds) / denominator


_MISSING = object()


class NativeForwardCounter:
    """Record one actual native forward call and the positions it processed."""

    def __init__(self) -> None:
        self.phase: str | None = None
        self.records: list[tuple[str, int]] = []

    def record(self, width: int) -> None:
        _require(self.phase in {"plastic_read", "session_prepare", "generation"},
                 "native forward outside a declared accounting phase")
        _require(type(width) is int and width > 0,
                 "native forward must process at least one position")
        self.records.append((str(self.phase), width))


@contextmanager
def counted_native_forwards(adapter):
    """Count real ``_native`` calls/positions for the current phase.

    ``sense``, ``encode_sites`` and ``forward_with_frame`` all reach the same
    private native pass, so one tap counts plastic sensing, session sensing,
    generation prefill and decode steps without inferring anything from the
    response tokens.
    """
    counter = NativeForwardCounter()
    stored = vars(adapter).get("_native", _MISSING)
    original = getattr(adapter, "_native")

    def wrapper(*args, **kwargs):
        ids = args[0] if args else kwargs.get("input_ids")
        _require(ids is not None, "native forward requires input IDs")
        counter.record(int(torch.as_tensor(ids).shape[-1]))
        return original(*args, **kwargs)

    setattr(adapter, "_native", wrapper)
    try:
        yield counter
    finally:
        if stored is _MISSING:
            try:
                delattr(adapter, "_native")
            except AttributeError:
                pass
        else:
            setattr(adapter, "_native", stored)


def phase_work_from_records(records: list[tuple[str, int]], *,
                            generated_ids: int, stop_reason: str,
                            stop_token: int | None) -> dict:
    """Phase/work counts from measured native calls, never from decoded text."""
    phases: dict[str, list[int]] = {"plastic_read": [],
                                    "session_prepare": [],
                                    "generation": []}
    for phase, width in records:
        _require(phase in phases, f"unknown measured phase {phase!r}")
        phases[phase].append(int(width))
    generation = phases["generation"]
    prefill = generation[0] if generation else 0
    decode = generation[1:]
    return {
        "plastic_feature_forward_calls": len(phases["plastic_read"]),
        "plastic_feature_prompt_positions": sum(phases["plastic_read"]),
        "session_sense_forward_calls": len(phases["session_prepare"]),
        "session_sense_prompt_positions": sum(phases["session_prepare"]),
        "generation_prefill_forward_calls": 1 if generation else 0,
        "generation_prefill_positions": prefill,
        "generation_decode_forward_calls": len(decode),
        "decode_positions": sum(decode),
        "generated_ids": int(generated_ids),
        "backward_calls": 0,
        "replay_calls": 0,
        "response_supervised_tokens": 0,
        "stop_reason": str(stop_reason),
        "stop_token": stop_token,
        "truncated": str(stop_reason) != "stop_token",
    }


def score_generation(expected: str, result, *,
                     candidate_answers: tuple[str, ...] = ()):
    """Apply the shared v2 grammar to one finalized generation."""
    return score_response(expected, result.text,
                          candidate_answers=candidate_answers,
                          truncated=result.truncated)


def generate_episode_v2(learner: NestedTextLearner, session: MemorySession,
                        episode: Episode, *, route_mode: str | None,
                        task_id: str, max_new_tokens: int,
                        stop_tokens: tuple[int, ...]):
    """One P-only request with phase timings and measured work counts.

    The total timer starts before ``plastic_read`` so request construction and
    sensing are included; v1 generation-only latency is left unchanged.
    """
    total_started = time.perf_counter()
    request = None
    counter = None
    result = None
    audit = None
    plastic_seconds = prepare_seconds = 0.0
    generation_seconds = release_seconds = 0.0
    try:
        with counted_native_forwards(learner.adapter) as counter:
            counter.phase = "plastic_read"
            started = time.perf_counter()
            read = read_plastic(learner, episode, route_mode)
            plastic_seconds = time.perf_counter() - started
            counter.phase = "session_prepare"
            started = time.perf_counter()
            request, audit = prepare_read_request(
                learner, session, episode, read, task_id=task_id)
            prepare_seconds = time.perf_counter() - started
            counter.phase = "generation"
            started = time.perf_counter()
            prompt_length = int(request.frame.sense.prompt_ids.shape[1])
            ids = session.generate(request, max_new_tokens=max_new_tokens,
                                   eos_token_id=tuple(stop_tokens))
            generated = tuple(int(token)
                              for token in ids[0, prompt_length:].tolist())
            _require(generated == tuple(int(token)
                                        for token in request.generated_ids),
                     "generated IDs differ from the request record")
            result = finalize_generation(
                generated,
                decode=lambda content: session.tokenizer.decode(
                    list(content), skip_special_tokens=True),
                stop_tokens=stop_tokens, max_new_tokens=max_new_tokens)
            generation_seconds = time.perf_counter() - started
    finally:
        if request is not None:
            started = time.perf_counter()
            request.release()
            release_seconds = time.perf_counter() - started
    total_seconds = time.perf_counter() - total_started
    _require(result is not None and audit is not None,
             "v2 generation did not produce a finalized result")
    latency = {"plastic_read_seconds": plastic_seconds,
               "session_prepare_seconds": prepare_seconds,
               "generation_finalize_seconds": generation_seconds,
               "release_seconds": release_seconds,
               "total_seconds": total_seconds}
    work = phase_work_from_records(
        counter.records, generated_ids=len(result.generated_ids),
        stop_reason=result.stop_reason, stop_token=result.stop_token)
    audit = {**audit, "scoring_version": SCORING_VERSION}
    return result, audit, latency, work


def score_episodes_v2(learner: NestedTextLearner, session: MemorySession,
                      episodes: list[Episode], *, route_mode: str | None,
                      max_new_tokens: int, stop_tokens: tuple[int, ...],
                      label: str, candidate_answers: tuple[str, ...] = (),
                      latency_rows: list[dict] | None = None,
                      work_rows: list[dict] | None = None) -> dict:
    results = []
    correct = 0
    failures = 0
    for episode in episodes:
        record = {"id": episode.row["id"], "label": label,
                  "expected": episode.expected,
                  "prompt_sha256": episode.row["prompt_sha256"],
                  "scoring_version": SCORING_VERSION}
        try:
            result, audit, latency, work = generate_episode_v2(
                learner, session, episode, route_mode=route_mode,
                task_id=f"{label}:{episode.row['id']}",
                max_new_tokens=max_new_tokens, stop_tokens=stop_tokens)
            scored = score_generation(episode.expected, result,
                                      candidate_answers=candidate_answers)
            record.update({
                "output": result.text,
                "correct": scored.correct,
                "reason": scored.reason,
                "scoring_version": scored.scoring_version,
                "generated_ids": list(result.generated_ids),
                "content_ids": list(result.content_ids),
                "stop_reason": result.stop_reason,
                "stop_token": result.stop_token,
                "max_new_tokens": result.max_new_tokens,
                "truncated": result.truncated,
                "request_audit": audit,
                "latency": latency,
                "work": work,
            })
            if latency_rows is not None:
                latency_rows.append({"id": record["id"], "label": label,
                                     **latency})
            if work_rows is not None:
                work_rows.append({"id": record["id"], "label": label,
                                  **work})
            correct += int(scored.correct)
        except BaseException as error:  # raw failure is preserved, never retried
            failures += 1
            record.update({"error": f"{type(error).__name__}: {error}",
                           "traceback": traceback.format_exc()})
        results.append(record)
    return {"label": label, "count": len(results),
            "correct": correct, "failures": failures,
            "accuracy": (correct / len(results)) if results else 0.0,
            "scoring_version": SCORING_VERSION, "results": results}


def summarize_latency(rows: list[dict]) -> dict:
    summary = {}
    for field in LATENCY_FIELDS:
        values = [float(row[field]) for row in rows
                  if isinstance(row.get(field), (int, float))]
        summary[field] = {"count": len(values),
                          "p50_ms": p50(values) * 1000.0,
                          "p95_ms": p95(values) * 1000.0}
    return summary


def sum_phase_work(rows: list[dict]) -> dict:
    totals: dict[str, Any] = {name: 0 for name in WORK_COUNTERS}
    reasons: dict[str, int] = {}
    truncated = 0
    for row in rows:
        for name in WORK_COUNTERS:
            totals[name] += int(row.get(name, 0))
        reason = row.get("stop_reason")
        if isinstance(reason, str):
            reasons[reason] = reasons.get(reason, 0) + 1
        truncated += int(bool(row.get("truncated")))
    totals["stop_reasons"] = dict(sorted(reasons.items()))
    totals["truncated_attempts"] = truncated
    totals["requests"] = len(rows)
    totals["forward_calls"] = {
        "plastic_read": totals["plastic_feature_forward_calls"],
        "session_prepare": totals["session_sense_forward_calls"],
        "generation_prefill": totals["generation_prefill_forward_calls"],
        "generation_decode": totals["generation_decode_forward_calls"],
        "total": (totals["plastic_feature_forward_calls"]
                  + totals["session_sense_forward_calls"]
                  + totals["generation_prefill_forward_calls"]
                  + totals["generation_decode_forward_calls"]),
    }
    return totals


def _merge_request_rows(work_rows: list[dict],
                        latency_rows: list[dict]) -> list[dict]:
    latency_by_key = {(row["id"], row["label"]): row
                      for row in latency_rows}
    requests = []
    for row in work_rows:
        merged = dict(row)
        latency = latency_by_key.get((row["id"], row["label"]))
        if latency is not None:
            merged.update({field: latency[field] for field in LATENCY_FIELDS})
        requests.append(merged)
    return requests


def candidate_answers_for(directory: Path, manifest: Mapping[str, Any], *,
                          roles: tuple[str, ...] = (
                              "development", "correction", "unaffected",
                              "wrong_world")) -> tuple[str, ...]:
    """Expected answers used only to reject conflicting v2 outputs."""
    answers = set()
    for role in roles:
        if role not in manifest["role_files"]:
            continue
        answers.update(row["expected_response"]
                       for row in load_rows(directory, manifest, role))
    _require(answers, "no candidate answers are available")
    return tuple(sorted(answers))


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------


def run_init(args, config: dict) -> int:
    """Initialize the declared null bank; zero updates and zero events."""
    output: Path = args.output
    _require(is_v2(config), "init mode requires the v2 protocol")
    validate_v2_config(config)
    manifest = load_episode_manifest(args.episodes)
    world_uid = (manifest["primary_world"] if args.world_role == "primary"
                 else manifest["wrong_world"])
    started = time.perf_counter()
    with exclusive_mps_worker(ROOT):
        load_started = time.perf_counter()
        adapter, tokenizer, lock = load_local_mps(
            args.model_path, args.model_lock)
        model_load_seconds = time.perf_counter() - load_started
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
        # Initialization never observes or updates; the checkpoint encodes that.
        _require(int(learner.accounting["events_observed"]) == 0,
                 "initialization created a learning event")
        _require(int(learner.accounting["optimizer_steps"]) == 0
                 and controller.clock.eligible_events == 0
                 and controller.revision == 0,
                 "initialization applied an update")
        _require(coordinator.transition_count == 0,
                 "initialization sealed an observation transition")
        _require(list(bank.expert_steps) == [0] * bank.expert_count,
                 "initialization advanced an expert counter")
        _require(all(not bool(expert.down.any()) for expert in bank.experts),
                 "initialization did not produce a null plastic read")
        stamp_before = adapter.parameter_stamp()
        order = TextAccessOrder()  # no teaching/development/heldout access
        _require(order.snapshot()["events"] == [],
                 "initialization accessed an episode role")
        checkpoint_started = time.perf_counter()
        checkpoint = learner.save_checkpoint(output / "full_checkpoint.pt")
        p_only_started = time.perf_counter()
        p_only_path = learner.save_p_only(output / "p_only" / "state.pt")
        save_seconds = time.perf_counter() - p_only_started
        checkpoint_seconds = p_only_started - checkpoint_started
        p_only = torch.load(p_only_path, map_location="cpu", weights_only=True)
        _require(isinstance(p_only, Mapping) and set(p_only) == P_ONLY_KEYS,
                 "initialization P-only state has unexpected keys")
        _require(not any(key in p_only for key in
                         ("optimizer", "dedup", "clock", "revision",
                          "coordinator", "observations", "moments", "sources")),
                 "initialization P-only state carries learning state")
        stamp_after = adapter.parameter_stamp()
        _require(stamp_before == stamp_after,
                 "native theta parameter stamp changed during initialization")
        _write_json(output / "p_only" / "manifest.json", {
            "format": P_ONLY_FORMAT,
            "bytes": p_only_path.stat().st_size,
            "sha256": _sha256_file(p_only_path),
            "keys": sorted(p_only),
            "forbidden_absent": True,
            "source_inputs": 0,
        })
        _write_json(output / "protocol.json", protocol.to_dict())
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
            "initial_read_is_null": all(
                not bool(expert.down.any()) for expert in bank.experts),
            "optimizer_state_entries": sum(
                len(bank.optimizer_state_for_expert(index))
                for index in range(bank.expert_count)),
            "clock": controller.clock.to_dict(),
            "revision": controller.revision,
            "dedup_events": len(controller.dedup.event_ids),
            "transitions": int(coordinator.transition_count),
            "learning_events": 0,
            "updates": 0,
            "controller_constructed": True,
            "coordinator_constructed": True,
            "updates_enabled": False,
        })
        _write_json(output / "access_order.json", {
            **order.snapshot(), "roles_accessed": [],
            "row_accesses": 0, "manifest_loaded": True})
        _write_json(output / "identity.json", {
            "personality_before": personality.to_dict(),
            "personality_after": learner.personality.to_dict(),
            "stable": personality == learner.personality,
            "observer_reports": len(observer.reports) if observer else 0,
            "spine_reports": len(spine.reports) if spine else 0,
        })
        wall = time.perf_counter() - started
        cost = {
            "mode": "init",
            "events": 0,
            "eligible_events": 0,
            "optimizer_steps": 0,
            "feature_vectors": 0,
            "response_tokens": 0,
            "processed_positions": 0,
            "wall_seconds": wall,
            "update_seconds": 0.0,
            "learning_update_seconds": 0.0,
            "teaching_events": 0,
            "end_to_end_seconds": wall,
            "teaching_end_to_end_seconds": wall,
            "duty_cycle_formula": (
                "learning_update_seconds / (teaching_end_to_end_seconds + "
                "evaluation_end_to_end_seconds)"),
            "model_load_seconds": model_load_seconds,
            "checkpoint_seconds": checkpoint_seconds,
            "p_only_seconds": save_seconds,
            "protocol_sha256": protocol.protocol_sha256,
            "peak_driver_bytes": peak_driver_bytes(),
            "checkpoint_bytes": checkpoint.stat().st_size,
            "p_only_bytes": p_only_path.stat().st_size,
            "model_loads": 1, "tokenizer_loads": 1,
            "teacher_requests": 0, "updates": 0, "learning_events": 0,
            "scoring_version": SCORING_VERSION,
        }
        _write_json(output / "cost.json", cost)
        _write_json(output / "run.json", {
            "format": PROMPT_FORM, "mode": "init", "status": "completed",
            "world_uid": world_uid, "protocol_sha256": protocol.protocol_sha256,
            "episodes_manifest_sha256": _sha256_file(
                args.episodes / "manifest.json"),
            "checkpoint": str(checkpoint), "p_only": str(p_only_path),
            "descriptive_only": True, "promotion": False,
            "updates": 0, "learning_events": 0,
            "non_claims": ["initialization only; no learning or capability claim",
                           "single seed is descriptive only"],
        })
    return 0


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


def run_p_only_v1(args, config: dict) -> int:
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
        (output / "p_only").mkdir(parents=True, exist_ok=True)
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
            wrong_world = manifest["wrong_world"]
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
        scored_groups = [*scores.values(), *preteach_scores.values(),
                         shuffled, reset_score, parity_score]
        if wrong_world is not None:
            scored_groups.extend([wrong_world, wrong_on_primary])
        scoring_attempts = sum(len(score["results"])
                               for score in scored_groups)
        audits = [item.get("request_audit")
                  for score in scored_groups
                  for item in score["results"] if item.get("request_audit")]
        audit_complete = len(audits) == scoring_attempts
        source_free = bool(audits) and audit_complete and all(
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
            "scoring_attempts": scoring_attempts,
            "audit_complete": audit_complete,
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
            "checkpoint_bytes": (None if teach_cost is None
                                 else teach_cost.get("checkpoint_bytes")),
            "p_only_bytes": args.p_state.stat().st_size,
            "model_loads": 1, "tokenizer_loads": 1,
            "scoring_attempts": scoring_attempts,
            "primary_generated_tokens": sum(
                len(tokenizer.encode(item.get("output", ""),
                                     add_special_tokens=False))
                for score in scores.values() for item in score["results"]),
            "generated_tokens": sum(
                len(tokenizer.encode(item.get("output", ""),
                                     add_special_tokens=False))
                for score in scored_groups for item in score["results"]),
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
            "preteach_scores": preteach_scores,
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


def run_p_only(args, config: dict) -> int:
    if is_v2(config):
        return run_p_only_v2(args, config)
    return run_p_only_v1(args, config)


def run_p_only_v2(args, config: dict) -> int:
    """V2 development evaluation with stop metadata and phase accounting."""
    evaluation_started = time.perf_counter()
    output: Path = args.output
    _require(is_v2(config), "v2 evaluation requires the v2 protocol")
    validate_v2_config(config)
    workload = config["workload"]
    declared_attempts = int(workload["scoring_attempts"])
    declared_groups = {str(key): int(value)
                       for key, value in workload["attempts"].items()}
    max_new_tokens = int(config["generation"]["max_new_tokens"])
    stop_tokens = tuple(int(token)
                        for token in config["generation"]["stop_tokens"])
    manifest = load_episode_manifest(args.episodes)
    primary_world = manifest["primary_world"]
    wrong_world_uid = manifest["wrong_world"]
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
        "scoring_version": SCORING_VERSION,
    })
    _require(args.wrong_world_p is not None,
             "the declared v2 workload requires the wrong-world P control")
    latency_rows: list[dict] = []
    work_rows: list[dict] = []
    with exclusive_mps_worker(ROOT):
        adapter, tokenizer, lock = load_local_mps(
            args.model_path, args.model_lock)
        protocol = protocol_for_world(
            config, lock, hidden_size=adapter.base.config.hidden_size,
            world_uid=primary_world)
        adapter.install_plastic(protocol.plastic_spec)
        validate_p_only_payload(state, protocol=protocol)
        learner = build_read_only_learner(protocol, config, adapter, state)
        (output / "p_only").mkdir(parents=True, exist_ok=True)
        ledger = RecordLedger(output / "p_only" / "ledger.sqlite")
        session = MemorySession(adapter, tokenizer, ledger)
        # Development first: no heldout is admitted by this workload.
        order = TextAccessOrder()
        order.advance("development")
        order.access("development")
        roles = ("development", "correction", "unaffected", "unknown",
                 "revoked", "legacy_retention", "wrong_world")
        dev_rows = {
            role: episodes_for_role(
                args.episodes, manifest, role, tokenizer, protocol,
                max_prompt_tokens=args.max_prompt_tokens,
                max_response_tokens=args.max_response_tokens,
                limit=args.max_dev)
            for role in roles
        }
        preteach = build_read_only_learner(protocol, config, adapter, None)
        reset = build_read_only_learner(protocol, config, adapter, None)
        candidates = candidate_answers_for(args.episodes, manifest)
        taught_answers = {row["expected_response"]
                          for role in ("development", "correction", "unaffected")
                          for row in (r.row for r in dev_rows[role])}

        def score(learner_value, episodes, *, route_mode, label):
            return score_episodes_v2(
                learner_value, session, episodes, route_mode=route_mode,
                max_new_tokens=max_new_tokens, stop_tokens=stop_tokens,
                label=label, candidate_answers=candidates,
                latency_rows=latency_rows, work_rows=work_rows)

        scores = {}
        preteach_scores = {}
        for role, rows in dev_rows.items():
            if not rows:
                continue
            scores[role] = score(learner, rows, route_mode=None,
                                 label=f"taught_{role}")
            preteach_scores[role] = score(preteach, rows, route_mode=None,
                                          label=f"preteach_{role}")
        shuffled = score(learner, dev_rows["development"],
                         route_mode="shuffled", label="shuffled_development")
        frozen = preteach_scores["development"]
        reset_score = score(reset, dev_rows["development"], route_mode=None,
                            label="reset_P_development")
        read_observer = PlasticReadObserver()
        parity_learner = build_read_only_learner(protocol, config, adapter, state)
        parity_learner.attach_read_observer(read_observer)
        parity_score = score(parity_learner, dev_rows["development"],
                             route_mode=None, label="spine_absent_development")
        primary_outputs = [item.get("output")
                           for item in scores["development"]["results"]]
        parity_outputs = [item.get("output") for item in parity_score["results"]]
        parity_equal = (primary_outputs == parity_outputs
                        and len(read_observer.reports) == len(dev_rows["development"])
                        and all(report["readonly"]
                                for report in read_observer.reports))
        wrong_world_score = None
        wrong_on_primary = None
        wrong_protocol = protocol_for_world(
            config, lock, hidden_size=adapter.base.config.hidden_size,
            world_uid=wrong_world_uid)
        wrong_state = torch.load(args.wrong_world_p, map_location="cpu",
                                 weights_only=True)
        validate_p_only_payload(wrong_state, protocol=wrong_protocol)
        wrong_learner = build_read_only_learner(
            wrong_protocol, config, adapter, wrong_state)
        wrong_world_score = score(wrong_learner, dev_rows["wrong_world"],
                                  route_mode=None, label="wrong_world_P")
        wrong_on_primary = score(wrong_learner, dev_rows["development"],
                                 route_mode=None,
                                 label="wrong_world_on_primary")
        scored_groups = [*scores.values(), *preteach_scores.values(),
                         shuffled, reset_score, parity_score,
                         wrong_world_score, wrong_on_primary]
        group_counts = {
            "taught_primary": sum(len(item["results"])
                                  for item in scores.values()),
            "preteach_primary": sum(len(item["results"])
                                    for item in preteach_scores.values()),
            "shuffled": len(shuffled["results"]),
            "reset_P": len(reset_score["results"]),
            "spine_absent": len(parity_score["results"]),
            "wrong_world_P": len(wrong_world_score["results"]),
            "wrong_world_on_primary": len(wrong_on_primary["results"]),
        }
        for name, required in declared_groups.items():
            _require(group_counts.get(name) == required,
                     f"declared workload group {name} requires {required} "
                     f"attempts; measured {group_counts.get(name)}")
        scoring_attempts = sum(len(item["results"])
                               for item in scored_groups)
        _require(scoring_attempts == declared_attempts,
                 f"declared workload requires {declared_attempts} scoring "
                 f"attempts; measured {scoring_attempts}")
        audits = [item.get("request_audit")
                  for group in scored_groups
                  for item in group["results"] if item.get("request_audit")]
        audit_complete = len(audits) == scoring_attempts
        source_free = bool(audits) and audit_complete and all(
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
            "scoring_attempts": scoring_attempts,
            "audit_complete": audit_complete,
            "source_free": source_free,
            "controller_constructed": False,
            "coordinator_constructed": False,
            "updates_enabled": False,
            "moments_loaded": False,
            "dedup_loaded": False,
            "clock_loaded": False,
            "ledger_records": audits[0]["ledger_records"] if audits else None,
            "source_text_present": not source_free,
            "scoring_version": SCORING_VERSION,
            "candidate_answers": len(candidates),
            "declared_attempts": declared_attempts,
            "group_attempts": group_counts,
        })
        _require(source_free, "P-only request constructed a source-bearing state")
        gates = config["gates"]["dev"]
        cost_limits = config["gates"]["cost"]
        new_accuracy = scores["development"]["accuracy"]
        preteach_accuracy = preteach_scores["development"]["accuracy"]
        gain = new_accuracy - preteach_accuracy
        unaffected = scores["unaffected"]["accuracy"]
        correction = scores["correction"]["accuracy"]
        retention_decline = max(
            0.0, preteach_scores["legacy_retention"]["accuracy"]
            - scores["legacy_retention"]["accuracy"])
        unknown_false = false_recall(scores["unknown"], taught_answers)
        revoked_false = false_recall(scores["revoked"], taught_answers)
        peak = peak_driver_bytes()
        latency_summary = summarize_latency(latency_rows)
        work_totals = sum_phase_work(work_rows)
        totals = [row["total_seconds"] for row in latency_rows]
        warm = totals[1:] if len(totals) > 1 else totals
        warm_p95 = p95(warm) * 1000.0
        teach_cost = None
        learning_update_seconds = None
        teaching_end_to_end_seconds = None
        if args.teach_cost is not None:
            teach_cost = json.loads(Path(args.teach_cost).read_text())
            _require(teach_cost.get("protocol_sha256") == protocol.protocol_sha256,
                     "teach cost report belongs to another protocol")
            learning_update_seconds = teach_cost.get(
                "learning_update_seconds", teach_cost.get("update_seconds"))
            teaching_end_to_end_seconds = teach_cost.get(
                "teaching_end_to_end_seconds", teach_cost.get("end_to_end_seconds"))
            _require(learning_update_seconds is not None
                     and teaching_end_to_end_seconds is not None,
                     "teach cost report lacks a duty-cycle component")
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
            "wrong_world_available": True,
            "wrong_world_on_primary_accuracy": (
                None if wrong_on_primary is None
                else wrong_on_primary["accuracy"]),
            "wrong_world_own_accuracy": (
                None if wrong_world_score is None
                else wrong_world_score["accuracy"]),
            "spine_absent_parity": parity_equal,
        }

        def build(evaluation_end_to_end_seconds: float):
            duty = None
            if teach_cost is not None:
                duty = workload_duty_cycle(
                    learning_update_seconds=float(learning_update_seconds),
                    teaching_end_to_end_seconds=float(
                        teaching_end_to_end_seconds),
                    evaluation_end_to_end_seconds=float(
                        evaluation_end_to_end_seconds))
            cost_value = {
                "peak_driver_bytes": peak,
                "warm_p95_overhead_ms": warm_p95,
                "warm_p95_excludes_first_request": len(totals) > 1,
                "duty_cycle": duty,
                "duty_cycle_source": (str(args.teach_cost)
                                      if teach_cost is not None else None),
                "duty_cycle_formula": (
                    "learning_update_seconds / (teaching_end_to_end_seconds + "
                    "evaluation_end_to_end_seconds)"),
                "checkpoint_bytes": (None if teach_cost is None
                                     else teach_cost.get("checkpoint_bytes")),
                "p_only_bytes": args.p_state.stat().st_size,
                "model_loads": 1, "tokenizer_loads": 1,
                "scoring_attempts": scoring_attempts,
                "primary_generated_tokens": sum(
                    len(item.get("generated_ids") or ())
                    for group in scores.values()
                    for item in group["results"]),
                "generated_tokens": work_totals["generated_ids"],
                "teacher_requests": 0,
                "updates": 0,
                "scoring_version": SCORING_VERSION,
                "learning_update_seconds": learning_update_seconds,
                "teaching_end_to_end_seconds": teaching_end_to_end_seconds,
                "evaluation_end_to_end_seconds": float(
                    evaluation_end_to_end_seconds),
                "evaluation_end_to_end_includes": (
                    "model and protocol initialization, all requests, scoring, "
                    "audits, workload accounting and report construction"),
                "latency_ms": latency_summary,
                "latency_total_semantics": (
                    "plastic_read + session_prepare + generation/finalize + "
                    "release measured from before plastic_read"),
                "work": work_totals,
            }
            failures_value = evaluate_dev_gate(
                metrics=metrics, cost=cost_value, gates=gates,
                cost_limits=cost_limits)
            report_value = {
                "label": "development", "descriptive_only": True,
                "promotion": False, "single_seed": True,
                "gate_failures": failures_value,
                "all_declared_gates_passed": not failures_value,
                "metrics": {**metrics, "cost": cost_value},
                "controls": {"preteach": preteach_scores.get("development", {}),
                             "reset_P": reset_score, "frozen": frozen,
                             "shuffled_route": shuffled,
                             "spine_absent": parity_score,
                             "wrong_world_P": wrong_world_score,
                             "wrong_world_on_primary": wrong_on_primary},
                "scores": scores,
                "preteach_scores": preteach_scores,
                "scoring_version": SCORING_VERSION,
                "workload": {"declared_attempts": declared_attempts,
                             "declared_groups": declared_groups,
                             "measured_groups": group_counts,
                             "scoring_attempts": scoring_attempts},
                "latency_ms": latency_summary,
                "work": work_totals,
            }
            return cost_value, failures_value, report_value

        cost, failures, report = build(
            time.perf_counter() - evaluation_started)
        # Include report serialization in the measured evaluation end-to-end.
        json.dumps(report, indent=2, sort_keys=True)
        cost, failures, report = build(
            time.perf_counter() - evaluation_started)
        json.dumps(report, indent=2, sort_keys=True)
        phase_work_report = {
            "format": "replica.nested_text.phase_work.v2",
            "scoring_version": SCORING_VERSION,
            "attempts": len(work_rows),
            "failed_attempts": scoring_attempts - len(work_rows),
            "scoring_attempts": scoring_attempts,
            "latency_ms": latency_summary,
            "totals": work_totals,
            "requests": _merge_request_rows(work_rows, latency_rows),
        }
        _write_json(output / "gates" / "dev.json", report)
        _write_json(output / "cost.json", cost)
        _write_json(output / "phase_work.json", phase_work_report)
        _write_json(output / "gates" / "access_before_heldout.json",
                    order.snapshot())
        if failures:
            order.record_dev_gate(report)
            _write_json(output / "gates" / "access_final.json", order.snapshot())
            _write_json(output / "run.json", {
                "format": PROMPT_FORM, "mode": "p_only_eval",
                "status": "stopped_before_heldout", "gate_failures": failures,
                "heldout_opened": False, "scoring_version": SCORING_VERSION,
                "scoring_attempts": scoring_attempts,
                "descriptive_only": True})
            return 2
        if args.no_heldout:
            order.record_dev_gate(report)
            _write_json(output / "gates" / "access_final.json", order.snapshot())
            _write_json(output / "run.json", {
                "format": PROMPT_FORM, "mode": "p_only_eval",
                "status": "development_passed_heldout_closed",
                "heldout_opened": False, "no_heldout": True,
                "scoring_version": SCORING_VERSION,
                "scoring_attempts": scoring_attempts,
                "descriptive_only": True})
            return 0
        order.record_dev_gate(report)
        order.open_heldout()
        heldout_rows = episodes_for_role(
            args.episodes, manifest, "sealed_final", tokenizer, protocol,
            max_prompt_tokens=args.max_prompt_tokens,
            max_response_tokens=args.max_response_tokens,
            limit=args.max_heldout)
        heldout_work_start = len(work_rows)
        heldout_latency_start = len(latency_rows)
        try:
            heldout = score(learner, heldout_rows, route_mode=None,
                            label="heldout")
            heldout_pre = score(preteach, heldout_rows, route_mode=None,
                                label="heldout_preteach")
        except BaseException:
            order.fail("heldout evaluation raised; no retry is permitted")
            _write_json(output / "gates" / "access_final.json", order.snapshot())
            raise
        heldout_work = sum_phase_work(work_rows[heldout_work_start:])
        heldout_latency = summarize_latency(latency_rows[heldout_latency_start:])
        _write_json(output / "p_only" / "heldout.json", {
            "taught": heldout, "preteach": heldout_pre,
            "descriptive_only": True, "single_seed": True, "retry": False})
        _write_json(output / "gates" / "heldout.json", {
            "heldout_opened": True, "retry": False,
            "taught_accuracy": heldout["accuracy"],
            "preteach_accuracy": heldout_pre["accuracy"],
            "failures": heldout["failures"],
            "attempts": len(work_rows) - heldout_work_start,
            "latency_ms": heldout_latency, "work": heldout_work,
            "scoring_version": SCORING_VERSION,
            "descriptive_only": True, "single_seed": True})
        _write_json(output / "gates" / "access_final.json", order.snapshot())
        _write_json(output / "run.json", {
            "format": PROMPT_FORM, "mode": "p_only_eval",
            "status": "completed", "heldout_opened": True,
            "scoring_version": SCORING_VERSION,
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
    parser.add_argument("--mode",
                        choices=("init", "teach", "p-only-eval", "resume"),
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
    parser.add_argument("--max-new-tokens", type=int, default=8,
                        help="v1 generation budget; v2 uses generation.max_new_tokens")
    parser.add_argument("--observer", action="store_true")
    parser.add_argument("--spine", action="store_true")
    parser.add_argument("--p-state", type=Path)
    parser.add_argument("--wrong-world-p", type=Path)
    parser.add_argument("--teach-cost", type=Path,
                        help="measured teach/init cost.json for the duty-cycle gate")
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
            validate_run_mode(config, args.mode)
            _require(config.get("one_clock_control_only") is True,
                     "the baseline must remain one-clock only")
            _require(config.get("replay_bytes") == 0,
                     "the one-clock baseline forbids replay bytes")
            if is_v2(config):
                validate_v2_config(config)
            if args.mode == "init":
                return run_init(args, config)
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

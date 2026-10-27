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
* behavioral ``p-only-eval`` additionally requires a preserved passing fit
  receipt that binds the exact P-only file and payload, the live config,
  protocol, episode manifest, scorer/grammar/stopping policy, fit-only
  candidate vocabulary, case coverage and attempts, and the teaching
  run/cost/canonical state hashes.  Preserved ``generated_ids``/``content_ids``
  are re-decoded with the sealed fast tokenizer before they are scored: the
  receipt binds the model revision and lock, the tokenizer file/config digests
  and the decode policy (``skip_special_tokens=True``,
  ``clean_up_tokenization_spaces=False``, stripped whitespace and the terminal
  stop token excluded before decode).  Only the tokenizer is loaded offline;
  no model weights and no MPS device are touched before admission.  An
  EOS-only generation can therefore never carry nonempty scored text.
  Correctness, minimum accuracy, coverage,
  generation/stop evidence, source freedom and generation failures are
  independently recomputed from preserved generations and authoritative fit
  targets; saved score/gate flags are never trusted alone.  ``load_rows`` is
  deny-by-default for every role outside teaching/correction, so development,
  control and sealed labels cannot be parsed before this receipt validates.
  Validation happens before ``torch.load``, before the native worker/model load
  and before any development-role access; other protocols are unaffected;
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
import math
import os
import sys
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

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
from replica_lab.nested_continuation import canonical_state  # noqa: E402
from replica_lab.nested_text_scoring import (ANSWER_GRAMMAR_VERSION,
                                             SCORING_VERSION,
                                             STOP_POLICY_VERSION, STOP_TOKENS,
                                             finalize_generation,
                                             score_response,
                                             is_truncated_stop_reason,
                                             sentence_complete_text)
PROMPT_FORM = "replica.nested_text.run.v1"
CONFIG_FORMAT_BEHAVIORAL = "replica.nested_text.behavioral_reference.v1"
CONFIG_FORMAT_V2 = "replica.nested_text.nl1.v2"
CONFIG_FORMAT_V1 = "replica.nested_text.nl1.v1"
FIT_RECEIPT_FORMAT = "replica.nested_text.fit_receipt.v1"
FIT_RECEIPT_SCHEMA_VERSION = 2
FIT_RECEIPT_FILE = "fit_receipt.json"
# Preserved fit generations are decoded again with this exact offline contract.
TOKENIZER_BINDING_FORMAT = "replica.nested_text.tokenizer_binding.v1"
DECODE_POLICY_VERSION = "replica.nested_text.decode.v1"
DECODE_SKIP_SPECIAL_TOKENS = True
DECODE_CLEAN_UP_TOKENIZATION_SPACES = False
DECODE_STRIP_WHITESPACE = True
DECODE_STOP_HANDLING = "terminal_stop_token_excluded_before_decode"
# Same sealed source the native loader pins; only the tokenizer is ever loaded here.
PINNED_MODEL_REPO = "openbmb/MiniCPM5-2B"
PINNED_MODEL_REVISION = "f97400052a43d642bbc6e9975e2397e3ae6a6b52"
TOKENIZER_FILE_PREFIXES = ("tokenizer", "special_tokens", "chat_template")
# Roles whose targets may be parsed while producing/validating the fit receipt.
FIT_TARGET_ROLES = frozenset({"teaching", "correction",
                              "wrong_world_teaching"})
# Every other role carries development/unaffected/unknown/revoked/retention,
# wrong-world query or sealed labels and is refused until admission.
DEVELOPMENT_CANDIDATE_ROLES = ("development", "correction", "unaffected",
                               "wrong_world")
FIT_CANDIDATE_ROLES = ("teaching", "correction")
_AUDIT_FIELDS = ("sources_present", "memory_spans", "selection_records",
                 "ledger_records", "controller_constructed",
                 "coordinator_constructed", "updates_enabled")
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


class RoleAccessGuard:
    """Process-scoped deny-by-default gate for non-fit label roles.

    Only admission after a validated fit receipt (or after proving the protocol
    does not require one) opens development/control/sealed label parsing.  The
    guard is checked inside :func:`load_rows`, so no candidate-vocabulary or
    episode helper can read those labels by another route.
    """

    def __init__(self) -> None:
        self._admission: str | None = None
        self._validated: set[str] = set()

    @property
    def admission(self) -> str | None:
        return self._admission

    def reset(self) -> None:
        self._admission = None
        self._validated.clear()

    def record_validated(self, receipt: Mapping[str, Any]) -> None:
        """Remember that ``validate_fit_receipt`` accepted this exact receipt."""
        digest = receipt.get("receipt_sha256")
        _require(isinstance(digest, str) and len(digest) == 64,
                 "validated fit receipt lacks its digest")
        self._validated.add(digest)

    def admit_fit_receipt(self, receipt: Mapping[str, Any]) -> None:
        digest = receipt.get("receipt_sha256")
        _require(receipt.get("status") == "passed"
                 and isinstance(digest, str) and digest in self._validated,
                 "development labels require a fully validated passing fit receipt")
        self._admission = f"fit_receipt:{digest}"

    def admit_without_fit_receipt(self, config: Mapping[str, Any]) -> None:
        _require(not fit_receipt_required(config),
                 "this protocol requires a fit receipt before development labels")
        self._admission = "protocol_without_fit_receipt"

    def require(self, role: str) -> None:
        if role in FIT_TARGET_ROLES:
            return
        _require(self._admission is not None,
                 f"development label role {role!r} cannot be parsed before a "
                 "validated fit receipt")


ROLE_ACCESS = RoleAccessGuard()


def reset_role_access() -> None:
    ROLE_ACCESS.reset()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RunnerError(message)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def _implementation_pins(config: Mapping[str, Any]) -> dict[str, str]:
    return {relative: _sha256_file(ROOT / relative)
            for relative in config.get("implementation_paths", ())}



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
def validate_episode_binding(args, config: Mapping[str, Any], manifest: Mapping[str, Any]) -> None:
    episodes = config["episodes"]
    expected_root = (ROOT / episodes["root"]).resolve()
    _require(Path(args.episodes).resolve() == expected_root,
             "CLI episode root differs from the v2 pinned root")
    _require(_sha256_file(args.episodes / "manifest.json")
             == episodes["manifest_sha256"],
             "Episode manifest digest differs from the v2 pin")
    _require(manifest.get("schema_version") == 2,
             "v2 requires the corrected schema-v2 episode manifest")


def is_v2(config: Mapping[str, Any]) -> bool:
    return config_format(config) in (CONFIG_FORMAT_V2, CONFIG_FORMAT_BEHAVIORAL)
def config_format(config: Mapping[str, Any]) -> str:
    value = config.get("format")
    _require(value in (CONFIG_FORMAT_V1, CONFIG_FORMAT_V2,
                       CONFIG_FORMAT_BEHAVIORAL),
             "unsupported nested-text protocol")
    return str(value)



def version_metadata(args) -> dict:
    return {"config_sha256": _sha256_file(args.protocol),
            "stop_policy_version": STOP_POLICY_VERSION}


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
    _require(is_v2(config),
             "v2 protocol or behavioral reference required")
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
    _require(generation.get("sentence_stop", {}).get("enabled") is True,
             "v2 requires the declared first-sentence completion stop")
    _require(generation["sentence_stop"].get("version") == STOP_POLICY_VERSION,
             "v2 stop-policy version differs")
    _require(generation.get("scoring_version") == SCORING_VERSION,
             "v2 generation/scoring versions differ")
    _require(generation.get("token_budget_credit") is False
             and generation.get("empty_generation_credit") is False,
             "v2 cannot credit empty or budget-truncated generations")
    _require(int(config["transaction"].get("steps_per_event", 1)) == 1,
             "the one-clock protocol requires exactly one step per event")
    workload = config["workload"]
    attempts = workload["attempts"]
    _require(sum(int(value) for value in attempts.values())
             == int(workload["scoring_attempts"]),
             "v2 workload attempts do not sum to the declared total")
    if config.get("training_admitted"):
        _require(int(workload["teaching_events_for_this_baseline"]) > 0,
                 "admitted training requires teaching events")
        if config_format(config) == CONFIG_FORMAT_BEHAVIORAL:
            _require(config["transaction"].get("objective")
                     == "verified_response_cross_entropy_through_frozen_decoder",
                     "behavioral reference requires the declared response objective")
            _require(config["transaction"].get("response_only_labels") is True,
                     "behavioral reference requires response-only labels")
            _require(config["training"].get("fit_eval_required") is True
                     and int(workload["fit_eval_attempts"]) > 0,
                     "behavioral reference requires a fit evaluation gate")
    else:
        _require(int(workload["teaching_events_for_this_baseline"]) == 0,
                 "this v2 baseline performs no teaching events")


def load_rows(directory: Path, manifest: Mapping[str, Any], role: str) -> list[dict]:
    _require(role in manifest["role_files"], f"undeclared episode role {role!r}")
    ROLE_ACCESS.require(role)
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


def teaching_rows(directory: Path, manifest: Mapping[str, Any], world_uid: str,
                  scope: str, *, limit: int) -> list[dict]:
    """Deterministic eligible teaching/correction rows for one world and scope."""
    files = ("teaching", "correction") if world_uid == manifest["primary_world"] \
        else ("wrong_world_teaching",)
    rows = []
    for role in files:
        rows.extend(load_rows(directory, manifest, role))
    rows = [row for row in rows if row["eligible"]
            and row["world_uid"] == world_uid
            and row["scope_uid"] == scope]
    rows.sort(key=lambda row: (float(row["target_available_at"]), row["id"]))
    return rows[:limit]


def episodes_for_teaching(directory: Path, manifest: Mapping[str, Any],
                          tokenizer, protocol: TextProtocol, *,
                          max_prompt_tokens: int, max_response_tokens: int,
                          limit: int) -> list[Episode]:
    rows = teaching_rows(directory, manifest, protocol.world_uid,
                         protocol.scope, limit=limit)
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


def _is_lower_hex(value: Any, length: int) -> bool:
    return (isinstance(value, str) and len(value) == length
            and all(character in "0123456789abcdef" for character in value))


def _tokenizer_files(lock: Mapping[str, Any]) -> dict[str, str]:
    """The sealed tokenizer-side files a decode must be reconstructed from."""
    _require(isinstance(lock, Mapping)
             and isinstance(lock.get("snapshot_files_sha256"), Mapping),
             "lock lacks the snapshot digests")
    files = {name: digest for name, digest in
             lock["snapshot_files_sha256"].items()
             if name.startswith(TOKENIZER_FILE_PREFIXES)}
    _require("tokenizer.json" in files, "lock lacks the pinned tokenizer.json")
    for name, digest in files.items():
        _require(_is_lower_hex(digest, 64),
                 f"lock tokenizer digest for {name} is invalid")
    return files


def _tokenizer_digest(lock: Mapping[str, Any]) -> str:
    return fingerprint(_tokenizer_files(lock))


def decode_policy() -> dict:
    """The exact decode settings preserved fit evidence must be replayed under."""
    return {
        "version": DECODE_POLICY_VERSION,
        "skip_special_tokens": DECODE_SKIP_SPECIAL_TOKENS,
        "clean_up_tokenization_spaces": DECODE_CLEAN_UP_TOKENIZATION_SPACES,
        "strip_whitespace": DECODE_STRIP_WHITESPACE,
        "stop_handling": DECODE_STOP_HANDLING,
        "stop_policy_version": STOP_POLICY_VERSION,
    }


def tokenizer_binding(lock: Mapping[str, Any]) -> dict:
    """Model revision, lock and tokenizer file digests plus decode settings."""
    _require(isinstance(lock, Mapping), "tokenizer binding requires the model lock")
    _require(lock.get("repo_id") == PINNED_MODEL_REPO,
             "the tokenizer binding requires the pinned MiniCPM5-2B source")
    _require(lock.get("revision") == PINNED_MODEL_REVISION,
             "the tokenizer binding requires the pinned MiniCPM5 revision")
    _require(_is_lower_hex(lock.get("config_sha256"), 64),
             "the tokenizer binding requires the sealed model config digest")
    files = _tokenizer_files(lock)
    return {
        "format": TOKENIZER_BINDING_FORMAT,
        "repo_id": lock["repo_id"],
        "revision": lock["revision"],
        "lock_sha256": _sha256_json(lock),
        "config_sha256": lock["config_sha256"],
        "snapshot_files_sha256_digest": _sha256_json(
            sorted(lock["snapshot_files_sha256"].items())),
        "tokenizer_files_sha256": dict(sorted(files.items())),
        "tokenizer_files_digest": fingerprint(files),
        "decoding": decode_policy(),
    }


def load_pinned_tokenizer(model_path: Path, lock: Mapping[str, Any]):
    """Load only the sealed fast tokenizer, offline; no model, weights or MPS."""
    binding = tokenizer_binding(lock)
    root = Path(model_path)
    _require(root.is_dir(), "the pinned tokenizer snapshot directory is missing")
    for name, digest in binding["tokenizer_files_sha256"].items():
        path = root / name
        _require(path.is_file(), f"the pinned tokenizer artifact is missing: {name}")
        _require(_sha256_file(path) == digest,
                 f"the pinned tokenizer artifact differs from the model lock: {name}")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        str(root), local_files_only=True, trust_remote_code=False)
    _require(getattr(tokenizer, "is_fast", False),
             "the pinned tokenizer must be a fast tokenizer")
    _require_fit_tokenizer(tokenizer)
    return tokenizer


def _require_fit_tokenizer(tokenizer) -> int:
    """The vocabulary size preserved token IDs must stay inside."""
    _require(callable(getattr(tokenizer, "decode", None)),
             "fit evidence requires a tokenizer with decode()")
    try:
        size = len(tokenizer)
    except TypeError as error:  # pragma: no cover - defensive contract
        raise RunnerError("fit evidence tokenizer lacks a vocabulary size") from error
    _require(type(size) is int and size > 0,
             "fit evidence tokenizer must declare a positive vocabulary size")
    return size


def decode_content(tokenizer, content_ids: Iterable[int]) -> str:
    """Decode one preserved content sequence under the pinned decode policy."""
    text = tokenizer.decode(
        list(content_ids), skip_special_tokens=DECODE_SKIP_SPECIAL_TOKENS,
        clean_up_tokenization_spaces=DECODE_CLEAN_UP_TOKENIZATION_SPACES)
    return text.strip() if DECODE_STRIP_WHITESPACE else text


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


def false_recall_v2(score: Mapping[str, Any],
                    taught_answers: Iterable[str]) -> float:
    """False recall under the shared v2 grammar used to score these rows.

    A control row is false recall when the output would be scored correct for any
    taught answer, including answers decorated by Markdown or terminal
    punctuation.  Failed generations carry no output and never count as recall.
    """
    answers = tuple(taught_answers)
    total = score["count"] or 1
    false = sum(1 for item in score["results"]
                if item.get("output") and any(
                    score_response(answer, item["output"]).correct
                    for answer in answers))
    return false / total


def attempt_counts(*, scoring_attempts: int, failures: int) -> dict:
    """Attempt accounting derived from the actual scored rows, never hardcoded."""
    _require(type(scoring_attempts) is int and scoring_attempts >= 0,
             "scoring attempts must be a nonnegative integer")
    _require(type(failures) is int and 0 <= failures <= scoring_attempts,
             "failed attempts cannot exceed the scoring attempts")
    return {"attempts": scoring_attempts - failures,
            "failed_attempts": failures,
            "scoring_attempts": scoring_attempts}


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
                        evaluation_end_to_end_seconds: float,
                        wrong_world_learning_update_seconds: float = 0.0,
                        wrong_world_teaching_end_to_end_seconds: float = 0.0) -> float:
    """Amortized duty over the complete declared teaching + evaluation workload.

    Missing or negative components fail closed; idle time is never added.
    """
    values = {"learning_update_seconds": learning_update_seconds,
              "teaching_end_to_end_seconds": teaching_end_to_end_seconds,
              "evaluation_end_to_end_seconds": evaluation_end_to_end_seconds,
              "wrong_world_learning_update_seconds": wrong_world_learning_update_seconds,
              "wrong_world_teaching_end_to_end_seconds": wrong_world_teaching_end_to_end_seconds}
    for name, value in values.items():
        _require(isinstance(value, (int, float)) and not isinstance(value, bool)
                 and float(value) >= 0.0,
                 f"duty-cycle component {name} is required and nonnegative")
    denominator = (float(teaching_end_to_end_seconds)
                   + float(wrong_world_teaching_end_to_end_seconds)
                   + float(evaluation_end_to_end_seconds))
    _require(denominator > 0.0, "duty-cycle denominator must be positive")
    return (float(learning_update_seconds)
            + float(wrong_world_learning_update_seconds)) / denominator


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
        "truncated": is_truncated_stop_reason(str(stop_reason)),
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
    sentence_stopped = False

    def stop_on_sentence(ids):
        nonlocal sentence_stopped
        text = decode_content(session.tokenizer, ids)
        if sentence_complete_text(text):
            sentence_stopped = True
            return True
        return False

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
                                   eos_token_id=tuple(stop_tokens),
                                   stop_predicate=stop_on_sentence)
            generated = tuple(int(token)
                              for token in ids[0, prompt_length:].tolist())
            _require(generated == tuple(int(token)
                                        for token in request.generated_ids),
                     "generated IDs differ from the request record")
            result = finalize_generation(
                generated,
                decode=lambda content: decode_content(session.tokenizer, content),
                stop_tokens=stop_tokens, max_new_tokens=max_new_tokens,
                completed_by="sentence_complete" if sentence_stopped else None)
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
                "answer_grammar_version": scored.grammar_version,
                "decode_policy_version": DECODE_POLICY_VERSION,
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
            record.update({"correct": False, "reason": "generation_failure",
                           "scoring_version": SCORING_VERSION,
                           "answer_grammar_version": ANSWER_GRAMMAR_VERSION,
                           "decode_policy_version": DECODE_POLICY_VERSION,
                           "truncated": None,
                           "error": f"{type(error).__name__}: {error}",
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

def response_teaching_work(*, prompt_positions: int, full_positions: int,
                           response_tokens: int, events: int,
                           steps_per_event: int) -> dict:
    """Count detached targets once and response-CE work per transaction step."""
    values = (prompt_positions, full_positions, response_tokens, events,
              steps_per_event)
    _require(all(type(value) is int and value >= 0 for value in values)
             and steps_per_event >= 1,
             "response teaching work requires nonnegative counts and positive steps")
    updates = events * steps_per_event
    return {
        "response_supervised_tokens": response_tokens * steps_per_event,
        # Detached prompt P + detached target F + steps * (read P + sense P + CE F).
        "processed_positions": ((1 + 2 * steps_per_event) * prompt_positions
                                + (1 + steps_per_event) * full_positions),
        "plastic_feature_prompt_positions": prompt_positions,
        "differentiable_read_prompt_positions": prompt_positions * steps_per_event,
        "session_sense_prompt_positions": prompt_positions * steps_per_event,
        "response_target_positions": full_positions,
        "response_ce_positions": full_positions * steps_per_event,
        "forward_calls": {
            "prompt_features": 2 * events,
            "differentiable_read": updates,
            "session_sense": updates,
            "response_ce": updates,
            "total": 2 * events + 3 * updates,
        },
        "backward_calls": updates,
        "response_ce_backward_calls": updates,
    }



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


def fit_candidate_answers_for(directory: Path, manifest: Mapping[str, Any], *,
                              world_uid: str,
                              scope: str) -> tuple[str, ...]:
    """Fit-only conflict vocabulary from authorized teaching/correction targets.

    Development, unaffected, sealed and other evaluation labels are never read
    here; :func:`load_rows` additionally refuses them until admission.
    """
    answers = set()
    for role in FIT_CANDIDATE_ROLES:
        if role not in manifest["role_files"]:
            continue
        answers.update(
            row["expected_response"]
            for row in load_rows(directory, manifest, role)
            if row["eligible"] and row["world_uid"] == world_uid
            and row["scope_uid"] == scope)
    _require(answers, "no authorized fit candidate answers are available")
    return tuple(sorted(answers))


def development_candidate_answers_for(directory: Path,
                                      manifest: Mapping[str, Any], *,
                                      roles: tuple[str, ...] =
                                      DEVELOPMENT_CANDIDATE_ROLES
                                      ) -> tuple[str, ...]:
    """Development-time conflict vocabulary, scoped after fit admission."""
    answers = set()
    for role in roles:
        if role not in manifest["role_files"]:
            continue
        answers.update(row["expected_response"]
                       for row in load_rows(directory, manifest, role))
    _require(answers, "no candidate answers are available")
    return tuple(sorted(answers))


# ---------------------------------------------------------------------------
# Fit receipt: dependent-development admission evidence
# ---------------------------------------------------------------------------

FIT_RECEIPT_KEYS = frozenset({
    "format", "schema_version", "status", "mode", "created_by", "world_uid",
    "scope", "config", "protocol", "episodes", "p_state", "tokenizer",
    "scoring", "coverage", "attempts", "teaching", "gate_report",
    "heldout_opened", "receipt_sha256"})
TEACH_RUN_KEYS = (
    "format", "mode", "status", "world_uid", "protocol_sha256",
    "config_sha256", "episodes_manifest_sha256", "implementation_sha256",
    "theta_stamp_unchanged", "updates", "learning_events", "expert_steps",
    "p_only_sha256", "p_only_manifest_sha256", "checkpoint_sha256",
    "checkpoint_bytes", "canonical_state_sha256", "cost_sha256",
    "checkpoint", "p_only")
TEACH_COST_KEYS = (
    "events", "optimizer_steps", "backward_calls", "wall_seconds",
    "learning_update_seconds", "teaching_end_to_end_seconds", "duty_cycle",
    "checkpoint_bytes", "p_only_bytes", "protocol_sha256", "config_sha256",
    "scoring_version")


def fit_receipt_required(config: Mapping[str, Any]) -> bool:
    """The behavioral training protocol depends on a passing fit evaluation."""
    return (config.get("format") == CONFIG_FORMAT_BEHAVIORAL
            and config.get("training_admitted") is True
            and bool(config.get("training", {}).get("fit_eval_required")))


def protocol_binding(protocol: TextProtocol) -> dict:
    """Every field needed to reconstruct the exact live text protocol."""
    return {
        "protocol_sha256": protocol.protocol_sha256,
        "world_uid": protocol.world_uid,
        "scope": protocol.scope,
        "theta": {"revision": protocol.theta_revision,
                  "lock_sha256": protocol.theta_lock_sha256,
                  "stamp_sha256": protocol.parameter_stamp_sha256},
        "theta_lock_sha256": protocol.theta_lock_sha256,
        "parameter_stamp_sha256": protocol.parameter_stamp_sha256,
        "tokenizer_sha256": protocol.tokenizer_sha256,
        "target_projection_sha256": protocol.target_projection_sha256,
        "read_path_sha256": protocol.read_path_sha256,
        "read_gain": protocol.read_gain,
        "feature_sites": list(protocol.feature_sites),
        "route": {"mode": protocol.route_mode,
                  "seed": protocol.route_seed,
                  "expert_count": protocol.expert_count},
        "bank": {"experts": protocol.expert_count,
                 "width": protocol.width,
                 "hidden_width": protocol.hidden_width,
                 "hidden_size": protocol.hidden_size},
    }


def p_only_binding(state: Mapping[str, Any]) -> dict:
    """The payload without weights: the immutable binding the receipt pins."""
    return {key: value for key, value in state.items() if key != "parameters"}


def p_only_binding_digest(state: Mapping[str, Any]) -> str:
    return _sha256_json(p_only_binding(state))


def p_state_summary(state_path: Path, state: Mapping[str, Any]) -> dict:
    payload = p_only_binding(state)
    summary = {"path": str(state_path), "sha256": _sha256_file(state_path),
               "bytes": Path(state_path).stat().st_size,
               "binding_sha256": _sha256_json(payload),
               "keys": sorted(state)}
    for key in ("format", "protocol_sha256", "scope", "theta", "bank", "route",
                "feature_producer", "read_path", "expert_steps"):
        _require(key in payload, f"P-only payload lacks {key}")
        summary[key] = payload[key]
    return summary


def _fit_cases(rows: Iterable[Mapping[str, Any]]) -> list[dict]:
    return [{"id": row["id"], "expected": row["expected_response"],
             "prompt_sha256": row["prompt_sha256"],
             "response_sha256": row["response_sha256"]} for row in rows]


def _case_digest(cases: Iterable[Mapping[str, Any]]) -> str:
    return _sha256_json([dict(case) for case in cases])


_FIT_STOP_REASONS = frozenset({"stop_token", "sentence_complete",
                               "token_budget", "missing_stop"})
_FIT_EVIDENCE_FIELDS = ("output", "generated_ids", "content_ids",
                        "stop_reason", "stop_token", "max_new_tokens")


def _fit_int_ids(value: Any, *, field: str, attempt: str,
                 vocab_size: int) -> tuple[int, ...]:
    _require(isinstance(value, (list, tuple))
             and all(type(item) is int for item in value),
             f"fit attempt {attempt!r} has an invalid {field} sequence")
    _require(all(0 <= item < vocab_size for item in value),
             f"fit attempt {attempt!r} has an out-of-vocabulary {field} token ID")
    return tuple(value)


def _is_fit_generation_failure(item: Mapping[str, Any]) -> bool:
    """Recognize only explicit, evidence-free generation failures."""
    flagged = "error" in item or item.get("reason") == "generation_failure"
    if not flagged:
        return False
    attempt = item.get("id")
    _require(item.get("reason") == "generation_failure",
             f"fit attempt {attempt!r} carries an error without a generation failure")
    _require(isinstance(item.get("error"), str) and item["error"].strip(),
             f"fit attempt {attempt!r} lacks its generation error")
    _require(item.get("correct") is False and item.get("truncated") is None,
             f"fit attempt {attempt!r} generation failure is scored")
    _require(not any(field in item for field in _FIT_EVIDENCE_FIELDS),
             f"fit attempt {attempt!r} generation failure carries scored evidence")
    _require(item.get("scoring_version") == SCORING_VERSION
             and item.get("answer_grammar_version") == ANSWER_GRAMMAR_VERSION
             and item.get("decode_policy_version") == DECODE_POLICY_VERSION,
             f"fit attempt {attempt!r} failure lacks the declared "
             "scorer/grammar/decoder")
    return True


def _recompute_fit_attempt(item: Mapping[str, Any], case: Mapping[str, Any], *,
                           candidates: tuple[str, ...],
                           max_new_tokens: int,
                           stop_tokens: frozenset[int],
                           tokenizer) -> dict:
    """Decode one preserved generation with the pinned tokenizer and rescore it."""
    attempt = case["id"]
    vocab_size = _require_fit_tokenizer(tokenizer)
    _require(item.get("label") == "fit",
             f"fit attempt {attempt!r} belongs to another evaluation label")
    _require(item.get("expected") == case["expected"]
             and item.get("prompt_sha256") == case["prompt_sha256"],
             f"fit attempt {attempt!r} differs from its authoritative fit target")
    if _is_fit_generation_failure(item):
        return {"failed": True, "correct": False, "truncated": False,
                "stop_reason": None, "generated": 0}
    _require(item.get("scoring_version") == SCORING_VERSION,
             f"fit attempt {attempt!r} was scored by another scorer")
    _require(item.get("answer_grammar_version") == ANSWER_GRAMMAR_VERSION,
             f"fit attempt {attempt!r} lacks the current answer grammar")
    _require(item.get("decode_policy_version") == DECODE_POLICY_VERSION,
             f"fit attempt {attempt!r} was decoded under another policy")
    output = item.get("output")
    _require(isinstance(output, str), f"fit attempt {attempt!r} lacks decoded output")
    generated = _fit_int_ids(item.get("generated_ids"),
                             field="generated_ids", attempt=attempt,
                             vocab_size=vocab_size)
    content = _fit_int_ids(item.get("content_ids"),
                           field="content_ids", attempt=attempt,
                           vocab_size=vocab_size)
    _require(generated and len(generated) <= max_new_tokens,
             f"fit attempt {attempt!r} generation violates the token budget")
    _require(content == generated[:len(content)],
             f"fit attempt {attempt!r} content is not the generated prefix")
    declared_budget = item.get("max_new_tokens")
    _require(type(declared_budget) is int and declared_budget == max_new_tokens,
             f"fit attempt {attempt!r} records another generation budget")
    stop_reason = item.get("stop_reason")
    _require(stop_reason in _FIT_STOP_REASONS,
             f"fit attempt {attempt!r} records an unknown stop reason")
    stop_token = item.get("stop_token")
    if stop_reason == "stop_token":
        _require(type(stop_token) is int and stop_token in stop_tokens,
                 f"fit attempt {attempt!r} stop token is not declared")
        _require(generated[-1] == stop_token
                 and len(content) == len(generated) - 1,
                 f"fit attempt {attempt!r} stop token disagrees with its generation")
    else:
        _require(stop_token is None and content == generated,
                 f"fit attempt {attempt!r} stop disposition disagrees with its generation")
        _require(not any(token in stop_tokens for token in generated),
                 f"fit attempt {attempt!r} hides a stop token in a non-token stop")
        if stop_reason == "sentence_complete":
            _require(sentence_complete_text(output),
                     f"fit attempt {attempt!r} claims an incomplete sentence stop")
        elif stop_reason == "token_budget":
            _require(len(generated) == max_new_tokens,
                     f"fit attempt {attempt!r} claims a token-budget stop it did not reach")
        else:
            _require(len(generated) < max_new_tokens,
                     f"fit attempt {attempt!r} claims a missing stop after budget exhaustion")
    _require(not any(token in stop_tokens for token in content),
             f"fit attempt {attempt!r} preserves a stop token inside its content")
    if not content:
        _require(output == "",
                 f"fit attempt {attempt!r} saves text with no content IDs")
    decoded = decode_content(tokenizer, content)
    _require(decoded == output,
             f"fit attempt {attempt!r} decoded IDs contradict its saved output")
    truncated = is_truncated_stop_reason(stop_reason)
    _require(item.get("truncated") is truncated,
             f"fit attempt {attempt!r} truncation disagrees with its stop reason")
    scored = score_response(case["expected"], output,
                            candidate_answers=candidates, truncated=truncated)
    _require(item.get("reason") == scored.reason
             and item.get("correct") is scored.correct,
             f"fit attempt {attempt!r} saved score contradicts the preserved evidence")
    _require(scored.scoring_version == SCORING_VERSION
             and scored.grammar_version == ANSWER_GRAMMAR_VERSION,
             f"fit attempt {attempt!r} was rescored under another contract")
    return {"failed": False, "correct": scored.correct,
            "truncated": truncated, "stop_reason": stop_reason,
            "generated": len(generated)}


def recompute_fit_evaluation(cases: list[dict], score: Mapping[str, Any], *,
                             declared_attempts: int, min_accuracy: float,
                             max_new_tokens: int,
                             stop_tokens: Iterable[int],
                             candidates: tuple[str, ...],
                             tokenizer) -> dict:
    """Recompute the whole fit gate from targets and preserved evidence.

    Saved ``correct`` flags, gate pass flags and gate accuracy are only compared
    against this independent result; none of them is trusted for admission.
    """
    _require(len(cases) == int(declared_attempts),
             "fit case coverage does not match the declared fit workload")
    _require(len({case["id"] for case in cases}) == len(cases),
             "fit workload cases are not unique")
    results = score.get("results")
    _require(isinstance(results, list), "fit gate report lacks per-attempt scores")
    results = [item for item in results if isinstance(item, Mapping)]
    ids = [item.get("id") for item in results]
    expected_ids = [case["id"] for case in cases]
    duplicates = sorted({value for value in ids
                         if value is not None and ids.count(value) > 1})
    missing = [value for value in expected_ids if value not in ids]
    unexpected = [value for value in ids if value not in expected_ids]
    _require(not duplicates and not missing and not unexpected,
             "fit gate case coverage is incomplete, duplicated or unexpected")
    _require(len(results) == len(expected_ids),
             "fit gate attempt count differs from its unique fit cases")
    by_id: dict[Any, Mapping[str, Any]] = {}
    for item in results:
        _require(item.get("id") not in by_id,
                 "fit gate repeats an authoritative fit case")
        by_id[item.get("id")] = item
    stop_set = frozenset(int(token) for token in stop_tokens)
    recomputed = [
        _recompute_fit_attempt(by_id[case["id"]], case,
                               candidates=candidates,
                               max_new_tokens=int(max_new_tokens),
                               stop_tokens=stop_set, tokenizer=tokenizer)
        for case in cases]
    audits = [item.get("request_audit") for item in results]
    audit_complete = bool(results) and all(
        isinstance(audit, Mapping)
        and all(field in audit for field in _AUDIT_FIELDS)
        for audit in audits)
    source_free = audit_complete and all(
        audit["sources_present"] is False
        and audit["memory_spans"] == 0
        and audit["selection_records"] == 0
        and audit["ledger_records"] == 0
        and audit["controller_constructed"] is False
        and audit["coordinator_constructed"] is False
        and audit["updates_enabled"] is False
        for audit in audits)
    failures = sum(1 for row in recomputed if row["failed"])
    correct = sum(1 for row in recomputed if row["correct"])
    accuracy = correct / len(recomputed)
    stop_reasons: dict[str, int] = {}
    for row in recomputed:
        reason = row["stop_reason"]
        if isinstance(reason, str):
            stop_reasons[reason] = stop_reasons.get(reason, 0) + 1
    gate_failures = []
    if accuracy < float(min_accuracy):
        gate_failures.append("min_fit_accuracy")
    if failures:
        gate_failures.append("generation_failures")
    if not audit_complete:
        gate_failures.append("incomplete_request_audit")
    coverage = {
        "role": "fit",
        "declared_attempts": int(declared_attempts),
        "measured_attempts": len(results),
        "unique_ids": len(set(ids)),
        "ids": ids,
        "cases_sha256": _case_digest(cases),
        "missing_ids": missing,
        "unexpected_ids": unexpected,
        "duplicate_ids": duplicates,
        "failed_attempts": failures,
        "source_free": source_free,
        "audit_complete": audit_complete,
    }
    attempts = {
        "scoring_attempts": len(results),
        "correct": correct,
        "accuracy": accuracy,
        "failures": failures,
        "truncated": sum(1 for row in recomputed if row["truncated"]),
        "stop_reasons": stop_reasons,
        "generated_tokens": sum(row["generated"] for row in recomputed),
        "min_accuracy": float(min_accuracy),
        "all_declared_gates_passed": not gate_failures,
        "gate_failures": gate_failures,
    }
    _require(score.get("label") == "fit"
             and int(score.get("count", -1)) == len(results)
             and int(score.get("correct", -1)) == correct
             and int(score.get("failures", -1)) == failures
             and math.isclose(float(score.get("accuracy", -1.0)), accuracy,
                              rel_tol=1e-12, abs_tol=1e-12),
             "fit gate score aggregates differ from independently recomputed evidence")
    passed = (not gate_failures
              and failures == 0
              and correct > 0
              and coverage["audit_complete"]
              and coverage["source_free"]
              and not missing and not unexpected and not duplicates
              and coverage["measured_attempts"] == int(declared_attempts)
              and coverage["unique_ids"] == int(declared_attempts))
    return {"coverage": coverage, "attempts": attempts,
            "gate_failures": gate_failures, "passed": passed}


def teaching_bundle(directory: Path, *, expected_updates: int | None = None) -> dict:
    """Read and cross-check one preserved teaching run before it is trusted."""
    directory = Path(directory)
    run_path = directory / "run.json"
    cost_path = directory / "cost.json"
    p_only_manifest_path = directory / "p_only" / "manifest.json"
    p_only_path = directory / "p_only" / "state.pt"
    checkpoint_path = directory / "full_checkpoint.pt"
    final_state_path = directory / "final_state.json"
    for path in (run_path, cost_path, p_only_manifest_path, p_only_path,
                 checkpoint_path, final_state_path):
        _require(path.is_file(), f"teaching artifact is missing: {path}")
    run = json.loads(run_path.read_text(encoding="utf-8"))
    cost = json.loads(cost_path.read_text(encoding="utf-8"))
    p_manifest = json.loads(p_only_manifest_path.read_text(encoding="utf-8"))
    final_state = json.loads(final_state_path.read_text(encoding="utf-8"))
    for key in TEACH_RUN_KEYS:
        _require(key in run,
                 f"teaching run.json lacks {key}; rerun teaching with this runner")
    _require(run["format"] == PROMPT_FORM and run["mode"] == "teach"
             and run["status"] == "completed",
             "teaching run.json is not a completed teach run")
    _require(run["theta_stamp_unchanged"] is True,
             "teaching run changed the native theta parameter stamp")
    _require(isinstance(run["implementation_sha256"], Mapping),
             "teaching run lacks implementation pins")
    for key in TEACH_COST_KEYS:
        _require(key in cost, f"teaching cost.json lacks {key}")
    _require(isinstance(p_manifest, Mapping)
             and p_manifest.get("format") == P_ONLY_FORMAT,
             "teaching P-only manifest is invalid")
    _require(p_manifest.get("forbidden_absent") is True
             and p_manifest.get("source_inputs") == 0,
             "teaching P-only export carries learning state or sources")
    _require(isinstance(final_state, Mapping)
             and isinstance(final_state.get("sha256"), str)
             and len(final_state["sha256"]) == 64,
             "teaching canonical state is invalid")
    p_only_sha = _sha256_file(p_only_path)
    checkpoint_sha = _sha256_file(checkpoint_path)
    _require(p_manifest["sha256"] == p_only_sha
             and p_manifest["bytes"] == p_only_path.stat().st_size,
             "teaching P-only manifest differs from the exported state")
    _require(run["p_only_sha256"] == p_only_sha
             and run["p_only_manifest_sha256"] == _sha256_file(p_only_manifest_path),
             "teaching run.json P-only hashes differ from the export")
    _require(run["checkpoint_sha256"] == checkpoint_sha
             and int(run["checkpoint_bytes"]) == checkpoint_path.stat().st_size,
             "teaching run.json checkpoint hashes differ from the checkpoint")
    _require(run["canonical_state_sha256"] == final_state["sha256"],
             "teaching run.json canonical state differs from final_state.json")
    _require(cost["protocol_sha256"] == run["protocol_sha256"],
             "teaching cost.json belongs to another protocol")
    _require(run["cost_sha256"] == _sha256_file(cost_path),
             "teaching run.json cost_sha256 differs from cost.json")
    _require(cost["config_sha256"] == run["config_sha256"],
             "teaching cost.json belongs to another config")
    _require(int(cost["checkpoint_bytes"]) == checkpoint_path.stat().st_size
             and int(cost["p_only_bytes"]) == p_only_path.stat().st_size,
             "teaching cost.json bytes differ from the exported artifacts")
    updates = int(run["updates"])
    _require(updates >= 1 and int(run["learning_events"]) == updates,
             "teaching run.json event accounting is invalid")
    _require(int(cost["events"]) == updates,
             "teaching cost.json events differ from run.json updates")
    _require(int(cost["optimizer_steps"]) == updates
             and int(cost["backward_calls"]) == updates,
             "teaching optimizer/backward accounting differs from its updates")
    expert_steps = [int(value) for value in run["expert_steps"]]
    _require(expert_steps and sum(expert_steps) == updates,
             "teaching expert counters do not cover the declared updates")
    if expected_updates is not None:
        _require(updates == int(expected_updates),
                 f"teaching run covers {updates} events; "
                 f"the workload declares {expected_updates}")
    wall = float(cost["wall_seconds"])
    _require(wall > 0, "teaching wall time must be positive")
    learning = float(cost["learning_update_seconds"])
    end_to_end = float(cost["teaching_end_to_end_seconds"])
    _require(learning >= 0.0 and end_to_end >= learning,
             "teaching cost timing is inconsistent")
    _require(math.isclose(float(cost["duty_cycle"]), learning / wall,
                          rel_tol=1e-9, abs_tol=1e-12),
             "teaching cost duty cycle does not match its timings")
    return {
        "run_path": str(run_path), "run_sha256": _sha256_file(run_path),
        "cost_path": str(cost_path), "cost_sha256": _sha256_file(cost_path),
        "p_only_manifest_path": str(p_only_manifest_path),
        "p_only_manifest_sha256": _sha256_file(p_only_manifest_path),
        "p_only_path": str(p_only_path), "p_only_sha256": p_only_sha,
        "p_only_bytes": p_only_path.stat().st_size,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_bytes": checkpoint_path.stat().st_size,
        "final_state_path": str(final_state_path),
        "final_state_sha256": _sha256_file(final_state_path),
        "canonical_state_sha256": final_state["sha256"],
        "protocol_sha256": run["protocol_sha256"],
        "config_sha256": run["config_sha256"],
        "episodes_manifest_sha256": run["episodes_manifest_sha256"],
        "world_uid": run["world_uid"],
        "updates": updates,
        "learning_events": int(run["learning_events"]),
        "expert_steps": expert_steps,
        "events": int(cost["events"]),
        "optimizer_steps": int(cost["optimizer_steps"]),
        "backward_calls": int(cost["backward_calls"]),
        "learning_update_seconds": learning,
        "teaching_end_to_end_seconds": end_to_end,
        "wall_seconds": wall,
        "duty_cycle": float(cost["duty_cycle"]),
        "theta_stamp_unchanged": True,
    }


def seal_fit_receipt(receipt: Mapping[str, Any]) -> dict:
    """Attach the digest over every field except ``receipt_sha256``."""
    body = {key: value for key, value in receipt.items()
            if key != "receipt_sha256"}
    sealed = dict(receipt)
    sealed["receipt_sha256"] = _sha256_json(body)
    return sealed


def load_fit_receipt(path: Path) -> dict:
    path = Path(path)
    _require(path.is_file(), f"fit receipt is missing: {path}")
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise RunnerError(f"fit receipt is unreadable: {error}") from error
    _require(isinstance(receipt, Mapping), "fit receipt must be a JSON object")
    return dict(receipt)


def load_model_lock(path: Path) -> dict:
    path = Path(path)
    _require(path.is_file(), f"model lock is missing: {path}")
    try:
        lock = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise RunnerError(f"model lock is unreadable: {error}") from error
    _require(isinstance(lock, Mapping)
             and isinstance(lock.get("snapshot_files_sha256"), Mapping),
             "model lock lacks the snapshot digests")
    return dict(lock)


def build_fit_receipt(*, config: Mapping[str, Any], config_path: Path,
                      manifest: Mapping[str, Any], manifest_path: Path,
                      episodes_root: Path, state_path: Path,
                      state: Mapping[str, Any], protocol: TextProtocol,
                      teach_dir: Path, teach_expected_updates: int,
                      fit_gate_path: Path, lock: Mapping[str, Any],
                      tokenizer,
                      wrong_world_teach_dir: Path | None = None) -> dict:
    """Bind one fit evaluation to its P-state, protocol, data, scorer and teach run."""
    gate = json.loads(Path(fit_gate_path).read_text(encoding="utf-8"))
    _require(gate.get("label") == "fit", "fit gate report has another label")
    score = gate.get("scores")
    _require(isinstance(score, Mapping)
             and isinstance(score.get("results"), list),
             "fit gate report lacks per-attempt scores")
    tokenizer_block = tokenizer_binding(lock)
    _require(tokenizer_block["revision"] == str(config["theta"]["revision"]),
             "the model lock revision differs from the configured theta revision")
    _require(tokenizer_block["tokenizer_files_digest"]
             == protocol.tokenizer_sha256,
             "the model lock tokenizer files differ from the fit protocol")
    primary = manifest["primary_world"]
    scope = f"{primary}/text"
    declared = int(config["workload"]["fit_eval_attempts"])
    rows = teaching_rows(episodes_root, manifest, primary, scope, limit=declared)
    cases = _fit_cases(rows)
    generation = config["generation"]
    candidates = fit_candidate_answers_for(
        episodes_root, manifest, world_uid=primary, scope=scope)
    recomputed = recompute_fit_evaluation(
        cases, score, declared_attempts=declared,
        min_accuracy=float(config["gates"]["fit"]["min_accuracy"]),
        max_new_tokens=int(generation["max_new_tokens"]),
        stop_tokens=generation["stop_tokens"], candidates=candidates,
        tokenizer=tokenizer)
    coverage = recomputed["coverage"]
    attempts = recomputed["attempts"]
    gate_failures = [str(value) for value in gate.get("gate_failures") or ()]
    _require(gate.get("scoring_version") == SCORING_VERSION
             and gate.get("stop_policy_version") == STOP_POLICY_VERSION
             and gate.get("answer_grammar_version") == ANSWER_GRAMMAR_VERSION
             and gate.get("decode_policy_version") == DECODE_POLICY_VERSION,
             "fit gate report lacks the declared "
             "scorer/grammar/stopping/decoder metadata")
    _require(gate_failures == recomputed["gate_failures"],
             "fit gate failures differ from independently recomputed evidence")
    _require((gate.get("all_declared_gates_passed") is True)
             == recomputed["passed"],
             "fit gate pass flag differs from independently recomputed evidence")
    _require(math.isclose(float(gate.get("fit_accuracy", -1.0)),
                          attempts["accuracy"], rel_tol=1e-12, abs_tol=1e-12),
             "fit gate accuracy differs from independently recomputed evidence")
    teaching = teaching_bundle(teach_dir, expected_updates=teach_expected_updates)
    _require(teaching["world_uid"] == primary,
             "teaching run belongs to another world")
    teaching_block = {"primary": teaching}
    if wrong_world_teach_dir is not None:
        wrong = teaching_bundle(
            Path(wrong_world_teach_dir),
            expected_updates=int(config["workload"]["wrong_world_teaching_events"]))
        _require(wrong["world_uid"] == manifest["wrong_world"],
                 "wrong-world teaching run belongs to another world")
        teaching_block["wrong_world"] = wrong
    _require(teaching["p_only_sha256"] == _sha256_file(state_path),
             "fit evaluation P-state is not the teaching run export")
    _require(teaching["protocol_sha256"] == protocol.protocol_sha256,
             "teaching run protocol differs from the fit protocol")
    _require(teaching["config_sha256"] == _sha256_file(config_path),
             "teaching run config differs from the fit config")
    _require(teaching["episodes_manifest_sha256"]
             == _sha256_file(manifest_path),
             "teaching run episodes differ from the fit episodes")
    passed = recomputed["passed"]
    receipt = {
        "format": FIT_RECEIPT_FORMAT,
        "schema_version": FIT_RECEIPT_SCHEMA_VERSION,
        "status": "passed" if passed else "failed",
        "mode": "fit-eval",
        "created_by": "scripts/run_nested_text.py",
        "world_uid": primary,
        "scope": scope,
        "config": {
            "path": str(config_path),
            "sha256": _sha256_file(config_path),
            "format": config.get("format"),
            "training_admitted": config.get("training_admitted") is True,
            "objective": config["transaction"].get("objective"),
            "steps_per_event": int(config["transaction"]["steps_per_event"]),
        },
        "protocol": protocol_binding(protocol),
        "episodes": {
            "root": str(episodes_root),
            "manifest_path": str(manifest_path),
            "format": manifest.get("format"),
            "schema_version": manifest.get("schema_version"),
            "manifest_sha256": _sha256_file(manifest_path),
            "files_sha256_digest": _sha256_json(
                sorted(manifest["files_sha256"].items())),
            "role_files_digest": _sha256_json(
                sorted(manifest["role_files"].items())),
            "primary_world": manifest["primary_world"],
            "wrong_world": manifest["wrong_world"],
        },
        "p_state": p_state_summary(state_path, state),
        "tokenizer": tokenizer_block,
        "scoring": {
            "scoring_version": SCORING_VERSION,
            "answer_grammar_version": ANSWER_GRAMMAR_VERSION,
            "stop_policy_version": STOP_POLICY_VERSION,
            "max_new_tokens": int(generation["max_new_tokens"]),
            "stop_tokens": [int(token) for token in generation["stop_tokens"]],
            "sentence_stop": generation["sentence_stop"],
            "target_eos_token": generation.get("target_eos_token"),
            "token_budget_credit": generation.get("token_budget_credit"),
            "empty_generation_credit": generation.get("empty_generation_credit"),
            "candidate_answers_sha256": _sha256_json(list(candidates)),
        },
        "coverage": coverage,
        "attempts": attempts,
        "teaching": teaching_block,
        "gate_report": {
            "path": str(fit_gate_path),
            "sha256": _sha256_file(fit_gate_path),
            "all_declared_gates_passed": recomputed["passed"],
            "gate_failures": gate_failures,
        },
        "heldout_opened": False,
    }
    return seal_fit_receipt(receipt)


def verify_fit_receipt_payload(receipt: Mapping[str, Any], *,
                               state: Mapping[str, Any]) -> None:
    """Compare the loaded P-only payload with the receipt's immutable binding."""
    _require(isinstance(state, Mapping) and set(state) == P_ONLY_KEYS,
             "fit receipt P-state lacks the declared P-only keys")
    block = receipt["p_state"]
    _require(p_only_binding_digest(state) == block["binding_sha256"],
             "fit receipt P-state payload binding differs")
    for key in ("format", "protocol_sha256", "scope", "theta", "bank", "route",
                "feature_producer", "read_path", "expert_steps"):
        _require(state.get(key) == block.get(key),
                 f"fit receipt P-state field {key} differs from the loaded payload")


def verify_fit_receipt_protocol(receipt: Mapping[str, Any], *,
                                protocol: TextProtocol) -> None:
    """Bind the receipt to the live protocol built from the loaded model."""
    _require(protocol_binding(protocol) == receipt["protocol"],
             "fit receipt protocol differs from the live text protocol")
    _require(receipt["p_state"]["protocol_sha256"] == protocol.protocol_sha256,
             "fit receipt P-state belongs to another protocol")


def validate_fit_receipt(receipt: Mapping[str, Any], *,
                         config: Mapping[str, Any], config_path: Path,
                         manifest: Mapping[str, Any], manifest_path: Path,
                         episodes_root: Path, state_path: Path,
                         lock: Mapping[str, Any],
                         state: Mapping[str, Any] | None = None,
                         tokenizer=None, model_path: Path | None = None) -> None:
    """Fail closed unless this receipt admits the live dependent evaluation.

    Everything except the payload digest is checked before ``torch.load``; the
    payload binding is checked immediately after the state is loaded and before
    the native worker or any development-role access.  Preserved fit IDs are
    decoded with the sealed fast tokenizer only: when no live tokenizer object
    is supplied, the tokenizer is loaded offline from ``model_path``; no model
    weights and no MPS device are touched.
    """
    _require(isinstance(receipt, Mapping), "fit receipt must be a mapping")
    _require(set(receipt) == FIT_RECEIPT_KEYS,
             "fit receipt fields differ from the declared schema")
    _require(receipt.get("format") == FIT_RECEIPT_FORMAT,
             "fit receipt format differs")
    _require(receipt.get("schema_version") == FIT_RECEIPT_SCHEMA_VERSION,
             "fit receipt schema version differs")
    _require(receipt.get("created_by") == "scripts/run_nested_text.py",
             "fit receipt was not produced by this runner")
    _require(seal_fit_receipt(receipt)["receipt_sha256"]
             == receipt.get("receipt_sha256"),
             "fit receipt digest differs")
    _require(receipt.get("mode") == "fit-eval", "fit receipt mode differs")
    _require(receipt.get("status") == "passed",
             f"fit receipt status is {receipt.get('status')!r}; "
             "development evaluation requires a passing fit")
    _require(receipt.get("heldout_opened") is False,
             "fit receipt must not have opened the heldout role")

    config_block = receipt["config"]
    _require(isinstance(config_block, Mapping), "fit receipt config block is invalid")
    _require(Path(config_block["path"]).resolve() == Path(config_path).resolve(),
             "fit receipt belongs to another config file")
    _require(_sha256_file(config_path) == config_block["sha256"],
             "fit receipt config digest differs")
    _require(config_block["format"] == config.get("format")
             and config_block["training_admitted"] is True,
             "fit receipt config is not the admitted behavioral protocol")
    _require(config_block["objective"]
             == config["transaction"].get("objective"),
             "fit receipt objective differs from the live config")
    _require(int(config_block["steps_per_event"])
             == int(config["transaction"]["steps_per_event"]),
             "fit receipt step policy differs from the live config")

    episodes = receipt["episodes"]
    _require(isinstance(episodes, Mapping), "fit receipt episode block is invalid")
    _require(Path(episodes["root"]).resolve() == Path(episodes_root).resolve(),
             "fit receipt belongs to another episode root")
    _require(Path(episodes["manifest_path"]).resolve()
             == Path(manifest_path).resolve(),
             "fit receipt belongs to another episode manifest")
    _require(_sha256_file(manifest_path) == episodes["manifest_sha256"],
             "fit receipt episode manifest digest differs")
    _require(episodes["format"] == manifest.get("format")
             and episodes["schema_version"] == manifest.get("schema_version"),
             "fit receipt episode manifest format differs")
    _require(episodes["files_sha256_digest"]
             == _sha256_json(sorted(manifest["files_sha256"].items()))
             and episodes["role_files_digest"]
             == _sha256_json(sorted(manifest["role_files"].items())),
             "fit receipt episode file digests differ")
    _require(episodes["primary_world"] == manifest["primary_world"]
             and episodes["wrong_world"] == manifest["wrong_world"],
             "fit receipt belongs to another episode world pair")

    protocol_block = receipt["protocol"]
    _require(isinstance(protocol_block, Mapping)
             and isinstance(protocol_block.get("bank"), Mapping),
             "fit receipt protocol block is invalid")
    _require(isinstance(lock, Mapping)
             and isinstance(lock.get("snapshot_files_sha256"), Mapping),
             "fit receipt validation requires the model lock")
    hidden_size = int(protocol_block["bank"]["hidden_size"])
    live = protocol_for_world(config, lock, hidden_size=hidden_size,
                              world_uid=manifest["primary_world"])
    _require(protocol_binding(live) == protocol_block,
             "fit receipt protocol is stale or differs from the live config/lock")
    _require(receipt["world_uid"] == manifest["primary_world"]
             and receipt["scope"] == f"{manifest['primary_world']}/text",
             "fit receipt world/scope differs from the episodes")

    tokenizer_block = receipt["tokenizer"]
    _require(isinstance(tokenizer_block, Mapping),
             "fit receipt tokenizer block is invalid")
    _require(tokenizer_block == tokenizer_binding(lock),
             "fit receipt tokenizer identity or decode policy differs "
             "from the model lock")
    _require(tokenizer_block["revision"] == protocol_block["theta"]["revision"]
             and tokenizer_block["tokenizer_files_digest"]
             == protocol_block["tokenizer_sha256"],
             "fit receipt tokenizer differs from its protocol")
    _require(tokenizer_block["decoding"]["version"] == DECODE_POLICY_VERSION
             and tokenizer_block["decoding"]["stop_policy_version"]
             == STOP_POLICY_VERSION,
             "fit receipt decode policy differs from the declared contract")

    p_state = receipt["p_state"]
    _require(isinstance(p_state, Mapping), "fit receipt P-state block is invalid")
    _require(Path(p_state["path"]).resolve() == Path(state_path).resolve(),
             "fit receipt binds another P-state file")
    _require(_sha256_file(state_path) == p_state["sha256"],
             "fit receipt P-state digest differs")
    _require(Path(state_path).stat().st_size == int(p_state["bytes"]),
             "fit receipt P-state byte count differs")
    _require(p_state["format"] == P_ONLY_FORMAT
             and p_state["protocol_sha256"] == protocol_block["protocol_sha256"]
             and p_state["scope"] == protocol_block["scope"],
             "fit receipt P-state binding differs from its protocol")
    _require(list(p_state["keys"]) == sorted(P_ONLY_KEYS),
             "fit receipt P-state key inventory differs")
    declared_binding = {key: p_state[key] for key in P_ONLY_KEYS
                        if key != "parameters"}
    _require(_sha256_json(declared_binding) == p_state["binding_sha256"],
             "fit receipt P-state binding digest differs")
    _require(p_state["theta"] == protocol_block["theta"],
             "fit receipt P-state theta binding differs")
    _require(p_state["bank"].get("experts") == protocol_block["bank"]["experts"]
             and p_state["bank"].get("width") == protocol_block["bank"]["width"]
             and p_state["bank"].get("hidden_width")
             == protocol_block["bank"]["hidden_width"]
             and p_state["bank"].get("seed") == int(config["bank"]["seed"]),
             "fit receipt P-state bank binding differs")
    _require(p_state["route"].get("mode") == protocol_block["route"]["mode"]
             and p_state["route"].get("seed") == protocol_block["route"]["seed"]
             and p_state["route"].get("expert_count")
             == protocol_block["route"]["expert_count"],
             "fit receipt P-state route binding differs")
    feature = p_state["feature_producer"]
    _require(list(feature.get("sites", ())) == protocol_block["feature_sites"]
             and feature.get("width") == protocol_block["bank"]["width"]
             and feature.get("hidden_size") == protocol_block["bank"]["hidden_size"]
             and feature.get("target_projection_seed")
             == int(config["features"]["target_projection_seed"])
             and feature.get("target_projection_sha256")
             == protocol_block["target_projection_sha256"],
             "fit receipt P-state feature producer differs")
    read = p_state["read_path"]
    _require(read.get("decoder_sha256") == protocol_block["read_path_sha256"]
             and float(read.get("gain", -1)) == protocol_block["read_gain"],
             "fit receipt P-state read path differs")

    scoring = receipt["scoring"]
    generation = config["generation"]
    _require(isinstance(scoring, Mapping), "fit receipt scoring block is invalid")
    _require(scoring["scoring_version"] == SCORING_VERSION
             and scoring["scoring_version"] == generation.get("scoring_version"),
             "fit receipt scorer version differs from the live scorer/config")
    _require(scoring["answer_grammar_version"] == ANSWER_GRAMMAR_VERSION,
             "fit receipt answer grammar differs from this scorer")
    _require(scoring["stop_policy_version"] == STOP_POLICY_VERSION
             and scoring["stop_policy_version"]
             == generation["sentence_stop"]["version"],
             "fit receipt stop policy differs from the live stopping policy")
    _require(int(scoring["max_new_tokens"])
             == int(generation["max_new_tokens"]),
             "fit receipt generation budget differs")
    _require([int(token) for token in scoring["stop_tokens"]]
             == [int(token) for token in generation["stop_tokens"]]
             == list(STOP_TOKENS),
             "fit receipt stop tokens differ")
    _require(scoring["sentence_stop"] == generation["sentence_stop"],
             "fit receipt sentence-stop policy differs")
    _require(scoring["target_eos_token"] == generation.get("target_eos_token")
             and scoring["token_budget_credit"] is False
             and scoring["empty_generation_credit"] is False,
             "fit receipt credits empty or budget-truncated generations")
    _require(scoring["candidate_answers_sha256"]
             == _sha256_json(list(fit_candidate_answers_for(
                 episodes_root, manifest,
                 world_uid=manifest["primary_world"],
                 scope=f"{manifest['primary_world']}/text"))),
             "fit receipt candidate answers differ")

    gate_block = receipt["gate_report"]
    _require(isinstance(gate_block, Mapping), "fit receipt gate block is invalid")
    gate_path = Path(gate_block["path"])
    _require(gate_path.is_file(), f"fit gate report is missing: {gate_path}")
    _require(_sha256_file(gate_path) == gate_block["sha256"],
             "fit receipt gate report digest differs")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    _require(gate.get("label") == "fit"
             and gate.get("all_declared_gates_passed") is True
             and gate_block["all_declared_gates_passed"] is True,
             "fit receipt records a failed fit gate")
    _require(gate.get("scoring_version") == SCORING_VERSION
             and gate.get("answer_grammar_version") == ANSWER_GRAMMAR_VERSION
             and gate.get("stop_policy_version") == STOP_POLICY_VERSION
             and gate.get("decode_policy_version") == DECODE_POLICY_VERSION
             and gate.get("config_sha256") == config_block["sha256"],
             "fit gate report was produced by another scorer, grammar, "
             "stopping policy, decoder or config")
    _require(list(gate.get("gate_failures") or ()) == []
             and list(gate_block["gate_failures"]) == [],
             "fit receipt records fit gate failures")
    score = gate.get("scores")
    _require(isinstance(score, Mapping)
             and isinstance(score.get("results"), list),
             "fit gate report lacks per-attempt scores")
    declared = int(config["workload"]["fit_eval_attempts"])
    rows = teaching_rows(episodes_root, manifest, manifest["primary_world"],
                         f"{manifest['primary_world']}/text", limit=declared)
    cases = _fit_cases(rows)
    candidates = fit_candidate_answers_for(
        episodes_root, manifest, world_uid=manifest["primary_world"],
        scope=f"{manifest['primary_world']}/text")
    decoder = tokenizer
    if decoder is None:
        _require(model_path is not None,
                 "fit receipt validation requires the pinned tokenizer "
                 "snapshot or a loaded tokenizer")
        decoder = load_pinned_tokenizer(Path(model_path), lock)
    recomputed = recompute_fit_evaluation(
        cases, score, declared_attempts=declared,
        min_accuracy=float(config["gates"]["fit"]["min_accuracy"]),
        max_new_tokens=int(generation["max_new_tokens"]),
        stop_tokens=generation["stop_tokens"], candidates=candidates,
        tokenizer=decoder)
    coverage = recomputed["coverage"]
    attempts = recomputed["attempts"]
    _require(receipt["coverage"] == coverage,
             "fit receipt case coverage differs from the declared fit workload")
    _require(receipt["attempts"] == attempts,
             "fit receipt attempt accounting differs from the preserved fit gate")
    _require(recomputed["passed"],
             "fit gate does not independently satisfy the declared accuracy, "
             f"coverage and generation rules: {recomputed['gate_failures']}")
    _require(list(gate.get("gate_failures") or ())
             == recomputed["gate_failures"]
             and (gate.get("all_declared_gates_passed") is True)
             == recomputed["passed"],
             "fit gate report disagrees with independently recomputed evidence")
    _require(math.isclose(float(gate.get("fit_accuracy", -1.0)),
                          attempts["accuracy"], rel_tol=1e-12, abs_tol=1e-12),
             "fit gate accuracy differs from independently recomputed evidence")

    teaching = receipt["teaching"]
    _require(isinstance(teaching, Mapping) and "primary" in teaching,
             "fit receipt lacks the primary teaching run")
    primary = teaching["primary"]
    _require(isinstance(primary, Mapping)
             and "run_path" in primary,
             "fit receipt primary teaching block is invalid")
    bundle = teaching_bundle(Path(primary["run_path"]).parent,
                             expected_updates=int(
                                 config["workload"][
                                     "teaching_events_for_this_baseline"]))
    _require(dict(primary) == bundle,
             "fit receipt teaching run/cost/state hashes are stale")
    _require(primary["world_uid"] == manifest["primary_world"],
             "fit receipt teaching run belongs to another world")
    _require(primary["p_only_sha256"] == p_state["sha256"],
             "fit receipt P-state is not the teaching run export")
    _require(primary["p_only_bytes"] == int(p_state["bytes"]),
             "fit receipt P-state bytes differ from the teaching export")
    _require(primary["protocol_sha256"] == protocol_block["protocol_sha256"]
             and primary["config_sha256"] == config_block["sha256"]
             and primary["episodes_manifest_sha256"]
             == episodes["manifest_sha256"],
             "fit receipt teaching run belongs to another protocol/config/episodes")
    _require([int(value) for value in p_state["expert_steps"]]
             == [int(value) for value in primary["expert_steps"]],
             "fit receipt P-state counters differ from the teaching run")
    if "wrong_world" in teaching:
        wrong = teaching["wrong_world"]
        _require(isinstance(wrong, Mapping) and "run_path" in wrong,
                 "fit receipt wrong-world teaching block is invalid")
        wrong_bundle = teaching_bundle(
            Path(wrong["run_path"]).parent,
            expected_updates=int(config["workload"]["wrong_world_teaching_events"]))
        _require(dict(wrong) == wrong_bundle,
                 "fit receipt wrong-world teaching hashes are stale")
        _require(wrong["world_uid"] == manifest["wrong_world"],
                 "fit receipt wrong-world teaching run belongs to another world")

    if state is not None:
        verify_fit_receipt_payload(receipt, state=state)
    ROLE_ACCESS.record_validated(receipt)


def bound_teach_cost(receipt: Mapping[str, Any], role: str,
                     path: Path | None) -> None:
    """Reject a duty-cycle cost report from another teaching run."""
    if path is None:
        return
    block = receipt["teaching"].get(role)
    _require(isinstance(block, Mapping),
             f"fit receipt lacks the {role} teaching cost binding")
    _require(Path(block["cost_path"]).resolve() == Path(path).resolve()
             and block["cost_sha256"] == _sha256_file(path),
             f"the {role} teach cost report differs from the fit receipt")


def require_fit_receipt(args, config: Mapping[str, Any], *,
                        manifest: Mapping[str, Any], manifest_path: Path,
                        episodes_root: Path) -> dict | None:
    """Admission gate run before ``torch.load`` and any native/development access."""
    reset_role_access()
    if not fit_receipt_required(config):
        _require(getattr(args, "fit_receipt", None) is None,
                 "--fit-receipt is only valid for the behavioral protocol")
        ROLE_ACCESS.admit_without_fit_receipt(config)
        return None
    path = getattr(args, "fit_receipt", None)
    _require(path is not None,
             "the behavioral protocol requires a passing fit receipt "
             "(--fit-receipt) before development evaluation")
    receipt = load_fit_receipt(Path(path))
    validate_fit_receipt(
        receipt, config=config, config_path=Path(args.protocol),
        manifest=manifest, manifest_path=manifest_path,
        episodes_root=episodes_root, state_path=Path(args.p_state),
        lock=load_model_lock(args.model_lock),
        model_path=Path(args.model_path))
    ROLE_ACCESS.admit_fit_receipt(receipt)
    return receipt


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------


def run_init(args, config: dict) -> int:
    """Initialize the declared null bank; zero updates and zero events."""
    output: Path = args.output
    _require(is_v2(config), "init mode requires the v2 protocol")
    validate_v2_config(config)
    manifest = load_episode_manifest(args.episodes)
    validate_episode_binding(args, config, manifest)
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
            "config_sha256": _sha256_file(args.protocol),
            "stop_policy_version": STOP_POLICY_VERSION,
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
            "config_sha256": _sha256_file(args.protocol),
            "stop_policy_version": STOP_POLICY_VERSION,
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
        state_history = []
        canonical_seconds = 0.0
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
            transaction = outcome.to_dict()
            transaction["gradient_norm_before_clipping"] = controller.last_gradient_norm
            transactions.append(transaction)
            canonical_started = time.perf_counter()
            state_history.append({
                "event_id": outcome.event_id,
                "event_index": len(state_history) + 1,
                "transaction": transaction,
                "legacy_state_digest": state_digest(learner),
                "canonical_state": canonical_state(learner),
            })
            canonical_seconds += time.perf_counter() - canonical_started
            observations.append({"event_id": outcome.event_id,
                                 "chunk": outcome.chunk.to_dict()})
        wall = time.perf_counter() - started
        stamp_after = adapter.parameter_stamp()
        _require(stamp_before == stamp_after,
                 "native theta parameter stamp changed during plastic learning")
        _write_jsonl(output / "transactions.jsonl", transactions)
        _write_jsonl(output / "state_history.jsonl", state_history)
        _write_jsonl(output / "observations.jsonl", observations)
        _write_json(output / "protocol.json", protocol.to_dict())
        checkpoint = learner.save_checkpoint(output / "full_checkpoint.pt")
        p_only_path = learner.save_p_only(output / "p_only" / "state.pt")
        final_canonical = canonical_state(learner)
        _write_json(output / "final_state.json", final_canonical)
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
            "canonical_state_seconds": canonical_seconds,
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
            "config_sha256": _sha256_file(args.protocol),
            "implementation_sha256": _implementation_pins(config),
            "theta_stamp_unchanged": stamp_before == stamp_after,
            "parameter_audit": "parameter_audit.json",
            "updates": len(episodes), "learning_events": len(episodes),
            "state_history": "state_history.jsonl",
            "final_state": "final_state.json",
            "canonical_state_sha256": final_canonical["sha256"],
            "legacy_state_digest": state_digest(learner),
            "checkpoint_sha256": _sha256_file(checkpoint),
            "checkpoint_bytes": checkpoint.stat().st_size,
            "p_only_sha256": _sha256_file(p_only_path),
            "checkpoint": str(checkpoint), "p_only": str(p_only_path),
            "descriptive_only": True, "promotion": False,
            "non_claims": ["no utility, capability or heldout claim",
                           "single seed is descriptive only"],
        })
    return 0


def run_response_teach(args, config: dict) -> int:
    """Bounded response-CE teaching with complete phase and position accounting."""
    output: Path = args.output
    _require(is_v2(config), "response-CE teaching requires the behavioral protocol")
    _require(config.get("training_admitted") is True,
             "response-CE training requires an admitted protocol")
    _require(config["transaction"]["objective_admitted_for_this_protocol"] is True
             and config["transaction"]["external_gradient_transaction"] is True,
             "the protocol must admit the response-CE gradient transaction")
    _require(config["transaction"].get("objective")
             == "verified_response_cross_entropy_through_frozen_decoder",
             "response-CE teaching requires the declared response objective")
    validate_v2_config(config)
    manifest = load_episode_manifest(args.episodes)
    validate_episode_binding(args, config, manifest)
    world_uid = (manifest["primary_world"] if args.world_role == "primary"
                 else manifest["wrong_world"])
    if args.world_role == "wrong_world":
        parent_path = (args.wrong_world_parent_checkpoint
                       or config["training"]["wrong_world_parent_checkpoint"])
    else:
        parent_path = (args.parent_checkpoint
                       or config["training"]["parent_checkpoint"])
    _require(parent_path is not None,
             "the behavioral reference requires a parent checkpoint")
    started_total = time.perf_counter()
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
        payload = torch.load(parent_path, map_location="cpu", weights_only=True)
        _require(payload.get("format") == "replica.nested_text.checkpoint.v1",
                 "parent is not a nested-text full checkpoint")
        _require(TextProtocol.from_dict(payload["text"]["protocol"]) == protocol,
                 "parent checkpoint protocol differs from the behavioral protocol")
        learner.load_checkpoint(payload)
        order = TextAccessOrder()
        order.access("teaching")
        stamp_before = adapter.parameter_stamp()
        episodes = episodes_for_teaching(
            args.episodes, manifest, tokenizer, protocol,
            max_prompt_tokens=args.max_prompt_tokens,
            max_response_tokens=args.max_response_tokens,
            limit=args.max_events)
        _require(episodes, "no eligible teaching episodes")
        update_limit = int(config["training"][
            "wrong_world_max_updates" if args.world_role == "wrong_world"
            else "max_updates"])
        _require(len(episodes) * int(protocol.steps_per_event) <= update_limit,
                 f"response-CE optimizer steps exceed declared update bound {update_limit}")
        (output / "teach").mkdir(parents=True, exist_ok=True)
        session = MemorySession(adapter, tokenizer,
                                RecordLedger(output / "teach" / "ledger.sqlite"))
        scope = empty_scope(protocol)
        transactions, observations = [], []
        started = time.perf_counter()
        update_seconds = 0.0
        for episode in episodes:
            update_started = time.perf_counter()
            outcome = learner.learn_event_response_ce(
                session, episode.event, scope=scope,
                prompt_ids=episode.prompt_ids, prompt_mask=episode.prompt_mask,
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
                 "native theta parameter stamp changed during response-CE learning")
        _write_jsonl(output / "transactions.jsonl", transactions)
        _write_jsonl(output / "observations.jsonl", observations)
        _write_json(output / "protocol.json", protocol.to_dict())
        checkpoint_started = time.perf_counter()
        checkpoint = learner.save_checkpoint(output / "full_checkpoint.pt")
        checkpoint_seconds = time.perf_counter() - checkpoint_started
        p_only_started = time.perf_counter()
        p_only_path = learner.save_p_only(output / "p_only" / "state.pt")
        p_only_seconds = time.perf_counter() - p_only_started
        p_only = torch.load(p_only_path, map_location="cpu", weights_only=True)
        final_canonical = canonical_state(learner)
        _write_json(output / "final_state.json", final_canonical)
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
        prompt_positions = sum(int(episode.prompt_ids.shape[1]) for episode in episodes)
        full_positions = sum(int(episode.input_ids.shape[1]) for episode in episodes)
        supervised = sum(int(episode.response_mask.sum()) for episode in episodes)
        events = len(episodes)
        response_work = response_teaching_work(
            prompt_positions=prompt_positions, full_positions=full_positions,
            response_tokens=supervised, events=events,
            steps_per_event=int(protocol.steps_per_event))
        _require(int(learner.accounting["optimizer_steps"])
                 == response_work["backward_calls"],
                 "optimizer steps differ from response-CE transaction accounting")
        end_to_end = time.perf_counter() - started_total
        cost = {
            "events": events,
            "eligible_events": int(learner.accounting["eligible_events"]),
            "optimizer_steps": int(learner.accounting["optimizer_steps"]),
            "feature_vectors": int(learner.accounting["feature_vectors"]),
            "response_tokens": int(learner.accounting["response_tokens"]),
            **response_work,
            "replay_calls": 0,
            "wall_seconds": wall,
            "update_seconds": update_seconds,
            "learning_update_seconds": update_seconds,
            "duty_cycle": (update_seconds / wall) if wall > 0 else None,
            "end_to_end_seconds": end_to_end,
            "teaching_end_to_end_seconds": end_to_end,
            "checkpoint_seconds": checkpoint_seconds,
            "p_only_seconds": p_only_seconds,
            "protocol_sha256": protocol.protocol_sha256,
            "peak_driver_bytes": peak_driver_bytes(),
            "checkpoint_bytes": checkpoint.stat().st_size,
            "p_only_bytes": p_only_path.stat().st_size,
            "model_loads": 1, "tokenizer_loads": 1,
            "teacher_requests": events,
            "scoring_version": SCORING_VERSION,
            **version_metadata(args),
        }
        _write_json(output / "cost.json", cost)
        _write_json(output / "run.json", {
            "format": PROMPT_FORM, "mode": "teach", "status": "completed",
            "world_uid": world_uid, "protocol_sha256": protocol.protocol_sha256,
            "episodes_manifest_sha256": _sha256_file(
                args.episodes / "manifest.json"),
            "config_sha256": _sha256_file(args.protocol),
            "implementation_sha256": _implementation_pins(config),
            "theta_stamp_unchanged": stamp_before == stamp_after,
            "checkpoint": str(checkpoint), "p_only": str(p_only_path),
            "checkpoint_sha256": _sha256_file(checkpoint),
            "checkpoint_bytes": checkpoint.stat().st_size,
            "p_only_sha256": _sha256_file(p_only_path),
            "p_only_bytes": p_only_path.stat().st_size,
            "p_only_manifest_sha256": _sha256_file(
                output / "p_only" / "manifest.json"),
            "canonical_state_sha256": final_canonical["sha256"],
            "final_state": "final_state.json",
            "expert_steps": list(bank.expert_steps),
            "cost_sha256": _sha256_file(output / "cost.json"),
            "objective": config["transaction"]["objective"],
            "updates": events, "learning_events": events,
            "descriptive_only": True, "promotion": False,
            "non_claims": ["bounded response-CE reference only",
                           "no utility, capability or heldout claim",
                           "single seed is descriptive only"],
        })
    return 0


def run_fit_eval(args, config: dict) -> int:
    """Evaluate fit rows against the exact teaching export and seal a receipt."""
    output: Path = args.output
    _require(is_v2(config), "fit evaluation requires the behavioral protocol")
    validate_v2_config(config)
    _require(fit_receipt_required(config),
             "fit evaluation requires the admitted behavioral protocol")
    teach_run = getattr(args, "teach_run", None)
    _require(teach_run is not None,
             "fit evaluation requires --teach-run: the teaching output that "
             "produced the P-only state and its cost/state hashes")
    wrong_world_teach_run = getattr(args, "wrong_world_teach_run", None)
    required_attempts = int(config["workload"]["fit_eval_attempts"])
    required_updates = int(config["workload"]["teaching_events_for_this_baseline"])
    manifest = load_episode_manifest(args.episodes)
    validate_episode_binding(args, config, manifest)
    teach = teaching_bundle(Path(teach_run), expected_updates=required_updates)
    _require(Path(teach["p_only_path"]).resolve() == Path(args.p_state).resolve(),
             "fit evaluation must score the exact teaching P-only export file")
    _require(teach["p_only_sha256"] == _sha256_file(args.p_state),
             "fit evaluation P-state digest differs from the teaching export")
    state = torch.load(args.p_state, map_location="cpu", weights_only=True)
    _require(isinstance(state, Mapping) and set(state) == P_ONLY_KEYS,
             "P-only state must contain exactly the declared weights/binding keys")
    evaluation_started = time.perf_counter()
    with exclusive_mps_worker(ROOT):
        adapter, tokenizer, lock = load_local_mps(
            args.model_path, args.model_lock)
        protocol = protocol_for_world(
            config, lock, hidden_size=adapter.base.config.hidden_size,
            world_uid=manifest["primary_world"])
        adapter.install_plastic(protocol.plastic_spec)
        validate_p_only_payload(state, protocol=protocol)
        _require(teach["protocol_sha256"] == protocol.protocol_sha256,
                 "teaching run protocol differs from the fit protocol")
        _require(teach["config_sha256"] == _sha256_file(args.protocol),
                 "teaching run config differs from the fit config")
        _require(teach["episodes_manifest_sha256"]
                 == _sha256_file(args.episodes / "manifest.json"),
                 "teaching run episodes differ from the fit episodes")
        _require([int(value) for value in state["expert_steps"]]
                 == [int(value) for value in teach["expert_steps"]],
                 "P-only expert counters differ from the teaching run")
        learner = build_read_only_learner(protocol, config, adapter, state)
        (output / "fit").mkdir(parents=True, exist_ok=True)
        session = MemorySession(adapter, tokenizer,
                                RecordLedger(output / "fit" / "ledger.sqlite"))
        episodes = episodes_for_teaching(
            args.episodes, manifest, tokenizer, protocol,
            max_prompt_tokens=args.max_prompt_tokens,
            max_response_tokens=args.max_response_tokens,
            limit=required_attempts)
        candidates = fit_candidate_answers_for(
            args.episodes, manifest, world_uid=manifest["primary_world"],
            scope=f"{manifest['primary_world']}/text")
        latency_rows, work_rows = [], []
        score = score_episodes_v2(
            learner, session, episodes, route_mode=None,
            max_new_tokens=int(config["generation"]["max_new_tokens"]),
            stop_tokens=tuple(int(token) for token in config["generation"]["stop_tokens"]),
            label="fit", candidate_answers=candidates,
            latency_rows=latency_rows, work_rows=work_rows)
        attempts = len(episodes)
        _require(attempts == required_attempts,
                 f"fit workload requires {required_attempts} attempts; measured {attempts}")
        audits = [item.get("request_audit") for item in score["results"]]
        audited = sum(1 for audit in audits if isinstance(audit, Mapping))
        audit_complete = audited == attempts and all(
            all(field in audit for field in _AUDIT_FIELDS)
            for audit in audits if isinstance(audit, Mapping))
        source_bearing = any(
            isinstance(audit, Mapping) and (
                audit.get("sources_present") is not False
                or audit.get("memory_spans") != 0
                or audit.get("selection_records") != 0
                or audit.get("ledger_records") != 0
                or audit.get("controller_constructed") is not False
                or audit.get("coordinator_constructed") is not False
                or audit.get("updates_enabled") is not False)
            for audit in audits)
        _require(not source_bearing,
                 "fit request constructed a source-bearing state")
        fit_accuracy = float(score["accuracy"])
        gate_failures = []
        if fit_accuracy < float(config["gates"]["fit"]["min_accuracy"]):
            gate_failures.append("min_fit_accuracy")
        if int(score["failures"]) > 0:
            gate_failures.append("generation_failures")
        if not audit_complete:
            gate_failures.append("incomplete_request_audit")
        counts = attempt_counts(scoring_attempts=attempts,
                                failures=int(score["failures"]))
        latency_summary = summarize_latency(latency_rows)
        work_totals = sum_phase_work(work_rows)
        cost = {
            "fit_accuracy": fit_accuracy,
            "scoring_attempts": attempts,
            "failed_attempts": int(score["failures"]),
            "generated_tokens": work_totals["generated_ids"],
            "evaluation_end_to_end_seconds": time.perf_counter() - evaluation_started,
            "peak_driver_bytes": peak_driver_bytes(),
            "latency_ms": latency_summary,
            "work": work_totals,
            "scoring_version": SCORING_VERSION,
            "answer_grammar_version": ANSWER_GRAMMAR_VERSION,
            **version_metadata(args),
        }
        _write_json(output / "request_audit.json", {
            "requests_audited": audited, "scoring_attempts": attempts,
            "audit_complete": audit_complete,
            "source_free": audit_complete and not source_bearing,
            "controller_constructed": False, "coordinator_constructed": False,
            "updates_enabled": False, "moments_loaded": False,
            "dedup_loaded": False, "clock_loaded": False,
            "ledger_records": audits[0]["ledger_records"]
            if isinstance(audits[0], Mapping) else None,
            "source_text_present": source_bearing,
            "answer_grammar_version": ANSWER_GRAMMAR_VERSION,
            **version_metadata(args),
        })
        _write_json(output / "cost.json", cost)
        _write_json(output / "phase_work.json", {
            "format": "replica.nested_text.phase_work.v2",
            "scoring_version": SCORING_VERSION,
            "answer_grammar_version": ANSWER_GRAMMAR_VERSION,
            **counts,
            "latency_ms": latency_summary,
            "totals": work_totals,
            "requests": _merge_request_rows(work_rows, latency_rows),
        })
        _write_json(output / "gates" / "fit.json", {
            "label": "fit", "fit_accuracy": fit_accuracy,
            "all_declared_gates_passed": not gate_failures,
            "gate_failures": gate_failures,
            "metrics": {"cost": cost},
            "scores": score,
            "latency_ms": latency_summary,
            "work": work_totals,
            "scoring_version": SCORING_VERSION,
            "answer_grammar_version": ANSWER_GRAMMAR_VERSION,
            "decode_policy_version": DECODE_POLICY_VERSION,
            **version_metadata(args),
        })
        receipt = build_fit_receipt(
            config=config, config_path=args.protocol, manifest=manifest,
            manifest_path=args.episodes / "manifest.json",
            episodes_root=args.episodes, state_path=args.p_state, state=state,
            protocol=protocol, teach_dir=Path(teach_run),
            teach_expected_updates=required_updates,
            fit_gate_path=output / "gates" / "fit.json",
            lock=lock, tokenizer=tokenizer,
            wrong_world_teach_dir=(Path(wrong_world_teach_run)
                                   if wrong_world_teach_run is not None
                                   else None))
        _require(receipt["status"] == ("passed" if not gate_failures else "failed"),
                 "fit receipt status differs from the fit gate")
        if receipt["status"] == "passed":
            validate_fit_receipt(
                receipt, config=config, config_path=args.protocol,
                manifest=manifest, manifest_path=args.episodes / "manifest.json",
                episodes_root=args.episodes, state_path=args.p_state,
                lock=lock, state=state, tokenizer=tokenizer)
        receipt_path = output / FIT_RECEIPT_FILE
        _write_json(receipt_path, receipt)
        _write_json(output / "run.json", {
            "format": PROMPT_FORM, "mode": "fit_eval",
            "status": "stopped_fit_gate" if gate_failures else "completed",
            "fit_accuracy": fit_accuracy,
            "gate_failures": gate_failures,
            "heldout_opened": False,
            "fit_receipt": FIT_RECEIPT_FILE,
            "fit_receipt_status": receipt["status"],
            "fit_receipt_sha256": _sha256_file(receipt_path),
            "answer_grammar_version": ANSWER_GRAMMAR_VERSION,
            "scoring_version": SCORING_VERSION,
            **version_metadata(args),
        })
    return 2 if gate_failures else 0


def run_p_only_v1(args, config: dict) -> int:
    output: Path = args.output
    reset_role_access()
    ROLE_ACCESS.admit_without_fit_receipt(config)
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
    validate_episode_binding(args, config, manifest)
    primary_world = manifest["primary_world"]
    wrong_world_uid = manifest["wrong_world"]
    # Dependent-development admission: this fails closed before torch.load, the
    # native worker/model load and any development-role access.
    fit_receipt = require_fit_receipt(
        args, config, manifest=manifest,
        manifest_path=args.episodes / "manifest.json",
        episodes_root=args.episodes)
    if fit_receipt is not None:
        bound_teach_cost(fit_receipt, "primary", args.teach_cost)
        bound_teach_cost(fit_receipt, "wrong_world", args.wrong_world_teach_cost)
        if (args.wrong_world_p is not None
                and "wrong_world" in fit_receipt["teaching"]):
            _require(fit_receipt["teaching"]["wrong_world"]["p_only_sha256"]
                     == _sha256_file(args.wrong_world_p),
                     "wrong-world P control differs from the fit receipt teaching run")
    state = torch.load(args.p_state, map_location="cpu", weights_only=True)
    _require(isinstance(state, Mapping) and set(state) == P_ONLY_KEYS,
             "P-only state must contain exactly the declared weights/binding keys")
    if fit_receipt is not None:
        verify_fit_receipt_payload(fit_receipt, state=state)
    _write_json(output / "p_only_input.json", {
        "path": str(args.p_state), "sha256": _sha256_file(args.p_state),
        "keys": sorted(state),
        "forbidden_present": sorted(
            key for key in state if key in
            ("optimizer", "dedup", "clock", "revision", "coordinator",
             "observations", "moments", "sources")),
        "scoring_version": SCORING_VERSION,
        "answer_grammar_version": ANSWER_GRAMMAR_VERSION,
        "fit_receipt": (None if fit_receipt is None else {
            "path": str(args.fit_receipt),
            "sha256": _sha256_file(args.fit_receipt),
            "receipt_sha256": fit_receipt["receipt_sha256"],
            "status": fit_receipt["status"],
            "p_state_sha256": fit_receipt["p_state"]["sha256"],
            "gate_report_sha256": fit_receipt["gate_report"]["sha256"],
        }),
        **version_metadata(args),
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
        if fit_receipt is not None:
            verify_fit_receipt_protocol(fit_receipt, protocol=protocol)
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
        candidates = development_candidate_answers_for(args.episodes, manifest)
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
        audit_complete = len(audits) == scoring_attempts and all(
            all(field in audit for field in _AUDIT_FIELDS)
            for audit in audits)
        source_bearing = any(
            audit.get("sources_present") is not False
            or audit.get("memory_spans") != 0
            or audit.get("selection_records") != 0
            or audit.get("ledger_records") != 0
            or audit.get("controller_constructed") is not False
            or audit.get("coordinator_constructed") is not False
            or audit.get("updates_enabled") is not False
            for audit in audits)
        source_free = audit_complete and not source_bearing
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
            "source_text_present": source_bearing,
            "scoring_version": SCORING_VERSION,
            "answer_grammar_version": ANSWER_GRAMMAR_VERSION,
            **version_metadata(args),
            "candidate_answers": len(candidates),
            "declared_attempts": declared_attempts,
            "group_attempts": group_counts,
        })
        _require(audit_complete,
                 "P-only request audit is incomplete; "
                 f"{scoring_attempts - len(audits)} attempts were not audited")
        _require(not source_bearing,
                 "P-only request constructed a source-bearing state")
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
        unknown_false = false_recall_v2(scores["unknown"], taught_answers)
        revoked_false = false_recall_v2(scores["revoked"], taught_answers)
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
        wrong_teach_cost = None
        wrong_learning_update_seconds = None
        wrong_teaching_end_to_end_seconds = None
        if args.wrong_world_teach_cost is not None:
            wrong_teach_cost = json.loads(
                Path(args.wrong_world_teach_cost).read_text())
            _require(wrong_teach_cost.get("protocol_sha256")
                     == wrong_protocol.protocol_sha256,
                     "wrong-world teach cost report belongs to another protocol")
            wrong_learning_update_seconds = wrong_teach_cost.get(
                "learning_update_seconds", wrong_teach_cost.get("update_seconds"))
            wrong_teaching_end_to_end_seconds = wrong_teach_cost.get(
                "teaching_end_to_end_seconds", wrong_teach_cost.get("end_to_end_seconds"))
            _require(wrong_learning_update_seconds is not None
                     and wrong_teaching_end_to_end_seconds is not None,
                     "wrong-world teach cost report lacks a duty-cycle component")
        _require(args.wrong_world_teach_cost is not None,
                 "v2 requires wrong-world teaching/init cost for complete duty accounting")
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

        receipt_binding = (None if fit_receipt is None else {
            "path": str(args.fit_receipt),
            "sha256": _sha256_file(args.fit_receipt),
            "receipt_sha256": fit_receipt["receipt_sha256"],
            "status": fit_receipt["status"],
            "p_state_sha256": fit_receipt["p_state"]["sha256"],
            "fit_accuracy": fit_receipt["attempts"]["accuracy"],
            "gate_report_sha256": fit_receipt["gate_report"]["sha256"],
        })
        failure_count = sum(int(group.get("failures", 0))
                            for group in scored_groups)
        counts = attempt_counts(scoring_attempts=scoring_attempts,
                                failures=failure_count)
        phase_work_report = {
            "format": "replica.nested_text.phase_work.v2",
            "scoring_version": SCORING_VERSION,
            "answer_grammar_version": ANSWER_GRAMMAR_VERSION,
            "fit_receipt_sha256": (None if receipt_binding is None
                                   else receipt_binding["sha256"]),
            **version_metadata(args),
            **counts,
            "latency_ms": latency_summary,
            "totals": work_totals,
            "requests": _merge_request_rows(work_rows, latency_rows),
        }
        def build(evaluation_end_to_end_seconds: float):
            duty = None
            if teach_cost is not None:
                duty = workload_duty_cycle(
                    learning_update_seconds=float(learning_update_seconds),
                    teaching_end_to_end_seconds=float(
                        teaching_end_to_end_seconds),
                    evaluation_end_to_end_seconds=float(
                        evaluation_end_to_end_seconds),
                    wrong_world_learning_update_seconds=float(
                        wrong_learning_update_seconds),
                    wrong_world_teaching_end_to_end_seconds=float(
                        wrong_teaching_end_to_end_seconds))
            cost_value = {
                "peak_driver_bytes": peak,
                "warm_p95_overhead_ms": warm_p95,
                "warm_p95_excludes_first_request": len(totals) > 1,
                "duty_cycle": duty,
                "duty_cycle_source": ({
                    "primary": str(args.teach_cost),
                    "wrong_world": str(args.wrong_world_teach_cost)}
                    if teach_cost is not None else None),
                "duty_cycle_formula": (
                    "(learning_update_seconds + wrong_world_learning_update_seconds) / "
                    "(teaching_end_to_end_seconds + wrong_world_teaching_end_to_end_seconds + "
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
                "answer_grammar_version": ANSWER_GRAMMAR_VERSION,
                "fit_receipt_sha256": (None if receipt_binding is None
                                       else receipt_binding["sha256"]),
                **version_metadata(args),
                "learning_update_seconds": learning_update_seconds,
                "teaching_end_to_end_seconds": teaching_end_to_end_seconds,
                "wrong_world_learning_update_seconds": wrong_learning_update_seconds,
                "wrong_world_teaching_end_to_end_seconds": wrong_teaching_end_to_end_seconds,
                "wrong_world_duty_cycle_source": (
                    str(args.wrong_world_teach_cost)
                    if wrong_teach_cost is not None else None),
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
                "answer_grammar_version": ANSWER_GRAMMAR_VERSION,
                "fit_receipt": receipt_binding,
                **version_metadata(args),
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
                "answer_grammar_version": ANSWER_GRAMMAR_VERSION,
                "fit_receipt_sha256": (None if receipt_binding is None
                                       else receipt_binding["sha256"]),
                **version_metadata(args),
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
                "answer_grammar_version": ANSWER_GRAMMAR_VERSION,
                "fit_receipt_sha256": (None if receipt_binding is None
                                       else receipt_binding["sha256"]),
                **version_metadata(args),
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
            **version_metadata(args),
            "descriptive_only": True, "single_seed": True})
        _write_json(output / "gates" / "access_final.json", order.snapshot())
        _write_json(output / "run.json", {
            "format": PROMPT_FORM, "mode": "p_only_eval",
            "status": "completed", "heldout_opened": True,
            "scoring_version": SCORING_VERSION,
            "answer_grammar_version": ANSWER_GRAMMAR_VERSION,
            "fit_receipt_sha256": (None if receipt_binding is None
                                   else receipt_binding["sha256"]),
            **version_metadata(args),
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
        stamp_before = adapter.parameter_stamp()
        personality = PersonalityIdentity.from_dict(
            payload["text"]["personality"])
        bank, coordinator, controller = build_learning_owners(
            protocol, config, personality=personality)
        resumed = NestedTextLearner(
            adapter, bank=bank, protocol=protocol, controller=controller,
            coordinator=coordinator, personality=personality,
            tokenizer_sha256=protocol.tokenizer_sha256)
        resumed.load_checkpoint(payload)
        before = state_digest(resumed)
        before_canonical = canonical_state(resumed)
        _write_json(output / "before_state.json", before_canonical)
        rows = []
        for role in ("teaching", "correction"):
            rows.extend(load_rows(args.episodes, manifest, role))
        rows = [row for row in rows if row["eligible"]
                and row["world_uid"] == protocol.world_uid]
        rows.sort(key=lambda row: (float(row["target_available_at"]), row["id"]))
        next_row = next((row for row in rows if row["id"] == args.next_event), None)
        _require(next_row is not None,
                 f"next event {args.next_event!r} is not an eligible teaching row")
        _require(args.next_event not in resumed.controller.dedup.event_ids,
                 "next event is already committed; choose an untaught event")
        episode = build_episode(next_row, tokenizer, protocol,
                                max_prompt_tokens=args.max_prompt_tokens,
                                max_response_tokens=args.max_response_tokens)
        resumed_outcome = resumed.learn_event(
            episode.event, prompt_ids=episode.prompt_ids,
            prompt_mask=episode.prompt_mask, input_ids=episode.input_ids,
            attention_mask=episode.attention_mask,
            response_mask=episode.response_mask)
        stamp_after = adapter.parameter_stamp()
        _require(stamp_before == stamp_after,
                 "native theta parameter stamp changed during resume")
        after_resumed = state_digest(resumed)
        after_canonical = canonical_state(resumed)
        transaction = resumed_outcome.to_dict()
        transaction["gradient_norm_before_clipping"] = controller.last_gradient_norm
        _write_json(output / "after_state.json", after_canonical)
        _write_jsonl(output / "state_history.jsonl", [{
            "event_id": resumed_outcome.event_id,
            "event_index": int(resumed.controller.revision),
            "transaction": transaction,
            "legacy_state_digest": after_resumed,
            "canonical_state": after_canonical,
        }])
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
        equal = None
        if args.expected_state is not None:
            expected = json.loads(Path(args.expected_state).read_text())
            equal = expected.get("after_state") == after_resumed
        status = ("completed_pending_independent_comparison" if equal is None
                  else "completed" if equal else "mismatch")
        checkpoint_sha = _sha256_file(args.checkpoint)
        _write_json(output / "resume.json", {
            "format": PROMPT_FORM, "mode": "resume", "status": status,
            "next_event": args.next_event,
            "before_state": before, "after_state": after_resumed,
            "before_canonical_sha256": before_canonical["sha256"],
            "after_canonical_sha256": after_canonical["sha256"],
            "before_state_file": "before_state.json",
            "after_state_file": "after_state.json",
            "state_history": "state_history.jsonl",
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": checkpoint_sha,
            "checkpoint_bytes": args.checkpoint.stat().st_size,
            "config_sha256": _sha256_file(args.protocol),
            "implementation_sha256": _implementation_pins(config),
            "protocol_sha256": protocol.protocol_sha256,
            "episodes_manifest_sha256": _sha256_file(args.episodes / "manifest.json"),
            "resumed_outcome": transaction,
            "transaction_retried": bool(resumed_outcome.transaction.retried),
            "restart_equal": equal,
            "descriptive_only": True,
        })
        _write_json(output / "run.json", {
            "format": PROMPT_FORM, "mode": "resume", "status": status,
            "restart_equal": equal,
            "before_state": before, "after_state": after_resumed,
            "canonical_state_sha256": after_canonical["sha256"],
            "theta_stamp_unchanged": stamp_before == stamp_after,
            "parameter_audit": "parameter_audit.json",
            "checkpoint_sha256": checkpoint_sha,
            "state_history": "state_history.jsonl",
            "final_state": "after_state.json",
            "config_sha256": _sha256_file(args.protocol),
            "implementation_sha256": _implementation_pins(config),
        })
    return 0 if equal is not False else 3




def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode",
                        choices=("init", "teach", "fit-eval", "p-only-eval", "resume"),
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
    parser.add_argument("--wrong-world-teach-cost", type=Path,
                        help="measured wrong-world teach/init cost.json for complete workload duty")
    parser.add_argument("--fit-receipt", type=Path,
                        help="p-only-eval: preserved passing fit receipt that admits development")
    parser.add_argument("--teach-run", type=Path,
                        help="fit-eval: teaching output that produced the exact P-only state")
    parser.add_argument("--wrong-world-teach-run", type=Path,
                        help="fit-eval: optional wrong-world teaching output bound into the receipt")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--parent-checkpoint", type=Path)
    parser.add_argument("--wrong-world-parent-checkpoint", type=Path)
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
    if args.mode == "fit-eval" and args.p_state is None:
        parser.error("--p-state is required for fit-eval")
    if args.mode == "fit-eval" and args.checkpoint is not None:
        parser.error("a full checkpoint is not a P-only input for fit-eval")
    if args.mode != "teach" and (args.parent_checkpoint is not None
                                 or args.wrong_world_parent_checkpoint is not None):
        parser.error("parent checkpoints are only valid for teach")
    if args.mode == "resume" and (args.checkpoint is None or args.next_event is None):
        parser.error("--checkpoint and --next-event are required for resume")
    if args.mode == "teach" and args.no_updates:
        parser.error("teach mode must perform its declared update transaction")
    if args.mode not in ("p-only-eval", "fit-eval") and args.p_state is not None:
        parser.error("--p-state is only valid for evaluation modes")
    if args.mode != "p-only-eval" and (args.wrong_world_p is not None
                                       or args.teach_cost is not None
                                       or args.wrong_world_teach_cost is not None):
        parser.error("wrong-world and duty-cycle paths are only valid for p-only-eval")
    if args.mode == "p-only-eval" and args.checkpoint is not None:
        parser.error("a full checkpoint is not a P-only input")
    if args.mode != "p-only-eval" and args.fit_receipt is not None:
        parser.error("--fit-receipt is only valid for p-only-eval")
    if args.mode != "fit-eval" and (args.teach_run is not None
                                    or args.wrong_world_teach_run is not None):
        parser.error("--teach-run/--wrong-world-teach-run are only valid for fit-eval")
    if args.mode == "fit-eval" and args.fit_receipt is not None:
        parser.error("fit-eval produces a fit receipt; it does not consume one")
    if args.max_events < 1 or args.max_new_tokens < 1:
        parser.error("positive event/generation budgets are required")
    try:
        _fresh_output(args.output)
        from replica_lab.profiling import RunProfiler
        with RunProfiler("mps", args.output) as profiler:
            config = json.loads(args.protocol.read_text(encoding="utf-8"))
            validate_run_mode(config, args.mode)
            if args.wrong_world_teach_cost is not None and not is_v2(config):
                parser.error("--wrong-world-teach-cost is only valid for v2")
            _require(config.get("one_clock_control_only") is True,
                     "the baseline must remain one-clock only")
            _require(config.get("replay_bytes") == 0,
                     "the one-clock baseline forbids replay bytes")
            if is_v2(config):
                validate_v2_config(config)
            if fit_receipt_required(config):
                if args.mode == "p-only-eval" and args.fit_receipt is None:
                    parser.error("the behavioral protocol requires a passing "
                                 "--fit-receipt before development evaluation")
                if args.mode == "fit-eval" and args.teach_run is None:
                    parser.error("fit-eval requires --teach-run: the teaching "
                                 "output that produced the P-only state")
            if args.mode == "init":
                return run_init(args, config)
            if args.mode == "fit-eval":
                return run_fit_eval(args, config)
            if args.mode == "teach":
                if is_v2(config):
                    return run_response_teach(args, config)
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

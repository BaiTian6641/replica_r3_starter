"""Readout-contract v2: exact selected-record value or explicit status word.

This module implements one predeclared hypothesis over the frozen
``memory_address_bank`` decisions:

    The v1 readout failure is an answer-format and completion-budget failure,
    not a wrong-record-selection failure.  A static prompt that requires one
    line containing either the shown record value or the shown status word,
    with ``max_new_tokens=32``, passes the predeclared exact/correction/
    unaffected/unknown/revoked gates in both a first process and a genuinely
    new post-restart query form.

The experiment reuses, unchanged:

* ``build_split_decisions`` / ``annotate_decision_values`` / ``freeze_decisions``
  for the selected-record decisions;
* ``load_fresh_checkpoint`` / ``_unpack_cache`` for read-only checkpoint
  verification;
* ``normalize_answer`` / ``score_literal_values`` / ``_labels_map`` /
  ``_target`` / ``evaluate_dev_literal_gate`` for literal and target joins.

It adds only:

* ``build_prompt_v2`` — a static one-line response contract, no exemplars;
* the v2 generation row with ``single_line`` / ``cap_hit`` / ``max_new_tokens``
  and no evaluation field;
* phase-scoped gates with mechanically verified denominators;
* a role-sealed label loader so ``heldout`` bytes cannot be opened before every
  development gate passes.

Two fresh-process native jobs are declared.  Job ``first`` verifies prepared
inputs and native02, freezes dev decisions, applies the literal gate, loads
MiniCPM once and evaluates the first-process phase.  Job ``restart`` additionally
verifies the restart02 checkpoint evidence and the first job receipt, then
evaluates restart and revocation phases and only then the role-sealed heldout
labels.  No gradient, optimizer update, checkpoint write or export occurs.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Mapping, Sequence

from .memory_address_bank import (
    SPLIT_ROLES as ADDRESS_SPLIT_ROLES,
    SPLITS,
    _canonical,
    _prepared_root,
    _rate,
    _require,
    _sha_bytes,
    _sha_file,
    _sha_json,
    _sha_text,
    _utc_now,
    _verify_prepared_hashes,
    _write_new,
    build_split_decisions,
    decision_key_register,
    freeze_decisions,
    write_frozen_decisions,
)
from .memory_bank_value import (
    annotate_decision_values,
    decision_status,
    evaluate_dev_literal_gate,
    normalize_answer,
    query_text_register,
    score_literal_values,
    _display,
    _labels_map,
    _target,
    _verify_prepared_against_report,
    _verify_sealed_source,
)
from .memory_causal_fresh import (
    CHECKPOINT_FORMAT as CAUSAL_CHECKPOINT_FORMAT,
    LEDGER_FORMAT as CAUSAL_LEDGER_FORMAT,
    RUN_FORMAT as CAUSAL_RUN_FORMAT,
    _unpack_cache,
    load_fresh_checkpoint,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIG_FORMAT = "replica.memory_bank_readout.config.v2"
PROMPT_FORMAT = "replica.memory_bank_readout.prompt.v2"
GENERATIONS_FORMAT = "replica.memory_bank_readout.generations.v2"
REPORT_FORMAT = "replica.memory_bank_readout.report.v2"
ROLES_FORMAT = "replica.memory_bank_readout.roles.v1"
ROLE_MANIFEST_FORMAT = "replica.memory_bank_readout.role_manifest.v1"

MAX_NEW_TOKENS = 32
MAX_PROMPT_TOKENS = 256
MAX_DECISIONS_PER_SPLIT = 152
MAX_WALL_SECONDS = 1200
ARTIFACT_CAP_BYTES = 64 * 1024 * 1024
STOP_TOKEN_IDS = (1, 130073)
FRAME_BRANCHES = ("memory",)
JOBS = ("first", "restart")
JOB_PHASES = {
    "first": ("first_process",),
    "restart": ("restart", "revocation"),
}

STATUS_VOCABULARY = ("unknown", "revoked", "ambiguous", "superseded")
STATUS_PROMPT = {status: status.upper() for status in STATUS_VOCABULARY}
LITERAL_FACT_VALUE_MIN = 0.85
LITERAL_UNKNOWN_SELECTIONS_MAX = 0
LITERAL_REVOKED_SELECTIONS_MAX = 0
PHASE_EXACT_MIN = 0.85
CORRECTION_MIN = 0.60
UNAFFECTED_MIN = 0.70
UNKNOWN_MIN = 0.90
REVOKED_MIN = 0.90
CONTROL_FALSE_VALUE_MAX = 0
INITIAL_FACT_EXACT_MIN = 0.85
CORRECTED_FACT_EXACT_MIN = 0.85
SINGLE_LINE_RATE_MIN = 1.0
ENDED_BY_EOS_RATE_MIN = 1.0
CAP_HITS_MAX = 0
QUERY_TEXT_DISJOINT_MIN = 1.0
OLD_QUERY_REGENERATION_MAX = 0

EXPECTED_DEV_COUNTS = {
    "decision_rows": 152,
    "fact_rows": 128,
    "unknown_rows": 16,
    "revoked_rows": 8,
    "first_process_rows": 72,
    "first_process_fact_rows": 64,
    "first_process_unknown_rows": 8,
    "first_process_correction_rows": 8,
    "first_process_unaffected_rows": 20,
    "restart_rows": 72,
    "restart_fact_rows": 64,
    "restart_unknown_rows": 8,
    "restart_correction_rows": 8,
    "restart_unaffected_rows": 20,
    "restart_revoked_status_rows": 4,
    "revocation_rows": 8,
    "initial_fact_rows": 64,
    "corrected_fact_rows": 64,
    "total_generated_rows": 152,
}

PROMPT_CONTRACT = (
    "Response contract: Output exactly one line and then stop.",
    "Do not explain, reason, quote, use Markdown, or add a prefix.",
    "Do not invent values.",
    "If a record value is shown, output exactly that value.",
    "If a record status is shown, output exactly that status word.",
    "Permitted status words: UNKNOWN, REVOKED, AMBIGUOUS, SUPERSEDED.",
)
PROMPT_FIELDS = [
    "response_contract",
    "query_text",
    "selected_record_value_or_runtime_status",
]

CLASSIFICATION = {
    "package": "selected_record_literal_value_and_model_readout_v2",
    "native": False,
    "model_execution": False,
    "tokenizer_load": False,
    "generation": False,
    "persistent_learning": False,
    "calibrated": False,
    "learned_selector": False,
    "bank_plus_compressed_arm": False,
    "compressed_arm": "separate_reference_not_combined",
    "oracle": "diagnostic_only_not_executed",
    "checkpoint_state_role": "validated_only_not_consumed_by_generation",
    "answer_source": "sealed_minicpm5_greedy_readout",
    "few_shot": False,
    "structured_decoding": False,
    "host_selected_answer": False,
    "retries": False,
}

LIMITATIONS = [
    "The hypothesis is a readout-contract hypothesis; it is not a persistent-learning, retention or calibrated-confidence claim.",
    "The bank selection layer is unchanged; prompt v2 changes only the response grammar and the completion budget.",
    "The prompt contains only the runtime query text, one authorized selected value or one runtime status word, and the static contract; it never contains an evaluation label.",
    "Heldout labels are role-sealed and opened only after every predeclared development gate passes; a failure opens no heldout label or heldout artifact.",
    "The four bank abstentions among the 128 dev fact rows remain misses in the fact denominator and are never relabeled.",
    "The restart02 checkpoint is a procedural post-restart anchor that is verified read-only; it is not an input to generation.",
    "No few-shot exemplars, structured decoding, host-selected candidates, retries or post-hoc answer extraction are used.",
    "Train is descriptive only, uses literal metrics only and is read after both development gates pass.",
    "No gradient, optimizer update, checkpoint write, base export, extension export or download occurs.",
]


def _resolve(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_json(path):
    return json.loads(Path(path).read_text())


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def _address_control_view(config):
    """Project the v2 protocol onto the frozen address-bank control contract."""
    return {
        "format": "replica.memory_address_bank.config.v1",
        "declared_at_utc": config["declared_at_utc"],
        "purpose": config["purpose"],
        "classification": dict(config["classification"]),
        "prepared": config["prepared"],
        "bank": config["bank"],
        "reader": config["reader"],
        "endpoint": {
            "split": "dev",
            "dev_fact_selection_identity_min": 0.90,
            "dev_unknown_selections_max": 0,
            "dev_revoked_selections_max": 0,
            "heldout": "evaluate_after_dev_pass_only",
            "tuning": "forbidden",
        },
        "roles": config["roles"],
        "reporting": config["reporting"],
        "budget": {
            "model_loads": 0,
            "tokenizer_loads": 0,
            "forward_passes": 0,
            "optimizer_updates": 0,
            "gradients": 0,
            "checkpoints_written": 0,
            "native_jobs": 0,
            "native_resource_reports": 0,
        },
        "limitations": config["limitations"],
    }


def _is_sha256(value):
    return (isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def validate_config(config):
    """Validate and freeze the v2 protocol without mutating it."""
    from .memory_address_bank import validate_config as _validate_address

    _require(isinstance(config, Mapping),
             "Readout v2 configuration must be an object")
    for name in ("format", "declared_at_utc", "purpose", "classification",
                 "prepared", "role_labels", "baseline", "bank", "reader",
                 "endpoint", "generation", "jobs", "roles", "reporting",
                 "budget", "limitations"):
        _require(name in config,
                 f"Readout v2 configuration is missing: {name}")
    _require(config["format"] == CONFIG_FORMAT,
             "Readout v2 configuration format differs")

    _validate_address(_address_control_view(config))
    _require(config["roles"] == ADDRESS_SPLIT_ROLES,
             "Split roles differ from the frozen address-bank roles")
    _require(config["bank"].get("value_comparison") is True,
             "The literal-value comparison must be declared")

    role_labels = config["role_labels"]
    _require(role_labels.get("format") == ROLES_FORMAT,
             "Role label format differs")
    _require(role_labels.get("manifest_format") == ROLE_MANIFEST_FORMAT,
             "Role manifest format differs")
    _require(dict(role_labels.get("files") or {}).get("dev")
             == "labels_dev.json"
             and dict(role_labels["files"]).get("heldout")
             == "labels_heldout.json"
             and dict(role_labels["files"]).get("train")
             == "labels_train.json",
             "Role label filenames differ")
    _require(set(role_labels["files"]) == set(SPLITS),
             "Role label files must name every split")
    pins = role_labels.get("pins") or {}
    for role in SPLITS:
        _require(_is_sha256(pins.get(role)),
                 f"Role label pin is invalid: {role}")
    _require(isinstance(pins.get("manifest"), str)
             and len(pins["manifest"]) == 64,
             "Role manifest pin is invalid")
    prepared_pins = config["prepared"]["pins"]
    _require(role_labels.get("source_labels_sha256")
             == prepared_pins["labels_sha256"],
             "Role labels must derive from the pinned prepared labels")

    baseline = config["baseline"]
    first = baseline.get("first_process") or {}
    restart = baseline.get("restart") or {}
    for label, item, phase, resumed in (
            ("first_process", first, "first_process", False),
            ("restart", restart, "restart", True)):
        _require(item.get("phase") == phase,
                 f"{label} baseline phase differs")
        _require(item.get("resumed") is resumed,
                 f"{label} baseline resumed flag differs")
        _require(item.get("report_format") == CAUSAL_RUN_FORMAT,
                 f"{label} baseline report format differs")
        _require(item.get("checkpoint_format") == CAUSAL_CHECKPOINT_FORMAT,
                 f"{label} baseline checkpoint format differs")
        _require(item.get("ledger_format") == CAUSAL_LEDGER_FORMAT,
                 f"{label} baseline ledger format differs")
        for key in ("checkpoint_sha256", "report_sha256", "ledger_sha256"):
            _require(_is_sha256(item.get(key)),
                     f"{label} baseline pin is invalid: {key}")
    _require(restart.get("loaded_checkpoint_sha256")
             == first.get("checkpoint_sha256"),
             "Restart checkpoint must load the pinned first-process state")
    _require(restart.get("old_observations_reencoded") == 0,
             "Restart baseline must re-encode zero old observations")
    _require(restart.get("old_queries_reencoded") == 0,
             "Restart baseline must re-encode zero old queries")
    _require(type(restart.get("restart_new_queries")) is int
             and restart["restart_new_queries"] == 216,
             "Restart baseline new-query count differs")
    _require(restart.get("fresh_process_parity") is True,
             "Restart baseline parity must be declared")
    _require(first.get("episodes") == 12,
             "First-process baseline episode count differs")
    _require(dict(first.get("split_episode_counts") or {})
             == {"train": 4, "dev": 4, "heldout": 4},
             "First-process split episode counts differ")

    endpoint = config["endpoint"]
    _require(endpoint.get("split") == "dev", "The endpoint split must be dev")
    pinned_endpoint = {
        "literal_fact_value_min": LITERAL_FACT_VALUE_MIN,
        "literal_unknown_selections_max": LITERAL_UNKNOWN_SELECTIONS_MAX,
        "literal_revoked_selections_max": LITERAL_REVOKED_SELECTIONS_MAX,
        "first_process_exact_min": PHASE_EXACT_MIN,
        "restart_exact_min": PHASE_EXACT_MIN,
        "correction_min": CORRECTION_MIN,
        "unaffected_min": UNAFFECTED_MIN,
        "unknown_min": UNKNOWN_MIN,
        "revoked_min": REVOKED_MIN,
        "control_false_value_max": CONTROL_FALSE_VALUE_MAX,
        "initial_fact_exact_min": INITIAL_FACT_EXACT_MIN,
        "corrected_fact_exact_min": CORRECTED_FACT_EXACT_MIN,
        "single_line_rate_min": SINGLE_LINE_RATE_MIN,
        "ended_by_eos_rate_min": ENDED_BY_EOS_RATE_MIN,
        "cap_hits_max": CAP_HITS_MAX,
        "query_text_disjoint_min": QUERY_TEXT_DISJOINT_MIN,
        "old_query_regeneration_max": OLD_QUERY_REGENERATION_MAX,
    }
    for name, value in pinned_endpoint.items():
        _require(endpoint.get(name) == value,
                 f"Endpoint threshold may not be tuned: {name}")
    _require(dict(endpoint.get("expected_dev_counts") or {})
             == EXPECTED_DEV_COUNTS,
             "Pinned development denominators differ")
    _require(endpoint.get("heldout")
             == "evaluate_after_all_dev_gates_pass_only",
             "Heldout must be evaluated only after every dev gate passes")
    _require(endpoint.get("tuning") == "forbidden",
             "Endpoint tuning must be forbidden")

    generation = config["generation"]
    _require(generation.get("prompt_format") == PROMPT_FORMAT,
             "Prompt format differs")
    _require(generation.get("max_new_tokens") == MAX_NEW_TOKENS,
             "Generation token budget may not be tuned")
    _require(generation.get("max_prompt_tokens") == MAX_PROMPT_TOKENS,
             "Prompt token budget may not be tuned")
    _require(generation.get("max_decisions_per_split")
             == MAX_DECISIONS_PER_SPLIT,
             "Per-split decision cap may not be tuned")
    _require(generation.get("wall_budget_seconds_per_job")
             == MAX_WALL_SECONDS,
             "Per-job wall budget may not be tuned")
    _require(generation.get("artifact_cap_bytes")
             == ARTIFACT_CAP_BYTES,
             "Artifact cap may not be tuned")
    _require(tuple(generation.get("stop_token_ids", ())) == STOP_TOKEN_IDS,
             "Generation stop token set differs")
    _require(tuple(generation.get("status_vocabulary", ()))
             == STATUS_VOCABULARY,
             "Status vocabulary differs")
    _require(generation.get("answer_normalization")
             == "strip+collapse_whitespace+casefold",
             "Answer normalization differs")
    for key in ("single_line_required", "eos_required"):
        _require(generation.get(key) is True,
                 f"Generation must require {key}")
    _require(generation.get("cap_hits_max") == CAP_HITS_MAX,
             "Cap-hit allowance may not be tuned")
    _require(tuple(generation.get("enabled_ports", ())) == (),
             "Generation must enable no ports")
    _require(tuple(generation.get("frame_branches", ()))
             == FRAME_BRANCHES,
             "Generation must use the empty-memory frame branch convention")
    _require(generation.get("core_impl") == "none",
             "Generation must use core_impl='none'")
    _require(generation.get("training") is False,
             "Generation must not train")
    _require(generation.get("record_memory") is False
             and generation.get("workspace") is False,
             "Generation must not use record memory or a workspace")
    _require(generation.get("few_shot") is False
             and generation.get("structured_decoding") is False
             and generation.get("host_selected_answer") is False,
             "Few-shot, structured decoding and host-selected answers are forbidden")
    _require(generation.get("model_loads_per_job") == 1,
             "Exactly one model load per job is allowed")

    jobs = config["jobs"]
    _require(tuple(jobs.get("order", ())) == JOBS,
             "Job order differs")
    _require(jobs.get("native_jobs") == 2,
             "Exactly two native jobs are allowed")
    _require(jobs.get("model_loads_total") == 2,
             "Exactly two total model loads are allowed")
    _require(list(jobs.get("phases", {}).get("first") or ())
             == list(JOB_PHASES["first"]),
             "First-process job phases differ")
    _require(list(jobs["phases"].get("restart") or ())
             == list(JOB_PHASES["restart"]),
             "Restart job phases differ")
    _require(jobs.get("first_process_gate_before_restart") is True,
             "The first-process gate must precede the restart job")
    _require(jobs.get("restart_revocation_gate_before_heldout") is True,
             "Restart and revocation gates must precede heldout")

    budget = config["budget"]
    _require(budget.get("model_loads") == 2,
             "The protocol allows exactly two model loads")
    _require(budget.get("tokenizer_loads") == 2,
             "The protocol allows exactly two tokenizer loads")
    for key in ("optimizer_updates", "gradients", "checkpoints_written",
                "base_export", "extension_export", "downloads"):
        _require(type(budget.get(key)) is int and budget[key] == 0,
                 f"Forbidden resource budget must be zero: {key}")
    _require(budget.get("fallback") == "disabled",
             "CPU/MPS fallback must be disabled")
    _require(isinstance(config["limitations"], Sequence)
             and config["limitations"], "Limitations are required")
    return config


# ---------------------------------------------------------------------------
# Role-sealed labels
# ---------------------------------------------------------------------------

def build_role_projection(labels_doc, episodes):
    """Project one evaluation-label document onto disjoint split roles.

    The projection is a preparation-time operation only.  It reads labels and
    the label-free prepared episodes, checks that every state's declared split
    agrees with the episode split, and writes no evaluation decision.
    """
    _require(labels_doc.get("evaluation_only") is True,
             "Prepared labels are not marked evaluation-only")
    labels = labels_doc.get("episodes")
    _require(isinstance(labels, Mapping) and labels,
             "Prepared evaluation labels are missing")
    by_split = {split: {} for split in SPLITS}
    episode_ids = {}
    for episode in episodes:
        episode_id = episode["episode_id"]
        split = episode.get("split")
        _require(split in SPLITS, "Prepared episode split differs")
        _require(episode_id not in episode_ids,
                 "Prepared episode IDs repeat")
        episode_ids[episode_id] = split
    _require(set(labels) == set(episode_ids),
             "Labels and prepared episodes name different episodes")
    for episode_id, split in episode_ids.items():
        states = labels[episode_id]
        _require(isinstance(states, Mapping) and states,
                 f"Labels are missing for episode: {episode_id}")
        projected = {}
        for state, state_labels in states.items():
            _require(state_labels.get("split") == split,
                     f"Label split differs from the episode split: {episode_id}")
            _require(state_labels.get("episode_id") == episode_id,
                     f"Label episode ID differs: {episode_id}")
            _require(state_labels.get("evaluation_only") is True,
                     "State labels are not evaluation-only")
            projected[state] = state_labels
        by_split[split][episode_id] = projected
    role_episode_ids = {split: sorted(by_split[split]) for split in SPLITS}
    all_ids = [frozenset(ids) for ids in role_episode_ids.values()]
    _require(sum(len(ids) for ids in all_ids) == len(episode_ids),
             "Role episode sets overlap")
    return {
        "episode_ids": role_episode_ids,
        "episodes": by_split,
    }


def seal_role_labels(prepared, output):
    """Write role-specific label files with per-file SHA256; never evaluate."""
    prepared = Path(prepared)
    output = Path(output)
    names = {
        "labels": "labels.json",
        "episodes": "episodes.json",
        "manifest": "manifest.json",
    }
    for name in names.values():
        _require((prepared / name).is_file(),
                 f"Prepared artifact is missing: {name}")
    labels_doc = _read_json(prepared / names["labels"])
    _require(labels_doc.get("format")
             == "replica.memory_causal_fresh.labels.v2",
             "Prepared labels format differs")
    episodes_doc = _read_json(prepared / names["episodes"])
    _require(episodes_doc.get("format")
             == "replica.memory_causal_fresh.episodes.v2",
             "Prepared episodes format differs")
    manifest = _read_json(prepared / names["manifest"])
    labels_sha = _sha_file(prepared / names["labels"])
    _require(manifest.get("labels_sha256") == labels_sha,
             "Prepared manifest label hash differs")
    episodes = episodes_doc.get("episodes")
    _require(isinstance(episodes, list) and episodes,
             "Prepared episodes are missing")
    projection = build_role_projection(labels_doc, episodes)

    if output.exists():
        _require(output.is_dir() and not any(output.iterdir()),
                 "Use a fresh role-label output directory")
    else:
        output.mkdir(parents=True, exist_ok=False)

    entries = {}
    for role in SPLITS:
        subset = projection["episodes"][role]
        payload = {
            "format": ROLES_FORMAT,
            "role": role,
            "evaluation_only": True,
            "source_labels_sha256": labels_sha,
            "source_manifest_sha256": _sha_file(prepared / names["manifest"]),
            "episode_ids": list(projection["episode_ids"][role]),
            "episodes": {episode_id: subset[episode_id]
                         for episode_id in projection["episode_ids"][role]},
        }
        data = (_canonical(payload) + "\n").encode("utf-8")
        filename = f"labels_{role}.json"
        path = output / filename
        _write_new(path, data)
        entries[role] = {
            "filename": filename,
            "sha256": _sha_bytes(data),
            "bytes": len(data),
            "episodes": len(payload["episode_ids"]),
            "states": sum(len(states) for states in subset.values()),
        }
        _require(_sha_file(path) == entries[role]["sha256"],
                 f"Sealed role bytes differ: {filename}")

    role_manifest = {
        "format": ROLE_MANIFEST_FORMAT,
        "source_labels_sha256": labels_sha,
        "source_manifest_sha256": _sha_file(prepared / names["manifest"]),
        "roles": entries,
        "role_episode_ids": projection["episode_ids"],
        "role_sets_disjoint": True,
        "heldout_written": True,
        "native_computation": False,
        "model_loaded": False,
        "tokenizer_loaded": False,
        "note": ("Role projection of the pinned prepared labels; only dev is "
                 "opened before every development gate passes."),
    }
    manifest_data = (_canonical(role_manifest) + "\n").encode("utf-8")
    _write_new(output / "manifest.json", manifest_data)
    role_manifest["manifest_sha256"] = _sha_bytes(manifest_data)
    return role_manifest


def _read_role_json(path):
    """Single seam for role-label reads; tests spy on this function."""
    return json.loads(Path(path).read_text())

def verify_role_manifest(roles, config):
    """Verify the sealed role manifest without opening any label file."""
    declaration = config["role_labels"]
    root = _resolve(roles if roles is not None else declaration["root"])
    path = root / "manifest.json"
    _require(path.is_file(), "Role label manifest is missing")
    _require(_sha_file(path) == declaration["pins"]["manifest"],
             "Role label manifest bytes differ from the pin")
    manifest = _read_json(path)
    _require(manifest.get("format") == declaration["manifest_format"],
             "Role label manifest format differs")
    _require(manifest.get("source_labels_sha256")
             == declaration["source_labels_sha256"],
             "Role manifest source labels hash differs")
    _require(manifest.get("role_sets_disjoint") is True,
             "Role episode sets must be declared disjoint")
    _require(manifest.get("heldout_written") is True,
             "The heldout role must be sealed at preparation time")
    _require(manifest.get("native_computation") is False
             and manifest.get("model_loaded") is False
             and manifest.get("tokenizer_loaded") is False,
             "Role sealing must not have loaded a model")
    entries = manifest.get("roles") or {}
    _require(set(entries) == set(SPLITS),
             "Role manifest must name every split")
    for role in SPLITS:
        entry = entries[role]
        _require(entry.get("filename") == declaration["files"][role],
                 f"Role manifest filename differs: {role}")
        _require(entry.get("sha256") == declaration["pins"][role],
                 f"Role manifest hash differs: {role}")
        _require((root / entry["filename"]).is_file(),
                 f"Sealed role file is missing: {role}")
    return {
        "path": _display(path),
        "sha256": declaration["pins"]["manifest"],
        "roles": {role: entries[role]["sha256"] for role in SPLITS},
        "role_sets_disjoint": True,
    }


def load_role_labels(roles, role, config, *, opened=None):
    """Load exactly one role's sealed labels and verify its pinned hash."""
    _require(role in SPLITS, f"Unknown label role: {role}")
    declaration = config["role_labels"]
    root = _resolve(roles if roles is not None else declaration["root"])
    filename = declaration["files"][role]
    path = root / filename
    _require(path.is_file(), f"Role label file is missing: {filename}")
    digest = _sha_file(path)
    _require(digest == declaration["pins"][role],
             f"Role label bytes differ from the pin: {role}")
    if opened is not None and role not in opened:
        opened.append(role)
    doc = _read_role_json(path)
    _require(doc.get("format") == declaration["format"],
             f"Role label format differs: {role}")
    _require(doc.get("role") == role, f"Role label role differs: {role}")
    _require(doc.get("evaluation_only") is True,
             f"Role labels are not evaluation-only: {role}")
    _require(doc.get("source_labels_sha256")
             == declaration["source_labels_sha256"],
             f"Role label source hash differs: {role}")
    labels = doc.get("episodes")
    _require(isinstance(labels, Mapping) and labels,
             f"Role labels are missing: {role}")
    _require(sorted(labels) == sorted(doc.get("episode_ids") or ()),
             f"Role label episode IDs differ: {role}")
    return labels


# ---------------------------------------------------------------------------
# Decisions and prompt v2
# ---------------------------------------------------------------------------

def build_readout_split_decisions(episodes, revocations, config, split):
    """Reuse the bank decision rule and add the runtime selected value."""
    validate_config(config)
    _require(split in SPLITS, f"Unknown split: {split}")
    selected = [episode for episode in episodes
                if episode.get("split") == split]
    _require(selected, f"No episodes are declared for split: {split}")
    decisions = build_split_decisions(
        episodes, revocations, _address_control_view(config), split)
    _require(len(decisions) <= config["generation"]["max_decisions_per_split"],
             f"Split exceeds the declared decision cap: {split}")
    annotated = annotate_decision_values(selected, decisions)
    _require(len({(row["episode_id"], row["state"], row["phase"],
                   row["query_id"]) for row in annotated}) == len(annotated),
             "Annotated decision keys are not unique")
    return annotated


def build_prompt_v2(decision, query_text):
    """Build the static one-line response contract from runtime inputs only.

    No label object, expected answer or evaluator field is accepted here.
    """
    _require(isinstance(query_text, str) and query_text.strip(),
             "Query text is missing")
    status = decision_status(decision)
    lines = list(PROMPT_CONTRACT)
    lines.append(f"Question: {query_text.strip()}")
    if status is None:
        value = decision.get("selected_value")
        _require(isinstance(value, str) and value,
                 "An authorized selection lacks its runtime value")
        lines.append(f"Record value: {value}")
    else:
        _require(status in STATUS_VOCABULARY,
                 f"Runtime status is outside the vocabulary: {status}")
        _require(decision.get("selected_value") is None,
                 "An abstained decision carries a literal value")
        lines.append(f"Record status: {STATUS_PROMPT[status]}")
    lines.append("Answer:")
    return "\n".join(lines)


def generation_row_v2(decision, query_text, *, prompt_sha256,
                      prompt_ids_sha256, prompt_tokens, generated_ids,
                      generated_text, max_new_tokens=MAX_NEW_TOKENS,
                      ended_by_eos=None):
    """Build one immutable, label-free v2 generation row."""
    _require(max_new_tokens == MAX_NEW_TOKENS,
             "Generation token budget may not be tuned")
    prompt = build_prompt_v2(decision, query_text)
    _require(_sha_text(prompt) == prompt_sha256,
             "Prompt hash differs from the exact prompt text")
    ids = [int(value) for value in generated_ids]
    _require(1 <= len(ids) <= max_new_tokens,
             "Generation exceeded its declared token budget")
    derived_eos = ids[-1] in STOP_TOKEN_IDS
    if ended_by_eos is not None:
        _require(bool(ended_by_eos) is derived_eos,
                 "Declared EOS outcome differs from the stop set")
    _require(isinstance(generated_text, str), "Generated text is missing")
    single_line = "\n" not in generated_text \
        and "\r" not in generated_text
    cap_hit = len(ids) == max_new_tokens
    normalized = normalize_answer(generated_text)
    status = decision_status(decision)
    if status is None:
        contract_exact = normalized == normalize_answer(
            decision["selected_value"])
    else:
        contract_exact = normalized == status
    return {
        "format": GENERATIONS_FORMAT,
        "job_phase": decision["phase"],
        "generation_id": f"generation-{decision['decision_id']}",
        "decision_id": decision["decision_id"],
        "episode_id": decision["episode_id"],
        "world_id": decision["world_id"],
        "split": decision["split"],
        "state": decision["state"],
        "phase": decision["phase"],
        "query_id": decision["query_id"],
        "query_form": decision.get("query_form"),
        "prompt_format": PROMPT_FORMAT,
        "prompt": prompt,
        "prompt_sha256": prompt_sha256,
        "prompt_ids_sha256": prompt_ids_sha256,
        "prompt_tokens": int(prompt_tokens),
        "selected_record_id": decision.get("selected_record_id"),
        "selected_value": decision.get("selected_value"),
        "runtime_status": status,
        "generated_ids": ids,
        "generated_text": generated_text,
        "normalized_answer": normalized,
        "ended_by_eos": derived_eos,
        "cap_hit": cap_hit,
        "single_line": single_line,
        "runtime_contract_exact": contract_exact,
        "stop_token_ids": list(STOP_TOKEN_IDS),
        "max_new_tokens": max_new_tokens,
        "model_readout": True,
    }


def generate_row_v2(session, tokenizer, decision, query_text, *,
                    maximum_prompt=MAX_PROMPT_TOKENS,
                    maximum_new=MAX_NEW_TOKENS):
    """Generate one v2 readout using the native generation convention.

    The request uses the empty-memory frame convention with no enabled ports,
    no record/workspace state, greedy decoding and no logits processor.
    """
    import torch
    from .architecture.contracts import TaskGraph
    from .architecture.loading import encode_prompt
    from .memory_system.ledger import EligibilityScope

    prompt = build_prompt_v2(decision, query_text)
    ids = encode_prompt(tokenizer, prompt, session.device)
    _require(ids.shape[1] <= maximum_prompt,
             f"Prompt exceeds the declared budget: {ids.shape[1]}")
    mask = torch.ones_like(ids, dtype=torch.bool)
    task = TaskGraph(task_id=f"bank-readout-v2-{decision['decision_id']}",
                     goal=prompt)
    scope = EligibilityScope(
        subject_id="bank-readout-v2", branch_id="literal-readout-v2",
        purpose="selected_record_readout_v2", as_of=0)
    request = session.prepare(
        prompt_ids=ids, prompt_mask=mask, query=prompt, scope=scope,
        task=task, training=False, visual=None, enabled_ports=(),
        core_impl="none", enabled_branches=FRAME_BRANCHES)
    try:
        token_ids = session.generate(
            request, max_new_tokens=maximum_new,
            eos_token_id=STOP_TOKEN_IDS)
        prompt_length = int(request.frame.sense.prompt_ids.shape[1])
        generated = [int(value) for value in
                     token_ids[0, prompt_length:].tolist()]
    finally:
        request.release()
    _require(1 <= len(generated) <= maximum_new,
             "Generation exceeded its declared token budget")
    text = tokenizer.decode(generated, skip_special_tokens=True)
    return generation_row_v2(
        decision, query_text,
        prompt_sha256=_sha_text(prompt),
        prompt_ids_sha256=_sha_json([int(value) for value in ids[0].tolist()]),
        prompt_tokens=int(ids.shape[1]),
        generated_ids=generated, generated_text=text,
        max_new_tokens=maximum_new)


def build_model_generator_v2(config, baseline_info, *, profiler=None,
                             deadline=None, device="mps"):
    """Load the sealed MiniCPM source once and return a v2 generator."""
    _require(device == "mps", "Native readout requires MPS")
    from .architecture.loading import load_local_mps
    from .memory_system.ledger import RecordLedger
    from .memory_system.session import MemorySession

    source = (baseline_info.get("source") or {}).get("source_pins")
    _require(isinstance(source, dict) and source.get("model_path")
             and source.get("model_lock"),
             "Sealed source pins are required for native generation")
    model, tokenizer, _lock = load_local_mps(
        _resolve(source["model_path"]),
        _resolve(source["model_lock"]), profiler=profiler)
    ledger = RecordLedger(":memory:")
    session = MemorySession(model, tokenizer, ledger)
    maximum_prompt = config["generation"]["max_prompt_tokens"]
    maximum_new = config["generation"]["max_new_tokens"]

    def generate(decisions, query_texts):
        rows = []
        for decision in decisions:
            if deadline is not None:
                _require(time.monotonic() <= deadline,
                         "Generation exceeded the declared wall budget")
            key = (decision["episode_id"], decision["phase"],
                   decision["query_id"])
            _require(key in query_texts,
                     f"Runtime query text is missing: {key}")
            rows.append(generate_row_v2(
                session, tokenizer, decision, query_texts[key],
                maximum_prompt=maximum_prompt, maximum_new=maximum_new))
        return tuple(rows)

    def close():
        ledger.close()

    generate.close = close
    return generate


# ---------------------------------------------------------------------------
# Row validation and immutable writes
# ---------------------------------------------------------------------------

def _decision_key(row):
    return (row["episode_id"], row["state"], row["phase"], row["query_id"])


def validate_generation_rows(rows, decisions, query_texts):
    """Require one immutable, label-free v2 row per decision."""
    rows = tuple(rows)
    _require(len(rows) == len(decisions),
             "Generation row count differs from the decision count")
    expected = {_decision_key(decision) for decision in decisions}
    seen = set()
    for row in rows:
        _require(row.get("format") == GENERATIONS_FORMAT,
                 "Generation format differs")
        key = _decision_key(row)
        _require(key in expected, "Generation row names an unknown decision")
        _require(key not in seen, "Generation keys repeat")
        seen.add(key)
        for forbidden in ("expected", "expected_answer", "gold_answer",
                          "label", "target_role", "cohort", "in_correction",
                          "in_unaffected"):
            _require(forbidden not in row,
                     f"Generation row contains an evaluation field: {forbidden}")
        decision = next(item for item in decisions
                        if _decision_key(item) == key)
        _require(row["decision_id"] == decision["decision_id"],
                 "Generation decision identity differs")
        _require(row["split"] == decision["split"],
                 "Generation split differs from its decision")
        _require(row["selected_record_id"]
                 == decision.get("selected_record_id"),
                 "Generation selection differs from its decision")
        _require(row["selected_value"] == decision.get("selected_value"),
                 "Generation literal value differs from its decision")
        _require(row["runtime_status"] == decision_status(decision),
                 "Generation status differs from its decision")
        text = row.get("generated_text")
        _require(isinstance(text, str), "Generated text is missing")
        _require(bool(row["single_line"])
                 == ("\n" not in text and "\r" not in text),
                 "Single-line flag differs from the generated text")
        ids = [int(value) for value in row["generated_ids"]]
        _require(1 <= len(ids) <= MAX_NEW_TOKENS,
                 "Generation exceeded its declared token budget")
        _require(bool(row["ended_by_eos"]) == (ids[-1] in STOP_TOKEN_IDS),
                 "EOS flag differs from the stop set")
        _require(bool(row["cap_hit"]) == (len(ids) == MAX_NEW_TOKENS),
                 "Cap-hit flag differs from the token count")
        _require(row["max_new_tokens"] == MAX_NEW_TOKENS,
                 "Generation token budget differs")
        runtime_key = (row["episode_id"], row["phase"], row["query_id"])
        _require(runtime_key in query_texts,
                 "Generation row names an unknown query text")
        prompt = build_prompt_v2(decision, query_texts[runtime_key])
        _require(row["prompt"] == prompt, "Prompt text differs")
        _require(row["prompt_sha256"] == _sha_text(prompt),
                 "Prompt hash differs")
    _require(seen == expected, "Generation rows are incomplete")
    return True


def write_generation_rows(output, rows, *, phase):
    _require(phase in JOB_PHASES["first"] + JOB_PHASES["restart"]
             + ("heldout",), f"Unknown generation phase: {phase}")
    _require(rows, f"Cannot write an empty generation set: {phase}")
    name = (f"generations_dev_{phase}.jsonl" if phase != "heldout"
            else "generations_heldout.jsonl")
    payload = b"".join((_canonical(row) + "\n").encode("utf-8")
                       for row in rows)
    path = Path(output) / name
    _write_new(path, payload)
    digest = _sha_file(path)
    return {
        "filename": name,
        "sha256": digest,
        "rows": len(rows),
        "first_write": True,
        "immutable": True,
    }


# ---------------------------------------------------------------------------
# Denominators and phase-scoped gates
# ---------------------------------------------------------------------------

def decision_counts(decisions, labels):
    """Mechanically compute every declared development denominator."""
    labels_by_episode = _labels_map(labels)
    rows = []
    for decision in decisions:
        state_labels, target = _target(labels_by_episode, decision)
        rows.append({
            "phase": decision["phase"],
            "state": decision["state"],
            "kind": target["kind"],
            "runtime_status": decision_status(decision),
            "in_correction": decision["query_id"]
            in set(state_labels.get("corrected_query_ids") or ()),
            "in_unaffected": decision["query_id"]
            in set(state_labels.get("unaffected_query_ids") or ()),
        })

    def phase_rows(phase):
        return [row for row in rows if row["phase"] == phase]

    def corrected_scope(phase, flag):
        return [row for row in phase_rows(phase)
                if row["state"] == "corrected" and row[flag]
                and row["kind"] == "fact"]

    first_process = phase_rows("first_process")
    restart = phase_rows("restart")
    revocation = phase_rows("revocation")
    counts = {
        "decision_rows": len(rows),
        "fact_rows": sum(row["kind"] == "fact" for row in rows),
        "unknown_rows": sum(row["kind"] == "unknown" for row in rows),
        "revoked_rows": sum(row["kind"] == "revoked" for row in rows),
        "first_process_rows": len(first_process),
        "first_process_fact_rows": sum(row["kind"] == "fact"
                                       for row in first_process),
        "first_process_unknown_rows": sum(row["kind"] == "unknown"
                                          for row in first_process),
        "first_process_correction_rows": len(
            corrected_scope("first_process", "in_correction")),
        "first_process_unaffected_rows": len(
            corrected_scope("first_process", "in_unaffected")),
        "restart_rows": len(restart),
        "restart_fact_rows": sum(row["kind"] == "fact" for row in restart),
        "restart_unknown_rows": sum(row["kind"] == "unknown"
                                    for row in restart),
        "restart_correction_rows": len(
            corrected_scope("restart", "in_correction")),
        "restart_unaffected_rows": len(
            corrected_scope("restart", "in_unaffected")),
        "restart_revoked_status_rows": sum(
            row["phase"] == "restart" and row["kind"] == "fact"
            and row["runtime_status"] == "revoked" for row in rows),
        "revocation_rows": len(revocation),
        "initial_fact_rows": sum(row["kind"] == "fact"
                                 and row["state"] == "initial"
                                 for row in rows),
        "corrected_fact_rows": sum(row["kind"] == "fact"
                                   and row["state"] == "corrected"
                                   for row in rows),
        "total_generated_rows": len(
            [row for row in rows if row["phase"] in JOB_PHASES["first"]
             + JOB_PHASES["restart"]]),
    }
    return counts


def validate_decision_counts(counts, config):
    expected = dict(config["endpoint"]["expected_dev_counts"])
    _require(counts == expected,
             "Mechanically computed denominators differ from the pins: "
             f"{counts} != {expected}")
    return True


def _block(rows):
    total = len(rows)
    correct = sum(row["correct"] for row in rows)
    false_value = sum(row["false_value"] for row in rows)
    status_correct = sum(row["status_correct"] for row in rows)
    return {
        "n": total,
        "correct_n": correct,
        "exact_answer_rate": _rate(correct, total),
        "status_correct_n": status_correct,
        "status_correct_rate": _rate(status_correct, total),
        "false_value_n": false_value,
        "false_value_rate": _rate(false_value, total),
    }


def _format_block(rows):
    total = len(rows)
    single = sum(bool(row["single_line"]) for row in rows)
    eos = sum(bool(row["ended_by_eos"]) for row in rows)
    cap = sum(bool(row["cap_hit"]) for row in rows)
    return {
        "n": total,
        "single_line_n": single,
        "single_line_rate": _rate(single, total),
        "ended_by_eos_n": eos,
        "ended_by_eos_rate": _rate(eos, total),
        "cap_hit_n": cap,
        "cap_hit_rate": _rate(cap, total),
        "empty_n": sum(not row["normalized_answer"] for row in rows),
        "generated_tokens": sum(len(row["generated_ids"]) for row in rows),
    }


def _score_generation(generation, labels_by_episode):
    state_labels, target = _target(labels_by_episode, generation)
    expected = target.get("expected")
    _require(isinstance(expected, str) and expected,
             "Evaluation target lacks an expected string")
    kind = target.get("kind")
    _require(kind in ("fact", "unknown", "revoked"),
             f"Unknown label kind: {kind}")
    answer = generation.get("normalized_answer")
    _require(isinstance(answer, str), "Generation lacks a normalized answer")
    cohort = (state_labels.get("cohort_by_query") or {}).get(
        generation["query_id"])
    _require(isinstance(cohort, str) and cohort, "Scoring cohort is missing")
    record_values = {
        normalize_answer(entry["value"])
        for entry in list(state_labels.get("live_entries") or ())
        + list(state_labels.get("negative_entries") or ())
        if isinstance(entry.get("value"), str) and entry["value"]
    }
    false_value = bool(
        kind in ("unknown", "revoked") and answer
        and answer in record_values and answer not in STATUS_VOCABULARY)
    return {
        "episode_id": generation["episode_id"],
        "world_id": generation["world_id"],
        "split": generation["split"],
        "state": generation["state"],
        "phase": generation["phase"],
        "query_id": generation["query_id"],
        "cohort": cohort,
        "complexity": state_labels.get("complexity"),
        "kind": kind,
        "expected": expected,
        "normalized_answer": answer,
        "runtime_status": generation["runtime_status"],
        "correct": answer == normalize_answer(expected),
        "status_correct": bool(
            kind in ("unknown", "revoked")
            and answer == normalize_answer(expected)),
        "false_value": false_value,
        "in_correction": generation["query_id"]
        in set(state_labels.get("corrected_query_ids") or ()),
        "in_unaffected": generation["query_id"]
        in set(state_labels.get("unaffected_query_ids") or ()),
        "single_line": bool(generation["single_line"]),
        "ended_by_eos": bool(generation["ended_by_eos"]),
        "cap_hit": bool(generation["cap_hit"]),
        "generated_ids": list(generation["generated_ids"]),
    }


def score_phase(generations, labels, decisions, phase):
    """Join one phase's frozen generations to the role-sealed labels."""
    _require(phase in JOB_PHASES["first"] + JOB_PHASES["restart"],
             f"Unknown phase: {phase}")
    expected_decisions = [decision for decision in decisions
                          if decision["phase"] == phase]
    _require(len(generations) == len(expected_decisions),
             f"Phase row count differs: {phase}")
    labels_by_episode = _labels_map(labels)
    scored = tuple(_score_generation(row, labels_by_episode)
                   for row in generations)
    control = [row for row in scored if row["kind"] in ("unknown", "revoked")]
    return {
        "phase": phase,
        "overall": _block(scored),
        "fact": _block([row for row in scored if row["kind"] == "fact"]),
        "correction": _block(
            [row for row in scored if row["state"] == "corrected"
             and row["in_correction"] and row["kind"] == "fact"]),
        "unaffected": _block(
            [row for row in scored if row["state"] == "corrected"
             and row["in_unaffected"] and row["kind"] == "fact"]),
        "unknown": _block([row for row in scored
                           if row["kind"] == "unknown"]),
        "revoked": _block([row for row in scored
                           if row["kind"] == "revoked"]),
        "control": _block(control),
        "format": _format_block(scored),
        "counts": {
            "rows": len(scored),
            "states": len({(row["episode_id"], row["state"])
                           for row in scored}),
            "episodes": len({row["episode_id"] for row in scored}),
        },
    }


def evaluate_state_gates(generations, labels, decisions, config):
    """Exact answer rates over all initial/corrected fact rows."""
    labels_by_episode = _labels_map(labels)
    scored = [_score_generation(row, labels_by_episode)
              for row in generations]
    expected = config["endpoint"]["expected_dev_counts"]
    checks = {}
    for state, key, minimum in (
            ("initial", "initial_fact_rows", INITIAL_FACT_EXACT_MIN),
            ("corrected", "corrected_fact_rows",
             CORRECTED_FACT_EXACT_MIN)):
        rows = [row for row in scored
                if row["state"] == state and row["kind"] == "fact"]
        block = _block(rows)
        checks[f"{state}_fact_exact"] = {
            "value": block["exact_answer_rate"],
            "operator": ">=",
            "threshold": minimum,
            "n": block["n"],
            "expected_n": expected[key],
            "correct_n": block["correct_n"],
            "passed": (block["n"] == expected[key]
                       and block["exact_answer_rate"] is not None
                       and block["exact_answer_rate"] >= minimum),
        }
    return {
        "evaluated_after_restart_gates": True,
        "checks": checks,
        "passed": all(check["passed"] for check in checks.values()),
    }


def _check(name, value, operator, threshold, *, n=None, expected_n=None,
           count=None):
    passed = value is not None
    if operator == ">=":
        passed = passed and value >= threshold
    elif operator == "<=":
        passed = passed and value <= threshold
    else:  # pragma: no cover - declared operators only
        raise ValueError(f"Unknown operator: {operator}")
    if expected_n is not None:
        passed = passed and n == expected_n
    return {
        "value": value,
        "operator": operator,
        "threshold": threshold,
        "n": n,
        "expected_n": expected_n,
        "count": count,
        "passed": bool(passed),
    }


def evaluate_phase_gate(phase_metrics, config):
    """Apply the predeclared phase gate; denominators must match the pins."""
    phase = phase_metrics["phase"]
    endpoint = config["endpoint"]
    expected = endpoint["expected_dev_counts"]
    format_block = phase_metrics["format"]
    checks = {}
    if phase == "first_process":
        checks["exact_answer"] = _check(
            "exact_answer", phase_metrics["overall"]["exact_answer_rate"],
            ">=", endpoint["first_process_exact_min"],
            n=phase_metrics["overall"]["n"],
            expected_n=expected["first_process_rows"],
            count=phase_metrics["overall"]["correct_n"])
    elif phase == "restart":
        checks["exact_answer"] = _check(
            "exact_answer", phase_metrics["overall"]["exact_answer_rate"],
            ">=", endpoint["restart_exact_min"],
            n=phase_metrics["overall"]["n"],
            expected_n=expected["restart_rows"],
            count=phase_metrics["overall"]["correct_n"])
    else:
        checks["revoked_exact"] = _check(
            "revoked_exact",
            phase_metrics["revoked"]["exact_answer_rate"],
            ">=", endpoint["revoked_min"],
            n=phase_metrics["revoked"]["n"],
            expected_n=expected["revocation_rows"],
            count=phase_metrics["revoked"]["correct_n"])
    if phase in ("first_process", "restart"):
        prefix = "first_process" if phase == "first_process" else "restart"
        checks["correction"] = _check(
            "correction", phase_metrics["correction"]["exact_answer_rate"],
            ">=", endpoint["correction_min"],
            n=phase_metrics["correction"]["n"],
            expected_n=expected[f"{prefix}_correction_rows"],
            count=phase_metrics["correction"]["correct_n"])
        checks["unaffected"] = _check(
            "unaffected", phase_metrics["unaffected"]["exact_answer_rate"],
            ">=", endpoint["unaffected_min"],
            n=phase_metrics["unaffected"]["n"],
            expected_n=expected[f"{prefix}_unaffected_rows"],
            count=phase_metrics["unaffected"]["correct_n"])
        checks["unknown"] = _check(
            "unknown", phase_metrics["unknown"]["exact_answer_rate"],
            ">=", endpoint["unknown_min"],
            n=phase_metrics["unknown"]["n"],
            expected_n=expected[f"{prefix}_unknown_rows"],
            count=phase_metrics["unknown"]["correct_n"])
    checks["control_false_value"] = _check(
        "control_false_value", phase_metrics["control"]["false_value_n"],
        "<=", endpoint["control_false_value_max"],
        n=phase_metrics["control"]["n"],
        expected_n=expected["revocation_rows"] if phase == "revocation"
        else expected["first_process_unknown_rows"]
        if phase == "first_process" else expected["restart_unknown_rows"],
        count=phase_metrics["control"]["false_value_n"])
    checks["single_line"] = _check(
        "single_line", format_block["single_line_rate"], ">=",
        endpoint["single_line_rate_min"], n=format_block["n"],
        expected_n=phase_metrics["overall"]["n"],
        count=format_block["single_line_n"])
    checks["ended_by_eos"] = _check(
        "ended_by_eos", format_block["ended_by_eos_rate"], ">=",
        endpoint["ended_by_eos_rate_min"], n=format_block["n"],
        expected_n=phase_metrics["overall"]["n"],
        count=format_block["ended_by_eos_n"])
    checks["cap_hits"] = _check(
        "cap_hits", format_block["cap_hit_n"], "<=",
        endpoint["cap_hits_max"], n=format_block["n"],
        expected_n=phase_metrics["overall"]["n"],
        count=format_block["cap_hit_n"])
    return {
        "phase": phase,
        "evaluated_split": "dev",
        "checks": checks,
        "passed": all(check["passed"] for check in checks.values()),
    }


def evaluate_literal_gate(dev_metrics, config):
    """Reuse the frozen v1 literal endpoint with the pinned v2 values."""
    endpoint = config["endpoint"]
    gate = evaluate_dev_literal_gate(dev_metrics, {
        "dev_literal_fact_value_min": endpoint["literal_fact_value_min"],
        "dev_literal_unknown_selections_max":
            endpoint["literal_unknown_selections_max"],
        "dev_literal_revoked_selections_max":
            endpoint["literal_revoked_selections_max"],
    })
    gate["evaluated_split"] = "dev"
    gate["denominator_pin"] = {
        "fact_rows": config["endpoint"]["expected_dev_counts"]["fact_rows"],
        "fact_value_exact_n": gate["checks"]["fact_value_exact"].get(
            "value_exact_n"),
    }
    _require(gate["checks"]["fact_value_exact"]["fact_n"]
             == config["endpoint"]["expected_dev_counts"]["fact_rows"],
             "Literal fact denominator differs from the pin")
    return gate


# ---------------------------------------------------------------------------
# Restart-query accounting and first-job receipt
# ---------------------------------------------------------------------------

def restart_query_accounting(episodes, split="dev"):
    """Verify post-restart query texts are disjoint from first-process texts."""
    declared = {"first_process": 0, "restart": 0, "revocation": 0}
    overlap = []
    revocation_reuse = 0
    for episode in episodes:
        if episode.get("split") != split:
            continue
        episode_id = episode["episode_id"]
        old = [query["text"] for query in episode.get("queries", ())]
        restart = [query["text"]
                   for query in episode.get("restart_queries", ())]
        revocation = [query["text"]
                      for query in episode.get("revocation_queries", ())]
        declared["first_process"] += len(old)
        declared["restart"] += len(restart)
        declared["revocation"] += len(revocation)
        shared = set(old) & set(restart)
        for text in sorted(shared):
            overlap.append({"episode_id": episode_id,
                            "restart_text_sha256": _sha_text(text)})
        revocation_reuse += len(set(old) & set(revocation))
    disjoint = not overlap
    return {
        "split": split,
        "declared_query_texts": declared,
        "restart_vs_first_process_overlap_n": len(overlap),
        "restart_texts_disjoint": disjoint,
        "disjoint_rate": 1.0 if disjoint else 0.0,
        "overlap": overlap,
        "revocation_texts_reusing_first_process_n": revocation_reuse,
        "old_observations_encoded": 0,
        "old_first_process_queries_regenerated": 0,
        "restart_new_queries": declared["restart"],
        "note": ("The restart phase uses new query texts; revocation reuses "
                 "the first-process distractor text and is a control."),
    }


def evaluate_restart_accounting(accounting, config):
    endpoint = config["endpoint"]
    checks = {
        "restart_texts_disjoint": _check(
            "restart_texts_disjoint",
            1.0 if accounting["restart_texts_disjoint"] else 0.0,
            ">=", endpoint["query_text_disjoint_min"],
            n=accounting["declared_query_texts"]["restart"],
            count=accounting["restart_vs_first_process_overlap_n"]),
        "old_first_process_queries_regenerated": _check(
            "old_first_process_queries_regenerated",
            accounting["old_first_process_queries_regenerated"], "<=",
            endpoint["old_query_regeneration_max"]),
        "restart_new_queries_positive": _check(
            "restart_new_queries_positive",
            1.0 if accounting["restart_new_queries"] > 0 else 0.0,
            ">=", 1.0, n=accounting["restart_new_queries"]),
        "old_observations_encoded": _check(
            "old_observations_encoded",
            accounting["old_observations_encoded"], "<=", 0),
    }
    return {"checks": checks,
            "passed": all(check["passed"] for check in checks.values())}


def _load_first_generations(first_output, config, decisions, query_texts):
    """Read and verify the immutable first-job receipt and its raw rows."""
    first_output = Path(first_output)
    report_path = first_output / "report.json"
    _require(report_path.is_file(), "The first-process report is missing")
    report = _read_json(report_path)
    _require(report.get("format") == REPORT_FORMAT,
             "First-process report format differs")
    _require(report.get("job") == "first",
             "The restart job requires a first-process report")
    protocol = report.get("protocol") or {}
    config_sha = protocol.get("config_sha256")
    _require(isinstance(config_sha, str) and len(config_sha) == 64,
             "First-process report lacks the configuration hash")
    _require(report["raw_outputs"]["verified_unchanged"] is True,
             "First-process raw outputs changed after their first write")
    gating = report.get("gating") or {}
    _require(gating.get("literal_dev", {}).get("passed") is True,
             "First-process literal gate did not pass")
    _require(((gating.get("first_process") or {}).get("gate") or {})
             .get("passed") is True,
             "The first-process gate did not pass; restart is forbidden")
    _require(gating.get("heldout", {}).get("status") == "not_run",
             "The first-process job must not evaluate heldout")

    files = report["raw_outputs"]["files"]
    decision_manifest = files.get("dev")
    _require(isinstance(decision_manifest, dict),
             "First-process decisions are missing")
    decision_path = first_output / decision_manifest["filename"]
    _require(decision_path.is_file(), "First-process decisions are missing")
    _require(_sha_file(decision_path) == decision_manifest["sha256"],
             "First-process decisions changed")
    frozen = freeze_decisions(decisions, split="dev")
    _require(decision_manifest["sha256"] == frozen.sha256,
             "First-process decisions differ from this fresh process")

    generation_manifest = files.get("dev_first_process")
    _require(isinstance(generation_manifest, dict),
             "First-process generation rows are missing")
    generation_path = first_output / generation_manifest["filename"]
    _require(generation_path.is_file(),
             "First-process generation rows are missing")
    _require(_sha_file(generation_path) == generation_manifest["sha256"],
             "First-process generation rows changed")
    rows = tuple(json.loads(line) for line in
                 generation_path.read_text().splitlines() if line)
    phase_decisions = [decision for decision in frozen.decisions
                       if decision["phase"] == "first_process"]
    validate_generation_rows(rows, phase_decisions, query_texts)
    metrics = report.get("gating", {}).get("first_process", {}).get("metrics")
    _require(isinstance(metrics, dict),
             "First-process phase metrics are missing")
    return {
        "report_path": str(report_path),
        "report_sha256": _sha_file(report_path),
        "report": report,
        "generations": rows,
        "generation_manifest": generation_manifest,
        "decision_manifest": decision_manifest,
    }


# ---------------------------------------------------------------------------
# Baseline verification (read-only)
# ---------------------------------------------------------------------------

def _verify_baseline_item(config, label, prepared_root, episodes_by_id):
    baseline = config["baseline"][label]
    report_path = _resolve(baseline["report"])
    checkpoint_path = _resolve(baseline["checkpoint"])
    ledger_path = _resolve(baseline["ledger"])
    for path, name in ((report_path, "report"),
                       (checkpoint_path, "checkpoint"),
                       (ledger_path, "ledger")):
        _require(path.is_file(), f"{label} baseline {name} is missing")
    _require(_sha_file(report_path) == baseline["report_sha256"],
             f"{label} report bytes differ from the pin")
    _require(_sha_file(checkpoint_path) == baseline["checkpoint_sha256"],
             f"{label} checkpoint bytes differ from the pin")
    _require(_sha_file(ledger_path) == baseline["ledger_sha256"],
             f"{label} ledger bytes differ from the pin")

    report = _read_json(report_path)
    _require(report.get("format") == baseline["report_format"],
             f"{label} report format differs")
    _require(report.get("phase") == baseline["phase"],
             f"{label} report phase differs")
    _require(report.get("resumed") is baseline["resumed"],
             f"{label} report resumed flag differs")
    checkpoint = report.get("checkpoint") or {}
    _require(checkpoint.get("sha256") == baseline["checkpoint_sha256"],
             f"{label} report checkpoint hash differs")
    _require(checkpoint.get("bytes") == checkpoint_path.stat().st_size,
             f"{label} report checkpoint byte count differs")
    _require((report.get("exactly_once") or {}).get("ledger_sha256")
             == baseline["ledger_sha256"],
             f"{label} report ledger hash differs")

    payload = load_fresh_checkpoint(
        checkpoint_path, expected=report["bindings"], device="cpu")
    _require(payload.get("format") == baseline["checkpoint_format"],
             f"{label} checkpoint format differs")
    _require(payload.get("phase") == baseline["phase"],
             f"{label} checkpoint phase differs")
    ledger = _read_json(ledger_path)
    _require(payload.get("ledger") == ledger,
             f"{label} checkpoint ledger differs")
    _require(payload["ledger"].get("format") == baseline["ledger_format"],
             f"{label} ledger format differs")
    counters = payload.get("counters") or {}
    encoded = report.get("encoded") or {}
    if label == "first_process":
        _require(counters.get("observation_encodes")
                 == encoded.get("observations"),
                 f"{label} observation counter differs")
        _require(counters.get("query_encodes") == encoded.get("queries"),
                 f"{label} query counter differs")
    else:
        _require(int(counters.get("observation_encodes", 0)) == 0
                 and encoded.get("observations") == 0,
                 "Restart checkpoint recorded an observation encode")
        _require(counters.get("query_encodes_restart")
                 == encoded.get("restart_new_queries"),
                 "Restart checkpoint query counter differs")

    packed_episodes = payload.get("episodes")
    _require(isinstance(packed_episodes, list)
             and len(packed_episodes) == len(episodes_by_id),
             f"{label} checkpoint episode count differs")
    split_counts = {}
    seen = set()
    for packed in packed_episodes:
        episode_id = packed.get("episode_id")
        _require(episode_id in episodes_by_id,
                 f"{label} checkpoint names an unknown episode")
        _require(episode_id not in seen,
                 f"{label} checkpoint episode IDs repeat")
        seen.add(episode_id)
        episode = episodes_by_id[episode_id]
        _unpack_cache(packed, episode, "cpu")
        split_counts[episode["split"]] = (
            split_counts.get(episode["split"], 0) + 1)
    _require(split_counts == dict(baseline["split_episode_counts"]),
             f"{label} checkpoint split counts differ")
    _require(seen == set(episodes_by_id),
             f"{label} checkpoint does not cache every prepared episode")

    if label == "first_process":
        first_view = dict(config)
        first_view["baseline"] = config["baseline"]["first_process"]
        prepared_info = _verify_prepared_against_report(
            prepared_root, first_view, report)
    else:
        prepared_info = None
        _require(report.get("encoded", {}).get("old_observations_reencoded")
                 == baseline["old_observations_reencoded"],
                 "Restart report re-encoded old observations")
        _require(report["encoded"].get("old_queries_reencoded")
                 == baseline["old_queries_reencoded"],
                 "Restart report re-encoded old queries")
        _require(report["encoded"].get("restart_new_queries")
                 == baseline["restart_new_queries"],
                 "Restart report new-query count differs")
        _require(report.get("fresh_process_parity") is True,
                 "Restart report fresh-process parity differs")
        _require((report.get("gates") or {}).get("mechanical_pass") is True,
                 "Restart report mechanical gate did not pass")
        _require(report.get("optimizer_updates") == 0,
                 "Restart report records an optimizer update")
        _require(checkpoint.get("loaded_sha256")
                 == baseline["loaded_checkpoint_sha256"],
                 "Restart checkpoint did not load the pinned first state")
        _require(checkpoint.get("new_checkpoint_saved") is True,
                 "Restart checkpoint write was not recorded")
    return {
        "label": label,
        "phase": payload["phase"],
        "report": {"path": _display(report_path),
                   "sha256": baseline["report_sha256"]},
        "checkpoint": {
            "path": _display(checkpoint_path),
            "sha256": baseline["checkpoint_sha256"],
            "bytes": checkpoint_path.stat().st_size,
            "loaded": True,
            "written": False,
            "loaded_sha256": checkpoint.get("loaded_sha256"),
        },
        "ledger": {"path": _display(ledger_path),
                   "sha256": baseline["ledger_sha256"],
                   "commits": len(ledger.get("commits", ())),
                   "attempts": len(ledger.get("attempts", ()))},
        "bindings_verified": True,
        "cached_state_verified": True,
        "counters": dict(counters),
        "encoded": dict(report.get("encoded") or {}),
        "prepared_from_report": prepared_info,
        "source": (_verify_sealed_source(prepared_root, report)
                   if label == "first_process" else None),
    }


def verify_baselines(config, prepared, *, jobs=JOBS, verify_source=True):
    """Read-only verification of prepared04 and the pinned native checkpoints."""
    validate_config(config)
    prepared_root = _prepared_root(prepared, config)
    episodes_doc = _read_json(prepared_root / "episodes.json")
    episodes_by_id = {}
    for episode in episodes_doc.get("episodes", ()):
        episodes_by_id[episode["episode_id"]] = episode
    _require(episodes_by_id, "Prepared episodes are missing")
    if tuple(jobs) == ("first",):
        first = _verify_baseline_item(
            config, "first_process", prepared_root, episodes_by_id)
        if not verify_source:
            first["source"] = None
        return {"first_process": first}
    first = _verify_baseline_item(
        config, "first_process", prepared_root, episodes_by_id)
    restart = _verify_baseline_item(
        config, "restart", prepared_root, episodes_by_id)
    if not verify_source:
        first["source"] = None
        restart["source"] = None
    return {"first_process": first, "restart": restart}


# ---------------------------------------------------------------------------
# Prepared inputs
# ---------------------------------------------------------------------------

def load_prepared(config, prepared=None):
    """Verify and load exactly the selection inputs; labels stay sealed."""
    validate_config(config)
    prepared_root = _prepared_root(prepared, config)
    verification = _verify_prepared_hashes(prepared_root, config)
    formats = config["prepared"]["formats"]
    episodes_doc = _read_json(prepared_root / "episodes.json")
    _require(episodes_doc.get("format") == formats["episodes"],
             "Prepared episodes format differs")
    revision_doc = _read_json(prepared_root / "revocations.json")
    _require(revision_doc.get("format") == formats["revocations"],
             "Prepared revocations format differs")
    episodes = episodes_doc.get("episodes")
    revocations = revision_doc.get("revocations")
    _require(isinstance(episodes, list) and episodes,
             "Prepared episodes are missing")
    _require(isinstance(revocations, dict),
             "Prepared revocations are missing")
    episode_ids = {episode["episode_id"] for episode in episodes}
    _require(len(episode_ids) == len(episodes),
             "Prepared episode IDs repeat")
    _require(set(revocations) <= episode_ids,
             "Revocations name an unknown episode")
    manifest = verification["manifest"]
    _require(manifest.get("episodes") == len(episodes),
             "Prepared episode count differs from its manifest")
    return {
        "root": prepared_root,
        "episodes": tuple(episodes),
        "revocations": dict(revocations),
        "manifest": manifest,
        "hashes": verification["hashes"],
    }


# ---------------------------------------------------------------------------
# Report construction
# ---------------------------------------------------------------------------

def _artifact_bytes(output):
    return sum(path.stat().st_size for path in Path(output).iterdir()
               if path.is_file())


def _prepare_output(output):
    output = Path(output)
    allowed = {"profile_samples.jsonl"}
    if output.exists():
        _require(output.is_dir(), "Output path is not a directory")
        present = {path.name for path in output.iterdir()}
        _require(present <= allowed, "Use a fresh output directory")
    else:
        output.mkdir(parents=True, exist_ok=False)
    return output


def _close_generator(generator):
    close = getattr(generator, "close", None)
    if callable(close):
        close()


def _classification(config, *, native, model_executed):
    classification = dict(config["classification"])
    classification.update(CLASSIFICATION)
    classification["native"] = bool(native)
    classification["model_execution"] = bool(model_executed)
    classification["tokenizer_load"] = bool(model_executed)
    classification["generation"] = ("model_readout_only"
                                    if model_executed else False)
    return classification


def _protocol_block(config, config_path, prepared, implementation):
    return {
        "config_path": (str(config_path) if config_path else None),
        "config_sha256": (_sha_file(config_path)
                          if config_path and Path(config_path).is_file()
                          else None),
        "decision_rule_sha256": _sha_json({
            "bank": config["bank"],
            "reader": config["reader"],
            "endpoint": config["endpoint"],
            "generation": config["generation"],
            "jobs": config["jobs"],
        }),
        "implementation_sha256": implementation,
        "prepared": prepared,
        "prompt_contract": {
            "format": PROMPT_FORMAT,
            "fields": list(PROMPT_FIELDS),
            "contract_lines": list(PROMPT_CONTRACT),
            "status_vocabulary": list(STATUS_VOCABULARY),
            "expected_answer_included": False,
            "few_shot": False,
            "structured_decoding": False,
            "host_selected_answer": False,
            "evaluation_labels_read": False,
            "answer_normalization":
                config["generation"]["answer_normalization"],
        },
        "generation_contract": {
            "max_new_tokens": config["generation"]["max_new_tokens"],
            "stop_token_ids": list(STOP_TOKEN_IDS),
            "single_line_required": True,
            "ended_by_eos_required": True,
            "cap_hits_max": config["generation"]["cap_hits_max"],
            "enabled_ports": [],
            "enabled_branches": list(FRAME_BRANCHES),
            "core_impl": "none",
            "empty_ledger": True,
            "record_memory": False,
            "workspace": False,
        },
        "thresholds": config["endpoint"],
        "expected_dev_counts": config["endpoint"]["expected_dev_counts"],
    }


def _implementation_hashes():
    return {
        name: _sha_file(PROJECT_ROOT / name)
        for name in (
            "src/replica_lab/memory_bank_readout_v2.py",
            "src/replica_lab/memory_bank_value.py",
            "src/replica_lab/memory_address_bank.py",
            "src/replica_lab/memory_causal_fresh.py",
            "src/replica_lab/memory_matched.py",
            "scripts/run_memory_bank_readout_v2.py",
        )}


def _resource_usage(config, *, job, model_executed, generation_rows):
    return {
        "job": job,
        "model_loads": 1 if model_executed else 0,
        "tokenizer_loads": 1 if model_executed else 0,
        "generation_rows": len(generation_rows),
        "generated_tokens": sum(len(row["generated_ids"])
                                for row in generation_rows),
        "optimizer_updates": 0,
        "gradients": 0,
        "checkpoints_written": 0,
        "base_exports": 0,
        "extension_exports": 0,
        "downloads": 0,
        "native_jobs": 1,
        "fallback": "disabled",
        "limits": {
            "max_model_loads_per_job":
                config["generation"]["model_loads_per_job"],
            "max_decisions_per_split":
                config["generation"]["max_decisions_per_split"],
            "max_new_tokens": config["generation"]["max_new_tokens"],
            "max_wall_seconds":
                config["generation"]["wall_budget_seconds_per_job"],
            "artifact_cap_bytes": config["generation"]["artifact_cap_bytes"],
        },
    }


def _base_report(config, *, job, config_path, prepared_info, baseline_info,
                 native):
    return {
        "format": REPORT_FORMAT,
        "job": job,
        "generated_at_utc": _utc_now(),
        "classification": _classification(
            config, native=native, model_executed=False),
        "roles": config["roles"],
        "protocol": _protocol_block(
            config, config_path, prepared_info,
            _implementation_hashes()),
        "baseline": baseline_info,
        "gating": {
            "order": (["literal_dev", "first_process"]
                      if job == "first" else
                      ["first_receipt", "literal_dev", "restart",
                       "revocation", "state_denominators",
                       "restart_accounting", "heldout", "train"]),
            "literal_dev": None,
            "first_process": {
                "status": "not_run" if job == "first" else "evaluated_prior",
                "gate": None,
                "metrics": None,
            },
            "restart": {"status": "not_run", "gate": None, "metrics": None},
            "revocation": {"status": "not_run", "gate": None,
                           "metrics": None},
            "state_denominators": {"status": "not_run", "gate": None},
            "restart_accounting": {"status": "not_run", "gate": None},
            "heldout": {
                "status": "not_run",
                "evaluated_after_dev_pass": False,
                "fields": 0,
                "generations": 0,
                "metrics": None,
            },
            "train": {
                "status": "not_run",
                "role": "descriptive_only",
                "reason": "dev_gates_pending",
                "model_generation": "not_run",
                "metrics": None,
            },
            "stopped_before_model": True,
            "stopped_before_restart": job == "restart",
            "stopped_before_heldout": True,
        },
        "role_label_access": {"opened": [], "heldout_opened": False},
        "raw_outputs": {"policy": config["reporting"]["raw_output_policy"],
                        "files": {},
                        "verified_unchanged": True},
        "resource_usage": _resource_usage(
            config, job=job, model_executed=False, generation_rows=()),
        "profile": None,
        "limitations": list(config["limitations"]),
    }


def _finish_report(report, config, output, profiler):
    if profiler is not None:
        report["profile"] = profiler.stop()
    _require(report["elapsed_seconds"]
             <= config["generation"]["wall_budget_seconds_per_job"],
             "Run exceeded the declared wall budget")
    existing = _artifact_bytes(output)
    size = 0
    for _ in range(5):
        report["artifact_bytes"] = existing + size
        payload = (_canonical(report) + "\n").encode("utf-8")
        if len(payload) == size:
            break
        size = len(payload)
    _require(existing + len(payload)
             <= config["generation"]["artifact_cap_bytes"],
             "Artifact cap exceeded")
    _write_new(Path(output) / "report.json", payload)


def _verify_generation_file(output, rows, manifest):
    path = Path(output) / manifest["filename"]
    _require(path.is_file(), f"Generation file is missing: {path.name}")
    payload = b"".join((_canonical(row) + "\n").encode("utf-8")
                       for row in rows)
    _require(_sha_file(path) == _sha_bytes(payload),
             f"Generation file changed: {path.name}")
    _require(manifest["sha256"] == _sha_bytes(payload),
             "Generation manifest hash changed")
    return True


def _write_frozen_dev(output, frozen):
    return write_frozen_decisions(output, frozen)


def _heldout_metrics(generations, labels, decisions):
    labels_by_episode = _labels_map(labels)
    scored = [_score_generation(row, labels_by_episode)
              for row in generations]

    def group(key):
        groups = {}
        for row in scored:
            value = row.get(key)
            groups.setdefault("unknown" if value is None else str(value),
                              []).append(row)
        return {name: _block(groups[name]) for name in sorted(groups)}

    return {
        "overall": _block(scored),
        "by_phase": group("phase"),
        "by_state": group("state"),
        "correction": _block([row for row in scored
                              if row["in_correction"]]),
        "unaffected": _block([row for row in scored
                              if row["in_unaffected"]]),
        "unknown": _block([row for row in scored
                           if row["kind"] == "unknown"]),
        "revoked": _block([row for row in scored
                           if row["kind"] == "revoked"]),
        "format": _format_block(scored),
        "counts": {
            "rows": len(scored),
            "episodes": len({row["episode_id"] for row in scored}),
            "states": len({(row["episode_id"], row["state"])
                           for row in scored}),
        },
    }


# ---------------------------------------------------------------------------
# Job orchestration
# ---------------------------------------------------------------------------

def run_readout_job(config, *, job, output, prepared=None, roles=None,
                    first_output=None, config_path=None,
                    generation_factory=None, native=False, profiler=None,
                    clock=time.monotonic, verify_source=True):
    """Run one predeclared fresh-process job, fail-closed.

    Job ``first``: prepared + native02 verification, frozen dev decisions,
    literal gate, one model load, first-process generation and gate.

    Job ``restart``: prepared + native02/restart02 verification, immutable
    first-job receipt and first-process gate, frozen dev decisions, literal
    gate, one model load, restart/revocation generation and gates, then
    role-sealed heldout generation and only then heldout labels, then
    descriptive train.
    """
    validate_config(config)
    _require(job in JOBS, f"Unknown job: {job}")
    output = _prepare_output(output)
    started = clock()
    deadline = started + config["generation"]["wall_budget_seconds_per_job"]
    opened_roles = []

    prepared_inputs = load_prepared(config, prepared)
    episodes, revocations = prepared_inputs["episodes"], \
        prepared_inputs["revocations"]
    role_manifest = verify_role_manifest(roles, config)
    prepared_info = {
        "root": _display(prepared_inputs["root"]),
        "episodes": len(prepared_inputs["episodes"]),
        "hashes": prepared_inputs["hashes"],
        "manifest": prepared_inputs["manifest"],
        "role_labels": role_manifest,
    }
    query_texts = query_text_register(episodes)
    baseline_info = verify_baselines(
        config, prepared_inputs["root"],
        jobs=("first",) if job == "first" else JOBS,
        verify_source=verify_source)

    dev_decisions = build_readout_split_decisions(
        episodes, revocations, config, "dev")
    frozen_dev = freeze_decisions(dev_decisions, split="dev")
    raw_files = {"dev": _write_frozen_dev(output, frozen_dev)}
    _require(decision_key_register(frozen_dev.decisions)["unique"],
             "Dev decision keys are not unique")

    report = _base_report(
        config, job=job, config_path=config_path,
        prepared_info=prepared_info, baseline_info=baseline_info,
        native=native)
    report["raw_outputs"]["files"] = dict(raw_files)
    report["counts"] = {
        "decisions": frozen_dev.row_count,
        "by_phase": {
            phase: sum(1 for decision in frozen_dev.decisions
                       if decision["phase"] == phase)
            for phase in JOB_PHASES["first"] + JOB_PHASES["restart"]},
        "decision_keys": decision_key_register(frozen_dev.decisions),
    }

    first_receipt = None
    if job == "restart":
        _require(first_output is not None,
                 "The restart job requires the first-process output directory")
        first_receipt = _load_first_generations(
            first_output, config, frozen_dev.decisions, query_texts)
        protocol = first_receipt["report"].get("protocol") or {}
        if report["protocol"]["config_sha256"] is not None:
            _require(protocol.get("config_sha256")
                     == report["protocol"]["config_sha256"],
                     "The first-process receipt used a different configuration")
        report["gating"]["first_receipt"] = {
            "status": "verified",
            "report_path": first_receipt["report_path"],
            "report_sha256": first_receipt["report_sha256"],
            "first_process_gate_passed": True,
            "generations": len(first_receipt["generations"]),
            "decision_sha256":
                first_receipt["decision_manifest"]["sha256"],
        }
        report["gating"]["first_process"] = {
            "status": "evaluated_prior",
            "gate": first_receipt["report"]["gating"]["first_process"].get(
                "gate"),
            "metrics": first_receipt["report"]["gating"]["first_process"].get(
                "metrics"),
        }
        report["gating"]["stopped_before_restart"] = False

    dev_labels = load_role_labels(roles, "dev", config, opened=opened_roles)
    report["role_label_access"]["opened"] = list(opened_roles)
    literal_scored = score_literal_values(frozen_dev, dev_labels)
    literal_metrics = literal_scored["splits"]["dev"]
    literal_gate = evaluate_literal_gate(literal_metrics, config)
    counts = decision_counts(frozen_dev.decisions, dev_labels)
    validate_decision_counts(counts, config)
    report["counts"]["dev_denominators"] = counts
    report["gating"]["literal_dev"] = literal_gate
    report["literal_metrics"] = {"dev": literal_metrics}

    if not literal_gate["passed"]:
        report["gating"]["first_process"] = {
            "status": "not_run",
            "reason": "dev_literal_failed",
            "gate": None,
            "metrics": None,
        }
        report["gating"]["heldout"] = {
            "status": "not_run",
            "evaluated_after_dev_pass": False,
            "reason": "dev_literal_failed",
            "fields": 0,
            "generations": 0,
            "metrics": None,
        }
        report["gating"]["train"]["reason"] = "dev_literal_failed"
        report["gating"]["stopped_before_model"] = True
        report["gating"]["stopped_before_heldout"] = True
        report["outcome"] = {
            "status": "stopped_at_dev_literal",
            "literal_dev": "failed",
            "model": "not_run",
            "heldout": "not_run",
            "train": "not_run",
            "claim": "no model, tokenizer or generation evidence",
        }
        report["elapsed_seconds"] = clock() - started
        _finish_report(report, config, output, profiler)
        return report

    _require(callable(generation_factory),
             "A generation factory is required after the literal gate")
    generator = generation_factory(
        deadline=deadline, baseline_info=baseline_info)
    all_generation_rows = []
    try:
        for phase in JOB_PHASES[job]:
            phase_decisions = [decision for decision in frozen_dev.decisions
                               if decision["phase"] == phase]
            rows = tuple(generator(phase_decisions, query_texts))
            validate_generation_rows(rows, phase_decisions, query_texts)
            manifest = write_generation_rows(output, rows, phase=phase)
            raw_files[f"dev_{phase}"] = manifest
            all_generation_rows.extend(rows)
            metrics = score_phase(rows, dev_labels, frozen_dev.decisions,
                                  phase)
            gate = evaluate_phase_gate(metrics, config)
            report["gating"][phase] = {
                "status": "evaluated",
                "gate": gate,
                "metrics": metrics,
            }
            _require(clock() <= deadline,
                     "Generation exceeded the declared wall budget")
            if not gate["passed"]:
                report["gating"]["heldout"] = {
                    "status": "not_run",
                    "evaluated_after_dev_pass": False,
                    "reason": f"dev_{phase}_failed",
                    "fields": 0,
                    "generations": 0,
                    "metrics": None,
                }
                report["gating"]["train"]["reason"] = f"dev_{phase}_failed"
                report["gating"]["stopped_before_heldout"] = True
                report["gating"]["stopped_before_model"] = False
                report["raw_outputs"]["files"] = dict(raw_files)
                report["raw_outputs"]["verified_unchanged"] = all(
                    [_verify_decisions(output, frozen_dev, raw_files["dev"])]
                    + [_verify_generation_file(
                        output, rows, raw_files[f"dev_{phase}"])
                       for phase in JOB_PHASES[job]
                       if f"dev_{phase}" in raw_files])
                report["outcome"] = {
                    "status": f"stopped_at_dev_{phase}",
                    "literal_dev": "passed",
                    "model": f"{phase}_failed",
                    "heldout": "not_run",
                    "train": "not_run",
                    "claim": ("prompt v2 readout did not pass the "
                              "predeclared development gate; no heldout "
                              "evidence"),
                }
                report["classification"] = _classification(
                    config, native=native, model_executed=native)
                report["resource_usage"] = _resource_usage(
                    config, job=job, model_executed=native,
                    generation_rows=all_generation_rows)
                report["counts"]["generations"] = {
                    phase: len([row for row in all_generation_rows
                                if row["phase"] == phase])
                    for phase in JOB_PHASES[job]}
                report["elapsed_seconds"] = clock() - started
                _finish_report(report, config, output, profiler)
                return report

        if job == "first":
            report["gating"]["restart"]["status"] = "not_run"
            report["gating"]["revocation"]["status"] = "not_run"
            report["gating"]["state_denominators"] = {
                "status": "not_run",
                "reason": "restart_job_pending",
                "gate": None,
            }
            report["gating"]["restart_accounting"] = {
                "status": "not_run",
                "reason": "restart_job_pending",
                "gate": None,
            }
            report["gating"]["heldout"]["reason"] = "restart_job_pending"
            report["gating"]["train"]["reason"] = "restart_job_pending"
            report["gating"]["stopped_before_heldout"] = True
            report["gating"]["stopped_before_model"] = False
            report["raw_outputs"]["files"] = dict(raw_files)
            report["raw_outputs"]["verified_unchanged"] = all(
                [_verify_decisions(output, frozen_dev, raw_files["dev"])]
                + [_verify_generation_file(
                    output, tuple(row for row in all_generation_rows
                                  if row["phase"] == phase),
                    raw_files[f"dev_{phase}"])
                   for phase in JOB_PHASES[job]])
            report["classification"] = _classification(
                config, native=native, model_executed=native)
            report["resource_usage"] = _resource_usage(
                config, job=job, model_executed=native,
                generation_rows=all_generation_rows)
            report["counts"]["generations"] = {
                phase: len([row for row in all_generation_rows
                            if row["phase"] == phase])
                for phase in JOB_PHASES[job]}
            report["outcome"] = {
                "status": "first_process_passed_restart_pending",
                "literal_dev": "passed",
                "model": "first_process_passed",
                "heldout": "not_run",
                "train": "not_run",
                "claim": ("selected-record literal value and the first-process "
                          "prompt-v2 readout; restart and heldout pending"),
            }
            report["elapsed_seconds"] = clock() - started
            _finish_report(report, config, output, profiler)
            return report

        # Restart job: combined state gates, restart accounting, heldout.
        combined = tuple(first_receipt["generations"]) \
            + tuple(all_generation_rows)
        state_gate = evaluate_state_gates(
            combined, dev_labels, frozen_dev.decisions, config)
        report["gating"]["state_denominators"] = {
            "status": "evaluated",
            "gate": state_gate,
        }
        accounting = restart_query_accounting(episodes, split="dev")
        accounting["old_observations_encoded"] = int(
            baseline_info["restart"]["encoded"].get(
                "old_observations_reencoded", -1))
        accounting["baseline_restart_new_queries"] = int(
            baseline_info["restart"]["encoded"].get(
                "restart_new_queries", -1))
        accounting["old_first_process_queries_regenerated"] = sum(
            1 for row in all_generation_rows
            if row["phase"] == "first_process")
        accounting_gate = evaluate_restart_accounting(accounting, config)
        report["gating"]["restart_accounting"] = {
            "status": "evaluated",
            "gate": accounting_gate,
            "metrics": accounting,
        }
        report["gating"]["stopped_before_model"] = False
        if not (state_gate["passed"] and accounting_gate["passed"]):
            report["gating"]["heldout"] = {
                "status": "not_run",
                "evaluated_after_dev_pass": False,
                "reason": "state_or_accounting_gate_failed",
                "fields": 0,
                "generations": 0,
                "metrics": None,
            }
            report["gating"]["train"]["reason"] = \
                "state_or_accounting_gate_failed"
            report["gating"]["stopped_before_heldout"] = True
            report["raw_outputs"]["files"] = dict(raw_files)
            report["raw_outputs"]["verified_unchanged"] = all(
                [_verify_decisions(output, frozen_dev, raw_files["dev"])]
                + [_verify_generation_file(
                    output, tuple(row for row in all_generation_rows
                                  if row["phase"] == phase),
                    raw_files[f"dev_{phase}"])
                   for phase in JOB_PHASES[job]])
            report["classification"] = _classification(
                config, native=native, model_executed=native)
            report["resource_usage"] = _resource_usage(
                config, job=job, model_executed=native,
                generation_rows=all_generation_rows)
            report["counts"]["generations"] = {
                phase: len([row for row in all_generation_rows
                            if row["phase"] == phase])
                for phase in JOB_PHASES[job]}
            report["outcome"] = {
                "status": "stopped_at_dev_state_or_accounting",
                "literal_dev": "passed",
                "model": "restart_revocation_passed",
                "heldout": "not_run",
                "train": "not_run",
                "claim": ("development phase gates passed but the combined "
                          "state or restart-accounting gate failed; no "
                          "heldout evidence"),
            }
            report["elapsed_seconds"] = clock() - started
            _finish_report(report, config, output, profiler)
            return report

        # Heldout: freeze decisions, generate without labels, then open the
        # sealed heldout label role exactly once.
        heldout_decisions = build_readout_split_decisions(
            episodes, revocations, config, "heldout")
        frozen_heldout = freeze_decisions(
            heldout_decisions, split="heldout")
        raw_files["heldout"] = write_frozen_decisions(
            output, frozen_heldout)
        heldout_generations = tuple(
            generator(frozen_heldout.decisions, query_texts))
        validate_generation_rows(
            heldout_generations, frozen_heldout.decisions, query_texts)
        raw_files["heldout_generations"] = write_generation_rows(
            output, heldout_generations, phase="heldout")
        _require(clock() <= deadline,
                 "Heldout generation exceeded the wall budget")
        heldout_labels = load_role_labels(
            roles, "heldout", config, opened=opened_roles)
        report["role_label_access"]["opened"] = list(opened_roles)
        report["role_label_access"]["heldout_opened"] = True
        heldout_metrics = _heldout_metrics(
            heldout_generations, heldout_labels, heldout_decisions)
        heldout_gate = {
            "enforced": False,
            "thresholds_predeclared": True,
            "tuned": False,
            "retried": False,
            "single_shot": True,
            "metrics_role": "post_freeze_confirmation_only",
        }
        report["gating"]["heldout"] = {
            "status": "evaluated",
            "evaluated_after_dev_pass": True,
            "reason": None,
            "fields": 0,
            "decisions": frozen_heldout.row_count,
            "generations": len(heldout_generations),
            "metrics": heldout_metrics,
            "gate": heldout_gate,
            "raw_rows": raw_files["heldout_generations"]["sha256"],
        }

        train_decisions = build_readout_split_decisions(
            episodes, revocations, config, "train")
        frozen_train = freeze_decisions(train_decisions, split="train")
        raw_files["train"] = write_frozen_decisions(output, frozen_train)
        train_labels = load_role_labels(
            roles, "train", config, opened=opened_roles)
        report["role_label_access"]["opened"] = list(opened_roles)
        train_literal = score_literal_values(frozen_train, train_labels)
        report["gating"]["train"] = {
            "status": "descriptive_only",
            "role": "descriptive_only",
            "reason": "all_dev_gates_passed",
            "model_generation": "not_run_descriptive_only",
            "decisions": frozen_train.row_count,
            "metrics": train_literal["splits"]["train"],
        }
        report["gating"]["stopped_before_heldout"] = False
        report["raw_outputs"]["files"] = dict(raw_files)
        verified = _verify_decisions(output, frozen_dev, raw_files["dev"])
        for phase in JOB_PHASES[job]:
            verified = verified and _verify_generation_file(
                output, tuple(row for row in all_generation_rows
                              if row["phase"] == phase),
                raw_files[f"dev_{phase}"])
        verified = verified and _verify_decisions(
            output, frozen_heldout, raw_files["heldout"])
        verified = verified and _verify_generation_file(
            output, heldout_generations,
            raw_files["heldout_generations"])
        verified = verified and _verify_decisions(
            output, frozen_train, raw_files["train"])
        report["raw_outputs"]["verified_unchanged"] = verified
        report["classification"] = _classification(
            config, native=native, model_executed=native)
        report["resource_usage"] = _resource_usage(
            config, job=job, model_executed=native,
            generation_rows=tuple(all_generation_rows)
            + heldout_generations)
        report["counts"] = {
            "decisions": (frozen_dev.row_count + frozen_heldout.row_count
                          + frozen_train.row_count),
            "by_split": {
                "dev": frozen_dev.row_count,
                "heldout": frozen_heldout.row_count,
                "train": frozen_train.row_count,
            },
            "generations": {
                "dev": len(all_generation_rows),
                "heldout": len(heldout_generations),
                "train": 0,
            },
            "decision_keys": decision_key_register(
                frozen_dev.decisions + frozen_heldout.decisions
                + frozen_train.decisions),
        }
        report["outcome"] = {
            "status": "dev_passed_heldout_evaluated",
            "literal_dev": "passed",
            "model": "restart_revocation_passed",
            "heldout": "evaluated",
            "train": "descriptive_only",
            "claim": ("selected-record literal value and the prompt-v2 MiniCPM "
                      "readout in a first process and new post-restart query "
                      "forms; not a persistent-learning or calibrated claim"),
        }
        report["elapsed_seconds"] = clock() - started
        _finish_report(report, config, output, profiler)
        return report
    finally:
        _close_generator(generator)


def _verify_decisions(output, frozen, manifest):
    path = Path(output) / manifest["filename"]
    _require(path.is_file(), f"Frozen decision file is missing: {path.name}")
    _require(_sha_file(path) == frozen.sha256,
             f"Frozen decision file changed: {path.name}")
    return True

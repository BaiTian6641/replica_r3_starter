"""Selected-record literal-value scoring and a bounded MiniCPM readout.

This experiment follows the explicit-record address bank without changing it.
It reuses the bank's decisions verbatim and adds only the literal ``value`` of
the record each bank decision selected.  The bank never reads evaluation
labels while selecting; this module writes and freezes the development
decisions before it parses ``labels.json``.

The sequence is fixed and fail-closed:

1. verify the pinned prepared04 inputs and the native02 checkpoint bindings;
2. freeze and write the development decisions with their selected values;
3. only then load the evaluation labels and score literal value exactness;
4. only if the development literal endpoint passes, load MiniCPM once and
   generate a readout for the development decisions;
5. only if the development model endpoint passes, build and evaluate heldout.

A development literal failure means no model, tokenizer or generation path is
touched.  A development model failure means no heldout decision or generation
is produced.  Train is descriptive only after both development gates pass.

Generation is a model readout, not a persistent-learning proof, a calibrated
confidence claim, or a bank-plus-compressed hybrid.  The compressed-vector arm
is referenced from its existing v4 receipt and never combined with this run.
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
    _fact_block,
    _labels_map,
    _prepared_root,
    _rate,
    _require,
    _score_row,
    _selection_block,
    _sha_bytes,
    _sha_file,
    _sha_json,
    _sha_text,
    _utc_now,
    _write_new,
    build_split_decisions,
    decision_key_register,
    freeze_decisions,
    validate_config as _validate_address_config,
    write_frozen_decisions,
)
from .memory_causal_fresh import (
    CHECKPOINT_FORMAT as CAUSAL_CHECKPOINT_FORMAT,
    LEDGER_FORMAT as CAUSAL_LEDGER_FORMAT,
    RUN_FORMAT as CAUSAL_RUN_FORMAT,
    _unpack_cache,
    load_fresh_checkpoint,
)
from .memory_matched import SPLITS as MATCHED_SPLITS

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIG_FORMAT = "replica.memory_bank_value.config.v1"
LITERAL_FORMAT = "replica.memory_bank_value.literal.v1"
GENERATIONS_FORMAT = "replica.memory_bank_value.generations.v1"
PROMPT_FORMAT = "replica.memory_bank_value.prompt.v1"
REPORT_FORMAT = "replica.memory_bank_value.report.v1"

DEV_LITERAL_FACT_VALUE_MIN = 0.85
DEV_LITERAL_UNKNOWN_SELECTIONS_MAX = 0
DEV_LITERAL_REVOKED_SELECTIONS_MAX = 0
DEV_MODEL_EXACT_ANSWER_MIN = 0.85
DEV_MODEL_CORRECTION_MIN = 0.60
DEV_MODEL_UNAFFECTED_MIN = 0.70
DEV_MODEL_UNKNOWN_CORRECT_MIN = 0.90
DEV_MODEL_REVOKED_CORRECT_MIN = 0.90

MAX_DECISIONS_PER_SPLIT = 152
MAX_NEW_TOKENS = 8
MAX_PROMPT_TOKENS = 256
MAX_WALL_SECONDS = 1200
ARTIFACT_CAP_BYTES = 64 * 1024 * 1024
STOP_TOKEN_IDS = (1, 130073)

LITERAL_SPLIT_ROLES = ADDRESS_SPLIT_ROLES
PHASE_QUERY_FIELDS = {
    "first_process": "queries",
    "restart": "restart_queries",
    "revocation": "revocation_queries",
}
ABSTENTION_STATUSES = {
    "abstain_no_address": "unknown",
    "abstain_revoked_only_address": "revoked",
    "abstain_ambiguous_address": "ambiguous",
    "abstain_superseded_only_address": "superseded",
}
CLASSIFICATION = {
    "package": "selected_record_literal_value_and_model_readout",
    "native": "declared_by_caller",
    "model_execution": "only_after_dev_literal_pass",
    "tokenizer_load": "only_after_dev_literal_pass",
    "generation": "model_readout_only",
    "persistent_learning": False,
    "calibrated": False,
    "learned_selector": False,
    "bank_plus_compressed_arm": False,
    "compressed_value_comparison": "referenced_v4_only",
    "compressed_arm": "separate_reference_not_combined",
    "oracle": "diagnostic_only_not_executed",
    "checkpoint_state_role": "validated_only_not_consumed_by_generation",
    "answer_source": "sealed_minicpm5_greedy_readout",
}
LIMITATIONS = [
    "Literal value exactness is exact runtime-record string equality, not semantic or learned retrieval.",
    "The bank selects records; the literal value is read from the selected runtime record and is never a label input.",
    "Generation is a bounded greedy MiniCPM readout of a prompt that contains only the query, an authorized selected value and a runtime status.",
    "The generated-answer rates are readout evidence, not calibrated confidence, open-ended answering or a persistent-learning claim.",
    "No optimizer, gradient, checkpoint write, base export or extension export occurs; the native02 checkpoint is read only.",
    "The compressed-vector arm is referenced from its existing v4 receipt and is never combined with this bank arm.",
    "Unknown and revoked answers are evaluated separately and never enter the literal fact denominator.",
    "Heldout evidence is reported only after the predeclared development literal and model endpoints pass; no threshold is tuned.",
    "Train evidence is descriptive only and is produced only after both development gates pass.",
]


# ---------------------------------------------------------------------------
# Configuration and derived address-bank control view
# ---------------------------------------------------------------------------

def _address_control_view(config):
    """Project this protocol onto the frozen address-bank control contract.

    Reusing the bank's own validator and preparer here keeps its decision rule,
    prepared pins and roles literally identical; this module only adds the
    literal-value and readout layers on top.
    """
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


def validate_config(config):
    """Validate and freeze the protocol without mutating it."""
    _require(isinstance(config, Mapping), "Bank-value configuration must be an object")
    for name in ("format", "declared_at_utc", "purpose", "classification",
                 "prepared", "baseline", "bank", "reader", "endpoint",
                 "generation", "roles", "reporting", "budget",
                 "compressed_reference", "limitations"):
        _require(name in config, f"Bank-value configuration is missing: {name}")
    _require(config["format"] == CONFIG_FORMAT,
             "Bank-value configuration format differs")

    _validate_address_config(_address_control_view(config))
    _require(config["roles"] == LITERAL_SPLIT_ROLES,
             "Split roles differ from the frozen roles")
    _require(config["bank"].get("value_comparison") is True,
             "The literal-value comparison must be declared")

    endpoint = config["endpoint"]
    _require(endpoint.get("split") == "dev",
             "The endpoint split must be dev")
    _require(endpoint.get("dev_literal_fact_value_min")
             == DEV_LITERAL_FACT_VALUE_MIN,
             "Dev literal value threshold may not be tuned")
    _require(type(endpoint.get("dev_literal_unknown_selections_max")) is int
             and endpoint["dev_literal_unknown_selections_max"]
             == DEV_LITERAL_UNKNOWN_SELECTIONS_MAX,
             "Dev literal unknown selection threshold may not be tuned")
    _require(type(endpoint.get("dev_literal_revoked_selections_max")) is int
             and endpoint["dev_literal_revoked_selections_max"]
             == DEV_LITERAL_REVOKED_SELECTIONS_MAX,
             "Dev literal revoked selection threshold may not be tuned")
    _require(endpoint.get("dev_model_exact_answer_min")
             == DEV_MODEL_EXACT_ANSWER_MIN,
             "Dev model answer threshold may not be tuned")
    _require(endpoint.get("dev_model_correction_min")
             == DEV_MODEL_CORRECTION_MIN,
             "Dev model correction threshold may not be tuned")
    _require(endpoint.get("dev_model_unaffected_min")
             == DEV_MODEL_UNAFFECTED_MIN,
             "Dev model unaffected threshold may not be tuned")
    _require(endpoint.get("dev_model_unknown_correct_min")
             == DEV_MODEL_UNKNOWN_CORRECT_MIN,
             "Dev model unknown threshold may not be tuned")
    _require(endpoint.get("dev_model_revoked_correct_min")
             == DEV_MODEL_REVOKED_CORRECT_MIN,
             "Dev model revoked threshold may not be tuned")
    _require(endpoint.get("heldout") == "evaluate_after_dev_pass_only",
             "Heldout must be evaluated only after dev passes")
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
    _require(generation.get("wall_budget_seconds") == MAX_WALL_SECONDS,
             "Wall budget may not be tuned")
    _require(generation.get("artifact_cap_bytes") == ARTIFACT_CAP_BYTES,
             "Artifact cap may not be tuned")
    _require(tuple(generation.get("stop_token_ids", ())) == STOP_TOKEN_IDS,
             "Generation stop token set differs")
    _require(tuple(generation.get("enabled_ports", ())) == (),
             "Generation must enable no ports")
    _require(tuple(generation.get("enabled_branches", ())) == (),
             "Generation must enable no record-memory or workspace branch")
    _require(generation.get("core_impl") == "none",
             "Generation must use core_impl='none'")
    _require(generation.get("training") is False,
             "Generation must not train")
    _require(generation.get("record_memory") is False
             and generation.get("workspace") is False,
             "Generation must not use record memory or a workspace")
    _require(generation.get("answer_normalization")
             == "strip+collapse_whitespace+casefold",
             "Answer normalization differs")
    _require(generation.get("model_loads") == 1,
             "Exactly one model load is allowed")

    baseline = config["baseline"]
    for key in ("checkpoint_sha256", "report_sha256", "ledger_sha256",
                "protocol_sha256"):
        _require(isinstance(baseline.get(key), str)
                 and len(baseline[key]) == 64,
                 f"Baseline pin is invalid: {key}")
    _require(baseline.get("report_format") == CAUSAL_RUN_FORMAT,
             "Baseline report format differs")
    _require(baseline.get("checkpoint_format") == CAUSAL_CHECKPOINT_FORMAT,
             "Baseline checkpoint format differs")
    _require(baseline.get("ledger_format") == CAUSAL_LEDGER_FORMAT,
             "Baseline ledger format differs")
    _require(baseline.get("phase") == "first_process",
             "Baseline phase must be first_process")
    _require(baseline.get("resumed") is False,
             "Baseline must be a fresh first process")
    _require(baseline.get("episodes") == 12,
             "Baseline episode count differs")
    _require(dict(baseline.get("split_episode_counts") or {})
             == {"train": 4, "dev": 4, "heldout": 4},
             "Baseline split episode counts differ")

    budget = config["budget"]
    _require(budget.get("model_loads") == 1,
             "The protocol allows exactly one model load")
    _require(budget.get("tokenizer_loads") == 1,
             "The protocol allows exactly one tokenizer load")
    for key in ("optimizer_updates", "gradients", "checkpoints_written",
                "base_export", "extension_export", "downloads"):
        _require(type(budget.get(key)) is int and budget[key] == 0,
                 f"Forbidden resource budget must be zero: {key}")
    _require(budget.get("native_jobs") == 1,
             "Exactly one native job is allowed")
    _require(budget.get("fallback") == "disabled",
             "CPU/MPS fallback must be disabled")

    reference = config["compressed_reference"]
    _require(reference.get("role") == "referenced_only_not_combined",
             "Compressed reference role differs")
    _require(isinstance(reference.get("path"), str)
             and isinstance(reference.get("sha256"), str)
             and len(reference["sha256"]) == 64,
             "Compressed reference pin is invalid")
    _require(isinstance(config["limitations"], Sequence)
             and config["limitations"], "Limitations are required")
    return config


# ---------------------------------------------------------------------------
# Prepared inputs and baseline checkpoint binding
# ---------------------------------------------------------------------------

def _resolve(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_json(path):
    return json.loads(Path(path).read_text())


def _prepared_documents(prepared):
    prepared = Path(prepared)
    names = {
        "episodes": "episodes.json",
        "revocations": "revocations.json",
        "labels": "labels.json",
        "inventory": "inventory.json",
        "manifest": "manifest.json",
        "config": "config.json",
        "protocol": "protocol.json",
    }
    documents = {}
    for key, name in names.items():
        path = prepared / name
        _require(path.is_file(), f"Prepared artifact is missing: {name}")
        documents[key] = {"path": path, "sha256": _sha_file(path)}
    return documents


def _verify_prepared_against_report(prepared, config, report):
    documents = _prepared_documents(prepared)
    pins = config["prepared"]["pins"]
    for key in ("episodes", "revocations", "labels", "manifest"):
        _require(documents[key]["sha256"] == pins[f"{key}_sha256"],
                 f"Prepared {key} bytes differ from the pinned input")
    generated = report.get("prepared") or {}
    for key in ("episodes", "revocations", "labels", "inventory"):
        _require(generated.get(f"{key}_sha256")
                 == documents[key]["sha256"],
                 f"Baseline report {key} hash differs from prepared04")
    _require(generated.get("config_sha256") == documents["config"]["sha256"],
             "Baseline report configuration hash differs from prepared04")
    _require(generated.get("protocol_path")
             == _display(documents["protocol"]["path"]),
             "Baseline report protocol path differs from prepared04")
    _require(report.get("protocol_sha256") == documents["protocol"]["sha256"],
             "Baseline protocol hash differs from prepared04")
    _require(report.get("protocol_sha256") == config["baseline"]["protocol_sha256"],
             "Baseline protocol pin differs")
    manifest = _read_json(documents["manifest"]["path"])
    _require(manifest.get("format") == config["prepared"]["formats"]["manifest"],
             "Prepared manifest format differs")
    for key in ("episodes_sha256", "revocations_sha256", "labels_sha256"):
        _require(manifest.get(key) == pins[key],
                 f"Prepared manifest {key} differs from the pinned input")
    return {
        "root": _display(Path(prepared)),
        "hashes": {f"{key}_sha256": item["sha256"]
                   for key, item in documents.items()},
        "protocol_sha256": documents["protocol"]["sha256"],
        "config_sha256": documents["config"]["sha256"],
    }


def _display(path):
    path = Path(path).resolve()
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _verify_sealed_source(prepared, report):
    protocol = _read_json(Path(prepared) / "protocol.json")
    from .architecture.contracts import fingerprint
    from .memory_diagnosis import _extension_pins, _source_pins
    source = protocol["source"]
    pins = _source_pins(_resolve(source["model_path"]),
                        _resolve(source["model_lock"]))
    lock = _read_json(_resolve(source["model_lock"]))
    snapshot = lock["snapshot_files_sha256"]
    pins["tokenizer_sha256"] = fingerprint({
        name: digest for name, digest in snapshot.items()
        if "tokenizer" in name.lower()
        or name == "special_tokens_map.json"})
    _require(pins == report["source_pins"],
             "Sealed source or tokenizer bytes differ from the baseline")
    writer = _extension_pins(_resolve(protocol["writer_extension"]["path"]))
    _require(writer == report["writer_pins"],
             "Frozen writer bytes differ from the baseline")
    return {"source_pins": pins, "writer_pins": writer}


def verify_baseline(config, *, prepared=None, device="cpu",
                    verify_source=True):
    """Read-only verification of the native02 checkpoint and prepared04 pins.

    The checkpoint is never written.  ``load_fresh_checkpoint`` verifies every
    declared binding against the baseline report, and ``_unpack_cache``
    validates each cached episode's records, live view, negatives and
    observation digests against prepared04 in this fresh process.
    """
    validate_config(config)
    baseline = config["baseline"]
    prepared_root = _prepared_root(prepared, config)

    report_path = _resolve(baseline["report"])
    checkpoint_path = _resolve(baseline["checkpoint"])
    ledger_path = _resolve(baseline["ledger"])
    _require(report_path.is_file(), "Baseline report is missing")
    _require(checkpoint_path.is_file(), "Baseline checkpoint is missing")
    _require(ledger_path.is_file(), "Baseline ledger is missing")
    _require(_sha_file(report_path) == baseline["report_sha256"],
             "Baseline report bytes differ from the pinned hash")
    _require(_sha_file(checkpoint_path) == baseline["checkpoint_sha256"],
             "Baseline checkpoint bytes differ from the pinned hash")
    _require(_sha_file(ledger_path) == baseline["ledger_sha256"],
             "Baseline ledger bytes differ from the pinned hash")

    report = _read_json(report_path)
    _require(report.get("format") == baseline["report_format"],
             "Baseline report format differs")
    _require(report.get("phase") == baseline["phase"],
             "Baseline phase differs")
    _require(report.get("resumed") is baseline["resumed"],
             "Baseline resumed flag differs")
    checkpoint = report.get("checkpoint") or {}
    _require(checkpoint.get("sha256") == baseline["checkpoint_sha256"],
             "Baseline report checkpoint hash differs")
    _require(checkpoint.get("bytes") == checkpoint_path.stat().st_size,
             "Baseline checkpoint byte count differs")
    _require(report.get("exactly_once", {}).get("ledger_sha256")
             == baseline["ledger_sha256"],
             "Baseline report ledger hash differs")

    prepared_info = _verify_prepared_against_report(
        prepared_root, config, report)
    payload = load_fresh_checkpoint(
        checkpoint_path, expected=report["bindings"], device=device)
    _require(payload.get("format") == baseline["checkpoint_format"],
             "Checkpoint format differs")
    _require(payload.get("phase") == baseline["phase"],
             "Checkpoint phase differs")
    ledger = _read_json(ledger_path)
    _require(payload.get("ledger") == ledger,
             "Checkpoint ledger differs from ledger.json")
    _require(payload["ledger"].get("format") == baseline["ledger_format"],
             "Checkpoint ledger format differs")
    encoded = report.get("encoded") or {}
    counters = payload.get("counters") or {}
    _require(counters.get("observation_encodes") == encoded.get("observations"),
             "Checkpoint observation counter differs")
    _require(counters.get("query_encodes") == encoded.get("queries"),
             "Checkpoint query counter differs")

    episodes = _read_json(prepared_root / "episodes.json")["episodes"]
    by_id = {}
    for episode in episodes:
        _require(episode["episode_id"] not in by_id,
                 "Prepared episode IDs repeat")
        by_id[episode["episode_id"]] = episode
    packed_episodes = payload.get("episodes")
    _require(isinstance(packed_episodes, list)
             and len(packed_episodes) == baseline["episodes"],
             "Checkpoint episode count differs")
    split_counts = {}
    cached = []
    seen = set()
    for packed in packed_episodes:
        episode_id = packed.get("episode_id")
        _require(episode_id in by_id,
                 "Checkpoint names an unknown prepared episode")
        _require(episode_id not in seen, "Checkpoint episode IDs repeat")
        seen.add(episode_id)
        episode = by_id[episode_id]
        cache = _unpack_cache(packed, episode, device)
        split_counts[episode["split"]] = (
            split_counts.get(episode["split"], 0) + 1)
        cached.append({
            "episode_id": episode_id,
            "split": episode["split"],
            "variant": episode["variant"],
            "layers": list(cache.layers),
            "states": sorted(cache.states),
            "writes": len(cache.writes),
            "entry_vectors": len(cache.entry_vectors),
            "query_vectors": len(cache.query_vectors),
            "observation_sha256": {
                name: item["observation_sha256"]
                for name, item in sorted(cache.states.items())},
        })
    _require(split_counts == dict(baseline["split_episode_counts"]),
             "Checkpoint split episode counts differ")
    _require(seen == set(by_id),
             "Checkpoint does not cache every prepared episode")
    source = _verify_sealed_source(prepared_root, report) \
        if verify_source else None
    return {
        "mode": "pinned_native02_checkpoint_read_only",
        "report": {"path": _display(report_path),
                   "sha256": baseline["report_sha256"]},
        "checkpoint": {"path": _display(checkpoint_path),
                       "sha256": baseline["checkpoint_sha256"],
                       "bytes": checkpoint_path.stat().st_size,
                       "loaded": True,
                       "written": False},
        "ledger": {"path": _display(ledger_path),
                   "sha256": baseline["ledger_sha256"],
                   "commits": len(ledger.get("commits", ())),
                   "attempts": len(ledger.get("attempts", ()))},
        "protocol_sha256": report["protocol_sha256"],
        "bindings_sha256": _sha_json(payload["bindings"]),
        "bindings_verified": True,
        "cached_state_verified": True,
        "cached_episodes": cached,
        "phase": payload["phase"],
        "counters": dict(counters),
        "prepared": prepared_info,
        "source": source,
    }


# ---------------------------------------------------------------------------
# Literal selected-record values
# ---------------------------------------------------------------------------

def _episode_value_index(episodes):
    values = {}
    for episode in episodes:
        for group in ("facts", "distractors", "corrections"):
            for record in episode.get(group, ()):
                record_id = record.get("record_id")
                value = record.get("value")
                _require(isinstance(record_id, str) and record_id,
                         "Episode record ID is missing")
                _require(isinstance(value, str) and value,
                         "Episode record value is missing")
                if record_id in values:
                    _require(values[record_id] == value,
                             f"Episode records conflict on value: {record_id}")
                    continue
                values[record_id] = value
    return values


def annotate_decision_values(episodes, decisions):
    """Attach the selected runtime record's literal value to frozen decisions.

    Values come only from the runtime episode records.  Evaluation labels are
    not accepted by this function.
    """
    values = _episode_value_index(episodes)
    annotated = []
    for decision in decisions:
        selected_ids = tuple(decision.get("selected_ids") or ())
        _require(len(selected_ids) <= 1,
                 "A bank decision selected more than one record")
        selected_value = None
        if selected_ids:
            record_id = selected_ids[0]
            _require(record_id in values,
                     "Selected runtime record has no literal value")
            selected_value = values[record_id]
        annotated.append({
            **dict(decision),
            "literal_format": LITERAL_FORMAT,
            "selected_value": selected_value,
        })
    return tuple(annotated)


def build_value_split_decisions(episodes, revocations, config, split):
    """Reuse the bank decision rule and add the runtime selected value."""
    validate_config(config)
    _require(split in MATCHED_SPLITS, f"Unknown split: {split}")
    selected = [episode for episode in episodes
                if episode.get("split") == split]
    decisions = build_split_decisions(
        episodes, revocations, _address_control_view(config), split)
    _require(len(decisions) <= config["generation"]["max_decisions_per_split"],
             f"Split exceeds the declared decision cap: {split}")
    annotated = annotate_decision_values(selected, decisions)
    _require(len({(row["episode_id"], row["state"], row["phase"],
                   row["query_id"]) for row in annotated}) == len(annotated),
             "Annotated decision keys are not unique")
    return annotated


def query_text_register(episodes):
    """Map ``(episode, phase, query)`` to the exact runtime query text."""
    texts = {}
    for episode in episodes:
        episode_id = episode["episode_id"]
        for phase, field in PHASE_QUERY_FIELDS.items():
            for query in episode.get(field, ()):
                key = (episode_id, phase, query["query_id"])
                _require(key not in texts, "Runtime query keys repeat")
                text = query.get("text")
                _require(isinstance(text, str) and text.strip(),
                         "Runtime query text is missing")
                texts[key] = text
    return texts


def _query_key(decision):
    return (decision["episode_id"], decision["phase"],
            decision["query_id"])


def _decision_key(decision):
    """Full decision identity: state is part of every frozen decision."""
    return (decision["episode_id"], decision["state"],
            decision["phase"], decision["query_id"])


def decision_status(decision):
    """Runtime abstention/revocation status; ``None`` for an authorized value."""
    if tuple(decision.get("selected_ids") or ()):
        return None
    reason = decision.get("reason")
    _require(reason in ABSTENTION_STATUSES,
             f"Unknown abstention reason: {reason}")
    return ABSTENTION_STATUSES[reason]


def build_prompt(decision, query_text):
    """Build the exact prompt from a decision and its runtime query text only.

    No label object, expected answer or evaluator field is accepted here.
    """
    _require(isinstance(query_text, str) and query_text.strip(),
             "Query text is missing")
    status = decision_status(decision)
    lines = [f"Question: {query_text.strip()}"]
    if status is None:
        value = decision.get("selected_value")
        _require(isinstance(value, str) and value,
                 "An authorized selection lacks its runtime value")
        lines.append(f"Record value: {value}")
    else:
        _require(decision.get("selected_value") is None,
                 "An abstained decision carries a literal value")
        lines.append(f"Record status: {status}")
    lines.append("Answer:")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Literal scoring and the development literal endpoint
# ---------------------------------------------------------------------------

def _target(labels_by_episode, decision):
    episode_id = decision["episode_id"]
    state = decision["state"]
    query_id = decision["query_id"]
    _require(episode_id in labels_by_episode,
             f"Labels are missing for episode: {episode_id}")
    state_labels = labels_by_episode[episode_id].get(state)
    _require(state_labels is not None,
             f"Labels are missing for state: {episode_id}/{state}")
    targets = state_labels.get("targets") or {}
    _require(query_id in targets,
             f"Labels are missing for query: {episode_id}/{state}/{query_id}")
    return state_labels, targets[query_id]


def _literal_row(decision, labels_by_episode):
    base = _score_row(decision, labels_by_episode)
    state_labels, target = _target(labels_by_episode, decision)
    expected = target.get("expected")
    _require(isinstance(expected, str) and expected,
             "Evaluation target lacks an expected string")
    selected_value = decision.get("selected_value")
    if base["selected_any"]:
        _require(isinstance(selected_value, str) and selected_value,
                 "Selected decision lacks its runtime literal value")
    else:
        _require(selected_value is None,
                 "Abstained decision carries a literal value")
    value_exact = bool(
        base["kind"] == "fact" and base["fact_identity"]
        and selected_value == expected)
    return {
        **base,
        "expected": expected,
        "selected_value": selected_value,
        "value_exact": value_exact,
        "value_selected_wrong": bool(
            base["kind"] == "fact" and base["selected_any"]
            and not value_exact),
        "value_abstained_wrong": bool(
            base["kind"] == "fact" and not base["selected_any"]),
        "in_correction": decision["query_id"]
        in set(state_labels.get("corrected_query_ids") or ()),
        "in_unaffected": decision["query_id"]
        in set(state_labels.get("unaffected_query_ids") or ()),
    }


def _value_fact_block(rows):
    block = _fact_block(rows)
    exact = sum(row["value_exact"] for row in rows)
    mismatch = sum(row["value_selected_wrong"] for row in rows)
    block.update({
        "fact_n": len(rows),
        "fact_selected_n": sum(row["selected_any"] for row in rows),
        "value_exact_n": exact,
        "value_exact_rate": _rate(exact, len(rows)),
        "value_selected_wrong_n": mismatch,
        "value_abstained_wrong_n": sum(row["value_abstained_wrong"]
                                       for row in rows),
        "value_missing_n": sum(
            row["selected_any"] and not row["selected_value"] for row in rows),
    })
    return block


def _value_selection_block(rows):
    block = _selection_block(rows)
    facts = [row for row in rows if row["kind"] == "fact"]
    block.update({
        "fact_value_exact_n": sum(row["value_exact"] for row in facts),
        "fact_value_exact_rate": _rate(
            sum(row["value_exact"] for row in facts), len(facts)),
        "fact_value_selected_wrong_n": sum(
            row["value_selected_wrong"] for row in facts),
        "fact_value_abstained_n": sum(
            row["value_abstained_wrong"] for row in facts),
        "correction": _value_fact_block(
            [row for row in rows if row["in_correction"]]),
        "unaffected": _value_fact_block(
            [row for row in rows if row["in_unaffected"]]),
    })
    return block


def _value_group(rows, key):
    groups = {}
    for row in rows:
        value = row.get(key)
        groups.setdefault("unknown" if value is None else str(value),
                          []).append(row)
    return {name: _value_selection_block(groups[name])
            for name in sorted(groups)}


def _literal_split_metrics(rows):
    return {
        "overall": _value_selection_block(rows),
        "by_phase": _value_group(rows, "phase"),
        "by_state": _value_group(rows, "state"),
        "by_cohort": _value_group(rows, "cohort"),
        "by_world": _value_group(rows, "world_id"),
        "by_complexity": _value_group(rows, "complexity"),
        "counts": {
            "decisions": len(rows),
            "episodes": len({row["episode_id"] for row in rows}),
            "worlds": len({row["world_id"] for row in rows}),
            "states": len({(row["episode_id"], row["state"]) for row in rows}),
        },
    }


def score_literal_values(decisions, labels, *, config=None):
    """Join frozen selected values to evaluation labels after the freeze."""
    rows = tuple(getattr(decisions, "decisions", decisions))
    _require(rows, "No decisions are available for literal scoring")
    labels_by_episode = _labels_map(labels)
    scored = []
    seen = set()
    for decision in rows:
        key = (decision["episode_id"], decision["state"],
               decision["phase"], decision["query_id"])
        _require(key not in seen, "Decision keys repeat during literal scoring")
        seen.add(key)
        scored.append(_literal_row(decision, labels_by_episode))
    splits = {}
    for split in SPLITS:
        subset = [row for row in scored if row["split"] == split]
        if subset:
            splits[split] = _literal_split_metrics(subset)
    _require(splits, "No declared split was scored")
    aggregate = {
        "overall": _value_selection_block(scored),
        "by_split": {split: _value_selection_block(
            [row for row in scored if row["split"] == split])
            for split in splits},
        "by_cohort": _value_group(scored, "cohort"),
        "by_world": _value_group(scored, "world_id"),
    }
    return {
        "splits": splits,
        "aggregate": aggregate,
        "decision_rows": len(scored),
        "splits_present": list(splits),
        "note": ("Value exactness is exact runtime selected-record string "
                 "equality against evaluation labels; labels never select."),
    }


def evaluate_dev_literal_gate(dev_metrics, endpoint=None):
    """Apply the predeclared literal endpoint using dev evidence only."""
    overall = dev_metrics.get("overall") if "overall" in dev_metrics \
        else dev_metrics
    thresholds = endpoint or {
        "dev_literal_fact_value_min": DEV_LITERAL_FACT_VALUE_MIN,
        "dev_literal_unknown_selections_max":
            DEV_LITERAL_UNKNOWN_SELECTIONS_MAX,
        "dev_literal_revoked_selections_max":
            DEV_LITERAL_REVOKED_SELECTIONS_MAX,
    }
    _require(thresholds.get("dev_literal_fact_value_min")
             == DEV_LITERAL_FACT_VALUE_MIN,
             "Dev literal value threshold may not be tuned")
    _require(thresholds.get("dev_literal_unknown_selections_max")
             == DEV_LITERAL_UNKNOWN_SELECTIONS_MAX,
             "Dev literal unknown threshold may not be tuned")
    _require(thresholds.get("dev_literal_revoked_selections_max")
             == DEV_LITERAL_REVOKED_SELECTIONS_MAX,
             "Dev literal revoked threshold may not be tuned")
    rate = overall.get("fact_value_exact_rate")
    unknown_selected = overall.get("unknown_selected_n")
    revoked_selected = overall.get("revoked_selected_n")
    checks = {
        "fact_value_exact": {
            "value": rate,
            "operator": ">=",
            "threshold": DEV_LITERAL_FACT_VALUE_MIN,
            "passed": rate is not None and rate >= DEV_LITERAL_FACT_VALUE_MIN,
            "value_exact_n": overall.get("fact_value_exact_n"),
            "fact_n": overall.get("fact_n"),
        },
        "unknown_selections": {
            "value": unknown_selected,
            "operator": "<=",
            "threshold": DEV_LITERAL_UNKNOWN_SELECTIONS_MAX,
            "passed": unknown_selected is not None
            and unknown_selected <= DEV_LITERAL_UNKNOWN_SELECTIONS_MAX,
        },
        "revoked_selections": {
            "value": revoked_selected,
            "operator": "<=",
            "threshold": DEV_LITERAL_REVOKED_SELECTIONS_MAX,
            "passed": revoked_selected is not None
            and revoked_selected <= DEV_LITERAL_REVOKED_SELECTIONS_MAX,
        },
    }
    return {
        "evaluated_split": "dev",
        "evaluated_first": True,
        "passed": all(check["passed"] for check in checks.values()),
        "checks": checks,
    }


# ---------------------------------------------------------------------------
# Bounded greedy generation
# ---------------------------------------------------------------------------

def normalize_answer(text):
    """Declared answer normalization: strip, collapse whitespace, casefold."""
    _require(isinstance(text, str), "Answer text must be a string")
    return " ".join(text.split()).casefold()


def generation_row(decision, query_text, *, prompt_sha256, prompt_ids_sha256,
                   prompt_tokens, generated_ids, generated_text,
                   max_new_tokens, ended_by_eos=None):
    """Build one immutable generation row without any evaluation field."""
    prompt = build_prompt(decision, query_text)
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
    return {
        "format": GENERATIONS_FORMAT,
        "generation_id": f"generation-{decision['decision_id']}",
        "decision_id": decision["decision_id"],
        "episode_id": decision["episode_id"],
        "world_id": decision["world_id"],
        "split": decision["split"],
        "state": decision["state"],
        "phase": decision["phase"],
        "query_id": decision["query_id"],
        "prompt_format": PROMPT_FORMAT,
        "prompt": prompt,
        "prompt_sha256": prompt_sha256,
        "prompt_ids_sha256": prompt_ids_sha256,
        "prompt_tokens": int(prompt_tokens),
        "selected_record_id": decision.get("selected_record_id"),
        "selected_value": decision.get("selected_value"),
        "runtime_status": decision_status(decision),
        "generated_ids": ids,
        "generated_text": generated_text,
        "normalized_answer": normalize_answer(generated_text),
        "ended_by_eos": derived_eos,
        "stop_token_ids": list(STOP_TOKEN_IDS),
        "max_new_tokens": max_new_tokens,
        "model_readout": True,
    }


def generate_row(session, tokenizer, decision, query_text, *,
                 maximum_prompt=MAX_PROMPT_TOKENS,
                 maximum_new=MAX_NEW_TOKENS):
    """Generate one readout using the native ``generate_row`` convention.

    The request enables no port and no record-memory/workspace branch and the
    session ledger is empty, so the prompt is the only observable input.
    """
    import torch
    from .architecture.contracts import TaskGraph
    from .architecture.loading import encode_prompt
    from .memory_system.ledger import EligibilityScope

    prompt = build_prompt(decision, query_text)
    ids = encode_prompt(tokenizer, prompt, session.device)
    _require(ids.shape[1] <= maximum_prompt,
             f"Prompt exceeds the declared budget: {ids.shape[1]}")
    mask = torch.ones_like(ids, dtype=torch.bool)
    task = TaskGraph(task_id=f"bank-value-{decision['decision_id']}",
                     goal=prompt)
    scope = EligibilityScope(
        subject_id="bank-value", branch_id="literal-readout",
        purpose="selected_record_readout", as_of=0)
    request = session.prepare(
        prompt_ids=ids, prompt_mask=mask, query=prompt, scope=scope,
        task=task, training=False, visual=None, enabled_ports=(),
        core_impl="none", enabled_branches=())
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
    return generation_row(
        decision, query_text,
        prompt_sha256=_sha_text(prompt),
        prompt_ids_sha256=_sha_json([int(value) for value in ids[0].tolist()]),
        prompt_tokens=int(ids.shape[1]),
        generated_ids=generated, generated_text=text,
        max_new_tokens=maximum_new)


def build_model_generator(config, baseline_info, *, profiler=None,
                          deadline=None, device="mps"):
    """Load the sealed MiniCPM source once and return a generation callable."""
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
            key = _query_key(decision)
            _require(key in query_texts,
                     f"Runtime query text is missing: {key}")
            rows.append(generate_row(
                session, tokenizer, decision, query_texts[key],
                maximum_prompt=maximum_prompt, maximum_new=maximum_new))
        return tuple(rows)

    def close():
        ledger.close()

    generate.close = close
    return generate


# ---------------------------------------------------------------------------
# Generation scoring and the development model endpoint
# ---------------------------------------------------------------------------

def _answer_block(rows):
    total = len(rows)
    correct = sum(row["correct"] for row in rows)
    return {
        "n": total,
        "correct_n": correct,
        "exact_answer_rate": _rate(correct, total),
        "ended_by_eos_n": sum(row["ended_by_eos"] for row in rows),
        "empty_answer_n": sum(row["empty_answer"] for row in rows),
        "answer_tokens": sum(row["answer_tokens"] for row in rows),
    }


def _answer_group(rows, key):
    groups = {}
    for row in rows:
        value = row.get(key)
        groups.setdefault("unknown" if value is None else str(value),
                          []).append(row)
    return {name: _answer_block(groups[name]) for name in sorted(groups)}


def _generation_row(generation, labels_by_episode):
    decision_key = {
        "episode_id": generation["episode_id"],
        "state": generation["state"],
        "phase": generation["phase"],
        "query_id": generation["query_id"],
    }
    state_labels, target = _target(labels_by_episode, decision_key)
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
    _require(isinstance(cohort, str) and cohort,
             "Scoring cohort is missing")
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
        "correct": answer == normalize_answer(expected),
        "ended_by_eos": bool(generation["ended_by_eos"]),
        "empty_answer": answer == "",
        "answer_tokens": len(generation.get("generated_ids") or ()),
        "in_correction": generation["query_id"]
        in set(state_labels.get("corrected_query_ids") or ()),
        "in_unaffected": generation["query_id"]
        in set(state_labels.get("unaffected_query_ids") or ()),
    }


def _generation_split_metrics(rows):
    return {
        "overall": _answer_block(rows),
        "by_phase": _answer_group(rows, "phase"),
        "by_state": _answer_group(rows, "state"),
        "by_cohort": _answer_group(rows, "cohort"),
        "by_world": _answer_group(rows, "world_id"),
        "correction": _answer_block(
            [row for row in rows if row["in_correction"]]),
        "unaffected": _answer_block(
            [row for row in rows if row["in_unaffected"]]),
        "unknown": _answer_block(
            [row for row in rows if row["kind"] == "unknown"]),
        "revoked": _answer_block(
            [row for row in rows if row["kind"] == "revoked"]),
        "counts": {
            "decisions": len(rows),
            "episodes": len({row["episode_id"] for row in rows}),
            "worlds": len({row["world_id"] for row in rows}),
        },
    }


def score_generations(generations, labels, *, config=None):
    """Join frozen generation rows to evaluation labels after generation."""
    rows = tuple(generations)
    _require(rows, "No generations are available for scoring")
    labels_by_episode = _labels_map(labels)
    scored = []
    seen = set()
    for generation in rows:
        key = (generation["episode_id"], generation["state"],
               generation["phase"], generation["query_id"])
        _require(key not in seen, "Generation keys repeat during scoring")
        seen.add(key)
        scored.append(_generation_row(generation, labels_by_episode))
    splits = {}
    for split in SPLITS:
        subset = [row for row in scored if row["split"] == split]
        if subset:
            splits[split] = _generation_split_metrics(subset)
    _require(splits, "No declared split was scored")
    return {
        "splits": splits,
        "aggregate": {
            "overall": _answer_block(scored),
            "by_split": {split: _answer_block(
                [row for row in scored if row["split"] == split])
                for split in splits},
        },
        "generation_rows": len(scored),
        "splits_present": list(splits),
    }


def evaluate_dev_model_gate(dev_metrics, endpoint=None):
    """Apply the predeclared model endpoint using dev readout only."""
    thresholds = endpoint or {
        "dev_model_exact_answer_min": DEV_MODEL_EXACT_ANSWER_MIN,
        "dev_model_correction_min": DEV_MODEL_CORRECTION_MIN,
        "dev_model_unaffected_min": DEV_MODEL_UNAFFECTED_MIN,
        "dev_model_unknown_correct_min": DEV_MODEL_UNKNOWN_CORRECT_MIN,
        "dev_model_revoked_correct_min": DEV_MODEL_REVOKED_CORRECT_MIN,
    }
    pinned = {
        "dev_model_exact_answer_min": DEV_MODEL_EXACT_ANSWER_MIN,
        "dev_model_correction_min": DEV_MODEL_CORRECTION_MIN,
        "dev_model_unaffected_min": DEV_MODEL_UNAFFECTED_MIN,
        "dev_model_unknown_correct_min": DEV_MODEL_UNKNOWN_CORRECT_MIN,
        "dev_model_revoked_correct_min": DEV_MODEL_REVOKED_CORRECT_MIN,
    }
    for name, value in pinned.items():
        _require(thresholds.get(name) == value,
                 f"{name} may not be tuned")
    overall = dev_metrics.get("overall") or {}
    blocks = {
        "exact_answer": (overall, "exact_answer_rate",
                         DEV_MODEL_EXACT_ANSWER_MIN),
        "correction": (dev_metrics.get("correction") or {},
                       "exact_answer_rate", DEV_MODEL_CORRECTION_MIN),
        "unaffected": (dev_metrics.get("unaffected") or {},
                       "exact_answer_rate", DEV_MODEL_UNAFFECTED_MIN),
        "unknown": (dev_metrics.get("unknown") or {},
                    "exact_answer_rate", DEV_MODEL_UNKNOWN_CORRECT_MIN),
        "revoked": (dev_metrics.get("revoked") or {},
                    "exact_answer_rate", DEV_MODEL_REVOKED_CORRECT_MIN),
    }
    checks = {}
    for name, (block, key, threshold) in blocks.items():
        value = block.get(key)
        count = block.get("n")
        checks[name] = {
            "value": value,
            "operator": ">=",
            "threshold": threshold,
            "n": count,
            "correct_n": block.get("correct_n"),
            "passed": (value is not None and count is not None
                       and count > 0 and value >= threshold),
        }
    return {
        "evaluated_split": "dev",
        "evaluated_after_literal_pass": True,
        "passed": all(check["passed"] for check in checks.values()),
        "checks": checks,
    }


# ---------------------------------------------------------------------------
# Immutable writes, reference, and orchestration
# ---------------------------------------------------------------------------

def _freeze_rows(rows, name):
    _require(rows, f"Cannot freeze an empty row set: {name}")
    payload = b"".join((_canonical(row) + "\n").encode("utf-8")
                       for row in rows)
    return payload, _sha_bytes(payload)


def write_generation_rows(output, rows, *, split):
    _require(split in SPLITS, f"Unknown split: {split}")
    name = f"generations_{split}.jsonl"
    payload, digest = _freeze_rows(tuple(rows), name)
    path = Path(output) / name
    _write_new(path, payload)
    _require(_sha_file(path) == digest,
             f"Written generation bytes differ: {name}")
    return {
        "filename": name,
        "sha256": digest,
        "rows": len(rows),
        "first_write": True,
        "immutable": True,
    }


def validate_generation_rows(rows, decisions):
    """Require one immutable, label-free generation row per decision."""
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
                          "label", "target_role"):
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
        _require(row["selected_value"]
                 == decision.get("selected_value"),
                 "Generation literal value differs from its decision")
        _require(row["runtime_status"] == decision_status(decision),
                 "Generation status differs from its decision")
    _require(seen == expected, "Generation rows are incomplete")
    return True


def compressed_reference(config):
    """Read the existing v4 receipt as a separate, never-combined reference."""
    reference = config["compressed_reference"]
    path = _resolve(reference["path"])
    _require(path.is_file(), "Compressed reference receipt is missing")
    _require(_sha_file(path) == reference["sha256"],
             "Compressed reference receipt bytes differ")
    receipt = _read_json(path)
    _require(receipt.get("format") == "replica.memory_acceptance.metrics.v4",
             "Compressed reference format differs")
    variant = receipt["candidate_variant"]
    blocks = receipt["metrics"]["variants"][variant]
    quality = blocks["quality"]
    return {
        "role": "referenced_only_not_combined",
        "path": _display(path),
        "sha256": reference["sha256"],
        "metric_version": receipt.get("metric_version"),
        "candidate_variant": variant,
        "decision": receipt.get("verdict", {}).get("decision"),
        "quality": {
            "n": quality.get("n"),
            "fact_n": quality.get("fact_n"),
            "accepted_correct": quality.get("accepted_correct"),
            "accepted_correct_rate": quality.get("accepted_correct_rate"),
            "accepted_precision": quality.get("accepted_precision"),
        },
        "correction": {
            "n": blocks["correction"].get("n"),
            "accepted_correct_rate":
                blocks["correction"].get("accepted_correct_rate"),
        },
        "unaffected": {
            "n": blocks["unaffected"].get("n"),
            "accepted_correct_rate":
                blocks["unaffected"].get("accepted_correct_rate"),
        },
        "unknown": {
            "n": blocks["unknown"].get("n"),
            "unknown_false_recall_rate":
                blocks["unknown"].get("unknown_false_recall_rate"),
            "abstention_rate": blocks["unknown"].get("abstention_rate"),
        },
        "bank_plus_compressed_arm": False,
        "not_an_upper_bound": True,
        "note": ("This is a separate compressed-vector receipt; its rates are "
                 "never combined with the bank literal or model readout."),
    }


def _classification(config, *, native, model_executed):
    classification = dict(config["classification"])
    classification.update(CLASSIFICATION)
    classification["native"] = bool(native)
    classification["model_execution"] = bool(model_executed)
    classification["tokenizer_load"] = bool(model_executed)
    classification["generation"] = ("model_readout_only"
                                    if model_executed else False)
    return classification


def _resource_usage(config, *, native, model_executed, generation_rows):
    return {
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
        "native_jobs": 1 if native else 0,
        "fallback": "disabled",
        "limits": {
            "max_model_loads": config["budget"]["model_loads"],
            "max_decisions_per_split":
                config["generation"]["max_decisions_per_split"],
            "max_new_tokens": config["generation"]["max_new_tokens"],
            "max_wall_seconds": config["generation"]["wall_budget_seconds"],
            "artifact_cap_bytes": config["generation"]["artifact_cap_bytes"],
        },
    }


def _base_report(config, *, prepared_info, baseline_info, config_path,
                 native, compressed):
    return {
        "format": REPORT_FORMAT,
        "generated_at_utc": _utc_now(),
        "classification": _classification(
            config, native=native, model_executed=False),
        "roles": config["roles"],
        "protocol": {
            "config_path": (str(config_path) if config_path else None),
            "config_sha256": (_sha_file(config_path)
                              if config_path and Path(config_path).is_file()
                              else None),
            "decision_rule_sha256": _sha_json({
                "bank": config["bank"],
                "reader": config["reader"],
                "endpoint": config["endpoint"],
                "generation": config["generation"],
            }),
            "implementation_sha256": {
                name: _sha_file(PROJECT_ROOT / name)
                for name in (
                    "src/replica_lab/memory_bank_value.py",
                    "src/replica_lab/memory_address_bank.py",
                    "src/replica_lab/memory_causal_fresh.py",
                    "src/replica_lab/memory_matched.py",
                    "scripts/run_memory_bank_value.py",
                )},
            "prepared": prepared_info,
            "baseline": baseline_info,
            "selection_rule": (
                "exact (entity, attribute) match against live records minus "
                "the declared view-only revocation"),
            "literal_value_rule": (
                "exact string equality between the selected runtime record "
                "value and the evaluation-only expected string"),
            "prompt_contract": {
                "format": PROMPT_FORMAT,
                "fields": ["query_text", "selected_record_value",
                           "runtime_status"],
                "expected_answer_included": False,
                "evaluation_labels_read": False,
                "status_vocabulary": dict(ABSTENTION_STATUSES),
                "answer_normalization":
                    config["generation"]["answer_normalization"],
            },
            "generation_contract": {
                "enabled_ports": list(config["generation"]["enabled_ports"]),
                "enabled_branches":
                    list(config["generation"]["enabled_branches"]),
                "core_impl": config["generation"]["core_impl"],
                "empty_ledger": True,
                "record_memory": False,
                "workspace": False,
                "max_new_tokens": config["generation"]["max_new_tokens"],
                "stop_token_ids": list(STOP_TOKEN_IDS),
            },
            "thresholds": config["endpoint"],
            "compressed_reference": compressed,
        },
        "gating": {
            "order": ["literal_dev", "model_dev", "heldout"],
            "literal_dev": None,
            "model_dev": {
                "status": "not_run",
                "reason": "dev_literal_pending",
                "gate": None,
            },
            "heldout": {
                "status": "not_run",
                "evaluated_after_dev_pass": False,
                "decisions": 0,
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
            "stopped_before_heldout": True,
        },
        "raw_outputs": {
            "policy": config["reporting"]["raw_output_policy"],
            "files": {},
            "verified_unchanged": True,
        },
        "resource_usage": _resource_usage(
            config, native=native, model_executed=False,
            generation_rows=()),
        "profile": None,
        "limitations": list(config["limitations"]),
    }


def _artifact_bytes(output):
    return sum(path.stat().st_size for path in Path(output).iterdir()
               if path.is_file())


def _prepare_output(output):
    """Accept only a missing directory or one holding profiler telemetry."""
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


def _generation_events(rows):
    return {
        "rows": len(rows),
        "generated_tokens": sum(len(row["generated_ids"]) for row in rows),
        "ended_by_eos": sum(row["ended_by_eos"] for row in rows),
    }


def run_bank_value(episodes, revocations, labels_loader, config, output, *,
                   prepared_info=None, baseline_info=None, config_path=None,
                   generation_factory=None, native=False, profiler=None,
                   clock=time.monotonic):
    """Run the frozen literal-then-model sequence over loaded inputs.

    ``labels_loader`` is called only after the development decisions are
    written.  ``generation_factory`` is called only after the development
    literal endpoint passes.
    """
    validate_config(config)
    _require(callable(labels_loader), "A scoring-label loader is required")
    episodes = tuple(episodes)
    revocations = dict(revocations)
    output = _prepare_output(output)
    started = clock()
    deadline = started + config["generation"]["wall_budget_seconds"]
    query_texts = query_text_register(episodes)
    compressed = compressed_reference(config)

    dev_decisions = build_value_split_decisions(
        episodes, revocations, config, "dev")
    frozen_dev = freeze_decisions(dev_decisions, split="dev")
    raw_files = {"dev": write_frozen_decisions(output, frozen_dev)}
    _require(decision_key_register(frozen_dev.decisions)["unique"],
             "Dev decision keys are not unique")

    labels_doc = labels_loader()
    labels = labels_doc.get("labels", labels_doc) \
        if isinstance(labels_doc, Mapping) else labels_doc
    literal_dev_scored = score_literal_values(frozen_dev, labels)
    literal_dev_metrics = literal_dev_scored["splits"]["dev"]
    literal_gate = evaluate_dev_literal_gate(
        literal_dev_metrics, config["endpoint"])

    report = _base_report(
        config, prepared_info=prepared_info,
        baseline_info=baseline_info, config_path=config_path,
        native=native, compressed=compressed)
    report["raw_outputs"]["files"] = dict(raw_files)
    report["gating"]["literal_dev"] = literal_gate
    report["literal_metrics"] = {"dev": literal_dev_metrics}
    report["counts"] = {
        "decisions": frozen_dev.row_count,
        "by_split": {"dev": frozen_dev.row_count},
        "decision_keys": decision_key_register(frozen_dev.decisions),
    }

    if not literal_gate["passed"]:
        report["gating"]["model_dev"] = {
            "status": "not_run",
            "reason": "dev_literal_failed",
            "gate": None,
        }
        report["gating"]["heldout"] = {
            "status": "not_run",
            "evaluated_after_dev_pass": False,
            "reason": "dev_literal_failed",
            "decisions": 0,
            "generations": 0,
            "metrics": None,
        }
        report["gating"]["train"] = {
            "status": "not_run",
            "role": "descriptive_only",
            "reason": "dev_literal_failed",
            "model_generation": "not_run",
            "metrics": None,
        }
        report["gating"]["stopped_before_model"] = True
        report["gating"]["stopped_before_heldout"] = True
        report["outcome"] = {
            "status": "stopped_at_dev_literal",
            "literal_dev": "failed",
            "model_dev": "not_run",
            "heldout": "not_run",
            "train": "not_run",
            "claim": "no model, tokenizer, generation or heldout evidence",
        }
        report["elapsed_seconds"] = clock() - started
        report["raw_outputs"]["verified_unchanged"] = _verify_decisions(
            output, frozen_dev, raw_files["dev"])
        _finish_report(report, config, output, profiler)
        return report

    _require(callable(generation_factory),
             "A generation factory is required after the literal gate")
    generator = generation_factory(
        deadline=deadline, baseline_info=baseline_info)
    try:
        dev_generations = tuple(
            generator(frozen_dev.decisions, query_texts))
        validate_generation_rows(dev_generations, frozen_dev.decisions)
        raw_files["generations_dev"] = write_generation_rows(
            output, dev_generations, split="dev")
        _require(clock() <= deadline,
                 "Development generation exceeded the wall budget")

        dev_generation_scored = score_generations(dev_generations, labels)
        model_dev_metrics = dev_generation_scored["splits"]["dev"]
        model_gate = evaluate_dev_model_gate(
            model_dev_metrics, config["endpoint"])
        report["raw_outputs"]["files"] = dict(raw_files)
        report["gating"]["model_dev"] = {
            "status": "evaluated",
            "reason": None,
            "gate": model_gate,
            "events": _generation_events(dev_generations),
        }
        report["generation_metrics"] = {"dev": model_dev_metrics}
        report["classification"] = _classification(
            config, native=native, model_executed=native)
        report["resource_usage"] = _resource_usage(
            config, native=native, model_executed=native,
            generation_rows=dev_generations)
        report["gating"]["stopped_before_model"] = False
        report["counts"]["decisions"] = frozen_dev.row_count
        report["counts"]["generations"] = len(dev_generations)

        if not model_gate["passed"]:
            report["gating"]["heldout"] = {
                "status": "not_run",
                "evaluated_after_dev_pass": False,
                "reason": "dev_model_failed",
                "decisions": 0,
                "generations": 0,
                "metrics": None,
            }
            report["gating"]["train"] = {
                "status": "not_run",
                "role": "descriptive_only",
                "reason": "dev_model_failed",
                "model_generation": "not_run",
                "metrics": None,
            }
            report["gating"]["stopped_before_heldout"] = True
            report["outcome"] = {
                "status": "stopped_at_dev_model",
                "literal_dev": "passed",
                "model_dev": "failed",
                "heldout": "not_run",
                "train": "not_run",
                "claim": ("literal value exactness plus a failing bounded "
                          "model readout; no heldout evidence"),
            }
            report["elapsed_seconds"] = clock() - started
            report["raw_outputs"]["verified_unchanged"] = all(
                [_verify_decisions(output, frozen_dev, raw_files["dev"]),
                 _verify_generations(output, dev_generations,
                                     raw_files["generations_dev"])])
            _finish_report(report, config, output, profiler)
            return report

        heldout_decisions = build_value_split_decisions(
            episodes, revocations, config, "heldout")
        frozen_heldout = freeze_decisions(
            heldout_decisions, split="heldout")
        raw_files["heldout"] = write_frozen_decisions(
            output, frozen_heldout)
        heldout_generations = tuple(
            generator(frozen_heldout.decisions, query_texts))
        validate_generation_rows(
            heldout_generations, frozen_heldout.decisions)
        raw_files["generations_heldout"] = write_generation_rows(
            output, heldout_generations, split="heldout")
        _require(clock() <= deadline,
                 "Heldout generation exceeded the wall budget")

        heldout_scored = score_generations(heldout_generations, labels)
        heldout_metrics = heldout_scored["splits"]["heldout"]
        heldout_literal = score_literal_values(frozen_heldout, labels)
        heldout_literal_metrics = heldout_literal["splits"]["heldout"]

        train_decisions = build_value_split_decisions(
            episodes, revocations, config, "train")
        frozen_train = freeze_decisions(train_decisions, split="train")
        raw_files["train"] = write_frozen_decisions(output, frozen_train)
        train_literal = score_literal_values(frozen_train, labels)
        train_literal_metrics = train_literal["splits"]["train"]

        report["raw_outputs"]["files"] = dict(raw_files)
        report["literal_metrics"]["heldout"] = heldout_literal_metrics
        report["literal_metrics"]["train"] = train_literal_metrics
        report["generation_metrics"]["heldout"] = heldout_metrics
        report["gating"]["heldout"] = {
            "status": "evaluated",
            "evaluated_after_dev_pass": True,
            "reason": None,
            "decisions": frozen_heldout.row_count,
            "generations": len(heldout_generations),
            "metrics": heldout_metrics,
            "literal_metrics": heldout_literal_metrics,
            "endpoint_enforced": False,
            "role": "post_freeze_observation_only",
        }
        report["gating"]["train"] = {
            "status": "descriptive_only",
            "role": "descriptive_only",
            "reason": "both_dev_gates_passed",
            "model_generation": "not_run_descriptive_only",
            "decisions": frozen_train.row_count,
            "metrics": train_literal_metrics,
        }
        report["gating"]["stopped_before_heldout"] = False
        report["counts"] = {
            "decisions": (frozen_dev.row_count + frozen_heldout.row_count
                          + frozen_train.row_count),
            "by_split": {
                "dev": frozen_dev.row_count,
                "heldout": frozen_heldout.row_count,
                "train": frozen_train.row_count,
            },
            "generations": {
                "dev": len(dev_generations),
                "heldout": len(heldout_generations),
                "train": 0,
            },
            "decision_keys": decision_key_register(
                frozen_dev.decisions + frozen_heldout.decisions
                + frozen_train.decisions),
        }
        report["resource_usage"] = _resource_usage(
            config, native=native, model_executed=native,
            generation_rows=dev_generations + heldout_generations)
        report["outcome"] = {
            "status": "dev_passed_heldout_evaluated",
            "literal_dev": "passed",
            "model_dev": "passed",
            "heldout": "evaluated",
            "train": "descriptive_only",
            "claim": ("selected-record literal value and a bounded MiniCPM "
                      "readout; not a persistent-learning or calibrated "
                      "claim"),
        }
        report["elapsed_seconds"] = clock() - started
        verified = True
        for split, frozen, key in (
                ("dev", frozen_dev, "dev"),
                ("heldout", frozen_heldout, "heldout"),
                ("train", frozen_train, "train")):
            verified = verified and _verify_decisions(
                output, frozen, raw_files[key])
        verified = verified and _verify_generations(
            output, dev_generations, raw_files["generations_dev"])
        verified = verified and _verify_generations(
            output, heldout_generations,
            raw_files["generations_heldout"])
        report["raw_outputs"]["verified_unchanged"] = verified
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


def _verify_generations(output, rows, manifest):
    path = Path(output) / manifest["filename"]
    _require(path.is_file(), f"Generation file is missing: {path.name}")
    _payload, digest = _freeze_rows(tuple(rows), path.name)
    _require(_sha_file(path) == digest,
             f"Generation file changed: {path.name}")
    _require(manifest["sha256"] == digest,
             "Generation manifest hash changed")
    return True


def _finish_report(report, config, output, profiler):
    if profiler is not None:
        report["profile"] = profiler.stop()
        report["resource_telemetry"] = {
            "source": "RunProfiler",
            "memory": report["profile"].get("memory"),
            "mps_recommended_max_memory_bytes": report["profile"].get(
                "mps_recommended_max_memory_bytes"),
            "host_start": report["profile"].get("host_start"),
            "host_end": report["profile"].get("host_end"),
            "errors": report["profile"].get("errors"),
        }
    _require(report["elapsed_seconds"]
             <= config["generation"]["wall_budget_seconds"],
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


def run_prepared_bank_value(config, *, prepared=None, output, config_path=None,
                            generation_factory=None, native=False,
                            profiler=None, verify_source=True):
    """Verify pinned prepared04 and native02, then run the frozen sequence."""
    validate_config(config)
    prepared_root = _prepared_root(prepared, config)
    _require((prepared_root / "episodes.json").is_file(),
             "Prepared episodes are missing")
    _require((prepared_root / "revocations.json").is_file(),
             "Prepared revocations are missing")
    _require((prepared_root / "labels.json").is_file(),
             "Prepared labels are missing")
    baseline_info = verify_baseline(
        config, prepared=prepared_root, device="cpu",
        verify_source=verify_source)
    formats = config["prepared"]["formats"]
    episodes_doc = _read_json(prepared_root / "episodes.json")
    _require(episodes_doc.get("format") == formats["episodes"],
             "Prepared episodes format differs")
    selection = episodes_doc.get("episodes")
    _require(isinstance(selection, list) and selection,
             "Prepared episodes are missing")
    revocations_doc = _read_json(prepared_root / "revocations.json")
    _require(revocations_doc.get("format") == formats["revocations"],
             "Prepared revocations format differs")
    revocations = revocations_doc.get("revocations")
    _require(isinstance(revocations, dict),
             "Prepared revocations are missing")
    episode_ids = {episode["episode_id"] for episode in selection}
    _require(len(episode_ids) == len(selection),
             "Prepared episode IDs repeat")
    _require(set(revocations) <= episode_ids,
             "Revocations name an unknown episode")
    manifest = _read_json(prepared_root / "manifest.json")
    _require(manifest.get("episodes") == len(selection),
             "Prepared episode count differs from its manifest")

    def labels_loader():
        doc = _read_json(prepared_root / "labels.json")
        _require(doc.get("format") == formats["labels"],
                 "Prepared labels format differs")
        _require(doc.get("evaluation_only") is True,
                 "Prepared labels are not marked evaluation-only")
        labels = doc.get("episodes")
        _require(isinstance(labels, dict) and labels,
                 "Prepared evaluation labels are missing")
        return labels

    return run_bank_value(
        selection, revocations, labels_loader, config, output,
        prepared_info=baseline_info["prepared"],
        baseline_info=baseline_info,
        config_path=config_path, generation_factory=generation_factory,
        native=native, profiler=profiler)

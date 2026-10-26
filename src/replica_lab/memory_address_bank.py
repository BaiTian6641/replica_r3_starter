"""Deterministic explicit-record address-bank control.

The bank is deliberately smaller than a learned memory path.  It consumes only
the runtime record views derived by :mod:`replica_lab.memory_matched`:

* ``episode_states`` supplies the dated states;
* ``_record_views`` supplies records, live records and superseded records;
* a declared view-only revocation removes named live records from the current
  view.

For every ``(episode, state, phase, query)`` the bank performs exactly one
exact ``(entity, attribute)`` address lookup.  It selects one record, or
abstains for no address, an ambiguous address, a superseded-only address or a
revoked-only address.  Reader layers never multiply or alter that decision.

Evaluation labels are not selection inputs.  The runner freezes and writes the
development decisions before it parses ``labels.json``; scoring joins the frozen
decision IDs to the evaluation-only labels afterwards.  The bank emits record
identities (selection evidence), not values, so its output is never presented as
equivalent to a compressed-vector accepted-correct result.  No model, tokenizer,
native job, checkpoint or generation path is used here.
"""
from __future__ import annotations

from dataclasses import dataclass
import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Callable, Mapping, Sequence

from .memory_matched import SPLITS, _record_views, episode_states

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIG_FORMAT = "replica.memory_address_bank.config.v1"
DECISIONS_FORMAT = "replica.memory_address_bank.decisions.v1"
REPORT_FORMAT = "replica.memory_address_bank.report.v1"

PHASES = ("first_process", "restart", "revocation")
PHASE_QUERY_FIELDS = {
    "first_process": "queries",
    "restart": "restart_queries",
    "revocation": "revocation_queries",
}
DEFAULT_PHASE_MAP = {
    "post_restart": ("restart", "revocation"),
    "revocation_only": ("revocation",),
    "all": PHASES,
}
ABSTAIN_REASONS = (
    "abstain_no_address",
    "abstain_ambiguous_address",
    "abstain_superseded_only_address",
    "abstain_revoked_only_address",
)
DEV_FACT_SELECTION_IDENTITY_MIN = 0.90
DEV_UNKNOWN_SELECTIONS_MAX = 0
DEV_REVOKED_SELECTIONS_MAX = 0
SPLIT_ROLES = {
    "train": {
        "role": "descriptive_only",
        "endpoint": False,
        "requires_dev_pass": False,
    },
    "dev": {
        "role": "primary_endpoint_gate",
        "endpoint": True,
        "requires_dev_pass": False,
    },
    "heldout": {
        "role": "post_freeze_confirmation",
        "endpoint": False,
        "requires_dev_pass": True,
    },
}
CLASSIFICATION = {
    "package": "explicit_record_address_bank_control",
    "native": False,
    "model_execution": False,
    "tokenizer_load": False,
    "generation": False,
    "calibrated": False,
    "learned_selector": False,
    "bank_plus_compressed_arm": False,
    "compressed_value_comparison": False,
    "oracle": "diagnostic_only_not_executed",
    "bank_output": "selection_evidence_only",
}
LIMITATIONS = [
    "Exact string address matching is not learned, semantic or calibrated retrieval.",
    "The bank emits record identities only; no literal bank value is compared with a compressed-vector accepted-correct result.",
    "There is no bank-plus-compressed hybrid arm; the oracle remains diagnostic-only and is not executed.",
    "No model, tokenizer, forward pass, generation, optimizer, checkpoint or native resource path is used.",
    "Heldout evidence is reported only after the frozen rule passes the predeclared dev endpoint; no threshold or fixture is tuned.",
    "Declared independent-world distractor targets are episode inputs; a correct distractor selection is counted separately from false recall.",
    "Selection evidence does not establish a language, generation or general-memory capability.",
]


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def _sha_text(value):
    return _sha_bytes(value.encode("utf-8"))


def _sha_json(value):
    return _sha_text(_canonical(value))


def _sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc_now():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def _rate(count, total):
    return count / total if total else None


def _is_sha256(value):
    return (isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


# ---------------------------------------------------------------------------
# Configuration and prepared inputs
# ---------------------------------------------------------------------------

def validate_config(config):
    """Validate and freeze the protocol object without mutating it."""
    _require(isinstance(config, Mapping), "Address-bank configuration must be an object")
    required = ("format", "declared_at_utc", "purpose", "classification",
                "prepared", "bank", "reader", "endpoint", "roles",
                "reporting", "budget", "limitations")
    for name in required:
        _require(name in config, f"Address-bank configuration is missing: {name}")
    _require(config["format"] == CONFIG_FORMAT,
             "Address-bank configuration format differs")

    prepared = config["prepared"]
    _require(isinstance(prepared, Mapping), "Prepared input declaration is missing")
    _require(tuple(prepared.get("selection_inputs", ()))
             == ("episodes.json", "revocations.json"),
             "Selection inputs must be exactly episodes and revocations")
    _require(tuple(prepared.get("scoring_only_inputs", ())) == ("labels.json",),
             "labels.json must be the only scoring-only input")
    formats = prepared.get("formats") or {}
    for key in ("episodes", "revocations", "labels", "manifest"):
        _require(isinstance(formats.get(key), str) and formats[key],
                 f"Prepared format pin is missing: {key}")
    pins = prepared.get("pins") or {}
    for key in ("episodes_sha256", "revocations_sha256", "labels_sha256",
                "manifest_sha256"):
        _require(_is_sha256(pins.get(key)), f"Prepared hash pin is invalid: {key}")
    manifest_requirements = prepared.get("manifest_requirements") or {}
    for key in ("native_computation", "model_loaded", "tokenizer_loaded"):
        _require(manifest_requirements.get(key) is False,
                 f"Prepared manifest requirement must be false: {key}")

    bank = config["bank"]
    _require(bank.get("match") == "exact",
             "Address-bank match must be exact")
    _require(tuple(bank.get("address_key", ())) == ("entity", "attribute"),
             "Address key must be (entity, attribute)")
    _require(tuple(bank.get("phase_order", ())) == PHASES,
             "Phase order differs")
    _require(dict(bank.get("phase_query_fields") or {}) == PHASE_QUERY_FIELDS,
             "Phase query fields differ")
    _require(bank.get("decision_unit") == "episode/state/phase/query",
             "Decision unit differs")
    _require(bank.get("scope") == "episode_world",
             "Bank scope must be episode/world")
    _require(bank.get("duplicate_record_id")
             == "dedupe_if_identical_else_reject",
             "Duplicate record-id policy differs")
    _require(bank.get("duplicate_address") == "abstain_ambiguous",
             "Duplicate-address policy differs")
    _require(tuple(bank.get("abstain_reasons", ())) == ABSTAIN_REASONS,
             "Abstain reason inventory differs")
    revocation = bank.get("revocation") or {}
    _require(revocation.get("scope") == "view_only",
             "Revocation scope must be view_only")
    _require(revocation.get("cached_history_retained") is True,
             "Revocation must retain cached history")
    phase_map = revocation.get("applies_at_phase_map") or {}
    _require(dict(phase_map) == {key: list(value)
                                 for key, value in DEFAULT_PHASE_MAP.items()},
             "Revocation phase map differs")

    reader = config["reader"]
    _require(reader.get("decision_count_multiplier") == 1,
             "Reader layers must not multiply decisions")
    _require(isinstance(reader.get("declared_source_layers"), Sequence)
             and len(reader.get("declared_source_layers")) >= 1,
             "Declared source layers are missing")

    endpoint = config["endpoint"]
    _require(endpoint.get("split") == "dev",
             "The endpoint split must be dev")
    _require(endpoint.get("dev_fact_selection_identity_min")
             == DEV_FACT_SELECTION_IDENTITY_MIN,
             "Dev fact selection identity threshold may not be tuned")
    _require(type(endpoint.get("dev_unknown_selections_max")) is int
             and endpoint["dev_unknown_selections_max"]
             == DEV_UNKNOWN_SELECTIONS_MAX,
             "Dev unknown selection threshold may not be tuned")
    _require(type(endpoint.get("dev_revoked_selections_max")) is int
             and endpoint["dev_revoked_selections_max"]
             == DEV_REVOKED_SELECTIONS_MAX,
             "Dev revoked selection threshold may not be tuned")
    _require(endpoint.get("heldout") == "evaluate_after_dev_pass_only",
             "Heldout must be evaluated only after dev passes")
    _require(endpoint.get("tuning") == "forbidden",
             "Endpoint tuning must be forbidden")

    roles = config["roles"]
    _require(set(roles) == set(SPLITS), "Explicit split roles are required")
    for split, expected in SPLIT_ROLES.items():
        actual = roles[split]
        for key, value in expected.items():
            _require(actual.get(key) == value,
                     f"Role {split}.{key} differs from the frozen role")

    reporting = config["reporting"]
    for group in ("split", "cohort", "world", "phase", "state", "complexity"):
        _require(group in reporting.get("groups", ()),
                 f"Reporting group is missing: {group}")
    policy = reporting.get("raw_output_policy") or {}
    _require(policy.get("fresh_output_required") is True,
             "Fresh output is required")
    _require(policy.get("immutable_first_write") is True,
             "Immutable first write is required")

    budget = config["budget"]
    for key in ("model_loads", "tokenizer_loads", "forward_passes",
                "optimizer_updates", "gradients", "checkpoints_written",
                "native_jobs", "native_resource_reports"):
        _require(type(budget.get(key)) is int and budget[key] == 0,
                 f"Native or training budget must be zero: {key}")
    _require(isinstance(config["limitations"], Sequence)
             and config["limitations"], "Limitations are required")
    return config


def _prepared_root(prepared, config):
    path = Path(prepared if prepared is not None else config["prepared"]["root"])
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_json(path):
    return json.loads(Path(path).read_text())


def _prepared_filenames():
    return {
        "episodes": "episodes.json",
        "revocations": "revocations.json",
        "labels": "labels.json",
        "manifest": "manifest.json",
    }


def _verify_prepared_hashes(prepared, config):
    prepared = Path(prepared)
    names = _prepared_filenames()
    pins = config["prepared"]["pins"]
    actual = {}
    for key, name in names.items():
        path = prepared / name
        _require(path.is_file(), f"Prepared artifact is missing: {name}")
        actual[f"{key}_sha256"] = _sha_file(path)
    _require(actual["manifest_sha256"] == pins["manifest_sha256"],
             "Prepared manifest hash differs")
    manifest = _read_json(prepared / names["manifest"])
    _require(manifest.get("format") == config["prepared"]["formats"]["manifest"],
             "Prepared manifest format differs")
    for key in ("episodes_sha256", "revocations_sha256", "labels_sha256"):
        _require(actual[key] == pins[key],
                 f"Prepared {key} bytes differ from the pinned input")
        _require(manifest.get(key) == pins[key],
                 f"Prepared manifest {key} differs from the pinned input")
    requirements = config["prepared"]["manifest_requirements"]
    for key, expected in requirements.items():
        _require(manifest.get(key) is expected,
                 f"Prepared manifest does not declare {key}={expected}")
    return {"hashes": actual, "manifest": manifest}


def load_selection_inputs(prepared, config):
    """Load exactly the selection inputs; labels.json is only hashed here."""
    validate_config(config)
    prepared = Path(prepared)
    verification = _verify_prepared_hashes(prepared, config)
    names = _prepared_filenames()
    formats = config["prepared"]["formats"]

    episodes_doc = _read_json(prepared / names["episodes"])
    _require(episodes_doc.get("format") == formats["episodes"],
             "Episodes format differs")
    episodes = episodes_doc.get("episodes")
    _require(isinstance(episodes, list) and episodes,
             "Prepared episodes are missing")
    revocations_doc = _read_json(prepared / names["revocations"])
    _require(revocations_doc.get("format") == formats["revocations"],
             "Revocations format differs")
    revocations = revocations_doc.get("revocations")
    _require(isinstance(revocations, dict), "Prepared revocations are missing")

    episode_ids = set()
    state_count = 0
    for episode in episodes:
        _require(isinstance(episode, dict), "Prepared episode is not an object")
        episode_id = episode.get("episode_id")
        _require(isinstance(episode_id, str) and episode_id,
                 "Prepared episode ID is missing")
        _require(episode_id not in episode_ids, "Prepared episode IDs repeat")
        episode_ids.add(episode_id)
        _require(episode.get("split") in SPLITS, "Prepared split differs")
        state_count += len(episode_states(episode))
    _require(set(revocations) <= episode_ids,
             "Revocations name an unknown episode")
    manifest = verification["manifest"]
    _require(manifest.get("episodes") == len(episodes),
             "Prepared episode count differs")
    _require(manifest.get("states") == state_count,
             "Prepared state count differs")
    return {
        "episodes": tuple(episodes),
        "revocations": dict(revocations),
        "prepared": str(prepared),
        "hashes": verification["hashes"],
        "manifest": manifest,
    }


def load_scoring_labels(prepared, config):
    """Load evaluation-only labels after decisions have been frozen."""
    validate_config(config)
    prepared = Path(prepared)
    verification = _verify_prepared_hashes(prepared, config)
    doc = _read_json(prepared / _prepared_filenames()["labels"])
    _require(doc.get("format") == config["prepared"]["formats"]["labels"],
             "Labels format differs")
    _require(doc.get("evaluation_only") is True,
             "Labels are not marked evaluation-only")
    labels = doc.get("episodes")
    _require(isinstance(labels, dict) and labels, "Evaluation labels are missing")
    return {
        "labels": labels,
        "hashes": verification["hashes"],
        "manifest": verification["manifest"],
    }


# ---------------------------------------------------------------------------
# Pure address resolution and decision construction
# ---------------------------------------------------------------------------

def _record_signature(record):
    fields = ("record_id", "entity", "attribute", "value", "observed_at",
              "kind", "supersedes", "origin_world_id")
    return _canonical({name: record.get(name) for name in fields})


def _dedupe_records(records):
    seen = {}
    ordered = []
    for record in records:
        record_id = record.get("record_id")
        _require(isinstance(record_id, str) and record_id,
                 "Record ID is missing")
        signature = _record_signature(record)
        if record_id in seen:
            _require(seen[record_id] == signature,
                     f"Duplicate record ID has conflicting content: {record_id}")
            continue
        seen[record_id] = signature
        ordered.append(record)
    return tuple(ordered)


def _record_meta(record, source_view):
    return {
        "record_id": record.get("record_id"),
        "origin_world_id": record.get("origin_world_id"),
        "observed_at": record.get("observed_at"),
        "record_kind": record.get("kind"),
        "source_view": source_view,
    }


def _by_address(records):
    index = {}
    for record in _dedupe_records(records):
        index.setdefault((record["entity"], record["attribute"]), []).append(record)
    return index


def resolve_address(entity, attribute, live_records, superseded_records, *,
                    revoked_ids=()):
    """Resolve one exact address against a live view minus revocations.

    ``live_records`` and ``superseded_records`` are runtime record views.
    ``revoked_ids`` names records hidden by the declared view-only revocation.
    No evaluator label or expected answer is accepted by this function.
    """
    _require(isinstance(entity, str) and entity, "Query entity is missing")
    _require(isinstance(attribute, str) and attribute, "Query attribute is missing")
    revoked_ids = frozenset(revoked_ids)
    live = _dedupe_records(live_records)
    superseded = _dedupe_records(superseded_records)
    overlap = ({record["record_id"] for record in live}
               & {record["record_id"] for record in superseded})
    _require(not overlap,
             "A record appears in both live and superseded views")
    address = (entity, attribute)
    live_at = [record for record in live
               if (record["entity"], record["attribute"]) == address]
    superseded_at = [record for record in superseded
                     if (record["entity"], record["attribute"]) == address]

    revoked_at = [record for record in live_at
                  if record["record_id"] in revoked_ids]
    remaining = [record for record in live_at
                 if record["record_id"] not in revoked_ids]

    all_ids = {record["record_id"] for record in live_at + superseded_at}
    excluded_superseded = sorted(record["record_id"]
                                 for record in superseded_at)
    excluded_revoked = sorted(record["record_id"] for record in revoked_at)
    excluded_reasons = [
        {"record_id": record_id, "reason": "superseded"}
        for record_id in excluded_superseded
    ] + [
        {"record_id": record_id, "reason": "revoked"}
        for record_id in excluded_revoked
    ]
    live_candidate_ids = sorted(record["record_id"] for record in remaining)
    base = {
        "address_record_ids": sorted(all_ids),
        "candidate_ids": live_candidate_ids,
        "live_candidate_ids": live_candidate_ids,
        "live_candidates": [_record_meta(record, "live")
                            for record in remaining],
        "excluded_superseded_ids": excluded_superseded,
        "excluded_revoked_ids": excluded_revoked,
        "excluded_reasons": excluded_reasons,
    }

    if len(remaining) == 1:
        record = remaining[0]
        return {
            **base,
            "decision": "select",
            "abstained": False,
            "reason": "selected_exact_address",
            "selected_ids": [record["record_id"]],
            "selected_record_id": record["record_id"],
            "selected_records": [_record_meta(record, "live")],
        }
    if len(remaining) > 1:
        return {
            **base,
            "decision": "abstain",
            "abstained": True,
            "reason": "abstain_ambiguous_address",
            "selected_ids": [],
            "selected_record_id": None,
            "selected_records": [],
        }
    if superseded_at:
        return {
            **base,
            "decision": "abstain",
            "abstained": True,
            "reason": "abstain_superseded_only_address",
            "selected_ids": [],
            "selected_record_id": None,
            "selected_records": [],
        }
    if revoked_at:
        return {
            **base,
            "decision": "abstain",
            "abstained": True,
            "reason": "abstain_revoked_only_address",
            "selected_ids": [],
            "selected_record_id": None,
            "selected_records": [],
        }
    return {
        **base,
        "decision": "abstain",
        "abstained": True,
        "reason": "abstain_no_address",
        "selected_ids": [],
        "selected_record_id": None,
        "selected_records": [],
    }


def _phase_map(config):
    raw = config["bank"]["revocation"]["applies_at_phase_map"]
    return {key: tuple(value) for key, value in raw.items()}


def revocation_applies(revocation, phase, *, phase_map=None):
    """Return whether the declared revocation view applies to a phase."""
    _require(phase in PHASES, f"Unknown phase: {phase}")
    if not revocation:
        return False
    _require(revocation.get("scope") == "view_only",
             "Revocation must be a view-only scope")
    applies_at = revocation.get("applies_at", "revocation_only")
    mapping = phase_map or DEFAULT_PHASE_MAP
    _require(applies_at in mapping,
             f"Unknown revocation applies_at value: {applies_at}")
    return phase in mapping[applies_at]


def _revocation_ids(revocation):
    if not revocation:
        return ()
    raw = revocation.get("record_ids")
    if raw is None:
        raw = [revocation.get("record_id")]
    _require(isinstance(raw, (list, tuple)) and raw,
             "Revocation record IDs are missing")
    ids = []
    for record_id in raw:
        _require(isinstance(record_id, str) and record_id,
                 "Revocation record ID is invalid")
        _require(record_id not in ids, "Revocation record IDs repeat")
        ids.append(record_id)
    return tuple(ids)


def _episode_records(episode):
    for group in ("facts", "distractors", "corrections"):
        for record in episode.get(group, ()):
            yield group, record


def _validate_revocation(episode, revocation):
    if not revocation:
        return None
    _require(isinstance(revocation, Mapping),
             "Episode revocation must be an object")
    _require(revocation.get("scope") == "view_only",
             "Episode revocation must be view-only")
    _require(revocation.get("cached_history_retained") is True,
             "Episode revocation must retain cached history")
    ids = _revocation_ids(revocation)
    records = {}
    for _, record in _episode_records(episode):
        record_id = record.get("record_id")
        signature = _record_signature(record)
        if record_id in records and records[record_id] != signature:
            raise ValueError(
                f"Episode record IDs conflict at selection time: {record_id}")
        records[record_id] = signature
    for record_id in ids:
        _require(record_id in records,
                 f"Revocation target is absent from the episode: {record_id}")
    target = next((record for _, record in _episode_records(episode)
                   if record.get("record_id") == ids[0]), None)
    for key in ("entity", "attribute"):
        declared = revocation.get(key)
        if declared is not None:
            _require(target is not None and target.get(key) == declared,
                     f"Revocation {key} differs from the target record")
    return {
        "record_ids": ids,
        "applies_at": revocation.get("applies_at", "revocation_only"),
        "scope": "view_only",
    }


def _decision_id(episode_id, state, phase, query_id):
    digest = _sha_json([episode_id, state, phase, query_id])
    return f"decision-{digest[:16]}"


def _decision_sort_key(decision):
    return (decision["episode_id"], decision["state"], decision["phase"],
            decision["query_id"])


def build_episode_decisions(episode, revocation, config, *, phases=PHASES):
    """Build the bank decisions for one episode from runtime inputs only."""
    _require(isinstance(episode, Mapping), "Episode must be an object")
    for key in ("episode_id", "world_id", "split", "facts", "distractors",
                "corrections", "queries", "restart_queries",
                "revocation_queries"):
        _require(key in episode, f"Episode is missing: {key}")
    _require(episode["split"] in SPLITS, "Episode split differs")
    phase_map = _phase_map(config)
    normalized = _validate_revocation(episode, revocation)
    decision_rows = []
    for state in episode_states(episode):
        records, live, superseded = _record_views(episode, state)
        live_index = _by_address(live)
        superseded_index = _by_address(superseded)
        local_queries = set()
        for phase in phases:
            _require(phase in PHASE_QUERY_FIELDS, f"Unknown phase: {phase}")
            field = PHASE_QUERY_FIELDS[phase]
            queries = episode.get(field, ())
            _require(isinstance(queries, (list, tuple)),
                     f"Phase queries are not a sequence: {field}")
            applied = normalized is not None and revocation_applies(
                normalized, phase, phase_map=phase_map)
            applied_ids = normalized["record_ids"] if applied else ()
            for query in queries:
                query_id = query.get("query_id")
                _require(isinstance(query_id, str) and query_id,
                         "Query ID is missing")
                _require(query_id not in local_queries,
                         f"Query ID repeats within a state: {query_id}")
                local_queries.add(query_id)
                _require(query.get("phase") == phase,
                         f"Query {query_id} phase marker differs")
                address = (query.get("entity"), query.get("attribute"))
                resolved = resolve_address(
                    address[0], address[1],
                    live_index.get(address, ()),
                    superseded_index.get(address, ()),
                    revoked_ids=applied_ids)
                decision_rows.append({
                    "format": DECISIONS_FORMAT,
                    "decision_id": _decision_id(
                        episode["episode_id"], state["name"], phase, query_id),
                    "episode_id": episode["episode_id"],
                    "world_id": episode["world_id"],
                    "split": episode["split"],
                    "state": state["name"],
                    "phase": phase,
                    "query_id": query_id,
                    "query_form": query.get("form"),
                    "entity": address[0],
                    "attribute": address[1],
                    "scope_episode_id": episode["episode_id"],
                    "scope_world_id": episode["world_id"],
                    "revocation_declared": normalized is not None,
                    "revocation_applies_at": (
                        normalized["applies_at"] if normalized else None),
                    "revocation_applied": applied,
                    "revocation_record_ids": sorted(applied_ids),
                    **resolved,
                })
    decision_rows.sort(key=_decision_sort_key)
    return tuple(decision_rows)


def build_split_decisions(episodes, revocations, config, split):
    """Build one split's decisions in deterministic input order."""
    validate_config(config)
    _require(split in SPLITS, f"Unknown split: {split}")
    selected = [episode for episode in episodes if episode.get("split") == split]
    _require(selected, f"No episodes are declared for split: {split}")
    episode_ids = set()
    decisions = []
    for episode in selected:
        episode_id = episode.get("episode_id")
        _require(episode_id not in episode_ids,
                 "Episode IDs repeat across the prepared input")
        episode_ids.add(episode_id)
        decisions.extend(build_episode_decisions(
            episode, revocations.get(episode_id), config))
    decisions.sort(key=_decision_sort_key)
    _require(len(decisions) == expected_decision_rows(selected),
             "Decision count differs from episode/state/phase/query inventory")
    return tuple(decisions)


def expected_decision_rows(episodes):
    """Exact unique decision count: episode x state x phase x query."""
    total = 0
    for episode in episodes:
        states = len(episode_states(episode))
        total += states * sum(len(episode.get(field, ()))
                              for field in PHASE_QUERY_FIELDS.values())
    return total


@dataclass(frozen=True)
class FrozenDecisions:
    split: str
    decisions: tuple
    data: bytes
    sha256: str

    @property
    def row_count(self):
        return len(self.decisions)


def decision_digest(decisions):
    """Canonical digest of a decision sequence, independent of Python objects."""
    return _sha_bytes(b"".join(
        (_canonical(decision) + "\n").encode("utf-8")
        for decision in decisions))


def freeze_decisions(decisions, *, split):
    """Freeze decisions to canonical JSONL bytes and a private deep copy."""
    _require(split in SPLITS, f"Unknown frozen split: {split}")
    rows = tuple(decisions)
    _require(rows, f"Cannot freeze an empty decision set: {split}")
    keys = set()
    for decision in rows:
        _require(decision.get("format") == DECISIONS_FORMAT,
                 "Decision format differs")
        key = (decision["episode_id"], decision["state"],
               decision["phase"], decision["query_id"])
        _require(key not in keys, "Decision keys repeat")
        keys.add(key)
    payload = b"".join((_canonical(decision) + "\n").encode("utf-8")
                       for decision in rows)
    frozen_rows = tuple(copy.deepcopy(decision) for decision in rows)
    return FrozenDecisions(split=split, decisions=frozen_rows,
                           data=payload, sha256=_sha_bytes(payload))


# ---------------------------------------------------------------------------
# Immutable raw output
# ---------------------------------------------------------------------------

def _fresh_output(output):
    output = Path(output)
    if output.exists():
        _require(output.is_dir() and not any(output.iterdir()),
                 "Use a fresh output directory")
    else:
        output.mkdir(parents=True, exist_ok=False)
    return output


def _write_new(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def write_frozen_decisions(output, frozen):
    """Write a frozen split once and verify the immutable bytes."""
    output = Path(output)
    name = f"decisions_{frozen.split}.jsonl"
    path = output / name
    _write_new(path, frozen.data)
    written = _sha_file(path)
    _require(written == frozen.sha256,
             f"Written decision bytes differ from the frozen digest: {name}")
    return {
        "filename": name,
        "sha256": frozen.sha256,
        "rows": frozen.row_count,
        "first_write": True,
        "immutable": True,
    }


# ---------------------------------------------------------------------------
# Scoring after decisions are frozen
# ---------------------------------------------------------------------------

def _fact_block(rows):
    facts = [row for row in rows if row["kind"] == "fact"]
    total = len(facts)
    selected = sum(row["selected_any"] for row in facts)
    identity = sum(row["fact_identity"] for row in facts)
    return {
        "n": total,
        "selected_n": selected,
        "abstained_n": total - selected,
        "identity_n": identity,
        "identity_rate": _rate(identity, total),
        "coverage": _rate(selected, total),
        "precision": _rate(identity, selected),
        "wrong_selection_n": selected - identity,
    }


def _selection_block(rows):
    total = len(rows)
    facts = [row for row in rows if row["kind"] == "fact"]
    unknown = [row for row in rows if row["kind"] == "unknown"]
    revoked = [row for row in rows if row["kind"] == "revoked"]
    fact_selected = sum(row["selected_any"] for row in facts)
    fact_identity = sum(row["fact_identity"] for row in facts)
    unknown_selected = sum(row["selected_any"] for row in unknown)
    revoked_selected = sum(row["selected_any"] for row in revoked)
    false_recall = sum(row["false_recall"] for row in rows)
    cross_world_selected = sum(row["cross_world_selected"] for row in rows)
    block = {
        "n": total,
        "kind_counts": {
            "fact": len(facts),
            "unknown": len(unknown),
            "revoked": len(revoked),
        },
        "fact_n": len(facts),
        "fact_selected_n": fact_selected,
        "fact_abstained_n": len(facts) - fact_selected,
        "fact_selection_identity_n": fact_identity,
        "fact_selection_identity": _rate(fact_identity, len(facts)),
        "fact_selection_coverage": _rate(fact_selected, len(facts)),
        "fact_selection_precision": _rate(fact_identity, fact_selected),
        "fact_wrong_selection_n": fact_selected - fact_identity,
        "unknown_n": len(unknown),
        "unknown_selected_n": unknown_selected,
        "unknown_abstained_n": len(unknown) - unknown_selected,
        "unknown_false_selection_rate": _rate(unknown_selected, len(unknown)),
        "revoked_n": len(revoked),
        "revoked_selected_n": revoked_selected,
        "revoked_abstained_n": len(revoked) - revoked_selected,
        "revoked_false_selection_rate": _rate(revoked_selected, len(revoked)),
        "abstention_n": sum(not row["selected_any"] for row in rows),
        "abstention_rate": _rate(
            sum(not row["selected_any"] for row in rows), total),
        "false_recall_n": false_recall,
        "false_recall_rate": _rate(false_recall, total),
        "cross_world_selected_n": cross_world_selected,
        "cross_world_correct_n": sum(row["cross_world_correct"]
                                     for row in rows),
        "cross_world_false_recall_n": sum(
            row["cross_world_false_recall"] for row in rows),
        "scope_violation_n": sum(row["scope_violation"] for row in rows),
        "correction": _fact_block(
            [row for row in rows if row["in_correction"]]),
        "unaffected": _fact_block(
            [row for row in rows if row["in_unaffected"]]),
    }
    return block


def _group(rows, key):
    groups = {}
    for row in rows:
        value = row.get(key)
        groups.setdefault("unknown" if value is None else str(value), []).append(row)
    return {name: _selection_block(groups[name]) for name in sorted(groups)}


def _split_metrics(rows):
    return {
        "overall": _selection_block(rows),
        "by_phase": _group(rows, "phase"),
        "by_state": _group(rows, "state"),
        "by_cohort": _group(rows, "cohort"),
        "by_world": _group(rows, "world_id"),
        "by_complexity": _group(rows, "complexity"),
        "counts": {
            "decisions": len(rows),
            "episodes": len({row["episode_id"] for row in rows}),
            "worlds": len({row["world_id"] for row in rows}),
            "states": len({(row["episode_id"], row["state"]) for row in rows}),
            "cohorts": len({row["cohort"] for row in rows}),
        },
    }


def _labels_map(labels):
    if isinstance(labels, Mapping) and "episodes" in labels \
            and isinstance(labels["episodes"], Mapping):
        return labels["episodes"]
    _require(isinstance(labels, Mapping), "Evaluation labels must be an object")
    return labels


def _score_row(decision, labels_by_episode):
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
    target = targets[query_id]
    kind = target.get("kind")
    _require(kind in ("fact", "unknown", "revoked"),
             f"Unknown label kind: {kind}")
    target_id = target.get("target_entry_id")
    if kind == "fact":
        _require(isinstance(target_id, str) and target_id,
                 "Fact target lacks a record identity")
    cohort = (state_labels.get("cohort_by_query") or {}).get(query_id)
    _require(isinstance(cohort, str) and cohort,
             "Scoring cohort is missing")
    label_split = state_labels.get("split", decision["split"])
    label_world = state_labels.get("world_id", decision["world_id"])
    _require(label_split == decision["split"],
             "Decision split differs from its evaluation label")
    _require(label_world == decision["world_id"],
             "Decision world differs from its evaluation label")

    selected_ids = tuple(decision["selected_ids"])
    selected_records = {record["record_id"]: record
                        for record in decision.get("selected_records", ())}
    _require(set(selected_ids) == set(selected_records),
             "Selected IDs and selected record records differ")
    selected_any = bool(selected_ids)
    exact = bool(kind == "fact" and selected_ids == (target_id,))
    selected_wrong = bool(kind == "fact" and selected_any and not exact)
    unknown_selected = bool(kind == "unknown" and selected_any)
    revoked_selected = bool(kind == "revoked" and selected_any)
    false_recall = bool(selected_wrong or unknown_selected or revoked_selected)
    cross_world = any(
        record.get("origin_world_id") != decision["world_id"]
        for record in selected_records.values())
    known_address_ids = set(decision["address_record_ids"])
    live_ids = set(decision["live_candidate_ids"])
    scope_violation = any(
        record_id not in known_address_ids or record_id not in live_ids
        for record_id in selected_ids)
    corrected = set(state_labels.get("corrected_query_ids") or ())
    unaffected = set(state_labels.get("unaffected_query_ids") or ())
    return {
        "episode_id": episode_id,
        "world_id": decision["world_id"],
        "split": decision["split"],
        "state": state,
        "phase": decision["phase"],
        "query_id": query_id,
        "cohort": cohort,
        "kind": kind,
        "complexity": state_labels.get("complexity"),
        "target_entry_id": target_id,
        "selected_ids": list(selected_ids),
        "selected_any": selected_any,
        "fact_identity": exact,
        "selected_wrong": selected_wrong,
        "unknown_selected": unknown_selected,
        "revoked_selected": revoked_selected,
        "false_recall": false_recall,
        "cross_world_selected": cross_world,
        "cross_world_correct": bool(cross_world and exact),
        "cross_world_false_recall": bool(cross_world and false_recall),
        "scope_violation": scope_violation,
        "in_correction": query_id in corrected,
        "in_unaffected": query_id in unaffected,
    }


def score_decisions(decisions, labels, *, config=None):
    """Join frozen decisions to evaluation labels; never select a record."""
    if isinstance(decisions, FrozenDecisions):
        rows = decisions.decisions
    else:
        rows = tuple(decisions)
    _require(rows, "No decisions are available for scoring")
    labels_by_episode = _labels_map(labels)
    scored = []
    seen = set()
    for decision in rows:
        key = (decision["episode_id"], decision["state"],
               decision["phase"], decision["query_id"])
        _require(key not in seen, "Decision keys repeat during scoring")
        seen.add(key)
        scored.append(_score_row(decision, labels_by_episode))
    splits = {}
    for split in SPLITS:
        subset = [row for row in scored if row["split"] == split]
        if subset:
            splits[split] = _split_metrics(subset)
    _require(splits, "No declared split was scored")
    aggregate = {
        "overall": _selection_block(scored),
        "by_split": {split: _selection_block(
            [row for row in scored if row["split"] == split])
            for split in splits},
        "by_cohort": _group(scored, "cohort"),
        "by_world": _group(scored, "world_id"),
    }
    return {
        "splits": splits,
        "aggregate": aggregate,
        "decision_rows": len(scored),
        "splits_present": list(splits),
        "note": ("Selection identity is exact record-ID equality against "
                 "evaluation labels; no bank value is compared."),
    }


def evaluate_dev_gate(dev_metrics, endpoint=None):
    """Apply the predeclared dev-only endpoint without any heldout input."""
    overall = dev_metrics.get("overall") if "overall" in dev_metrics \
        else dev_metrics
    identity = overall.get("fact_selection_identity")
    unknown_selected = overall.get("unknown_selected_n")
    revoked_selected = overall.get("revoked_selected_n")
    thresholds = endpoint or {
        "dev_fact_selection_identity_min": DEV_FACT_SELECTION_IDENTITY_MIN,
        "dev_unknown_selections_max": DEV_UNKNOWN_SELECTIONS_MAX,
        "dev_revoked_selections_max": DEV_REVOKED_SELECTIONS_MAX,
    }
    _require(thresholds.get("dev_fact_selection_identity_min")
             == DEV_FACT_SELECTION_IDENTITY_MIN,
             "Dev confidence threshold may not be tuned")
    _require(thresholds.get("dev_unknown_selections_max")
             == DEV_UNKNOWN_SELECTIONS_MAX,
             "Dev unknown threshold may not be tuned")
    _require(thresholds.get("dev_revoked_selections_max")
             == DEV_REVOKED_SELECTIONS_MAX,
             "Dev revoked threshold may not be tuned")
    checks = {
        "fact_selection_identity": {
            "value": identity,
            "operator": ">=",
            "threshold": DEV_FACT_SELECTION_IDENTITY_MIN,
            "passed": identity is not None
            and identity >= DEV_FACT_SELECTION_IDENTITY_MIN,
            "identity_n": overall.get("fact_selection_identity_n"),
            "fact_n": overall.get("fact_n"),
        },
        "unknown_selections": {
            "value": unknown_selected,
            "operator": "<=",
            "threshold": DEV_UNKNOWN_SELECTIONS_MAX,
            "passed": unknown_selected is not None
            and unknown_selected <= DEV_UNKNOWN_SELECTIONS_MAX,
        },
        "revoked_selections": {
            "value": revoked_selected,
            "operator": "<=",
            "threshold": DEV_REVOKED_SELECTIONS_MAX,
            "passed": revoked_selected is not None
            and revoked_selected <= DEV_REVOKED_SELECTIONS_MAX,
        },
    }
    return {
        "evaluated_split": "dev",
        "evaluated_first": True,
        "passed": all(check["passed"] for check in checks.values()),
        "checks": checks,
    }


# ---------------------------------------------------------------------------
# Frozen control orchestration
# ---------------------------------------------------------------------------

def decision_key_register(decisions):
    keys = [(row["episode_id"], row["state"], row["phase"], row["query_id"])
            for row in decisions]
    return {
        "rows": len(keys),
        "unique_keys": len(set(keys)),
        "unique": len(keys) == len(set(keys)),
    }


def _decision_counts(decisions):
    counts = {}
    for split in SPLITS:
        subset = [row for row in decisions if row["split"] == split]
        if subset:
            counts[split] = {
                "decisions": len(subset),
                "episodes": len({row["episode_id"] for row in subset}),
                "states": len({(row["episode_id"], row["state"])
                               for row in subset}),
                "worlds": len({row["world_id"] for row in subset}),
            }
    return counts


def _verify_written(output, frozen, manifest):
    path = Path(output) / manifest["filename"]
    _require(path.is_file(), f"Frozen decision file is missing: {manifest['filename']}")
    _require(_sha_file(path) == frozen.sha256,
             f"Frozen decision file changed: {manifest['filename']}")
    _require(manifest["sha256"] == frozen.sha256,
             "Frozen decision manifest hash changed")
    return True


def _classification(config):
    classification = dict(config.get("classification") or {})
    classification.update(CLASSIFICATION)
    classification.setdefault("not_native", True)
    classification["value_comparison"] = "none"
    classification["compressed_arm"] = "absent"
    classification["oracle_role"] = "diagnostic_only"
    return classification


def _base_report(config, *, prepared_info, config_path):
    return {
        "format": REPORT_FORMAT,
        "generated_at_utc": _utc_now(),
        "classification": _classification(config),
        "roles": config["roles"],
        "protocol": {
            "config_path": str(config_path) if config_path else None,
            "config_sha256": (_sha_file(config_path)
                              if config_path and Path(config_path).is_file()
                              else None),
            "decision_rule_sha256": _sha_json({
                "bank": config["bank"],
                "reader": config["reader"],
                "endpoint": config["endpoint"],
            }),
            "prepared": prepared_info,
            "reader_layer_independence": config["reader"],
            "selection_rule": (
                "exact (entity, attribute) match against live records minus "
                "the declared view-only revocation"),
            "value_comparison": "none",
        },
        "gating": {
            "order": ["dev"],
            "development_evaluated_first": True,
            "dev": None,
            "heldout": {
                "status": "not_run",
                "evaluated_after_dev_pass": False,
            },
            "stopped_before_heldout": True,
        },
        "raw_outputs": {
            "policy": config["reporting"]["raw_output_policy"],
            "files": {},
            "verified_unchanged": True,
        },
        "artifact_policy": {
            "native_jobs": 0,
            "model_loads": 0,
            "tokenizer_loads": 0,
            "forward_passes": 0,
            "optimizer_updates": 0,
            "gradients": 0,
            "checkpoints_written": 0,
            "native_resource_reports": 0,
            "reports": "small JSON/JSONL only",
            "model_execution": False,
            "generation": False,
        },
        "limitations": list(config["limitations"]),
    }


def run_control(episodes, revocations, labels_loader, config, output, *,
                prepared_info=None, config_path=None):
    """Run dev-first gating over already loaded selection inputs.

    ``labels_loader`` is called only after the development decisions have been
    frozen and written.  A callback may therefore assert that no label payload
    is touched before the immutable decision file exists.
    """
    validate_config(config)
    _require(callable(labels_loader), "A scoring-label loader is required")
    episodes = tuple(episodes)
    revocations = dict(revocations)
    output = _fresh_output(output)
    _require(expected_decision_rows(episodes) > 0,
             "Prepared episodes declare no decisions")
    dev_decisions = build_split_decisions(episodes, revocations, config, "dev")
    frozen_dev = freeze_decisions(dev_decisions, split="dev")
    raw_files = {"dev": write_frozen_decisions(output, frozen_dev)}
    _require(decision_key_register(frozen_dev.decisions)["unique"],
             "Dev decision keys are not unique")

    labels_doc = labels_loader()
    labels = labels_doc.get("labels", labels_doc) \
        if isinstance(labels_doc, Mapping) else labels_doc
    dev_scored = score_decisions(frozen_dev, labels)
    dev_metrics = dev_scored["splits"]["dev"]
    dev_gate = evaluate_dev_gate(dev_metrics, config["endpoint"])

    report = _base_report(config, prepared_info=prepared_info,
                          config_path=config_path)
    report["raw_outputs"]["files"] = dict(raw_files)
    report["gating"]["dev"] = dev_gate

    if not dev_gate["passed"]:
        report["counts"] = {
            "decisions": frozen_dev.row_count,
            "by_split": _decision_counts(frozen_dev.decisions),
            "decision_keys": decision_key_register(frozen_dev.decisions),
        }
        report["split_metrics"] = {"dev": dev_metrics}
        report["aggregate_metrics"] = dev_scored["aggregate"]
        report["gating"]["heldout"] = {
            "status": "not_run",
            "evaluated_after_dev_pass": False,
            "reason": "dev_endpoint_failed",
            "decisions": 0,
            "metrics": None,
        }
        report["gating"]["stopped_before_heldout"] = True
        report["outcome"] = {
            "status": "stopped_at_dev",
            "dev": "failed",
            "heldout": "not_run",
            "claim": "no heldout decision or metric was produced",
        }
        report["raw_outputs"]["verified_unchanged"] = _verify_written(
            output, frozen_dev, raw_files["dev"])
        _write_new(output / "report.json",
                   (_canonical(report) + "\n").encode("utf-8"))
        return report

    train_decisions = ()
    heldout_decisions = ()
    train_frozen = None
    heldout_frozen = None
    if any(episode["split"] == "train" for episode in episodes):
        train_decisions = build_split_decisions(
            episodes, revocations, config, "train")
        train_frozen = freeze_decisions(train_decisions, split="train")
        raw_files["train"] = write_frozen_decisions(output, train_frozen)
    _require(any(episode["split"] == "heldout" for episode in episodes),
             "The heldout split is required after dev passes")
    heldout_decisions = build_split_decisions(
        episodes, revocations, config, "heldout")
    heldout_frozen = freeze_decisions(heldout_decisions, split="heldout")
    raw_files["heldout"] = write_frozen_decisions(output, heldout_frozen)

    all_decisions = train_decisions + frozen_dev.decisions + heldout_decisions
    all_scored = score_decisions(all_decisions, labels)
    _require(all_scored["splits"]["dev"]["overall"] == dev_metrics["overall"],
             "Dev metrics changed after full frozen scoring")
    report["raw_outputs"]["files"] = dict(raw_files)
    verified = True
    for split, frozen in (("train", train_frozen),
                          ("dev", frozen_dev),
                          ("heldout", heldout_frozen)):
        if frozen is not None:
            verified = verified and _verify_written(
                output, frozen, raw_files[split])
    report["raw_outputs"]["verified_unchanged"] = verified
    report["counts"] = {
        "decisions": len(all_decisions),
        "by_split": _decision_counts(all_decisions),
        "decision_keys": decision_key_register(all_decisions),
    }
    report["split_metrics"] = all_scored["splits"]
    report["aggregate_metrics"] = all_scored["aggregate"]
    report["gating"]["heldout"] = {
        "status": "evaluated",
        "evaluated_after_dev_pass": True,
        "reason": None,
        "decisions": heldout_frozen.row_count,
        "metrics": all_scored["splits"]["heldout"],
        "endpoint_enforced": False,
        "role": "post_freeze_observation_only",
    }
    report["gating"]["stopped_before_heldout"] = False
    report["outcome"] = {
        "status": "dev_passed_heldout_evaluated",
        "dev": "passed",
        "heldout": "evaluated",
        "claim": ("explicit-record selection evidence under a frozen rule; "
                  "not a native, generation or calibrated capability claim"),
    }
    _write_new(output / "report.json",
               (_canonical(report) + "\n").encode("utf-8"))
    return report


def run_prepared_control(config, *, prepared=None, output, config_path=None):
    """Verify pinned prepared inputs, freeze dev, then load labels and score."""
    validate_config(config)
    prepared_root = _prepared_root(prepared, config)
    selection = load_selection_inputs(prepared_root, config)
    prepared_info = {
        "root": str(prepared_root),
        "mode": "pinned_prepared_inputs",
        "hashes": selection["hashes"],
        "manifest_sha256": selection["hashes"]["manifest_sha256"],
        "selection_inputs": list(config["prepared"]["selection_inputs"]),
        "scoring_only_inputs": list(config["prepared"]["scoring_only_inputs"]),
    }

    def labels_loader():
        return load_scoring_labels(prepared_root, config)["labels"]

    return run_control(
        selection["episodes"], selection["revocations"], labels_loader,
        config, output, prepared_info=prepared_info, config_path=config_path)

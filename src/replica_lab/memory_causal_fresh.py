"""Fresh disjoint causal-selector evaluation with versioned durable writes.

The v2 protocol closes the Task 5 gaps left by ``memory_causal_refresh``:

* a fresh, disjoint paraphrase/distractor/correction/unknown fixture with
  explicit train/dev/heldout roles and post-restart queries;
* deterministic full/causal/shuffled/disabled/oracle selector arms;
* durable ingestion that is idempotent per ``(episode, record, version)`` and
  rejects duplicate or restart replay of the identical record version;
* a separately identified, query-local accepted refresh that is admitted only
  for a live record matching its committed durable version.  Refresh never
  writes a durable record, optimizer update, decay or persistent state; it
  rebuilds a query-local history for the current readout only, and stable
  attempt IDs retry idempotently;
* live revocation and revoked-query controls.  Runtime selection consumes only
  runtime-available revocation/staleness metadata.  A revoked-target query that
  selects another live record is scored as wrong retrieval, never as an
  evaluator-induced abstention;
* checkpoints bound to protocol, fixture, source/model/tokenizer, writer,
  representation and implementation hashes.  Only durable state is serialized;
  query-local refresh attempts are audit metadata;
* restart evaluation of genuinely new queries with zero old-observation
  re-encoding and byte-identical same-query parity;
* per-phase cached-history encode, rebuild, readout, checkpoint and wall-clock
  cost accounting;
* inventory accounting in explicit units: declared query rows, inventory case
  rows and arm-expanded metric rows are reconciled rather than conflated.

Raw rows stay diagnostics.  Candidate quality in the new scorer uses the actual
``query_key_value`` readout.  ``exact_key_value`` is a mixture diagnostic, not
an upper bound, and oracle rows are a separate gold-target population.  Nothing
here trains, unfreezes, or promotes the rejected writer, and no learned-selector
or generation claim is made.
"""
from __future__ import annotations

from contextlib import contextmanager
import copy
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

import torch

from .architecture.contracts import fingerprint
from .memory_acceptance_v3 import (
    acceptance_metrics as acceptance_metrics_v3,
    evaluate_acceptance as evaluate_acceptance_v3,
    validate_config as validate_acceptance_v3,
)
from .memory_diagnosis import (
    _cosine_heads, _display_path, _extension_pins, _freeze, _parameter_versions,
    _resolve_path, _sha_file, _sha_text, _source_pins, _utc_now, _write_json,
    _write_jsonl, observed_binding, query_key,
)
from .memory_matched import (
    CORRECTION_DAY, FACT_REPRESENTATION, INITIAL_DAY, SPLITS, _entry_role,
    _fact_of, _record_views, episode_states, generate_matched_episodes,
    render_fact_query, validate_matched_episodes,
)
from .memory_causal_refresh import _history_writes
from .memory_update_dynamics import (
    _build_state, _observation_sha, _read_weights, _readout_row,
    _recompute_ranking, _sha_json, _state_sha,
)
from .neural import SGDN2Memory, SGDN2Write

CONFIG_FORMAT = "replica.memory_causal_fresh.config.v2"
PROTOCOL_FORMAT = "replica.memory_causal_fresh.protocol.v2"
EPISODES_FORMAT = "replica.memory_causal_fresh.episodes.v2"
LABELS_FORMAT = "replica.memory_causal_fresh.labels.v2"
REVOCATIONS_FORMAT = "replica.memory_causal_fresh.revocations.v2"
PREPARED_FORMAT = "replica.memory_causal_fresh.prepared.v2"
INVENTORY_FORMAT = "replica.memory_causal_fresh.inventory.v2"
CHECKPOINT_FORMAT = "replica.memory_causal_fresh.checkpoint.v3"
RUN_FORMAT = "replica.memory_causal_fresh.run.v2"
DECISION_FORMAT = "replica.memory_causal_fresh.decision.v2"
LEDGER_FORMAT = "replica.memory_causal_fresh.ledger.v2"
REFRESH_AUDIT_FORMAT = "replica.memory_causal_fresh.refresh_audit.v1"

FULL = "shared_mean/full"
CAUSAL = "shared_mean/causal_query_refresh"
SHUFFLED = "shared_mean/causal_query_refresh_shuffled_values"
DISABLED = "shared_mean/disabled"
ORACLE = "shared_mean/oracle_target_refresh_last"
VARIANT_NAMES = (FULL, CAUSAL, SHUFFLED, DISABLED, ORACLE)

FIRST_PROCESS = "first_process"
RESTART = "restart"
REVOCATION = "revocation"
PHASE_ORDER = (FIRST_PROCESS, RESTART, REVOCATION)
PHASE_QUERY_FIELDS = {
    FIRST_PROCESS: "queries",
    RESTART: "restart_queries",
    REVOCATION: "revocation_queries",
}
FORBIDDEN_ROW_KEYS = frozenset({"expected", "expected_answer", "gold_answer"})
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _canonical(value):
    return json.dumps(value, sort_keys=True, allow_nan=False)


def _canonical_bytes(value):
    return (_canonical(value) + "\n").encode("utf-8")


def _rows_sha(rows):
    return _sha_text("".join(_canonical(row) + "\n" for row in rows))


def _check_no_label_keys(value, *, path="row"):
    if isinstance(value, dict):
        for key, item in value.items():
            _require(key not in FORBIDDEN_ROW_KEYS,
                     f"Raw policy row contains an evaluator answer: {path}.{key}")
            _check_no_label_keys(item, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _check_no_label_keys(item, path=f"{path}[{index}]")


def _opaque(prefix, key):
    return f"{prefix}-{hashlib.sha256(key.encode('utf-8')).hexdigest()[:10]}"


# ---------------------------------------------------------------------------
# Fixture generation and labels
# ---------------------------------------------------------------------------

def _query(query_id, entity, attribute, form, phase, cohort):
    return {
        "query_id": query_id, "entity": entity, "attribute": attribute,
        "form": form, "text": render_fact_query(entity, attribute, form),
        "phase": phase, "cohort": cohort,
    }


def _record_addresses(episode):
    addresses = set()
    for group in ("facts", "distractors", "corrections"):
        for record in episode[group]:
            addresses.add((record["entity"], record["attribute"]))
    return addresses


def _restart_unknown(episode, *, attempts=64):
    for attempt in range(attempts):
        entity = _opaque(
            "ent", f"{episode['episode_id']}:restart-unknown:{attempt}:entity")
        attribute = _opaque(
            "attr", f"{episode['episode_id']}:restart-unknown:{attempt}:attribute")
        taken = _record_addresses(episode)
        for query in episode["queries"] + episode.get("restart_queries", []):
            taken.add((query["entity"], query["attribute"]))
        if (entity, attribute) not in taken:
            return entity, attribute
    raise ValueError("Could not allocate a fresh restart-unknown address")


def _augment_episode(episode, config):
    plan = config["query_plan"]
    first_forms = tuple(plan["first_process_forms"])
    restart_forms = tuple(plan["restart_forms"])
    _require(len(first_forms) == len(restart_forms) == 2,
             "The fresh query plan requires two paraphrase forms per phase")
    _require(not set(first_forms) & set(restart_forms),
             "First-process and restart paraphrase forms must be disjoint")
    facts = episode["facts"]
    distractors = episode["distractors"]
    _require(distractors, "A fresh episode requires a distractor for revocation")
    base_unknown = next(query for query in episode["queries"]
                        if query["query_id"] == "unknown")
    observed = _record_addresses(episode)
    _require((base_unknown["entity"], base_unknown["attribute"]) not in observed,
             "The declared unknown address is observed")

    augmented = dict(episode)
    first, restart = [], []
    for index, fact in enumerate(facts):
        for suffix, form in zip(("a", "b"), first_forms):
            first.append(_query(f"p{index}{suffix}", fact["entity"],
                                fact["attribute"], form, FIRST_PROCESS,
                                "paraphrase"))
    first_distractor = distractors[0]
    first.append(_query("p-distractor", first_distractor["entity"],
                        first_distractor["attribute"],
                        plan["distractor_forms"]["first_process"],
                        FIRST_PROCESS, "distractor"))
    first.append(_query("p-unknown", base_unknown["entity"],
                        base_unknown["attribute"],
                        plan["unknown_forms"]["first_process"],
                        FIRST_PROCESS, "unknown"))
    for index, fact in enumerate(facts):
        for suffix, form in zip(("a", "b"), restart_forms):
            restart.append(_query(f"r{index}{suffix}", fact["entity"],
                                  fact["attribute"], form, RESTART,
                                  "paraphrase"))
    restart_distractor = distractors[1] if len(distractors) > 1 else distractors[0]
    restart.append(_query("r-distractor", restart_distractor["entity"],
                          restart_distractor["attribute"],
                          plan["distractor_forms"]["restart"], RESTART,
                          "distractor"))
    augmented["queries"] = first
    augmented["restart_queries"] = restart
    restart_entity, restart_attribute = _restart_unknown(augmented)
    restart.append(_query("r-unknown", restart_entity, restart_attribute,
                          plan["unknown_forms"]["restart"], RESTART, "unknown"))
    _require(len({query["query_id"] for query in first}) == len(first),
             "First-process query IDs must be unique")
    _require(len({query["query_id"] for query in restart}) == len(restart),
             "Restart query IDs must be unique")
    _require(not ({query["text"] for query in first}
                  & {query["text"] for query in restart}),
             "Restart queries must be genuinely new versus the first process")
    augmented["revocation_queries"] = [_query(
        "v-distractor", first_distractor["entity"],
        first_distractor["attribute"],
        plan["distractor_forms"]["first_process"], REVOCATION, "revoked")]
    _require(augmented["revocation_queries"][0]["text"]
             == next(query for query in first
                     if query["query_id"] == "p-distractor")["text"],
             "The revocation control must reuse the live distractor query text")
    return augmented


def _entry_public(record, role):
    return {
        "record_id": record["record_id"], "entity": record["entity"],
        "attribute": record["attribute"], "value": record["value"],
        "role": role,
        "source_family": "independent_world" if role == "distractor" else "source",
        "origin_world_id": record["origin_world_id"],
        "observed_at": record["observed_at"], "live": role != "superseded_negative",
    }


def _target_row(query, *, kind, entry, role, source_family, expected, target_id):
    return {
        "expected": expected, "kind": kind, "target_entry_id": target_id,
        "target_role": role, "target_source_family": source_family,
        "query_form": query["form"],
        "query_text_sha256": _sha_json(query["text"]),
    }


def derive_fresh_labels(episode, state, revocation):
    """Derive evaluation-only labels from observed records and declared controls."""
    records, live, negative = _record_views(episode, state)
    role_by_id = {}
    live_entries, negative_entries = [], []
    for record in live:
        role = _entry_role(record, state, episode)
        role_by_id[record["record_id"]] = role
        live_entries.append(_entry_public(record, role))
    for record in negative:
        role_by_id[record["record_id"]] = "superseded_negative"
        negative_entries.append(_entry_public(record, "superseded_negative"))
    live_by_key = {(entry["entity"], entry["attribute"]): entry
                   for entry in live_entries}
    revoked_id = revocation["record_id"]
    _require(revoked_id in {entry["record_id"] for entry in live_entries},
             "The revocation target is not live in the declared state")
    _require(revoked_id not in {entry["record_id"] for entry in negative_entries},
             "The revocation target is stale in the declared state")

    targets, phase_by_query, cohort_by_query = {}, {}, {}
    for phase in (FIRST_PROCESS, RESTART):
        unknown = 0
        for query in episode[PHASE_QUERY_FIELDS[phase]]:
            query_id = query["query_id"]
            _require(query_id not in targets, "Query IDs must be unique per state")
            entry = live_by_key.get((query["entity"], query["attribute"]))
            if entry is None:
                _require(query["cohort"] == "unknown",
                         f"Only a declared unknown query may be unobserved: {query_id}")
                unknown += 1
                targets[query_id] = _target_row(
                    query, kind="unknown", entry=None, role="unknown",
                    source_family="none", expected="unknown", target_id=None)
            else:
                _require(query["cohort"] != "unknown",
                         f"A declared unknown query resolved to a live record: {query_id}")
                targets[query_id] = _target_row(
                    query, kind="fact", entry=entry, role=entry["role"],
                    source_family=entry["source_family"],
                    expected=entry["value"], target_id=entry["record_id"])
            phase_by_query[query_id] = phase
            cohort_by_query[query_id] = query["cohort"]
        _require(unknown == 1,
                 f"The {phase} phase must expose exactly one unknown query")
    for query in episode["revocation_queries"]:
        query_id = query["query_id"]
        _require(query_id not in targets, "Revocation query IDs must be unique")
        targets[query_id] = _target_row(
            query, kind="revoked", entry=None, role="revoked",
            source_family="source", expected="revoked", target_id=None)
        phase_by_query[query_id] = REVOCATION
        cohort_by_query[query_id] = "revoked"
    corrected = [query_id for query_id, target in targets.items()
                 if target["kind"] == "fact"
                 and target["target_role"] == "correction_target"]
    unaffected = [query_id for query_id, target in targets.items()
                  if target["kind"] == "fact" and target["target_role"] == "retained"]
    unknown_ids = sorted(query_id for query_id, target in targets.items()
                         if target["kind"] == "unknown")
    if state["name"] == "corrected" and episode["corrections"]:
        _require(corrected, "The corrected state must expose a correction target")
        _require(unaffected, "The corrected state must retain an unaffected fact")
    return {
        "episode_id": episode["episode_id"], "world_id": episode["world_id"],
        "family_id": episode["family_id"], "split": episode["split"],
        "complexity": episode["complexity"], "variant": episode["variant"],
        "state": state["name"], "cutoff": state["cutoff"],
        "observed_record_ids": [record["record_id"] for record in records],
        "live_entries": live_entries, "negative_entries": negative_entries,
        "revoked_entry_ids": [revoked_id],
        "targets": targets, "phase_by_query": phase_by_query,
        "cohort_by_query": cohort_by_query,
        "corrected_query_ids": sorted(corrected),
        "unaffected_query_ids": sorted(unaffected),
        "unknown_query_ids": unknown_ids,
        "evaluation_only": True,
    }


def generate_fresh_fixture(config):
    """Generate the deterministic fresh fixture without a model or tokenizer."""
    base_plan = config["base_generator"]
    base = generate_matched_episodes(
        seed=config["seed"],
        families_per_split=base_plan["families_per_split"],
        complexities=tuple(base_plan["complexities"]),
        variants=base_plan["variants"])
    validate_matched_episodes(
        base, families_per_split=base_plan["families_per_split"],
        complexities=tuple(base_plan["complexities"]),
        variants=base_plan["variants"])
    episodes = [_augment_episode(copy.deepcopy(episode), config) for episode in base]
    revocations = {}
    for episode in episodes:
        target = episode["distractors"][0]
        revocations[episode["episode_id"]] = {
            "record_id": target["record_id"],
            "entity": target["entity"], "attribute": target["attribute"],
            "applies_at": config["revocation"]["applies_at"],
            "scope": config["revocation"]["scope"],
            "cached_history_retained": True,
        }
    labels_by_episode = {}
    for episode in episodes:
        by_state = {}
        for state in episode_states(episode):
            by_state[state["name"]] = derive_fresh_labels(
                episode, state, revocations[episode["episode_id"]])
        labels_by_episode[episode["episode_id"]] = by_state
    audit = validate_fresh_fixture(
        episodes, labels_by_episode, revocations, config=config)
    return episodes, labels_by_episode, revocations, audit


def _iter_records(episode):
    for group in ("facts", "distractors", "corrections"):
        for record in episode[group]:
            yield record


def _leak_check(episode):
    values = {record["value"] for record in _iter_records(episode)}
    for query in (episode["queries"] + episode["restart_queries"]
                  + episode["revocation_queries"]):
        text = query["text"]
        _require("val-" not in text, "Query text leaks a value token")
        for value in values:
            _require(value not in text, "Query text leaks an observed value")


def _label_free_episode(episode):
    def scan(value, path):
        if isinstance(value, dict):
            for key, item in value.items():
                lowered = key.lower()
                _require(not any(marker in lowered for marker in (
                    "expected", "gold", "answer", "label")),
                    f"Episode input carries an evaluator field: {path}.{key}")
                scan(item, f"{path}.{key}")
        elif isinstance(value, list):
            for index, item in enumerate(value):
                scan(item, f"{path}[{index}]")
    scan(episode, "episode")


def validate_fresh_fixture(episodes, labels_by_episode, revocations, *, config,
                           base_episodes=None):
    _require(isinstance(episodes, list) and episodes, "Fresh episodes are missing")
    episode_ids, world_ids, record_ids, values = set(), set(), set(), set()
    split_counts = {split: 0 for split in SPLITS}
    state_count = query_count = 0
    for episode in episodes:
        episode_id = episode["episode_id"]
        _require(episode_id not in episode_ids, "Episode IDs must be unique")
        episode_ids.add(episode_id)
        _require(episode["world_id"] not in world_ids, "World IDs must be unique")
        world_ids.add(episode["world_id"])
        _require(episode["split"] in SPLITS, "Episode split differs")
        split_counts[episode["split"]] += 1
        _require(set(PHASE_QUERY_FIELDS.values()) <= set(episode),
                 "Episode phase query sets are missing")
        _require(revocations[episode_id]["record_id"]
                 == episode["distractors"][0]["record_id"],
                 "Revocation target differs from the declared distractor")
        _label_free_episode(episode)
        _leak_check(episode)
        _require(not ({query["text"] for query in episode["queries"]}
                      & {query["text"] for query in episode["restart_queries"]}),
                 "Restart queries repeat first-process text")
        _require(not ({query["query_id"] for query in episode["queries"]}
                      & {query["query_id"] for query in episode["restart_queries"]}),
                 "Restart query IDs repeat the first-process IDs")
        _require(all(query["phase"] == FIRST_PROCESS for query in episode["queries"]),
                 "First-process phase marker differs")
        _require(all(query["phase"] == RESTART
                     for query in episode["restart_queries"]),
                 "Restart phase marker differs")
        _require(all(query["phase"] == REVOCATION
                     for query in episode["revocation_queries"]),
                 "Revocation phase marker differs")
        for record in _iter_records(episode):
            _require(record["record_id"] not in record_ids,
                     "Record IDs must be unique")
            record_ids.add(record["record_id"])
            values.add(record["value"])
        _require(len({fact["value"] for fact in episode["facts"]})
                 == len(episode["facts"]),
                 "Source fact values must be unique within an episode")
        stored = labels_by_episode[episode_id]
        for state in episode_states(episode):
            state_count += 1
            _require(state["name"] in stored, "State labels are missing")
            expected = derive_fresh_labels(
                episode, state, revocations[episode_id])
            _require(stored[state["name"]] == expected,
                     "Stored labels differ from the derived labels")
            query_count += len(expected["targets"])
    for split in SPLITS:
        _require(split_counts[split] > 0, f"The {split} split is empty")
    if base_episodes is not None:
        base_episode_ids = {episode["episode_id"] for episode in base_episodes}
        base_world_ids = {episode["world_id"] for episode in base_episodes}
        base_records = {record["record_id"]
                        for episode in base_episodes for record in _iter_records(episode)}
        base_values = {record["value"]
                       for episode in base_episodes for record in _iter_records(episode)}
        _require(not (episode_ids & base_episode_ids), "Fresh episode IDs overlap the base fixture")
        _require(not (world_ids & base_world_ids), "Fresh world IDs overlap the base fixture")
        _require(not (record_ids & base_records), "Fresh record IDs overlap the base fixture")
        _require(not (values & base_values), "Fresh values overlap the base fixture")
    return {
        "episodes": len(episodes), "states": state_count,
        "declared_queries": query_count,
        "splits": split_counts,
        "complexities": sorted({episode["complexity"] for episode in episodes}),
        "first_process_queries": sum(len(episode["queries"]) for episode in episodes),
        "restart_queries": sum(len(episode["restart_queries"]) for episode in episodes),
        "revocation_queries": sum(len(episode["revocation_queries"])
                                  for episode in episodes),
        "paraphrase_texts": sum(1 for episode in episodes
                                for query in episode["queries"] + episode["restart_queries"]
                                if query["cohort"] == "paraphrase"),
        "distractor_texts": sum(1 for episode in episodes
                                for query in episode["queries"] + episode["restart_queries"]
                                if query["cohort"] == "distractor"),
        "unknown_texts": sum(1 for episode in episodes
                             for query in episode["queries"] + episode["restart_queries"]
                             if query["cohort"] == "unknown"),
        "revocation_texts": sum(len(episode["revocation_queries"])
                                for episode in episodes),
        "restart_texts_disjoint": True,
        "label_free_inputs": True,
        "answer_leakage": False,
        "disjoint_from_base": base_episodes is not None,
    }


# ---------------------------------------------------------------------------
# Prepared fixture pins and planned work
# ---------------------------------------------------------------------------

ARM_CASE_COUNTS = {"fact": 4, "unknown": 3, "revoked": 3}


def _inventory_rows(episodes, labels_by_episode):
    rows = []
    for episode in episodes:
        labels_states = labels_by_episode[episode["episode_id"]]
        for state_name in (state["name"] for state in episode_states(episode)):
            labels = labels_states[state_name]
            for phase in PHASE_ORDER:
                for query in episode[PHASE_QUERY_FIELDS[phase]]:
                    target = labels["targets"][query["query_id"]]
                    rows.append([
                        episode["episode_id"], episode["split"], state_name,
                        phase, query["query_id"], query["cohort"],
                        target["kind"], target["target_role"]])
    return rows


def planned_workload(episodes, layers, labels_by_episode=None):
    layer_count = len(layers)
    observation_encodes = query_first = query_restart = contexts = declared = 0
    inventory_case_rows = 0
    for episode in episodes:
        states = episode_states(episode)
        contexts += len(states)
        phase_queries = sum(len(episode[field])
                            for field in PHASE_QUERY_FIELDS.values())
        declared += phase_queries
        inventory_case_rows += len(states) * phase_queries
        seen = set()
        for state in states:
            records, _, _ = _record_views(episode, state)
            seen.update(record["record_id"] for record in records)
        observation_encodes += len(seen) * layer_count
        first_texts = {query["text"] for query in episode["queries"]}
        query_first += len(first_texts) * layer_count
        encoded = set(first_texts)
        new_restart = {query["text"] for query in episode["restart_queries"]} - encoded
        query_restart += len(new_restart) * layer_count
        encoded |= new_restart
        new_revocation = ({query["text"] for query in episode["revocation_queries"]}
                          - encoded)
        query_restart += len(new_revocation) * layer_count
    if labels_by_episode is None:
        arm_expanded = None
    else:
        rows = _inventory_rows(episodes, labels_by_episode)
        _require(len(rows) == inventory_case_rows,
                 "Inventory case rows differ from the declared state expansion")
        arm_expanded = sum(
            ARM_CASE_COUNTS[row[6]] * len(VARIANT_NAMES) * layer_count
            for row in rows)
    return {
        "episodes": len(episodes), "contexts": contexts,
        "layers": layer_count,
        "observation_encodes_first_process": observation_encodes,
        "query_encodes_first_process": query_first,
        "query_encodes_restart": query_restart,
        "state_rebuilds_per_process": contexts * 2,
        "declared_query_rows": declared,
        "inventory_case_rows": inventory_case_rows,
        "arm_expanded_metric_rows": arm_expanded,
        "units": {
            "declared_query_rows": (
                "unique declared query rows per episode across phases; no state "
                "duplication"),
            "inventory_case_rows": (
                "episode/state/phase/query cases; states duplicate the declared "
                "phase queries"),
            "arm_expanded_metric_rows": (
                "inventory case rows x variants x layers x arms, with "
                "exact_key_value present only for fact targets"),
            "reconciliation": (
                "inventory_case_rows == sum(state_count * phase_query_count); "
                "arm_expanded_metric_rows == sum(arm_case_counts[kind] * "
                "variants * layers over inventory_case_rows)"),
        },
    }


def build_inventory(episodes, labels_by_episode, layers=None):
    rows = _inventory_rows(episodes, labels_by_episode)
    per_layer = sum(
        ARM_CASE_COUNTS[row[6]] * len(VARIANT_NAMES) for row in rows)
    units = {
        "inventory_case_rows": len(rows),
        "declared_query_rows": sum(
            sum(len(episode[field]) for field in PHASE_QUERY_FIELDS.values())
            for episode in episodes),
        "per_layer_arm_expanded_metric_rows": per_layer,
        "arm_case_counts": dict(ARM_CASE_COUNTS),
        "variants": len(VARIANT_NAMES),
        "note": (
            "declared query rows are before state duplication; inventory case "
            "rows are state-expanded; arm-expanded rows multiply by variants, "
            "layers and kind-eligible arms (exact_key_value only for facts)"),
    }
    if layers is not None:
        units["layers"] = len(layers)
        units["arm_expanded_metric_rows"] = per_layer * len(layers)
    return {"format": INVENTORY_FORMAT, "rows": rows, "units": units}


def _validate_config(config):
    _require(isinstance(config, dict), "Fresh configuration must be an object")
    _require(config.get("format") == CONFIG_FORMAT,
             "Fresh configuration format differs")
    for key in ("declared_at_utc", "purpose", "seed", "base_generator",
                "query_plan", "revocation", "restart", "layers", "engine",
                "representation", "numeric_policy", "variant_names",
                "thresholds", "selector", "acceptance", "limits",
                "stop_conditions", "budget", "resource_policy",
                "checkpoint_policy", "raw_output_policy", "source",
                "writer_extension", "implementation_paths", "limitations",
                "disjoint_from", "durable_ingestion", "query_local_refresh",
                "inventory_units"):
        _require(key in config, f"Fresh configuration is missing: {key}")
    _require(type(config["seed"]) is int and config["seed"] >= 0,
             "Fresh fixture seed must be a nonnegative integer")
    _require(tuple(config["variant_names"]) == VARIANT_NAMES,
             "Fresh variant inventory differs")
    durable = config["durable_ingestion"]
    _require(durable.get("identity")
             == "episode_id/record_id/record_version",
             "Durable ingestion identity differs")
    _require(durable.get("record_version")
             == "sha256(canonical(record_id,entity,attribute,value,"
                "observed_at,supersedes,origin_world_id))",
             "Durable record-version rule differs")
    _require(durable.get("duplicate_replay") == "reject"
             and durable.get("restart_replay") == "reject"
             and durable.get("refresh_source") == "excluded",
             "Durable replay/refresh policy differs")
    refresh = config["query_local_refresh"]
    _require(refresh.get("enabled") is True
             and refresh.get("owner") == "query_local_refresh",
             "Query-local refresh must be explicitly enabled and owned")
    _require(refresh.get("persistence") == "query_local_only"
             and refresh.get("persistent_memory_writes") == 0
             and refresh.get("optimizer_updates") == 0
             and refresh.get("decay_updates") == 0,
             "Refresh must not persist or mutate durable state")
    _require(refresh.get("requires_live_record") is True
             and refresh.get("requires_durable_version") is True,
             "Refresh must require a live record matching its durable version")
    retry = refresh.get("retry") or {}
    _require(retry.get("attempt_id") == "stable_per_variant_episode_state_query_layer"
             and retry.get("idempotent") is True,
             "Query-local refresh retry policy differs")
    per_query = refresh.get("per_query") or {}
    _require(type(per_query.get("max_applied_per_query")) is int
             and per_query["max_applied_per_query"] >= 1,
             "Query-local refresh requires a positive per-query bound")
    positive = refresh.get("positive_control") or {}
    _require(positive.get("admitted_refresh_changes_state") is True
             and positive.get("shuffled_refresh_perturbs_values") is True
             and isinstance(positive.get("oracle_recovery_raw_top1_min"),
                            (int, float)),
             "Admitted-refresh positive controls are undeclared")
    selector = config["selector"]
    _require(selector.get("runtime_inputs_only") is True
             and "sealed evaluator labels" in selector.get("forbidden_inputs", [])
             and "evaluator expected answers" in selector.get("forbidden_inputs", []),
             "Selector must use runtime inputs only")
    revocation = config["revocation"]
    _require(revocation.get("runtime_metadata_only") is True
             and revocation.get("evaluator_label_kind_forces_abstention") is False,
             "Revocation must not force evaluator-induced abstention")
    _require(tuple(config["layers"]) and len(set(config["layers"]))
             == len(config["layers"]), "Fresh layers must be a unique sequence")
    _require(config["engine"]["reducer"] == "shared_mean",
             "Fresh engine reducer differs")
    _require(config["representation"]["name"] == FACT_REPRESENTATION,
             "Fresh representation differs from the fact binding")
    _require(config["numeric_policy"]["dtype"] == "float32",
             "Fresh protocol requires float32")
    for key in ("heads", "key_size", "value_size"):
        _require(type(config["engine"][key]) is int
                 and config["engine"][key] >= 1,
                 f"Fresh engine {key} is invalid")
    _require(config["budget"]["optimizer_updates"] == 0
             and config["budget"]["gradient_examples"] == 0
             and config["budget"]["native_training_jobs"] == 0,
             "Fresh protocol must not train")
    _require(config["revocation"]["scope"] == "view_only"
             and config["revocation"]["cached_history_retained"] is True,
             "Revocation must be a view-only control over retained history")
    _require(config["checkpoint_policy"]["overwrite"] is False
             and config["checkpoint_policy"]["retention"] == "retain_both",
             "Fresh checkpoints must not be overwritten")
    _require(config["raw_output_policy"]["fresh_output_required"] is True
             and config["raw_output_policy"]["immutable_first_write"] is True,
             "Fresh raw outputs must be immutable first writes")
    _require(config["restart"]["old_observation_encodes_allowed"] == 0,
             "Restart must not re-encode old observations")
    limits = config["limits"]
    _require(limits.get("max_durable_refresh_commits") == 0,
             "Durable refresh commits must be forbidden")
    _require(limits.get("max_duplicate_observe_commits") == 0,
             "Duplicate observation commits must be forbidden")
    _require(type(limits.get("max_query_local_refresh_applied")) is int
             and limits["max_query_local_refresh_applied"] >= 1,
             "Query-local refresh needs a positive applied-attempt bound")
    _require(type(limits.get("max_record_versions_per_record")) is int
             and limits["max_record_versions_per_record"] >= 1,
             "Record-version bound is invalid")
    validate_acceptance_v3(config["acceptance"], require_inventory=False)
    gates = config["acceptance"]["gates"]
    for gate in ("accepted_correct_rate_min", "accepted_precision_min",
                 "correction_accepted_correct_rate_min",
                 "unaffected_accepted_correct_rate_min",
                 "unknown_false_recall_max", "oracle_raw_top1_rate_min"):
        _require(gate in gates, f"Acceptance gate is missing: {gate}")
    return config


def _tokenizer_digest(lock):
    snapshot = lock.get("snapshot_files_sha256")
    _require(isinstance(snapshot, dict) and snapshot,
             "The sealed lock has no snapshot-file pins")
    tokenizer = {name: digest for name, digest in snapshot.items()
                 if "tokenizer" in name.lower()
                 or name == "special_tokens_map.json"}
    _require(tokenizer, "The sealed lock pins no tokenizer files")
    return fingerprint(tokenizer)


def _resolved_pins(config):
    source_pins = _source_pins(
        _resolve_path(config["source"]["model_path"]),
        _resolve_path(config["source"]["model_lock"]))
    _require(source_pins["model_lock_sha256"]
             == config["source"]["model_lock_sha256"],
             "Sealed model lock bytes differ from the configuration pin")
    _require(source_pins["required_revision"]
             == config["source"]["required_revision"],
             "Sealed model revision differs from the configuration pin")
    writer_pins = _extension_pins(
        _resolve_path(config["writer_extension"]["path"]))
    for key in ("weights_sha256", "manifest_sha256", "manifest_digest"):
        _require(writer_pins[key] == config["writer_extension"][key],
                 f"Frozen writer {key} differs from the configuration pin")
    lock = json.loads(
        _resolve_path(config["source"]["model_lock"]).read_text())
    source_pins = dict(source_pins)
    source_pins["tokenizer_sha256"] = _tokenizer_digest(lock)
    return source_pins, writer_pins


def _implementation_sha256(paths):
    result = {}
    for relative in paths:
        path = PROJECT_ROOT / relative
        _require(path.is_file(), f"Implementation pin path is missing: {relative}")
        result[relative] = _sha_file(path)
    return result


def _base_disjoint_episodes(config):
    directory = _resolve_path(config["disjoint_from"]["path"])
    episodes_path = directory / "episodes.json"
    _require(episodes_path.is_file(), "The disjoint base fixture is missing")
    _require(_sha_file(episodes_path)
             == config["disjoint_from"]["episodes_sha256"],
             "Disjoint base episodes differ from the configuration pin")
    return json.loads(episodes_path.read_text())["episodes"]


def prepare_fresh(output, *, config_path=None, config=None):
    """Write the deterministic fresh fixture, labels, controls and pins.

    No model weights, tokenizer, neural forward or optimizer are used.
    """
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Prepare output must be fresh or empty")
    if config is None:
        _require(config_path is not None, "Prepare requires a configuration")
        config_path = Path(config_path)
        config = json.loads(config_path.read_text())
    config = _validate_config(copy.deepcopy(config))
    episodes, labels_by_episode, revocations, _ = generate_fresh_fixture(config)
    base = _base_disjoint_episodes(config)
    audit = validate_fresh_fixture(
        episodes, labels_by_episode, revocations, config=config,
        base_episodes=base)
    source_pins, writer_pins = _resolved_pins(config)
    implementation = _implementation_sha256(config["implementation_paths"])

    output.mkdir(parents=True, exist_ok=True)
    config_snapshot = output / "config.json"
    config_snapshot.write_bytes(_canonical_bytes(config))
    episodes_path = output / "episodes.json"
    labels_path = output / "labels.json"
    revocations_path = output / "revocations.json"
    episodes_path.write_bytes(_canonical_bytes({
        "format": EPISODES_FORMAT, "seed": config["seed"],
        "base_fixture": {
            "path": _display_path(_resolve_path(config["disjoint_from"]["path"])),
            "episodes_sha256": config["disjoint_from"]["episodes_sha256"],
        },
        "episodes": episodes,
    }))
    labels_path.write_bytes(_canonical_bytes({
        "format": LABELS_FORMAT, "evaluation_only": True,
        "note": ("Derived from observed records and declared controls; never "
                 "read by the policy or writer."),
        "episodes": labels_by_episode,
    }))
    revocations_path.write_bytes(_canonical_bytes({
        "format": REVOCATIONS_FORMAT,
        "note": "Evaluation-only live revocation; cached observation history is retained.",
        "revocations": revocations,
    }))
    inventory = build_inventory(episodes, labels_by_episode,
                                layers=config["layers"])
    inventory_path = output / "inventory.json"
    inventory_path.write_bytes(_canonical_bytes(inventory))

    protocol = copy.deepcopy(config)
    protocol["format"] = PROTOCOL_FORMAT
    protocol["config_format"] = CONFIG_FORMAT
    protocol["config_path"] = _display_path(config_snapshot)
    protocol["config_sha256"] = _sha_file(config_snapshot)
    protocol["source_prepared"] = {
        "path": _display_path(output),
        "config_sha256": _sha_file(config_snapshot),
        "episodes_sha256": _sha_file(episodes_path),
        "labels_sha256": _sha_file(labels_path),
        "revocations_sha256": _sha_file(revocations_path),
        "inventory_sha256": _sha_file(inventory_path),
    }
    protocol["source_pins"] = source_pins
    protocol["writer_pins"] = writer_pins
    protocol["implementation_sha256"] = implementation
    protocol["planned_workload"] = planned_workload(
        episodes, protocol["layers"], labels_by_episode)
    protocol["planned_inventory_sha256"] = _sha_json(inventory)
    protocol["fixture_audit"] = audit
    protocol_path = output / "protocol.json"
    protocol_path.write_bytes(_canonical_bytes(protocol))
    manifest = {
        "format": PREPARED_FORMAT,
        "native_computation": False,
        "model_loaded": False,
        "tokenizer_loaded": False,
        "seed": config["seed"],
        "episodes": len(episodes),
        "states": sum(len(states) for states in labels_by_episode.values()),
        "config_sha256": _sha_file(config_snapshot),
        "episodes_sha256": _sha_file(episodes_path),
        "labels_sha256": _sha_file(labels_path),
        "revocations_sha256": _sha_file(revocations_path),
        "inventory_sha256": _sha_file(inventory_path),
        "protocol_sha256": _sha_file(protocol_path),
        "planned_workload": protocol["planned_workload"],
        "fixture_audit": audit,
    }
    _write_json(output / "manifest.json", manifest)
    return manifest


def _read_prepared(prepared):
    prepared = Path(prepared)
    required = ("config.json", "episodes.json", "labels.json",
                "revocations.json", "inventory.json", "protocol.json",
                "manifest.json")
    for name in required:
        _require((prepared / name).is_file(), f"Prepared artifact is missing: {name}")
    return {
        "config": json.loads((prepared / "config.json").read_text()),
        "episodes": json.loads((prepared / "episodes.json").read_text()),
        "labels": json.loads((prepared / "labels.json").read_text()),
        "revocations": json.loads((prepared / "revocations.json").read_text()),
        "inventory": json.loads((prepared / "inventory.json").read_text()),
        "protocol": json.loads((prepared / "protocol.json").read_text()),
        "manifest": json.loads((prepared / "manifest.json").read_text()),
    }


def verify_fresh_prepared(prepared):
    """Re-derive and re-verify every prepared byte and pin."""
    prepared = Path(prepared)
    data = _read_prepared(prepared)
    manifest, protocol = data["manifest"], data["protocol"]
    config = _validate_config(data["config"])
    _require(manifest.get("format") == PREPARED_FORMAT,
             "Prepared manifest format differs")
    _require(manifest.get("native_computation") is False
             and manifest.get("model_loaded") is False
             and manifest.get("tokenizer_loaded") is False,
             "Preparation must not perform native computation")
    pins = protocol["source_prepared"]
    _require(protocol.get("format") == PROTOCOL_FORMAT,
             "Prepared protocol format differs")
    for name, pin in (("config.json", "config_sha256"),
                      ("episodes.json", "episodes_sha256"),
                      ("labels.json", "labels_sha256"),
                      ("revocations.json", "revocations_sha256"),
                      ("inventory.json", "inventory_sha256")):
        _require(_sha_file(prepared / name) == pins[pin],
                 f"Prepared {name} bytes differ from the protocol pin")
    _require(manifest["protocol_sha256"] == _sha_file(prepared / "protocol.json"),
             "Prepared protocol bytes differ from the manifest pin")
    for name in ("episodes_sha256", "labels_sha256", "revocations_sha256",
                 "inventory_sha256", "config_sha256"):
        _require(manifest[name] == pins[name],
                 f"Prepared manifest {name} differs from the protocol")
    _require(data["episodes"].get("format") == EPISODES_FORMAT
             and data["labels"].get("format") == LABELS_FORMAT
             and data["revocations"].get("format") == REVOCATIONS_FORMAT
             and data["inventory"].get("format") == INVENTORY_FORMAT,
             "Prepared formats differ")
    episodes = data["episodes"]["episodes"]
    labels_by_episode = data["labels"]["episodes"]
    revocations = data["revocations"]["revocations"]
    _require(data["episodes"].get("seed") == config["seed"],
             "Prepared fixture seed differs from the configuration")
    _require(data["episodes"].get("base_fixture", {}).get("episodes_sha256")
             == config["disjoint_from"]["episodes_sha256"],
             "Prepared base-fixture pin differs from the configuration")
    regenerated, regenerated_labels, regenerated_revocations, _ = \
        generate_fresh_fixture(config)
    _require(episodes == regenerated,
             "Prepared episodes differ from deterministic regeneration")
    _require(labels_by_episode == regenerated_labels,
             "Prepared labels differ from deterministic regeneration")
    _require(revocations == regenerated_revocations,
             "Prepared revocations differ from deterministic regeneration")
    audit = validate_fresh_fixture(
        episodes, labels_by_episode, revocations, config=config,
        base_episodes=_base_disjoint_episodes(config))
    _require(audit == protocol["fixture_audit"],
             "Prepared fixture audit differs from the protocol")
    _require(audit == manifest["fixture_audit"],
             "Prepared fixture audit differs from the manifest")
    inventory = build_inventory(episodes, labels_by_episode,
                                layers=protocol["layers"])
    _require(inventory == data["inventory"],
             "Prepared inventory differs from the regenerated plan")
    _require(_sha_json(inventory) == protocol["planned_inventory_sha256"],
             "Prepared inventory digest differs from the protocol")
    _require(planned_workload(episodes, protocol["layers"], labels_by_episode)
             == protocol["planned_workload"]
             == manifest["planned_workload"],
             "Prepared planned workload differs")
    source_pins, writer_pins = _resolved_pins(config)
    _require(source_pins == protocol["source_pins"],
             "Prepared source pins differ from the sealed source")
    _require(writer_pins == protocol["writer_pins"],
             "Prepared writer pins differ from the frozen writer")
    implementation = _implementation_sha256(config["implementation_paths"])
    _require(implementation == protocol["implementation_sha256"],
             "Implementation bytes changed after preparation")
    return {
        "status": "verified", "prepared": _display_path(prepared),
        "episodes": audit["episodes"], "states": audit["states"],
        "protocol_sha256": _sha_file(prepared / "protocol.json"),
        "fixture_audit": audit,
        "implementation_sha256": implementation,
    }


# ---------------------------------------------------------------------------
# Exactly-once write ledger and cost accounting
# ---------------------------------------------------------------------------

def record_version(record):
    """Explicit durable record-version identity for one observed record.

    The version covers the observed content and its supersession edge, so a
    replay of the same record bytes is the same version while a genuine update
    with new content or a new supersession target is a new version.
    """
    payload = {
        key: record.get(key) for key in (
            "record_id", "entity", "attribute", "value", "observed_at",
            "supersedes", "origin_world_id")
    }
    return _sha_json(payload)


def durable_identity(record):
    return {"record_id": record["record_id"],
            "record_version": record_version(record)}


class DuplicateWrite(ValueError):
    """An identical durable record version was replayed after commit."""


class WriteLedger:
    """Idempotent durable ingestion keyed by episode, record and version.

    Durable writes accept only ``source="observation"``.  The commit identity is
    ``(episode_id, record_id, record_version)``.  Replaying the identical
    version under a new attempt ID raises :class:`DuplicateWrite`; a new version
    of the same record may commit as a distinct durable update.  A retried
    attempt returns its original outcome without a second write.  Only durable
    state is serialized, so a fresh process rejects the same version replay.

    Query-local accepted refresh is deliberately *not* committed here.  See
    :class:`QueryLocalRefresh`; it is audit metadata and cannot mutate durable
    state.
    """

    def __init__(self):
        self._commits = {}
        self._attempts = {}
        self._rejections = {}
        self._rejected_records = {}
        self._commit_source = {}
        self._attempt_source = {}

    @staticmethod
    def _key(episode_id, record_id):
        return f"{episode_id}:{record_id}"

    @staticmethod
    def _attempt_identity(episode_id, record_id, record_version):
        return {"episode_id": episode_id, "record_id": record_id,
                "record_version": record_version}

    def is_committed(self, episode_id, record_id, record_version=None):
        key = self._key(episode_id, record_id)
        versions = self._commits.get(key)
        if not versions:
            return False
        if record_version is None:
            return True
        return record_version in versions

    def committed_versions(self, episode_id, record_id):
        return sorted(self._commits.get(self._key(episode_id, record_id), {}))

    def commit(self, *, attempt_id, episode_id, record_id, record_version,
               state, source="observation"):
        _require(isinstance(attempt_id, str) and attempt_id,
                 "A ledger attempt requires a stable attempt ID")
        _require(source == "observation",
                 "Durable ingestion accepts only observed records")
        _require(isinstance(record_version, str) and len(record_version) == 64,
                 "A durable record requires a version identity")
        key = self._key(episode_id, record_id)
        identity = self._attempt_identity(episode_id, record_id, record_version)
        previous = self._attempts.get(attempt_id)
        if previous is not None:
            _require(previous["episode_id"] == episode_id
                     and previous["record_id"] == record_id
                     and previous["record_version"] == record_version,
                     "A retried attempt changed its write identity")
            if previous["status"] == "committed":
                return {"status": "retry", "commit_index": previous["index"]}
            return {"status": "duplicate_rejected", "retry": True,
                    "reason": "already_committed"}
        versions = self._commits.setdefault(key, {})
        self._attempt_source[source] = self._attempt_source.get(source, 0) + 1
        if record_version in versions:
            attempt = {"attempt_id": attempt_id, **identity, "state": state,
                       "source": source, "status": "rejected",
                       "index": versions[record_version]["index"]}
            self._attempts[attempt_id] = attempt
            self._rejections[source] = self._rejections.get(source, 0) + 1
            self._rejected_records[key] = self._rejected_records.get(key, 0) + 1
            raise DuplicateWrite(
                f"Record {record_id} in {episode_id} version {record_version} "
                f"was already committed; {source} replay rejected")
        index = sum(len(items) for items in self._commits.values())
        attempt = {"attempt_id": attempt_id, **identity, "state": state,
                   "source": source, "status": "committed", "index": index}
        versions[record_version] = attempt
        self._attempts[attempt_id] = attempt
        self._commit_source[source] = self._commit_source.get(source, 0) + 1
        return {"status": "committed", "commit_index": index,
                "record_version": record_version}

    def require_committed(self, episode_id, record_id, record_version):
        _require(self.is_committed(episode_id, record_id, record_version),
                 f"Observation version is not committed: {record_id}")
        return self._commits[self._key(episode_id, record_id)][record_version]

    def probe_replay(self, episode_id, record_id, record_version):
        return self.is_committed(episode_id, record_id, record_version)

    def counts(self):
        return {key: len(versions)
                for key, versions in sorted(self._commits.items())
                if versions}

    def durable_snapshot(self):
        """Canonical durable identity used to prove refresh has no effect."""
        return _sha_json({
            key: {version: {"state": item["state"],
                            "index": item["index"]}
                  for version, item in sorted(versions.items())}
            for key, versions in sorted(self._commits.items())})

    def summary(self):
        counts = self.counts()
        return {
            "committed_writes": sum(counts.values()),
            "unique_records": len(counts),
            "max_commits_per_record": max(counts.values()) if counts else 0,
            "per_record_commit_counts": counts,
            "per_record_versions": {
                key: sorted(versions)
                for key, versions in sorted(self._commits.items()) if versions},
            "commit_source_counts": dict(sorted(self._commit_source.items())),
            "attempt_source_counts": dict(sorted(self._attempt_source.items())),
            "rejection_counts": dict(sorted(self._rejections.items())),
            "rejected_record_count": len(self._rejected_records),
            "durable_snapshot_sha256": self.durable_snapshot(),
        }

    def retry_probe(self):
        """Retry a recorded attempt and prove the durable set is unchanged."""
        if not self._attempts:
            return {"retried": False, "idempotent": True}
        attempt_id = sorted(self._attempts)[0]
        attempt = self._attempts[attempt_id]
        before = (self.durable_snapshot(),
                  dict(sorted(self._rejections.items())),
                  dict(sorted(self._commit_source.items())))
        outcome = self.commit(
            attempt_id=attempt_id, episode_id=attempt["episode_id"],
            record_id=attempt["record_id"],
            record_version=attempt["record_version"], state=attempt["state"],
            source=attempt["source"])
        after = (self.durable_snapshot(),
                 dict(sorted(self._rejections.items())),
                 dict(sorted(self._commit_source.items())))
        return {"retried": True,
                "idempotent": before == after,
                "outcome": outcome["status"]}

    def to_dict(self):
        return {
            "format": LEDGER_FORMAT,
            "identity": "episode_id/record_id/record_version",
            "commits": [
                {"episode_id": item["episode_id"],
                 "record_id": item["record_id"],
                 "record_version": item["record_version"],
                 "state": item["state"], "source": item["source"],
                 "index": item["index"]}
                for key, versions in sorted(self._commits.items())
                for version, item in sorted(versions.items())],
            "attempts": [
                {"attempt_id": item["attempt_id"],
                 "episode_id": item["episode_id"],
                 "record_id": item["record_id"],
                 "record_version": item["record_version"],
                 "state": item["state"], "source": item["source"],
                 "status": item["status"], "index": item["index"]}
                for key, item in sorted(self._attempts.items())],
            "rejections": dict(sorted(self._rejections.items())),
            "rejected_records": dict(sorted(self._rejected_records.items())),
            "commit_counts": self.counts(),
            "commit_source_counts": dict(sorted(self._commit_source.items())),
            "attempt_source_counts": dict(sorted(self._attempt_source.items())),
            "durable_snapshot_sha256": self.durable_snapshot(),
        }

    @classmethod
    def from_dict(cls, payload):
        _require(isinstance(payload, dict)
                 and payload.get("format") == LEDGER_FORMAT,
                 "Invalid durable write ledger")
        _require(payload.get("identity")
                 == "episode_id/record_id/record_version",
                 "Durable ledger identity differs")
        ledger = cls()
        seen = set()
        indexes = set()
        for item in payload.get("commits", []):
            key = cls._key(item["episode_id"], item["record_id"])
            version = item["record_version"]
            _require(isinstance(version, str) and len(version) == 64,
                     "Durable commit lacks a record version")
            _require((key, version) not in seen,
                     "Durable ledger contains duplicate record versions")
            _require(item.get("source") == "observation",
                     "Durable ledger contains a non-observation source")
            _require(type(item["index"]) is int and item["index"] not in indexes,
                     "Durable ledger commit indexes are not unique")
            seen.add((key, version))
            indexes.add(item["index"])
            ledger._commits.setdefault(key, {})[version] = {
                "episode_id": item["episode_id"],
                "record_id": item["record_id"],
                "record_version": version, "state": item["state"],
                "source": item["source"], "status": "committed",
                "index": item["index"], "attempt_id": None}
        _require(sorted(indexes) == list(range(len(indexes))),
                 "Durable ledger commit indexes are not contiguous")
        for item in payload.get("attempts", []):
            _require(item["attempt_id"] not in ledger._attempts,
                     "Durable ledger attempt IDs are not unique")
            ledger._attempts[item["attempt_id"]] = {
                "attempt_id": item["attempt_id"],
                "episode_id": item["episode_id"],
                "record_id": item["record_id"],
                "record_version": item["record_version"],
                "state": item["state"], "source": item["source"],
                "status": item["status"], "index": item["index"]}
        committed_attempts = {item["attempt_id"] for item in ledger._attempts.values()
                              if item["status"] == "committed"}
        _require(len(committed_attempts) == len(seen),
                 "Durable ledger committed attempts are incomplete")
        if "commit_counts" in payload:
            _require(payload["commit_counts"] == ledger.counts(),
                     "Durable ledger per-record commit counts differ")
        ledger._rejections = dict(payload.get("rejections", {}))
        ledger._rejected_records = dict(payload.get("rejected_records", {}))
        ledger._commit_source = dict(payload.get("commit_source_counts", {}))
        ledger._attempt_source = dict(payload.get("attempt_source_counts", {}))
        if "durable_snapshot_sha256" in payload:
            _require(payload["durable_snapshot_sha256"]
                     == ledger.durable_snapshot(),
                     "Durable ledger snapshot digest differs")
        return ledger


class QueryLocalRefresh:
    """Audit-only accepted refresh over a query-local history.

    An attempt is admitted only when it is enabled, names a live record whose
    exact durable version is committed, and stays within the per-query policy.
    Admission never touches the durable ledger.  The returned attempt record is
    audit metadata; callers rebuild a query-local history from cached writes.
    Repeated calls with a stable attempt ID return the original outcome; the
    same attempt ID with a different identity fails closed.
    """

    def __init__(self, policy=None):
        self.policy = dict(policy or {})
        self._attempts = {}
        self._per_query = {}
        self._status_counts = {}
        self._durable_snapshot = None

    @property
    def enabled(self):
        return self.policy.get("enabled") is True

    def bind_durable(self, ledger):
        self._durable_snapshot = ledger.durable_snapshot()

    def assert_durable_unchanged(self, ledger):
        _require(self._durable_snapshot is not None,
                 "Query-local refresh is not bound to durable state")
        _require(ledger.durable_snapshot() == self._durable_snapshot,
                 "Query-local refresh mutated durable state")

    def attempt(self, *, attempt_id, ledger, episode_id, record_id,
                record_version, live_ids, query_id, layer, state,
                variant=None):
        _require(isinstance(attempt_id, str) and attempt_id,
                 "A refresh attempt requires a stable attempt ID")
        identity = {"episode_id": episode_id, "record_id": record_id,
                    "record_version": record_version, "query_id": query_id,
                    "layer": layer, "variant": variant}
        scope_key = (episode_id, state, query_id, variant)
        previous = self._attempts.get(attempt_id)
        if previous is not None:
            _require(previous["identity"] == identity,
                     "A retried refresh attempt changed its identity")
            return {"status": previous["status"],
                    "retry": True, "reason": previous.get("reason"),
                    "query_local": True, "durable_mutation": False}
        max_per_query = self.policy.get("per_query", {}).get(
            "max_applied_per_query")
        _require(type(max_per_query) is int and max_per_query >= 0,
                 "Query-local refresh requires a per-query bound")
        reason = None
        if not self.enabled:
            reason = "refresh_disabled"
        elif self.policy.get("persistence") != "query_local_only":
            reason = "refresh_not_query_local"
        elif record_id not in live_ids:
            reason = "record_not_live"
        elif (self.policy.get("requires_durable_version") is not True
              or not ledger.is_committed(episode_id, record_id,
                                         record_version)):
            reason = "durable_version_mismatch"
        elif self._per_query.get(scope_key, 0) >= max_per_query:
            reason = "per_query_limit"
        if reason is not None:
            self._attempts[attempt_id] = {
                "identity": identity, "status": "rejected",
                "reason": reason, "query_local": True}
            self._status_counts["rejected"] = \
                self._status_counts.get("rejected", 0) + 1
            return {"status": "rejected", "retry": False, "reason": reason,
                    "query_local": True}
        self._per_query[scope_key] = self._per_query.get(scope_key, 0) + 1
        self._attempts[attempt_id] = {
            "identity": identity, "status": "applied", "reason": None,
            "query_local": True}
        self._status_counts["applied"] = \
            self._status_counts.get("applied", 0) + 1
        return {"status": "applied", "retry": False, "reason": None,
                "query_local": True, "durable_mutation": False}

    def retry_probe(self, ledger):
        """Retry an applied attempt and prove audit and durable state match."""
        for attempt_id, item in sorted(self._attempts.items()):
            if item["status"] != "applied":
                continue
            identity = item["identity"]
            before = (dict(self._status_counts), ledger.durable_snapshot())
            ledger.durable_snapshot()
            retry = self.attempt(
                attempt_id=attempt_id, ledger=ledger,
                episode_id=identity["episode_id"],
                record_id=identity["record_id"],
                record_version=identity["record_version"],
                live_ids={identity["record_id"]},
                query_id=identity["query_id"], layer=identity["layer"],
                state="retry", variant=identity.get("variant"))
            after = (dict(self._status_counts), ledger.durable_snapshot())
            return {"retried": True, "idempotent": before == after,
                    "outcome": retry["status"],
                    "durable_unchanged": before[1] == after[1]}
        return {"retried": False, "idempotent": True,
                "durable_unchanged": True}

    def summary(self):
        return {
            "format": REFRESH_AUDIT_FORMAT,
            "owner": "query_local",
            "persistent_memory_writes": 0,
            "optimizer_updates": 0,
            "decay_updates": 0,
            "enabled": self.enabled,
            "policy": copy.deepcopy(self.policy),
            "attempts": len(self._attempts),
            "status_counts": dict(sorted(self._status_counts.items())),
            "applied_per_query": {
                "|".join("" if part is None else str(part) for part in key):
                    value
                for key, value in sorted(self._per_query.items())},
            "durable_snapshot_sha256": self._durable_snapshot,
            "attempt_records": [
                {"attempt_id": attempt_id,
                 **item["identity"], "status": item["status"],
                 "reason": item["reason"]}
                for attempt_id, item in sorted(self._attempts.items())],
        }


class CostLedger:
    """Monotonic per-phase cached-history cost accounting."""

    def __init__(self):
        self._started = time.monotonic()
        self._phase_started = self._started
        self.counts = {}
        self.seconds = {}
        self.phases = []
        self.phase = "start"

    def add(self, name, count=1, seconds=0.0):
        if count:
            self.counts[name] = self.counts.get(name, 0) + count
        if seconds:
            self.seconds[name] = self.seconds.get(name, 0.0) + float(seconds)

    @contextmanager
    def timed(self, name, count=1):
        started = time.monotonic()
        try:
            yield
        finally:
            self.add(name, count=count, seconds=time.monotonic() - started)

    @contextmanager
    def observe_phase(self, name):
        self.phase = name
        started = time.monotonic()
        try:
            yield
        finally:
            self.phases.append({
                "phase": name,
                "seconds": time.monotonic() - started,
                "counts": dict(sorted(self.counts.items())),
                "seconds_by_operation": {
                    key: round(value, 6)
                    for key, value in sorted(self.seconds.items())},
            })

    def wall_seconds(self):
        return time.monotonic() - self._started

    def to_dict(self):
        return {
            "wall_seconds": self.wall_seconds(),
            "counts": dict(sorted(self.counts.items())),
            "seconds_by_operation": {
                key: round(value, 6) for key, value in sorted(self.seconds.items())},
            "phases": list(self.phases),
        }


def _control_record(record_id, *, value, entity="control-entity"):
    return {
        "record_id": record_id, "entity": entity,
        "attribute": "control-attribute", "value": value,
        "observed_at": 1, "supersedes": None,
        "origin_world_id": "control-world",
    }


def run_exactly_once_control(*, episode_id="control-episode",
                             record_ids=("rec-control-a", "rec-control-b"),
                             state="initial"):
    """Exercise durable replay, versioning, query-local refresh and restart."""
    ledger = WriteLedger()
    records = {record_id: _control_record(record_id, value=f"value-{index}")
               for index, record_id in enumerate(record_ids)}
    for record_id, record in records.items():
        ledger.commit(attempt_id=f"{episode_id}:observe:{record_id}",
                      episode_id=episode_id, record_id=record_id,
                      record_version=record_version(record),
                      state=state, source="observation")
    counts_before = ledger.counts()
    retry = ledger.retry_probe()
    duplicate_rejected = restart_replay_rejected = False
    first = records[record_ids[0]]
    try:
        ledger.commit(attempt_id=f"{episode_id}:replay:{record_ids[0]}",
                      episode_id=episode_id, record_id=record_ids[0],
                      record_version=record_version(first),
                      state=state, source="observation")
    except DuplicateWrite:
        duplicate_rejected = True
    updated = dict(first, value="updated-value")
    version_update = ledger.commit(
        attempt_id=f"{episode_id}:update:{record_ids[0]}",
        episode_id=episode_id, record_id=record_ids[0],
        record_version=record_version(updated), state="corrected",
        source="observation")
    versions_after_update = len(ledger.committed_versions(
        episode_id, record_ids[0]))
    policy = {
        "enabled": True,
        "owner": "query_local_refresh",
        "persistence": "query_local_only",
        "requires_durable_version": True,
        "per_query": {"max_applied_per_query": 1},
        "retry": {"attempt_id": "stable", "idempotent": True},
    }
    refresh = QueryLocalRefresh(policy)
    refresh.bind_durable(ledger)
    durable_before = ledger.durable_snapshot()
    second = records[record_ids[1]]
    applied = refresh.attempt(
        attempt_id="refresh-applied", ledger=ledger, episode_id=episode_id,
        record_id=record_ids[1], record_version=record_version(second),
        live_ids=set(record_ids), query_id="query-live", layer=0,
        state=state)
    retried = refresh.attempt(
        attempt_id="refresh-applied", ledger=ledger, episode_id=episode_id,
        record_id=record_ids[1], record_version=record_version(second),
        live_ids=set(record_ids), query_id="query-live", layer=0,
        state="retry")
    non_live = refresh.attempt(
        attempt_id="refresh-non-live", ledger=ledger, episode_id=episode_id,
        record_id=record_ids[1], record_version=record_version(second),
        live_ids=set(), query_id="query-non-live", layer=0, state=state)
    mismatched = refresh.attempt(
        attempt_id="refresh-mismatch", ledger=ledger, episode_id=episode_id,
        record_id=record_ids[1], record_version="f" * 64,
        live_ids=set(record_ids), query_id="query-mismatch", layer=0,
        state=state)
    limit = refresh.attempt(
        attempt_id="refresh-limit", ledger=ledger, episode_id=episode_id,
        record_id=record_ids[1], record_version=record_version(second),
        live_ids=set(record_ids), query_id="query-live", layer=1,
        state=state)
    refresh.bind_durable(ledger)
    try:
        refresh.assert_durable_unchanged(ledger)
        durable_unchanged = ledger.durable_snapshot() == durable_before
    except ValueError:
        durable_unchanged = False
    refresh_probe = refresh.retry_probe(ledger)
    restored = WriteLedger.from_dict(
        json.loads(json.dumps(ledger.to_dict())))
    try:
        restored.commit(attempt_id=f"{episode_id}:restart:{record_ids[0]}",
                        episode_id=episode_id, record_id=record_ids[0],
                        record_version=record_version(first),
                        state=state, source="observation")
    except DuplicateWrite:
        restart_replay_rejected = True
    durable_refresh_source = ledger.summary()["commit_source_counts"].get(
        "accepted_refresh", 0)
    outcome = {
        "counts_before": len(counts_before),
        "counts_after": len(ledger.counts()),
        "durable_retry": retry,
        "refresh_retry": refresh_probe,
        "duplicate_observe_rejected": duplicate_rejected,
        "restart_replay_rejected": restart_replay_rejected,
        "version_update_committed": version_update["status"] == "committed",
        "versions_after_update": versions_after_update,
        "max_commits_per_record": max(ledger.counts().values()),
        "refresh_applied": applied["status"] == "applied",
        "refresh_retry_status": retried["status"],
        "refresh_retry_idempotent": retried["status"] == "applied",
        "refresh_non_live_rejected": non_live["status"] == "rejected",
        "refresh_non_live_reason": non_live["reason"],
        "refresh_mismatch_rejected": mismatched["status"] == "rejected",
        "refresh_mismatch_reason": mismatched["reason"],
        "refresh_per_query_limit_rejected": limit["status"] == "rejected",
        "refresh_durable_unchanged": durable_unchanged,
        "refresh_audit": refresh.summary(),
        "refresh_commits": durable_refresh_source,
        "durable_mutations_from_refresh": 0 if durable_unchanged else 1,
    }
    outcome["passed"] = (
        retry["idempotent"]
        and refresh_probe["idempotent"]
        and refresh_probe["durable_unchanged"]
        and duplicate_rejected
        and restart_replay_rejected
        and outcome["version_update_committed"]
        and outcome["refresh_applied"]
        and outcome["refresh_retry_status"] == "applied"
        and outcome["refresh_non_live_rejected"]
        and outcome["refresh_mismatch_rejected"]
        and outcome["refresh_per_query_limit_rejected"]
        and durable_unchanged
        and durable_refresh_source == 0
    )
    return outcome


# ---------------------------------------------------------------------------
# Cached episode builder and checkpoints
# ---------------------------------------------------------------------------

@dataclass
class EpisodeCache:
    episode: dict
    layers: tuple
    states: dict = field(default_factory=dict)
    writes: dict = field(default_factory=dict)
    entry_vectors: dict = field(default_factory=dict)
    query_vectors: dict = field(default_factory=dict)
    query_phases: dict = field(default_factory=dict)
    text_vectors: dict = field(default_factory=dict)
    weights: dict = field(default_factory=dict)
    state_shas: dict = field(default_factory=dict)


def _entry_row(write):
    key = write.keys[0, 0].detach().float().cpu()
    value = write.values[0, 0].detach().float().cpu()
    _require(bool(torch.isfinite(key).all())
             and bool(torch.isfinite(value).all()),
             "Writer produced a non-finite key/value vector")
    return {
        "write_key": key.tolist(), "value_vector": value.tolist(),
        "write_key_sha256": _sha_json(key.tolist()),
        "value_sha256": _sha_json(value.tolist()),
        "key_norm": float(torch.linalg.vector_norm(key)),
        "value_norm": float(torch.linalg.vector_norm(value)),
    }


def _encode_query_text(cache, model, tokenizer, query, costs):
    text_sha = _sha_json(query["text"])
    if text_sha not in cache.text_vectors:
        vectors = {}
        for layer in cache.layers:
            with costs.timed("query_encoding"):
                vectors[layer] = query_key(
                    model, tokenizer, query["text"], layer=layer)
        cache.text_vectors[text_sha] = vectors
    else:
        costs.add("query_vectors_reused", len(cache.layers))
    for layer in cache.layers:
        cache.query_vectors[(query["query_id"], layer)] = \
            cache.text_vectors[text_sha][layer]
    cache.query_phases[query["query_id"]] = (query["phase"], text_sha)


def build_episode_cache(model, tokenizer, episode, layers, costs):
    cache = EpisodeCache(episode=episode, layers=tuple(layers))
    for state in episode_states(episode):
        records, live, negative = _record_views(episode, state)
        cache.states[state["name"]] = {
            "state": state, "records": list(records),
            "live_ids": [record["record_id"] for record in live],
            "negative_ids": [record["record_id"] for record in negative],
            "observation_sha256": _observation_sha(episode, state),
        }
        for record in records:
            record_id = record["record_id"]
            for layer in layers:
                if (record_id, layer) in cache.writes:
                    continue
                with costs.timed("observation_encoding"):
                    write = observed_binding(
                        model, tokenizer, _fact_of(record), layer=layer)
                cache.writes[(record_id, layer)] = write
                cache.entry_vectors[(record_id, layer)] = _entry_row(write)
    for query in episode["queries"]:
        _encode_query_text(cache, model, tokenizer, query, costs)
    return cache


def add_phase_queries(cache, model, tokenizer, phase, costs):
    _require(phase in (RESTART, REVOCATION), "Only post-restart phases are added")
    queries = cache.episode[PHASE_QUERY_FIELDS[phase]]
    new_texts = 0
    for query in queries:
        if query["query_id"] in cache.query_phases:
            _require(cache.query_phases[query["query_id"]][0] == phase,
                     "A query ID was reused across phases")
            continue
        before = len(cache.text_vectors)
        _encode_query_text(cache, model, tokenizer, query, costs)
        if len(cache.text_vectors) > before:
            new_texts += 1
    return {"queries": len(queries), "new_query_texts": new_texts}


def _context(cache, state_name):
    state = cache.states[state_name]
    return {
        "episode": cache.episode, "state": state["state"],
        "records": state["records"], "live_ids": state["live_ids"],
        "negative_ids": state["negative_ids"], "writes": cache.writes,
        "entry_vectors": cache.entry_vectors,
        "query_vectors": cache.query_vectors,
        "observation_sha256": state["observation_sha256"],
    }


def _variant_kind(variant):
    return {FULL: "full", CAUSAL: "full", SHUFFLED: "shuffled",
            DISABLED: "zero", ORACLE: "full"}[variant]


def _build_weights(cache, records, layers, primitive, device, costs, *,
                   shuffle_values):
    with costs.timed("state_rebuild"):
        history = _history_writes(
            {"records": records, "writes": cache.writes}, records, layers,
            shuffle_values=shuffle_values)
        states = _build_state(
            primitive, layers, history, reducer="shared_mean",
            gate_control=None, device=device)
    return states["shared"].weights


def _weights(cache, state_name, kind, primitive, device, costs):
    key = (state_name, kind)
    if key in cache.weights:
        return cache.weights[key]
    if kind == "zero":
        weights = primitive.initial_state(1, device=device).weights
        costs.add("zero_states")
    else:
        weights = _build_weights(
            cache, cache.states[state_name]["records"], cache.layers,
            primitive, device, costs, shuffle_values=(kind == "shuffled"))
    cache.weights[key] = weights
    return weights


def _weights_sha(cache, state_name, kind, primitive, device, costs):
    key = (state_name, kind)
    if key not in cache.state_shas:
        cache.state_shas[key] = _state_sha(
            _weights(cache, state_name, kind, primitive, device, costs))
    return cache.state_shas[key]


def _pack_write(write):
    return {"keys": write.keys.detach().cpu(),
            "values": write.values.detach().cpu(),
            "erase": write.erase.detach().cpu(),
            "write_gate": write.write_gate.detach().cpu(),
            "decay": write.decay.detach().cpu(),
            "chunk_index": write.chunk_index,
            "mask": None if write.mask is None else write.mask.detach().cpu()}


def _unpack_write(packed, device):
    return SGDN2Write(
        packed["keys"].to(device), packed["values"].to(device),
        packed["erase"].to(device), packed["write_gate"].to(device),
        packed["decay"].to(device), packed["chunk_index"],
        None if packed["mask"] is None else packed["mask"].to(device))


def _pack_cache(cache):
    return {
        "episode_id": cache.episode["episode_id"],
        "world_id": cache.episode["world_id"],
        "split": cache.episode["split"],
        "complexity": cache.episode["complexity"],
        "variant": cache.episode["variant"],
        "layers": list(cache.layers),
        "states": {
            name: {
                "cutoff": item["state"]["cutoff"],
                "record_ids": [record["record_id"] for record in item["records"]],
                "live_ids": list(item["live_ids"]),
                "negative_ids": list(item["negative_ids"]),
                "observation_sha256": item["observation_sha256"],
            }
            for name, item in sorted(cache.states.items())},
        "writes": [
            {"record_id": record_id, "layer": layer, "write": _pack_write(write)}
            for (record_id, layer), write in sorted(cache.writes.items())],
        "entry_vectors": [
            {"record_id": record_id, "layer": layer,
             "entry": cache.entry_vectors[(record_id, layer)]}
            for (record_id, layer) in sorted(cache.entry_vectors)],
        "text_vectors": [
            {"text_sha256": text_sha, "layer": layer,
             "vector": vector.detach().cpu()}
            for text_sha, vectors in sorted(cache.text_vectors.items())
            for layer, vector in sorted(vectors.items())],
        "queries": [
            {"query_id": query_id, "phase": phase, "text_sha256": text_sha}
            for query_id, (phase, text_sha)
            in sorted(cache.query_phases.items())],
    }


def _unpack_cache(packed, episode, device):
    _require(packed["episode_id"] == episode["episode_id"],
             "Cached episode identity differs")
    for key in ("world_id", "split", "complexity", "variant"):
        _require(packed[key] == episode[key],
                 f"Cached episode {key} differs")
    layers = tuple(packed["layers"])
    _require(len(layers) == len(packed["layers"]),
             "Cached layers are missing")
    cache = EpisodeCache(episode=episode, layers=layers)
    for state in episode_states(episode):
        records, live, negative = _record_views(episode, state)
        stored = packed["states"].get(state["name"])
        _require(stored is not None, "Cached state is missing")
        _require(stored["cutoff"] == state["cutoff"],
                 "Cached state cutoff differs")
        _require(stored["record_ids"]
                 == [record["record_id"] for record in records],
                 "Cached state records differ from the fixture")
        _require(stored["live_ids"]
                 == [record["record_id"] for record in live],
                 "Cached live records differ from the fixture")
        _require(stored["negative_ids"]
                 == [record["record_id"] for record in negative],
                 "Cached negative records differ from the fixture")
        _require(stored["observation_sha256"]
                 == _observation_sha(episode, state),
                 "Cached observations differ from the fixture")
        cache.states[state["name"]] = {
            "state": state, "records": list(records),
            "live_ids": list(stored["live_ids"]),
            "negative_ids": list(stored["negative_ids"]),
            "observation_sha256": stored["observation_sha256"],
        }
    for item in packed["writes"]:
        cache.writes[(item["record_id"], item["layer"])] = _unpack_write(
            item["write"], device)
    for item in packed["entry_vectors"]:
        cache.entry_vectors[(item["record_id"], item["layer"])] = item["entry"]
    for item in packed["text_vectors"]:
        cache.text_vectors.setdefault(item["text_sha256"], {})[
            item["layer"]] = item["vector"].to(device)
    for item in packed["queries"]:
        _require(item["text_sha256"] in cache.text_vectors,
                 "Cached query text vector is missing")
        for layer in layers:
            cache.query_vectors[(item["query_id"], layer)] = \
                cache.text_vectors[item["text_sha256"]][layer]
        cache.query_phases[item["query_id"]] = (item["phase"],
                                                item["text_sha256"])
    return cache


def _check_binding(expected, actual, *, label):
    _require(isinstance(expected, dict) and isinstance(actual, dict),
             f"Checkpoint {label} binding is malformed")
    _require(set(expected) == set(actual),
             f"Checkpoint {label} binding keys differ")
    for key in expected:
        _require(expected[key] == actual[key],
                 f"Checkpoint {label} differs at {key}")
    return True


def expected_bindings(*, protocol, fixture_hashes, source_pins,
                      writer_receipt, implementation_sha256, representation,
                      engine):
    protocol_sha = protocol.get("protocol_sha256")
    _require(isinstance(protocol_sha, str) and len(protocol_sha) == 64,
             "A protocol sha256 is required for checkpoint binding")
    _require(isinstance(implementation_sha256, dict) and implementation_sha256,
             "Implementation hashes are required for checkpoint binding")
    _require(isinstance(source_pins, dict) and source_pins,
             "Source pins are required for checkpoint binding")
    writer = {"weights_sha256": writer_receipt.get("weights_sha256")}
    declared = protocol.get("writer_extension") or {}
    if declared.get("manifest_sha256"):
        writer["manifest_sha256"] = declared["manifest_sha256"]
    writer["manifest_digest"] = writer_receipt.get("manifest_digest")
    _require(all(isinstance(value, str) and value
                 for value in writer.values()),
             "Writer receipt lacks its bound identity")
    return {
        "protocol_sha256": protocol_sha,
        "fixture": dict(fixture_hashes),
        "source": dict(source_pins),
        "writer": writer,
        "representation": {
            "name": representation["name"],
            "sha256": representation["sha256"],
            "layers": list(representation["layers"]),
        },
        "implementation": dict(implementation_sha256),
        "engine": dict(engine),
    }


def save_fresh_checkpoint(path, *, bindings, caches, ledger, phase, counters,
                          payload_extra=None):
    destination = Path(path)
    if destination.exists():
        raise FileExistsError("Use a new fresh causal checkpoint")
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": CHECKPOINT_FORMAT,
        "phase": phase,
        "bindings": bindings,
        "episodes": [_pack_cache(cache) for cache in caches],
        "ledger": ledger.to_dict(),
        "counters": dict(counters),
    }
    if payload_extra:
        payload.update(payload_extra)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return destination


def load_fresh_checkpoint(path, *, expected, device="cpu"):
    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    _require(isinstance(payload, dict)
             and payload.get("format") == CHECKPOINT_FORMAT,
             "Invalid fresh causal checkpoint")
    actual = payload.get("bindings")
    _require(isinstance(actual, dict), "Fresh checkpoint has no bindings")
    for label in ("protocol_sha256", "fixture", "source", "writer",
                  "representation", "implementation", "engine"):
        _require(label in actual and label in expected,
                 f"Fresh checkpoint binding is missing: {label}")
        if label == "protocol_sha256":
            _require(actual[label] == expected[label],
                     "Fresh checkpoint protocol differs")
        else:
            _check_binding(expected[label], actual[label], label=label)
    return payload


# ---------------------------------------------------------------------------
# Selection, readout and ranking
# ---------------------------------------------------------------------------

def _select(cache, state_name, query, layer, *, thresholds, revoked_ids):
    state = cache.states[state_name]
    live = list(state["live_ids"])
    candidates = [record_id for record_id in live if record_id not in revoked_ids]
    excluded = [record_id for record_id in live if record_id in revoked_ids]
    if not candidates:
        return {"selected_record_id": None, "accepted": False, "scores": {},
                "top_score": None, "second_score": None,
                "reason": "no_live_observation", "candidate_ids": [],
                "excluded_revoked_ids": excluded}
    query_vector = cache.query_vectors[(query["query_id"], layer)]
    scores = {
        record_id: _cosine_heads(
            query_vector[0, 0].detach().float().cpu(),
            cache.writes[(record_id, layer)].keys[0, 0].detach().float().cpu())
        for record_id in candidates}
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    top_id, top = ordered[0]
    second = ordered[1][1] if len(ordered) > 1 else None
    accepted = bool(top >= thresholds["address_cosine_floor"]
                    and top - (second or 0.0) >= thresholds["address_margin_floor"])
    return {"selected_record_id": top_id if accepted else None,
            "accepted": accepted, "scores": scores, "top_score": top,
            "second_score": second,
            "reason": ("query_address_accepted" if accepted
                       else "query_address_below_threshold"),
            "candidate_ids": candidates, "excluded_revoked_ids": excluded}


def _decide(variant, cache, state_name, query, layer, labels, *,
            thresholds, revoked_ids):
    target = labels["targets"][query["query_id"]]
    if variant == ORACLE:
        selected = (target["target_entry_id"]
                    if target["kind"] == "fact" else None)
        if selected is not None and selected in revoked_ids:
            selected = None
        return {"selected_record_id": selected,
                "accepted": selected is not None, "scores": {},
                "top_score": None, "second_score": None,
                "reason": ("gold_target_oracle" if selected is not None
                           else "oracle_no_fact_target"),
                "candidate_ids": [], "excluded_revoked_ids": sorted(revoked_ids)}
    if variant in (CAUSAL, SHUFFLED):
        return _select(cache, state_name, query, layer,
                       thresholds=thresholds, revoked_ids=revoked_ids)
    return {"selected_record_id": None, "accepted": False, "scores": {},
            "top_score": None, "second_score": None,
            "reason": "no_refresh", "candidate_ids": [],
            "excluded_revoked_ids": sorted(revoked_ids)}


def _refresh_attempt(refresh, ledger, cache, variant, state_name, query,
                     layer, decision):
    selected = decision["selected_record_id"]
    attempt_id = (f"{variant}|{cache.episode['episode_id']}|{state_name}|"
                  f"{query['query_id']}|{layer}")
    if selected is None:
        return {"status": "not_attempted", "attempt_id": None,
                "selected_record_id": None, "record_version": None,
                "reason": decision["reason"], "query_local": True,
                "durable_mutation": False}
    record = next(item for item in cache.states[state_name]["records"]
                  if item["record_id"] == selected)
    outcome = refresh.attempt(
        attempt_id=attempt_id, ledger=ledger,
        episode_id=cache.episode["episode_id"], record_id=selected,
        record_version=record_version(record),
        live_ids=cache.states[state_name]["live_ids"],
        query_id=query["query_id"], layer=layer, state=state_name,
        variant=variant)
    return {"status": outcome["status"], "attempt_id": attempt_id,
            "selected_record_id": selected,
            "record_version": record_version(record),
            "reason": outcome.get("reason"), "query_local": True,
            "durable_mutation": False}


def _decision_row(variant, cache, state_name, phase, query, layer, labels,
                  decision, attempt, *, revoked_ids, weights_source,
                  base_state_sha, state_sha):
    state = cache.states[state_name]
    applied = attempt["status"] == "applied"
    return {
        "format": DECISION_FORMAT, "variant": variant,
        "policy_class": ("gold_target_oracle" if variant == ORACLE
                         else "causal_current_query" if variant in (CAUSAL, SHUFFLED)
                         else "control"),
        "phase": phase, "episode_id": cache.episode["episode_id"],
        "world_id": cache.episode["world_id"], "split": cache.episode["split"],
        "state": state_name, "case_id": query["query_id"], "layer": layer,
        "cohort": labels["cohort_by_query"][query["query_id"]],
        "target_role": labels["targets"][query["query_id"]]["target_role"],
        "selected_record_id": decision["selected_record_id"],
        "accepted": decision["accepted"], "reason": decision["reason"],
        "top_score": decision["top_score"],
        "second_score": decision["second_score"],
        "scores": decision["scores"],
        "candidate_ids": decision["candidate_ids"],
        "excluded_revoked_ids": decision["excluded_revoked_ids"],
        "committed_record_ids": [record["record_id"]
                                 for record in state["records"]],
        "committed_record_versions": {
            record["record_id"]: record_version(record)
            for record in state["records"]},
        "refresh_attempt": attempt,
        "refresh_applied": applied,
        "refresh_persistent": False,
        "weights_source": weights_source,
        "base_state_sha256": base_state_sha,
        "state_sha256": state_sha,
        "refresh_changed_state": (applied and state_sha != base_state_sha),
        "causal_fallback_to_full": (
            variant in (CAUSAL, SHUFFLED) and not applied
            and weights_source == "full_state"),
        "oracle_no_refresh_fallback": (
            variant == ORACLE and not applied
            and weights_source == "full_state"),
        "used_gold_target": variant == ORACLE,
        "used_future_queries": False, "reread_source": False,
        "reencoded_old_observations": False,
    }


def _evaluate_query(cache, state_name, labels, query, variant, *,
                    primitive, device, thresholds, ledger, refresh_audit,
                    costs, phase, revoked_ids):
    rows, decisions = [], []
    target = labels["targets"][query["query_id"]]
    kind = target["kind"]
    target_id = target["target_entry_id"]
    state = cache.states[state_name]
    record_ids = [record["record_id"] for record in state["records"]]
    final_write = record_ids[-1] if record_ids else None
    zero_weights = _weights(cache, state_name, "zero", primitive, device, costs)
    zero_sha = _weights_sha(cache, state_name, "zero", primitive, device, costs)
    base_kind = _variant_kind(variant)
    base_weights = _weights(cache, state_name, base_kind, primitive, device,
                            costs)
    base_sha = _weights_sha(cache, state_name, base_kind, primitive, device,
                            costs)
    base_source = {"full": "full_state", "shuffled": "shuffled_state",
                   "zero": "zero_state"}[base_kind]
    for layer in cache.layers:
        decision = _decide(
            variant, cache, state_name, query, layer, labels,
            thresholds=thresholds, revoked_ids=revoked_ids)
        attempt = _refresh_attempt(
            refresh_audit, ledger, cache, variant, state_name, query, layer,
            decision)
        applied = attempt["status"] == "applied"
        if applied:
            selected = attempt["selected_record_id"]
            history = state["records"] + [
                record for record in state["records"]
                if record["record_id"] == selected]
            weights = _build_weights(
                cache, history, cache.layers, primitive, device, costs,
                shuffle_values=(variant == SHUFFLED))
            state_sha = _state_sha(weights)
            weights_source = ("query_local_refresh_shuffled"
                              if variant == SHUFFLED
                              else "query_local_refresh")
        else:
            weights = base_weights
            state_sha = base_sha
            weights_source = base_source
        variant_doc = {"name": variant, "reducer": "shared_mean"}
        vector = cache.query_vectors[(query["query_id"], layer)]
        layer_rows = []
        if kind == "fact" and target_id is not None:
            key_vector = cache.writes[(target_id, layer)].keys[0, 0] \
                .detach().float().cpu()
            with costs.timed("readout"):
                readout = _read_weights(
                    weights, key_vector.view(1, 1, *key_vector.shape))
            layer_rows.append(_readout_row(
                variant=variant_doc, arm="exact_key_value", context=_context(
                    cache, state_name), query=query, layer=layer,
                readout=readout, readout_space="value",
                final_write_id=final_write, state_sha256=state_sha))
        with costs.timed("readout"):
            readout = _read_weights(weights, vector)
        layer_rows.append(_readout_row(
            variant=variant_doc, arm="query_key_value",
            context=_context(cache, state_name), query=query, layer=layer,
            readout=readout, readout_space="value",
            final_write_id=final_write, state_sha256=state_sha))
        with costs.timed("readout"):
            readout = vector[0, 0].detach().float().cpu()
        layer_rows.append(_readout_row(
            variant=variant_doc, arm="query_key_address",
            context=_context(cache, state_name), query=query, layer=layer,
            readout=readout, readout_space="key",
            final_write_id=final_write, state_sha256=state_sha))
        with costs.timed("readout"):
            readout = _read_weights(zero_weights, vector)
        layer_rows.append(_readout_row(
            variant=variant_doc, arm="context_removed",
            context=_context(cache, state_name), query=query, layer=layer,
            readout=readout, readout_space="value",
            final_write_id=final_write, state_sha256=zero_sha))
        for row in layer_rows:
            row["refresh_applied"] = applied
            row["refresh_attempt_id"] = attempt["attempt_id"]
            row["weights_source"] = weights_source
            row["runtime_selected_record_id"] = decision["selected_record_id"]
            row["runtime_accepted"] = decision["accepted"]
            row["runtime_reason"] = decision["reason"]
            row["used_gold_target"] = variant == ORACLE
            rows.append(row)
        decisions.append(_decision_row(
            variant, cache, state_name, phase, query, layer, labels,
            decision, attempt, revoked_ids=revoked_ids,
            weights_source=weights_source, base_state_sha=base_sha,
            state_sha=state_sha))
    for row in rows:
        row["phase"] = phase
        row["codebook_key"] = (
            f"{cache.episode['episode_id']}/{state_name}/{phase}")
    return decisions, rows


def _codebook(cache, state_name, phase, revoked_ids):
    state = cache.states[state_name]
    entries = {}
    for layer in cache.layers:
        entries[str(layer)] = {
            record_id: dict(cache.entry_vectors[(record_id, layer)])
            for record_id in [record["record_id"] for record in state["records"]]}
    return {
        "format": "replica.memory_causal_fresh.codebook.v1",
        "episode_id": cache.episode["episode_id"],
        "world_id": cache.episode["world_id"],
        "state": state_name, "cutoff": state["state"]["cutoff"],
        "phase": phase,
        "observation_sha256": state["observation_sha256"],
        "layers": entries,
        "live_entry_ids": sorted(set(state["live_ids"]) - set(revoked_ids)),
        "negative_entry_ids": sorted(state["negative_ids"]),
        "revoked_entry_ids": sorted(set(state["live_ids"]) & set(revoked_ids)),
        "all_entry_ids": sorted(record["record_id"] for record in state["records"]),
    }


def _fresh_ranking(row, codebook, labels, thresholds, revoked_ids):
    """Rank a readout while keeping policy blocks separate from correctness.

    ``policy_blocked`` is derived only from runtime metadata (the declared
    revocation view and the superseded/live record sets).  It is never derived
    from the sealed evaluator label kind.  A revoked-target query that accepts
    another live record therefore remains an accepted wrong retrieval instead
    of being converted into evaluator-induced abstention.
    """
    ranking = _recompute_ranking(row, codebook, labels, thresholds)
    kind = labels["targets"][row["case_id"]]["kind"]
    top = ranking["top_entry_id"]
    blocked = None
    if top in set(revoked_ids):
        blocked = "revoked"
    elif top in set(codebook["negative_entry_ids"]):
        blocked = "stale"
    readout_accepted = bool(ranking["accepted"])
    policy_blocked = blocked is not None
    accepted = readout_accepted and not policy_blocked
    ranking["readout_accepted"] = readout_accepted
    ranking["policy_blocked"] = blocked
    ranking["blocked_acceptance"] = blocked
    ranking["policy_blocked_acceptance"] = readout_accepted and policy_blocked
    ranking["revoked_blocked"] = blocked == "revoked"
    ranking["stale_blocked"] = blocked == "stale"
    ranking["revoked_query_kind"] = kind == "revoked"
    ranking["accepted"] = accepted
    top1 = bool(ranking["top1"])
    ranking["false_recall"] = bool(accepted and not top1)
    if accepted and not top1:
        if kind == "revoked":
            false_recall_kind = "revoked_target_other_live"
        elif kind == "unknown":
            false_recall_kind = "unknown_address"
        elif top in set(codebook["negative_entry_ids"]):
            false_recall_kind = "superseded"
        elif ranking.get("target_role") is None:
            false_recall_kind = "other_source"
        else:
            false_recall_kind = ranking.get("false_recall_kind") or "other_source"
    else:
        false_recall_kind = None
    ranking["false_recall_kind"] = false_recall_kind
    runtime_selected = row.get("runtime_selected_record_id")
    ranking["runtime_selected_record_id"] = runtime_selected
    ranking["runtime_accepted"] = bool(row.get("runtime_accepted"))
    ranking["selection_correct"] = bool(
        runtime_selected is not None and runtime_selected == ranking["target_entry_id"])
    ranking["selected_other_live"] = bool(
        ranking["runtime_accepted"]
        and not ranking["selection_correct"]
        and runtime_selected is not None)
    ranking["refresh_applied"] = bool(row.get("refresh_applied"))
    ranking["refresh_attempt_id"] = row.get("refresh_attempt_id")
    ranking["weights_source"] = row.get("weights_source")
    ranking["runtime_selection_available"] = "runtime_accepted" in row
    return ranking


def _rank(readouts, decisions, codebooks, labels_by_episode, revocations,
          thresholds):
    decision_by_key = {
        (row["variant"], row["episode_id"], row["state"], row["case_id"],
         row["phase"], row["layer"]): row for row in decisions}
    rankings = []
    for row in readouts:
        key = (row["episode_id"], row["state"], row["phase"])
        labels = labels_by_episode[row["episode_id"]][row["state"]]
        runtime_revoked = (
            (revocations[row["episode_id"]]["record_id"],)
            if row["phase"] == REVOCATION else ())
        decision = decision_by_key.get(
            (row["variant"], row["episode_id"], row["state"], row["case_id"],
             row["phase"], row["layer"]))
        _require(decision is not None, "Readout row has no matching decision")
        enriched = dict(row)
        enriched["runtime_selected_record_id"] = decision[
            "selected_record_id"]
        enriched["runtime_accepted"] = decision["accepted"]
        enriched["runtime_reason"] = decision["reason"]
        enriched["refresh_applied"] = decision["refresh_applied"]
        enriched["refresh_attempt_id"] = (
            decision["refresh_attempt"]["attempt_id"])
        enriched["weights_source"] = decision["weights_source"]
        ranking = _fresh_ranking(
            enriched, codebooks[key], labels, thresholds, runtime_revoked)
        ranking["phase"] = row["phase"]
        ranking["cohort"] = labels["cohort_by_query"][row["case_id"]]
        ranking["split"] = labels["split"]
        ranking["runtime_revoked_ids"] = list(runtime_revoked)
        ranking["runtime_negative_ids"] = list(
            codebooks[key]["negative_entry_ids"])
        rankings.append(ranking)
    return rankings


def _cohort_blocks(rankings):
    groups = {}
    for row in rankings:
        key = f"{row['phase']}/{row['cohort']}/{row['variant']}"
        groups.setdefault(key, []).append(row)
    blocks = {}
    for key, rows in sorted(groups.items()):
        n = len(rows)
        accepted = sum(row["accepted"] for row in rows)
        correct = sum(row["accepted"] and row["top1"] for row in rows)
        blocks[key] = {
            "n": n, "raw_top1": sum(row["top1"] for row in rows),
            "raw_top1_rate": (sum(row["top1"] for row in rows) / n if n else None),
            "accepted": accepted,
            "accepted_correct": correct,
            "accepted_correct_rate": correct / n if n else None,
            "accepted_precision": correct / accepted if accepted else None,
            "stale_blocked": sum(row["stale_blocked"] for row in rows),
            "revoked_blocked": sum(row["revoked_blocked"] for row in rows),
            "stale_accepted": sum(row["stale_blocked"] and row["accepted"]
                                  for row in rows),
            "revoked_accepted": sum(row["revoked_blocked"] and row["accepted"]
                                    for row in rows),
            "policy_blocked": sum(bool(row["policy_blocked"]) for row in rows),
            "policy_blocked_acceptance": sum(
                row["policy_blocked_acceptance"] for row in rows),
            "revoked_target_other_live": sum(
                row["false_recall_kind"] == "revoked_target_other_live"
                for row in rows),
            "note": "descriptive cohort block; acceptance-v3 metrics are separate",
        }
    return blocks


def _revocation_summary(rankings):
    summary = {}
    for variant in VARIANT_NAMES:
        live = [row for row in rankings
                if row["variant"] == variant and row["phase"] == FIRST_PROCESS
                and row["cohort"] == "distractor"]
        revoked = [row for row in rankings
                   if row["variant"] == variant and row["phase"] == REVOCATION]
        live_accepted = sum(row["accepted"] for row in live)
        revoked_accepted = sum(row["accepted"] for row in revoked)
        revoked_policy_accepted = sum(
            row["accepted"] and row["policy_blocked"] == "revoked"
            for row in revoked)
        other_live = sum(
            row["false_recall_kind"] == "revoked_target_other_live"
            for row in revoked)
        summary[variant] = {
            "live_distractor_rows": len(live),
            "live_accepted": live_accepted,
            "live_accepted_rate": (live_accepted / len(live) if live else None),
            "revoked_rows": len(revoked),
            "revoked_accepted": revoked_accepted,
            "revoked_acceptance_rate": (revoked_accepted / len(revoked)
                                        if revoked else None),
            "revoked_policy_accepted": revoked_policy_accepted,
            "revoked_target_other_live_accepted": other_live,
            "revoked_blocked": sum(row["revoked_blocked"] for row in revoked),
            "control_meaningful": live_accepted > 0,
            "passed": revoked_policy_accepted == 0,
        }
    live_reference = next((item for item in summary.values()
                           if item["live_accepted"] > 0), None)
    return {
        "by_variant": summary,
        "any_live_reference_accepted": live_reference is not None,
        "all_revoked_policy_acceptance_zero": all(
            item["revoked_policy_accepted"] == 0 for item in summary.values()),
        "revoked_target_other_live_is_wrong_retrieval": True,
        "paired_text": "revocation query reuses the live distractor query text",
    }


def _representation(protocol, model):
    spec = getattr(model.manifest, "fast_memory", None)
    _require(spec is not None, "The adapter declares no fast-memory writer")
    engine = protocol["engine"]
    for key in ("heads", "key_size", "value_size"):
        _require(getattr(spec, key) == engine[key],
                 f"Writer {key} differs from the protocol engine")
    for key in ("update_rule", "depth_mode", "depth_writers"):
        _require(getattr(spec, key) == engine[key],
                 f"Writer {key} differs from the protocol engine")
    declared = protocol["representation"]
    _require(declared["name"] == FACT_REPRESENTATION,
             "Fresh representation name differs from the fact binding")
    representation = {
        "name": FACT_REPRESENTATION,
        "layers": list(protocol["layers"]),
        "heads": spec.heads, "key_size": spec.key_size,
        "value_size": spec.value_size,
        "dtype": protocol["numeric_policy"]["dtype"],
        "observation_renderer": declared["observation_renderer"],
        "query_renderer": declared["query_renderer"],
        "query_pooling": declared["query_pooling"],
        "key_space": declared["key_space"],
        "value_space": declared["value_space"],
        "declared_config": dict(declared),
        "numeric": {
            "torch_use_deterministic_algorithms":
                protocol["numeric_policy"].get(
                    "torch_use_deterministic_algorithms"),
            "mps_fast_math": protocol["numeric_policy"].get("mps_fast_math"),
        },
    }
    representation["sha256"] = fingerprint(representation)
    return representation


# ---------------------------------------------------------------------------
# Metrics, limits and the run
# ---------------------------------------------------------------------------

def _metric_bundle(rows, acceptance, *, strict=True):
    if not rows:
        verdict = {"decision": "invalid", "error": "no ranking rows",
                   "fail_closed": True, "checks": {}}
        return {"metrics": None, "verdict": verdict,
                "coverage_valid": False}
    metrics = acceptance_metrics_v3(rows, acceptance)
    require_revoked = any(row.get("phase") == REVOCATION for row in rows)
    try:
        verdict = evaluate_acceptance_v3(
            metrics, acceptance, require_revoked=require_revoked)
    except ValueError as error:
        if strict:
            raise ValueError(
                f"Acceptance coverage failed closed: {error}") from error
        verdict = {"decision": "invalid", "error": str(error),
                   "fail_closed": True, "checks": {}}
        return {"metrics": metrics, "verdict": verdict,
                "coverage_valid": False}
    return {"metrics": metrics, "verdict": verdict, "coverage_valid": True}


def refresh_positive_control_fixture(*, layers=(0,), device="cpu"):
    """Deterministic single-record fixture for admitted-refresh controls.

    A one-record state is recoverable by construction: its query key is the
    record key, so the target value must be the top value.  Re-Admitting the
    same record into a query-local history changes the state, and a two-record
    shuffled history deranges value donors.
    """
    generator = torch.Generator(device="cpu").manual_seed(20261102)
    heads, key_size, value_size = 4, 16, 16

    def write(key, value):
        return SGDN2Write(
            keys=key.view(1, 1, heads, key_size),
            values=value.view(1, 1, heads, value_size),
            erase=torch.zeros(1, 1, heads, key_size),
            write_gate=torch.ones(1, 1, heads, value_size),
            decay=torch.ones(1, 1, heads, key_size),
            chunk_index=0)

    keys = [torch.randn(heads, key_size, generator=generator)
            for _ in range(2)]
    values = [torch.randn(heads, value_size, generator=generator)
              for _ in range(2)]
    writes = {"rec-1": write(keys[0], values[0]),
              "rec-2": write(keys[1], values[1])}
    records = [{"record_id": record_id} for record_id in writes]
    context = {"records": records,
               "writes": {(record_id, layer): value
                          for record_id, value in writes.items()
                          for layer in layers}}
    primitive = SGDN2Memory(heads, key_size, value_size)

    def state(history, *, shuffle=False):
        history_writes = _history_writes(
            context, history, layers, shuffle_values=shuffle)
        return _build_state(
            primitive, layers, history_writes, reducer="shared_mean",
            gate_control=None, device=device)["shared"].weights

    base = state([records[0]])
    refreshed = state([records[0], records[0]])
    paired = state(records)
    shuffled = state(records, shuffle=True)
    query = keys[0].view(1, 1, heads, key_size)
    base_readout = _read_weights(base, query)
    refresh_readout = _read_weights(refreshed, query)
    paired_readout = _read_weights(paired, query)
    shuffled_readout = _read_weights(shuffled, query)
    target_value = values[0]
    target_cosine = _cosine_heads(refresh_readout, target_value)
    other_cosine = _cosine_heads(refresh_readout, values[1])
    oracle_top1 = bool(target_cosine > other_cosine and target_cosine > 0.99)
    refresh_changed_state = _state_sha(base) != _state_sha(refreshed)
    refresh_changed_readout = (
        _sha_json(base_readout.tolist())
        != _sha_json(refresh_readout.tolist()))
    shuffled_changed_state = _state_sha(paired) != _state_sha(shuffled)
    shuffled_changed_readout = (
        _sha_json(paired_readout.tolist())
        != _sha_json(shuffled_readout.tolist()))
    result = {
        "oracle_fixture": "single_record_query_key_recoverable",
        "oracle_target_cosine": target_cosine,
        "oracle_other_cosine": other_cosine,
        "oracle_recovers_known_fixture": oracle_top1,
        "admitted_refresh_changes_state": refresh_changed_state,
        "admitted_refresh_changes_readout": refresh_changed_readout,
        "shuffled_refresh_changes_state": shuffled_changed_state,
        "shuffled_refresh_perturbs_values": shuffled_changed_readout,
        "durable_writes": 0,
        "optimizer_updates": 0,
    }
    result["passed"] = (
        oracle_top1 and refresh_changed_state and refresh_changed_readout
        and shuffled_changed_state and shuffled_changed_readout)
    return result


def _refresh_positive_controls(rankings, decisions):
    applied = [row for row in decisions if row["refresh_applied"]]
    causal = [row for row in applied if row["variant"] == CAUSAL]
    shuffled = [row for row in applied if row["variant"] == SHUFFLED]
    oracle = [row for row in applied if row["variant"] == ORACLE]
    state_changed = sum(bool(row["refresh_changed_state"]) for row in applied)
    # Paired causal/shuffled refreshes share the same selected record and query
    # but derange values; a non-degenerate writer must produce a different state.
    by_key = {}
    for row in decisions:
        by_key[(row["episode_id"], row["state"], row["case_id"],
                row["layer"], row["variant"])] = row
    shuffled_perturbed = 0
    for (episode, state, case, layer, variant), row in by_key.items():
        if variant != SHUFFLED or not row["refresh_applied"]:
            continue
        causal_row = by_key.get((episode, state, case, layer, CAUSAL))
        if (causal_row is not None and causal_row["refresh_applied"]
                and causal_row["state_sha256"] != row["state_sha256"]):
            shuffled_perturbed += 1
    oracle_rankings = [row for row in rankings
                       if row["variant"] == ORACLE
                       and row["arm"] == "query_key_value"]
    return {
        "applied_refreshes": len(applied),
        "causal_refreshes": len(causal),
        "shuffled_refreshes": len(shuffled),
        "oracle_refreshes": len(oracle),
        "refresh_changed_state_rows": state_changed,
        "admitted_refresh_changes_state": bool(state_changed),
        "shuffled_pairs": sum(
            1 for (episode, state, case, layer, variant), row in by_key.items()
            if variant == SHUFFLED and row["refresh_applied"]
            and (episode, state, case, layer, CAUSAL) in by_key),
        "shuffled_refresh_perturbs_values": shuffled_perturbed > 0,
        "shuffled_perturbed_pairs": shuffled_perturbed,
        "oracle_applied": bool(oracle),
        "oracle_raw_top1": sum(row["top1"] for row in oracle_rankings),
        "oracle_query_rows": len(oracle_rankings),
        "oracle_recovery_rate": (
            sum(row["top1"] for row in oracle_rankings) / len(oracle_rankings)
            if oracle_rankings else None),
        "oracle_is_diagnostic_not_upper_bound": True,
    }


def _limits_check(protocol, costs, checkpoint_bytes, *, first_process,
                  exactly_once):
    limits = protocol["limits"]
    observations = costs.counts.get("observation_encoding", 0)
    queries = costs.counts.get("query_encoding", 0)
    rebuilds = costs.counts.get("state_rebuild", 0)
    readouts = costs.counts.get("readout_rows", 0)
    checks = {
        "wall_seconds": costs.wall_seconds() <= limits["max_wall_seconds_per_process"],
        "observation_encodes": observations <= limits["max_observation_encodes"],
        "query_encodes": queries <= limits["max_query_encodes_per_process"],
        "state_rebuilds": rebuilds <= limits["max_state_rebuilds_per_process"],
        "readout_rows": readouts <= limits["max_readout_rows_per_process"],
        "checkpoint_bytes": checkpoint_bytes <= limits["max_checkpoint_bytes"],
        "restart_observation_encodes": (first_process
                                        or observations == 0),
        "durable_refresh_commits": exactly_once[
            "durable_refresh_commits"] == 0,
        "query_local_refresh_applied": exactly_once[
            "query_local_refresh_applied"]
        <= limits["max_query_local_refresh_applied"],
        "record_versions": exactly_once["max_record_versions_per_record"]
        <= limits["max_record_versions_per_record"],
    }
    return {"checks": checks, "passed": all(checks.values()),
            "observed": {"observation_encodes": observations,
                         "query_encodes": queries,
                         "state_rebuilds": rebuilds,
                         "readout_rows": readouts,
                         "wall_seconds": costs.wall_seconds(),
                         "checkpoint_bytes": checkpoint_bytes,
                         "durable_refresh_commits": exactly_once[
                             "durable_refresh_commits"],
                         "query_local_refresh_applied": exactly_once[
                             "query_local_refresh_applied"],
                         "max_record_versions_per_record": exactly_once[
                             "max_record_versions_per_record"]}}


def _raw_digests(caches, decisions, readouts, rankings, codebooks):
    result = {}
    for phase in PHASE_ORDER:
        phase_decisions = [row for row in decisions if row["phase"] == phase]
        phase_readouts = [row for row in readouts if row["phase"] == phase]
        phase_rankings = [row for row in rankings if row["phase"] == phase]
        if not (phase_decisions or phase_readouts or phase_rankings):
            result[phase] = None
            continue
        phase_codebooks = {f"{key[0]}/{key[1]}/{key[2]}": value
                           for key, value in codebooks.items()
                           if key[2] == phase}
        result[phase] = {
            "decisions_sha256": _rows_sha(phase_decisions),
            "readouts_sha256": _rows_sha(phase_readouts),
            "rankings_sha256": _rows_sha(phase_rankings),
            "codebooks_sha256": _sha_json(phase_codebooks),
            "decision_rows": len(phase_decisions),
            "readout_rows": len(phase_readouts),
            "ranking_rows": len(phase_rankings),
        }
    return result


def run_memory_causal_fresh(model, tokenizer, protocol, *, output_dir, episodes,
                            labels_by_episode, revocations, writer_extension_receipt,
                            source_pins, implementation_sha256, fixture_hashes,
                            device="cpu", resume_checkpoint=None,
                            baseline_report=None, profiler=None):
    """Run the fresh causal-selector evaluation on CPU or MPS."""
    started = time.monotonic()
    _require(protocol.get("format") == PROTOCOL_FORMAT,
             "Fresh causal evaluation requires its versioned protocol")
    _require(tuple(protocol["variant_names"]) == VARIANT_NAMES,
             "Fresh causal variant inventory differs")
    _require(protocol["budget"]["optimizer_updates"] == 0,
             "Fresh causal evaluation requires zero optimizer updates")
    if protocol.get("source_prepared"):
        for key in ("episodes_sha256", "labels_sha256", "revocations_sha256",
                    "inventory_sha256"):
            _require(protocol["source_prepared"][key] == fixture_hashes[key],
                     f"Prepared fixture {key} differs from the protocol")
    if protocol.get("implementation_sha256"):
        _require(protocol["implementation_sha256"] == implementation_sha256,
                 "Implementation bytes differ from the protocol")
    if protocol.get("source_pins"):
        _require(source_pins == protocol["source_pins"],
                 "Source/model/tokenizer pins differ from the protocol")
    _require(isinstance(revocations, dict) and revocations,
             "Fresh revocation controls are missing")
    layers = tuple(protocol["layers"])
    workload = planned_workload(episodes, layers, labels_by_episode)
    if protocol.get("planned_workload"):
        _require(protocol["planned_workload"] == workload,
                 "Protocol planned workload differs from the fixture")
    thresholds = protocol["thresholds"]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    filenames = ("decisions.jsonl", "readouts.jsonl", "rankings.jsonl",
                 "codebooks.json", "ledger.json", "refresh_audit.json",
                 "report.json")
    if resume_checkpoint is None:
        filenames += ("policy_checkpoint.pt",)
    else:
        filenames += ("policy_checkpoint_restart.pt",)
    for name in filenames:
        _require(not (output_dir / name).exists(), f"Fresh output required: {name}")

    model.eval()
    _freeze(model)
    versions = _parameter_versions(model)
    declared_writer = protocol.get("writer_extension") or {}
    if declared_writer:
        _require(writer_extension_receipt.get("weights_sha256")
                 == declared_writer["weights_sha256"],
                 "Loaded writer weights differ from the protocol")
        if declared_writer.get("manifest_digest"):
            _require(writer_extension_receipt.get("manifest_digest")
                     == declared_writer["manifest_digest"],
                     "Loaded writer manifest differs from the protocol")
    primitive = SGDN2Memory(
        protocol["engine"]["heads"], protocol["engine"]["key_size"],
        protocol["engine"]["value_size"])
    representation = _representation(protocol, model)
    bindings = expected_bindings(
        protocol=protocol, fixture_hashes=fixture_hashes,
        source_pins=source_pins, writer_receipt=writer_extension_receipt,
        implementation_sha256=implementation_sha256,
        representation=representation, engine=protocol["engine"])
    costs = CostLedger()
    if profiler is not None:
        profiler.mark("fresh_evaluation_start", resumed=resume_checkpoint is not None,
                      device=device)

    expected_episodes = {episode["episode_id"]: episode for episode in episodes}
    resumed = resume_checkpoint is not None
    refresh_audit = QueryLocalRefresh(protocol["query_local_refresh"])
    if not resumed:
        caches = {}
        ledger = WriteLedger()
        with costs.observe_phase(FIRST_PROCESS):
            for episode in episodes:
                cache = build_episode_cache(
                    model, tokenizer, episode, layers, costs)
                caches[episode["episode_id"]] = cache
                for state_name in (state["name"]
                                   for state in episode_states(episode)):
                    for record in cache.states[state_name]["records"]:
                        record_id = record["record_id"]
                        version = record_version(record)
                        if ledger.is_committed(episode["episode_id"],
                                               record_id, version):
                            ledger.require_committed(
                                episode["episode_id"], record_id, version)
                            continue
                        ledger.commit(
                            attempt_id=(f"{episode['episode_id']}:observe:"
                                        f"{record_id}:{version}"),
                            episode_id=episode["episode_id"],
                            record_id=record_id, record_version=version,
                            state=state_name, source="observation")
        encoded = {
            "contexts": sum(len(cache.states) for cache in caches.values()),
            "observations": costs.counts.get("observation_encoding", 0),
            "queries": costs.counts.get("query_encoding", 0),
            "query_vectors_reused": costs.counts.get("query_vectors_reused", 0),
            "old_observations_reencoded": 0,
            "old_queries_reencoded": 0,
            "restart_new_queries": 0,
        }
        costs.add("cached_observation_writes",
                  sum(len(cache.writes) for cache in caches.values()))
        costs.add("cached_query_vectors",
                  sum(len(cache.query_vectors) for cache in caches.values()))
        refresh_audit.bind_durable(ledger)
    else:
        with costs.timed("cache_load"):
            payload = load_fresh_checkpoint(
                resume_checkpoint, expected=bindings, device=device)
        checkpoint_bytes = Path(resume_checkpoint).stat().st_size
        costs.add("checkpoint_bytes", checkpoint_bytes)
        costs.add("checkpoint_loads")
        ledger = WriteLedger.from_dict(payload["ledger"])
        refresh_audit.bind_durable(ledger)
        packed = {item["episode_id"]: item for item in payload["episodes"]}
        _require(set(packed) == set(expected_episodes),
                 "Cached episode inventory differs")
        caches = {}
        with costs.observe_phase(RESTART):
            for episode in episodes:
                cache = _unpack_cache(
                    packed[episode["episode_id"]], episode, device)
                _require(tuple(cache.layers) == layers,
                         "Cached layers differ from the protocol")
                for state_name in (state["name"]
                                   for state in episode_states(episode)):
                    for record in cache.states[state_name]["records"]:
                        ledger.require_committed(
                            episode["episode_id"], record["record_id"],
                            record_version(record))
                for query in episode["queries"]:
                    _require(cache.query_phases.get(query["query_id"], (None,))[0]
                             == FIRST_PROCESS,
                             "Cached first-process query is missing")
                    _require(cache.query_phases[query["query_id"]][1]
                             == _sha_json(query["text"]),
                             "Cached query text differs from the fixture")
                before_observation = costs.counts.get("observation_encoding", 0)
                added = add_phase_queries(
                    cache, model, tokenizer, RESTART, costs)
                added_revocation = add_phase_queries(
                    cache, model, tokenizer, REVOCATION, costs)
                _require(costs.counts.get("observation_encoding", 0)
                         == before_observation,
                         "Restart must not encode old observations")
                costs.add("restart_query_sets", added["queries"])
                costs.add("restart_revocation_queries",
                          added_revocation["queries"])
                costs.add("restart_new_query_texts",
                          added["new_query_texts"]
                          + added_revocation["new_query_texts"])
                caches[episode["episode_id"]] = cache
            _require(costs.counts.get("observation_encoding", 0) == 0,
                     "Restart must not encode any observation")
            restart_encoded = costs.counts.get("query_encoding", 0)
            _require(restart_encoded > 0,
                     "Restart must encode genuinely new queries")
        encoded = {
            "contexts": 0,
            "observations": costs.counts.get("observation_encoding", 0),
            "queries": restart_encoded,
            "query_vectors_reused": costs.counts.get("query_vectors_reused", 0),
            "old_observations_reencoded": 0,
            "old_queries_reencoded": 0,
            "restart_new_queries": restart_encoded,
        }
        costs.add("cached_observation_writes_loaded",
                  sum(len(cache.writes) for cache in caches.values()))
        costs.add("cached_query_vectors_loaded",
                  sum(len(cache.query_vectors) for cache in caches.values()))

    phases = [FIRST_PROCESS]
    if resumed:
        phases += [RESTART, REVOCATION]
    decisions, readouts, codebooks = [], [], {}
    for phase in phases:
        with costs.observe_phase(f"evaluate_{phase}"):
            for episode in episodes:
                cache = caches[episode["episode_id"]]
                for state in episode_states(episode):
                    state_name = state["name"]
                    labels = labels_by_episode[episode["episode_id"]][state_name]
                    revoked_ids = (
                        (revocations[episode["episode_id"]]["record_id"],)
                        if phase == REVOCATION else ())
                    codebooks[(episode["episode_id"], state_name, phase)] = \
                        _codebook(cache, state_name, phase, revoked_ids)
                    queries = episode[PHASE_QUERY_FIELDS[phase]]
                    for variant in VARIANT_NAMES:
                        for query in queries:
                            new_decisions, new_rows = _evaluate_query(
                                cache, state_name, labels, query, variant,
                                primitive=primitive, device=device,
                                thresholds=thresholds, ledger=ledger,
                                refresh_audit=refresh_audit,
                                costs=costs, phase=phase,
                                revoked_ids=revoked_ids)
                            decisions.extend(new_decisions)
                            readouts.extend(new_rows)
                            if len(readouts) > protocol["limits"][
                                    "max_readout_rows_per_process"]:
                                raise ValueError("Readout row limit exceeded")

    if not resumed:
        checkpoint_path = output_dir / "policy_checkpoint.pt"
        with costs.timed("checkpoint_save"):
            save_fresh_checkpoint(
                checkpoint_path, bindings=bindings,
                caches=[caches[episode["episode_id"]] for episode in episodes],
                ledger=ledger, phase=FIRST_PROCESS,
                counters={"observation_encodes": encoded["observations"],
                          "query_encodes": encoded["queries"]})
        checkpoint_bytes = checkpoint_path.stat().st_size
        costs.add("checkpoint_bytes", checkpoint_bytes)
        checkpoint_report = {
            "path": _display_path(checkpoint_path),
            "sha256": _sha_file(checkpoint_path),
            "bytes": checkpoint_bytes,
            "new_checkpoint_saved": True,
            "loaded": False,
        }
    else:
        restart_checkpoint = output_dir / "policy_checkpoint_restart.pt"
        with costs.timed("checkpoint_save"):
            save_fresh_checkpoint(
                restart_checkpoint, bindings=bindings,
                caches=[caches[episode["episode_id"]] for episode in episodes],
                ledger=ledger, phase=RESTART,
                counters={"query_encodes_restart": encoded["queries"]})
        restart_bytes = restart_checkpoint.stat().st_size
        costs.add("checkpoint_bytes", restart_bytes)
        checkpoint_report = {
            "loaded_path": _display_path(resume_checkpoint),
            "loaded_sha256": _sha_file(resume_checkpoint),
            "loaded_bytes": checkpoint_bytes,
            "load_seconds": costs.seconds.get("cache_load", 0.0),
            "path": _display_path(restart_checkpoint),
            "sha256": _sha_file(restart_checkpoint),
            "bytes": restart_bytes,
            "new_checkpoint_saved": True,
            "loaded": True,
        }

    rankings = _rank(readouts, decisions, codebooks, labels_by_episode,
                     revocations, thresholds)
    costs.add("readout_rows", len(readouts))
    _require(_parameter_versions(model) == versions,
             "Frozen writer parameters changed")
    refresh_audit.assert_durable_unchanged(ledger)
    for row in decisions + readouts:
        _check_no_label_keys(row)
    _write_jsonl(output_dir / "decisions.jsonl", decisions)
    _write_jsonl(output_dir / "readouts.jsonl", readouts)
    _write_jsonl(output_dir / "rankings.jsonl", rankings)
    codebook_doc = {f"{key[0]}/{key[1]}/{key[2]}": value
                    for key, value in sorted(codebooks.items())}
    _write_json(output_dir / "codebooks.json", {"codebooks": codebook_doc})
    _write_json(output_dir / "ledger.json", ledger.to_dict())
    _write_json(output_dir / "refresh_audit.json", refresh_audit.summary())

    raw = _raw_digests(caches, decisions, readouts, rankings, codebooks)
    acceptance = protocol["acceptance"]
    first_rows = [row for row in rankings if row["phase"] == FIRST_PROCESS]
    metric_bundles = {"first_process_all": _metric_bundle(first_rows, acceptance)}
    if resumed:
        restart_rows = [row for row in rankings if row["phase"] == RESTART]
        revocation_rows = [row for row in rankings if row["phase"] == REVOCATION]
        post_rows = restart_rows + revocation_rows
        restart_heldout = [row for row in post_rows if row["split"] == "heldout"]
        metric_bundles["restart_all"] = _metric_bundle(post_rows, acceptance)
        metric_bundles["restart_heldout"] = _metric_bundle(
            restart_heldout, acceptance)
        metric_bundles["restart_by_split"] = {
            split: _metric_bundle(
                [row for row in post_rows if row["split"] == split],
                acceptance, strict=False)
            for split in ("train", "dev", "heldout")}
        metric_bundles["revocation_descriptive"] = {
            "metrics": None, "verdict": None,
            "cohort": _cohort_blocks(revocation_rows),
            "revocation": metric_bundles["restart_all"]["metrics"][
                "variants"][acceptance["candidate_variant"]]["revoked"]}
    cohorts = _cohort_blocks(rankings)
    revocation = _revocation_summary(rankings) if resumed else None

    retry = ledger.retry_probe()
    exactly_once = ledger.summary()
    exactly_once["retry"] = retry
    restart_replay_rejected = False
    if resumed:
        probe = WriteLedger.from_dict(ledger.to_dict())
        probe_episode = episodes[0]
        probe_record = _record_views(
            probe_episode, episode_states(probe_episode)[0])[0][0]
        try:
            probe.commit(
                attempt_id="restart-replay-probe",
                episode_id=probe_episode["episode_id"],
                record_id=probe_record["record_id"],
                record_version=record_version(probe_record), state="initial",
                source="observation")
        except DuplicateWrite:
            restart_replay_rejected = True
    refresh_summary = refresh_audit.summary()
    refresh_probe = refresh_audit.retry_probe(ledger)
    exactly_once.update({
        "restart_replay_rejected": restart_replay_rejected,
        "durable_refresh_commits": exactly_once[
            "commit_source_counts"].get("accepted_refresh", 0),
        "durable_identity": "episode_id/record_id/record_version",
        "record_version_counts": {
            key: len(versions) for key, versions in
            exactly_once["per_record_versions"].items()},
        "max_record_versions_per_record": max(
            (len(versions) for versions in
             exactly_once["per_record_versions"].values()), default=0),
        "query_local_refresh": refresh_summary,
        "query_local_refresh_applied": refresh_summary[
            "status_counts"].get("applied", 0),
        "query_local_refresh_rejected": refresh_summary[
            "status_counts"].get("rejected", 0),
        "query_local_refresh_retry": refresh_probe,
        "query_local_refresh_durable_unchanged": refresh_probe[
            "durable_unchanged"],
        "duplicate_observe_commits_rejected": exactly_once[
            "rejection_counts"].get("observation", 0),
    })
    exactly_once["passed"] = (
        exactly_once["max_record_versions_per_record"]
        <= protocol["limits"]["max_record_versions_per_record"]
        and exactly_once["durable_refresh_commits"] == 0
        and exactly_once["duplicate_observe_commits_rejected"]
        <= protocol["limits"]["max_duplicate_observe_commits"]
        and exactly_once["query_local_refresh_applied"]
        <= protocol["limits"]["max_query_local_refresh_applied"]
        and retry["idempotent"]
        and refresh_probe["idempotent"]
        and refresh_probe["durable_unchanged"]
        and (not resumed or exactly_once["restart_replay_rejected"])
    )
    exactly_once["ledger_sha256"] = _sha_file(output_dir / "ledger.json")
    exactly_once["refresh_audit_sha256"] = _sha_file(
        output_dir / "refresh_audit.json")
    _require(exactly_once["durable_refresh_commits"] == 0,
             "Query-local refresh created a durable commit")
    _require(exactly_once["query_local_refresh_applied"]
             <= protocol["limits"]["max_query_local_refresh_applied"],
             "Query-local refresh applied beyond the predeclared limit")
    _require(exactly_once["duplicate_observe_commits_rejected"]
             <= protocol["limits"]["max_duplicate_observe_commits"],
             "Duplicate observation commits exceeded the predeclared limit")
    _require(exactly_once["max_record_versions_per_record"]
             <= protocol["limits"]["max_record_versions_per_record"],
             "Record version bound exceeded")

    blocked = {
        "stale_blocked": sum(row["stale_blocked"] for row in rankings),
        "revoked_blocked": sum(row["revoked_blocked"] for row in rankings),
        "revoked_query_kind": sum(row["revoked_query_kind"]
                                  for row in rankings),
        "policy_blocked_acceptance": sum(
            row["policy_blocked_acceptance"] for row in rankings),
        "stale_accepted": sum(row["stale_blocked"] and row["accepted"]
                              for row in rankings),
        "revoked_policy_accepted": sum(
            row["accepted"] and row["policy_blocked"] == "revoked"
            for row in rankings),
        "revoked_target_other_live": sum(
            row["false_recall_kind"] == "revoked_target_other_live"
            for row in rankings),
    }
    _require(blocked["stale_accepted"] == 0,
             "Stale records must not be accepted")
    _require(blocked["revoked_policy_accepted"] == 0,
             "Revoked records must not be accepted by runtime policy")

    parity = None
    if resumed:
        _require(isinstance(baseline_report, dict),
                 "Resume requires the first-process baseline report")
        baseline_raw = baseline_report["raw"][FIRST_PROCESS]
        current_raw = raw[FIRST_PROCESS]
        raw_match = baseline_raw == current_raw
        metrics_match = (_sha_json(baseline_report["metrics"][
            "first_process_all"]["metrics"])
            == _sha_json(metric_bundles["first_process_all"]["metrics"]))
        protocol_match = (baseline_report["protocol_sha256"]
                          == protocol["protocol_sha256"])
        parity = {
            "baseline_protocol_match": protocol_match,
            "first_process_raw_match": raw_match,
            "first_process_metrics_match": metrics_match,
            "same_query_parity": raw_match and metrics_match and protocol_match,
        }
        _require(parity["same_query_parity"],
                 "Fresh-process same-query parity differs")

    refresh_controls = _refresh_positive_controls(rankings, decisions)
    positive_control = refresh_positive_control_fixture(device="cpu")
    refresh_controls["positive_control_fixture"] = positive_control
    mechanical = {
        "frozen_writer": _parameter_versions(model) == versions,
        "requires_grad_all_false": all(
            not parameter.requires_grad for parameter in model.parameters()),
        "zero_optimizer_updates": protocol["budget"]["optimizer_updates"] == 0,
        "exactly_once": exactly_once["passed"],
        "checkpoint_binding": True,
        "restart_without_old_observation_encode": (
            not resumed or encoded["observations"] == 0),
        "restart_new_queries_encoded": (
            not resumed or encoded["restart_new_queries"] > 0),
        "same_query_parity": (not resumed or parity["same_query_parity"]),
        "revocation_blocks_acceptance": (
            not resumed or (revocation["all_revoked_policy_acceptance_zero"]
                            and blocked["revoked_policy_accepted"] == 0)),
        "stale_blocks_acceptance": blocked["stale_accepted"] == 0,
        "query_local_refresh_durable_unchanged": exactly_once[
            "query_local_refresh_durable_unchanged"],
        "causal_refresh_admitted": (
            not resumed or exactly_once["query_local_refresh_applied"] > 0),
        "refresh_positive_control": positive_control["passed"],
        "coverage_valid": metric_bundles["first_process_all"]["coverage_valid"],
        "representation_bound": True,
    }
    decisive = metric_bundles.get(acceptance["decisive_cohort"]) \
        if resumed else metric_bundles["first_process_all"]
    oracle_rows = (metric_bundles["restart_all"] if resumed
                   else metric_bundles["first_process_all"])
    oracle = oracle_rows["metrics"]["variants"][ORACLE]["quality"]
    oracle_ok = (oracle["raw_top1_rate"] is not None
                 and oracle["raw_top1_rate"]
                 >= acceptance["gates"]["oracle_raw_top1_rate_min"])
    if not resumed:
        decision = ("invalid" if not decisive["coverage_valid"]
                    else "inconclusive")
    elif not decisive["coverage_valid"]:
        decision = "invalid"
    elif all(mechanical.values()) and decisive["verdict"]["decision"] == "accepted":
        decision = "accepted"
    elif oracle_ok:
        decision = "rejected"
    else:
        decision = "inconclusive"
    limits = _limits_check(
        protocol, costs, checkpoint_report["bytes"],
        first_process=not resumed, exactly_once=exactly_once)
    _require(limits["passed"], "Fresh protocol limit exceeded")

    report = {
        "format": RUN_FORMAT, "status": "completed", "decision": decision,
        "phase": RESTART if resumed else FIRST_PROCESS,
        "started_at_utc": _utc_now(),
        "elapsed_seconds": time.monotonic() - started,
        "protocol_sha256": protocol["protocol_sha256"],
        "bindings": bindings,
        "writer_extension_receipt": dict(writer_extension_receipt),
        "representation": representation,
        "optimizer_updates": 0,
        "native": device == "mps", "device": device,
        "resumed": resumed,
        "encoded": encoded,
        "checkpoint": checkpoint_report,
        "exactly_once": exactly_once,
        "refresh": refresh_summary,
        "refresh_controls": refresh_controls,
        "refresh_positive_control": positive_control,
        "revocation": revocation,
        "blocked": blocked,
        "inventory": {
            "declared_query_rows": workload["declared_query_rows"],
            "inventory_case_rows": workload["inventory_case_rows"],
            "arm_expanded_metric_rows": workload["arm_expanded_metric_rows"],
            "process_ranking_rows": len(rankings),
            "process_phases": list(phases),
            "units": workload["units"],
            "note": ("declared query rows are not inventory case rows and are "
                     "not arm-expanded metric rows"),
        },
        "representation_compatibility": {
            "protocol_bound": True, "source_bound": True,
            "writer_bound": True, "representation_bound": True,
            "implementation_bound": True, "engine_bound": True,
        },
        "costs": costs.to_dict(),
        "limits": limits,
        "metrics": metric_bundles,
        "cohorts": cohorts,
        "gates": {"mechanical": mechanical,
                  "mechanical_pass": all(mechanical.values()),
                  "quality_decision": decisive["verdict"]["decision"],
                  "oracle_raw_bound": oracle_ok},
        "parity": parity,
        "classification": {
            "full_history": "frozen learned-memory baseline",
            "causal_refresh": ("hybrid query-addressed cached-observation "
                               "retrieval plus neural refresh"),
            "oracle_refresh": ("gold-target diagnostic population; not a legal "
                               "policy and not a candidate-quality upper bound"),
            "disabled": "zero state and zero value readout control",
            "shuffled": "cyclically deranged value-vector control",
            "exact_key_value": ("mixture diagnostic for a gold key readout; "
                                "not an upper bound"),
            "query_key_value": "primary candidate-quality readout",
            "query_local_refresh_at_most_once_per_query": True,
            "durable_refresh_commits": 0,
            "learned_selector": False,
            "useful_memory_claim": False,
        },
        "generation": {
            "status": "not_measured", "executed": False,
            "note": ("Generation is excluded from this protocol; no generated-answer "
                     "claim follows from retrieval/readout rows."),
        },
        "raw": raw,
        "raw_policy": {
            "decisions_and_readouts_label_free": True,
            "files_immutable_first_write": True,
        },
        "stop_conditions": list(protocol["stop_conditions"]),
        "limitations": list(protocol["limitations"]),
    }
    _write_json(output_dir / "report.json", report)
    if profiler is not None:
        profiler.mark("fresh_evaluation_done", decision=decision,
                      observation_encodes=encoded["observations"],
                      query_encodes=encoded["queries"],
                      state_rebuilds=costs.counts.get("state_rebuild", 0),
                      checkpoint_bytes=checkpoint_report["bytes"])
    return report

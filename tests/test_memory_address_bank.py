"""Contracts for the explicit-record address-bank control.

Synthetic tests cover exact selection, correction versus unaffected facts,
unknown/revoked abstention, ambiguity, duplicates, episode scope, reader
independence, decision-before-label ordering and dev stop gating.  One optional
integration test replays the real prepared04 fixture when that ignored local
artifact is provisioned; no native command, model or tokenizer is used.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from replica_lab.memory_address_bank import (
    ABSTAIN_REASONS,
    DEFAULT_PHASE_MAP,
    DEV_FACT_SELECTION_IDENTITY_MIN,
    REPORT_FORMAT,
    build_split_decisions,
    decision_digest,
    decision_key_register,
    evaluate_dev_gate,
    expected_decision_rows,
    freeze_decisions,
    load_scoring_labels,
    load_selection_inputs,
    resolve_address,
    revocation_applies,
    run_control,
    run_prepared_control,
    score_decisions,
    validate_config,
)
from replica_lab.memory_matched import CORRECTION_DAY, INITIAL_DAY

ROOT = Path(__file__).resolve().parents[1]
REAL_CONFIG = ROOT / "configs/memory_address_bank_v1.json"
PREPARED04 = ROOT / "data/memory_causal_fresh_20261026/prepared04"
FORBIDDEN_DECISION_KEYS = {
    "expected", "expected_answer", "gold", "gold_answer", "label",
    "audit_label", "kind", "role", "target_role", "cohort", "value",
}


def _config():
    return json.loads(REAL_CONFIG.read_text())


def _fact(record_id, entity, attribute, value, *, observed_at=INITIAL_DAY,
          world_id="world-a"):
    return {
        "record_id": record_id, "entity": entity, "attribute": attribute,
        "value": value, "observed_at": observed_at,
        "source_world_id": world_id,
    }


def _correction(record_id, entity, attribute, value, supersedes,
                *, observed_at=CORRECTION_DAY):
    return {
        "record_id": record_id, "entity": entity, "attribute": attribute,
        "value": value, "observed_at": observed_at,
        "supersedes": supersedes,
    }


def _distractor(record_id, entity, attribute, value, *, world_id="world-b"):
    return {
        "record_id": record_id, "entity": entity, "attribute": attribute,
        "value": value, "observed_at": INITIAL_DAY,
        "origin_record_id": record_id, "origin_world_id": world_id,
    }


def _query(query_id, entity, attribute, phase, cohort="paraphrase", form=0):
    return {
        "query_id": query_id, "entity": entity, "attribute": attribute,
        "phase": phase, "cohort": cohort, "form": form,
        "text": f"address {entity}/{attribute}",
    }


def _page(page_id, observed_at, lines):
    return {
        "page_id": page_id, "observed_at": observed_at,
        "date": "2026-03-01" if observed_at == INITIAL_DAY else "2026-04-01",
        "lines": list(lines),
    }


def _episode(episode_id, *, split="dev", world_id="world-a",
             facts, corrections=(), distractors=(), queries=(),
             restart_queries=(), revocation_queries=(),
             complexity="correction", variant=0):
    observations = [_page(f"page-{episode_id}-initial", INITIAL_DAY,
                          ["line" for _ in facts])]
    if corrections:
        observations.append(_page(f"page-{episode_id}-corrected",
                                  CORRECTION_DAY, ["update line"]))
    return {
        "episode_id": episode_id, "family_id": f"family-{episode_id}",
        "world_id": world_id, "split": split, "complexity": complexity,
        "variant": variant, "facts": list(facts),
        "corrections": list(corrections), "distractors": list(distractors),
        "observations": observations, "queries": list(queries),
        "restart_queries": list(restart_queries),
        "revocation_queries": list(revocation_queries),
    }


def _simple_episode(episode_id, split):
    world = f"world-{episode_id}"
    facts = [
        _fact("rec-a", "ent-a", "attr-a", "val-a", world_id=world),
        _fact("rec-b", "ent-b", "attr-b", "val-b", world_id=world),
    ]
    queries = [
        _query("q-a", "ent-a", "attr-a", "first_process"),
        _query("q-b", "ent-b", "attr-b", "first_process"),
    ]
    restart = [
        _query("qr-a", "ent-a", "attr-a", "restart", form=1),
        _query("qr-b", "ent-b", "attr-b", "restart", form=1),
    ]
    return _episode(episode_id, split=split, world_id=world, facts=facts,
                    queries=queries, restart_queries=restart,
                    complexity="pair")


def _correction_episode(*, split="dev", revoke=True, ambiguous=False):
    world = "world-a"
    facts = [
        _fact("rec-a1", "ent-a", "attr-a", "val-a1", world_id=world),
        _fact("rec-b1", "ent-b", "attr-b", "val-b1", world_id=world),
    ]
    if ambiguous:
        facts.append(_fact("rec-a1x", "ent-a", "attr-a", "val-a1x",
                           world_id=world))
    corrections = [
        _correction("rec-a2", "ent-a", "attr-a", "val-a2", "rec-a1"),
    ]
    distractors = [
        _distractor("rec-d1", "ent-d", "attr-d", "val-d1"),
    ]
    queries = [
        _query("q-a", "ent-a", "attr-a", "first_process"),
        _query("q-b", "ent-b", "attr-b", "first_process"),
        _query("q-d", "ent-d", "attr-d", "first_process",
               cohort="distractor", form=2),
        _query("q-unknown", "ent-u", "attr-u", "first_process",
               cohort="unknown", form=3),
    ]
    restart = [
        _query("qr-a", "ent-a", "attr-a", "restart", form=2),
        _query("qr-b", "ent-b", "attr-b", "restart", form=2),
        _query("qr-d", "ent-d", "attr-d", "restart",
               cohort="distractor", form=3),
        _query("qr-unknown", "ent-u2", "attr-u2", "restart",
               cohort="unknown", form=0),
    ]
    revocation = [
        _query("q-revoked", "ent-d", "attr-d", "revocation",
               cohort="revoked", form=2),
    ]
    episode = _episode(
        "episode-a", split=split, world_id=world, facts=facts,
        corrections=corrections, distractors=distractors, queries=queries,
        restart_queries=restart, revocation_queries=revocation)
    revocations = {}
    if revoke:
        revocations[episode["episode_id"]] = {
            "record_id": "rec-d1", "entity": "ent-d", "attribute": "attr-d",
            "applies_at": "post_restart", "scope": "view_only",
            "cached_history_retained": True,
        }
    return episode, revocations


def _labels_for_episode(episode):
    correction_ids = {item["record_id"] for item in episode["corrections"]}
    correction_addresses = {(item["entity"], item["attribute"])
                            for item in episode["corrections"]}
    corrections = {item["supersedes"] for item in episode["corrections"]}
    labels = {}
    for state_name, cutoff in (("initial", INITIAL_DAY),
                               ("corrected", CORRECTION_DAY)):
        if state_name == "corrected" and not episode["corrections"]:
            continue
        records = (episode["facts"] + episode["distractors"]
                   + episode["corrections"])
        live = [record for record in records
                if record["observed_at"] <= cutoff
                and record["record_id"] not in corrections]
        live_by_address = {}
        for record in live:
            live_by_address.setdefault(
                (record["entity"], record["attribute"]), record)
        targets, phase_by_query, cohort_by_query = {}, {}, {}
        corrected, unaffected, unknown = [], [], []
        for phase, field in (("first_process", "queries"),
                             ("restart", "restart_queries"),
                             ("revocation", "revocation_queries")):
            for query in episode[field]:
                query_id = query["query_id"]
                cohort_by_query[query_id] = query["cohort"]
                phase_by_query[query_id] = phase
                address = (query["entity"], query["attribute"])
                if phase == "revocation":
                    targets[query_id] = {
                        "expected": "revoked", "kind": "revoked",
                        "target_entry_id": None, "target_role": "revoked",
                        "target_source_family": "source",
                        "query_form": query["form"],
                    }
                    continue
                record = live_by_address.get(address)
                if record is None:
                    targets[query_id] = {
                        "expected": "unknown", "kind": "unknown",
                        "target_entry_id": None, "target_role": "unknown",
                        "target_source_family": "none",
                        "query_form": query["form"],
                    }
                    unknown.append(query_id)
                    continue
                role = "source"
                if record["record_id"] in correction_ids:
                    role = "correction_target"
                elif state_name == "corrected":
                    role = "retained"
                family = ("independent_world"
                          if record.get("origin_world_id", episode["world_id"])
                          != episode["world_id"] else "source")
                targets[query_id] = {
                    "expected": record["value"], "kind": "fact",
                    "target_entry_id": record["record_id"],
                    "target_role": role,
                    "target_source_family": family,
                    "query_form": query["form"],
                }
                if role == "correction_target" or (
                        state_name == "corrected"
                        and address in correction_addresses):
                    corrected.append(query_id)
                elif role == "retained" and family == "source":
                    unaffected.append(query_id)
        labels[state_name] = {
            "episode_id": episode["episode_id"],
            "world_id": episode["world_id"],
            "family_id": episode["family_id"],
            "split": episode["split"],
            "complexity": episode["complexity"],
            "variant": episode["variant"],
            "state": state_name,
            "cutoff": cutoff,
            "targets": targets,
            "phase_by_query": phase_by_query,
            "cohort_by_query": cohort_by_query,
            "corrected_query_ids": sorted(corrected),
            "unaffected_query_ids": sorted(unaffected),
            "unknown_query_ids": sorted(unknown),
            "evaluation_only": True,
        }
    return labels


def _labels_for(episodes):
    return {episode["episode_id"]: _labels_for_episode(episode)
            for episode in episodes}


def _by_query(decisions):
    return {(row["state"], row["phase"], row["query_id"]): row
            for row in decisions}


def _assert_no_label_fields(value, path="decision"):
    if isinstance(value, dict):
        for key, item in value.items():
            assert key not in FORBIDDEN_DECISION_KEYS, f"{path}.{key}"
            _assert_no_label_fields(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_no_label_fields(item, f"{path}[{index}]")


# ---------------------------------------------------------------------------
# Synthetic selection contracts
# ---------------------------------------------------------------------------

def test_exact_correction_unaffected_unknown_and_revoked():
    config = _config()
    episode, revocations = _correction_episode(revoke=True)
    decisions = build_split_decisions(
        [episode], revocations, config, "dev")
    indexed = _by_query(decisions)
    assert len(decisions) == expected_decision_rows([episode])
    assert decision_key_register(decisions)["unique"] is True

    initial_a = indexed[("initial", "first_process", "q-a")]
    corrected_a = indexed[("corrected", "first_process", "q-a")]
    assert initial_a["selected_ids"] == ["rec-a1"]
    assert corrected_a["selected_ids"] == ["rec-a2"]
    assert corrected_a["excluded_superseded_ids"] == ["rec-a1"]
    assert corrected_a["excluded_reasons"] == [
        {"record_id": "rec-a1", "reason": "superseded"}]

    for state in ("initial", "corrected"):
        unaffected = indexed[(state, "first_process", "q-b")]
        assert unaffected["selected_ids"] == ["rec-b1"]
        assert unaffected["decision"] == "select"

    unknown = indexed[("initial", "first_process", "q-unknown")]
    assert unknown["reason"] == "abstain_no_address"
    assert unknown["selected_ids"] == []

    first_distractor = indexed[("initial", "first_process", "q-d")]
    assert first_distractor["selected_ids"] == ["rec-d1"]
    assert first_distractor["revocation_applied"] is False

    for state in ("initial", "corrected"):
        revoked = indexed[(state, "revocation", "q-revoked")]
        assert revoked["reason"] == "abstain_revoked_only_address"
        assert revoked["selected_ids"] == []
        assert revoked["excluded_revoked_ids"] == ["rec-d1"]
        restart_distractor = indexed[(state, "restart", "qr-d")]
        assert restart_distractor["reason"] == "abstain_revoked_only_address"
        assert restart_distractor["selected_ids"] == []

    _assert_no_label_fields(decisions)
    assert "val-" not in json.dumps(decisions)


def test_ambiguity_superseded_only_and_duplicate_handling():
    config = _config()
    record_a = _fact("rec-a", "ent-a", "attr-a", "val-a")
    record_a_other = _fact("rec-a-other", "ent-a", "attr-a", "val-other")

    ambiguous = resolve_address(
        "ent-a", "attr-a", [record_a, record_a_other], [])
    assert ambiguous["reason"] == "abstain_ambiguous_address"
    assert ambiguous["selected_ids"] == []
    assert ambiguous["live_candidate_ids"] == ["rec-a", "rec-a-other"]

    superseded = resolve_address("ent-a", "attr-a", [], [record_a])
    assert superseded["reason"] == "abstain_superseded_only_address"
    assert superseded["excluded_superseded_ids"] == ["rec-a"]

    revoked = resolve_address(
        "ent-a", "attr-a", [record_a], [], revoked_ids={"rec-a"})
    assert revoked["reason"] == "abstain_revoked_only_address"
    assert revoked["excluded_revoked_ids"] == ["rec-a"]

    missing = resolve_address("ent-z", "attr-z", [record_a], [])
    assert missing["reason"] == "abstain_no_address"

    duplicate = resolve_address(
        "ent-a", "attr-a", [record_a, dict(record_a)], [])
    assert duplicate["reason"] == "selected_exact_address"
    assert duplicate["selected_ids"] == ["rec-a"]

    conflict = dict(record_a)
    conflict["value"] = "val-different"
    with pytest.raises(ValueError, match="conflicting content"):
        resolve_address("ent-a", "attr-a", [record_a, conflict], [])

    episode, _ = _correction_episode(revoke=False)
    episode["facts"].append(
        _fact("rec-a1x", "ent-a", "attr-a", "val-a1x", world_id="world-a"))
    decisions = build_split_decisions([episode], {}, config, "dev")
    initial = _by_query(decisions)[("initial", "first_process", "q-a")]
    assert initial["reason"] == "abstain_ambiguous_address"
    assert set(initial["live_candidate_ids"]) == {"rec-a1", "rec-a1x"}


def test_decisions_never_cross_episode_or_world_scope():
    config = _config()
    episode_a = _simple_episode("episode-a", "dev")
    episode_b = _simple_episode("episode-b", "dev")
    episode_b["facts"][0]["record_id"] = "rec-b-world"
    episode_b["facts"][0]["entity"] = "ent-a"
    episode_b["facts"][0]["attribute"] = "attr-a"
    decisions = build_split_decisions(
        [episode_a, episode_b], {}, config, "dev")
    selected = {(row["episode_id"], row["query_id"]): row["selected_ids"]
                for row in decisions if row["decision"] == "select"}
    assert selected[("episode-a", "q-a")] == ["rec-a"]
    assert selected[("episode-b", "q-a")] == ["rec-b-world"]
    for row in decisions:
        assert row["scope_world_id"] == row["world_id"]
        if row["episode_id"] == "episode-a":
            assert "rec-b-world" not in row["selected_ids"]


def test_reader_layers_never_multiply_decisions():
    config = _config()
    episode = _simple_episode("episode-a", "dev")
    first = build_split_decisions([episode], {}, config, "dev")
    changed = deepcopy(config)
    changed["reader"]["declared_source_layers"] = [0, 1, 2, 3]
    validate_config(changed)
    second = build_split_decisions([episode], {}, changed, "dev")
    assert len(first) == expected_decision_rows([episode])
    assert len(first) == len(second)
    assert decision_digest(first) == decision_digest(second)
    assert all("layer" not in row for row in first)
    assert decision_key_register(first) == {
        "rows": len(first), "unique_keys": len(first), "unique": True}


def test_revocation_phase_mapping_is_declared():
    config = _config()
    episode, revocations = _correction_episode()
    revocation = revocations[episode["episode_id"]]
    assert revocation_applies(revocation, "first_process") is False
    assert revocation_applies(revocation, "restart") is True
    assert revocation_applies(revocation, "revocation") is True
    assert DEFAULT_PHASE_MAP["post_restart"] == ("restart", "revocation")
    assert tuple(config["bank"]["abstain_reasons"]) == ABSTAIN_REASONS


def test_endpoint_cannot_be_tuned():
    config = _config()
    changed = deepcopy(config)
    changed["endpoint"]["dev_fact_selection_identity_min"] = 0.80
    with pytest.raises(ValueError, match="may not be tuned"):
        validate_config(changed)
    changed = deepcopy(config)
    changed["endpoint"]["dev_unknown_selections_max"] = 1
    with pytest.raises(ValueError, match="may not be tuned"):
        validate_config(changed)
    changed = deepcopy(config)
    changed["endpoint"]["dev_revoked_selections_max"] = 1
    with pytest.raises(ValueError, match="may not be tuned"):
        validate_config(changed)


def test_dev_pass_freezes_before_labels_and_then_writes_heldout(tmp_path):
    config = _config()
    dev = _simple_episode("episode-dev", "dev")
    heldout = _simple_episode("episode-heldout", "heldout")
    labels = _labels_for([dev, heldout])
    output = tmp_path / "control-pass"

    def labels_loader():
        # Labels are first requested after immutable dev decisions exist.
        assert (output / "decisions_dev.jsonl").is_file()
        assert not (output / "decisions_heldout.jsonl").exists()
        assert not (output / "report.json").exists()
        return labels

    report = run_control(
        [dev, heldout], {}, labels_loader, config, output,
        prepared_info={"mode": "synthetic"})
    assert report["format"] == REPORT_FORMAT
    assert report["gating"]["dev"]["passed"] is True
    assert report["gating"]["dev"]["evaluated_first"] is True
    assert report["gating"]["heldout"]["status"] == "evaluated"
    assert report["gating"]["stopped_before_heldout"] is False
    assert (output / "decisions_dev.jsonl").is_file()
    assert (output / "decisions_heldout.jsonl").is_file()
    assert report["raw_outputs"]["verified_unchanged"] is True
    assert report["counts"]["decision_keys"]["unique"] is True
    assert report["split_metrics"]["dev"]["overall"][
        "fact_selection_identity"] == 1.0
    assert report["split_metrics"]["heldout"]["overall"][
        "fact_selection_identity"] == 1.0
    assert report["classification"]["model_execution"] is False
    assert report["classification"]["generation"] is False
    assert report["classification"]["bank_plus_compressed_arm"] is False


def test_dev_failure_stops_before_any_heldout_decision_or_metric(tmp_path):
    config = _config()
    dev, revocations = _correction_episode(ambiguous=True)
    heldout = _simple_episode("episode-heldout", "heldout")
    labels = _labels_for([dev, heldout])
    output = tmp_path / "control-fail"

    def labels_loader():
        assert (output / "decisions_dev.jsonl").is_file()
        return labels

    report = run_control(
        [dev, heldout], revocations, labels_loader, config, output,
        prepared_info={"mode": "synthetic"})
    assert report["gating"]["dev"]["passed"] is False
    assert report["gating"]["heldout"]["status"] == "not_run"
    assert report["gating"]["heldout"]["decisions"] == 0
    assert report["gating"]["heldout"]["metrics"] is None
    assert report["gating"]["stopped_before_heldout"] is True
    assert report["outcome"]["status"] == "stopped_at_dev"
    assert set(report["split_metrics"]) == {"dev"}
    assert not (output / "decisions_heldout.jsonl").exists()
    assert not (output / "decisions_train.jsonl").exists()
    assert (output / "decisions_dev.jsonl").is_file()
    assert report["raw_outputs"]["verified_unchanged"] is True
    dev_overall = report["split_metrics"]["dev"]["overall"]
    assert dev_overall["fact_selection_identity"] < DEV_FACT_SELECTION_IDENTITY_MIN
    assert dev_overall["unknown_selected_n"] == 0
    assert dev_overall["revoked_selected_n"] == 0


# ---------------------------------------------------------------------------
# Real prepared04 replay (optional ignored artifact)
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.skipif(
    not all((PREPARED04 / name).is_file() for name in
            ("episodes.json", "revocations.json", "labels.json", "manifest.json")),
    reason=("optional ignored prepared04 replay; provision "
            "data/memory_causal_fresh_20261026/prepared04 first"))
def test_prepared04_replay_freezes_dev_before_labels_and_passes(tmp_path):
    config = _config()
    selection = load_selection_inputs(PREPARED04, config)
    episodes = selection["episodes"]
    revocations = selection["revocations"]
    assert len(episodes) == 12
    assert expected_decision_rows(episodes) == 456

    dev = build_split_decisions(episodes, revocations, config, "dev")
    assert len(dev) == 152
    assert decision_key_register(dev)["unique"] is True
    assert all("layer" not in row for row in dev)
    _assert_no_label_fields(dev)
    digest_before = decision_digest(dev)
    frozen = freeze_decisions(dev, split="dev")

    labels = load_scoring_labels(PREPARED04, config)["labels"]
    scored = score_decisions(frozen, labels)
    dev_metrics = scored["splits"]["dev"]
    gate = evaluate_dev_gate(dev_metrics, config["endpoint"])
    assert gate["passed"] is True
    overall = dev_metrics["overall"]
    assert overall["fact_n"] == 128
    assert overall["fact_selection_identity_n"] == 124
    assert overall["fact_selection_identity"] == 124 / 128
    assert overall["unknown_selected_n"] == 0
    assert overall["revoked_selected_n"] == 0
    assert overall["correction"]["n"] == 16
    assert overall["correction"]["identity_n"] == 16
    assert overall["unaffected"]["n"] == 40
    assert overall["unaffected"]["identity_n"] == 40
    assert overall["scope_violation_n"] == 0
    assert decision_digest(dev) == digest_before

    report = run_prepared_control(
        config, prepared=PREPARED04, output=tmp_path / "replay01",
        config_path=REAL_CONFIG)
    assert report["gating"]["dev"]["passed"] is True
    assert report["gating"]["heldout"]["status"] == "evaluated"
    assert report["counts"]["decisions"] == 456
    assert report["counts"]["by_split"] == {
        "train": {"decisions": 152, "episodes": 4, "states": 8, "worlds": 4},
        "dev": {"decisions": 152, "episodes": 4, "states": 8, "worlds": 4},
        "heldout": {"decisions": 152, "episodes": 4, "states": 8, "worlds": 4},
    }
    assert report["raw_outputs"]["verified_unchanged"] is True
    assert (tmp_path / "replay01/decisions_dev.jsonl").is_file()
    assert (tmp_path / "replay01/decisions_train.jsonl").is_file()
    assert (tmp_path / "replay01/decisions_heldout.jsonl").is_file()
    assert report["split_metrics"]["heldout"]["overall"][
        "fact_selection_identity"] == 124 / 128
    assert report["classification"]["native"] is False
    assert report["classification"]["calibrated"] is False
    assert report["outcome"]["status"] == "dev_passed_heldout_evaluated"

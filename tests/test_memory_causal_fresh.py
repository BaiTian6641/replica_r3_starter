"""CPU contracts for the fresh disjoint causal-selector protocol.

These tests use the real frozen writer on a random tiny adapter: real fact
binding, real shared-state reduction, real save/reload and real ranking.  They
establish fixture disjointness/leakage rules, exactly-once writes, live
revocation and stale blocking, checkpoint representation binding, genuinely new
post-restart queries, same-query parity and cached-history cost accounting.
They never run native MPS, never train and never claim a learned selector.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from architecture_helpers import tiny_adapter
from conftest import require_local_artifacts
from replica_lab.memory_causal_fresh import (
    CAUSAL, CONFIG_FORMAT, DISABLED, FIRST_PROCESS, FULL, ORACLE,
    PROTOCOL_FORMAT, RESTART, REVOCATION, SHUFFLED, VARIANT_NAMES,
    DuplicateWrite, QueryLocalRefresh, WriteLedger, _control_record,
    generate_fresh_fixture, load_fresh_checkpoint, prepare_fresh,
    record_version, run_exactly_once_control, run_memory_causal_fresh,
    verify_fresh_prepared,
)
from test_memory_diagnosis import SPEC, Tokenizer

ROOT = Path(__file__).resolve().parents[1]
REAL_CONFIG = ROOT / "configs/memory_causal_fresh_v2.json"


def _test_config(**overrides):
    config = json.loads(REAL_CONFIG.read_text())
    config.update({
        "seed": 4242,
        "base_generator": {
            "families_per_split": {"train": 1, "dev": 1, "heldout": 1},
            "complexities": ["correction", "interference"],
            "variants": 2,
        },
        "layers": [0, 2],
        "engine": {
            "heads": SPEC.heads, "key_size": SPEC.key_size,
            "value_size": SPEC.value_size, "update_rule": SPEC.update_rule,
            "depth_mode": SPEC.depth_mode,
            "depth_writers": SPEC.depth_writers,
            "reducer": "shared_mean",
        },
        "thresholds": {
            "match_cosine_floor": -2.0, "match_margin_floor": -2.0,
            "address_cosine_floor": -2.0, "address_margin_floor": -2.0,
        },
        "acceptance": deepcopy(config["acceptance"]),
        "limits": {
            "max_wall_seconds_per_process": 300,
            "max_observation_encodes": 512,
            "max_query_encodes_per_process": 1024,
            "max_state_rebuilds_per_process": 8192,
            "max_readout_rows_per_process": 40000,
            "max_checkpoint_bytes": 67108864,
            "max_duplicate_observe_commits": 0,
            "max_durable_refresh_commits": 0,
            "max_query_local_refresh_applied": 20000,
            "max_record_versions_per_record": 1,
        },
    })
    config["acceptance"]["bootstrap"] = {
        "seed": 17, "resamples": 16, "confidence": 0.90}
    config["acceptance"]["decisive_cohort"] = "restart_heldout"
    config.update(overrides)
    return config


def _protocol(config):
    protocol = deepcopy(config)
    protocol["format"] = PROTOCOL_FORMAT
    protocol["protocol_sha256"] = "a" * 64
    return protocol


def _fixture_hashes():
    return {
        "episodes_sha256": "1" * 64,
        "labels_sha256": "2" * 64,
        "revocations_sha256": "3" * 64,
        "inventory_sha256": "4" * 64,
    }


def _source_pins():
    return {
        "model_path": "checkpoints/minicpm5_source",
        "model_lock_sha256": "5" * 64,
        "required_revision": "6" * 64,
        "tokenizer_sha256": "7" * 64,
        "config_sha256": "8" * 64,
    }


def _receipt():
    writer = json.loads(REAL_CONFIG.read_text())["writer_extension"]
    return {"weights_sha256": writer["weights_sha256"],
            "manifest_digest": writer["manifest_digest"]}


def _implementation():
    return {"src/replica_lab/memory_causal_fresh.py": "c" * 64}


def _run_pair(tmp_path, *, config=None, protocol=None):
    config = config or _test_config()
    protocol = protocol or _protocol(config)
    episodes, labels, revocations, audit = generate_fresh_fixture(config)
    kwargs = dict(
        episodes=episodes, labels_by_episode=labels,
        revocations=revocations, writer_extension_receipt=_receipt(),
        source_pins=_source_pins(), implementation_sha256=_implementation(),
        fixture_hashes=_fixture_hashes(), device="cpu")
    first = run_memory_causal_fresh(
        tiny_adapter(fast_memory=SPEC).eval(), Tokenizer(), protocol,
        output_dir=tmp_path / "first", **kwargs)
    resumed = run_memory_causal_fresh(
        tiny_adapter(fast_memory=SPEC).eval(), Tokenizer(), protocol,
        output_dir=tmp_path / "resumed",
        resume_checkpoint=tmp_path / "first/policy_checkpoint.pt",
        baseline_report=first, **kwargs)
    return first, resumed, episodes, labels, revocations, audit, protocol


def _rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def test_fresh_fixture_is_deterministic_split_role_and_leak_free():
    config = _test_config()
    first = generate_fresh_fixture(config)
    again = generate_fresh_fixture(config)
    assert first[:3] == again[:3]
    episodes, labels, revocations, audit = first
    assert audit["episodes"] == 12 and audit["states"] == 24
    assert audit["splits"] == {"train": 4, "dev": 4, "heldout": 4}
    assert audit["paraphrase_texts"] > 0
    assert audit["distractor_texts"] > 0
    assert audit["unknown_texts"] > 0
    assert audit["revocation_texts"] == 12
    assert audit["restart_texts_disjoint"] is True
    assert audit["label_free_inputs"] is True
    assert audit["answer_leakage"] is False
    for episode in episodes:
        assert episode["split"] in {"train", "dev", "heldout"}
        first_texts = {query["text"] for query in episode["queries"]}
        restart_texts = {query["text"] for query in episode["restart_queries"]}
        assert first_texts and restart_texts
        assert not (first_texts & restart_texts)
        values = {record["value"]
                  for group in ("facts", "distractors", "corrections")
                  for record in episode[group]}
        addresses = {(record["entity"], record["attribute"])
                     for group in ("facts", "distractors", "corrections")
                     for record in episode[group]}
        for query in (episode["queries"] + episode["restart_queries"]
                      + episode["revocation_queries"]):
            assert "val-" not in query["text"]
            assert not any(value in query["text"] for value in values)
        unknown = next(query for query in episode["restart_queries"]
                       if query["cohort"] == "unknown")
        assert (unknown["entity"], unknown["attribute"]) not in addresses
        revocations_query = episode["revocation_queries"][0]
        assert revocations_query["text"] == next(
            query["text"] for query in episode["queries"]
            if query["query_id"] == "p-distractor")
        assert revocations[episode["episode_id"]]["record_id"] \
            == episode["distractors"][0]["record_id"]
        for state_labels in labels[episode["episode_id"]].values():
            assert state_labels["evaluation_only"] is True
            assert state_labels["unknown_query_ids"]
            assert len([query_id for query_id, target
                        in state_labels["targets"].items()
                        if target["kind"] == "revoked"]) == 1
        corrected = labels[episode["episode_id"]]["corrected"]
        assert corrected["corrected_query_ids"]
        assert corrected["unaffected_query_ids"]
        assert corrected["revoked_entry_ids"]


def test_durable_versions_and_query_local_refresh_are_separate():
    outcome = run_exactly_once_control()
    assert outcome["passed"] is True
    assert outcome["counts_before"] == outcome["counts_after"] == 2
    assert outcome["durable_retry"]["idempotent"] is True
    assert outcome["duplicate_observe_rejected"] is True
    assert outcome["restart_replay_rejected"] is True
    assert outcome["version_update_committed"] is True
    assert outcome["versions_after_update"] == 2
    assert outcome["refresh_applied"] is True
    assert outcome["refresh_retry_status"] == "applied"
    assert outcome["refresh_retry_idempotent"] is True
    assert outcome["refresh_non_live_rejected"] is True
    assert outcome["refresh_non_live_reason"] == "record_not_live"
    assert outcome["refresh_mismatch_rejected"] is True
    assert outcome["refresh_mismatch_reason"] == "durable_version_mismatch"
    assert outcome["refresh_per_query_limit_rejected"] is True
    assert outcome["refresh_durable_unchanged"] is True
    assert outcome["refresh_commits"] == 0
    assert outcome["refresh_audit"]["persistent_memory_writes"] == 0
    assert outcome["refresh_audit"]["optimizer_updates"] == 0

    ledger = WriteLedger()
    record = _control_record("rec", value="one")
    version = record_version(record)
    ledger.commit(attempt_id="a", episode_id="ep", record_id="rec",
                  record_version=version, state="initial",
                  source="observation")
    with pytest.raises(ValueError, match="stable attempt ID"):
        ledger.commit(attempt_id="", episode_id="ep", record_id="rec",
                      record_version=version, state="initial",
                      source="observation")
    with pytest.raises(ValueError, match="version identity"):
        ledger.commit(attempt_id="b", episode_id="ep", record_id="rec",
                      record_version="short", state="initial",
                      source="observation")
    with pytest.raises(ValueError, match="only observed records"):
        ledger.commit(attempt_id="c", episode_id="ep", record_id="rec",
                      record_version=version, state="initial",
                      source="accepted_refresh")
    retry = ledger.commit(attempt_id="a", episode_id="ep", record_id="rec",
                          record_version=version, state="corrected",
                          source="observation")
    assert retry["status"] == "retry"
    assert ledger.counts() == {"ep:rec": 1}
    updated = _control_record("rec", value="two")
    ledger.commit(attempt_id="d", episode_id="ep", record_id="rec",
                  record_version=record_version(updated), state="corrected",
                  source="observation")
    assert ledger.counts() == {"ep:rec": 2}
    restored = WriteLedger.from_dict(json.loads(json.dumps(ledger.to_dict())))
    with pytest.raises(DuplicateWrite):
        restored.commit(attempt_id="e", episode_id="ep", record_id="rec",
                        record_version=version, state="initial",
                        source="observation")
    assert restored.counts() == {"ep:rec": 2}
    assert restored.durable_snapshot() == ledger.durable_snapshot()


def test_refresh_requires_live_matching_version_and_cannot_mutate_durable():
    record = _control_record("rec", value="value")
    ledger = WriteLedger()
    ledger.commit(attempt_id="obs", episode_id="ep", record_id="rec",
                  record_version=record_version(record), state="initial",
                  source="observation")
    before = ledger.durable_snapshot()
    refresh = QueryLocalRefresh({
        "enabled": True, "owner": "query_local_refresh",
        "persistence": "query_local_only",
        "requires_durable_version": True,
        "per_query": {"max_applied_per_query": 1},
    })
    refresh.bind_durable(ledger)
    for _ in range(2):
        outcome = refresh.attempt(
            attempt_id="stable", ledger=ledger, episode_id="ep",
            record_id="rec", record_version=record_version(record),
            live_ids={"rec"}, query_id="q", layer=0, state="initial")
        assert outcome["status"] == "applied"
        assert outcome["durable_mutation"] is False
    assert ledger.durable_snapshot() == before
    assert refresh.summary()["status_counts"] == {"applied": 1}
    assert refresh.summary()["durable_snapshot_sha256"] == before


def test_fresh_run_restart_revocation_and_cost_contracts(tmp_path):
    first, resumed, episodes, labels, revocations, audit, protocol = \
        _run_pair(tmp_path)
    planned = {
        "observations": 144, "first_queries": 216,
        "base_rebuilds": 48, "restart_queries": 216,
    }
    assert first["phase"] == FIRST_PROCESS
    assert first["decision"] == "inconclusive"
    assert first["resumed"] is False
    assert first["encoded"]["observations"] == planned["observations"]
    assert first["encoded"]["queries"] == planned["first_queries"]
    assert first["encoded"]["old_observations_reencoded"] == 0
    assert first["costs"]["counts"]["observation_encoding"] \
        == planned["observations"]
    assert first["costs"]["counts"]["state_rebuild"] > planned["base_rebuilds"]
    assert first["costs"]["seconds_by_operation"]["observation_encoding"] > 0
    assert first["checkpoint"]["bytes"] > 0
    assert first["limits"]["passed"] is True
    assert first["exactly_once"]["max_record_versions_per_record"] == 1
    assert first["exactly_once"]["durable_refresh_commits"] == 0
    assert first["exactly_once"]["query_local_refresh_applied"] > 0
    assert first["exactly_once"]["query_local_refresh_durable_unchanged"]
    assert first["exactly_once"]["passed"] is True
    assert first["revocation"] is None
    assert first["generation"]["status"] == "not_measured"
    assert first["classification"]["learned_selector"] is False
    assert first["classification"]["useful_memory_claim"] is False
    assert first["classification"]["exact_key_value"] != "upper bound"
    assert first["gates"]["mechanical"]["restart_without_old_observation_encode"]
    assert first["gates"]["mechanical"]["stale_blocks_acceptance"]
    assert first["gates"]["mechanical"]["causal_refresh_admitted"]
    assert first["metrics"]["first_process_all"]["metrics"] is not None
    assert first["inventory"]["declared_query_rows"] == 228
    assert first["inventory"]["inventory_case_rows"] == 456
    assert first["inventory"]["arm_expanded_metric_rows"] == 17520
    assert first["inventory"]["process_ranking_rows"] == 8400
    assert first["inventory"]["process_phases"] == [FIRST_PROCESS]

    assert resumed["phase"] == RESTART
    assert resumed["resumed"] is True
    assert resumed["encoded"]["observations"] == 0
    assert resumed["encoded"]["old_observations_reencoded"] == 0
    assert resumed["encoded"]["restart_new_queries"] == planned["restart_queries"]
    assert resumed["costs"]["counts"].get("observation_encoding", 0) == 0
    assert resumed["costs"]["counts"]["query_encoding"] \
        == planned["restart_queries"]
    assert resumed["costs"]["counts"]["state_rebuild"] > planned["base_rebuilds"]
    assert resumed["costs"]["counts"]["checkpoint_loads"] == 1
    assert resumed["parity"]["same_query_parity"] is True
    assert resumed["exactly_once"]["restart_replay_rejected"] is True
    assert resumed["exactly_once"]["query_local_refresh_applied"] > 0
    assert resumed["exactly_once"]["durable_refresh_commits"] == 0
    assert resumed["revocation"]["all_revoked_policy_acceptance_zero"] is True
    assert resumed["revocation"]["any_live_reference_accepted"] is True
    assert resumed["blocked"]["revoked_policy_accepted"] == 0
    assert resumed["blocked"]["stale_accepted"] == 0
    assert resumed["blocked"]["revoked_target_other_live"] > 0
    assert resumed["decision"] in {"rejected", "invalid", "inconclusive"}
    assert resumed["decision"] != "accepted"
    assert resumed["metrics"]["restart_all"]["metrics"] is not None
    assert resumed["limits"]["passed"] is True
    assert (tmp_path / "resumed/policy_checkpoint_restart.pt").is_file()
    assert (tmp_path / "resumed/ledger.json").is_file()
    assert (tmp_path / "resumed/refresh_audit.json").is_file()

    first_rankings = _rows(tmp_path / "first/rankings.jsonl")
    resumed_rankings = _rows(tmp_path / "resumed/rankings.jsonl")
    assert all(not row["accepted"] for row in first_rankings
               if row["stale_blocked"] or row["revoked_blocked"])
    assert all(not row["accepted"] for row in resumed_rankings
               if row["stale_blocked"] or row["revoked_blocked"])
    assert {row["phase"] for row in resumed_rankings} == {
        FIRST_PROCESS, RESTART, REVOCATION}
    assert {row["variant"] for row in resumed_rankings} == set(VARIANT_NAMES)
    decisions = _rows(tmp_path / "first/decisions.jsonl")
    assert decisions
    assert all(row["used_future_queries"] is False for row in decisions)
    assert all(row["reread_source"] is False for row in decisions)
    assert all(row["reencoded_old_observations"] is False for row in decisions)
    assert any(row["variant"] == ORACLE and row["used_gold_target"] is True
               for row in decisions)
    assert all(row["used_gold_target"] is False for row in decisions
               if row["variant"] in (CAUSAL, SHUFFLED, FULL, DISABLED))
    assert any(row["refresh_applied"] for row in decisions)
    assert all(row["refresh_persistent"] is False for row in decisions)
    assert any(row["weights_source"] == "query_local_refresh"
               for row in decisions)
    assert any(row["weights_source"] == "query_local_refresh_shuffled"
               for row in decisions)
    readouts = _rows(tmp_path / "first/readouts.jsonl")
    for row in readouts:
        assert "expected" not in row and "expected_answer" not in row


def test_admitted_refresh_changes_state_oracle_recovers_and_shuffle_perturbs(
        tmp_path):
    first, resumed, episodes, labels, revocations, audit, protocol = \
        _run_pair(tmp_path)
    controls = first["refresh_controls"]
    assert controls["applied_refreshes"] > 0
    assert controls["causal_refreshes"] > 0
    assert controls["shuffled_refreshes"] > 0
    assert controls["oracle_refreshes"] > 0
    assert controls["refresh_changed_state_rows"] > 0
    assert controls["admitted_refresh_changes_state"] is True
    assert controls["shuffled_refresh_perturbs_values"] is True
    assert controls["shuffled_perturbed_pairs"] > 0
    assert controls["oracle_recovery_rate"] is not None
    assert controls["oracle_is_diagnostic_not_upper_bound"] is True
    fixture_control = controls["positive_control_fixture"]
    assert fixture_control["passed"] is True
    assert fixture_control["oracle_recovers_known_fixture"] is True
    assert fixture_control["oracle_target_cosine"] > 0.99
    assert fixture_control["admitted_refresh_changes_state"] is True
    assert fixture_control["admitted_refresh_changes_readout"] is True
    assert fixture_control["shuffled_refresh_changes_state"] is True
    assert fixture_control["shuffled_refresh_perturbs_values"] is True
    assert fixture_control["durable_writes"] == 0
    assert fixture_control["optimizer_updates"] == 0

    decisions = _rows(tmp_path / "first/decisions.jsonl")
    applied = [row for row in decisions if row["refresh_applied"]]
    assert applied
    assert all(row["state_sha256"] != row["base_state_sha256"]
               or row["refresh_changed_state"] is False
               for row in applied)
    changed = [row for row in applied if row["refresh_changed_state"]]
    assert changed
    assert all(row["variant"] in (CAUSAL, SHUFFLED, ORACLE)
               for row in applied)
    assert all(row["refresh_attempt"]["durable_mutation"] is False
               for row in applied)

    readouts = _rows(tmp_path / "first/readouts.jsonl")
    index = {(row["variant"], row["episode_id"], row["state"],
              row["case_id"], row["layer"], row["arm"]): row
             for row in readouts}
    comparisons = 0
    for key, row in index.items():
        if row["variant"] != CAUSAL or row["arm"] != "query_key_value":
            continue
        if not row["refresh_applied"]:
            continue
        full = index.get((FULL, *key[1:]))
        if full is None:
            continue
        comparisons += 1
        assert row["readout_sha256"] != full["readout_sha256"]
    assert comparisons > 0

    oracle = first["metrics"]["first_process_all"]["metrics"]["variants"][ORACLE]["quality"]
    assert oracle["refresh_applied_rows"] > 0
    assert oracle["raw_top1_all"] >= 1
    assert first["metrics"]["first_process_all"]["metrics"]["variants"].get(FULL) is not None
    assert first["metrics"]["first_process_all"]["metrics"]["variants"][CAUSAL]["quality"][
        "refresh_applied_rows"] > 0
    assert first["metrics"]["first_process_all"]["metrics"]["variants"][FULL]["quality"][
        "refresh_applied_rows"] == 0
    diagnostics = first["metrics"]["first_process_all"]["metrics"]["diagnostics"]
    assert diagnostics["query_local_refresh_evidence"] is True
    assert diagnostics["candidate_equals_full_quality"] is False
    assert diagnostics["fallback_is_explicit"] is True

    resumed_controls = resumed["refresh_controls"]
    assert resumed_controls["admitted_refresh_changes_state"] is True
    assert resumed_controls["shuffled_refresh_perturbs_values"] is True


def test_runtime_revocation_and_evaluator_kind_are_scored_separately(tmp_path):
    first, resumed, episodes, labels, revocations, audit, protocol = \
        _run_pair(tmp_path)
    rankings = _rows(tmp_path / "resumed/rankings.jsonl")
    revoked = [row for row in rankings if row["phase"] == REVOCATION]
    assert revoked
    other_live = [row for row in revoked
                  if row["false_recall_kind"] == "revoked_target_other_live"]
    assert other_live
    # The sealed label kind did not force abstention; these are wrong retrievals.
    assert all(row["accepted"] is True for row in other_live)
    assert all(row["policy_blocked"] is None for row in other_live)
    assert all(row["revoked_query_kind"] is True for row in other_live)
    assert all(row["target_entry_id"] is None for row in other_live)
    assert resumed["blocked"]["revoked_target_other_live"] > 0
    assert resumed["blocked"]["revoked_policy_accepted"] == 0
    assert all(not row["accepted"] for row in revoked
               if row["policy_blocked"] is not None)

    candidate = protocol["acceptance"]["candidate_variant"]
    metrics = resumed["metrics"]["restart_all"]["metrics"]["variants"]
    revoked_block = metrics[candidate]["revoked"]
    assert revoked_block["n"] > 0
    assert revoked_block["revoked_target_other_live"] > 0
    assert revoked_block["revoked_false_recall"] > 0
    assert revoked_block["revoked_abstention_rate"] is not None
    unknown_block = metrics[candidate]["unknown"]
    assert unknown_block["n"] > 0
    assert unknown_block["unknown_false_recall"] > 0

    # Stale corrections are runtime-blocked, not evaluator-labelled abstentions.
    stale = [row for row in rankings if row["stale_blocked"]]
    assert stale
    assert all(row["policy_blocked"] == "stale" for row in stale)
    assert all(not row["accepted"] for row in stale)
    assert resumed["blocked"]["stale_accepted"] == 0

    # Restart/rebuild does not change durable versions or revoke replay safety.
    assert resumed["exactly_once"]["durable_identity"] == (
        "episode_id/record_id/record_version")
    assert resumed["exactly_once"]["max_record_versions_per_record"] == 1
    assert resumed["exactly_once"]["restart_replay_rejected"] is True


def test_checkpoint_bindings_reject_every_category(tmp_path):
    config = _test_config()
    protocol = _protocol(config)
    episodes, labels, revocations, _ = generate_fresh_fixture(config)
    first = run_memory_causal_fresh(
        tiny_adapter(fast_memory=SPEC).eval(), Tokenizer(), protocol,
        output_dir=tmp_path / "first", episodes=episodes,
        labels_by_episode=labels, revocations=revocations,
        writer_extension_receipt=_receipt(),
        source_pins=_source_pins(),
        implementation_sha256=_implementation(),
        fixture_hashes=_fixture_hashes(), device="cpu")
    checkpoint = tmp_path / "first/policy_checkpoint.pt"
    base = json.loads(json.dumps(first["bindings"]))
    assert load_fresh_checkpoint(checkpoint, expected=base)["format"]

    def mutate(path, value):
        def apply(bindings):
            target = bindings
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = value
        return apply

    cases = [
        (["protocol_sha256"], "d" * 64, "protocol"),
        (["fixture", "episodes_sha256"], "e" * 64, "fixture"),
        (["source", "model_path"], "other-source", "source"),
        (["writer", "weights_sha256"], "f" * 64, "writer"),
        (["representation", "sha256"], "0" * 64, "representation"),
        (["implementation", "src/replica_lab/memory_causal_fresh.py"],
         "1" * 64, "implementation"),
        (["engine", "heads"], SPEC.heads + 1, "engine"),
    ]
    for path, value, label in cases:
        expected = json.loads(json.dumps(base))
        mutate(path, value)(expected)
        with pytest.raises(ValueError, match=label):
            load_fresh_checkpoint(checkpoint, expected=expected)


def test_representation_mismatch_is_rejected_before_encoding(tmp_path):
    config = _test_config()
    protocol = _protocol(config)
    protocol["engine"]["heads"] = SPEC.heads + 1
    episodes, labels, revocations, _ = generate_fresh_fixture(config)
    with pytest.raises(ValueError, match="heads differs"):
        run_memory_causal_fresh(
            tiny_adapter(fast_memory=SPEC).eval(), Tokenizer(), protocol,
            output_dir=tmp_path / "mismatch", episodes=episodes,
            labels_by_episode=labels, revocations=revocations,
            writer_extension_receipt=_receipt(),
            source_pins=_source_pins(),
            implementation_sha256=_implementation(),
            fixture_hashes=_fixture_hashes(), device="cpu")
    assert not (tmp_path / "mismatch/ledger.json").exists()


@require_local_artifacts(
    "checkpoints/minicpm5_source",
    "configs/minicpm5_sealed_lock.local.json")
def test_prepare_verify_and_pin_tamper(tmp_path):
    config = _test_config()
    prepared = tmp_path / "prepared"
    manifest = prepare_fresh(prepared, config=config)
    assert manifest["native_computation"] is False
    assert manifest["model_loaded"] is False
    assert manifest["episodes"] == 12
    receipt = verify_fresh_prepared(prepared)
    assert receipt["status"] == "verified"
    assert receipt["fixture_audit"]["answer_leakage"] is False

    with pytest.raises(ValueError, match="fresh or empty"):
        prepare_fresh(prepared, config=config)

    labels_path = prepared / "labels.json"
    labels_path.write_text("{}\n")
    with pytest.raises(ValueError, match="bytes differ from the protocol pin"):
        verify_fresh_prepared(prepared)


def test_native_script_requires_capacity_guard():
    spec = importlib.util.spec_from_file_location(
        "fresh_native", ROOT / "scripts/run_memory_causal_fresh.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    import os
    for key in ("REPLICA_CAPACITY_GUARD", "PYTORCH_ENABLE_MPS_FALLBACK",
                "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        os.environ.pop(key, None)
    with pytest.raises(RuntimeError, match="capacity guard"):
        module.execute(SimpleNamespace())


def test_protocol_predeclares_limits_seeds_controls_and_policy():
    config = json.loads(REAL_CONFIG.read_text())
    assert config["format"] == CONFIG_FORMAT
    assert config["seed"] == 20261102
    assert config["seed"] not in (20261026, 20261027)
    assert config["acceptance"]["metric_version"] == "memory.acceptance.v3"
    assert config["acceptance"]["primary_quality_arm"] == "query_key_value"
    assert config["acceptance"]["diagnostic_arms"] == [
        "exact_key_value", "query_key_address", "context_removed"]
    assert config["acceptance"]["candidate_variant"] == CAUSAL
    assert config["acceptance"]["oracle_variant"] == ORACLE
    assert config["acceptance"]["decisive_cohort"] == "restart_heldout"
    assert config["acceptance"]["bootstrap"]["seed"] == 17
    assert config["budget"]["optimizer_updates"] == 0
    assert config["budget"]["native_training_jobs"] == 0
    assert config["budget"]["native_diagnostic_jobs"] == 2
    assert config["limits"]["max_durable_refresh_commits"] == 0
    assert config["limits"]["max_query_local_refresh_applied"] > 0
    assert config["limits"]["max_record_versions_per_record"] == 1
    assert config["limits"]["max_duplicate_observe_commits"] == 0
    assert config["durable_ingestion"]["identity"] == (
        "episode_id/record_id/record_version")
    assert config["durable_ingestion"]["refresh_source"] == "excluded"
    assert config["query_local_refresh"]["enabled"] is True
    assert config["query_local_refresh"]["persistence"] == "query_local_only"
    assert config["query_local_refresh"]["persistent_memory_writes"] == 0
    assert config["query_local_refresh"]["optimizer_updates"] == 0
    assert config["query_local_refresh"]["decay_updates"] == 0
    assert config["query_local_refresh"]["per_query"][
        "max_applied_per_query"] >= 1
    assert config["query_local_refresh"]["positive_control"][
        "oracle_recovery_raw_top1_min"] > 0
    assert config["restart"]["old_observation_encodes_allowed"] == 0
    assert config["revocation"]["scope"] == "view_only"
    assert config["revocation"]["runtime_metadata_only"] is True
    assert config["revocation"]["evaluator_label_kind_forces_abstention"] is False
    assert config["revocation"]["cached_history_retained"] is True
    assert config["checkpoint_policy"]["format"] == (
        "replica.memory_causal_fresh.checkpoint.v3")
    assert config["checkpoint_policy"]["refresh_attempts_serialized"] is False
    assert config["checkpoint_policy"]["retention"] == "retain_both"
    assert config["checkpoint_policy"]["overwrite"] is False
    assert config["raw_output_policy"]["immutable_first_write"] is True
    assert config["raw_output_policy"]["no_generation_claims"] is True
    assert config["resource_policy"]["cpu_fallback"] is False
    assert config["numeric_policy"]["torch_use_deterministic_algorithms"] is True
    assert config["numeric_policy"]["mps_fast_math"] == "disabled"
    assert config["representation"]["name"] == "fact_binding_key_value.v1"
    assert config["inventory_units"]["must_reconcile"] is True
    assert all((ROOT / name).is_file() for name in config["implementation_paths"])
    assert ("abort_on_checkpoint_binding_mismatch"
            in config["stop_conditions"])
    assert ("abort_on_durable_refresh_commit"
            in config["stop_conditions"])
    assert ("abort_on_missing_coverage" in config["stop_conditions"])
    assert ("never_promote_quality_without_oracle_bound"
            in config["stop_conditions"])

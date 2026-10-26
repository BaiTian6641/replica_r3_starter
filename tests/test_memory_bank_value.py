"""Contracts for the selected-record literal-value and MiniCPM readout.

Synthetic tests cover literal value exactness, correction versus unaffected
facts, unknown/revoked abstention, decision-before-label ordering, dev literal
and model stop behaviour, prompt exclusion of evaluation labels, the EOS stop
set, immutable generation rows, and the frozen endpoint constants.  One
optional integration test reads the real native02 checkpoint and prepared04
fixture; it loads no model, writes no checkpoint and runs no native job.
"""
from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
import json
from pathlib import Path

import pytest

from replica_lab.memory_address_bank import decision_digest
from replica_lab.memory_bank_value import (
    ARTIFACT_CAP_BYTES,
    DEV_LITERAL_FACT_VALUE_MIN,
    GENERATIONS_FORMAT,
    LITERAL_FORMAT,
    MAX_DECISIONS_PER_SPLIT,
    MAX_NEW_TOKENS,
    PROMPT_FORMAT,
    REPORT_FORMAT,
    STOP_TOKEN_IDS,
    annotate_decision_values,
    build_prompt,
    build_value_split_decisions,
    compressed_reference,
    decision_status,
    evaluate_dev_literal_gate,
    evaluate_dev_model_gate,
    generate_row,
    generation_row,
    normalize_answer,
    query_text_register,
    run_bank_value,
    score_generations,
    score_literal_values,
    validate_config,
    validate_generation_rows,
    verify_baseline,
    write_generation_rows,
)
from replica_lab.memory_matched import CORRECTION_DAY, INITIAL_DAY

ROOT = Path(__file__).resolve().parents[1]
REAL_CONFIG = ROOT / "configs/memory_bank_value_v1.json"
PREPARED04 = ROOT / "data/memory_causal_fresh_20261026/prepared04"
NATIVE02 = ROOT / "runs/memory_causal_fresh_20261026/native02"


def _config():
    return json.loads(REAL_CONFIG.read_text())


def _fact(record_id, entity, attribute, value, *, world_id, observed_at=INITIAL_DAY):
    return {
        "record_id": record_id, "entity": entity, "attribute": attribute,
        "value": value, "observed_at": observed_at,
        "source_world_id": world_id,
    }


def _correction(record_id, entity, attribute, value, supersedes):
    return {
        "record_id": record_id, "entity": entity, "attribute": attribute,
        "value": value, "observed_at": CORRECTION_DAY,
        "supersedes": supersedes,
    }


def _distractor(record_id, entity, attribute, value, *, world_id):
    return {
        "record_id": record_id, "entity": entity, "attribute": attribute,
        "value": value, "observed_at": INITIAL_DAY,
        "origin_record_id": record_id, "origin_world_id": world_id,
    }


def _query(query_id, entity, attribute, phase, *, cohort="paraphrase", form=0):
    return {
        "query_id": query_id, "entity": entity, "attribute": attribute,
        "phase": phase, "cohort": cohort, "form": form,
        "text": f"Question about {entity} and {attribute}?",
    }


def _episode(prefix, split):
    world = f"world-{prefix}"
    facts = [
        _fact(f"rec-{prefix}-a", f"ent-{prefix}-a", f"attr-{prefix}-a",
              f"val-{prefix}-a", world_id=world),
        _fact(f"rec-{prefix}-b", f"ent-{prefix}-b", f"attr-{prefix}-b",
              f"val-{prefix}-b", world_id=world),
    ]
    correction = _correction(
        f"rec-{prefix}-a2", f"ent-{prefix}-a", f"attr-{prefix}-a",
        f"val-{prefix}-a2", f"rec-{prefix}-a")
    distractor = _distractor(
        f"rec-{prefix}-d", f"ent-{prefix}-d", f"attr-{prefix}-d",
        f"val-{prefix}-d", world_id=f"world-{prefix}-other")
    queries = [
        _query("q-a", f"ent-{prefix}-a", f"attr-{prefix}-a",
               "first_process"),
        _query("q-b", f"ent-{prefix}-b", f"attr-{prefix}-b",
               "first_process"),
        _query("q-d", f"ent-{prefix}-d", f"attr-{prefix}-d",
               "first_process", cohort="distractor", form=2),
        _query("q-unknown", f"ent-{prefix}-u", f"attr-{prefix}-u",
               "first_process", cohort="unknown", form=3),
    ]
    restart = [
        _query("r-a", f"ent-{prefix}-a", f"attr-{prefix}-a", "restart",
               form=2),
        _query("r-b", f"ent-{prefix}-b", f"attr-{prefix}-b", "restart",
               form=2),
        _query("r-unknown", f"ent-{prefix}-u2", f"attr-{prefix}-u2",
               "restart", cohort="unknown", form=0),
    ]
    revocation = [
        _query("q-revoked", f"ent-{prefix}-d", f"attr-{prefix}-d",
               "revocation", cohort="revoked", form=2),
    ]
    observations = [
        {"page_id": f"page-{prefix}-initial", "observed_at": INITIAL_DAY,
         "date": "2026-03-01", "lines": ["initial line"]},
        {"page_id": f"page-{prefix}-corrected", "observed_at": CORRECTION_DAY,
         "date": "2026-04-01", "lines": ["corrected line"]},
    ]
    return {
        "episode_id": f"episode-{prefix}", "family_id": f"family-{prefix}",
        "world_id": world, "split": split, "complexity": "correction",
        "variant": 0, "facts": facts, "corrections": [correction],
        "distractors": [distractor], "observations": observations,
        "queries": queries, "restart_queries": restart,
        "revocation_queries": revocation,
    }


def _revocations(episode):
    return {episode["episode_id"]: {
        "record_id": f"rec-{episode['episode_id'].split('-')[-1]}-d",
        "entity": episode["distractors"][0]["entity"],
        "attribute": episode["distractors"][0]["attribute"],
        "applies_at": "post_restart", "scope": "view_only",
        "cached_history_retained": True,
    }}


def _labels_for_episode(episode):
    prefix = episode["episode_id"].split("-")[-1]
    correction = episode["corrections"][0]
    labels = {}
    for state_name, cutoff in (("initial", INITIAL_DAY),
                               ("corrected", CORRECTION_DAY)):
        records = (episode["facts"] + episode["distractors"]
                   + episode["corrections"])
        superseded = {item["supersedes"] for item in episode["corrections"]
                      if item["observed_at"] <= cutoff}
        live = [record for record in records
                if record["observed_at"] <= cutoff
                and record["record_id"] not in superseded]
        by_address = {(record["entity"], record["attribute"]): record
                      for record in live}
        targets, cohort_by_query, phase_by_query = {}, {}, {}
        corrected_ids, unaffected_ids = [], []
        for phase, field in (("first_process", "queries"),
                             ("restart", "restart_queries"),
                             ("revocation", "revocation_queries")):
            for query in episode[field]:
                query_id = query["query_id"]
                cohort_by_query[query_id] = query["cohort"]
                phase_by_query[query_id] = phase
                if phase == "revocation":
                    targets[query_id] = {
                        "expected": "revoked", "kind": "revoked",
                        "target_entry_id": None, "target_role": "revoked",
                        "target_source_family": "source",
                        "query_form": query["form"],
                    }
                    continue
                record = by_address.get((query["entity"],
                                         query["attribute"]))
                if record is None:
                    targets[query_id] = {
                        "expected": "unknown", "kind": "unknown",
                        "target_entry_id": None, "target_role": "unknown",
                        "target_source_family": "none",
                        "query_form": query["form"],
                    }
                    continue
                family = ("independent_world"
                          if record.get("origin_world_id",
                                        episode["world_id"])
                          != episode["world_id"] else "source")
                if record["record_id"] == correction["record_id"]:
                    role = "correction_target"
                    corrected_ids.append(query_id)
                elif family == "source" and state_name == "corrected":
                    role = "retained"
                    unaffected_ids.append(query_id)
                else:
                    role = "source"
                if family == "source" and state_name == "initial":
                    unaffected_ids.append(query_id)
                targets[query_id] = {
                    "expected": record["value"], "kind": "fact",
                    "target_entry_id": record["record_id"],
                    "target_role": role,
                    "target_source_family": family,
                    "query_form": query["form"],
                }
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
            "corrected_query_ids": sorted(set(corrected_ids)),
            "unaffected_query_ids": sorted(set(unaffected_ids)),
            "unknown_query_ids": [],
            "evaluation_only": True,
        }
    return labels


def _labels_for(episodes):
    return {episode["episode_id"]: _labels_for_episode(episode)
            for episode in episodes}


def _by_key(decisions):
    return {(row["state"], row["phase"], row["query_id"]): row
            for row in decisions}


def _correct_generator(decisions, query_texts):
    rows = []
    for decision in decisions:
        key = (decision["episode_id"], decision["phase"],
               decision["query_id"])
        prompt = build_prompt(decision, query_texts[key])
        if decision["selected_value"] is not None:
            answer = decision["selected_value"]
        else:
            answer = decision_status(decision)
        rows.append(generation_row(
            decision, query_texts[key],
            prompt_sha256=__import__("hashlib").sha256(
                prompt.encode()).hexdigest(),
            prompt_ids_sha256=("0" * 64),
            prompt_tokens=8,
            generated_ids=[100, 130073] if answer else [130073],
            generated_text=answer, max_new_tokens=MAX_NEW_TOKENS))
    return tuple(rows)


def _wrong_generator(decisions, query_texts):
    rows = []
    for decision in decisions:
        key = (decision["episode_id"], decision["phase"],
               decision["query_id"])
        prompt = build_prompt(decision, query_texts[key])
        rows.append(generation_row(
            decision, query_texts[key],
            prompt_sha256=__import__("hashlib").sha256(
                prompt.encode()).hexdigest(),
            prompt_ids_sha256=("0" * 64),
            prompt_tokens=8, generated_ids=[100, 130073],
            generated_text="not-the-answer",
            max_new_tokens=MAX_NEW_TOKENS))
    return tuple(rows)


def _factory(generator, calls):
    def factory(*, deadline=None, baseline_info=None):
        calls.append("factory")
        return generator
    return factory


# ---------------------------------------------------------------------------
# Literal value contracts
# ---------------------------------------------------------------------------

def test_literal_values_correction_unaffected_unknown_and_revoked():
    config = _config()
    dev = _episode("dev", "dev")
    decisions = build_value_split_decisions(
        [dev], _revocations(dev), config, "dev")
    indexed = _by_key(decisions)
    assert len(decisions) <= MAX_DECISIONS_PER_SPLIT
    assert decision_digest(decisions) == decision_digest(decisions)
    assert all(row["literal_format"] == LITERAL_FORMAT for row in decisions)

    initial_a = indexed[("initial", "first_process", "q-a")]
    corrected_a = indexed[("corrected", "first_process", "q-a")]
    assert initial_a["selected_value"] == "val-dev-a"
    assert corrected_a["selected_value"] == "val-dev-a2"
    assert indexed[("initial", "first_process", "q-b")][
        "selected_value"] == "val-dev-b"
    assert indexed[("initial", "first_process", "q-d")][
        "selected_value"] == "val-dev-d"
    for state in ("initial", "corrected"):
        assert indexed[(state, "first_process", "q-unknown")][
            "selected_value"] is None
        assert indexed[(state, "revocation", "q-revoked")][
            "selected_value"] is None
        assert indexed[(state, "restart", "r-unknown")][
            "selected_value"] is None

    scored = score_literal_values(decisions, _labels_for([dev]))
    overall = scored["splits"]["dev"]["overall"]
    assert overall["fact_value_exact_rate"] == 1.0
    assert overall["fact_value_exact_n"] == overall["fact_n"]
    assert overall["unknown_selected_n"] == 0
    assert overall["revoked_selected_n"] == 0
    assert overall["correction"]["value_exact_rate"] == 1.0
    assert overall["unaffected"]["value_exact_rate"] == 1.0
    assert overall["correction"]["fact_n"] > 0
    assert overall["unaffected"]["fact_n"] > 0


def test_values_come_from_runtime_records_not_labels():
    config = _config()
    dev = _episode("dev", "dev")
    decisions = build_value_split_decisions(
        [dev], _revocations(dev), config, "dev")
    labels = _labels_for([dev])
    changed = deepcopy(labels)
    for state_labels in changed[dev["episode_id"]].values():
        for target in state_labels["targets"].values():
            if target["kind"] == "fact":
                target["expected"] = "val-leaked"
    before = score_literal_values(decisions, labels)["splits"]["dev"]["overall"]
    after = score_literal_values(decisions, changed)["splits"]["dev"]["overall"]
    assert before["fact_value_exact_rate"] == 1.0
    assert after["fact_value_exact_rate"] == 0.0
    annotated = annotate_decision_values([dev], decisions)
    assert [row["selected_value"] for row in annotated] == [
        row["selected_value"] for row in decisions]


def test_literal_gate_is_frozen_and_rejects_selections():
    config = _config()
    dev = _episode("dev", "dev")
    decisions = build_value_split_decisions(
        [dev], _revocations(dev), config, "dev")
    metrics = score_literal_values(
        decisions, _labels_for([dev]))["splits"]["dev"]
    gate = evaluate_dev_literal_gate(metrics, config["endpoint"])
    assert gate["passed"] is True
    assert gate["evaluated_first"] is True

    failed = deepcopy(metrics)
    failed["overall"]["fact_value_exact_rate"] = (
        DEV_LITERAL_FACT_VALUE_MIN - 0.01)
    assert evaluate_dev_literal_gate(
        failed, config["endpoint"])["passed"] is False
    unknown = deepcopy(metrics)
    unknown["overall"]["unknown_selected_n"] = 1
    assert evaluate_dev_literal_gate(
        unknown, config["endpoint"])["passed"] is False
    revoked = deepcopy(metrics)
    revoked["overall"]["revoked_selected_n"] = 1
    assert evaluate_dev_literal_gate(
        revoked, config["endpoint"])["passed"] is False
    tuned = deepcopy(config["endpoint"])
    tuned["dev_literal_fact_value_min"] = 0.8
    with pytest.raises(ValueError, match="may not be tuned"):
        evaluate_dev_literal_gate(metrics, tuned)


# ---------------------------------------------------------------------------
# Prompt and generation contracts
# ---------------------------------------------------------------------------

def test_prompt_contains_only_query_value_or_runtime_status():
    config = _config()
    dev = _episode("dev", "dev")
    decisions = build_value_split_decisions(
        [dev], _revocations(dev), config, "dev")
    texts = query_text_register([dev])
    indexed = _by_key(decisions)

    fact = indexed[("initial", "first_process", "q-a")]
    fact_prompt = build_prompt(fact, texts[
        (dev["episode_id"], "first_process", "q-a")])
    assert "Question about ent-dev-a and attr-dev-a?" in fact_prompt
    assert "Record value: val-dev-a" in fact_prompt
    assert "val-dev-b" not in fact_prompt
    assert "val-dev-d" not in fact_prompt
    assert "expected" not in fact_prompt
    assert "target_entry_id" not in fact_prompt

    unknown = indexed[("initial", "first_process", "q-unknown")]
    unknown_prompt = build_prompt(unknown, texts[
        (dev["episode_id"], "first_process", "q-unknown")])
    assert "Record status: unknown" in unknown_prompt
    assert "val-dev" not in unknown_prompt
    assert decision_status(unknown) == "unknown"

    revoked = indexed[("initial", "revocation", "q-revoked")]
    revoked_prompt = build_prompt(revoked, texts[
        (dev["episode_id"], "revocation", "q-revoked")])
    assert "Record status: revoked" in revoked_prompt
    assert "val-dev" not in revoked_prompt
    assert decision_status(revoked) == "revoked"

    # A decoy label payload never enters the prompt builder.
    labels = _labels_for([dev])
    for target in labels[dev["episode_id"]]["initial"]["targets"].values():
        target["expected"] = "LEAKED-EXPECTED-ANSWER"
    assert "LEAKED-EXPECTED-ANSWER" not in build_prompt(
        fact, texts[(dev["episode_id"], "first_process", "q-a")])


def test_generation_rows_use_stop_set_and_frozen_budget():
    assert STOP_TOKEN_IDS == (1, 130073)
    config = _config()
    dev = _episode("dev", "dev")
    decisions = build_value_split_decisions(
        [dev], _revocations(dev), config, "dev")
    texts = query_text_register([dev])
    fact = decisions[0]
    prompt = build_prompt(fact, texts[(fact["episode_id"], fact["phase"],
                                       fact["query_id"])])
    digest = __import__("hashlib").sha256(prompt.encode()).hexdigest()

    eos_one = generation_row(
        fact, texts[(fact["episode_id"], fact["phase"], fact["query_id"])],
        prompt_sha256=digest, prompt_ids_sha256="0" * 64, prompt_tokens=4,
        generated_ids=[7, 1], generated_text="val", max_new_tokens=8)
    assert eos_one["ended_by_eos"] is True
    assert eos_one["stop_token_ids"] == [1, 130073]
    eos_two = generation_row(
        fact, texts[(fact["episode_id"], fact["phase"], fact["query_id"])],
        prompt_sha256=digest, prompt_ids_sha256="0" * 64, prompt_tokens=4,
        generated_ids=[7, 130073], generated_text="val", max_new_tokens=8)
    assert eos_two["ended_by_eos"] is True
    ordinary = generation_row(
        fact, texts[(fact["episode_id"], fact["phase"], fact["query_id"])],
        prompt_sha256=digest, prompt_ids_sha256="0" * 64, prompt_tokens=4,
        generated_ids=[7, 8], generated_text="val", max_new_tokens=8)
    assert ordinary["ended_by_eos"] is False
    with pytest.raises(ValueError, match="token budget"):
        generation_row(
            fact,
            texts[(fact["episode_id"], fact["phase"], fact["query_id"])],
            prompt_sha256=digest, prompt_ids_sha256="0" * 64,
            prompt_tokens=4, generated_ids=[1] * 9,
            generated_text="val", max_new_tokens=8)
    with pytest.raises(ValueError, match="differs from the stop set"):
        generation_row(
            fact,
            texts[(fact["episode_id"], fact["phase"], fact["query_id"])],
            prompt_sha256=digest, prompt_ids_sha256="0" * 64,
            prompt_tokens=4, generated_ids=[7, 8],
            generated_text="val", max_new_tokens=8, ended_by_eos=True)
    for forbidden in ("expected", "expected_answer", "gold_answer", "label"):
        assert forbidden not in eos_one
    assert eos_one["format"] == GENERATIONS_FORMAT
    assert eos_one["prompt_format"] == PROMPT_FORMAT


def test_generate_row_uses_declared_request_convention():
    import torch

    class FakeTokenizer:
        eos_token_id = 1

        def apply_chat_template(self, messages, tokenize=False,
                                add_generation_prompt=True,
                                enable_thinking=False):
            return messages[0]["content"]

        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": [10, 11, 12]}

        def decode(self, ids, skip_special_tokens=True):
            return "val-dev-a"

    class FakeRequest:
        def __init__(self):
            self.frame = SimpleNamespace(
                sense=SimpleNamespace(
                    prompt_ids=torch.tensor([[10, 11, 12]])))
            self.released = False

        def release(self):
            self.released = True

    class FakeSession:
        device = "cpu"

        def __init__(self):
            self.prepare_kwargs = None
            self.generate_kwargs = None
            self.request = None

        def prepare(self, **kwargs):
            self.prepare_kwargs = kwargs
            self.request = FakeRequest()
            return self.request

        def generate(self, request, **kwargs):
            self.generate_kwargs = kwargs
            return torch.tensor([[10, 11, 12, 999, 130073]])

    config = _config()
    dev = _episode("dev", "dev")
    decisions = build_value_split_decisions(
        [dev], _revocations(dev), config, "dev")
    texts = query_text_register([dev])
    fact = decisions[0]
    tokenizer, session = FakeTokenizer(), FakeSession()
    row = generate_row(
        session, tokenizer, fact,
        texts[(fact["episode_id"], fact["phase"], fact["query_id"])])
    kwargs = session.prepare_kwargs
    assert kwargs["enabled_ports"] == ()
    assert kwargs["enabled_branches"] == ()
    assert kwargs["core_impl"] == "none"
    assert kwargs["training"] is False
    assert kwargs["visual"] is None
    assert f"Record value: {fact['selected_value']}" in kwargs["query"]
    assert session.generate_kwargs["max_new_tokens"] == MAX_NEW_TOKENS
    assert session.generate_kwargs["eos_token_id"] == STOP_TOKEN_IDS
    assert row["generated_ids"] == [999, 130073]
    assert row["ended_by_eos"] is True
    assert session.request.released is True


def test_generation_scoring_correction_unaffected_unknown_and_revoked():
    config = _config()
    dev = _episode("dev", "dev")
    decisions = build_value_split_decisions(
        [dev], _revocations(dev), config, "dev")
    texts = query_text_register([dev])
    generations = _correct_generator(decisions, texts)
    validate_generation_rows(generations, decisions)
    labels = _labels_for([dev])
    metrics = score_generations(generations, labels)["splits"]["dev"]
    assert metrics["overall"]["exact_answer_rate"] == 1.0
    assert metrics["correction"]["exact_answer_rate"] == 1.0
    assert metrics["unaffected"]["exact_answer_rate"] == 1.0
    assert metrics["unknown"]["exact_answer_rate"] == 1.0
    assert metrics["revoked"]["exact_answer_rate"] == 1.0
    gate = evaluate_dev_model_gate(metrics, config["endpoint"])
    assert gate["passed"] is True
    assert gate["evaluated_after_literal_pass"] is True

    wrong = _wrong_generator(decisions, texts)
    wrong_metrics = score_generations(
        wrong, labels)["splits"]["dev"]
    assert evaluate_dev_model_gate(
        wrong_metrics, config["endpoint"])["passed"] is False
    broken = deepcopy(metrics)
    broken["unknown"]["exact_answer_rate"] = 0.5
    assert evaluate_dev_model_gate(
        broken, config["endpoint"])["passed"] is False
    broken = deepcopy(metrics)
    broken["correction"]["exact_answer_rate"] = 0.5
    assert evaluate_dev_model_gate(
        broken, config["endpoint"])["passed"] is False
    tuned = deepcopy(config["endpoint"])
    tuned["dev_model_exact_answer_min"] = 0.5
    with pytest.raises(ValueError, match="may not be tuned"):
        evaluate_dev_model_gate(metrics, tuned)
    with pytest.raises(ValueError, match="row count"):
        validate_generation_rows(generations[:-1], decisions)
    duplicated = list(generations)
    duplicated[-1] = dict(duplicated[0])
    with pytest.raises(ValueError, match="repeat|differs"):
        validate_generation_rows(duplicated, decisions)
    leaked = dict(generations[0])
    leaked["expected"] = "val-dev-a"
    with pytest.raises(ValueError, match="evaluation field"):
        validate_generation_rows([leaked] + list(generations[1:]), decisions)


# ---------------------------------------------------------------------------
# Sequence and stop contracts
# ---------------------------------------------------------------------------

def test_decisions_are_written_before_labels_and_full_pass_runs_heldout(
        tmp_path):
    config = _config()
    dev = _episode("dev", "dev")
    heldout = _episode("hold", "heldout")
    train = _episode("train", "train")
    episodes = [dev, heldout, train]
    revocations = {**_revocations(dev), **_revocations(heldout),
                   **_revocations(train)}
    labels = _labels_for(episodes)
    output = tmp_path / "pass"
    calls = []

    def labels_loader():
        assert (output / "decisions_dev.jsonl").is_file()
        assert not (output / "decisions_heldout.jsonl").exists()
        assert not (output / "decisions_train.jsonl").exists()
        assert not (output / "generations_dev.jsonl").exists()
        assert not (output / "report.json").exists()
        return labels

    report = run_bank_value(
        episodes, revocations, labels_loader, config, output,
        prepared_info={"mode": "synthetic"},
        baseline_info={"mode": "synthetic"},
        generation_factory=_factory(_correct_generator, calls))
    assert report["format"] == REPORT_FORMAT
    assert calls == ["factory"]
    assert report["gating"]["literal_dev"]["passed"] is True
    assert report["gating"]["model_dev"]["gate"]["passed"] is True
    assert report["gating"]["heldout"]["status"] == "evaluated"
    assert report["gating"]["heldout"]["evaluated_after_dev_pass"] is True
    assert report["gating"]["train"]["status"] == "descriptive_only"
    assert report["gating"]["train"]["model_generation"] == (
        "not_run_descriptive_only")
    assert report["gating"]["stopped_before_heldout"] is False
    for name in ("decisions_dev.jsonl", "decisions_heldout.jsonl",
                 "decisions_train.jsonl", "generations_dev.jsonl",
                 "generations_heldout.jsonl", "report.json"):
        assert (output / name).is_file(), name
    assert not (output / "generations_train.jsonl").exists()
    assert report["raw_outputs"]["verified_unchanged"] is True
    assert report["counts"]["by_split"]["dev"] == len(
        build_value_split_decisions(episodes, revocations, config, "dev"))
    assert report["counts"]["generations"]["train"] == 0
    assert report["resource_usage"]["checkpoints_written"] == 0
    assert report["resource_usage"]["optimizer_updates"] == 0
    assert report["resource_usage"]["gradients"] == 0
    assert report["resource_usage"]["base_exports"] == 0
    assert report["resource_usage"]["extension_exports"] == 0
    assert report["protocol"]["compressed_reference"][
        "bank_plus_compressed_arm"] is False


def test_literal_failure_stops_before_model_and_heldout(tmp_path):
    config = _config()
    dev = _episode("dev", "dev")
    heldout = _episode("hold", "heldout")
    episodes = [dev, heldout]
    revocations = {**_revocations(dev), **_revocations(heldout)}
    labels = _labels_for(episodes)
    changed = deepcopy(labels)
    for state_labels in changed[dev["episode_id"]].values():
        for target in state_labels["targets"].values():
            if target["kind"] == "fact":
                target["expected"] = "val-wrong"
    output = tmp_path / "literal-fail"
    calls = []

    def labels_loader():
        assert (output / "decisions_dev.jsonl").is_file()
        return changed

    report = run_bank_value(
        episodes, revocations, labels_loader, config, output,
        prepared_info={"mode": "synthetic"},
        baseline_info={"mode": "synthetic"},
        generation_factory=_factory(_correct_generator, calls))
    assert calls == []
    assert report["gating"]["literal_dev"]["passed"] is False
    assert report["gating"]["model_dev"]["status"] == "not_run"
    assert report["gating"]["model_dev"]["reason"] == "dev_literal_failed"
    assert report["gating"]["heldout"]["status"] == "not_run"
    assert report["gating"]["heldout"]["decisions"] == 0
    assert report["gating"]["train"]["status"] == "not_run"
    assert report["gating"]["stopped_before_model"] is True
    assert report["gating"]["stopped_before_heldout"] is True
    assert report["outcome"]["status"] == "stopped_at_dev_literal"
    assert report["classification"]["model_execution"] is False
    assert report["resource_usage"]["model_loads"] == 0
    assert (output / "decisions_dev.jsonl").is_file()
    assert not (output / "generations_dev.jsonl").exists()
    assert not (output / "decisions_heldout.jsonl").exists()
    assert not (output / "decisions_train.jsonl").exists()


def test_model_failure_stops_before_heldout_and_train(tmp_path):
    config = _config()
    dev = _episode("dev", "dev")
    heldout = _episode("hold", "heldout")
    episodes = [dev, heldout]
    revocations = {**_revocations(dev), **_revocations(heldout)}
    labels = _labels_for(episodes)
    output = tmp_path / "model-fail"
    calls = []

    report = run_bank_value(
        episodes, revocations, lambda: labels, config, output,
        prepared_info={"mode": "synthetic"},
        baseline_info={"mode": "synthetic"},
        generation_factory=_factory(_wrong_generator, calls))
    assert calls == ["factory"]
    assert report["gating"]["literal_dev"]["passed"] is True
    assert report["gating"]["model_dev"]["gate"]["passed"] is False
    assert report["gating"]["heldout"]["status"] == "not_run"
    assert report["gating"]["heldout"]["reason"] == "dev_model_failed"
    assert report["gating"]["heldout"]["decisions"] == 0
    assert report["gating"]["heldout"]["generations"] == 0
    assert report["gating"]["train"]["status"] == "not_run"
    assert report["gating"]["stopped_before_heldout"] is True
    assert report["outcome"]["status"] == "stopped_at_dev_model"
    assert (output / "generations_dev.jsonl").is_file()
    assert not (output / "decisions_heldout.jsonl").exists()
    assert not (output / "generations_heldout.jsonl").exists()
    assert not (output / "decisions_train.jsonl").exists()
    assert "generation_metrics" in report
    assert report["raw_outputs"]["verified_unchanged"] is True


def test_generation_rows_are_immutable_first_writes(tmp_path):
    config = _config()
    dev = _episode("dev", "dev")
    decisions = build_value_split_decisions(
        [dev], _revocations(dev), config, "dev")
    texts = query_text_register([dev])
    rows = _correct_generator(decisions, texts)
    output = tmp_path / "immutable"
    output.mkdir()
    manifest = write_generation_rows(output, rows, split="dev")
    assert manifest["first_write"] is True
    assert manifest["immutable"] is True
    assert manifest["rows"] == len(rows)
    path = output / manifest["filename"]
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        write_generation_rows(output, rows, split="dev")
    assert path.read_bytes() == before


# ---------------------------------------------------------------------------
# Configuration and reference contracts
# ---------------------------------------------------------------------------

def test_endpoints_and_resource_bounds_cannot_be_tuned():
    config = _config()
    for key, value in (
            ("dev_literal_fact_value_min", 0.5),
            ("dev_literal_unknown_selections_max", 1),
            ("dev_literal_revoked_selections_max", 1),
            ("dev_model_exact_answer_min", 0.5),
            ("dev_model_correction_min", 0.1),
            ("dev_model_unaffected_min", 0.1),
            ("dev_model_unknown_correct_min", 0.1),
            ("dev_model_revoked_correct_min", 0.1)):
        changed = deepcopy(config)
        changed["endpoint"][key] = value
        with pytest.raises(ValueError, match="may not be tuned"):
            validate_config(changed)

    changed = deepcopy(config)
    changed["generation"]["max_new_tokens"] = 9
    with pytest.raises(ValueError, match="may not be tuned"):
        validate_config(changed)
    changed = deepcopy(config)
    changed["generation"]["stop_token_ids"] = [1]
    with pytest.raises(ValueError, match="stop token set"):
        validate_config(changed)
    changed = deepcopy(config)
    changed["generation"]["enabled_branches"] = ["memory"]
    with pytest.raises(ValueError, match="branch"):
        validate_config(changed)
    changed = deepcopy(config)
    changed["budget"]["checkpoints_written"] = 1
    with pytest.raises(ValueError, match="must be zero"):
        validate_config(changed)
    changed = deepcopy(config)
    changed["budget"]["model_loads"] = 2
    with pytest.raises(ValueError, match="exactly one model load"):
        validate_config(changed)
    assert config["generation"]["max_decisions_per_split"] == (
        MAX_DECISIONS_PER_SPLIT)
    assert config["generation"]["artifact_cap_bytes"] == ARTIFACT_CAP_BYTES
    assert config["generation"]["max_new_tokens"] == MAX_NEW_TOKENS


def test_compressed_reference_is_separate_and_pinned():
    reference = compressed_reference(_config())
    assert reference["role"] == "referenced_only_not_combined"
    assert reference["metric_version"] == "memory.acceptance.v4"
    assert reference["bank_plus_compressed_arm"] is False
    assert reference["sha256"] == (
        "d348e476b15886f68cd3373a65cf4a8e80072b93d5589cd150368a1ad7629afe")
    assert reference["quality"]["accepted_correct_rate"] is not None


# ---------------------------------------------------------------------------
# Real artifacts (optional, read-only)
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.skipif(
    not all((NATIVE02 / name).is_file()
            for name in ("policy_checkpoint.pt", "run_report.json",
                         "ledger.json"))
    or not all((PREPARED04 / name).is_file()
               for name in ("episodes.json", "revocations.json",
                            "labels.json", "manifest.json",
                            "protocol.json", "config.json")),
    reason=("optional ignored native02/prepared04 artifacts; provision "
            "runs/memory_causal_fresh_20261026/native02 and "
            "data/memory_causal_fresh_20261026/prepared04 first"))
def test_native02_checkpoint_bindings_and_cached_state():
    config = _config()
    info = verify_baseline(config, prepared=PREPARED04, verify_source=False)
    assert info["checkpoint"]["written"] is False
    assert info["checkpoint"]["sha256"] == (
        "6e680c62aa848c95864d3fe8a4c979c6241b9a05af226c921c7af454197b6db3")
    assert info["bindings_verified"] is True
    assert info["cached_state_verified"] is True
    assert info["phase"] == "first_process"
    assert info["ledger"]["commits"] == 72
    assert info["ledger"]["attempts"] == 72
    assert len(info["cached_episodes"]) == 12
    assert {item["split"] for item in info["cached_episodes"]} == {
        "train", "dev", "heldout"}
    assert all(item["states"] == ["corrected", "initial"]
               for item in info["cached_episodes"])
    assert all(item["writes"] > 0 and item["query_vectors"] > 0
               for item in info["cached_episodes"])

    changed = deepcopy(config)
    changed["baseline"]["checkpoint_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="checkpoint bytes differ"):
        verify_baseline(changed, prepared=PREPARED04, verify_source=False)


@pytest.mark.integration
@pytest.mark.skipif(
    not all((PREPARED04 / name).is_file()
            for name in ("episodes.json", "revocations.json",
                         "labels.json", "manifest.json")),
    reason=("optional ignored prepared04 replay; provision "
            "data/memory_causal_fresh_20261026/prepared04 first"))
def test_prepared04_dev_literal_values_pass_the_frozen_endpoint():
    config = _config()
    episodes = json.loads(
        (PREPARED04 / "episodes.json").read_text())["episodes"]
    revocations = json.loads(
        (PREPARED04 / "revocations.json").read_text())["revocations"]
    labels = json.loads(
        (PREPARED04 / "labels.json").read_text())["episodes"]
    dev = build_value_split_decisions(episodes, revocations, config, "dev")
    assert len(dev) == 152
    assert decision_digest(dev) == decision_digest(dev)
    metrics = score_literal_values(dev, labels)["splits"]["dev"]
    gate = evaluate_dev_literal_gate(metrics, config["endpoint"])
    assert gate["passed"] is True
    overall = metrics["overall"]
    assert overall["fact_n"] == 128
    assert overall["fact_value_exact_n"] == 124
    assert overall["fact_value_exact_rate"] == 124 / 128
    assert overall["unknown_selected_n"] == 0
    assert overall["revoked_selected_n"] == 0
    assert overall["correction"]["value_exact_rate"] == 1.0
    assert overall["unaffected"]["value_exact_rate"] == 1.0
    assert normalize_answer("  Val-A  ") == "val-a"

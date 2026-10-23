"""Frozen dev selection and offline accounting with a fake tokenizer, no model."""
import copy
import importlib.util
import json
from pathlib import Path
import random

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("prepare_infrastructure_preflight_test",
                                            ROOT / "scripts/prepare_infrastructure_preflight.py")
PREPARE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREPARE)
FACTORY = __import__("make_infrastructure_slice")


class FakeTokenizer:
    eos_token_id = 2

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking):
        assert tokenize is False and add_generation_prompt is True and enable_thinking is False
        assert len(messages) == 1 and messages[0]["role"] == "user"
        return "<chat>" + messages[0]["content"] + "</chat>"

    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return [7] * max(1, (len(text) + 11) // 12)


class OverlongNativeTokenizer(FakeTokenizer):
    def encode(self, text, *, add_special_tokens):
        if text.startswith("<chat>"):
            return [7] * 513
        return super().encode(text, add_special_tokens=add_special_tokens)


class OverlongWriterTokenizer(FakeTokenizer):
    def encode(self, text, *, add_special_tokens):
        if not text.startswith("<chat>") and " | event_time=" in text:
            return [7] * 513
        return super().encode(text, add_special_tokens=add_special_tokens)


@pytest.fixture(scope="module")
def release_data(tmp_path_factory):
    root = tmp_path_factory.mktemp("preflight") / "release"
    FACTORY.build_release(root)
    manifest, sources, payloads = PREPARE.load_release(root)
    return root, manifest, sources, payloads


def _selected(release_data):
    _, _, sources, payloads = release_data
    return PREPARE.select_cases(payloads, sources)


def test_fixed_selection_is_role_family_scenario_balanced_and_order_independent(release_data):
    _, _, sources, payloads = release_data
    selected = _selected(release_data)
    assert len(selected) == 16 and len({p["group_id"] for p in selected}) == 16
    assert {p["split"] for p in selected} == {"dev"}
    assert PREPARE.Counter(p["target"]["action"] for p in selected) == {"answer": 8, "ask": 4, "unknown": 4}
    expected_times = {"late_correction": 9, "late_observation_ask": 9,
                      "late_observation_unknown": 5, "tombstone": 5}
    for p in selected:
        source = sources[p["sources"][0]["source_id"]]
        assert p["query"]["at"] == expected_times[source["scenario"]]
        stratum = [x["group_id"] for x in payloads if x["split"] == "dev"
                   and sources[x["sources"][0]["source_id"]]["family"] == source["family"]
                   and sources[x["sources"][0]["source_id"]]["scenario"] == source["scenario"]]
        assert p["group_id"] == min(stratum)
    shuffled = payloads.copy()
    random.Random(11).shuffle(shuffled)
    assert [p["id"] for p in PREPARE.select_cases(shuffled, sources)] == [p["id"] for p in selected]


def test_missing_stratum_and_incomplete_world_are_rejected(release_data):
    _, _, sources, payloads = release_data
    chosen = _selected(release_data)[0]
    chosen_source = sources[chosen["sources"][0]["source_id"]]
    without_stratum = [p for p in payloads if not (p["split"] == "dev"
        and sources[p["sources"][0]["source_id"]]["family"] == chosen_source["family"]
        and sources[p["sources"][0]["source_id"]]["scenario"] == chosen_source["scenario"])]
    with pytest.raises(ValueError, match="missing fixed development stratum"):
        PREPARE.select_cases(without_stratum, sources)
    without_query = [p for p in payloads if p["id"] != chosen["id"]]
    with pytest.raises(ValueError, match="exactly two queries"):
        PREPARE.select_cases(without_query, sources)


def test_selection_rejects_changed_action_distribution(release_data):
    _, _, sources, payloads = release_data
    changed = copy.deepcopy(payloads)
    selected_id = _selected(release_data)[0]["id"]
    next(p for p in changed if p["id"] == selected_id)["target"]["action"] = "unknown"
    with pytest.raises(ValueError, match="action balance"):
        PREPARE.select_cases(changed, sources)


def test_token_views_supervise_one_eos_and_do_not_add_writer_tokens(release_data):
    row = PREPARE.lines(release_data[0] / "exports/dev.jsonl")[0]
    tokenizer = FakeTokenizer()
    actual = PREPARE.token_lengths(row, tokenizer)
    assert actual["answer"] == len(tokenizer.encode(row["response"], add_special_tokens=False))
    assert actual["eos"] == 1
    assert actual["supervised_answer_plus_eos"] == actual["answer"] + 1
    assert actual["native_full"] == actual["native_prompt"] + actual["answer"] + 1
    assert actual["source_writer"] == len(tokenizer.encode("\n".join(row["memory_texts"]), add_special_tokens=False))
    assert actual["native_full"] != actual["native_prompt"] + actual["source_writer"] + actual["answer"] + 1


def test_summary_preserves_measured_costs_and_declared_empirical_quantiles():
    assert PREPARE.summarize([20, 1, 3, 2]) == {
        "count": 4, "sum": 26, "min": 1, "mean": 6.5, "p50": 2, "p95": 3, "max": 20}
    with pytest.raises(ValueError, match="empty accounting"):
        PREPARE.summarize([])


def test_semantic_controls_recompute_missing_fields_and_reject_constant_fit(release_data):
    root, _, sources, _ = release_data
    selected = _selected(release_data)
    result = PREPARE.controls(selected, sources, root)
    assert result["examples"] == 16
    assert result["constant_perfect_fit_rejected"] is True
    assert result["best_constant"]["correct"] == 4
    assert all(row["total"] == 16 and row["rate"] == row["correct"] / 16 for row in result["constant_scores"])
    missing = result["missing_fact_controls"]
    assert sum(row["answer_to_abstention"] for row in missing) == 8
    assert all(row["expected"]["action"] in {"ask", "unknown"} for row in missing)
    assert all(row["removed_fact_count"] >= 1 for row in missing)
    asks = [row for row in missing if row["expected"]["action"] == "ask"]
    for row in asks:
        fields = FACTORY._design()["families"][row["family"]]["fact_fields"]
        assert row["expected"]["response"] == "Please provide " + ", ".join(fields) + "."
    assert "no native generation" in result["kind"]
    assert result["shuffled_evidence_model_condition"] == "pending"


def _prepare_args(tmp_path, monkeypatch):
    # Temporary test artifacts are deliberately outside the repository; relative
    # path presentation is independent of the preparation/solver/token logic.
    monkeypatch.setattr(PREPARE, "relative", lambda path: str(Path(path).resolve()))
    lock = tmp_path / "fake-lock.json"
    lock.write_text('{"fixture":true}\n')
    return tmp_path / "prepared", tmp_path / "fake-source", lock


def test_prepare_writes_exact_case_lock_complete_accounting_and_automated_review(release_data, tmp_path, monkeypatch):
    root, _, _, _ = release_data
    output, model, lock = _prepare_args(tmp_path, monkeypatch)
    result = PREPARE.prepare(root, output, FakeTokenizer(), model, lock)
    assert result["examples"] == 2000 and result["selected"] == 16
    spec = json.loads((output / "evaluation_spec.json").read_bytes())
    cases = PREPARE.lines(output / "dev_cases.jsonl")
    assert [row["id"] for row in cases] == [row["id"] for row in spec["selected_cases"]]
    assert spec["data_sha256"] == PREPARE.sha(output / "dev_cases.jsonl")
    assert spec["max_new_tokens"] == 32 and spec["max_length"] == 512
    assert spec["device"] == "mps" and spec["dtype"] == "float32" and spec["cpu_fallback"] is False
    assert spec["decoding"] == "greedy" and spec["seed"] == 17
    accounting = json.loads((output / "token_accounting.json").read_bytes())
    assert accounting["admitted"] == 2000 and accounting["overlong_ids"] == []
    assert accounting["truncated_examples"] == 0
    assert accounting["summary"]["overall"]["native_full"]["count"] == 2000
    assert accounting["summary"]["by_role"]["train"]["native_full"]["count"] == 1200
    assert set(accounting["summary"]["by_family"]) == set(PREPARE.FAMILIES)
    review = json.loads((output / "stratified_review.json").read_bytes())
    assert "Automated" in review["kind"] and "not a human" in review["kind"]
    assert review["sample_episodes"] == 320 and review["sample_worlds"] == 160
    assert len(review["critical_strata"]) == 16
    case_groups = {p["id"]: p["group_id"] for p in release_data[3]}
    for stratum in review["critical_strata"]:
        assert len(stratum["episode_ids"]) == 20
        assert len({case_groups[i] for i in stratum["episode_ids"]}) == 10
        assert stratum["independent_raw_source_checks_passed"] is True
    with pytest.raises(ValueError, match="fresh preflight directory"):
        PREPARE.prepare(root, output, FakeTokenizer(), model, lock)


@pytest.mark.parametrize("tokenizer", [OverlongNativeTokenizer(), OverlongWriterTokenizer()],
                         ids=["native-full", "separate-source-writer"])
def test_prepare_rejects_either_overlong_view_without_creating_output(release_data, tmp_path, monkeypatch, tokenizer):
    root = release_data[0]
    output, model, lock = _prepare_args(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="admission"):
        PREPARE.prepare(root, output, tokenizer, model, lock)
    assert not output.exists()

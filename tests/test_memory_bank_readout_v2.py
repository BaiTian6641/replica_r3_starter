"""CPU contracts for the predeclared prompt-v2 readout experiment.

Covered here:

* prompt v2 excludes evaluation labels and requires the exact value or an
  explicit status word;
* the 32-token cap, single-line and EOS format flags;
* mechanically pinned phase/state denominators and phase-scoped gates;
* restart query-text disjointness and old-query accounting;
* role-sealed labels, heldout-open spying and fail-closed ordering;
* the restart job's dependence on a passing first-process receipt;
* fresh-process, read-only checkpoint verification (integration marker).

No test loads a model, writes a checkpoint or runs a native job.  The
orchestration tests inject a contract-following fake generator.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from conftest import require_local_artifacts
from replica_lab.memory_address_bank import _sha_file, _sha_text
from replica_lab.memory_bank_readout_v2 import (
    EXPECTED_DEV_COUNTS,
    MAX_NEW_TOKENS,
    PROMPT_CONTRACT,
    PROMPT_FORMAT,
    REPORT_FORMAT,
    ROLE_MANIFEST_FORMAT,
    STATUS_VOCABULARY,
    build_prompt_v2,
    build_readout_split_decisions,
    decision_counts,
    decision_status,
    evaluate_literal_gate,
    evaluate_phase_gate,
    evaluate_restart_accounting,
    evaluate_state_gates,
    generation_row_v2,
    load_prepared,
    load_role_labels,
    restart_query_accounting,
    run_readout_job,
    score_phase,
    score_literal_values,
    seal_role_labels,
    validate_config,
    validate_decision_counts,
    validate_generation_rows,
    verify_baselines,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/memory_bank_readout_v2.json"
PREPARED = ROOT / "data/memory_causal_fresh_20261026/prepared04"
ROLES = ROOT / "data/memory_causal_fresh_20261026/prepared04_roles"


def _config():
    return json.loads(CONFIG_PATH.read_text())


def _dev_decisions(config=None):
    config = config or _config()
    prepared = load_prepared(config)
    return build_readout_split_decisions(
        prepared["episodes"], prepared["revocations"], config, "dev")


def _contract_answer(decision):
    status = decision_status(decision)
    return decision["selected_value"] if status is None else status


def _row(decision, query_text, text=None):
    answer = _contract_answer(decision) if text is None else text
    generated = [1]
    return generation_row_v2(
        decision, query_text,
        prompt_sha256=_sha_text(build_prompt_v2(decision, query_text)),
        prompt_ids_sha256="0" * 64,
        prompt_tokens=12,
        generated_ids=generated,
        generated_text=answer)


def _rows_for(phase, text=None, config=None):
    config = config or _config()
    prepared = load_prepared(config)
    decisions = _dev_decisions(config)
    from replica_lab.memory_bank_readout_v2 import query_text_register
    query_texts = query_text_register(prepared["episodes"])
    selected = [decision for decision in decisions
                if decision["phase"] == phase]
    return [ _row(decision,
                  query_texts[(decision["episode_id"], decision["phase"],
                               decision["query_id"])], text)
             for decision in selected ]


def _fake_generator(calls, *, text=None, record_open=None):
    """Contract-following generator factory; records calls."""
    def factory(*, deadline, baseline_info):
        calls.append({"baseline": bool(baseline_info)})

        def generate(decisions, query_texts):
            rows = []
            for decision in decisions:
                key = (decision["episode_id"], decision["phase"],
                       decision["query_id"])
                answer = (_contract_answer(decision)
                          if text is None else text)
                generated = [1]
                rows.append(generation_row_v2(
                    decision, query_texts[key],
                    prompt_sha256=_sha_text(
                        build_prompt_v2(decision, query_texts[key])),
                    prompt_ids_sha256="0" * 64,
                    prompt_tokens=12,
                    generated_ids=generated,
                    generated_text=answer))
            if record_open is not None:
                record_open("generated", tuple(
                    decision["phase"] for decision in decisions))
            return tuple(rows)

        return generate

    return factory


def _stub_baseline(config):
    return {
        "first_process": {
            "label": "first_process",
            "phase": "first_process",
            "checkpoint": {"written": False, "loaded": True},
            "encoded": {"old_observations_reencoded": 0,
                        "old_queries_reencoded": 0,
                        "restart_new_queries": 0},
            "source": None,
        },
        "restart": {
            "label": "restart",
            "phase": "restart",
            "checkpoint": {"written": False, "loaded": True,
                           "loaded_sha256":
                               config["baseline"]["first_process"][
                                   "checkpoint_sha256"]},
            "encoded": {"old_observations_reencoded": 0,
                        "old_queries_reencoded": 0,
                        "restart_new_queries": 216},
            "source": None,
        },
    }


@pytest.fixture()
def stub_baselines(monkeypatch):
    import replica_lab.memory_bank_readout_v2 as module

    def _verify(config, prepared, *, jobs=("first", "restart"),
                verify_source=True):
        stub = _stub_baseline(config)
        return ({"first_process": stub["first_process"]}
                if tuple(jobs) == ("first",) else stub)

    monkeypatch.setattr(module, "verify_baselines", _verify)


@pytest.fixture()
def role_spy(monkeypatch):
    import replica_lab.memory_bank_readout_v2 as module

    log = []
    original = module._read_role_json

    def spy(path):
        log.append(Path(path).name)
        return original(path)

    monkeypatch.setattr(module, "_read_role_json", spy)
    return log


# ---------------------------------------------------------------------------
# Configuration and denominators
# ---------------------------------------------------------------------------

def test_config_is_frozen_and_rejects_tuning():
    config = _config()
    assert validate_config(config) is config
    tampered = deepcopy(config)
    tampered["generation"]["max_new_tokens"] = 8
    with pytest.raises(ValueError):
        validate_config(tampered)
    tampered = deepcopy(config)
    tampered["endpoint"]["expected_dev_counts"]["fact_rows"] = 124
    with pytest.raises(ValueError):
        validate_config(tampered)
    tampered = deepcopy(config)
    tampered["endpoint"]["heldout"] = "evaluate_always"
    with pytest.raises(ValueError):
        validate_config(tampered)


def test_denominators_are_mechanically_computed_and_pinned():
    config = _config()
    decisions = _dev_decisions(config)
    dev_labels = load_role_labels(None, "dev", config)
    counts = decision_counts(decisions, dev_labels)
    assert counts == EXPECTED_DEV_COUNTS
    assert validate_decision_counts(counts, config)
    # The four bank abstentions stay in the fact denominator.
    assert counts["fact_rows"] - 4 == 124
    assert counts["restart_revoked_status_rows"] == 4


def test_literal_gate_reuses_pinned_v1_endpoint():
    config = _config()
    decisions = _dev_decisions(config)
    dev_labels = load_role_labels(None, "dev", config)
    metrics = score_literal_values(decisions, dev_labels)["splits"]["dev"]
    gate = evaluate_literal_gate(metrics, config)
    assert gate["passed"] is True
    assert gate["checks"]["fact_value_exact"]["fact_n"] == 128
    assert gate["checks"]["fact_value_exact"]["value_exact_n"] == 124


# ---------------------------------------------------------------------------
# Prompt v2 and generation-row format
# ---------------------------------------------------------------------------

def test_prompt_v2_requires_contract_and_excludes_labels():
    prompt = build_prompt_v2(
        {"episode_id": "e", "state": "initial", "phase": "first_process",
         "query_id": "q", "selected_ids": ["rec-1"],
         "selected_value": "val-shown",
         "reason": None},
        "What is the value?")
    for line in PROMPT_CONTRACT:
        assert line in prompt
    assert "Record value: val-shown" in prompt
    assert prompt.count("val-shown") == 1
    assert "\nAnswer:\n" not in prompt and prompt.endswith("Answer:")
    for forbidden in ("expected", "cohort", "split", "val-other",
                      "target_entry_id"):
        assert forbidden not in prompt


def test_prompt_v2_status_vocabulary_and_grammar():
    for reason, word in (
            ("abstain_no_address", "unknown"),
            ("abstain_revoked_only_address", "revoked"),
            ("abstain_ambiguous_address", "ambiguous"),
            ("abstain_superseded_only_address", "superseded")):
        prompt = build_prompt_v2(
            {"episode_id": "e", "state": "initial",
             "phase": "revocation", "query_id": "q", "selected_ids": [],
             "selected_value": None, "reason": reason}, "Q?")
        assert f"Record status: {word.upper()}" in prompt
        assert "Record value:" not in prompt
        assert word in STATUS_VOCABULARY
    assert "Permitted status words: UNKNOWN, REVOKED, AMBIGUOUS, SUPERSEDED." \
        in PROMPT_CONTRACT


def test_generation_row_cap_single_line_and_eos_flags():
    decision = _dev_decisions()[0]
    text = "val-one"
    prompt_sha = _sha_text(build_prompt_v2(decision, "Q?"))
    eos_row = generation_row_v2(
        decision, "Q?", prompt_sha256=prompt_sha,
        prompt_ids_sha256="0" * 64, prompt_tokens=10,
        generated_ids=[7, 1], generated_text=text)
    assert eos_row["ended_by_eos"] is True
    assert eos_row["cap_hit"] is False
    assert eos_row["single_line"] is True
    assert eos_row["max_new_tokens"] == MAX_NEW_TOKENS == 32
    assert eos_row["format"].endswith("generations.v2")
    for forbidden in ("expected", "cohort", "in_correction"):
        assert forbidden not in eos_row

    cap_row = generation_row_v2(
        decision, "Q?", prompt_sha256=prompt_sha,
        prompt_ids_sha256="0" * 64, prompt_tokens=10,
        generated_ids=list(range(1, MAX_NEW_TOKENS + 1)),
        generated_text="x")
    assert cap_row["cap_hit"] is True
    assert cap_row["ended_by_eos"] is False

    multi = generation_row_v2(
        decision, "Q?", prompt_sha256=prompt_sha,
        prompt_ids_sha256="0" * 64, prompt_tokens=10,
        generated_ids=[1], generated_text="line1\nline2")
    assert multi["single_line"] is False

    with pytest.raises(ValueError):
        generation_row_v2(
            decision, "Q?", prompt_sha256=prompt_sha,
            prompt_ids_sha256="0" * 64, prompt_tokens=10,
            generated_ids=list(range(1, MAX_NEW_TOKENS + 2)),
            generated_text="x")


def test_generation_rows_immutable_and_label_free():
    from replica_lab.memory_bank_readout_v2 import query_text_register

    config = _config()
    prepared = load_prepared(config)
    decisions = _dev_decisions(config)
    query_texts = query_text_register(prepared["episodes"])
    rows = _rows_for("first_process", config=config)
    assert validate_generation_rows(rows, [
        decision for decision in decisions
        if decision["phase"] == "first_process"], query_texts)
    injected = deepcopy(rows[0])
    injected["expected"] = "leak"
    with pytest.raises(ValueError):
        validate_generation_rows([injected] + list(rows[1:]), [
            decision for decision in decisions
            if decision["phase"] == "first_process"], query_texts)


# ---------------------------------------------------------------------------
# Phase and state gates
# ---------------------------------------------------------------------------

def test_phase_gates_use_pinned_denominators():
    config = _config()
    dev_labels = load_role_labels(None, "dev", config)
    decisions = _dev_decisions(config)
    for phase, expected_rows in (
            ("first_process", 72), ("restart", 72), ("revocation", 8)):
        metrics = score_phase(_rows_for(phase, config=config),
                              dev_labels, decisions, phase)
        gate = evaluate_phase_gate(metrics, config)
        assert gate["passed"] is True, gate
        assert gate["checks"]["single_line"]["n"] == expected_rows
        assert gate["checks"]["ended_by_eos"]["n"] == expected_rows
        assert gate["checks"]["cap_hits"]["count"] == 0
    fp = score_phase(_rows_for("first_process", config=config),
                     dev_labels, decisions, "first_process")
    assert fp["correction"]["n"] == 8
    assert fp["unaffected"]["n"] == 20
    assert fp["unknown"]["n"] == 8
    assert fp["control"]["n"] == 8
    restart = score_phase(_rows_for("restart", config=config),
                          dev_labels, decisions, "restart")
    assert restart["correction"]["n"] == 8
    assert restart["unaffected"]["n"] == 20
    assert restart["control"]["n"] == 8


def test_phase_gate_fails_on_wrong_answer_and_format():
    config = _config()
    dev_labels = load_role_labels(None, "dev", config)
    decisions = _dev_decisions(config)
    wrong = score_phase(_rows_for("first_process", text="Apologies",
                                  config=config),
                        dev_labels, decisions, "first_process")
    gate = evaluate_phase_gate(wrong, config)
    assert gate["passed"] is False
    assert gate["checks"]["exact_answer"]["passed"] is False
    assert gate["checks"]["control_false_value"]["passed"] is True

    # Multi-line and cap-hit rows must fail the format checks even when the
    # normalized answer would otherwise be correct.
    rows = _rows_for("revocation", config=config)
    bad = []
    for index, row in enumerate(rows):
        if index == 0:
            row = dict(row, generated_text="revoked\nprefix",
                       single_line=False)
        elif index == 1:
            row = dict(row, generated_ids=list(range(1, 33)),
                       ended_by_eos=False, cap_hit=True)
        bad.append(row)
    metrics = score_phase(tuple(bad), dev_labels, decisions, "revocation")
    gate = evaluate_phase_gate(metrics, config)
    assert gate["passed"] is False
    assert gate["checks"]["single_line"]["passed"] is False
    assert gate["checks"]["ended_by_eos"]["passed"] is False
    assert gate["checks"]["cap_hits"]["passed"] is False


def test_state_gates_cover_all_128_fact_rows():
    config = _config()
    dev_labels = load_role_labels(None, "dev", config)
    decisions = _dev_decisions(config)
    combined = tuple(_rows_for("first_process", config=config)) \
        + tuple(_rows_for("restart", config=config))
    gate = evaluate_state_gates(combined, dev_labels, decisions, config)
    assert gate["passed"] is True
    assert gate["checks"]["initial_fact_exact"]["n"] == 64
    assert gate["checks"]["corrected_fact_exact"]["n"] == 64
    # The four abstentions are counted as misses, never relabeled.
    assert gate["checks"]["corrected_fact_exact"]["correct_n"] <= 64


def test_restart_query_texts_are_disjoint_and_accounted():
    config = _config()
    prepared = load_prepared(config)
    accounting = restart_query_accounting(prepared["episodes"], split="dev")
    assert accounting["restart_texts_disjoint"] is True
    assert accounting["restart_vs_first_process_overlap_n"] == 0
    assert accounting["disjoint_rate"] == 1.0
    assert accounting["restart_new_queries"] > 0
    assert accounting["old_first_process_queries_regenerated"] == 0
    gate = evaluate_restart_accounting(accounting, config)
    assert gate["passed"] is True

    overlapping = dict(accounting, restart_texts_disjoint=False,
                       restart_vs_first_process_overlap_n=1,
                       overlap=[{"episode_id": "e"}])
    assert evaluate_restart_accounting(overlapping, config)["passed"] is False
    regenerated = dict(accounting,
                       old_first_process_queries_regenerated=1)
    assert evaluate_restart_accounting(regenerated, config)["passed"] is False


# ---------------------------------------------------------------------------
# Role sealing and heldout-open ordering
# ---------------------------------------------------------------------------

def test_role_sealing_is_disjoint_and_pinned(tmp_path):
    manifest = seal_role_labels(PREPARED, tmp_path / "roles")
    assert manifest["format"] == ROLE_MANIFEST_FORMAT
    assert manifest["role_sets_disjoint"] is True
    roles = manifest["roles"]
    assert set(roles) == {"dev", "heldout", "train"}
    episode_sets = []
    for role in ("dev", "heldout", "train"):
        path = tmp_path / "roles" / roles[role]["filename"]
        assert _sha_file(path) == roles[role]["sha256"]
        doc = json.loads(path.read_text())
        assert doc["role"] == role
        assert doc["evaluation_only"] is True
        assert doc["source_labels_sha256"] == manifest["source_labels_sha256"]
        episode_sets.append(set(doc["episode_ids"]))
    assert not (episode_sets[0] & episode_sets[1])
    assert not (episode_sets[0] & episode_sets[2])
    assert not (episode_sets[1] & episode_sets[2])
    manifest_doc = json.loads((tmp_path / "roles" / "manifest.json").read_text())
    assert manifest_doc["format"] == ROLE_MANIFEST_FORMAT
    assert manifest_doc["heldout_written"] is True
    assert manifest_doc["model_loaded"] is False


def test_role_loader_refuses_tampered_pin(tmp_path):
    config = _config()
    seal_role_labels(PREPARED, tmp_path / "roles")
    config["role_labels"]["root"] = str(tmp_path / "roles")
    dev_path = tmp_path / "roles" / "labels_dev.json"
    doc = json.loads(dev_path.read_text())
    doc["note"] = "tampered"
    dev_path.write_text(json.dumps(doc, sort_keys=True,
                                   separators=(",", ":")) + "\n")
    with pytest.raises(ValueError):
        load_role_labels(tmp_path / "roles", "dev", config)
    config["role_labels"]["pins"] = {
        role: _sha_file(tmp_path / "roles" / f"labels_{role}.json")
        for role in ("dev", "heldout", "train")}
    config["role_labels"]["pins"]["manifest"] = "0" * 64
    labels = load_role_labels(tmp_path / "roles", "dev", config)
    assert labels


def _broken_dev_role_config(tmp_path):
    """Seal roles, corrupt every dev fact target, and pin the new file."""
    import replica_lab.memory_bank_readout_v2 as module

    config = _config()
    seal_role_labels(PREPARED, tmp_path / "roles")
    dev_path = tmp_path / "roles" / "labels_dev.json"
    doc = json.loads(dev_path.read_text())
    for states in doc["episodes"].values():
        for state in states.values():
            for target in state["targets"].values():
                if target["kind"] == "fact":
                    target["expected"] = "val-not-present"
    dev_path.write_text(json.dumps(doc, sort_keys=True,
                                   separators=(",", ":")) + "\n")
    manifest_path = tmp_path / "roles" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["roles"]["dev"]["sha256"] = _sha_file(dev_path)
    manifest["roles"]["dev"]["bytes"] = dev_path.stat().st_size
    manifest_path.write_text(json.dumps(manifest, sort_keys=True,
                                        separators=(",", ":")) + "\n")
    config["role_labels"]["root"] = str(tmp_path / "roles")
    config["role_labels"]["pins"] = {
        "dev": _sha_file(dev_path),
        "heldout": _sha_file(tmp_path / "roles" / "labels_heldout.json"),
        "train": _sha_file(tmp_path / "roles" / "labels_train.json"),
        "manifest": _sha_file(manifest_path),
    }
    return config


def test_literal_failure_stops_before_model_and_heldout(
        tmp_path, monkeypatch, stub_baselines, role_spy):
    config = _broken_dev_role_config(tmp_path)
    calls = []
    output = tmp_path / "first"
    report = run_readout_job(
        config, job="first", output=output, prepared=PREPARED,
        roles=tmp_path / "roles",
        generation_factory=_fake_generator(calls),
        config_path=CONFIG_PATH, verify_source=False)
    assert report["format"] == REPORT_FORMAT
    assert report["outcome"]["status"] == "stopped_at_dev_literal"
    assert report["gating"]["literal_dev"]["passed"] is False
    assert report["gating"]["stopped_before_model"] is True
    assert report["gating"]["heldout"]["status"] == "not_run"
    assert calls == []
    assert role_spy == ["labels_dev.json"]
    assert not (output / "generations_dev_first_process.jsonl").exists()
    assert not (output / "decisions_heldout.jsonl").exists()
    assert not (output / "generations_heldout.jsonl").exists()
    assert report["role_label_access"]["heldout_opened"] is False


def test_first_job_failure_never_reads_restart_or_heldout(
        tmp_path, monkeypatch, stub_baselines, role_spy):
    config = _config()
    output = tmp_path / "first"
    report = run_readout_job(
        config, job="first", output=output, prepared=PREPARED, roles=ROLES,
        generation_factory=_fake_generator([], text="Apologies"),
        config_path=CONFIG_PATH, verify_source=False)
    assert report["outcome"]["status"] == "stopped_at_dev_first_process"
    assert report["gating"]["first_process"]["gate"]["passed"] is False
    assert report["gating"]["heldout"]["status"] == "not_run"
    assert role_spy == ["labels_dev.json"]
    assert not (output / "decisions_heldout.jsonl").exists()
    assert not (output / "generations_heldout.jsonl").exists()
    assert not (output / "decisions_train.jsonl").exists()


def test_restart_job_requires_first_receipt(
        tmp_path, stub_baselines, role_spy):
    config = _config()
    calls = []
    output = tmp_path / "restart"
    with pytest.raises(ValueError):
        run_readout_job(
            config, job="restart", output=output, prepared=PREPARED,
            roles=ROLES, first_output=tmp_path / "missing",
            generation_factory=_fake_generator(calls),
            config_path=CONFIG_PATH, verify_source=False)
    assert calls == []
    assert role_spy == []
    assert not (output / "decisions_heldout.jsonl").exists()
    assert not (output / "generations_heldout.jsonl").exists()


def test_restart_job_rejects_failed_first_gate(
        tmp_path, stub_baselines, role_spy):
    config = _config()
    first = tmp_path / "first"
    first.mkdir()
    report = {
        "format": REPORT_FORMAT,
        "job": "first",
        "protocol": {"config_sha256": _sha_file(CONFIG_PATH)},
        "raw_outputs": {"verified_unchanged": True, "files": {}},
        "gating": {
            "literal_dev": {"passed": True},
            "first_process": {"passed": False},
            "heldout": {"status": "not_run"},
        },
    }
    (first / "report.json").write_text(json.dumps(report))
    calls = []
    with pytest.raises(ValueError):
        run_readout_job(
            config, job="restart", output=tmp_path / "restart",
            prepared=PREPARED, roles=ROLES, first_output=first,
            generation_factory=_fake_generator(calls),
            config_path=CONFIG_PATH, verify_source=False)
    assert calls == []
    assert role_spy == []


def test_restart_failure_keeps_heldout_sealed(
        tmp_path, stub_baselines, role_spy):
    config = _config()
    first_calls = []
    first = tmp_path / "first"
    first_report = run_readout_job(
        config, job="first", output=first, prepared=PREPARED, roles=ROLES,
        generation_factory=_fake_generator(first_calls),
        config_path=CONFIG_PATH, verify_source=False)
    assert first_report["gating"]["first_process"]["gate"]["passed"] is True
    assert "labels_heldout.json" not in role_spy

    role_spy.clear()
    restart_calls = []
    restart_output = tmp_path / "restart"
    report = run_readout_job(
        config, job="restart", output=restart_output, prepared=PREPARED,
        roles=ROLES, first_output=first,
        generation_factory=_fake_generator(restart_calls, text="Apologies"),
        config_path=CONFIG_PATH, verify_source=False)
    assert report["outcome"]["status"] == "stopped_at_dev_restart"
    assert report["gating"]["restart"]["gate"]["passed"] is False
    assert report["gating"]["heldout"]["status"] == "not_run"
    assert report["role_label_access"]["heldout_opened"] is False
    assert role_spy == ["labels_dev.json"]
    assert not (restart_output / "decisions_heldout.jsonl").exists()
    assert not (restart_output / "generations_heldout.jsonl").exists()
    assert restart_calls


def test_restart_pass_opens_heldout_only_after_dev_gates(
        tmp_path, stub_baselines, role_spy):
    import replica_lab.memory_bank_readout_v2 as module

    config = _config()
    first = tmp_path / "first"
    first_report = run_readout_job(
        config, job="first", output=first, prepared=PREPARED, roles=ROLES,
        generation_factory=_fake_generator([]),
            config_path=CONFIG_PATH, verify_source=False)
    assert first_report["gating"]["literal_dev"]["passed"] is True
    assert first_report["gating"]["heldout"]["status"] == "not_run"

    restart_output = tmp_path / "restart"
    heldout_state = {}
    original = module._read_role_json

    def spy(path):
        if Path(path).name == "labels_heldout.json":
            heldout_state["generations_exists"] = (
                restart_output / "generations_heldout.jsonl").is_file()
            heldout_state["decisions_exists"] = (
                restart_output / "decisions_heldout.jsonl").is_file()
            heldout_state["restart_generation_exists"] = (
                restart_output / "generations_dev_restart.jsonl").is_file()
            heldout_state["revocation_generation_exists"] = (
                restart_output / "generations_dev_revocation.jsonl").is_file()
        return original(path)

    module._read_role_json = spy
    try:
        report = run_readout_job(
            config, job="restart", output=restart_output,
            prepared=PREPARED, roles=ROLES, first_output=first,
            generation_factory=_fake_generator([]),
            config_path=CONFIG_PATH, verify_source=False)
    finally:
        module._read_role_json = original

    assert report["gating"]["restart"]["gate"]["passed"] is True
    assert report["gating"]["revocation"]["gate"]["passed"] is True
    assert report["gating"]["state_denominators"]["gate"]["passed"] is True
    assert report["gating"]["restart_accounting"]["gate"]["passed"] is True
    assert report["gating"]["heldout"]["status"] == "evaluated"
    assert report["role_label_access"]["opened"] == [
        "dev", "heldout", "train"]
    assert report["role_label_access"]["heldout_opened"] is True
    assert heldout_state == {
        "generations_exists": True,
        "decisions_exists": True,
        "restart_generation_exists": True,
        "revocation_generation_exists": True,
    }
    assert (restart_output / "generations_heldout.jsonl").is_file()
    assert (restart_output / "decisions_heldout.jsonl").is_file()
    assert (restart_output / "decisions_train.jsonl").is_file()
    assert report["gating"]["train"]["status"] == "descriptive_only"
    assert report["raw_outputs"]["verified_unchanged"] is True
    assert report["resource_usage"]["gradients"] == 0
    assert report["resource_usage"]["checkpoints_written"] == 0


def test_restart_accounting_reports_no_old_observation_encoding(
        tmp_path, stub_baselines, role_spy):
    config = _config()
    first = tmp_path / "first"
    run_readout_job(
        config, job="first", output=first, prepared=PREPARED, roles=ROLES,
        generation_factory=_fake_generator([]),
            config_path=CONFIG_PATH, verify_source=False)
    report = run_readout_job(
        config, job="restart", output=tmp_path / "restart",
        prepared=PREPARED, roles=ROLES, first_output=first,
        generation_factory=_fake_generator([]),
            config_path=CONFIG_PATH, verify_source=False)
    accounting = report["gating"]["restart_accounting"]["metrics"]
    assert accounting["old_observations_encoded"] == 0
    assert accounting["old_first_process_queries_regenerated"] == 0
    assert accounting["restart_new_queries"] == 36
    assert report["baseline"]["restart"]["checkpoint"]["written"] is False
    assert report["baseline"]["first_process"]["checkpoint"]["written"] is False


# ---------------------------------------------------------------------------
# Real-artifact CPU replay (no model, no native job)
# ---------------------------------------------------------------------------

@require_local_artifacts(
    "runs/memory_causal_fresh_20261026/native02",
    "runs/memory_causal_fresh_20261026/restart02",
)
@pytest.mark.integration
def test_fresh_process_checkpoint_verification():
    config = _config()
    result = verify_baselines(config, PREPARED, verify_source=False)
    assert set(result) == {"first_process", "restart"}
    first = result["first_process"]
    restart = result["restart"]
    assert first["checkpoint"]["written"] is False
    assert restart["checkpoint"]["written"] is False
    assert restart["checkpoint"]["loaded_sha256"] == \
        config["baseline"]["first_process"]["checkpoint_sha256"]
    assert restart["encoded"]["old_observations_reencoded"] == 0
    assert restart["encoded"]["old_queries_reencoded"] == 0
    assert restart["encoded"]["restart_new_queries"] == 216
    assert first["bindings_verified"] is True
    assert restart["bindings_verified"] is True
    assert first["cached_state_verified"] is True
    assert restart["cached_state_verified"] is True

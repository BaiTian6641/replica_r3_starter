"""Deterministic orchestration contracts for the Spark text-memory runner.

These tests use only the owned tiny CPU fixture.  They never load the sealed
Spark source, never claim native FP8 admission, and never read derived caches.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
import torch

import replica_lab.spark_text_memory_runner as runner
from replica_lab.architecture.spine_text import SparkTextMemoryPath
from replica_lab.spark_text_memory_runner import (
    ARMS, CPU_FIXTURE_LABEL, NATIVE_LABEL, CapExceeded, GenerationFailure,
    PreparedDataset, Refused, RunnerError, WorkLedger, WorkProfile,
    build_arm_views, build_cpu_fixture_backend, build_native_backend,
    greedy_generate, load_protocol, load_runner_state, prepare_development_memory,
    run_development, run_fit, run_protocol, score_answer,
)


@pytest.fixture(scope="module")
def protocol():
    return load_protocol("configs/spark_text_memory_protocol_v1.json")


@pytest.fixture(scope="module")
def dataset():
    return PreparedDataset.load("data/spark_text_memory_v1")


@pytest.fixture(scope="module")
def fixture_profile(protocol):
    return WorkProfile.from_protocol(
        protocol.config, fixture_context_tokens=256, fixture_max_new_tokens=2)


@pytest.fixture(scope="module")
def fit_run(protocol, dataset, fixture_profile, tmp_path_factory):
    output = tmp_path_factory.mktemp("fit") / "run"
    ledger = WorkLedger(fixture_profile)
    backend = build_cpu_fixture_backend(
        dataset, fixture_profile, seed=int(protocol.config["seed"]))
    ledger.load_backend()
    state = run_fit(protocol, dataset, backend, output, fixture_profile, ledger,
                    response_lr=1e-4)
    return {"output": output, "state": state, "backend": backend,
            "ledger": ledger}


@pytest.fixture(scope="module")
def full_run(protocol, tmp_path_factory):
    output = tmp_path_factory.mktemp("full") / "run"
    report = run_protocol(
        protocol_path="configs/spark_text_memory_protocol_v1.json",
        prepared_root="data/spark_text_memory_v1",
        output_root=output, backend_kind=CPU_FIXTURE_LABEL, device="cpu",
        fixture_context_tokens=256, fixture_max_new_tokens=2,
        restart_in_process=False)
    return {"output": output, "report": report}


def test_protocol_and_manifest_are_hash_bound_with_sealed_final_closed(protocol, dataset):
    assert protocol.protocol_id == "spark_text_memory_v1"
    assert [arm["id"] for arm in protocol.arms] == list(ARMS)
    assert dataset.manifest["counts"] == {
        "arms": 9, "development_cases": 12, "development_labels": 12,
        "fit_events": 4, "observations": 13, "parity_requests": 2,
        "procedures": 4, "restart_parity_requests": 1, "task_workspaces": 1,
    }
    assert dataset.manifest["sealed_final"] == {
        "materialized": False, "opened": False, "path": None}
    assert set(dataset.manifest["files"]) == {
        "observations.jsonl", "fit_events.jsonl", "development.jsonl",
        "development_labels.jsonl", "parity_requests.jsonl", "procedures.json",
        "tasks.json", "worlds.json"}
    fresh = PreparedDataset.load("data/spark_text_memory_v1")
    with pytest.raises(Refused, match="development labels require"):
        fresh.development_cases()
    with pytest.raises(Refused, match="development labels require"):
        fresh.parity_requests()


def test_corrected_work_bounds_match_the_declared_protocol_schedule(protocol):
    native = WorkProfile.from_protocol(protocol.config)
    caps = native.corrected_caps()
    assert native.context_tokens == 2048
    assert native.max_new_tokens == 32
    assert native.generation_requests == 111
    assert caps["training_forwards"] == 4
    assert caps["plastic_forward_calls"] == 4
    assert caps["plastic_backward_calls"] == 4
    assert caps["backward_calls"] == 8
    assert caps["backward_processed_positions"] == 8192
    assert caps["plastic_feature_vectors"] == 8192
    assert caps["plastic_backward_feature_vectors"] == 8192
    assert caps["optimizer_steps"] == 8
    assert caps["query_sensing_forwards"] == 3560
    assert caps["native_query_passes"] == 3560
    assert caps["forward_calls"] == 7120
    assert caps["autoregressive_forwards"] == 3552
    assert caps["generated_tokens"] == 3552
    assert caps["processed_text_positions"] == 14683680
    assert caps["retrieval_calls"] == 115
    assert caps["verification_calls"] == 112
    assert caps["checkpoint_writes"] == 5
    assert caps["retained_checkpoint_artifacts"] == 4
    assert caps["cache_hits"] == 0 and caps["cache_misses"] == 115
    declared = native.declared_schedule()
    assert declared["forward_calls"] == 7120
    assert declared["processed_text_positions"] == 14683680
    assert declared["plastic_forward_calls"] == 4
    assert declared["plastic_backward_calls"] == 4
    fixture_default = WorkProfile.from_protocol(
        protocol.config, backend_kind=CPU_FIXTURE_LABEL)
    assert fixture_default.label == CPU_FIXTURE_LABEL
    assert fixture_default.context_tokens == 512
    assert fixture_default.max_new_tokens == 4
    assert fixture_default.native_admissible is False


def test_ledger_caps_and_zero_new_work_retry_are_explicit(fixture_profile):
    ledger = WorkLedger(fixture_profile)
    ledger.set_phase("cold_start")
    before = dict(ledger.counters)
    result = ledger.retry_zero_new_work("event-retry")
    assert result["charged_new_work"] is False
    assert result["optimizer_steps"] == result["forward_calls"] == result["backward_calls"] == 0
    assert ledger.counters["retry_events"] == 1
    assert ledger.counters["native_retries"] == 0
    assert all(ledger.counters[key] == before[key] for key in
               ("training_forwards", "backward_calls", "optimizer_steps"))
    with pytest.raises(CapExceeded):
        for _ in range(fixture_profile.fit_events + 1):
            ledger.charge("fit_events")


def test_cpu_fixture_is_never_native_admissible(dataset, fixture_profile):
    backend = build_cpu_fixture_backend(
        dataset, fixture_profile, seed=123)
    assert backend.label == CPU_FIXTURE_LABEL
    assert backend.native_admissible is False
    assert backend.adapter.extension.spine.codec_version == runner.TEXT_CODEC_VERSION
    assert backend.adapter.extension.spine.representation_version == runner.TEXT_REPRESENTATION_VERSION
    assert runner._native_adapter_capability() is None  # current adapter blocker
    with pytest.raises(Refused, match="cpu_fixture"):
        build_cpu_fixture_backend(
            dataset, WorkProfile.from_protocol(
                load_protocol("configs/spark_text_memory_protocol_v1.json").config))


def test_native_backend_refuses_without_authorization_and_without_evidence(
        protocol, dataset, tmp_path):
    output = tmp_path / "native-out"
    with pytest.raises(Refused) as missing:
        build_native_backend(
            protocol, dataset, authorization_path=tmp_path / "missing.json",
            backend_evidence_path=tmp_path / "missing-backend.json",
            output_root=output, device="mps")
    assert missing.value.check == "authorization"
    receipt = {
        "authorization_id": "a1", "protocol_id": protocol.protocol_id,
        "protocol_sha256": "0" * 64, "protocol_commit": "0" * 40,
        "implementation_sha256": {"x": "y"},
        "prepared_manifest_sha256": dataset.manifest_sha256,
        "source_identity": "official_fp8",
        "execution_format": "official_native_fp8",
        "execution_backend": protocol.config["launch"][
            "required_execution_binding"]["execution_backend"],
        "backend_evidence_sha256": "1" * 64, "attempt_id": "attempt",
        "output_root": str(output), "issued_by": "test",
        "expires_at_utc": "2099-01-01T00:00:00+00:00",
    }
    authorization = tmp_path / "authorization.json"
    authorization.write_text(json.dumps(receipt))
    with pytest.raises(Refused) as wrong:
        build_native_backend(
            protocol, dataset, authorization_path=authorization,
            backend_evidence_path=tmp_path / "missing-backend.json",
            output_root=output, device="mps")
    assert wrong.value.check == "authorization"
    assert "protocol hash differs" in wrong.value.detail


def test_fit_emits_four_declared_response_updates_and_retains_four_checkpoints(fit_run):
    state = fit_run["state"]
    ledger = fit_run["ledger"]
    receipt = state.receipt
    assert receipt["status"] == "fit_executed"
    assert receipt["execution_complete"] is True
    assert receipt["fit_events"] == 4
    assert receipt["optimizer_steps"] == 8
    assert receipt["optimizer_scopes"] == {"response": 4, "plastic": 4, "total": 8}
    assert receipt["actual_parameter_updates"] == 4
    assert receipt["actual_plastic_updates"] == 4
    assert receipt["changed_trainable_names"]
    assert receipt["changed_expert_names"]
    assert receipt["changed_tensor_count"] == len(receipt["changed_trainable_names"])
    assert receipt["plastic_changed_tensor_count"] == len(receipt["changed_expert_names"])
    plastic = receipt["plastic_learning"]
    assert plastic["evaluated"] is True
    assert plastic["actual_updates"] == 4
    assert plastic["optimizer_steps"] == 4
    assert plastic["forward_calls"] == 4
    assert plastic["backward_calls"] == 4
    assert plastic["fit_evaluation_forward_calls"] == 4
    assert plastic["changed_trainable_names"] == receipt["changed_expert_names"]
    assert plastic["clock_events"] == 4
    assert plastic["revisions"] == [1, 2, 3, 4]
    assert plastic["plastic_changed"] is True
    assert plastic["pre_fit_plastic_digest"] != plastic["retained_plastic_digest"]
    assert all(row["target"]["representation"]
               == "core.memory_projection(versioned_answer_vector)"
               for row in plastic["per_event"])
    assert all(row["target"]["target_sha256"] for row in plastic["per_event"])
    assert receipt["checkpoint_writes"] == 5
    assert len(receipt["retained_checkpoints"]) == 4
    assert receipt["evicted_checkpoints"] == ["fit/event-01"]
    assert ledger.counters["backward_processed_positions"] == 4 * 256
    assert ledger.counters["plastic_feature_vectors"] == 4 * 256
    assert ledger.counters["plastic_backward_feature_vectors"] == 4 * 256
    assert ledger.counters["training_forwards"] == 4
    assert ledger.counters["plastic_forward_calls"] == 4
    assert ledger.counters["plastic_backward_calls"] == 4
    assert ledger.counters["backward_calls"] == 8
    assert ledger.counters["optimizer_steps"] == 8
    assert ledger.counters["plastic_fit_evaluation_forwards"] == 4
    assert ledger.counters["retrieval_calls"] == 4
    assert ledger.counters["cache_misses"] == 4
    assert ledger.counters["cache_hits"] == 0
    assert len(state.path.texts) == 3
    assert "fit-alpha-open" not in state.path.texts  # superseded correction
    assert set(state.path.procedures) == {"alpha_procedure", "beta_procedure"}
    assert "fit-beta-task" in state.path.tasks
    assert len(state.path.tasks["fit-beta-task"].observed_completions) == 1
    assert state.path.response_steps == 4
    assert state.path.response_parameter_updates == 4
    assert state.path.experiment.bank.expert_steps == [4, 0]
    assert state.path.experiment.controller.clock.eligible_events == 4
    assert receipt["derived_caches_required"] is False


def test_zero_learning_rate_is_refused_before_work(protocol, dataset, fixture_profile, tmp_path):
    backend = build_cpu_fixture_backend(dataset, fixture_profile, seed=7)
    ledger = WorkLedger(fixture_profile)
    with pytest.raises(Refused) as refused:
        run_fit(protocol, dataset, backend, tmp_path / "zero", fixture_profile,
                ledger, response_lr=0.0)
    assert refused.value.check == "zero_learning_rate"
    assert ledger.counters["training_forwards"] == 0


def test_noop_response_step_is_rejected_and_preserved(monkeypatch, protocol, dataset,
                                                       fixture_profile, tmp_path):
    backend = build_cpu_fixture_backend(dataset, fixture_profile, seed=9)
    ledger = WorkLedger(fixture_profile)

    def fake_step(self, query, *, labels, extra_passes=()):
        return {
            "execution_complete": True, "physical_optimizer_steps": 1,
            "actual_parameter_update": False, "changed_trainable_names": [],
            "changed_expert_names": [], "train_fit": "unevaluated",
            "transfer": "unevaluated", "retention": "unevaluated",
            "useful_learning": "unevaluated", "learning_evidence": {},
        }

    monkeypatch.setattr(SparkTextMemoryPath, "response_step", fake_step)
    output = tmp_path / "noop"
    with pytest.raises(RunnerError, match="zero-change"):
        run_fit(protocol, dataset, backend, output, fixture_profile, ledger)
    assert ledger.counters["no_op_steps"] == 1
    failure = json.loads((output / "failure.json").read_text())
    assert failure["status"] == "failed"
    assert failure["stage"] == "fit"


def test_development_has_twelve_cases_per_arm_and_two_parity_requests(full_run):
    report = full_run["report"]
    work = report["work"]
    actual = work["actual"]
    assert report["execution_complete"] is True
    assert report["development"]["cases_per_arm"] == 12
    assert report["development"]["cases_total"] == 108
    assert report["development"]["parity_requests"] == 2
    assert work["reconciled"] is True
    assert work["cap_violations"] == []
    assert actual["fit_events"] == 4
    assert actual["development_cases"] == 108
    assert actual["training_forwards"] == 4
    assert actual["plastic_forward_calls"] == 4
    assert actual["plastic_backward_calls"] == 4
    assert actual["backward_calls"] == 8
    assert actual["backward_processed_positions"] == 1024
    assert actual["plastic_feature_vectors"] == 1024
    assert actual["plastic_backward_feature_vectors"] == 1024
    assert actual["optimizer_steps"] == 8
    assert actual["plastic_fit_evaluation_forwards"] == 4
    assert actual["generation_requests"] == 111
    assert actual["autoregressive_forwards"] == 222
    assert actual["generated_tokens"] == 222
    assert actual["retrieval_calls"] == 115
    assert actual["verification_calls"] == 112
    assert actual["checkpoint_writes"] == 5
    assert actual["retained_checkpoint_artifacts"] == 4
    assert actual["model_loads"] == 2
    assert actual["tokenizer_loads"] == 2
    assert actual["reloads"] == 1
    assert actual["restarts"] == 1
    assert actual["cache_hits"] == 0
    assert actual["cache_misses"] == 115
    assert actual["teacher_passes"] == 0
    assert actual["replay_events"] == 0
    assert actual["consolidation_events"] == 0
    assert actual["native_retries"] == 0
    assert actual["query_sensing_forwards"] == 119
    assert actual["native_query_passes"] == 119
    assert actual["forward_calls"] == 349
    assert actual["processed_text_positions"] == 88431
    assert work["per_arm"] == {arm: 12 for arm in ARMS}
    assert set(work["cost_phases"]) >= {"cold_start", "steady_state", "amortized"}


def test_fresh_process_restart_removes_source_and_matches_parity(full_run):
    restart = full_run["report"]["restart"]
    assert restart["process"] == "fresh_process"
    assert restart["source_evidence_removed"] is True
    assert restart["derived_caches_restored"] is False
    assert restart["parity_match_exact_tokens"] is True
    inventory = restart["inventory_after_source_removal"]
    assert inventory["exact_text_records"] == 0
    assert inventory["evidence_records"] == 0
    assert inventory["procedures"] == 0
    assert inventory["task_workspaces"] == 0
    assert inventory["plastic_revision"] == 4
    assert inventory["plastic_clock_events"] == 4
    assert inventory["plastic_expert_steps"] == [4, 0]
    assert full_run["output"].joinpath("restart/restart_report.json").is_file()


def test_development_rows_match_causal_arm_information(full_run):
    path = full_run["output"] / "development/trajectories.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    assert len(rows) == 108
    by_arm = {arm: [row for row in rows if row["arm_id"] == arm] for arm in ARMS}
    assert all(len(value) == 12 for value in by_arm.values())
    full = by_arm["full_eligible_context"][0]
    assert full["information"]["source_text_included"] is True
    assert full["information"]["removed_source_text"] is False
    oracle = by_arm["selective_exact_and_procedure"][0]
    assert oracle["information"]["procedure_hint"] is not None
    assert oracle["query"]["oracle_selected"] is True
    assert oracle["query"]["host_eligible_record_ids"]
    for arm, rows_for_arm in by_arm.items():
        for row in rows_for_arm:
            assert row["query"]["oracle_selected"] is (arm == "selective_exact_and_procedure")
            # Answer-aware record identities appear only in the oracle control.
            if arm != "selective_exact_and_procedure":
                assert row["query"]["host_eligible_record_ids"] in (None, [])
    p_only = by_arm["source_free_P_only"][0]
    assert p_only["query"]["memory_source"] == "P"
    assert p_only["query"]["selected_record_ids"] == []
    assert p_only["information"]["removed_source_text"] is True
    combined = by_arm["E_plus_P"][0]
    assert combined["query"]["memory_source"] == "E_plus_P"
    assert combined["query"]["context_memory_source"] == "P"
    assert combined["query"]["selected_record_ids"]
    inapplicable = by_arm["inapplicable_procedure"][0]
    assert inapplicable["information"]["procedure_hint"]
    assert inapplicable["query"]["selected_record_ids"] == []
    shuffled = by_arm["shuffled_memory"][0]
    assert shuffled["query"]["memory_source"] == "E"
    assert shuffled["query"]["selected_record_ids"]


def test_plastic_update_changes_p_and_source_free_reads_use_it(
        protocol, dataset, fit_run, fixture_profile):
    state = fit_run["state"]
    dataset.open_development(state.receipt)
    memory = prepare_development_memory(state.path, protocol, dataset)
    views = build_arm_views(
        protocol, fit_run["backend"].adapter, state.path, memory,
        trained_plastic=state.trained_plastic, pre_plastic=state.pre_plastic)
    trained_digest = runner._plastic_digest(state.trained_plastic)
    pre_digest = runner._plastic_digest(state.pre_plastic)
    assert trained_digest != pre_digest
    assert runner._plastic_digest(
        views["source_free_P_only"].path.experiment.controller.checkpoint_payload()
    ) == trained_digest
    assert runner._plastic_digest(
        views["no_update"].path.experiment.controller.checkpoint_payload()
    ) == pre_digest
    case = dataset.development_cases()[0]
    ids, mask = fit_run["backend"].encode_prompt(
        f"Question: {case['question']}\nAnswer:", fixture_profile.context_tokens)
    ledger = WorkLedger(fixture_profile)
    slots = {}
    for arm in ("source_free_P_only", "no_update"):
        view = views[arm]
        view.path.experiment.invalidate(reason="regression_no_cache")
        query = view.path.prepare_query(
            str(case["question"]), ids, mask,
            token_available_at=tuple(10 for _ in range(int(ids.shape[1]))),
            query_time=10, memory_source="P", plastic_expert_id=0,
            enabled_branches=("memory",))
        slots[arm] = query.context.slots["memory"].detach().clone()
        assert query.selected_record_ids == ()
    assert not torch.equal(slots["source_free_P_only"], slots["no_update"])
    assert state.receipt["plastic_learning"]["plastic_changed"] is True
    assert state.receipt["plastic_learning"]["changed_trainable_names"]


def test_generation_masks_terminal_stops_and_preserves_raw_output(full_run):
    rows = [json.loads(line) for line in
            (full_run["output"] / "development/trajectories.jsonl").read_text().splitlines()
            if line]
    row = rows[0]
    raw = row["raw"]
    assert raw["terminal_stop_ids_excluded"]
    assert raw["terminal_stop_tokens_emitted"] == []
    assert len(raw["token_ids"]) == 2
    assert raw["autoregressive_forwards"] == 2
    assert raw["generated_tokens"] == 2
    assert len(raw["steps"]) == 2
    assert all("top_token_ids" in step for step in raw["steps"])
    assert row["execution_complete"] is True
    assert row["score"]["scoring"] == "normalized_exact_or_contains"


def test_arm_views_distinguish_reset_shuffle_and_inapplicable_procedure(
        protocol, dataset, fit_run):
    state = fit_run["state"]
    dataset.open_development(state.receipt)
    memory = prepare_development_memory(state.path, protocol, dataset)
    views = build_arm_views(
        protocol, fit_run["backend"].adapter, state.path, memory,
        trained_plastic=state.trained_plastic, pre_plastic=state.pre_plastic)
    assert views["full_eligible_context"].source_prompt_removed is False
    assert views["full_eligible_context"].information[
        "answer_aware_record_ids_used"] is False
    for arm in ARMS:
        assert views[arm].information["answer_aware_record_ids_used"] is (arm ==
            "selective_exact_and_procedure")
    assert views["source_free_P_only"].path.texts == {}
    assert views["source_free_P_only"].path.experiment.evidence.pending == 0
    assert views["source_free_P_only"].controller_source == "trained"
    assert views["no_update"].controller_source == "pre_teaching"
    assert views["reset_P"].controller_source == "pre_teaching_reset"
    shuffled_records = views["shuffled_memory"].path.experiment.evidence.active
    assert shuffled_records
    assert any(record.source_lineage.get("shuffled_from") != record.record_id
               for record in shuffled_records)
    case = dataset.development_cases()[0]
    content, information = runner._prompt_for(views["inapplicable_procedure"],
                                               case, dataset)
    assert information["inapplicable_world"] is True
    assert information["procedure_hint"]
    assert content.startswith("Procedure ")
    assert views["E_plus_P"].mode == "E_plus_P"
    assert views["reset_P"].mode == "E_plus_P"


def test_durable_runner_state_rejects_tampering(protocol, dataset, fit_run, tmp_path):
    output = tmp_path / "copy"
    shutil.copytree(fit_run["output"], output)
    receipt = json.loads((output / "fit/fit_receipt.json").read_text())
    backend = build_cpu_fixture_backend(dataset, WorkProfile.from_protocol(
        protocol.config, fixture_context_tokens=256, fixture_max_new_tokens=2),
        seed=int(protocol.config["seed"]))
    artifact = output / receipt["runner_state"]["path"]
    artifact.write_bytes(artifact.read_bytes() + b"tamper")
    with pytest.raises(Refused, match="differs"):
        load_runner_state(output, receipt, protocol, dataset, backend)


def test_development_failure_is_preserved_with_partial_trajectory(
        monkeypatch, protocol, dataset, fit_run):
    state = fit_run["state"]
    dataset.open_development(state.receipt)
    ledger = WorkLedger(fit_run["state"].ledger.profile)

    def fail(*args, **kwargs):
        raise GenerationFailure("injected generation failure",
                                partial={"token_ids": [], "decoded": ""})

    monkeypatch.setattr(runner, "greedy_generate", fail)
    with pytest.raises(GenerationFailure):
        run_development(protocol, dataset, fit_run["backend"], fit_run["output"],
                        fit_run["state"].ledger.profile, ledger, state)
    failure_path = fit_run["output"] / "failure.json"
    assert failure_path.is_file()
    failure = json.loads(failure_path.read_text())
    assert failure["status"] == "failed"
    assert failure["stage"] == "development"
    assert failure["error"]["type"] == "GenerationFailure"
    assert failure["partial"]["case_id"] == "dev-01"
    assert ledger.counters["failure_rows"] == 1


def test_protocol_reports_resolved_contracts_without_precision_substitution(full_run):
    report = full_run["report"]
    assert report["protocol_mismatches"] == []
    assert report["backend"]["label"] == CPU_FIXTURE_LABEL
    assert report["backend"]["native_admissible"] is False
    assert report["backend"]["precision_substitution"] is False
    assert report["native_admission"]["attempted"] is False
    assert report["status"] == "complete"


def test_default_cli_cpu_fixture_without_optional_lengths(protocol, tmp_path):
    output = tmp_path / "default-cli"
    root = Path(__file__).resolve().parents[1]
    command = [
        sys.executable, str(root / "scripts/run_spark_text_memory_v1.py"),
        "--stage", "all", "--backend", CPU_FIXTURE_LABEL, "--device", "cpu",
        "--output", str(output),
        "--config", str(root / "configs/spark_text_memory_protocol_v1.json"),
        "--prepared", str(root / "data/spark_text_memory_v1"),
    ]
    result = subprocess.run(command, cwd=root, text=True, capture_output=True,
                            timeout=300)
    assert result.returncode == 0, result.stderr
    report = json.loads((output / "run_report.json").read_text())
    assert report["work_profile"] == {
        "label": CPU_FIXTURE_LABEL, "context_tokens": 512,
        "max_new_tokens": 4, "native_admissible": False}
    actual = report["work"]["actual"]
    assert report["execution_complete"] is True
    assert report["status"] == "complete"
    assert actual["model_loads"] == 2
    assert actual["tokenizer_loads"] == 2
    assert actual["training_forwards"] == 4
    assert actual["plastic_forward_calls"] == 4
    assert actual["plastic_backward_calls"] == 4
    assert actual["backward_calls"] == 8
    assert actual["backward_processed_positions"] == 2048
    assert actual["plastic_feature_vectors"] == 2048
    assert actual["plastic_backward_feature_vectors"] == 2048
    assert actual["optimizer_steps"] == 8
    assert actual["generation_requests"] == 111
    assert actual["autoregressive_forwards"] == 444
    assert actual["generated_tokens"] == 444
    assert actual["retrieval_calls"] == 115
    assert actual["verification_calls"] == 112
    assert actual["checkpoint_writes"] == 5
    assert actual["retained_checkpoint_artifacts"] == 4
    assert actual["reloads"] == 1
    assert actual["cache_misses"] == 115
    assert actual["query_sensing_forwards"] == 119
    assert actual["native_query_passes"] == 119
    assert actual["forward_calls"] == 571
    assert actual["processed_text_positions"] == 290970
    assert report["restart"]["process"] == "fresh_process"


def test_answer_scoring_distinguishes_exact_from_containment():
    exact = score_answer("ready", "ready")
    assert exact["exact_match"] and exact["matched"]
    contained = score_answer("unknown", "The outcome is unknown.")
    assert contained["exact_match"] is False
    assert contained["matched"] is True
    wrong = score_answer("closed", "open")
    assert wrong["matched"] is False

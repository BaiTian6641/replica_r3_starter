"""Self-contained CPU tests for the staged vision curriculum.

These tests build the curriculum corpus from the admitted repository sources,
never load a native model, never run a native job and never score a real final
split. They cover split/final seals, provenance, abstention coverage,
equal-compute schedules, optimizer routing, sensing leakage, component-vs-
baseline retention policy and report validation.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from replica_lab.contracts import fingerprint  # noqa: E402
from replica_lab.vision_curriculum import (  # noqa: E402
    ABSTAIN_KINDS, ARMS, AUDIT_FORBIDDEN_V2_DESCRIPTORS, CONTROLS,
    DIAGNOSTIC_FINAL_DESCRIPTORS,
    PROCEDURAL_PARAMETERS_SANITY, PROJECTOR_1D_TENSORS, PROJECTOR_2D_TENSORS,
    PROJECTOR_PARAMETER_COUNT, PROJECTOR_TENSORS, RUN_REPORT_FORMAT,
    SCORE_REPORT_FORMAT, SPLIT_TASKS, SPLITS, UNKNOWN_ANSWER,
    VisionCurriculumError, _question_leaks, _required_run_checks,
    answer_correct, assemble_score_report, assign_controls, audit_dataset,
    build_arm_declarations, build_curriculum, build_finalize_receipt,
    build_procedural, build_protocol, build_training_plan, caption_token_f1,
    dataset_pins, finalize_receipt_paths, is_unknown_answer, load_config,
    load_run_report, normalise_answer, protocol_sha256,
    require_finalize_allowed, require_global_finalize_allowed,
    reserve_finalize, retention_policy, score_caption, score_procedural,
    sha256_bytes, sha256_file, validate_config, validate_finalize_receipt,
    validate_generation_rows, validate_lr_reports, validate_protocol,
    validate_reload_report, validate_run_report, validate_score_claims,
    validate_score_report, verify_data_pins, verify_implementation_pins,
    verify_retention_pins, write_curriculum, write_finalize_receipt,
    write_protocol,
)

BASE_CONFIG = json.loads(
    (ROOT / "configs/vision_curriculum_v1.json").read_text(encoding="utf-8"))
TEMPLATE = json.loads(
    (ROOT / "configs/minicpm5_vision_v1_projector_wide_v1.json").read_text(
        encoding="utf-8"))
PREPARED = ROOT / "data/vision_curriculum_20261025"


def small_config(seed: int = 101, worlds: int = 8) -> dict:
    config = copy.deepcopy(BASE_CONFIG)
    config["seed"] = seed
    config["procedural"]["worlds_per_task"] = worlds
    config["data_root"] = "data/vision_curriculum_test"
    return validate_config(config)


@pytest.fixture(scope="session")
def procedural():
    return build_procedural(small_config())


@pytest.fixture(scope="session")
def built(tmp_path_factory, procedural):
    config = small_config()
    built = build_curriculum(config, ROOT)
    root = tmp_path_factory.mktemp("curriculum") / "data"
    write_curriculum(root, built)
    return {"root": root, "built": built, "config": config}


@pytest.fixture(scope="session")
def dataset(built):
    return audit_dataset(built["root"])


@pytest.fixture(scope="session")
def prepared_protocol():
    assert PREPARED.joinpath("protocol.json").is_file(), (
        "run scripts/prepare_vision_curriculum.py first")
    protocol = json.loads((PREPARED / "protocol.json").read_text())
    validate_protocol(protocol)
    return protocol


def _tamper(path: Path, needle: bytes, replacement: bytes):
    text = path.read_bytes()
    assert needle in text
    path.write_bytes(text.replace(needle, replacement, 1))


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def test_config_constants_and_validation():
    config = load_config(ROOT / "configs/vision_curriculum_v1.json")
    assert list(ARMS) == ["muon8bit", "adamw_control"]
    assert config["curriculum"]["total_updates"] == 128
    assert config["curriculum"]["total_examples"] == 512
    assert config["limits"]["max_updates"] == 320
    assert config["resource_policy"]["workers"] == 1
    assert config["resource_policy"]["cpu_fallback"] is False
    assert config["resource_policy"]["driver_limit_bytes"] >= 50 * 1024 ** 3
    assert config["retention"]["baseline"]["correct"] == 22
    assert config["retention"]["baseline"]["cases"] == 32
    assert config["train"]["warm_start"] is None
    assert config["train"]["init_recipe"] == "fresh_seeded_normal_std0.02"
    assert config["curriculum"]["selection"]["arm_selection"][
        "both_arms_required"] is True
    assert PROCEDURAL_PARAMETERS_SANITY["projector_hidden"] == 4096
    assert PROCEDURAL_PARAMETERS_SANITY["projector_parameter_count"] == 11_542_016


@pytest.mark.parametrize("mutate", [
    lambda c: c["train"].update(warm_start={"path": "checkpoints/v1"}),
    lambda c: c["retention"]["baseline"].update(correct=20),
    lambda c: c["resource_policy"].update(cpu_fallback=True),
    lambda c: c["action_limits"].update(host_actions="allowed"),
    lambda c: c["procedural"]["descriptor_pools"]["audit"].append(
        ["red", "circle"]),
    lambda c: c["curriculum"]["stage_b"].update(
        sources=["procedural_dev"]),
    lambda c: c["limits"].update(max_updates=100),
])
def test_config_rejects_declared_policy_violations(mutate):
    config = copy.deepcopy(BASE_CONFIG)
    mutate(config)
    with pytest.raises(VisionCurriculumError):
        validate_config(config)


# ---------------------------------------------------------------------------
# Procedural generation: disjointness, abstention, determinism
# ---------------------------------------------------------------------------


def test_procedural_counts_coverage_and_abstention(procedural):
    manifest = procedural["manifest"]
    assert manifest["format"] == "replica.vision_curriculum_procedural_manifest.v1"
    counts = manifest["counts"]
    assert counts["split_worlds"] == {"train": 48, "dev": 40, "audit": 40}
    assert counts["split_model_rows"] == {"train": 96, "dev": 80, "audit": 80}
    for split in SPLITS:
        counts = manifest["unknown_coverage"][split]
        assert set(counts) == set(ABSTAIN_KINDS)
        assert sum(counts.values()) == 2 * manifest["counts"][
            "unknown_worlds"][split]
        assert all(value >= 1 for value in counts.values())
    abstain = [case for case in manifest["cases"]
               if case["task"] == "abstain"]
    assert abstain
    for case in abstain:
        assert case["label"] == UNKNOWN_ANSWER
        assert case["check"]["kind"] == "abstain"
        assert case["check"]["reason"] in ABSTAIN_KINDS
        assert not _question_leaks(case["question"], "unknown")
    # Every answer task has a decisive answer change; abstention never does.
    for pair in manifest["pairs"]:
        if pair["task"] == "abstain":
            assert pair["answers"] == {"a": UNKNOWN_ANSWER,
                                       "b": UNKNOWN_ANSWER}
            assert pair["answers_differ"] is False
        else:
            assert pair["answers_differ"] is True
            assert UNKNOWN_ANSWER not in pair["answers"].values()


def test_procedural_descriptor_holdout(procedural):
    pools = {split: {tuple(item) for item in
                     procedural["manifest"]["descriptor_pools"][split]}
             for split in SPLITS}
    assert pools["audit"].isdisjoint(pools["train"])
    assert pools["audit"].isdisjoint(pools["dev"])
    assert pools["train"].isdisjoint(pools["dev"])
    assert pools["audit"].isdisjoint(AUDIT_FORBIDDEN_V2_DESCRIPTORS)
    assert pools["train"].isdisjoint(DIAGNOSTIC_FINAL_DESCRIPTORS)
    assert all(asset["split"] in SPLITS for asset in procedural["assets"])


def test_procedural_generation_is_deterministic(procedural):
    again = build_procedural(small_config())
    assert again["manifest"]["generation_fingerprint"] == \
        procedural["manifest"]["generation_fingerprint"]
    assert again["files"] == procedural["files"]


def test_procedural_model_rows_are_label_free(procedural):
    for row in procedural["rows"]:
        assert set(row) == {"case_id", "pair_id", "asset_id", "image",
                            "image_sha256", "question"}
        assert "answer" not in row and "label" not in row
    answers = {case["case_id"]: case["answer"]
               for case in procedural["cases"]}
    for row in procedural["rows"]:
        assert answers[row["case_id"]] not in row
        assert not _question_leaks(row["question"],
                                   answers[row["case_id"]])


# ---------------------------------------------------------------------------
# Curriculum build, splits, seals and the untouched final pointer
# ---------------------------------------------------------------------------


def test_curriculum_manifest_counts_and_seals(built, dataset):
    manifest = dataset.manifest
    assert manifest["splits"]["audit"] == ["curriculum_procedural_audit"]
    assert manifest["splits"]["sealed"] == ["external_final_eval_untouched"]
    assert manifest["counts"]["external"]["train"] == 15
    assert manifest["counts"]["external"]["development"] == 7
    assert manifest["counts"]["external"]["final_sealed"] == 8
    assert dataset.seals["external_final_consumed"] is False
    assert dataset.seals["audit_scored"] is False
    pointer = json.loads(
        (built["root"] / "stage_a/final_sealed.json").read_text())
    assert pointer["consumed"] is False and pointer["sealed"] is True
    assert pointer["rows"] == 8
    assert pointer["seal_sha256"] == dataset.seals["external_final_seal_sha256"]
    # No final image or caption was copied.
    final_images = list((built["root"] / "external").rglob("*"))
    assert not any("final" in path.name for path in final_images)
    # The audit and development manifests carry their own seals.
    audit_manifest = json.loads(
        (built["root"] / "procedural/audit_manifest.json").read_text())
    assert audit_manifest["seal_sha256"] == dataset.seals["audit_seal_sha256"]
    assert audit_manifest["usage"] == "scored_once_after_frozen_dev_report"
    development = json.loads(
        (built["root"] / "procedural/development_manifest.json").read_text())
    assert development["usage"] == "bounded_development_selection_only"
    assert development["seal_sha256"] == dataset.seals["dev_seal_sha256"]


def test_dataset_pins_verify(dataset):
    pins = dataset_pins(dataset.root)
    verify_data_pins(pins, dataset.root)
    assert pins["counts"]["audit_cases"] == dataset.manifest["counts"][
        "audit_cases"]
    assert pins["seals"]["audit"] == dataset.seals["audit_seal_sha256"]


def test_audit_dataset_rejects_tampered_audit_label(built):
    root = built["root"]
    path = root / "stage_b/audit/labels.jsonl"
    original = path.read_bytes()
    try:
        _tamper(path, b'"answer":"unknown"', b'"answer":"yes"')
        with pytest.raises(VisionCurriculumError):
            audit_dataset(root)
    finally:
        path.write_bytes(original)
    audit_dataset(root)


def test_audit_dataset_rejects_tampered_asset(built):
    root = built["root"]
    asset = next((root / "procedural/assets").glob("*.png"))
    original = asset.read_bytes()
    try:
        asset.write_bytes(original[:-1] + bytes([original[-1] ^ 0xFF]))
        with pytest.raises(VisionCurriculumError):
            audit_dataset(root)
    finally:
        asset.write_bytes(original)
    audit_dataset(root)


def test_dataset_rejects_audit_access_in_training(dataset):
    training = dataset.training_examples()
    training_ids = {row["case_id"] for row in training}
    for row in dataset.generation_cases("audit"):
        assert row["case_id"] not in training_ids
    for row in dataset.generation_cases("dev"):
        assert row["case_id"] not in training_ids
    # Stage A uses only external train; Stage B uses only the two train sources.
    sources = {row["source"] for row in training}
    assert sources == {"external_train", "procedural_source_train",
                       "procedural_train"}
    assert all(row["split"] == "train" for row in training)


def test_unknown_training_examples_have_verifier_reasons(dataset):
    labels = dataset.stage_b_labels["train"]
    unknown = [row for row in labels.values()
               if row["answer"] == UNKNOWN_ANSWER]
    assert {row["kind"] for row in unknown} == set(ABSTAIN_KINDS)
    for row in unknown:
        assert row["check"]["kind"] == "abstain"
        assert row["label_visibility"] == "verifier_only"


# ---------------------------------------------------------------------------
# Schedule and equal-compute arms
# ---------------------------------------------------------------------------


def test_schedule_stages_and_probe_budget(dataset):
    schedule = dataset.schedule
    assert schedule["updates"] == 128
    assert schedule["examples"] == 512
    assert [stage["stage"] for stage in schedule["stages"]] == ["A", "B"]
    assert schedule["stages"][0]["updates"] == 16
    assert schedule["stages"][0]["batch"] == 4
    assert schedule["stages"][1]["updates"] == 112
    assert schedule["stages"][1]["batch"] == 4
    assert schedule["probe"]["updates_per_candidate"] == 16
    assert schedule["probe"]["batch"] == 2
    assert schedule["probe"]["total_probe_updates"] == 64
    assert schedule["probe"]["total_probe_examples"] == 128
    assert schedule["probe"]["selection_split"] == "procedural_dev"
    assert len(schedule["dev_probe_case_ids"]) == 24
    assert schedule["schedule_sha256"] == dataset.manifest["schedule"][
        "schedule_sha256"]


def test_schedule_never_references_dev_or_audit(built, dataset):
    train_ids = {row["case_id"] for row in dataset.training_examples()}
    dev_ids = {row["case_id"] for row in dataset.dev_rows()}
    audit_ids = {row["case_id"] for row in dataset.audit_rows()}
    scheduled = {case_id for stage in dataset.schedule["stages"]
                 for batch in stage["batches"] for case_id in batch}
    assert scheduled <= train_ids
    assert not (scheduled & dev_ids)
    assert not (scheduled & audit_ids)
    for stage in dataset.schedule["stages"]:
        for batch in stage["batches"]:
            assert len(batch) == stage["batch"]


def test_arm_declarations_equal_compute():
    arms = build_arm_declarations(small_config())
    assert arms["equal_compute"]["same_schedule_for_both_arms"] is True
    assert arms["equal_compute"]["same_initialization_for_both_arms"] is True
    assert arms["equal_compute"]["updates_per_arm"] == 128
    assert arms["equal_compute"]["examples_per_arm"] == 512
    assert arms["equal_compute"]["probe_updates_per_arm"] == 32
    assert arms["equal_compute"]["probe_examples_per_arm"] == 64
    assert arms["projector_hidden"] == 4096
    assert arms["projector_parameter_count"] == 11_542_016
    for arm in ARMS:
        names = sorted(name for group in arms["arms"][arm]["groups"]
                       for name in group["param_names"])
        assert names == sorted(PROJECTOR_TENSORS)
    assert arms["arms"]["adamw_control"]["declared_lr"] == 0.0003
    assert arms["arms"]["muon8bit"]["declared_lr"] == 0.0001


# ---------------------------------------------------------------------------
# Optimizer routing (CPU torch)
# ---------------------------------------------------------------------------


def _tiny_projector():
    import torch
    return torch.nn.Sequential(
        torch.nn.LayerNorm(8), torch.nn.Linear(8, 16),
        torch.nn.GELU(), torch.nn.Linear(16, 4))


def test_optimizer_routing_splits_2d_and_1d():
    import torch
    from replica_lab.optimizers import ReplicaOptimizer
    from replica_lab.vision_qualification import projector_optimizer_groups
    projector = _tiny_projector()
    train = {"lr_new": 1e-3, "momentum": 0.95, "ns_steps": 5,
             "block_size": 2048, "weight_decay": 0.01}
    groups = projector_optimizer_groups(projector, train, prefix="projector.")
    algorithms = {name: group["algorithm"] for group in groups
                  for name in group["param_names"]}
    for name, parameter in projector.named_parameters():
        expected = "muon8bit" if parameter.ndim == 2 else "adamw"
        assert algorithms["projector." + name] == expected
    optimizer = ReplicaOptimizer(groups, "muon8bit")
    loss = projector(torch.randn(2, 8)).sum()
    loss.backward()
    optimizer.step()
    assert set(optimizer.audit()["state_bytes_by_dtype"]) == {
        "torch.int8", "torch.float32"}


def test_adamw_control_routes_every_tensor_to_adamw():
    import torch
    from replica_lab.optimizers import ReplicaOptimizer
    projector = _tiny_projector()
    named = dict(projector.named_parameters())
    groups = [{
        "params": list(named.values()),
        "param_names": [f"vision_projector.{name}" for name in named],
        "group_name": "vision_projector_adamw_control",
        "algorithm": "adamw", "lr": 3e-4, "weight_decay": 0.01,
        "betas": (0.9, 0.999), "eps": 1e-8,
    }]
    optimizer = ReplicaOptimizer(groups, "adamw")
    assert all(group["algorithm"] == "adamw"
               for group in optimizer.param_groups)
    (projector(torch.randn(2, 8)).sum()).backward()
    optimizer.step()
    state = optimizer.state_dict()["state"]
    assert state
    for entry in state.values():
        assert {str(value.dtype) for value in entry.values()
                if hasattr(value, "dtype")} == {"torch.float32"}


def test_projector_parameter_count_matches_declaration():
    import torch
    projector = torch.nn.Sequential(
        torch.nn.LayerNorm(768), torch.nn.Linear(768, 4096),
        torch.nn.GELU(), torch.nn.Linear(4096, 2048))
    assert sum(parameter.numel() for parameter in projector.parameters()) \
        == PROJECTOR_PARAMETER_COUNT
    names = [name for name, _ in projector.named_parameters()]
    assert sorted("vision_projector." + name for name in names) == \
        sorted(PROJECTOR_TENSORS)


def test_fresh_seeded_initialization_is_identical_and_not_warm():
    import os
    import torch
    keys = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE",
            "PYTORCH_ENABLE_MPS_FALLBACK", "REPLICA_CAPACITY_GUARD")
    saved = {key: os.environ.get(key) for key in keys}
    try:
        from scripts.run_vision_curriculum import (
            initialize_projector, projector_state_sha256)
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    projector = _tiny_projector()
    first = initialize_projector(projector, 17)
    second = initialize_projector(projector, 17)
    assert first == second
    third = initialize_projector(projector, 18)
    assert third != first
    assert projector_state_sha256(projector) == third
    weights = [module.weight for module in projector
               if isinstance(module, torch.nn.Linear)]
    assert weights and all(not torch.equal(weight,
                                           torch.zeros_like(weight))
                           for weight in weights)


# ---------------------------------------------------------------------------
# Sensing leakage and generation validation
# ---------------------------------------------------------------------------


def test_sensing_inputs_never_contain_gold_labels(dataset):
    for row in dataset.training_examples():
        assert set(row) & {"answer", "label"} == set() or \
            row.get("answer") == row.get("response")
        assert not _question_leaks(row["question"], row["response"])
        assert row["response"].casefold() not in row["question"].casefold()
    for split in ("dev", "audit"):
        for case in dataset.generation_cases(split):
            assert "answer" not in case["question"]
            assert not _question_leaks(case["question"], case["label"])
    assert _question_leaks("Is the answer yes?", "yes")
    assert not _question_leaks("Is the answer yes?", "no")


def test_generation_rows_reject_malformed_and_success_claims():
    cases = [{"case_id": "c1"}, {"case_id": "c2"}]
    rows = [{"control": control, "case_id": case,
             "generated_text": "yes"}
            for control in CONTROLS for case in ("c1", "c2")]
    validate_generation_rows(rows, ("c1", "c2"))
    with pytest.raises(VisionCurriculumError):
        validate_generation_rows(rows + [rows[0]], ("c1", "c2"))
    with pytest.raises(VisionCurriculumError):
        validate_generation_rows(rows + [{"control": "oracle",
                                          "case_id": "c1",
                                          "generated_text": "yes"}],
                                 ("c1", "c2"))
    with pytest.raises(VisionCurriculumError):
        validate_generation_rows([{**rows[0], "success": True}]
                                 + rows[1:], ("c1", "c2"))
    with pytest.raises(VisionCurriculumError):
        validate_generation_rows(rows[:-1], ("c1", "c2"))


def test_control_assignment_is_deterministic_and_deranged():
    cases = [{"case_id": f"w{world}-p0-{side}", "world_id": f"w{world}",
              "task": "spatial"}
             for world in range(6) for side in ("a", "b")]
    first = assign_controls(cases, seed_offset=7919)
    second = assign_controls(cases, seed_offset=7919)
    assert first == second
    for case in cases:
        case_id = case["case_id"]
        assert first["wrong"][case_id] != case_id
        assert first["shuffled"][case_id] != case_id
        assert first["shuffled"][case_id].split("-p0-")[0] != \
            case["world_id"]


# ---------------------------------------------------------------------------
# Procedural scoring, unknown accuracy and caption metric
# ---------------------------------------------------------------------------


def _synthetic_cases():
    cases = [
        {"case_id": "c1", "world_id": "w1", "task": "spatial",
         "kind": None, "label": "yes"},
        {"case_id": "c2", "world_id": "w2", "task": "count",
         "kind": None, "label": "3"},
        {"case_id": "c3", "world_id": "w3", "task": "abstain",
         "kind": "absent_referent", "label": UNKNOWN_ANSWER},
        {"case_id": "c4", "world_id": "w4", "task": "abstain",
         "kind": "ambiguous_referent", "label": UNKNOWN_ANSWER},
    ]
    return cases


def _synthetic_generations(cases, *, correct=None, unknown_abstain=True):
    correct = correct or {case["case_id"]: case["label"] for case in cases}
    generations = []
    for case in cases:
        for control in CONTROLS:
            if control == "correct":
                if case["label"] == UNKNOWN_ANSWER and not unknown_abstain:
                    text = "yes"
                else:
                    text = correct[case["case_id"]]
            else:
                text = "no"
            generations.append({"control": control,
                                "case_id": case["case_id"],
                                "generated_text": text})
    return generations


def test_score_procedural_unknown_and_control_metrics():
    cases = _synthetic_cases()
    generations = _synthetic_generations(cases)
    score = score_procedural(
        cases, generations, seed=1, bootstrap_resamples=50,
        bootstrap_confidence=0.9)
    assert score["cases"] == 4
    assert score["controls"]["correct"]["accuracy"] == 1.0
    assert score["unknown"]["abstain_cases"] == 2
    assert score["unknown"]["abstention_accuracy"] == 1.0
    assert score["unknown"]["false_answer_on_abstain"] == 0.0
    assert score["unknown"]["false_abstention_on_answerable"] == 0.0
    for control in ("wrong", "shuffled", "noimage"):
        assert score["paired"][control]["correct_margin"] == 1.0
        assert "margin_bootstrap" in score["paired"][control]
    failing = _synthetic_generations(cases, unknown_abstain=False)
    broken = score_procedural(cases, failing, seed=1, bootstrap_resamples=20,
                              bootstrap_confidence=0.9,
                              gates=BASE_CONFIG["evaluation"]["gates"])
    assert broken["unknown"]["false_answer_on_abstain"] == 1.0
    assert "max_false_answer_on_abstain" in broken["gate_failures"]
    assert not broken["learned_quality_pass"]


def test_score_procedural_rejects_incomplete_or_success_rows():
    cases = _synthetic_cases()
    rows = _synthetic_generations(cases)
    with pytest.raises(VisionCurriculumError):
        score_procedural(cases, rows[:-1], seed=1, bootstrap_resamples=10,
                         bootstrap_confidence=0.9)
    bad = [{**row, "verdict": "accepted"} for row in rows]
    with pytest.raises(VisionCurriculumError):
        score_procedural(cases, bad, seed=1, bootstrap_resamples=10,
                         bootstrap_confidence=0.9)


def test_answer_normalisation_and_unknown_detection():
    assert answer_correct(" yes. ", "yes")
    assert answer_correct("Unknown", "unknown")
    assert answer_correct("cannot determine.", "unknown")
    assert not answer_correct("unknown", "yes")
    assert not answer_correct("yes", "unknown")
    assert is_unknown_answer("Unknown.")
    assert is_unknown_answer("cannot determine")
    assert not is_unknown_answer("3")


def test_caption_token_f1_and_loss_aggregation():
    assert caption_token_f1("a red car", "a red car") == 1.0
    assert caption_token_f1("a blue car", "a red car") < 1.0
    assert caption_token_f1("", "a red car") == 0.0
    labels = [{"case_id": "x", "answer": "a red car"},
              {"case_id": "y", "answer": "two dogs"}]
    generations = [{"case_id": "x", "generated_text": "a red car",
                    "response_loss": 0.5},
                   {"case_id": "y", "generated_text": "a dog",
                    "response_loss": 1.5}]
    score = score_caption(labels, generations)
    assert score["cases"] == 2
    assert score["teacher_forced_response_loss"] == 1.0
    assert 0.0 < score["token_f1"] < 1.0
    assert score["loss_cases"] == 2
    assert score["claim"] == \
        "narrow_alignment_metric_only_no_semantic_quality_claim"
    with pytest.raises(VisionCurriculumError):
        score_caption(labels, [{"case_id": "x", "generated_text": "a red car"}])


# ---------------------------------------------------------------------------
# Component-vs-baseline retention policy
# ---------------------------------------------------------------------------


def test_component_nonregression_is_separate_from_baseline_capability():
    gates = BASE_CONFIG["evaluation"]["gates"]
    passing = retention_policy(
        measured_correct=22, measured_cases=32,
        baseline_correct=22, baseline_cases=32,
        token_parity=True, gates=gates)
    assert passing["baseline_eligible"] is False
    assert passing["overall_baseline_eligible"] is False
    assert passing["baseline_failures"] == ["baseline_capability_ineligible"]
    assert passing["component_nonregression_pass"] is True
    assert passing["overall_eligible"] is False
    dropped = retention_policy(
        measured_correct=21, measured_cases=32,
        baseline_correct=22, baseline_cases=32,
        token_parity=True, gates=gates)
    assert dropped["component_nonregression_pass"] is False
    assert "max_text_retention_drop" in dropped["component_failures"]
    tokens = retention_policy(
        measured_correct=22, measured_cases=32,
        baseline_correct=22, baseline_cases=32,
        token_parity=False, gates=gates)
    assert tokens["component_nonregression_pass"] is False
    assert "text_tokens_changed" in tokens["component_failures"]


# ---------------------------------------------------------------------------
# Report validation: run, reload, score, finalize
# ---------------------------------------------------------------------------


def _run_report(protocol, *, arm="muon8bit", init_hash="a" * 64,
                schedule_sha256=None):
    schedule_sha256 = schedule_sha256 or protocol["schedule"]["schedule_sha256"]
    route = []
    for name in PROJECTOR_TENSORS:
        route.append({
            "name": name, "shape": [4096, 768] if name in
            PROJECTOR_2D_TENSORS else [768],
            "algorithm": ("muon8bit" if arm == "muon8bit"
                          and name in PROJECTOR_2D_TENSORS else "adamw"),
            "group_name": "vision_projector_muon8bit"
            if arm == "muon8bit" and name in PROJECTOR_2D_TENSORS
            else "vision_projector_adamw",
        })
    report = {
        "format": RUN_REPORT_FORMAT,
        "version": protocol["version"],
        "stage": "train",
        "arm": arm,
        "protocol_sha256": protocol_sha256(protocol),
        "data_pins": dict(protocol["data"]["manifest"]),
        "implementation_sha256": dict(protocol["implementation_sha256"]),
        "initialization": {
            "recipe": protocol["train"]["init_recipe"],
            "seed": int(protocol["train"]["init_seed"]),
            "projector_init_sha256": init_hash,
            "warm_start": None,
            "matches_forbidden_parent": False,
        },
        "trainable": {
            "names": sorted(PROJECTOR_TENSORS),
            "parameter_count": PROJECTOR_PARAMETER_COUNT,
            "frozen_names": ["base.model.embed_tokens.weight"],
            "trainable_count": len(PROJECTOR_TENSORS),
        },
        "gradient_audit": {
            "nonprojector_gradient_names": [],
            "projector_gradient_names": sorted(PROJECTOR_TENSORS),
            "backward_only_probe": True,
            "probe_applied_update": False,
            "probe_label": "backward_only_no_update",
            "loss": 1.0,
            "projector_gradients_finite": True,
            "projector_gradients_nonzero": True,
        },
        "optimizer_route": route,
        "optimizer_audit": {
            "optimizer": "muon8bit" if arm == "muon8bit" else "adamw",
            "parameter_dtype": "float32",
            "state_dtype": ("int8_momentum+float32_aux"
                            if arm == "muon8bit" else "float32"),
            "groups": ([
                {"group_name": "vision_projector_muon8bit",
                 "algorithm": "muon8bit",
                 "param_names": list(PROJECTOR_2D_TENSORS),
                 "lr": 0.0001},
                {"group_name": "vision_projector_adamw",
                 "algorithm": "adamw",
                 "param_names": list(PROJECTOR_1D_TENSORS),
                 "lr": 0.0001},
            ] if arm == "muon8bit" else [
                {"group_name": "vision_projector_adamw_control",
                 "algorithm": "adamw",
                 "param_names": list(PROJECTOR_TENSORS),
                 "lr": 0.0001},
            ]),
        },
        "schedule": {
            "schedule_sha256": schedule_sha256,
            "updates": protocol["schedule"]["updates"],
            "examples": protocol["schedule"]["examples"],
            "completed_updates": protocol["schedule"]["updates"],
            "probe_updates_per_candidate":
                protocol["schedule"]["probe"]["updates_per_candidate"],
        },
        "lr_selection": {
            "arm": arm, "split": "procedural_dev",
            "measure": "mean_dev_response_loss",
            "candidates": [0.0001, 0.0003],
            "selected_lr": 0.0001,
            "results": [
                {"lr": candidate, "dev_loss": 1.0 + candidate,
                 "init_sha256": init_hash, "dev_losses": [1.0],
                 "route": [], "optimizer_audit": {}}
                for candidate in (0.0001, 0.0003)
            ],
            "probe_case_ids": [],
        },
        "training": {
            "updates": protocol["schedule"]["updates"],
            "examples": protocol["schedule"]["examples"],
            "completed": True,
            "response_only_labels": True,
            "nonprojector_gradients_absent": True,
            "base_parameter_stamp_unchanged": True,
            "max_seconds": protocol["train"]["max_seconds"],
        },
        "retention": {
            "policy": "component_nonregression.v1",
            "baseline_correct": 22, "baseline_cases": 32,
            "measured_correct": 22, "measured_cases": 32,
            "measured_accuracy": 22 / 32,
            "overall_baseline_eligible": False,
            "component_nonregression_pass": True,
            "token_parity": True,
            "pre_sha256": "1" * 64,
            "post_sha256": "2" * 64,
        },
        "dev": {
            "cases": protocol["splits"]["dev"]["rows"],
            "generations_sha256": "b" * 64,
            "cases_sha256": "3" * 64,
            "seal_sha256": protocol["splits"]["dev"]["seal_sha256"],
            "dataset_dev_cases_sha256": "4" * 64,
            "response_loss": 1.25,
            "caption_cases_sha256": "5" * 64,
            "caption_labels_sha256": "6" * 64,
            "caption_generations_sha256": "7" * 64,
            "scored_splits": ["curriculum_dev", "diagnostic_dev",
                              "external_development"],
            "audit_touched": False,
        },
        "artifacts": {name: "9" * 64
                      for name in (
                          "training_trace.jsonl", "lr_probe.json",
                          "backward_probe.json", "dev_cases.jsonl",
                          "dev_generations.jsonl", "caption_cases.jsonl",
                          "caption_labels.jsonl", "caption_generations.jsonl",
                          "retention_pre.jsonl", "retention_post.jsonl",
                          "extension/manifest.json",
                          "extension/extensions.safetensors")},
        "extension": {
            "label": "final", "format": "replica.minicpm5.extensions.v4",
            "base_saved": False, "weights_sha256": "c" * 64,
            "receipt_sha256": "d" * 64, "manifest_digest": "e" * 64,
            "tensor_names": sorted(PROJECTOR_TENSORS),
        },
        "checks": {name: True for name in _required_run_checks()},
        "audit_scored": False,
        "external_final_consumed": False,
        "costs": {"downloads": 0, "workers": 1,
                  "peak_driver_bytes": 1024,
                  "updates": protocol["schedule"]["updates"]},
        "claims": {
            "perception_quality": False,
            "semantic_quality": False,
            "overall_baseline_eligible": False,
            "scope": "staged_curriculum_engineering_and_learned_quality_only",
        },
        "implementation_runtime": {"device": "mps"},
        "_run_report_sha256": "f" * 64,
    }
    return report


def test_run_report_validation_accepts_declared_evidence(prepared_protocol):
    report = _run_report(prepared_protocol)
    result = validate_run_report(report, prepared_protocol)
    assert result["engineering_pass"] is True
    assert result["engineering_failures"] == []
    for arm in ARMS:
        other = _run_report(prepared_protocol, arm=arm)
        validate_run_report(other, prepared_protocol)


def test_run_report_validation_rejects_premature_audit_and_warm_start(
        prepared_protocol):
    report = _run_report(prepared_protocol)
    report["audit_scored"] = True
    with pytest.raises(VisionCurriculumError):
        validate_run_report(report, prepared_protocol)
    report = _run_report(prepared_protocol)
    report["initialization"]["warm_start"] = {"path": "v1"}
    with pytest.raises(VisionCurriculumError):
        validate_run_report(report, prepared_protocol)
    report = _run_report(prepared_protocol)
    report["checks"]["strict_extension_export"] = False
    with pytest.raises(VisionCurriculumError):
        validate_run_report(report, prepared_protocol)
    report = _run_report(prepared_protocol)
    report["claims"]["semantic_quality"] = True
    with pytest.raises(VisionCurriculumError):
        validate_run_report(report, prepared_protocol)


def test_lr_reports_require_identical_initialization_and_schedule(
        prepared_protocol):
    left = _run_report(prepared_protocol, arm="muon8bit")
    right = _run_report(prepared_protocol, arm="adamw_control")
    validate_lr_reports(left, right)
    right["initialization"]["projector_init_sha256"] = "9" * 64
    with pytest.raises(VisionCurriculumError):
        validate_lr_reports(left, right)
    right = _run_report(prepared_protocol, arm="adamw_control")
    right["schedule"]["schedule_sha256"] = "8" * 64
    with pytest.raises(VisionCurriculumError):
        validate_lr_reports(left, right)


def test_reload_report_validation_requires_bitwise_equality(
        prepared_protocol):
    run = _run_report(prepared_protocol)
    reload = {
        "format": "replica.vision_curriculum.reload_report.v1",
        "version": prepared_protocol["version"],
        "run_report_sha256": run["_run_report_sha256"],
        "protocol_sha256": protocol_sha256(prepared_protocol),
        "arm": run["arm"],
        "zero_updates": True,
        "extension_weights_sha256": run["extension"]["weights_sha256"],
        "dev_generations_equal": True,
        "retention_generations_equal": True,
        "extension_unchanged": True,
        "dev_cases_sha256": run["dev"]["cases_sha256"],
        "dataset_dev_cases_sha256": run["dev"]["dataset_dev_cases_sha256"],
        "dev_generations_sha256": run["dev"]["generations_sha256"],
        "retention_sha256": run["retention"]["post_sha256"],
        "checks": {"zero_updates": True, "extension_unchanged": True,
                   "dev_equal": True, "retention_equal": True,
                   "strict_export": True},
        "claims": {"bitwise_reload": True, "quality_claim": False},
        "guard": {}, "packages": {},
    }
    validate_reload_report(reload, run, prepared_protocol)
    reload["dev_generations_equal"] = False
    with pytest.raises(VisionCurriculumError):
        validate_reload_report(reload, run, prepared_protocol)
    reload["dev_generations_equal"] = True
    reload["zero_updates"] = False
    with pytest.raises(VisionCurriculumError):
        validate_reload_report(reload, run, prepared_protocol)


def test_score_report_separates_engineering_quality_and_baseline(
        prepared_protocol):
    run = _run_report(prepared_protocol)
    cases = _synthetic_cases()
    generations = _synthetic_generations(cases)
    labels = [{"case_id": "cap", "answer": "a red car"}]
    caption_rows = [{"case_id": "cap", "generated_text": "a red car",
                     "response_loss": 0.5}]
    report = assemble_score_report(
        protocol=prepared_protocol, run_report=run, cases=cases,
        generations=generations, caption_labels=labels,
        caption_generations=caption_rows,
        retention={"measured_correct": 22, "token_parity": True})
    validate_score_report(report)
    assert report["engineering"]["pass"] is True
    assert report["eligibility"]["overall_baseline_eligible"] is False
    assert report["eligibility"]["baseline_eligible"] is False
    assert report["eligibility"]["component_nonregression_pass"] is True
    assert report["learned_quality"]["external_development_caption"][
        "claim"] == \
        "narrow_alignment_metric_only_no_semantic_quality_claim"
    assert report["audit_scored"] is False
    bad = copy.deepcopy(report)
    bad["claims"]["semantic_quality"] = True
    with pytest.raises(VisionCurriculumError):
        validate_score_report(bad)
    bad = copy.deepcopy(report)
    bad["audit_scored"] = True
    with pytest.raises(VisionCurriculumError):
        validate_score_report(bad)


def _comparison(protocol, *, losses=None):
    losses = losses or {"muon8bit": 1.0, "adamw_control": 1.2}
    reports = {}
    for position, arm in enumerate(ARMS):
        reports[arm] = {
            "run_report_sha256": ("a" if arm == "muon8bit" else "b") * 64,
            "extension_weights_sha256": ("c" if arm == "muon8bit" else "d") * 64,
            "projector_init_sha256": "e" * 64,
            "schedule_sha256": protocol["schedule"]["schedule_sha256"],
            "updates": protocol["schedule"]["updates"],
            "examples": protocol["schedule"]["examples"],
            "probe_updates_per_candidate":
                protocol["schedule"]["probe"]["updates_per_candidate"],
            "training_max_seconds": protocol["train"]["max_seconds"],
            "dev_response_loss": losses[arm],
        }
    return {"rule": {"measure": "mean_dev_response_loss",
                     "tie_break": "declared_arm_order",
                     "both_arms_required": True},
            "order": list(ARMS), "reports": reports,
            "selected_arm": "muon8bit"}


def test_global_reservation_and_finalize_receipt(tmp_path, prepared_protocol):
    project = tmp_path / "project"
    protocol_sha = protocol_sha256(prepared_protocol)
    data_sha = prepared_protocol["data"]["manifest"]["sha256"]
    paths = require_global_finalize_allowed(
        protocol_sha, project_root=project, data_manifest_sha256=data_sha)
    comparison = _comparison(prepared_protocol)
    reservation = reserve_finalize(
        prepared_protocol, protocol_sha, project_root=project,
        comparison=comparison, selected_arm="muon8bit",
        dev_report_sha256="a" * 64, score_report_sha256="f" * 64)
    with pytest.raises(VisionCurriculumError):
        require_global_finalize_allowed(
            protocol_sha, project_root=project, data_manifest_sha256=data_sha)
    with pytest.raises(VisionCurriculumError):
        reserve_finalize(
            prepared_protocol, protocol_sha, project_root=project,
            comparison=comparison, selected_arm="muon8bit",
            dev_report_sha256="a" * 64, score_report_sha256="f" * 64)
    receipt = build_finalize_receipt(
        protocol=prepared_protocol, comparison=comparison,
        reservation=reservation, selected_arm="muon8bit",
        selected_checkpoint={"weights_sha256": "c" * 64,
                             "frozen_at_utc": "2026-10-25T00:00:00Z",
                             "frozen_before_audit": True},
        dev_report_sha256="a" * 64, score_report_sha256="f" * 64,
        audit={"scored_once": True,
               "seal_sha256": prepared_protocol["splits"]["audit"][
                   "seal_sha256"],
               "cases": prepared_protocol["splits"]["audit"]["rows"],
               "cases_sha256": "3" * 64,
               "generations_sha256": "1" * 64,
               "score": {
                   "cases": prepared_protocol["splits"]["audit"]["rows"],
                   "controls": {"correct": {"accuracy": 0.0,
                                            "unique_worlds": 1}},
                   "unknown": {"false_abstention_on_answerable": 0.0,
                               "abstention_accuracy": 0.0,
                               "false_answer_on_abstain": 1.0},
                   "paired": {
                       "wrong": {"correct_margin": 0.0},
                       "shuffled": {"correct_margin": 0.0},
                       "noimage": {"correct_margin": 0.0}},
                   "learned_quality_pass": False,
                   "gate_failures": [
                       "min_correct_accuracy", "min_unique_families",
                       "min_correct_margin_wrong",
                       "min_correct_margin_shuffled",
                       "min_correct_margin_noimage", "min_unknown_accuracy",
                       "max_false_answer_on_abstain"]},
               "learned_quality_pass": False,
               "gate_failures": [
                   "min_correct_accuracy", "min_unique_families",
                   "min_correct_margin_wrong", "min_correct_margin_shuffled",
                   "min_correct_margin_noimage", "min_unknown_accuracy",
                   "max_false_answer_on_abstain"]},
        external_final={"consumed_once": True,
                        "seal_sha256": prepared_protocol["splits"]
                        ["external_final"]["seal_sha256"],
                        "rows": prepared_protocol["splits"]["external_final"]
                        ["rows"],
                        "cases_sha256": "4" * 64,
                        "generations_sha256": "2" * 64,
                        "caption": {"cases": 8, "loss_cases": 8,
                                    "token_f1": 0.5,
                                    "teacher_forced_response_loss": 1.0}})
    validate_finalize_receipt(receipt, prepared_protocol, "a" * 64)
    write_finalize_receipt(paths["receipt"], receipt)
    with pytest.raises(VisionCurriculumError):
        write_finalize_receipt(paths["receipt"], receipt)
    with pytest.raises(VisionCurriculumError):
        require_global_finalize_allowed(
            protocol_sha, project_root=project, data_manifest_sha256=data_sha)
    bad = copy.deepcopy(receipt)
    bad["selected_arm"] = "adamw_control"
    with pytest.raises(VisionCurriculumError):
        validate_finalize_receipt(bad, prepared_protocol, "a" * 64)
    bad = copy.deepcopy(receipt)
    bad["external_final"]["caption"]["loss_cases"] = 7
    with pytest.raises(VisionCurriculumError):
        validate_finalize_receipt(bad, prepared_protocol, "a" * 64)
    bad = copy.deepcopy(receipt)
    bad["audit"]["cases"] = 4
    with pytest.raises(VisionCurriculumError):
        validate_finalize_receipt(bad, prepared_protocol, "a" * 64)



def test_run_report_requires_artifacts_dev_loss_and_group_lrs(prepared_protocol):
    report = _run_report(prepared_protocol)
    del report["artifacts"]
    with pytest.raises(VisionCurriculumError):
        validate_run_report(report, prepared_protocol)
    report = _run_report(prepared_protocol)
    del report["dev"]["response_loss"]
    with pytest.raises(VisionCurriculumError):
        validate_run_report(report, prepared_protocol)
    report = _run_report(prepared_protocol)
    report["optimizer_audit"]["groups"][1]["lr"] = 0.0002
    with pytest.raises(VisionCurriculumError):
        validate_run_report(report, prepared_protocol)
    report = _run_report(prepared_protocol)
    report["optimizer_audit"]["groups"][0]["lr"] = 0.0003
    with pytest.raises(VisionCurriculumError):
        validate_run_report(report, prepared_protocol)
    report = _run_report(prepared_protocol)
    report["training"]["max_seconds"] = 3600
    with pytest.raises(VisionCurriculumError):
        validate_run_report(report, prepared_protocol)


def test_score_report_rejects_self_contradictory_acceptance(prepared_protocol):
    run = _run_report(prepared_protocol)
    cases = _synthetic_cases()
    report = assemble_score_report(
        protocol=prepared_protocol, run_report=run, cases=cases,
        generations=_synthetic_generations(cases),
        caption_labels=[{"case_id": "cap", "answer": "a red car"}],
        caption_generations=[{"case_id": "cap", "generated_text": "a red car",
                              "response_loss": 0.5}],
        retention={"measured_correct": 22, "token_parity": True})
    validate_score_report(report)
    bad = copy.deepcopy(report)
    bad["learned_quality"]["pass"] = True
    bad["learned_quality"]["failures"] = ["min_correct_accuracy"]
    with pytest.raises(VisionCurriculumError):
        validate_score_report(bad)
    bad = copy.deepcopy(report)
    bad["eligibility"]["curriculum_qualified"] = True
    bad["learned_quality"]["pass"] = False
    bad["learned_quality"]["failures"] = ["min_correct_accuracy"]
    bad["eligibility"]["learned_quality_pass"] = False
    with pytest.raises(VisionCurriculumError):
        validate_score_report(bad)
    bad = copy.deepcopy(report)
    bad["learned_quality"]["retention"]["measured_accuracy"] = 0.9
    with pytest.raises(VisionCurriculumError):
        validate_score_report(bad)
    bad = copy.deepcopy(report)
    bad["learned_quality"]["external_development_caption"]["loss_cases"] = 0
    with pytest.raises(VisionCurriculumError):
        validate_score_report(bad)


def _fake_generation_session():
    import torch
    class Tokenizer:
        eos_token_id = 0

        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt,
                                enable_thinking):
            return messages[0]["content"]

        def __call__(self, text, *, add_special_tokens):
            return {"input_ids": [5]}

        def decode(self, ids, *, skip_special_tokens):
            return "no"

    class Sense:
        prompt_ids = torch.tensor([[5]])

    class Frame:
        sense = Sense()

    class Request:
        frame = Frame()

        def release(self):
            return None

    class Session:
        device = "cpu"

        def prepare(self, **kwargs):
            return Request()

        def generate(self, request, *, max_new_tokens):
            return torch.tensor([[5, 6]])

    return Session(), Tokenizer()


def test_noimage_generation_records_null_image_hash():
    import os
    keys = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE",
            "PYTORCH_ENABLE_MPS_FALLBACK", "REPLICA_CAPACITY_GUARD")
    saved = {key: os.environ.get(key) for key in keys}
    try:
        from scripts.run_vision_curriculum import generate_row
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    session, tokenizer = _fake_generation_session()
    case = {"case_id": "c1", "image_sha256": "a" * 64}
    row = generate_row(
        session, tokenizer, control="noimage", case=case,
        question="Question: test?", visual=None, scope=None,
        maximum_prompt=32, maximum_new=8, image_case_id=None)
    assert row["image_sha256"] is None
    row = generate_row(
        session, tokenizer, control="correct", case=case,
        question="Question: test?", visual=None, scope=None,
        maximum_prompt=32, maximum_new=8, image_case_id="c1",
        image_sha256="a" * 64)
    assert row["image_sha256"] == "a" * 64


def test_muon_auxiliary_lr_stays_fixed_while_matrix_lr_is_selected():
    import os
    import torch
    keys = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE",
            "PYTORCH_ENABLE_MPS_FALLBACK", "REPLICA_CAPACITY_GUARD")
    saved = {key: os.environ.get(key) for key in keys}
    try:
        from scripts.run_vision_curriculum import build_optimizer
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    projector = _tiny_projector()

    class Fake:
        vision_projector = projector

    config = small_config()
    optimizer, route, audit = build_optimizer(
        Fake(), "muon8bit", 0.0003, config)
    lrs = {group["group_name"]: group["lr"]
           for group in optimizer.param_groups}
    assert lrs["vision_projector_muon8bit"] == 0.0003
    assert lrs["vision_projector_adamw"] == 0.0001
    audit_lrs = {group["group_name"]: group["lr"]
                 for group in audit["groups"]}
    assert audit_lrs["vision_projector_adamw"] == 0.0001
    assert audit_lrs["vision_projector_muon8bit"] == 0.0003


def test_retention_pin_verification_detects_changed_bytes(tmp_path):
    import shutil
    protocol = copy.deepcopy(BASE_CONFIG)
    prompts = ROOT / BASE_CONFIG["retention"]["prompts"]["path"]
    labels = ROOT / BASE_CONFIG["retention"]["labels"]["path"]
    shutil.copy(prompts, tmp_path / "prompts.jsonl")
    shutil.copy(labels, tmp_path / "labels.jsonl")
    value = {
        "retention": {
            "prompts": {"path": "prompts.jsonl",
                        "sha256": BASE_CONFIG["retention"]["prompts"]["sha256"]},
            "labels": {"path": "labels.jsonl",
                       "sha256": BASE_CONFIG["retention"]["labels"]["sha256"]},
        }
    }
    assert verify_retention_pins(value, tmp_path)
    with (tmp_path / "prompts.jsonl").open("ab") as stream:
        stream.write(b"\n")
    with pytest.raises(VisionCurriculumError):
        verify_retention_pins(value, tmp_path)

# ---------------------------------------------------------------------------
# Prepared protocol and data pins (repository deliverable)
# ---------------------------------------------------------------------------


def test_prepared_protocol_preregisters_every_required_field(
        prepared_protocol):
    protocol = prepared_protocol
    assert protocol["format"] == "replica.vision_curriculum.protocol.v1"
    assert protocol["claims"] == {
        "perception_quality": False,
        "semantic_quality": False,
        "overall_baseline_eligible": False,
        "component_nonregression_separate": True,
        "scope": "staged_curriculum_engineering_and_learned_quality_only",
    }
    assert protocol["audit_policy"]["score_once"] is True
    assert protocol["audit_policy"]["requires_frozen_dev_report_sha256"] is True
    assert protocol["audit_policy"]["external_final_untouched"] is True
    assert protocol["resource_policy"]["exclusive_mps_worker"] is True
    assert protocol["resource_policy"]["workers"] == 1
    assert protocol["resource_policy"]["cpu_fallback"] is False
    assert protocol["resource_policy"]["driver_limit_bytes"] == 50 * 1024 ** 3
    assert protocol["resource_policy"]["sustained_swap_growth_bytes"] == \
        512 * 1024 ** 2
    assert protocol["resource_policy"]["minimum_host_free_percent"] == 15
    assert protocol["trainable_groups"]["parameter_count"] == 11_542_016
    assert protocol["train"]["warm_start"] is None
    assert protocol["train"]["init_seed"] == 17
    assert protocol["schedule"]["updates"] == 128
    assert protocol["schedule"]["examples"] == 512
    assert protocol["schedule"]["probe"]["updates_per_candidate"] == 16
    assert protocol["schedule"]["probe"]["total_probe_updates"] == 64
    assert protocol["schedule"]["probe"]["total_probe_examples"] == 128
    assert protocol["schedule"]["arm_selection"]["both_arms_required"] is True
    assert protocol["splits"]["audit"]["usage"] == "scored_once_after_freeze"
    assert protocol["splits"]["external_final"]["rows"] == 8
    assert protocol["external"]["consumed"] is False
    assert protocol["evaluation"]["controls"] == list(CONTROLS)
    assert protocol["evaluation"]["bootstrap"]["cluster"] == "world"
    assert protocol["retention"]["policy"] == "component_nonregression.v1"
    assert protocol["retention"]["require_token_parity"] is True
    assert protocol["checkpoint_policy"]["probe_checkpoints"] is False
    assert protocol["checkpoint_policy"]["base_copies"] == 0
    assert protocol["limits"]["max_driver_bytes"] == 50 * 1024 ** 3
    assert protocol["limits"]["max_updates"] == 320
    assert protocol["action_limits"]["host_actions"] == "none"
    assert protocol["action_limits"]["success_claims"] == "none"
    assert len(protocol["stop_conditions"]) >= 5


def test_prepared_data_pins_and_implementation_pins(prepared_protocol):
    verify_data_pins(prepared_protocol["data"],
                     PREPARED)
    verify_implementation_pins(prepared_protocol, ROOT)
    manifest = json.loads((PREPARED / "manifest.json").read_text())
    assert manifest["format"] == "replica.vision_curriculum_manifest.v1"
    assert manifest["status"] == "staged_not_admitted"
    assert manifest["counts"]["audit_cases"] == 120
    assert manifest["counts"]["dev_cases"] == 240
    assert manifest["counts"]["train_examples"] == 279
    assert manifest["unknown_coverage"]["audit"] == {
        kind: 8 for kind in ABSTAIN_KINDS}
    assert manifest["arms"]["equal_compute"]["updates_per_arm"] == 128
    audit_manifest = json.loads(
        (PREPARED / "procedural/audit_manifest.json").read_text())
    audit_pool = {tuple(item) for item in audit_manifest["descriptor_pool"]}
    assert audit_pool.isdisjoint(AUDIT_FORBIDDEN_V2_DESCRIPTORS)
    assert audit_manifest["seal_sha256"] == manifest["seals"]["audit"]


def test_prepared_audit_seal_changes_when_audit_bytes_change(tmp_path):
    import shutil
    root = tmp_path / "audit_tamper"
    shutil.copytree(PREPARED, root)
    path = root / "stage_b/audit/model_rows.jsonl"
    data = path.read_bytes()
    assert b'"question":"' in data
    path.write_bytes(data.replace(b'"question":"', b'"question":"X', 1))
    with pytest.raises(VisionCurriculumError):
        audit_dataset(root)


def test_no_final_split_rows_copied_into_prepared_curriculum(prepared_protocol):
    external = load_config(ROOT / "configs/vision_curriculum_v1.json")[
        "external"]
    final_manifest = json.loads((ROOT / external["root"]
                                 / external["manifests"]
                                 ["final_eval_untouched"]["path"]).read_text())
    final_ids = {row["row_id"] for row in final_manifest["rows"]}
    managed = []
    for relative in ("stage_a/train/labels.jsonl",
                     "stage_a/development/labels.jsonl",
                     "stage_b/diagnostic_train/labels.jsonl",
                     "stage_b/diagnostic_dev/labels.jsonl",
                     "stage_b/train/labels.jsonl",
                     "stage_b/dev/labels.jsonl",
                     "stage_b/audit/labels.jsonl"):
        managed.extend(json.loads(line)["case_id"]
                       for line in (PREPARED / relative).read_text(
                           encoding="utf-8").splitlines() if line.strip())
    assert final_ids.isdisjoint(managed)
    assert len([case for case in managed if case.startswith("liuhaotian")]) == 22

"""CPU-only contracts for the smaller-step launch-readiness declaration.

The readiness checker may observe an authorization receipt and a committed
protocol, but it can never mint authorization, load a model or run a launch.
These tests prove the assessment is deterministic and current, that the exact
limits, gates and raw-generation contract are preserved, that commit binding is
recorded as facts rather than inferred, and that tampering or staleness is
refused.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from test_vision_smaller_step_launch_packet_v1 import _populate_clean_root

ROOT = Path(__file__).resolve().parents[1]
CHECKER = ROOT / "scripts/check_vision_smaller_step_launch_readiness_v1.py"
DECLARATION = "configs/vision_smaller_step_launch_readiness_v1.json"
REPORT = ("reports/vision_smaller_step_launch_readiness_v1_20261027/"
          "readiness.json")
REVIEWED_COMMIT = "4c5a84851009898bc876c0b8aa3ccb596e04037d"
CONFIG = "configs/vision_smaller_step_lr3e5_v1.json"
CONFIG_SHA256 = "931851ecc954cea91d97aa7a087c1a2bef528172379b06d453e6bec93ef4abad"
PACKET = ("reports/vision_smaller_step_launch_packet_v1_20261027/"
          "launch_packet.json")
PACKET_FILE_SHA256 = (
    "e19e5995f90cab2c54033b2673e583761c9148697fce1eaa200c79bc8f54af6c")
PACKET_CANONICAL_SHA256 = (
    "9be3a8d2edc7d0b4d819197e22cd316a82affd7b83345cc95c2b2ef7d591a770")
UNSATISFIED_NOW = [
    "authorization_receipt_issued",
    "protocol_commit_recorded",
    "review_launch_gate_passed",
    "exclusive_worker_and_capacity_guard",
    "native_execution_within_declared_budget",
]


def _module():
    spec = importlib.util.spec_from_file_location(
        "vision_smaller_step_launch_readiness_for_tests", CHECKER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def checker():
    return _module()


@pytest.fixture(scope="module")
def clean_root(tmp_path_factory):
    # The launch packet pins the shared fixture module's own bytes, so its
    # CONFIG_INPUTS cannot be extended without invalidating the checked-in
    # packet and readiness report.  Add the declaration and repository
    # metadata this assessment binds on top of the shared root.
    root = _populate_clean_root(
        tmp_path_factory.mktemp("vision_smaller_step_launch_readiness_root"))
    declaration = root / DECLARATION
    declaration.parent.mkdir(parents=True, exist_ok=True)
    if not declaration.exists() and not declaration.is_symlink():
        declaration.symlink_to(ROOT / DECLARATION)
    repository = ROOT / ".git"
    if repository.is_dir() and not (root / ".git").exists() \
            and not (root / ".git").is_symlink():
        (root / ".git").write_text(f"gitdir: {repository}\n")
    return root


@pytest.fixture(scope="module")
def report(checker, clean_root):
    return checker.assess(root=clean_root)


@pytest.fixture(scope="module")
def declaration(checker):
    return checker._load_declaration(ROOT / DECLARATION)


def test_assessment_is_deterministic_current_and_preparation_only(
        checker, report, clean_root):
    rebuilt = checker.assess(root=clean_root)
    assert checker.report_bytes(report) == checker.report_bytes(rebuilt)
    assert report["format"] == checker.FORMAT
    assert report["deterministic"] is True
    assert report["native_execution"] is False
    assert report["model_loads"] == 0
    assert report["authorization_created"] is False
    assert report["decision"]["preparation_complete"] is True
    assert report["decision"]["launch_allowed"] is False
    assert report["decision"]["native_execution"] is False
    assert report["decision"]["model_loads"] == 0
    assert report["decision"]["authorization_created"] is False
    assert report["readiness_sha256"] == checker._readiness_sha256(report)
    checked_in = ROOT / REPORT
    assert checked_in.is_file()
    assert checked_in.read_bytes() == checker.report_bytes(report)
    assert checker.verify_report(report=report, root=clean_root) == report


def test_subject_binds_the_exact_protocol_and_packet(checker, report):
    protocol = ROOT / CONFIG
    packet_path = ROOT / PACKET
    packet = json.loads(packet_path.read_text())
    assert _sha256(protocol) == CONFIG_SHA256
    assert report["subject"]["protocol_sha256"] == CONFIG_SHA256
    assert report["subject"]["packet_file_sha256"] == (
        _sha256(packet_path))
    assert report["subject"]["packet_file_sha256"] == PACKET_FILE_SHA256
    assert report["subject"]["packet_canonical_sha256"] == (
        packet["packet_sha256"])
    assert report["subject"]["packet_canonical_sha256"] == (
        PACKET_CANONICAL_SHA256)
    assert report["subject"]["packet_current"] is True
    assert report["subject"]["lr"] == 3e-05
    assert report["subject"]["completed_lr"] == 3e-04
    assert report["subject"]["update_batch_example_ids"] == [
        "caption:vba2-w013-p0-a",
        "caption:vba2-w025-p0-a",
        "answer:vba2-w000-p0-a",
        "answer:vba2-w036-p0-b",
    ]
    assert packet["native_execution"] is False
    assert packet["model_loads"] == 0
    assert packet["authorization_created"] is False


def test_limits_gates_and_raw_generation_are_preserved(checker, report):
    preserved = report["preserved_contract"]
    assert preserved["limits"] == {
        "decoder_forward_calls": 4824,
        "visual_observation_calls": 168,
        "backward_calls": 4,
        "optimizer_steps": 1,
        "wall_seconds": 1200,
        "artifact_bytes": 33554432,
        "artifact_includes_final_report": True,
        "model_loads": 1,
        "downloads": 0,
        "retries": 0,
        "workers": 1,
        "cpu_fallback": False,
    }
    assert preserved["campaign_cumulative"] == {
        "decoder_forward_calls": 4824,
        "visual_observation_calls": 168,
        "visual_encoder_calls": 168,
        "visual_encoder_cache_misses": 168,
        "visual_forward_calls": 4824,
        "projector_passes": 4824,
        "backward_calls": 4,
        "optimizer_steps": 1,
        "generated_tokens": 3800,
    }
    optimization = preserved["optimization_gate"]
    assert optimization["max_update_batch_mean_nll_delta"] == 0.25
    assert optimization["max_wider_answer_mean_nll_delta"] == 0.25
    assert optimization["max_wider_caption_mean_nll_delta"] == 0.25
    assert optimization["optimizer_owned_tensors"] == 6
    assert optimization["backbone_gradient_tensors"] == 0
    assert optimization["projector_tensors_only_changed"] is True
    assert optimization["status_on_fail"] == "optimization_gate_failed"
    functional = preserved["functional_fit_gate"]
    assert functional["answer_correct_min"] == 1
    assert functional["answer_correct_denominator"] == 14
    assert functional["caption_token_f1_gain_min"] == 0.1
    assert functional["caption_semantic_f1_min"] == 0.05
    assert functional["correct_vs_sibling_accuracy_margin_min"] == 0.05
    assert functional["correct_vs_shuffled_accuracy_margin_min"] == 0.05
    assert functional["correct_vs_noimage_accuracy_margin_min"] == 0.05
    assert functional["eos_only_rate_max"] == 0.5
    assert functional["whitespace_only_rate_max"] == 0.1
    assert functional["development_on_fail"] == "closed"
    assert preserved["decision"]["requires_all_gates"] is True
    assert preserved["decision"]["development_authorized_here"] is False
    assert preserved["decision"]["heldout_authorized"] is False
    assert preserved["decision"]["v0_authorized"] is False
    assert preserved["decision"]["promotion"] is False
    assert preserved["decision"]["retry_authorized"] is False
    raw = preserved["raw_generation"]
    assert raw["controls"] == ["correct", "sibling", "shuffled", "noimage"]
    assert raw["generation_roles"] == ["answer", "caption"]
    assert raw["readouts"] == ["prefix_0", "prefix_1"]
    assert raw["expected_generated_tokens"] == 3800
    assert raw["empty_text_policy"] == (
        "separate_eos_only_from_whitespace_only")
    assert raw["eos_only_not_counted_correct"] is True
    assert raw["whitespace_only_not_counted_correct"] is True
    output = "runs/vision_smaller_step_lr3e5_v1_20261027/readouts"
    expected_paths = {
        f"{output}/prefix_0/readout.json",
        f"{output}/prefix_0/records.jsonl",
        f"{output}/prefix_0/train_answer_generations.jsonl",
        f"{output}/prefix_0/train_caption_generations.jsonl",
        f"{output}/prefix_1/readout.json",
        f"{output}/prefix_1/records.jsonl",
        f"{output}/prefix_1/train_answer_generations.jsonl",
        f"{output}/prefix_1/train_caption_generations.jsonl",
        "runs/vision_smaller_step_lr3e5_v1_20261027/readouts/"
        "update_batch_pre/records.jsonl",
        "runs/vision_smaller_step_lr3e5_v1_20261027/readouts/"
        "update_batch_post/records.jsonl",
    }
    assert set(raw["required_paths"]) == expected_paths


def test_commit_binding_is_recorded_but_unbound_in_any_receipt(report):
    binding = report["commit_binding"]
    assert binding["reviewed_commit"] == REVIEWED_COMMIT
    assert binding["config_introducing_commit"] == (
        "46af5dd463e03d723f37afee8347a328db5302df")
    assert binding["config_blob_sha256"] == CONFIG_SHA256
    for entry in binding["reviewed_paths"]:
        assert entry["matches"] is True
        assert entry["sha256"] == entry["commit_blob_sha256"]
        assert entry["sha256"] == _sha256(ROOT / entry["path"])
    assert {entry["path"] for entry in binding["reviewed_paths"]} >= {
        CONFIG, PACKET, "scripts/vision_smaller_step_v1.py",
        "scripts/build_vision_smaller_step_launch_packet_v1.py",
        "tests/test_vision_smaller_step_v1.py",
        "tests/test_vision_smaller_step_launch_packet_v1.py",
    }
    assert binding["receipt_protocol_commit"] is None
    assert binding["commit_bound"] is False
    assert binding["recorded_in_receipt"] is False


def test_authorization_is_absent_and_never_minted(checker, report):
    authorization = report["authorization"]
    assert authorization["present"] is False
    assert authorization["status"] == "absent"
    assert authorization["receipt_sha256"] is None
    assert authorization["protocol_commit"] is None
    assert authorization["commit_bound"] is False
    assert authorization["native_execution_authorized"] is False
    assert authorization["violations"] == ["authorization_receipt_missing"]
    assert not (ROOT / authorization["path"]).exists()
    assert report["authorization_created"] is False
    requirements = report["remaining_requirements"]
    assert [item["id"] for item in requirements] == (
        checker.EXPECTED_REQUIREMENT_IDS)
    unsatisfied = [item["id"] for item in requirements
                   if not item["satisfied"]]
    assert unsatisfied == UNSATISFIED_NOW
    fresh = next(item for item in requirements
                 if item["id"] == "fresh_output_available")
    assert fresh["satisfied"] is True
    assert all(item["blocking"] is True for item in requirements)
    assert all(item["satisfied_by_readiness"] is False
               for item in requirements)
    assert report["launch"]["refusals"] == [
        "packet_is_not_an_authorization_receipt",
        "packet_is_preparation_only",
        "authorization_receipt_missing",
        "protocol_commit_unbound",
        "commit_binding_unverified",
        "native_execution_not_authorized",
    ]
    assert report["launch"]["disposition"] == "review_only_launch_refused"
    inspected = report["authorization"]["inspected_existing_receipts"]
    assert len(inspected) == 3
    assert {entry["path"] for entry in inspected} == {
        "reports/remediation_20261027/"
        "vision_learnability_step1_authorization.json",
        "reports/remediation_20261027/"
        "vision_learnability_step1_authorization_attempt02.json",
        "reports/remediation_20261027/"
        "vision_learnability_step1_authorization_attempt03.json",
    }
    for entry in inspected:
        assert entry["protocol_name"] == (
            "vision_learnability_step1_reproduction_v1")
        assert entry["binds_subject_config"] is False
        assert entry["config_sha256"] != report["subject"]["protocol_sha256"]
        assert entry["config_sha256"] in (
            "8487e1c4cc9b4b818b7d46513daa5c149bfc1e2a8af2bd8dcdac787d3f2827b8",
            "b7bb9ded894ba053fa17f380a408697317b8d6862a4cac53c8b8f6ec80b1e32a",
        )
        assert entry["protocol_commit"] in (
            "dfe39a077269e11f328895c9a20ff2e3cca337a3",
            "ca8669b90ce394581cdf3ecbf6709226439898c9",
        )
    assert any(check["id"] == "existing_authorizations_inspected"
               for check in report["checks"])


def test_valid_receipt_is_observed_without_becoming_launch_authority(
        checker, declaration, tmp_path):
    receipt = {
        "format": declaration["authorization"]["format"],
        "status": "authorized",
        "protocol_name": "vision_smaller_step_lr3e5_v1",
        "protocol_path": CONFIG,
        "protocol_commit": REVIEWED_COMMIT,
        "config_sha256": CONFIG_SHA256,
        "native_execution_authorized": True,
        "dev_authorized": False,
        "heldout_authorized": False,
        "v0_authorized": False,
        "semantic_audit_authorized": False,
        "promotion_authorized": False,
        "retry_authorized": False,
        "issued_at_utc": "2026-10-27T00:00:00+00:00",
    }
    target = tmp_path / declaration["authorization"]["path"]
    target.parent.mkdir(parents=True)
    target.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    state = checker._authorization_state(
        declaration, {"name": "vision_smaller_step_lr3e5_v1"}, tmp_path,
        git_root=ROOT)
    assert state["present"] is True
    assert state["status"] == "present_valid"
    assert state["violations"] == []
    assert state["commit_bound"] is True
    assert state["config_sha256_bound"] is True
    assert state["native_execution_authorized"] is True
    assert state["closed_roles_respected"] is True
    launch = checker._launch_state(declaration, state, tmp_path)
    assert launch["output_exists"] is False
    assert "native_execution_not_authorized" not in launch["refusals"]
    assert "packet_is_not_an_authorization_receipt" in launch["refusals"]
    assert launch["disposition"] == "review_only_launch_refused"


def test_invalid_receipts_are_recorded_but_never_upgraded(
        checker, declaration, tmp_path):
    base = {
        "format": declaration["authorization"]["format"],
        "status": "authorized",
        "protocol_name": "vision_smaller_step_lr3e5_v1",
        "protocol_path": CONFIG,
        "protocol_commit": "0" * 40,
        "config_sha256": CONFIG_SHA256,
        "native_execution_authorized": True,
        "dev_authorized": False,
        "heldout_authorized": False,
        "v0_authorized": False,
        "semantic_audit_authorized": False,
        "promotion_authorized": False,
        "retry_authorized": False,
        "issued_at_utc": "2026-10-27T00:00:00+00:00",
    }
    target = tmp_path / declaration["authorization"]["path"]
    target.parent.mkdir(parents=True)
    for mutate, expected in (
            (lambda item: item.update(dev_authorized=True),
             "receipt_opens_closed_field:dev_authorized"),
            (lambda item: item.update(native_execution_authorized=False),
             "receipt_does_not_authorize_native_execution"),
            (lambda item: item.update(status="proposed"),
             "receipt_status_not_authorized"),
            (lambda item: item.update(config_sha256="1" * 64),
             "receipt_config_sha256_differs"),
    ):
        receipt = copy.deepcopy(base)
        mutate(receipt)
        target.write_text(json.dumps(receipt) + "\n")
        state = checker._authorization_state(
            declaration, {"name": "vision_smaller_step_lr3e5_v1"}, tmp_path,
            git_root=ROOT)
        assert state["status"] == "present_invalid"
        assert expected in state["violations"]
        assert state["commit_bound"] is False
    assert state["status"] == "present_invalid"


@pytest.mark.parametrize("section, mutate", [
    ("subject", lambda item: item.update(packet_file_sha256="2" * 64)),
    ("subject", lambda item: item.update(lr=1e-05)),
    ("preserved_contract",
     lambda item: item["limits"].update(decoder_forward_calls=4825)),
    ("preserved_contract",
     lambda item: item["limits"].update(artifact_bytes=33554433)),
    ("preserved_contract",
     lambda item: item["optimization_gate"].update(
         max_update_batch_mean_nll_delta=0.5)),
    ("preserved_contract",
     lambda item: item["functional_fit_gate"].update(answer_correct_min=0)),
    ("preserved_contract",
     lambda item: item["functional_fit_gate"].update(
         eos_only_not_counted_correct=False)),
    ("preserved_contract",
     lambda item: item["decision"].update(retry_authorized=True)),
    ("preserved_contract",
     lambda item: item["raw_generation"]["required_paths"].pop()),
    ("preserved_contract",
     lambda item: item["raw_generation"]["controls"].remove("noimage")),
    ("preserved_contract",
     lambda item: item["raw_generation"].update(
         expected_generated_tokens=3801)),
    ("preserved_contract",
     lambda item: item["campaign_cumulative"].update(
         visual_observation_calls=167)),
    ("commit_binding", lambda item: item.update(reviewed_commit="abc")),
    ("commit_binding", lambda item: item.update(receipt_protocol_commit="0" * 40)),
    ("authorization",
     lambda item: item["inspected_existing_receipts"][0].update(
         binds_subject_config=True)),
    ("launch",
     lambda item: item.update(output="runs/some_other_arm")),
    ("closed_roles", lambda item: item.update(development=True)),
])
def test_tampered_declaration_is_refused(checker, clean_root, tmp_path,
                                         section, mutate):
    declaration = json.loads((ROOT / DECLARATION).read_text())
    mutate(declaration[section])
    target = tmp_path / "readiness.json"
    target.write_text(json.dumps(declaration) + "\n")
    with pytest.raises(checker.ReadinessError):
        checker.assess(target, root=clean_root)


def test_declaration_cannot_allow_launch_or_reorder_requirements(
        checker, clean_root, tmp_path):
    for mutate in (
            lambda item: item["decision"].update(launch_allowed=True),
            lambda item: item["decision"].update(native_execution=True),
            lambda item: item["decision"].update(authorization_created=True),
            lambda item: item["remaining_requirements"].reverse(),
            lambda item: item["remaining_requirements"][0].update(
                blocking=False),
    ):
        declaration = json.loads((ROOT / DECLARATION).read_text())
        mutate(declaration)
        target = tmp_path / "readiness.json"
        target.write_text(json.dumps(declaration) + "\n")
        with pytest.raises(checker.ReadinessError):
            checker.assess(target, root=clean_root)


def test_fresh_output_presence_blocks_the_launch(
        checker, declaration, tmp_path):
    authorization = {"present": False, "protocol_commit": None,
                     "commit_bound": False,
                     "native_execution_authorized": False}
    state = checker._launch_state(declaration, authorization, tmp_path)
    assert state["output_exists"] is False
    (tmp_path / declaration["launch"]["output"]).mkdir(parents=True)
    state = checker._launch_state(declaration, authorization, tmp_path)
    assert state["output_exists"] is True
    assert state["refusals"][-1] == "fresh_output_exists"
    assert state["disposition"] == "review_only_launch_refused"


def test_stale_and_malformed_assessments_are_refused(
        checker, report, clean_root, tmp_path):
    clone = copy.deepcopy(report)
    clone["preserved_contract"]["limits"]["decoder_forward_calls"] = 1
    clone["readiness_sha256"] = checker._readiness_sha256(clone)
    with pytest.raises(checker.StaleReadinessError):
        checker.verify_report(report=clone, root=clean_root)
    bad_hash = copy.deepcopy(report)
    bad_hash["readiness_sha256"] = "0" * 64
    with pytest.raises(checker.ReadinessError):
        checker.verify_report(report=bad_hash, root=clean_root)
    with pytest.raises(checker.ReadinessError):
        checker.verify_report(report_path=tmp_path / "missing.json",
                              root=clean_root)

def test_scan_detects_a_receipt_that_binds_the_subject(
        checker, tmp_path):
    target = tmp_path / "reports/example/authorization.json"
    target.parent.mkdir(parents=True)
    target.write_text(json.dumps({
        "format": "replica.vision_smaller_step.authorization.v1",
        "status": "authorized",
        "protocol_name": "vision_smaller_step_lr3e5_v1",
        "protocol_path": CONFIG,
        "protocol_commit": "0" * 40,
        "config_sha256": CONFIG_SHA256,
    }) + "\n")
    records = checker._scan_authorizations(tmp_path, {
        "protocol_sha256": CONFIG_SHA256, "protocol_path": CONFIG})
    assert len(records) == 1
    assert records[0]["path"] == "reports/example/authorization.json"
    assert records[0]["binds_subject_config"] is True

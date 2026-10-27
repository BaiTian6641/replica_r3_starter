"""Fail-closed tests for the corrected caption lineage and sealed synthetic eval.

Everything runs on CPU, offline.  No model, tensor library or native command is
used.  Temporary workspaces copy only the small local artifacts required by the
pins; the real data roots are never modified.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

import pytest

from conftest import require_local_artifacts
from replica_lab.contracts import canonical_json


CAPTION_ARTIFACTS = "data/external_caption_tranche_20261025_v2/raw/images"

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BUILDER = _load("caption_lineage_builder_20261027",
                "scripts/build_caption_lineage_20261027.py")
EVAL_BUILDER = _load("synthetic_scene_eval_builder_20261027",
                     "scripts/build_synthetic_scene_eval_20261027.py")
VALIDATOR = _load("caption_lineage_eval_validator_20261027",
                  "scripts/validate_caption_lineage_and_eval_20261027.py")

LINEAGE_CONFIG = "configs/caption_lineage_correction_20261027.json"
EVAL_CONFIG = "configs/synthetic_scene_eval_20261027.json"
RECEIPT_PATH = "configs/caption_admission_receipt_20261027.json"
AS_OF = "2026-10-27"

FILES = (
    LINEAGE_CONFIG,
    EVAL_CONFIG,
    "configs/external_caption_asset_qa_v1.json",
    "reports/external_caption_asset_qa_20261027/admission_decision.json",
    "reports/remediation_20261027/sealed_caption_review_exposure.json",
    "reports/caption_admission_preparation_20261027/admission_packet.json",
    "scripts/build_caption_lineage_20261027.py",
    "scripts/build_synthetic_scene_eval_20261027.py",
    "scripts/validate_caption_lineage_and_eval_20261027.py",
)
TREES = (
    "data/caption_corrected_train_20261027",
    "data/synthetic_scene_eval_20261027",
)


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _hash_of(value: dict, field: str) -> str:
    return _sha(canonical_json({key: item for key, item in value.items()
                                if key != field}).encode("utf-8"))


def _rows_text(rows) -> str:
    return "".join(canonical_json(row) + "\n" for row in rows)


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")


def _workspace(tmp_path: Path) -> Path:
    root = tmp_path / "ws"
    for relative in FILES:
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    tranche = root / "data/external_caption_tranche_20261025_v2"
    tranche.mkdir(parents=True, exist_ok=True)
    for relative in ("manifest.json", "corpus.jsonl"):
        shutil.copy2(ROOT / "data/external_caption_tranche_20261025_v2" / relative,
                     tranche / relative)
    shutil.copytree(ROOT / "data/external_caption_tranche_20261025_v2/manifests",
                    tranche / "manifests")
    shutil.copytree(ROOT / "data/external_caption_tranche_20261025_v2/raw/images",
                    tranche / "raw/images")
    for relative in TREES:
        shutil.copytree(ROOT / relative, root / relative)
    return root


def _validate(root: Path, *, require_admission=False):
    return VALIDATOR.validate(
        root / LINEAGE_CONFIG, root / EVAL_CONFIG, project_root=root,
        as_of=AS_OF, require_admission=require_admission)


def _lineage_payload(root: Path):
    config = json.loads((root / LINEAGE_CONFIG).read_text(encoding="utf-8"))
    return config, VALIDATOR.validate_lineage(config, project_root=root, checks=[])


def _refresh_lineage(root: Path, rows=None, exclusions=None) -> None:
    config = json.loads((root / LINEAGE_CONFIG).read_text(encoding="utf-8"))
    rows = rows if rows is not None else VALIDATOR.load_rows(root / config["outputs"]["rows"])
    exclusions = (exclusions if exclusions is not None
                  else VALIDATOR.load_rows(root / config["outputs"]["exclusions"]))
    (root / config["outputs"]["rows"]).write_text(_rows_text(rows), encoding="utf-8")
    (root / config["outputs"]["exclusions"]).write_text(
        _rows_text(exclusions), encoding="utf-8")
    manifest = json.loads((root / config["outputs"]["lineage_manifest"])
                          .read_text(encoding="utf-8"))
    manifest["rows"] = [{
        "row_id": row["row_id"],
        "group_key": row["group_key"],
        "parent_row_id": row["parent"]["tranche_row_id"],
        "row_sha256": row["row_sha256"],
        "image_sha256": row["asset"]["image_sha256"],
        "target_sha256": row["target"]["caption_sha256"],
        "normalized_caption_sha256": row["provenance"]["normalized_caption_sha256"],
        "local_path": row["asset"]["local_path"],
        "role": "train",
    } for row in rows]
    manifest["counts"].update(rows=len(rows), exclusions=len(exclusions))
    manifest["rows_sha256"] = _sha(_rows_text(rows).encode("utf-8"))
    manifest["exclusions_sha256"] = _sha(_rows_text(exclusions).encode("utf-8"))
    manifest["manifest_sha256"] = _hash_of(manifest, "manifest_sha256")
    _write_json(root / config["outputs"]["lineage_manifest"], manifest)


def _seal_receipt(receipt: dict) -> dict:
    body = {key: value for key, value in receipt.items()
            if key != "receipt_sha256"}
    sealed = dict(body)
    sealed["receipt_sha256"] = _sha(canonical_json(body).encode("utf-8"))
    return sealed


def _receipt(root: Path, **overrides) -> dict:
    manifest = json.loads((root / "data/caption_corrected_train_20261027/"
                           "lineage_manifest.json").read_text(encoding="utf-8"))
    seal = json.loads((root / "data/synthetic_scene_eval_20261027/seal.json")
                      .read_text(encoding="utf-8"))
    payload = {
        "format": VALIDATOR.RECEIPT_FORMAT,
        "scope": "training_admission",
        "trust": "trusted_local_reviewed_receipt_not_a_signature",
        "lineage_id": manifest["lineage_id"],
        "decision": "admit_training",
        "decided_by": "test-reviewer",
        "decided_at": "2026-10-27T12:00:00Z",
        "bindings": {
            "lineage_config_sha256": _sha((root / LINEAGE_CONFIG).read_bytes()),
            "eval_config_sha256": _sha((root / EVAL_CONFIG).read_bytes()),
            "lineage_manifest_sha256": manifest["manifest_sha256"],
            "lineage_rows_sha256": manifest["rows_sha256"],
            "lineage_exclusions_sha256": manifest["exclusions_sha256"],
            "eval_manifest_sha256":
                seal["sealed_artifacts"]["manifest_sha256"],
            "eval_rows_sha256": seal["sealed_artifacts"]["rows_sha256"],
            "eval_seal_sha256": seal["seal_sha256"],
            "validator_version": VALIDATOR.VALIDATOR_VERSION,
        },
        "admitted_row_ids": [row["row_id"] for row in manifest["rows"]],
        "conditions": ["research-use only"],
    }
    payload.update(overrides)
    receipt = _seal_receipt(payload)
    _write_json(root / RECEIPT_PATH, receipt)
    return receipt


# ---------------------------------------------------------------------------
# Positive path


@require_local_artifacts(CAPTION_ARTIFACTS)
def test_real_artifacts_are_valid_and_training_is_not_admitted(tmp_path):
    root = _workspace(tmp_path)
    report = _validate(root)
    assert report["status"] == "valid"
    assert report["lineage"]["rows"] == 9
    assert report["lineage"]["exclusions"] == 21
    assert report["lineage"]["exposed_rows"] == 8
    assert report["eval"]["rows"] == 24
    assert report["eval"]["images"] == 24
    assert report["admission"]["status"] == "not_admitted"
    assert report["admission"]["reason"] == "receipt_absent"
    assert report["admission"]["admitted_row_ids"] == []
    assert report["non_claims"]["training_admitted"] is False
    seal = json.loads((root / "data/synthetic_scene_eval_20261027/seal.json")
                      .read_text(encoding="utf-8"))
    assert report["eval"]["seal_sha256"] == seal["seal_sha256"]
    assert seal["attestations"]["sealed_before_any_model_use"] is True
    assert seal["attestations"]["model_executions_at_seal"] == 0


@require_local_artifacts(CAPTION_ARTIFACTS)
def test_generators_are_deterministic_and_do_not_change_the_seal(tmp_path):
    root = _workspace(tmp_path)
    lineage_config = json.loads((root / LINEAGE_CONFIG).read_text(encoding="utf-8"))
    eval_config = json.loads((root / EVAL_CONFIG).read_text(encoding="utf-8"))
    first = BUILDER.build(lineage_config, project_root=root)
    second = BUILDER.build(lineage_config, project_root=root)
    assert first["manifest_text"] == second["manifest_text"]
    assert first["rows_text"] == second["rows_text"]
    assert first["manifest"]["manifest_sha256"] == second["manifest"]["manifest_sha256"]

    eval_first = EVAL_BUILDER.build(eval_config, project_root=root)
    eval_second = EVAL_BUILDER.build(eval_config, project_root=root)
    assert eval_first["rows_sha256"] == eval_second["rows_sha256"]
    assert eval_first["specs_sha256"] == eval_second["specs_sha256"]
    assert eval_first["image_bytes"] == eval_second["image_bytes"]
    seal = json.loads((root / "data/synthetic_scene_eval_20261027/seal.json")
                      .read_text(encoding="utf-8"))
    assert seal["sealed_artifacts"]["rows_sha256"] == eval_first["rows_sha256"]
    assert seal["sealed_artifacts"]["scene_specs_sha256"] == eval_first["specs_sha256"]


@require_local_artifacts(CAPTION_ARTIFACTS)
def test_all_external_rows_are_accounted_and_exposed_rows_excluded(tmp_path):
    root = _workspace(tmp_path)
    config = json.loads((root / LINEAGE_CONFIG).read_text(encoding="utf-8"))
    exposure = json.loads((root / "reports/remediation_20261027/"
                          "sealed_caption_review_exposure.json")
                          .read_text(encoding="utf-8"))
    exclusions = VALIDATOR.load_rows(root / config["outputs"]["exclusions"])
    rows = VALIDATOR.load_rows(root / config["outputs"]["rows"])
    exposed_reasons = {item["row_id"]: item["reason_code"] for item in exclusions
                       if item["row_id"] in exposure["exposed_row_ids"]}
    assert set(exposed_reasons) == set(exposure["exposed_row_ids"])
    assert set(exposed_reasons.values()) == {"sealed_exposed_quarantine"}
    assert not ({row["row_id"] for row in rows}
                & set(exposure["exposed_row_ids"]))
    assert len(exclusions) + len(rows) == 30
    for row in rows:
        assert "caption-corrected-20261027:" in row["row_id"]
        assert row["parent"]["qa_decision"] == "reject"
        assert row["parent"]["qa_decision_relabeled"] is False


# ---------------------------------------------------------------------------
# Duplicate, tamper, leakage and exposed-row fail-closed tests


@require_local_artifacts(CAPTION_ARTIFACTS)
def test_duplicate_group_helper_rejects_shared_keys(tmp_path):
    root = _workspace(tmp_path)
    _, lineage = _lineage_payload(root)
    rows = copy.deepcopy(lineage["rows"])
    assert VALIDATOR.find_duplicate_groups(rows) == {}
    rows[1]["asset"]["image_sha256"] = rows[0]["asset"]["image_sha256"]
    rows[2]["target"]["caption_sha256"] = rows[0]["target"]["caption_sha256"]
    rows[3]["parent"]["tranche_row_id"] = rows[0]["parent"]["tranche_row_id"]
    groups = VALIDATOR.find_duplicate_groups(rows)
    assert groups["image_sha256"]
    assert groups["target_sha256"]
    assert groups["root_group_key"]


@require_local_artifacts(CAPTION_ARTIFACTS)
def test_validator_duplicate_check_is_wired_to_lineage_validation(tmp_path, monkeypatch):
    root = _workspace(tmp_path)
    monkeypatch.setattr(VALIDATOR, "find_duplicate_groups",
                        lambda rows: {"image_sha256": {"x": ["a", "b"]}})
    config = json.loads((root / LINEAGE_CONFIG).read_text(encoding="utf-8"))
    with pytest.raises(VALIDATOR.GuardError) as error:
        VALIDATOR.validate_lineage(config, project_root=root, checks=[])
    assert error.value.check == "lineage.duplicates"


@require_local_artifacts(CAPTION_ARTIFACTS)
def test_rows_file_tamper_is_rejected(tmp_path):
    root = _workspace(tmp_path)
    path = root / "data/caption_corrected_train_20261027/rows.jsonl"
    rows = VALIDATOR.load_rows(path)
    rows[0]["target"]["caption"] = "a tampered training caption"
    path.write_text(_rows_text(rows), encoding="utf-8")
    report = _validate(root)
    assert report["status"] == "invalid"
    assert report["failed_check"] == "lineage.rows_hash"


@require_local_artifacts(CAPTION_ARTIFACTS)
def test_manifest_tamper_is_rejected(tmp_path):
    root = _workspace(tmp_path)
    path = root / "data/caption_corrected_train_20261027/lineage_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["selection"]["rule"] = "loosened rule"
    _write_json(path, manifest)
    report = _validate(root)
    assert report["status"] == "invalid"
    assert report["failed_check"] == "lineage.manifest_self_hash"


@require_local_artifacts(CAPTION_ARTIFACTS)
def test_eval_image_tamper_is_rejected(tmp_path):
    root = _workspace(tmp_path)
    image = sorted((root / "data/synthetic_scene_eval_20261027/images").iterdir())[0]
    image.write_bytes(image.read_bytes() + b"\0")
    report = _validate(root)
    assert report["status"] == "invalid"
    assert report["failed_check"].startswith("eval.")


@require_local_artifacts(CAPTION_ARTIFACTS)
def test_eval_seal_tamper_is_rejected(tmp_path):
    root = _workspace(tmp_path)
    path = root / "data/synthetic_scene_eval_20261027/seal.json"
    seal = json.loads(path.read_text(encoding="utf-8"))
    seal["attestations"]["model_executions_at_seal"] = 1
    seal["seal_sha256"] = _hash_of(seal, "seal_sha256")
    _write_json(path, seal)
    report = _validate(root)
    assert report["status"] == "invalid"
    assert report["failed_check"] == "eval.seal_attestation"

    seal["seal_sha256"] = "0" * 64
    _write_json(path, seal)
    report = _validate(root)
    assert report["status"] == "invalid"
    assert report["failed_check"] == "eval.seal_digest"


@require_local_artifacts(CAPTION_ARTIFACTS)
def test_scene_spec_and_row_tamper_are_rejected(tmp_path):
    root = _workspace(tmp_path)
    specs_path = root / "data/synthetic_scene_eval_20261027/scene_specs.json"
    specs = json.loads(specs_path.read_text(encoding="utf-8"))
    specs["scenes"][0]["caption"] = "a mismatched caption"
    _write_json(specs_path, specs)
    report = _validate(root)
    assert report["status"] == "invalid"
    assert report["failed_check"] == "eval.seal_binding.scene_specs_sha256"

    root = _workspace(tmp_path / "second")
    rows_path = root / "data/synthetic_scene_eval_20261027/rows.jsonl"
    rows = VALIDATOR.load_rows(rows_path)
    rows[0]["target"]["caption"] = "a tampered caption"
    rows_path.write_text(_rows_text(rows), encoding="utf-8")
    report = _validate(root)
    assert report["status"] == "invalid"
    assert report["failed_check"] == "eval.seal_binding.rows_sha256"


@require_local_artifacts(CAPTION_ARTIFACTS)
@pytest.mark.parametrize("field", ["row_id", "image_sha256", "caption"])
def test_eval_training_leakage_fails_closed(tmp_path, field):
    root = _workspace(tmp_path)
    config = json.loads((root / LINEAGE_CONFIG).read_text(encoding="utf-8"))
    lineage = VALIDATOR.validate_lineage(config, project_root=root, checks=[])
    evals = VALIDATOR.validate_eval(json.loads((root / EVAL_CONFIG)
                                                .read_text(encoding="utf-8")),
                                    project_root=root, lineage=lineage, checks=[])
    training = lineage["rows"][0]
    leaked = evals["rows"][0]
    if field == "row_id":
        leaked["row_id"] = training["row_id"]
    elif field == "image_sha256":
        leaked["asset"]["image_sha256"] = training["asset"]["image_sha256"]
    else:
        leaked["target"]["caption"] = training["target"]["caption"]
        leaked["provenance"]["normalized_caption_sha256"] = \
            training["provenance"]["normalized_caption_sha256"]
    with pytest.raises(VALIDATOR.GuardError) as error:
        VALIDATOR.validate_separation(lineage, evals, config, checks=[])
    assert error.value.check == "separation.keys"


@require_local_artifacts(CAPTION_ARTIFACTS)
def test_exposed_row_cannot_be_marked_eligible_or_omitted(tmp_path):
    root = _workspace(tmp_path)
    config = json.loads((root / LINEAGE_CONFIG).read_text(encoding="utf-8"))
    exposure = json.loads((root / "reports/remediation_20261027/"
                          "sealed_caption_review_exposure.json")
                          .read_text(encoding="utf-8"))
    excluded_id = exposure["exposed_row_ids"][0]

    exclusions = VALIDATOR.load_rows(root / config["outputs"]["exclusions"])
    for item in exclusions:
        if item["row_id"] == excluded_id:
            item["training_eligible"] = True
    _refresh_lineage(root, exclusions=exclusions)
    with pytest.raises(VALIDATOR.GuardError) as error:
        VALIDATOR.validate_lineage(json.loads((root / LINEAGE_CONFIG)
                                              .read_text(encoding="utf-8")),
                                   project_root=root, checks=[])
    assert error.value.check == "lineage.exclusions"

    root = _workspace(tmp_path / "second")
    exclusions = VALIDATOR.load_rows(root / config["outputs"]["exclusions"])
    exclusions = [item for item in exclusions if item["row_id"] != excluded_id]
    _refresh_lineage(root, exclusions=exclusions)
    with pytest.raises(VALIDATOR.GuardError) as error:
        VALIDATOR.validate_lineage(json.loads((root / LINEAGE_CONFIG)
                                              .read_text(encoding="utf-8")),
                                   project_root=root, checks=[])
    assert error.value.check == "lineage.exclusions"


@require_local_artifacts(CAPTION_ARTIFACTS)
def test_config_cannot_admit_an_exposed_parent(tmp_path):
    root = _workspace(tmp_path)
    config = json.loads((root / LINEAGE_CONFIG).read_text(encoding="utf-8"))
    exposure = json.loads((root / "reports/remediation_20261027/"
                          "sealed_caption_review_exposure.json")
                          .read_text(encoding="utf-8"))
    config["selection"]["admitted_row_ids"].append(exposure["exposed_row_ids"][0])
    _write_json(root / LINEAGE_CONFIG, config)
    with pytest.raises(VALIDATOR.GuardError) as error:
        VALIDATOR.validate_lineage(config, project_root=root, checks=[])
    assert error.value.check == "lineage.selection"


@require_local_artifacts(CAPTION_ARTIFACTS)
def test_eval_row_with_exposed_identity_is_rejected(tmp_path):
    root = _workspace(tmp_path)
    config = json.loads((root / LINEAGE_CONFIG).read_text(encoding="utf-8"))
    lineage = VALIDATOR.validate_lineage(config, project_root=root, checks=[])
    evals = VALIDATOR.validate_eval(json.loads((root / EVAL_CONFIG)
                                                .read_text(encoding="utf-8")),
                                    project_root=root, lineage=lineage, checks=[])
    evals["rows"][0]["row_id"] = lineage["exposed"][0]
    with pytest.raises(VALIDATOR.GuardError) as error:
        VALIDATOR.validate_separation(lineage, evals, config, checks=[])
    assert error.value.check == "separation.exposed"


# ---------------------------------------------------------------------------
# Admission receipt


@require_local_artifacts(CAPTION_ARTIFACTS)
def test_missing_receipt_never_admits(tmp_path):
    root = _workspace(tmp_path)
    report = _validate(root)
    assert report["admission"]["status"] == "not_admitted"
    assert report["admission"]["reason"] == "receipt_absent"
    assert report["admission"]["eligible_row_ids"]
    with pytest.raises(VALIDATOR.GuardError) as error:
        _validate(root, require_admission=True)
    assert error.value.check == "admission.required"


@require_local_artifacts(CAPTION_ARTIFACTS)
def test_reviewed_receipt_admits_only_the_nine_rows(tmp_path):
    root = _workspace(tmp_path)
    receipt = _receipt(root)
    assert receipt["receipt_sha256"] == _seal_receipt(receipt)["receipt_sha256"]
    report = _validate(root)
    assert report["status"] == "valid"
    assert report["admission"]["status"] == "admitted"
    assert report["admission"]["scope"] == "training_admission"
    assert report["admission"]["trust"] == \
        "trusted_local_reviewed_receipt_not_a_signature"
    assert report["admission"]["receipt_sha256"] == receipt["receipt_sha256"]
    manifest = json.loads((root / "data/caption_corrected_train_20261027/"
                           "lineage_manifest.json").read_text(encoding="utf-8"))
    expected_ids = {row["row_id"] for row in manifest["rows"]}
    assert len(expected_ids) == 9
    assert set(report["admission"]["eligible_row_ids"]) == expected_ids
    assert len(report["admission"]["admitted_row_ids"]) == 9
    assert set(report["admission"]["admitted_row_ids"]) == \
        set(report["admission"]["eligible_row_ids"])
    admitted = _validate(root, require_admission=True)
    assert admitted["admission"]["status"] == "admitted"


@require_local_artifacts(CAPTION_ARTIFACTS)
def test_tampered_or_unknown_receipt_fields_are_blocked(tmp_path):
    root = _workspace(tmp_path)
    receipt = _receipt(root)
    receipt["decision"] = "admit_training "
    _write_json(root / RECEIPT_PATH, receipt)
    report = _validate(root)
    assert report["admission"]["status"] == "blocked"
    assert "self-digest" in report["admission"]["reason"]

    root = _workspace(tmp_path / "fields")
    receipt = _receipt(root)
    receipt.pop("conditions")
    _write_json(root / RECEIPT_PATH, _seal_receipt(receipt))
    report = _validate(root)
    assert report["admission"]["status"] == "blocked"
    assert "fields" in report["admission"]["reason"]


@require_local_artifacts(CAPTION_ARTIFACTS)
def test_stale_receipt_bindings_are_blocked(tmp_path):
    root = _workspace(tmp_path)
    receipt = _receipt(root)
    receipt["bindings"]["lineage_manifest_sha256"] = "0" * 64
    _write_json(root / RECEIPT_PATH, _seal_receipt(receipt))
    report = _validate(root)
    assert report["admission"]["status"] == "blocked"
    assert "binding" in report["admission"]["reason"]

    root = _workspace(tmp_path / "rows")
    receipt = _receipt(root)
    receipt["bindings"]["lineage_rows_sha256"] = "0" * 64
    _write_json(root / RECEIPT_PATH, _seal_receipt(receipt))
    report = _validate(root)
    assert report["admission"]["status"] == "blocked"
    assert "binding" in report["admission"]["reason"]

    root = _workspace(tmp_path / "config")
    receipt = _receipt(root)
    receipt["bindings"]["lineage_config_sha256"] = "0" * 64
    _write_json(root / RECEIPT_PATH, _seal_receipt(receipt))
    report = _validate(root)
    assert report["admission"]["status"] == "blocked"
    assert "binding" in report["admission"]["reason"]


@require_local_artifacts(CAPTION_ARTIFACTS)
def test_wrong_rows_or_review_fields_are_blocked(tmp_path):
    root = _workspace(tmp_path)
    receipt = _receipt(root)
    receipt["admitted_row_ids"] = receipt["admitted_row_ids"][:8]
    _write_json(root / RECEIPT_PATH, _seal_receipt(receipt))
    report = _validate(root)
    assert report["admission"]["status"] == "blocked"
    assert "rows" in report["admission"]["reason"]

    root = _workspace(tmp_path / "exposed")
    receipt = _receipt(root)
    report_before = _validate(root)
    exposed_id = json.loads(
        (root / "reports/remediation_20261027/sealed_caption_review_exposure.json")
        .read_text(encoding="utf-8"))["exposed_row_ids"][0]
    receipt["admitted_row_ids"] = \
        receipt["admitted_row_ids"][:8] + [exposed_id]
    _write_json(root / RECEIPT_PATH, _seal_receipt(receipt))
    report = _validate(root)
    assert report_before["status"] == "valid"
    assert report["admission"]["status"] == "blocked"
    assert "rows" in report["admission"]["reason"]

    root = _workspace(tmp_path / "reviewer")
    receipt = _receipt(root, decided_by="")
    _write_json(root / RECEIPT_PATH, _seal_receipt(receipt))
    report = _validate(root)
    assert report["admission"]["status"] == "blocked"
    assert "decided_by" in report["admission"]["reason"]

    root = _workspace(tmp_path / "timestamp")
    receipt = _receipt(root, decided_at="yesterday")
    _write_json(root / RECEIPT_PATH, _seal_receipt(receipt))
    report = _validate(root)
    assert report["admission"]["status"] == "blocked"
    assert "decided_at" in report["admission"]["reason"]


@require_local_artifacts(CAPTION_ARTIFACTS)
def test_rejecting_receipt_is_valid_but_never_admits(tmp_path):
    root = _workspace(tmp_path)
    _receipt(root, decision="reject_training")
    report = _validate(root)
    assert report["status"] == "valid"
    assert report["admission"]["status"] == "not_admitted"
    assert report["admission"]["reason"] == "receipt_decision_reject"
    assert report["admission"]["admitted_row_ids"] == []
    with pytest.raises(VALIDATOR.GuardError) as error:
        _validate(root, require_admission=True)
    assert error.value.check == "admission.required"


# ---------------------------------------------------------------------------
# Unsigned admission evidence packet


def _packet(root: Path) -> dict:
    return json.loads((root / VALIDATOR.PACKET_PATH).read_text(encoding="utf-8"))


def _write_packet(root: Path, packet: dict) -> None:
    _write_json(root / VALIDATOR.PACKET_PATH, packet)


def _verify_packet(root: Path, packet: dict):
    config = json.loads((root / LINEAGE_CONFIG).read_text(encoding="utf-8"))
    eval_config = json.loads((root / EVAL_CONFIG).read_text(encoding="utf-8"))
    lineage = VALIDATOR.validate_lineage(config, project_root=root, checks=[])
    evals = VALIDATOR.validate_eval(eval_config, project_root=root,
                                    lineage=lineage, checks=[])
    return VALIDATOR.verify_admission_packet(
        packet, admission=config["admission"], config=config,
        eval_config=eval_config, lineage=lineage, evals=evals,
        lineage_config_sha256=_sha((root / LINEAGE_CONFIG).read_bytes()),
        eval_config_sha256=_sha((root / EVAL_CONFIG).read_bytes()))


@require_local_artifacts(CAPTION_ARTIFACTS)
def test_unsigned_packet_matches_receipt_schema_without_signing(tmp_path):
    root = _workspace(tmp_path)
    packet = _packet(root)
    config = json.loads((root / LINEAGE_CONFIG).read_text(encoding="utf-8"))
    expected = VALIDATOR.build_admission_packet(
        root / LINEAGE_CONFIG, root / EVAL_CONFIG, project_root=root)
    assert packet == expected
    assert not (root / RECEIPT_PATH).exists()

    assert packet["format"] == VALIDATOR.PACKET_FORMAT
    assert packet["status"] == "prepared_unsigned_awaiting_review"
    assert packet["not_a_receipt"] is True and packet["unsigned"] is True
    assert packet["signature_present"] is False
    assert set(packet["receipt_fields"]) == \
        set(config["admission"]["required_fields"])
    assert tuple(packet["pending_reviewer_fields"]) == \
        VALIDATOR.PACKET_PENDING_FIELDS
    receipt_fields = packet["receipt_fields"]
    for field in ("decision", "decided_by", "decided_at", "receipt_sha256"):
        assert receipt_fields[field] is None
    assert receipt_fields["conditions"] == []
    assert receipt_fields["format"] == VALIDATOR.RECEIPT_FORMAT
    assert receipt_fields["scope"] == "training_admission"

    evidence = packet["evidence"]
    assert evidence["lineage"]["counts"] == {
        "rows": 9, "exclusions": 21, "exposed": 8,
        "parent_rows_accounted": 30, "eval_rows": 24, "eval_images": 24}
    assert evidence["lineage"]["exclusion_reason_counts"] == {
        "semantic_mismatch": 1, "semantic_unclear": 5,
        "validation_only_role": 7, "sealed_exposed_quarantine": 8}
    assert len(evidence["lineage"]["eligible_rows"]) == 9
    assert len(evidence["lineage"]["exclusions"]) == 21
    assert evidence["lineage"]["parent_decision"]["decision"] == "reject"
    assert evidence["lineage"]["parent_decision"]["relabeled"] is False
    exposure = evidence["lineage"]["exposure"]
    assert len(exposure["exposed_row_ids"]) == 8
    assert exposure["history_preserved"] is True
    assert exposure["untouched_eval_status_restored"] is False
    assert evidence["admission"]["training_admitted_row_ids"] == []
    assert evidence["admission"]["receipt_created"] is False

    diagnostic = evidence["evaluation"]["diagnostic_scope"]
    assert diagnostic["diagnostic_only"] is True
    assert diagnostic["natural_image_validation"] is False
    assert diagnostic["broad_visual_validation"] is False
    assert diagnostic["rows"] == diagnostic["images"] == 24
    assert diagnostic["benchmark_quality"] == "none"
    assert packet["non_claims"]["training_admitted"] is False

    report = _validate(root)
    assert report["status"] == "valid"
    assert report["admission_packet"]["status"] == "prepared_unsigned"
    assert report["admission_packet"]["unsigned"] is True
    assert report["admission_packet"]["packet_sha256"] == _sha(
        (root / VALIDATOR.PACKET_PATH).read_bytes())
    assert report["admission"]["status"] == "not_admitted"
    assert report["non_claims"]["training_admitted"] is False


@require_local_artifacts(CAPTION_ARTIFACTS)
def test_missing_or_unsigned_packet_fields_fail_closed(tmp_path):
    root = _workspace(tmp_path)
    packet = _packet(root)
    mutations = []

    def drop_top(value):
        value.pop("unsigned")

    mutations.append(("top field", drop_top, "admission.packet.fields"))

    def drop_pending(value):
        value["pending_reviewer_fields"].pop()

    mutations.append(("pending field", drop_pending, "admission.packet.pending"))

    def drop_binding(value):
        value["receipt_fields"]["bindings"].pop("lineage_rows_sha256")

    mutations.append(("binding", drop_binding, "admission.packet.content"))

    def drop_admitted(value):
        value["receipt_fields"]["admitted_row_ids"].pop()

    mutations.append(("admitted ids", drop_admitted, "admission.packet.content"))

    def drop_evidence(value):
        value["evidence"]["lineage"].pop("exclusions")

    mutations.append(("evidence", drop_evidence, "admission.packet.content"))

    def set_decision(value):
        value["receipt_fields"]["decision"] = "admit_training"

    mutations.append(("decision", set_decision, "admission.packet.unsigned"))

    def set_digest(value):
        value["receipt_fields"]["receipt_sha256"] = "a" * 64

    mutations.append(("digest", set_digest, "admission.packet.unsigned"))

    def set_conditions(value):
        value["receipt_fields"]["conditions"] = ["research-use only"]

    mutations.append(("conditions", set_conditions, "admission.packet.unsigned"))

    def set_signed(value):
        value["unsigned"] = False

    mutations.append(("signed", set_signed, "admission.packet.unsigned"))

    for label, mutate, expected_check in mutations:
        mutated = copy.deepcopy(packet)
        mutate(mutated)
        with pytest.raises(VALIDATOR.GuardError) as error:
            _verify_packet(root, mutated)
        assert error.value.check == expected_check, label


@require_local_artifacts(CAPTION_ARTIFACTS)
def test_extra_packet_fields_and_stale_bindings_are_rejected(tmp_path):
    root = _workspace(tmp_path)
    packet = _packet(root)
    packet["decision"] = "admit_training"
    with pytest.raises(VALIDATOR.GuardError) as error:
        _verify_packet(root, packet)
    assert error.value.check == "admission.packet.fields"

    packet = _packet(root)
    packet["receipt_fields"]["bindings"]["lineage_rows_sha256"] = "0" * 64
    _write_packet(root, packet)
    report = _validate(root)
    assert report["status"] == "invalid"
    assert report["failed_check"].startswith("admission.packet.")
    assert report["non_claims"]["training_admitted"] is False


@require_local_artifacts(CAPTION_ARTIFACTS)
def test_packet_cannot_include_exposed_or_validation_rows(tmp_path):
    root = _workspace(tmp_path)
    packet = _packet(root)
    exposure = json.loads((root / "reports/remediation_20261027/"
                          "sealed_caption_review_exposure.json")
                          .read_text(encoding="utf-8"))
    exposed_id = exposure["exposed_row_ids"][0]
    packet["receipt_fields"]["admitted_row_ids"][0] = exposed_id
    with pytest.raises(VALIDATOR.GuardError) as error:
        _verify_packet(root, packet)
    assert error.value.check == "admission.packet.content"

    packet = _packet(root)
    packet["evidence"]["lineage"]["exclusions"] = [
        item for item in packet["evidence"]["lineage"]["exclusions"]
        if item["row_id"] != exposed_id]
    with pytest.raises(VALIDATOR.GuardError) as error:
        _verify_packet(root, packet)
    assert error.value.check == "admission.packet.content"

    config = json.loads((root / LINEAGE_CONFIG).read_text(encoding="utf-8"))
    validation_id = next(item["row_id"] for item in
                         packet["evidence"]["lineage"]["exclusions"]
                         if item["reason_code"] == "validation_only_role")
    packet = _packet(root)
    packet["receipt_fields"]["admitted_row_ids"][0] = validation_id
    with pytest.raises(VALIDATOR.GuardError) as error:
        _verify_packet(root, packet)
    assert error.value.check == "admission.packet.content"
    assert validation_id in config["selection"]["validation_only_row_ids"]


@require_local_artifacts(CAPTION_ARTIFACTS)
def test_synthetic_eval_is_declared_diagnostic_only(tmp_path):
    root = _workspace(tmp_path)
    eval_config = json.loads((root / EVAL_CONFIG).read_text(encoding="utf-8"))
    limits = " ".join(eval_config["benchmark_limits"]).casefold()
    assert "no natural images" in limits
    assert "cannot support claims about natural-image" in limits
    assert "not a quality benchmark" in limits
    assert eval_config["non_claims"]["external_data"] is False
    assert eval_config["non_claims"]["benchmark_quality"] == "none"
    assert eval_config["non_claims"]["human_reviewed"] is False
    packet = _packet(root)
    diagnostic = packet["evidence"]["evaluation"]["diagnostic_scope"]
    assert diagnostic["diagnostic_only"] is True
    assert diagnostic["natural_image_validation"] is False
    assert diagnostic["broad_visual_validation"] is False
    assert set(diagnostic["forbidden_claims"]) >= {
        "natural-image caption quality", "benchmark superiority"}


@require_local_artifacts(CAPTION_ARTIFACTS)
def test_packet_and_missing_receipt_never_admit_training(tmp_path):
    root = _workspace(tmp_path)
    report = _validate(root)
    assert report["status"] == "valid"
    assert report["admission_packet"]["status"] == "prepared_unsigned"
    assert report["admission"]["status"] == "not_admitted"
    assert report["admission"]["admitted_row_ids"] == []
    assert not (root / RECEIPT_PATH).exists()
    with pytest.raises(VALIDATOR.GuardError) as error:
        _validate(root, require_admission=True)
    assert error.value.check == "admission.required"


@require_local_artifacts(CAPTION_ARTIFACTS)
def test_eval_contract_cannot_be_relabelled_natural_image_validation(tmp_path):
    root = _workspace(tmp_path)
    lineage_config = json.loads((root / LINEAGE_CONFIG).read_text(encoding="utf-8"))
    lineage = VALIDATOR.validate_lineage(lineage_config, project_root=root,
                                         checks=[])
    eval_config = json.loads((root / EVAL_CONFIG).read_text(encoding="utf-8"))
    eval_config["benchmark_limits"] = [
        "Natural-image caption quality is measured by this partition."]
    with pytest.raises(VALIDATOR.GuardError) as error:
        VALIDATOR.validate_eval(eval_config, project_root=root,
                                lineage=lineage, checks=[])
    assert error.value.check == "eval.diagnostic_scope"

    eval_config = json.loads((root / EVAL_CONFIG).read_text(encoding="utf-8"))
    eval_config["non_claims"]["benchmark_quality"] = "natural-image benchmark"
    with pytest.raises(VALIDATOR.GuardError) as error:
        VALIDATOR.validate_eval(eval_config, project_root=root,
                                lineage=lineage, checks=[])
    assert error.value.check == "eval.diagnostic_scope"


@require_local_artifacts(CAPTION_ARTIFACTS)
def test_packet_copied_to_receipt_path_is_refused_as_unsigned(tmp_path):
    root = _workspace(tmp_path)
    packet = _packet(root)
    _write_json(root / RECEIPT_PATH, packet)
    report = _validate(root)
    assert report["status"] == "valid"
    assert report["admission"]["status"] == "blocked"
    assert "fields" in report["admission"]["reason"]
    assert report["admission"]["admitted_row_ids"] == []
    assert report["non_claims"]["training_admitted"] is False

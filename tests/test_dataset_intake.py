"""External tranche selection: pinned identity, reserved eval, fail-closed terms."""
import json
from pathlib import Path

import pytest

from replica_lab.dataset_intake import (SELECTIONS, audit, build_tranche)
from replica_lab.vision_qualification import QualificationError

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "docs/datasets/20261024_Source_Catalog.json"


def source(source_id, *, revision="a" * 40, license_claim="mit", family=None,
           auto_admit=False, admitted=False):
    return {"id": source_id, "immutable_revision": revision,
            "card_license_claim": license_claim,
            "split_group_key": family or f"family-{source_id}",
            "auto_admit": auto_admit, "training_admitted": admitted,
            "content_inspected": False, "normalized": False,
            "card_sha256": "b" * 64, "metadata_sha256": "c" * 64,
            "category": "fixture"}


def selection(source_id, intent):
    return {"id": source_id, "role": "fixture", "intent": intent,
            "requires_assets": True, "notes": "fixture"}


def test_real_tranche_reserves_eval_and_reports_stage_counts():
    record = build_tranche(json.loads(CATALOG.read_text()))
    stages = record["stage_counts"]
    assert stages["catalogued"] >= 34 and stages["metadata_pinned"] >= 22
    assert stages["acquired"] == 0 and stages["admitted"] == 0
    assert stages["tranche_entries"] == len(SELECTIONS)
    assert set(record["reserved_eval_ids"]) == {
        "HuggingFaceM4/VQAv2", "AI4Math/MathVista", "MMMU/MMMU",
        "xiaowu0162/longmemeval-v2"}
    assert set(record["blocked_ids"]) == {"GQA", "VSR", "liuhaotian/LLaVA-Instruct-150K"}
    for item in record["selections"]:
        if item["intent"] == "eval":
            assert item["reserved_before_training"] is True
        assert not item["auto_admit"] and not item["training_admitted"]
        assert item["acquired_bytes"] == 0


def test_missing_terms_or_pins_fail_closed():
    catalog = {"sources": [
        source("unpinned", revision=None),
        source("no-terms", license_claim=None),
        source("good"),
    ]}
    tranche = build_tranche(catalog, [
        selection("unpinned", "train"), selection("no-terms", "train"),
        selection("good", "train")])
    statuses = {item["id"]: item["status"] for item in tranche["selections"]}
    assert statuses["unpinned"] == "blocked_unpinned"
    assert statuses["no-terms"] == "blocked_terms"
    assert statuses["good"] == "candidate_pending_assets"
    assert tranche["training_eligible_ids"] == []


def test_family_reserved_for_eval_blocks_training_derivatives():
    catalog = {"sources": [source("train-view", family="same"),
                           source("eval-view", family="same")]}
    tranche = build_tranche(catalog, [selection("train-view", "train"),
                                      selection("eval-view", "eval")])
    statuses = {item["id"]: item["status"] for item in tranche["selections"]}
    assert statuses["eval-view"].startswith("reserved_eval")
    assert statuses["train-view"] == "blocked_family_overlap"
    assert tranche["training_eligible_ids"] == []
    assert tranche["reserved_eval_ids"] == ["eval-view"]


def test_pre_admitted_or_duplicate_catalogue_entries_are_rejected():
    with pytest.raises(QualificationError, match="pre-admit"):
        build_tranche({"sources": [source("pre", auto_admit=True)]},
                      [selection("pre", "train")])
    with pytest.raises(QualificationError, match="Duplicate"):
        build_tranche({"sources": [source("same"), source("same")]},
                      [selection("same", "train")])
    with pytest.raises(QualificationError, match="not catalogued"):
        build_tranche({"sources": [source("known")]}, [selection("absent", "eval")])


def test_audit_records_catalog_and_tranche_hashes(tmp_path):
    record = audit(CATALOG, tmp_path / "audit.json")
    assert len(record["catalog_sha256"]) == 64
    assert len(record["tranche_sha256"]) == 64
    assert record["assumption_checks"]["evaluation_reserved_before_training"] is True
    assert record["assumption_checks"]["no_component_auto_admitted"] is True
    assert (tmp_path / "audit.json").is_file()

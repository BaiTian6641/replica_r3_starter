import copy
import json
from pathlib import Path

import pytest

from replica_lab.baseline_competence import (
    CompetenceError, FORMAT, binding_for, canonical_sha, load_spec, validate_receipt,
)
from replica_lab.persistent_learner_runner import Protocol, PreparedDataset, Refused

ROOT = Path(__file__).resolve().parents[1]


def protocol():
    return Protocol.load(ROOT / "configs/spark_persistent_learner_protocol_v2.json")


def passing_evidence():
    p = protocol()
    spec, cases = load_spec(p)
    return {"format": FORMAT, "binding": binding_for(p, spec, context_tokens=256, max_new_tokens=32),
        "rows": [{"case_id": c["case_id"], "complete": True, "truncated": False,
                  "prompt_sha256": canonical_sha(spec["prompt_template"].format(context=c["context"], question=c["question"])),
                  "token_ids": [1], "rendered_sha256": "a"*64,
                  "prompt_tokens": 64, "generated_tokens": 1, "decoded": c["expected"]} for c in cases],
        "frozen_base_unchanged": True, "optimizer_steps": 0, "correct": 8, "false_recall": 0, "passed": True}


@pytest.mark.parametrize("damage", ["first_only", "duplicate", "wrong_answer", "wrong_scope", "truncated", "weights_changed", "empty"])
def test_missing_incomplete_or_failed_baseline_never_passes(damage):
    r = passing_evidence()
    if damage == "first_only": r["rows"] = r["rows"][:1]
    if damage == "duplicate": r["rows"][1] = r["rows"][0]
    if damage == "wrong_answer": r["rows"][1]["decoded"] = "unverified"
    if damage == "wrong_scope": r["binding"]["thinking_mode"] = "enabled"
    if damage == "truncated": r["rows"][0]["truncated"] = True
    if damage == "weights_changed": r["frozen_base_unchanged"] = False
    if damage == "empty": r = {}
    with pytest.raises(CompetenceError): validate_receipt(protocol(), r)


def test_complete_preregistered_baseline_and_budget_binding():
    r = passing_evidence()
    assert validate_receipt(protocol(), r)["passed"]
    with pytest.raises(CompetenceError): validate_receipt(protocol(), r, max_new_tokens=8)
    spec, cases = load_spec(protocol())
    assert all(c["split"] == "pre_fit_calibration" for c in cases)
    assert len({c["world_id"] for c in cases}) == 8
    assert spec["pass_criteria"]["correct_min"] == 8


def test_failed_competence_stops_before_any_fit_path_or_label_access(tmp_path, monkeypatch):
    import replica_lab.baseline_competence as gate
    import replica_lab.persistent_learner_runner as runner
    def fail(*args, **kwargs): raise CompetenceError("failed")
    monkeypatch.setattr(gate, "run_competence", fail)
    monkeypatch.setattr(runner, "_new_path", lambda *a, **k: pytest.fail("dependent learning path created"))
    with pytest.raises(CompetenceError): runner.run_fit(protocol(), None, None, tmp_path, None, None)
    p = protocol()
    dataset = PreparedDataset.load(ROOT / "data/spark_persistent_learner_v2",
        expected_sha256=p.config["dataset"]["manifest_sha256"], protocol_sha256=p.sha256)
    receipt = {"format": runner.FIT_RECEIPT_FORMAT, "protocol_id": p.protocol_id,
               "protocol_sha256": p.sha256, "prepared_manifest_sha256": dataset.manifest_sha256,
               "status": "fit_executed", "execution_complete": True}
    with pytest.raises(Refused, match="competence"): dataset.open_development(receipt)
    with pytest.raises(Refused, match="development_labels_closed"): dataset.development_cases()

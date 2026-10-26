"""Acceptance-aware memory metric tests with synthetic readouts."""
import hashlib
import json

import pytest

from replica_lab.memory_acceptance import (
    CONFIG_FORMAT, CURRENT_STATUS, MANIFEST_FORMAT, RECEIPT_FORMAT,
    SUPERSEDED_STATUS, VERSION, acceptance_metrics, evaluate_acceptance,
    load_current_receipt, load_receipt, receipt_digest, rescore_ranking_file,
    validate_config, validate_receipt)


def _config():
    return {
        "format": CONFIG_FORMAT,
        "metric_version": VERSION,
        "candidate_variant": "causal",
        "oracle_variant": "oracle",
        "gates": {
            "accepted_correct_rate_min": 0.85,
            "accepted_precision_min": 0.95,
            "correction_accepted_correct_rate_min": 0.6,
            "unaffected_accepted_correct_rate_min": 0.7,
            "unknown_false_recall_max": 0.1,
            "oracle_raw_top1_rate_min": 0.85,
        },
        "bootstrap": {"seed": 17, "resamples": 100, "confidence": 0.95},
    }


def _row(case="q1", layer=13, *, variant="causal", accepted=True, top1=True,
         norm=1.0, top=0.8, second=0.5, kind="fact", role="correction_target",
         world="w1", episode="e1", state="initial", arm="query_key_value"):
    return {
        "variant": variant, "episode_id": episode, "world_id": world,
        "state": state, "case_id": case, "arm": arm, "layer": layer,
        "kind": kind, "target_role": role, "accepted": accepted, "top1": top1,
        "readout_norm": norm, "top_cosine": top, "second_cosine": second,
    }


def _metrics(rows):
    return acceptance_metrics(rows, resamples=100)


def _receipt():
    return {
        "format": RECEIPT_FORMAT,
        "metric_version": VERSION,
        "source": {"path": "rankings.jsonl", "sha256": "a" * 64,
                   "rows": 1, "original_bytes_unchanged": True},
        "candidate_variant": "causal",
        "oracle_variant": "oracle",
        "metrics": {"causal": {"query": {"n": 1}},
                    "oracle": {"query": {"n": 1}}},
        "verdict": {"decision": "rejected", "checks": {}},
        "generation": {"status": "not_measured", "accuracy": None},
        "claims": {"raw_rankings_are_not_accepted_retrievals": True},
    }


def test_zero_vector_gets_raw_diagnostic_but_no_accepted_credit():
    metrics = _metrics([_row(accepted=True, top1=True, norm=0.0)])
    block = metrics["causal"]["query"]
    assert block["raw_top1"] == 1
    assert block["accepted"] == 0
    assert block["accepted_correct"] == 0
    assert block["zero_or_tied_top1"] == 1
    assert block["abstention_rate"] == 1.0


def test_empty_memory_and_ambiguous_tie_receive_no_usable_credit():
    rows = [
        _row(variant="disabled", accepted=False, top1=True, norm=0.0),
        _row(layer=27, variant="disabled", accepted=True, top1=True, top=0.5, second=0.5),
    ]
    metrics = _metrics(rows)
    block = metrics["disabled"]["query"]
    assert block["raw_top1"] == 1
    assert block["accepted"] == 0
    assert block["accepted_correct"] == 0
    assert block["zero_or_tied_top1"] == 1


def test_rejected_confidence_and_wrong_acceptance_are_distinct():
    metrics = _metrics([
        _row(case="reject", accepted=False, top1=True),
        _row(case="wrong", accepted=True, top1=False),
        _row(case="correct", accepted=True, top1=True),
    ])
    block = metrics["causal"]["query"]
    assert block["raw_top1"] == 2
    assert block["accepted"] == 2
    assert block["accepted_correct"] == 1
    assert block["accepted_wrong"] == 1
    assert block["accepted_precision"] == 0.5
    assert block["acceptance_coverage"] == 2 / 3
    assert block["abstention_rate"] == 1 / 3


def test_query_correctness_uses_all_layers_but_coverage_uses_any_layer():
    metrics = _metrics([
        _row(layer=13, accepted=True, top1=True),
        _row(layer=27, accepted=False, top1=True),
    ])
    readout = metrics["causal"]["readout"]
    query = metrics["causal"]["query"]
    assert readout["accepted_correct"] == 1
    assert readout["n"] == 2
    assert query["n"] == 1
    assert query["accepted_correct"] == 0
    assert query["accepted"] == 1
    assert query["accepted_wrong"] == 1
    assert query["accepted_precision"] == 0.0
    assert query["acceptance_coverage"] == 1.0
    assert metrics["causal"]["counts"]["independent_worlds"] == 1

def test_address_and_context_arms_do_not_change_acceptance_denominator():
    value = _row(arm="query_key_value", accepted=True, top1=True)
    address = _row(arm="query_key_address", accepted=True, top1=True)
    removed = _row(arm="context_removed", accepted=True, top1=True)
    metrics = _metrics([value, address, removed])
    assert metrics["causal"]["counts"]["layer_readouts"] == 1
    assert metrics["causal"]["counts"]["queries"] == 1
    assert metrics["causal"]["query"]["accepted_correct"] == 1
    assert metrics["causal"]["query"]["accepted_precision"] == 1.0


def test_unknown_false_recall_is_separate_from_fact_precision():
    metrics = _metrics([
        _row(case="unknown", kind="unknown", role="unknown",
             accepted=True, top1=False),
        _row(case="fact", kind="fact", accepted=True, top1=True),
    ])
    block = metrics["causal"]["query"]
    unknown = metrics["causal"]["unknown"]
    assert block["accepted_precision"] == 0.5
    assert unknown["unknown_false_recall"] == 1
    assert unknown["unknown_false_recall_rate"] == 1.0


def test_all_gates_pass_only_with_complete_correct_coverage():
    rows = []
    for case, role in (("c1", "correction_target"), ("c2", "correction_target"),
                       ("u1", "retained"), ("u2", "retained"),
                       ("x1", "unknown"), ("x2", "unknown")):
        for layer in (13, 27):
            rows.append(_row(case=case, layer=layer, role=role,
                             kind="unknown" if role == "unknown" else "fact",
                             accepted=role != "unknown", top1=role != "unknown"))
    for layer in (13, 27):
        rows.append(_row(case="oracle", layer=layer, variant="oracle",
                         accepted=True, top1=True))
    verdict = evaluate_acceptance(_metrics(rows), _config())
    assert verdict["decision"] == "accepted"
    assert verdict["acceptance_coverage_policy"] == "report_only"


def test_missing_coverage_and_metric_version_are_rejected():
    with pytest.raises(ValueError, match="missing"):
        evaluate_acceptance(_metrics([]), _config())
    config = _config()
    config["metric_version"] = "memory.acceptance.v1"
    with pytest.raises(ValueError, match="memory.acceptance.v2"):
        evaluate_acceptance(_metrics([_row()]), config)


def test_rescore_preserves_source_bytes_and_separates_generation(tmp_path):
    path = tmp_path / "rankings.jsonl"
    rows = []
    for case, role in (("c1", "correction_target"), ("c2", "correction_target"),
                       ("u1", "retained"), ("u2", "retained"),
                       ("x1", "unknown"), ("x2", "unknown")):
        for layer in (13, 27):
            rows.append(_row(case=case, layer=layer, role=role,
                             kind="unknown" if role == "unknown" else "fact",
                             accepted=role != "unknown", top1=role != "unknown"))
    for layer in (13, 27):
        rows.append(_row(case="oracle", layer=layer, variant="oracle",
                         accepted=True, top1=True))
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    original = path.read_bytes()
    receipt = rescore_ranking_file(path, _config())
    assert receipt["source"]["original_bytes_unchanged"] is True
    assert path.read_bytes() == original
    assert receipt["metric_version"] == VERSION
    assert receipt["generation"]["status"] == "not_measured"
    assert receipt["claims"]["zero_and_ambiguous_tied_readouts_receive_no_credit"] is True


def test_uncertainty_resets_estimable_worlds_per_metric():
    metrics = acceptance_metrics([
        _row(case="a1", world="a", accepted=True, top1=True),
        _row(case="a2", world="a", accepted=True, top1=False),
        _row(case="a3", world="a", accepted=True, top1=False),
        _row(case="b1", world="b", accepted=False, top1=True),
        _row(case="b2", world="b", accepted=False, top1=False),
        _row(case="b3", world="b", accepted=False, top1=True),
        _row(case="c1", world="c", kind="unknown", role="unknown",
             accepted=True, top1=False),
        _row(case="c2", world="c", kind="unknown", role="unknown",
             accepted=False, top1=False),
        _row(case="c3", world="c", kind="unknown", role="unknown",
             accepted=False, top1=False),
    ], bootstrap_seed=17, resamples=100)
    uncertainty = metrics["causal"]["uncertainty"]
    assert metrics["causal"]["by_world"]["b"]["accepted_precision"] is None
    assert metrics["causal"]["by_world"]["c"]["accepted_correct_rate"] is None
    assert uncertainty["accepted_correct_rate"]["clusters"] == 2
    assert uncertainty["accepted_correct_rate"]["mean"] == pytest.approx(1 / 6)
    assert uncertainty["accepted_precision"]["clusters"] == 2
    assert uncertainty["accepted_precision"]["mean"] == pytest.approx(1 / 6)
    assert uncertainty["acceptance_coverage"]["clusters"] == 3
    assert uncertainty["acceptance_coverage"]["mean"] == pytest.approx(4 / 9)
    assert uncertainty["acceptance_coverage"]["lower"] == pytest.approx(0.052777777777778034)
    assert uncertainty["acceptance_coverage"]["upper"] == pytest.approx(0.7777777777777778)
    assert uncertainty["unit"] == "world"
    assert uncertainty["layer_readouts_are_not_independent"] is True


def test_receipt_version_format_and_supersession_are_explicit(tmp_path):
    receipt = _receipt()
    assert validate_receipt(receipt, config=_config()) is receipt
    with pytest.raises(ValueError, match="Unsupported acceptance metric version"):
        validate_receipt({**receipt, "metric_version": "memory.acceptance.v1"},
                         config=_config())
    with pytest.raises(ValueError, match="Unsupported acceptance metric version"):
        validate_receipt({**receipt, "metric_version": "memory.acceptance.v3"},
                         config=_config())
    with pytest.raises(ValueError, match="Unsupported acceptance receipt format"):
        validate_receipt({**receipt, "format": "replica.memory_acceptance.metrics.v1"})
    missing = _receipt()
    del missing["source"]
    with pytest.raises(ValueError, match="missing: source"):
        validate_receipt(missing)
    malformed = _receipt()
    malformed["source"] = {"sha256": "short"}
    with pytest.raises(ValueError, match="source hash is malformed"):
        validate_receipt(malformed)
    digest = receipt_digest(receipt)
    assert validate_receipt(receipt, superseded={digest}) is receipt
    with pytest.raises(ValueError, match="superseded"):
        validate_receipt(receipt, superseded={digest}, require_current=True)

    current_path = tmp_path / "current.json"
    current_path.write_text(json.dumps(receipt))
    stale = json.loads(json.dumps(receipt))
    stale["verdict"] = {"decision": "accepted", "checks": {}}
    stale_path = tmp_path / "stale.json"
    stale_path.write_text(json.dumps(stale))
    manifest = {
        "format": MANIFEST_FORMAT,
        "metric_version": VERSION,
        "authoritative": {
            "path": current_path.name,
            "status": CURRENT_STATUS,
            "file_sha256": hashlib.sha256(current_path.read_bytes()).hexdigest(),
            "receipt_digest": digest,
        },
        "receipts": [{
            "path": stale_path.name,
            "status": SUPERSEDED_STATUS,
            "receipt_digest": receipt_digest(stale),
        }],
    }
    manifest_path = tmp_path / "receipt_manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    assert load_current_receipt(manifest_path, config=_config()) == receipt
    assert load_receipt(stale_path, manifest_path=manifest_path) == stale
    with pytest.raises(ValueError, match="superseded"):
        load_receipt(stale_path, manifest_path=manifest_path, require_current=True)
    with pytest.raises(ValueError, match="requires a receipt manifest"):
        load_receipt(stale_path, require_current=True)


def test_config_version_and_format_are_validated_before_rescore():
    config = _config()
    config["metric_version"] = "memory.acceptance.v1"
    with pytest.raises(ValueError, match="memory.acceptance.v2"):
        validate_config(config)
    config = _config()
    config["format"] = "replica.memory_acceptance.config.v1"
    with pytest.raises(ValueError, match="Unsupported acceptance config format"):
        validate_config(config)

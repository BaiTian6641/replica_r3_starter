"""DC-R1: native record-readout preparation before retrieval or storage work.

This module compares five read-only conditions under one output grammar and
token/decode policy:

* exact context;
* the exact selected literal record;
* the compressed selected record (a deterministic 8-bit codec);
* a selected record carrying a wrong value;
* no memory record.

It reports exact-literal reconstruction, compressed reconstruction, per-layer
accuracy, "accepted layer" accuracy and complete-layer-group accuracy as
separate outcomes.  A useful readout is established before any learned-retrieval
failure can be interpreted or any storage expansion can be admitted.

The module is preparation only.  It builds a deterministic role-sealed fixture,
validates the protocol, scores already-generated layer outputs per case and
fails closed before native execution.  No torch import, model load, tokenizer
load, optimizer step, gradient, download or native command exists here.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path
import sys
from typing import Any, Callable, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dc_protocol_common_20261027 import (  # noqa: E402
    EVALUATION_CASE_FORBIDDEN_FIELDS, ProtocolError, assert_no_answer_shortcut,
    assert_prompt_matches, assert_routing_is_answer_free, canonical,
    check_work_budget, find_forbidden_fields, normalize_answer,
    preflight_receipt, read_json, read_jsonl, require, seal_manifest,
    sha256_json, tokenizer_binding_sha, validate_baseline_model,
    validate_guards, validate_manifest, validate_tokenizer_binding,
    verify_decoded_text, write_json, write_jsonl,
)

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_ID = "dc_r1_native_value_readout_v1"
CONFIG_FORMAT = "replica.dc_r1_native_value.protocol.v1"
MANIFEST_FORMAT = "replica.dc_r1_native_value.prepared.v1"
CASE_FORMAT = "replica.dc_r1_native_value.eval_case.v1"
CONDITION_FORMAT = "replica.dc_r1_native_value.condition.v1"
LABEL_FORMAT = "replica.dc_r1_native_value.label.v1"
PREDICTION_FORMAT = "replica.dc_r1_native_value.prediction.v1"
SCORE_FORMAT = "replica.dc_r1_native_value.score_report.v1"
CODEC_FORMAT = "replica.dc_r1_native_value.codec.v1"

CONFIG_PATH = ROOT / "configs" / "dc_r1_native_value_protocol_v1.json"
PREPARED_DIR = ROOT / "data" / "dc_r1_native_value_20261027"
DEV_CASES = 12
SEALED_CASES = 4
LAYER_GROUP = (13, 27, 41)
CONDITIONS = ("exact_context", "selected_literal", "compressed_record",
              "wrong_value", "no_memory")
FILES = (
    "dev_cases.jsonl",
    "sealed_final_cases.jsonl",
    "dev_conditions_exact_context.jsonl",
    "dev_conditions_selected_literal.jsonl",
    "dev_conditions_compressed_record.jsonl",
    "dev_conditions_wrong_value.jsonl",
    "dev_conditions_no_memory.jsonl",
    "sealed_final_conditions_exact_context.jsonl",
    "sealed_final_conditions_selected_literal.jsonl",
    "sealed_final_conditions_compressed_record.jsonl",
    "sealed_final_conditions_wrong_value.jsonl",
    "sealed_final_conditions_no_memory.jsonl",
    "sealed_labels_dev.jsonl",
    "sealed_labels_final.jsonl",
    "codec.json",
)
SYLLABLES = (
    "ka", "ro", "mi", "ta", "ne", "su", "vi", "lo", "za", "pe",
    "du", "fi", "go", "ha", "je", "qu", "we", "xi", "yo", "bu",
)
LOSSLESS_CODES = 16
DEV_LOSSY_BUCKETS = {4, 9, 11}
FINAL_LOSSY_BUCKETS = {14}


def _word(*parts: Any) -> str:
    digest = hashlib.sha256(
        canonical(list(parts)).encode("utf-8")).hexdigest()
    indices = [int(digest[index:index + 2], 16) % len(SYLLABLES)
               for index in range(0, 9, 2)]
    return "".join(SYLLABLES[index] for index in indices)


def _codec() -> dict:
    entries = []
    for code in range(LOSSLESS_CODES):
        canonical_literal = _word("r1", "canonical", code)
        alternate_literal = _word("r1", "alternate", code)
        entries.append({
            "code": code,
            "canonical_literal": canonical_literal,
            "members": [canonical_literal, alternate_literal],
        })
    return {
        "format": CODEC_FORMAT,
        "codec_id": "dc_r1_scalar8_v1",
        "description": (
            "Deterministic 8-bit bucket codec.  A bucket has a canonical "
            "literal and one alternate member; the compressed code carries "
            "only the bucket, so alternate members are unrecoverable.  This "
            "is a native compressed-record stand-in, not an answer index."),
        "bits": 8,
        "bucket_count": LOSSLESS_CODES,
        "entries": entries,
    }


def _codec_entry(codec: Mapping[str, Any], code: int) -> dict:
    for entry in codec["entries"]:
        if entry["code"] == code:
            return entry
    raise ProtocolError("codec", f"unknown code {code}")


def decode_compressed_literal(codec: Mapping[str, Any], code: int) -> str:
    return _codec_entry(codec, int(code))["canonical_literal"]


def _record(index: int, *, sealed: bool, codec: Mapping[str, Any]) -> dict:
    prefix = "sealed" if sealed else "dev"
    group = f"r1src_{prefix}_{index:02d}"
    code = index % LOSSLESS_CODES
    entry = _codec_entry(codec, code)
    lossy = index in (FINAL_LOSSY_BUCKETS if sealed else DEV_LOSSY_BUCKETS)
    literal = entry["members"][1] if lossy else entry["canonical_literal"]
    entity = _word("r1-entity", group)
    wrong_entry = _codec_entry(codec, (code + 5) % LOSSLESS_CODES)
    wrong_literal = wrong_entry["canonical_literal"]
    if wrong_literal == literal:
        wrong_entry = _codec_entry(codec, (code + 7) % LOSSLESS_CODES)
        wrong_literal = wrong_entry["canonical_literal"]
    require(wrong_literal != literal, "fixture",
            "wrong-value control equals the target literal")
    source = (f"Source record {group}: the phrase for {entity} is {literal}.")
    query = f"Which phrase belongs to {entity}?"
    return {
        "record_id": f"r1rec_{prefix}_{index:02d}",
        "source_group": group,
        "split_role": "sealed_final" if sealed else "dev",
        "entity": entity,
        "literal": literal,
        "compressed_code": code,
        "codec_id": codec["codec_id"],
        "layer_group": list(LAYER_GROUP),
        "lossy_bucket": lossy,
        "source_text": source,
        "query": query,
        "wrong_value": wrong_literal,
        "source_context_present": False,
        "source_kv_loaded": False,
    }


def _case(record: Mapping[str, Any]) -> dict:
    return {
        "format": CASE_FORMAT,
        "case_id": f"r1case_{record['record_id']}",
        "split_role": record["split_role"],
        "source_group": record["source_group"],
        "entity": record["entity"],
        "query": record["query"],
        "record_id": record["record_id"],
        "layer_group": list(record["layer_group"]),
        "inference_inputs": ["prompt", "selected_record_evidence"],
        "source_context_present": False,
        "source_kv_loaded": False,
    }


def _condition(record: Mapping[str, Any], condition: str,
               codec: Mapping[str, Any]) -> dict:
    case = _case(record)
    if condition == "exact_context":
        prompt = (f"Use the exact source record and answer with only the "
                  f"recorded phrase.\nSource: {record['source_text']}\n"
                  f"Question: {record['query']}")
        evidence = {"kind": "source_text", "source_text": record["source_text"]}
    elif condition == "selected_literal":
        prompt = (f"Use the selected literal record and answer with only the "
                  f"recorded phrase.\nRecord: {record['literal']}\n"
                  f"Question: {record['query']}")
        evidence = {"kind": "literal", "literal": record["literal"]}
    elif condition == "compressed_record":
        prompt = (f"Use the selected compressed record and answer with only "
                  f"the recorded phrase.\nCodec: {codec['codec_id']}\n"
                  f"Code: {record['compressed_code']}\n"
                  f"Question: {record['query']}")
        evidence = {
            "kind": "compressed_code",
            "codec_id": codec["codec_id"],
            "code": record["compressed_code"],
        }
    elif condition == "wrong_value":
        prompt = (f"Use the selected literal record and answer with only the "
                  f"recorded phrase.\nRecord: {record['wrong_value']}\n"
                  f"Question: {record['query']}")
        evidence = {"kind": "literal", "literal": record["wrong_value"]}
    elif condition == "no_memory":
        prompt = ("No memory record is selected. If no authoritative record "
                  f"is available, answer UNKNOWN.\nQuestion: {record['query']}")
        evidence = {"kind": "none"}
    else:
        raise ProtocolError("fixture", f"unknown condition {condition!r}")
    row = {
        "format": CONDITION_FORMAT,
        "case_id": case["case_id"],
        "condition": condition,
        "prompt": prompt,
        "prompt_sha256": sha256_json({"prompt": prompt}),
        "evidence": evidence,
        "source_context_present": condition == "exact_context",
        "source_kv_loaded": False,
        "routing_metadata": {"mode": "fixed", "expert_id": 0},
    }
    return row


def _label(record: Mapping[str, Any]) -> dict:
    return {
        "format": LABEL_FORMAT,
        "case_id": f"r1case_{record['record_id']}",
        "split_role": record["split_role"],
        "source_group": record["source_group"],
        "record_id": record["record_id"],
        "target": record["literal"],
        "compressed_code": record["compressed_code"],
        "lossy_bucket": record["lossy_bucket"],
        "layer_group": list(record["layer_group"]),
        "wrong_value": record["wrong_value"],
    }


def build_fixture(config_path: str | Path, output: str | Path) -> dict:
    config = read_json(config_path)
    require(config.get("protocol_id") == PROTOCOL_ID, "build",
            "config protocol_id differs")
    require(config.get("prepared", {}).get("dev_cases") == DEV_CASES,
            "build", "dev case count differs")
    require(config.get("prepared", {}).get("sealed_cases") == SEALED_CASES,
            "build", "sealed case count differs")
    root = Path(output)
    require(not root.exists(), "build", f"output is not fresh: {root}")
    root.mkdir(parents=True)
    codec = _codec()
    write_json(root / "codec.json", codec)

    dev_records = [_record(index, sealed=False, codec=codec)
                   for index in range(DEV_CASES)]
    final_records = [_record(DEV_CASES + offset, sealed=True, codec=codec)
                     for offset in range(SEALED_CASES)]
    dev_cases = [_case(record) for record in dev_records]
    final_cases = [_case(record) for record in final_records]
    write_jsonl(root / "dev_cases.jsonl", dev_cases)
    write_jsonl(root / "sealed_final_cases.jsonl", final_cases)
    for condition in CONDITIONS:
        write_jsonl(root / f"dev_conditions_{condition}.jsonl",
                    [_condition(record, condition, codec)
                     for record in dev_records])
        write_jsonl(root / f"sealed_final_conditions_{condition}.jsonl",
                    [_condition(record, condition, codec)
                     for record in final_records])
    write_jsonl(root / "sealed_labels_dev.jsonl",
                [_label(record) for record in dev_records])
    write_jsonl(root / "sealed_labels_final.jsonl",
                [_label(record) for record in final_records])

    def reconstruction(records: Sequence[Mapping[str, Any]]) -> float:
        return sum(
            1 for record in records
            if decode_compressed_literal(codec, record["compressed_code"])
            == record["literal"]) / len(records)

    body = {
        "format": MANIFEST_FORMAT,
        "protocol_id": PROTOCOL_ID,
        "declared_role": "deterministic_role_sealed_native_record_fixture",
        "native_computation": False,
        "model_loaded": False,
        "tokenizer_loaded": False,
        "downloads": 0,
        "heldout_opened": False,
        "seed": config.get("seed"),
        "counts": {
            "dev_cases": len(dev_cases),
            "sealed_final_cases": len(final_cases),
            "conditions_per_case": len(CONDITIONS),
            "dev_condition_rows": len(dev_cases) * len(CONDITIONS),
            "sealed_final_condition_rows": len(final_cases) * len(CONDITIONS),
            "dev_labels": len(dev_records),
            "sealed_final_labels": len(final_records),
        },
        "layer_group": list(LAYER_GROUP),
        "codec_id": codec["codec_id"],
        "compressed_reconstruction": {
            "dev": reconstruction(dev_records),
            "sealed_final": reconstruction(final_records),
            "dev_lossy_records": sorted(
                record["record_id"] for record in dev_records
                if record["lossy_bucket"]),
            "sealed_final_lossy_records": sorted(
                record["record_id"] for record in final_records
                if record["lossy_bucket"]),
        },
        "source_groups": {
            "dev": sorted({row["source_group"] for row in dev_cases}),
            "sealed_final": sorted({row["source_group"] for row in final_cases}),
        },
        "files": {name: name for name in FILES},
    }
    manifest = seal_manifest(root, body)
    _validate_fixture(root, config, validate_labels=True,
                      config_binding=False)
    return manifest


def _validate_fixture(root: Path, config: Mapping[str, Any], *,
                      validate_labels: bool,
                      config_binding: bool = True) -> dict:
    manifest = validate_manifest(
        root, config, expected_files=FILES, manifest_format=MANIFEST_FORMAT,
        protocol_id=PROTOCOL_ID, require_config_binding=config_binding)
    codec = read_json(root / "codec.json")
    require(codec.get("format") == CODEC_FORMAT, "fixture",
            "codec format differs")
    require(codec.get("bits") == 8, "fixture", "codec must be 8-bit")
    require(codec.get("codec_id") == manifest.get("codec_id"), "fixture",
            "codec id differs from manifest")
    codes = [entry.get("code") for entry in codec.get("entries", [])]
    require(codes == list(range(LOSSLESS_CODES)), "fixture",
            "codec codes must be exactly 0..15")
    for entry in codec["entries"]:
        require(entry.get("canonical_literal") in entry.get("members", []),
                "fixture", "codec canonical literal is not a member")
        require(len(set(entry.get("members", []))) == len(entry["members"]),
                "fixture", "codec bucket members repeat")

    dev_cases = read_jsonl(root / "dev_cases.jsonl")
    final_cases = read_jsonl(root / "sealed_final_cases.jsonl")
    labels_dev = read_jsonl(root / "sealed_labels_dev.jsonl")
    labels_final = read_jsonl(root / "sealed_labels_final.jsonl")
    require(len(dev_cases) == DEV_CASES, "fixture", "dev case count differs")
    require(len(final_cases) == SEALED_CASES, "fixture",
            "sealed final case count differs")
    require(len(labels_dev) == DEV_CASES, "fixture", "dev label count differs")
    require(len(labels_final) == SEALED_CASES, "fixture",
            "sealed final label count differs")
    require({row["case_id"] for row in dev_cases}
            == {row["case_id"] for row in labels_dev}, "fixture",
            "dev case/label ids differ")
    require({row["case_id"] for row in final_cases}
            == {row["case_id"] for row in labels_final}, "fixture",
            "sealed final case/label ids differ")

    label_by_id = {row["case_id"]: row for row in labels_dev + labels_final}
    for cases, split in ((dev_cases, "dev"), (final_cases, "sealed_final")):
        for case in cases:
            require(case.get("format") == CASE_FORMAT, "fixture",
                    "case format differs")
            require(case.get("split_role") == split, "fixture",
                    "case split_role differs")
            found = find_forbidden_fields(
                case, EVALUATION_CASE_FORBIDDEN_FIELDS)
            require(not found, "fixture",
                    f"evaluation case carries forbidden fields: {found}")
            require(case.get("source_context_present") is False, "fixture",
                    "query case must not carry source context")
            require(case.get("source_kv_loaded") is False, "fixture",
                    "query case must not carry source KV")
            require(case.get("layer_group") == list(LAYER_GROUP), "fixture",
                    "layer group differs from declared ports")
            label = label_by_id[case["case_id"]]
            require(label["source_group"] == case["source_group"], "fixture",
                    "case/label source group differs")
            require(label["record_id"] == case["record_id"], "fixture",
                    "case/label record id differs")
            require(label["target"].casefold()
                    not in case["query"].casefold(), "fixture",
                    "target leaked into the query")

    for condition in CONDITIONS:
        for prefix, cases, split in (
            ("dev", dev_cases, "dev"),
            ("sealed_final", final_cases, "sealed_final"),
        ):
            path = root / f"{prefix}_conditions_{condition}.jsonl"
            rows = read_jsonl(path)
            require(len(rows) == len(cases), "fixture",
                    f"{condition} row count differs for {prefix}")
            require({row["case_id"] for row in rows}
                    == {row["case_id"] for row in cases}, "fixture",
                    f"{condition} case ids differ for {prefix}")
            for row in rows:
                require(row.get("format") == CONDITION_FORMAT, "fixture",
                        "condition row format differs")
                require(row.get("condition") == condition, "fixture",
                        "condition label differs")
                require(row.get("source_kv_loaded") is False, "fixture",
                        "condition row would carry source KV")
                require(row.get("source_context_present")
                        == (condition == "exact_context"), "fixture",
                        "source context declaration differs")
                require(row["prompt_sha256"]
                        == sha256_json({"prompt": row["prompt"]}), "fixture",
                        "condition prompt hash differs")
                assert_routing_is_answer_free(row.get("routing_metadata"))
                label = label_by_id[row["case_id"]]
                evidence = row.get("evidence", {})
                require(evidence.get("kind") in (
                    "source_text", "literal", "compressed_code", "none"),
                    "fixture", "unknown evidence kind")
                if condition == "exact_context":
                    require(evidence.get("kind") == "source_text", "fixture",
                            "exact context must supply source text")
                    require(label["target"] in evidence.get("source_text", ""),
                            "fixture", "exact context omits the target")
                elif condition == "selected_literal":
                    require(evidence.get("kind") == "literal"
                            and evidence.get("literal") == label["target"],
                            "fixture", "selected literal differs from target")
                elif condition == "compressed_record":
                    require(evidence.get("kind") == "compressed_code", "fixture",
                            "compressed evidence kind differs")
                    require(evidence.get("code") == label["compressed_code"],
                            "fixture", "compressed code differs from label")
                    require(evidence.get("codec_id") == codec["codec_id"],
                            "fixture", "compressed codec id differs")
                    require(label["target"].casefold()
                            not in row["prompt"].casefold(), "fixture",
                            "compressed prompt leaks the target literal")
                elif condition == "wrong_value":
                    require(evidence.get("kind") == "literal", "fixture",
                            "wrong-value evidence kind differs")
                    require(evidence.get("literal") != label["target"],
                            "fixture", "wrong-value control equals target")
                    require(evidence.get("literal") == label["wrong_value"],
                            "fixture", "wrong-value control is not sealed")
                    require(label["target"].casefold()
                            not in row["prompt"].casefold(), "fixture",
                            "wrong-value prompt leaks the target literal")
                elif condition == "no_memory":
                    require(evidence == {"kind": "none"}, "fixture",
                            "no-memory evidence must be empty")
                    require(label["target"].casefold()
                            not in row["prompt"].casefold(), "fixture",
                            "no-memory prompt leaks the target literal")

    def reconstruction(records: Sequence[Mapping[str, Any]]) -> float:
        return sum(
            1 for record in records
            if decode_compressed_literal(codec, record["compressed_code"])
            == record["target"]) / len(records)

    dev_recs = [dict(label) for label in labels_dev]
    final_recs = [dict(label) for label in labels_final]
    expected_dev = 1.0 - len(DEV_LOSSY_BUCKETS) / DEV_CASES
    expected_final = 1.0 - len(FINAL_LOSSY_BUCKETS) / SEALED_CASES
    require(reconstruction(dev_recs) == expected_dev, "fixture",
            "dev compressed reconstruction differs from declaration")
    require(reconstruction(final_recs) == expected_final, "fixture",
            "sealed final compressed reconstruction differs")
    require(expected_dev == config["compressed_reconstruction"]["dev"],
            "config", "config dev reconstruction declaration differs")
    require(expected_final
            == config["compressed_reconstruction"]["sealed_final"], "config",
            "config sealed reconstruction declaration differs")

    dev_groups = {row["source_group"] for row in dev_cases}
    final_groups = {row["source_group"] for row in final_cases}
    require(not (dev_groups & final_groups), "fixture",
            "sealed final groups overlap development groups")
    require(len(dev_groups) == DEV_CASES, "fixture",
            "development source groups are not independent")
    require(len(final_groups) == SEALED_CASES, "fixture",
            "sealed final source groups are not independent")
    if validate_labels:
        require(len({row["case_id"] for row in labels_dev}) == DEV_CASES,
                "fixture", "dev labels duplicate a case")
    return {
        "format": "replica.dc_r1_native_value.validation.v1",
        "protocol_id": PROTOCOL_ID,
        "status": "valid",
        "manifest_sha256": manifest["manifest_sha256"],
        "counts": manifest["counts"],
        "compressed_reconstruction": manifest["compressed_reconstruction"],
        "native_computation": False,
        "labels_read_for_construction_audit": bool(validate_labels),
        "heldout_opened": False,
    }


# ---------------------------------------------------------------------------
# Protocol config validation


def validate_config(config: Mapping[str, Any]) -> dict:
    require(config.get("format") == CONFIG_FORMAT, "config",
            "config format differs")
    require(config.get("protocol_id") == PROTOCOL_ID, "config",
            "protocol_id differs")
    require(config.get("study") == "DC-R1", "config", "study must be DC-R1")
    classification = config.get("classification", {})
    require(classification.get("preparation_only") is True, "config",
            "preparation_only must be true")
    require(classification.get("native_execution_allowed") is False, "config",
            "native execution must be false before authorization")
    require(classification.get("model_execution") is False, "config",
            "no model execution in preparation")
    require(classification.get("tokenizer_load") is False, "config",
            "no tokenizer load in preparation")
    require(classification.get("learned_retrieval") is False, "config",
            "learned retrieval remains closed")
    require(classification.get("storage_expansion") is False, "config",
            "storage expansion remains closed")
    validate_guards(config)
    validate_baseline_model(config)
    validate_tokenizer_binding(config.get("tokenizer_binding"))
    representation = config.get("representation", {})
    require(representation.get("synthetic_feature_injection") is False,
            "representation", "synthetic feature injection is forbidden")
    require(representation.get("answer_index_bypass") is False,
            "representation", "answer-index bypass is forbidden")
    require(representation.get("compressed_code_is_not_an_answer_index")
            is True, "representation",
            "the compressed code must not be an answer index")
    require(representation.get("source_kv_loaded") is False,
            "representation", "source KV must be absent")
    parity = config.get("baseline_parity", {})
    require(parity.get("no_extension_is_default") is True, "baseline_parity",
            "no-extension default is required")
    require(parity.get("disabled_extension_preserves_native_logits") is True,
            "baseline_parity", "native logit parity is required")
    require(parity.get("disabled_extension_preserves_native_greedy_tokens")
            is True, "baseline_parity", "native greedy parity is required")
    require(parity.get("logit_rtol") == 0.0 and parity.get("logit_atol") == 0.0,
            "baseline_parity", "exact baseline parity is required")
    require(parity.get("no_memory_condition_abstains_instead_of_inventing")
            is True, "baseline_parity",
            "no-memory parity must abstain rather than invent")

    integration = config.get("integration_only_fixture", {})
    require(integration.get("protocol_id") == "value_readout_abstention_v1",
            "integration_only_fixture", "22-class identity differs")
    require(integration.get("prepared_root")
            == "data/value_readout_abstention_20261027/fixture02",
            "integration_only_fixture", "22-class prepared root differs")
    require(integration.get("status") == "integration_only", "config",
            "the 22-class fixture must remain explicitly integration-only")
    require(integration.get("modified") is False, "config",
            "the 22-class fixture must not be modified")
    require(integration.get("historical_evidence_preserved") is True, "config",
            "historical 22-class evidence must be preserved")
    require(integration.get("used_as_parent") is False, "config",
            "the 22-class fixture cannot be a parent")
    require(integration.get("used_as_scoring_oracle") is False, "config",
            "the 22-class fixture cannot be a scoring oracle")
    require(integration.get("claims_allowed") == [], "config",
            "the 22-class fixture allows no claim here")

    require(config.get("hypothesis", {}).get("id") == "DC-R1-H1", "config",
            "hypothesis id differs")
    conditions = config.get("conditions", [])
    require([row.get("condition") for row in conditions] == list(CONDITIONS),
            "conditions", "condition order/identity differs")
    for row in conditions:
        require(row.get("source_context_present")
                == (row.get("condition") == "exact_context"), "conditions",
                "source context declaration differs")
        require(row.get("source_kv_loaded") is False, "conditions",
                "source KV must be absent")
        require(row.get("per_case_scoring") is True, "conditions",
                "per-case scoring is required")
        require(row.get("identical_output_grammar") is True, "conditions",
                "all conditions must share the output grammar")
        require(row.get("same_decode_policy") is True, "conditions",
                "all conditions must share the decode policy")
    require(config.get("conditions_use_identical_scorer") is True,
            "conditions", "all conditions must use one scorer")

    readout = config.get("readout_gate", {})
    require(readout.get("useful_readout_required_before_retrieval") is True,
            "readout_gate", "readout must precede retrieval interpretation")
    require(readout.get("learned_retrieval_interpreted") is False,
            "readout_gate", "learned retrieval is not evaluated here")
    require(readout.get("learned_retrieval_status")
            == "blocked_until_useful_readout", "readout_gate",
            "learned retrieval status differs")
    require(readout.get("storage_expansion_admitted") is False,
            "readout_gate", "storage expansion cannot be admitted here")
    require(readout.get("storage_expansion_status")
            == "blocked_until_useful_readout", "readout_gate",
            "storage expansion status differs")
    require(readout.get("accepted_layer_and_complete_group_reported") is True,
            "readout_gate", "both layer metrics are required")
    require(readout.get("per_layer_reporting") is True, "readout_gate",
            "per-layer reporting is required")
    require(readout.get("exact_literal_and_compressed_reconstruction_reported")
            is True, "readout_gate",
            "literal/compressed reconstruction must be reported")
    require(readout.get("thresholds_condition_independent") is True,
            "readout_gate", "thresholds may not depend on individual targets")

    anti = config.get("anti_shortcut", {})
    require(anti.get("answer_index_bypass_allowed") is False, "anti_shortcut",
            "answer-index bypass is forbidden")
    require(anti.get("answer_bearing_routing_metadata") is False,
            "anti_shortcut", "answer-bearing routing metadata is forbidden")
    require(anti.get("corrupted_controls_must_fail") is True, "anti_shortcut",
            "corrupted controls must fail")
    require(anti.get("controls_use_identical_scorer") is True,
            "anti_shortcut", "controls must use the identical scorer")
    require(anti.get("compressed_code_is_not_an_answer_index") is True,
            "anti_shortcut", "compressed codes must not be answer indexes")
    require(anti.get("target_in_query_allowed") is False, "anti_shortcut",
            "targets may not be placed in the query")
    require(anti.get("target_in_condition_evidence") == [
        "exact_context", "selected_literal", "wrong_value"], "anti_shortcut",
        "target-bearing condition evidence declaration differs")

    prepared = config.get("prepared", {})
    require(prepared.get("root") == str(PREPARED_DIR.relative_to(ROOT)),
            "prepared", "prepared root differs")
    require(prepared.get("dev_cases") == DEV_CASES, "prepared",
            "dev case count differs")
    require(prepared.get("sealed_cases") == SEALED_CASES, "prepared",
            "sealed case count differs")
    require(prepared.get("layer_group") == list(LAYER_GROUP), "prepared",
            "layer group differs")
    counts = prepared.get("counts", {})
    require(counts.get("dev_cases") == DEV_CASES, "prepared",
            "dev count differs")
    require(counts.get("sealed_final_cases") == SEALED_CASES, "prepared",
            "sealed count differs")
    require(counts.get("dev_condition_rows")
            == DEV_CASES * len(CONDITIONS), "prepared",
            "dev condition count differs")
    require(counts.get("sealed_final_condition_rows")
            == SEALED_CASES * len(CONDITIONS), "prepared",
            "sealed condition count differs")
    require(counts.get("optimizer_steps") == 0, "prepared",
            "DC-R1 is read-only and has zero optimizer steps")
    require(prepared.get("manifest_sha256") is None
            or isinstance(prepared.get("manifest_sha256"), str),
            "prepared", "manifest digest must be null or a string")

    reconstruction = config.get("compressed_reconstruction", {})
    require(reconstruction.get("dev") == 0.75, "config",
            "dev compressed reconstruction declaration differs")
    require(reconstruction.get("sealed_final") == 0.75, "config",
            "sealed compressed reconstruction declaration differs")

    evaluation = config.get("evaluation", {})
    require(evaluation.get("generated_answer_scoring") is True, "evaluation",
            "generated-answer scoring is required")
    require(evaluation.get("tokenizer_and_decode_binding_required") is True,
            "evaluation", "tokenizer/decode binding is required")
    require(evaluation.get("sealed_final_one_shot") is True, "evaluation",
            "sealed final must be one-shot")
    require(evaluation.get("fit_then_dev_then_sealed_final") is True,
            "evaluation", "evaluation order differs")
    gates = evaluation.get("gates", {})
    for key in ("exact_context_accepted_min",
                "selected_literal_accepted_min",
                "selected_literal_complete_min",
                "compressed_reconstruction_min",
                "compressed_accepted_min", "compressed_complete_min",
                "no_memory_abstain_min", "no_memory_non_abstain_max",
                "wrong_value_correct_target_max",
                "no_layer_acceptance_reporting"):
        require(key in gates, "evaluation", f"missing gate {key}")
    require(gates["no_layer_acceptance_reporting"] is True, "evaluation",
            "per-layer reporting cannot be disabled")

    budget = config.get("budget", {})
    check_work_budget({}, budget)
    physical = budget.get("physical_limits", {})
    require(physical.get("max_optimizer_steps") == 0, "budget",
            "DC-R1 must declare zero optimizer steps")
    require(physical.get("max_backward_calls") == 0, "budget",
            "DC-R1 must declare zero backward calls")
    require(physical.get("max_forward_calls")
            == DEV_CASES * len(CONDITIONS) * len(LAYER_GROUP)
            + SEALED_CASES * len(CONDITIONS) * len(LAYER_GROUP), "budget",
            "forward ceiling differs from the declared readout workload")
    require(budget.get("preparation", {}).get("native_jobs") == 0, "budget",
            "preparation must declare zero native jobs")
    require(budget.get("authorized_run_ceiling", {}).get("model_loads") == 1,
            "budget", "exactly one model load may be authorized")
    require(budget.get("authorized_run_ceiling", {}).get("downloads") == 0,
            "budget", "downloads remain forbidden")

    launch = config.get("launch", {})
    require(launch.get("review_only") is True, "launch",
            "launch declaration must be review-only")
    require(launch.get("native_execution") is False, "launch",
            "native execution must be false")
    require(launch.get("requires_authorization_receipt") is True, "launch",
            "separate authorization is required")
    require(launch.get("fresh_output_path") is True, "launch",
            "fresh output path is required")
    require(launch.get("on_missing_receipt") == "refuse_nonzero", "launch",
            "missing authorization must refuse nonzero")
    return {
        "status": "valid",
        "protocol_id": PROTOCOL_ID,
        "conditions": list(CONDITIONS),
        "layer_group": list(LAYER_GROUP),
    }


def validate_prepared(prepared: str | Path = PREPARED_DIR,
                      config_path: str | Path = CONFIG_PATH) -> dict:
    config = read_json(config_path)
    validate_config(config)
    return _validate_fixture(Path(prepared), config, validate_labels=True)


# ---------------------------------------------------------------------------
# Per-case readout scoring


def _layer_correct(text: str, target: str) -> bool:
    return normalize_answer(text) == normalize_answer(target)


def _score_condition_prediction(
        row: Mapping[str, Any], condition_row: Mapping[str, Any],
        case: Mapping[str, Any], label: Mapping[str, Any],
        binding: Mapping[str, Any],
        decode_fn: Callable[[Sequence[int]], str]) -> dict:
    require(row.get("format") == PREDICTION_FORMAT, "prediction",
            "prediction format differs")
    require(row.get("protocol_id") == PROTOCOL_ID, "prediction",
            "prediction protocol differs")
    assert_no_answer_shortcut(row)
    assert_routing_is_answer_free(row.get("routing_metadata", {}))
    assert_prompt_matches(condition_row, row)
    condition = condition_row["condition"]
    require(row.get("condition") == condition, "prediction",
            "prediction condition differs")
    require(row.get("source_context_present")
            == condition_row["source_context_present"], "prediction",
            "prediction source-context flag differs from the condition")
    require(row.get("source_kv_loaded") is False, "prediction",
            "source KV must be absent in every DC-R1 condition")
    require(row.get("prompt_source_spans") == 0, "prediction",
            "source prompt spans must not be attached in DC-R1 readout")
    outputs = row.get("layer_outputs")
    require(isinstance(outputs, list), "prediction",
            "layer_outputs must be a list")
    require([item.get("layer") for item in outputs]
            == list(case["layer_group"]), "prediction",
            "layer outputs do not match the declared layer group")
    require(len({item.get("layer") for item in outputs}) == len(outputs),
            "prediction", "layer outputs repeat a layer")
    target = label["target"]
    layer_rows: list[dict] = []
    for item in outputs:
        require(isinstance(item, Mapping), "prediction",
                "layer output must be an object")
        generated = {
            "generated_ids": item.get("generated_ids"),
            "generated_text": item.get("generated_text"),
            "empty_generation": item.get("empty_generation"),
            "tokenizer_binding_sha256": row.get("tokenizer_binding_sha256"),
            "decode_policy": row.get("decode_policy"),
        }
        text = verify_decoded_text(generated, binding, decode_fn)
        abstain = item.get("abstain")
        require(isinstance(abstain, bool), "prediction",
                "layer abstain flag must be boolean")
        require(abstain == (normalize_answer(text) == "unknown"),
                "prediction",
                "abstain flag and generated UNKNOWN text are inconsistent")
        layer_rows.append({
            "layer": item["layer"],
            "generated_text": text,
            "abstained": abstain,
            "correct": _layer_correct(text, target),
            "empty_generation": text == "",
        })
    accepted = any(item["correct"] for item in layer_rows)
    complete = all(item["correct"] for item in layer_rows)
    followed_wrong = (
        condition == "wrong_value" and any(
            normalize_answer(item["generated_text"])
            == normalize_answer(condition_row["evidence"].get("literal", ""))
            for item in layer_rows))
    return {
        "case_id": case["case_id"],
        "condition": condition,
        "source_group": case["source_group"],
        "target": target,
        "layers": layer_rows,
        "accepted_layer_correct": accepted,
        "complete_group_correct": complete,
        "any_abstain": any(item["abstained"] for item in layer_rows),
        "all_abstain": all(item["abstained"] for item in layer_rows),
        "followed_wrong_literal": followed_wrong,
    }


def _metric_block(rows: Sequence[Mapping[str, Any]],
                  layer_group: Sequence[int]) -> dict:
    count = len(rows)
    require(count > 0, "scoring", "empty metric block")
    return {
        "cases": count,
        "accepted_layer_accuracy": (
            sum(1 for row in rows if row["accepted_layer_correct"]) / count),
        "complete_group_accuracy": (
            sum(1 for row in rows if row["complete_group_correct"]) / count),
        "abstain_rate": (
            sum(1 for row in rows if row["any_abstain"]) / count),
        "all_abstain_rate": (
            sum(1 for row in rows if row["all_abstain"]) / count),
        "wrong_value_follow_rate": (
            sum(1 for row in rows if row["followed_wrong_literal"]) / count),
        "per_layer_accuracy": {
            str(layer): (
                sum(1 for row in rows
                    for item in row["layers"] if item["layer"] == layer
                    and item["correct"])
                / count)
            for layer in layer_group
        },
    }


def score_r1(
        predictions_path: str | Path,
        *,
        dev_cases_path: str | Path,
        dev_labels_path: str | Path,
        dev_conditions: Mapping[str, str | Path],
        config_path: str | Path = CONFIG_PATH,
        decode_fn: Callable[[Sequence[int]], str]) -> dict:
    """Score already-generated DC-R1 layer outputs.  No tokenizer is loaded."""
    config = read_json(config_path)
    validate_config(config)
    binding = validate_tokenizer_binding(config["tokenizer_binding"])
    require(callable(decode_fn), "scoring",
            "a sealed-tokenizer decode function is required")
    cases = {row["case_id"]: row for row in read_jsonl(dev_cases_path)}
    labels = {row["case_id"]: row for row in read_jsonl(dev_labels_path)}
    require(set(cases) == set(labels), "scoring",
            "dev case/label coverage differs")
    condition_rows: dict[tuple[str, str], dict] = {}
    for condition in CONDITIONS:
        path = dev_conditions.get(condition)
        require(path is not None, "scoring",
                f"missing condition file for {condition}")
        for row in read_jsonl(path):
            key = (condition, row["case_id"])
            require(key not in condition_rows, "scoring",
                    f"duplicate condition row {key}")
            condition_rows[key] = row
    require(set(condition_rows) == {
        (condition, case_id) for condition in CONDITIONS for case_id in cases},
        "scoring", "condition coverage differs")

    seen: set[tuple[str, str]] = set()
    scored: list[dict] = []
    for row in read_jsonl(predictions_path):
        condition = row.get("condition")
        case_id = row.get("case_id")
        require(condition in CONDITIONS, "scoring",
                f"unknown condition {condition!r}")
        require(case_id in cases, "scoring", f"unknown case {case_id!r}")
        key = (condition, case_id)
        require(key not in seen, "scoring", f"duplicate prediction {key}")
        seen.add(key)
        scored.append(_score_condition_prediction(
            row, condition_rows[key], cases[case_id], labels[case_id],
            binding, decode_fn))
    expected = {(condition, case_id) for condition in CONDITIONS
                for case_id in cases}
    missing = sorted(expected - seen)
    require(not missing, "scoring", f"missing predictions: {missing[:8]}")

    by_condition = {condition: [row for row in scored
                               if row["condition"] == condition]
                    for condition in CONDITIONS}
    metrics = {
        condition: _metric_block(rows, LAYER_GROUP)
        for condition, rows in by_condition.items()
    }
    codec = read_json(Path(dev_cases_path).parent / "codec.json")
    reconstruction = sum(
        1 for label in labels.values()
        if decode_compressed_literal(codec, label["compressed_code"])
        == label["target"]) / len(labels)
    exact_literal_available = 1.0
    compressed_readout = metrics["compressed_record"]
    selected_readout = metrics["selected_literal"]
    no_memory = metrics["no_memory"]
    wrong_value = metrics["wrong_value"]
    gates_cfg = config["evaluation"]["gates"]
    gate_results: dict[str, dict] = {
        "exact_context_accepted": {
            "value": metrics["exact_context"]["accepted_layer_accuracy"],
            "min": gates_cfg["exact_context_accepted_min"],
            "passed": metrics["exact_context"]["accepted_layer_accuracy"]
            >= gates_cfg["exact_context_accepted_min"]},
        "selected_literal_accepted": {
            "value": selected_readout["accepted_layer_accuracy"],
            "min": gates_cfg["selected_literal_accepted_min"],
            "passed": selected_readout["accepted_layer_accuracy"]
            >= gates_cfg["selected_literal_accepted_min"]},
        "selected_literal_complete": {
            "value": selected_readout["complete_group_accuracy"],
            "min": gates_cfg["selected_literal_complete_min"],
            "passed": selected_readout["complete_group_accuracy"]
            >= gates_cfg["selected_literal_complete_min"]},
        "compressed_reconstruction": {
            "value": reconstruction,
            "min": gates_cfg["compressed_reconstruction_min"],
            "passed": reconstruction >= gates_cfg["compressed_reconstruction_min"]},
        "compressed_accepted": {
            "value": compressed_readout["accepted_layer_accuracy"],
            "min": gates_cfg["compressed_accepted_min"],
            "passed": compressed_readout["accepted_layer_accuracy"]
            >= gates_cfg["compressed_accepted_min"]},
        "compressed_complete": {
            "value": compressed_readout["complete_group_accuracy"],
            "min": gates_cfg["compressed_complete_min"],
            "passed": compressed_readout["complete_group_accuracy"]
            >= gates_cfg["compressed_complete_min"]},
        "no_memory_abstain": {
            "value": no_memory["abstain_rate"],
            "min": gates_cfg["no_memory_abstain_min"],
            "passed": no_memory["abstain_rate"]
            >= gates_cfg["no_memory_abstain_min"]},
        "no_memory_non_abstain": {
            "value": 1.0 - no_memory["abstain_rate"],
            "max": gates_cfg["no_memory_non_abstain_max"],
            "passed": 1.0 - no_memory["abstain_rate"]
            <= gates_cfg["no_memory_non_abstain_max"]},
    }
    wrong_correct = sum(
        1 for row in by_condition["wrong_value"]
        for item in row["layers"] if item["correct"]) / (
            len(by_condition["wrong_value"]) * len(LAYER_GROUP))
    gate_results["wrong_value_correct_target"] = {
        "value": wrong_correct,
        "max": gates_cfg["wrong_value_correct_target_max"],
        "passed": wrong_correct <= gates_cfg["wrong_value_correct_target_max"],
    }
    require(metrics["wrong_value"]["accepted_layer_accuracy"]
            <= gates_cfg["selected_literal_accepted_min"], "scoring",
            "wrong-value control unexpectedly passes the readout gate")
    require(config["readout_gate"]["learned_retrieval_interpreted"] is False,
            "readout_gate", "learned retrieval must remain uninterpreted")
    require(config["readout_gate"]["storage_expansion_admitted"] is False,
            "readout_gate", "storage expansion must remain blocked")

    readout_useful = all(row["passed"] for row in gate_results.values())
    return {
        "format": SCORE_FORMAT,
        "protocol_id": PROTOCOL_ID,
        "status": "readout_useful" if readout_useful else "readout_not_useful",
        "native_computation": False,
        "answer_index_used": False,
        "tokenizer_binding_sha256": tokenizer_binding_sha(binding),
        "conditions": metrics,
        "reconstruction": {
            "exact_literal_available_rate": exact_literal_available,
            "compressed_reconstruction_accuracy": reconstruction,
            "literal_minus_compressed": (exact_literal_available
                                        - reconstruction),
            "per_case": [
                {
                    "case_id": label["case_id"],
                    "target": label["target"],
                    "compressed_code": label["compressed_code"],
                    "reconstructed_literal": decode_compressed_literal(
                        codec, label["compressed_code"]),
                    "exact": (decode_compressed_literal(
                        codec, label["compressed_code"]) == label["target"]),
                    "lossy_bucket": label["lossy_bucket"],
                }
                for label in sorted(labels.values(),
                                    key=lambda row: row["case_id"])
            ],
        },
        "accepted_layer_vs_complete_group": {
            condition: {
                "accepted_layer_accuracy":
                    metrics[condition]["accepted_layer_accuracy"],
                "complete_group_accuracy":
                    metrics[condition]["complete_group_accuracy"],
                "gap": metrics[condition]["accepted_layer_accuracy"]
                - metrics[condition]["complete_group_accuracy"],
                "per_layer_accuracy":
                    metrics[condition]["per_layer_accuracy"],
            }
            for condition in CONDITIONS
        },
        "gates": gate_results,
        "readout_useful": readout_useful,
        "learned_retrieval_interpretable": False,
        "learned_retrieval_status": "blocked_or_not_evaluated",
        "storage_expansion_allowed": False,
        "storage_expansion_status": "blocked_until_useful_readout_and_new_protocol",
        "controls_use_identical_scorer": True,
        "conditions_use_identical_scorer": True,
        "source_kv_loaded": False,
    }


def preflight(config_path: str | Path = CONFIG_PATH,
              prepared: str | Path = PREPARED_DIR,
              *, launch: bool = False,
              authorization: str | Path | None = None) -> dict:
    config = read_json(config_path)
    validate_config(config)
    manifest_path = Path(prepared) / "manifest.json"
    manifest = read_json(manifest_path)
    receipt, code = preflight_receipt(
        config_path=config_path, manifest_path=manifest_path, config=config,
        protocol_id=PROTOCOL_ID,
        manifest_sha256=manifest.get("manifest_sha256"),
        authorization_path=authorization, launch=launch)
    receipt["exit_code"] = code
    return receipt

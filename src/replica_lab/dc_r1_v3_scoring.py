"""DC-R1 v3 scoring and independently verified development evidence."""
from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Callable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from dc_protocol_common_20261027 import (
    canonical, normalize_answer, read_json, read_jsonl, require, sha256_file,
    sha256_json, tokenizer_binding_sha, validate_baseline_model, validate_guards,
    validate_manifest, validate_tokenizer_binding, verify_decoded_text,
)
import dc_r1_native_value_v2_20261027 as legacy
from replica_lab import artifact_publication as publication

PROTOCOL_ID = "dc_r1_native_value_readout_v3"
PREDICTION_FORMAT = "replica.dc_r1_native_value.prediction.v3"
SCORE_FORMAT = "replica.dc_r1_native_value.score_report.v3"
CONFIG_PATH = ROOT / "configs/dc_r1_native_value_protocol_v3.json"
PREPARED_DIR = ROOT / "data/dc_r1_native_value_v2_20261027"
CONDITIONS = ("exact_context", "selected_literal", "compressed_record",
              "wrong_value", "no_memory")
LAYERS = (13, 27, 41)


def load_config(path: str | Path = CONFIG_PATH,
                prepared: str | Path | None = None) -> dict:
    config = read_json(path)
    require(config.get("format") == "replica.dc_r1_native_value.protocol.v3",
            "config", "format differs")
    require(config.get("protocol_id") == PROTOCOL_ID, "config", "protocol id differs")
    validate_guards(config)
    validate_baseline_model(config)
    validate_tokenizer_binding(config["tokenizer_binding"])
    parent = config["parent"]
    require(sha256_file(ROOT / "configs/dc_r1_native_value_protocol_v2.json")
            == parent["config_sha256"], "lineage", "v2 config changed")
    require(parent["prepared_manifest_sha256"]
            == config["prepared"]["manifest_sha256"], "lineage",
            "prepared manifest parent binding differs")
    prepared_root = Path(prepared) if prepared is not None else PREPARED_DIR
    manifest = read_json(prepared_root / "manifest.json")
    body = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    require(manifest["manifest_sha256"] == sha256_json(body), "prepared",
            "manifest self digest differs")
    require(manifest["manifest_sha256"] == config["prepared"]["manifest_sha256"],
            "prepared", "config does not bind manifest")
    return config


def validate_prepared(config: Mapping[str, Any], *, defer_final_labels: bool,
                      prepared: str | Path = PREPARED_DIR) -> dict:
    prepared = Path(prepared)
    return validate_manifest(
        prepared, config, expected_files=legacy.FILES,
        manifest_format=legacy.MANIFEST_FORMAT, protocol_id=legacy.PROTOCOL_ID,
        skip_hash_files=(("sealed_labels_final.jsonl",) if defer_final_labels else ()))


def _cases(prepared: Path, split: str) -> tuple[dict, dict, dict]:
    prefix = "dev" if split == "dev" else "sealed_final"
    cases = {row["case_id"]: row for row in read_jsonl(prepared / f"{prefix}_cases.jsonl")}
    labels_name = "sealed_labels_dev.jsonl" if split == "dev" else "sealed_labels_final.jsonl"
    labels = {row["case_id"]: row for row in read_jsonl(prepared / labels_name)}
    conditions = {}
    for condition in CONDITIONS:
        for row in read_jsonl(prepared / f"{prefix}_conditions_{condition}.jsonl"):
            conditions[(condition, row["case_id"])] = row
    require(set(cases) == set(labels), "scoring", "case/label coverage differs")
    require(set(conditions) == {(condition, case_id)
                               for condition in CONDITIONS for case_id in cases},
            "scoring", "condition coverage differs")
    return cases, labels, conditions


def _layer_evidence(item: Mapping[str, Any], *, condition: str, case_id: str,
                    target: str, binding: Mapping[str, Any], decode_fn: Callable,
                    decode_metadata: Mapping[str, Any]) -> dict:
    raw = item.get("raw_generated_ids")
    content = item.get("generated_ids")
    terminal = item.get("terminal_stop_ids")
    for name, value in (("raw_generated_ids", raw), ("generated_ids", content),
                        ("terminal_stop_ids", terminal)):
        require(isinstance(value, list) and all(
            isinstance(token, int) and not isinstance(token, bool) and token >= 0
            for token in value), "prediction", f"{name} must be nonnegative ints")
    stops = set(binding["decode_policy"]["stop_tokens"])
    require(raw == content + terminal, "prediction",
            "raw IDs must equal scored content followed by terminal stops")
    require(not any(token in stops for token in content), "prediction",
            "terminal stops cannot enter scored content")
    require(len(terminal) <= 1 and all(token in stops for token in terminal),
            "prediction", "invalid terminal stop stream")
    stopped = item.get("stopped")
    require(stopped is bool(terminal), "prediction",
            "stopped flag does not match terminal stop stream")
    reason = item.get("stop_reason")
    require(isinstance(reason, str) and reason, "prediction", "stop reason missing")
    if terminal:
        require(reason == "eos", "prediction", "terminal stop must use eos reason")
    generated = dict(item)
    generated.update(decode_metadata)
    text = verify_decoded_text(generated, binding, decode_fn)
    work = item.get("work")
    require(isinstance(work, Mapping), "physical_work", "layer work missing")
    prompt_tokens = work.get("prefill_tokens")
    require(isinstance(prompt_tokens, int) and not isinstance(prompt_tokens, bool)
            and prompt_tokens > 0, "physical_work", "prefill tokens must be positive")
    count = len(raw)
    expected = {
        "generation_requests": 1,
        "generation_prefill_forwards": 1,
        "generation_decode_forwards": max(count - 1, 0),
        "autoregressive_forwards": count,
        "forward_calls": count,
        "generated_tokens": count,
        "prefill_tokens": prompt_tokens,
        "processed_positions": count * prompt_tokens + count * (count - 1) // 2,
        "layer_readout_streams": 1,
    }
    require(dict(work) == expected, "physical_work",
            f"work differs for {condition}/{case_id}: {dict(work)} != {expected}")
    require(item.get("physical_stream_id") == f"{condition}/{case_id}/layer:{item.get('layer')}",
            "physical_work", "layer stream identity differs")
    completion_valid = bool(terminal) and reason == "eos"
    normalized = normalize_answer(text)
    target_normalized = normalize_answer(target)
    correct = completion_valid and normalized != "" and normalized == target_normalized
    unknown = completion_valid and normalized == "unknown"
    invented = completion_valid and normalized not in ("", "unknown", target_normalized)
    return {
        "layer": item["layer"],
        "raw_generated_ids": list(raw),
        "terminal_stop_ids": list(terminal),
        "generated_ids": list(content),
        "generated_text": text,
        "empty_generation": text == "",
        "stopped": stopped,
        "stop_reason": reason,
        "completion_valid": completion_valid,
        "correct": correct,
        "unknown": unknown,
        "invented_non_target": invented,
        "work": expected,
        "physical_stream_id": item["physical_stream_id"],
    }


def _metric(rows: Sequence[Mapping[str, Any]]) -> dict:
    count = len(rows)
    require(count > 0, "scoring", "empty condition")
    layers = [item for row in rows for item in row["layers"]]
    per_layer = {}
    for layer in LAYERS:
        selected = [item for item in layers if item["layer"] == layer]
        per_layer[str(layer)] = {
            "accepted_accuracy": sum(item["correct"] for item in selected) / count,
            "non_abstention_rate": sum(not item["unknown"] for item in selected) / count,
            "hallucination_rate": sum(item["invented_non_target"] for item in selected) / count,
            "empty_generation_rate": sum(item["empty_generation"] for item in selected) / count,
            "completion_valid_rate": sum(item["completion_valid"] for item in selected) / count,
        }
    return {
        "cases": count,
        "layer_outputs": len(layers),
        "accepted_layer_accuracy": sum(row["accepted"] for row in rows) / count,
        "complete_group_accuracy": sum(row["complete"] for row in rows) / count,
        "group_abstention_rate": sum(row["group_abstained"] for row in rows) / count,
        "group_non_abstention_rate": sum(not row["group_abstained"] for row in rows) / count,
        "hallucination_rate": sum(any(item["invented_non_target"] for item in row["layers"])
                                  for row in rows) / count,
        "per_layer": per_layer,
    }


def _physical(rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any],
              enforce_budget: bool) -> dict:
    names = ("generation_requests", "generation_prefill_forwards",
             "generation_decode_forwards", "autoregressive_forwards",
             "forward_calls", "generated_tokens", "prefill_tokens",
             "processed_positions", "layer_readout_streams")
    layers = [item for row in rows for item in row["layers"]]
    totals = {name: sum(item["work"][name] for item in layers) for name in names}
    require(totals["generation_requests"] == len(layers)
            == totals["generation_prefill_forwards"] == totals["layer_readout_streams"],
            "physical_work", "request/prefill/stream counts differ")
    require(totals["autoregressive_forwards"] == totals["forward_calls"]
            == totals["generated_tokens"], "physical_work", "forward/token counts differ")
    if enforce_budget:
        limits = config["budget"]["physical_limits"]
        aliases = {"generation_prefill_forwards": "prefill_forwards",
                   "generation_decode_forwards": "decode_forwards",
                   "autoregressive_forwards": "forward_calls"}
        for name, value in totals.items():
            budget_name = aliases.get(name, name)
            require(value <= limits[f"max_{budget_name}"], "physical_work",
                    f"{name}={value} exceeds ceiling {limits[f'max_{budget_name}']}")
    return {"complete": True, "totals": totals,
            "budget_check": "passed" if enforce_budget else "not_applicable_preparation"}


def _gates(metrics: Mapping[str, Any], config: Mapping[str, Any]) -> dict:
    thresholds = config["evaluation"]["gates"]
    no_memory = metrics["no_memory"]
    gates = {
        "exact_context_accepted": metrics["exact_context"]["accepted_layer_accuracy"]
            >= thresholds["exact_context_accepted_min"],
        "selected_literal_accepted": metrics["selected_literal"]["accepted_layer_accuracy"]
            >= thresholds["selected_literal_accepted_min"],
        "selected_literal_complete": metrics["selected_literal"]["complete_group_accuracy"]
            >= thresholds["selected_literal_complete_min"],
        "wrong_value_correct_target": metrics["wrong_value"]["accepted_layer_accuracy"]
            <= thresholds["wrong_value_correct_target_max"],
        "no_memory_group_abstain": no_memory["group_abstention_rate"]
            >= thresholds["no_memory_group_abstain_min"],
        "no_memory_group_non_abstain": no_memory["group_non_abstention_rate"]
            <= thresholds["no_memory_group_non_abstain_max"],
        "no_memory_hallucination": no_memory["hallucination_rate"]
            <= thresholds["no_memory_hallucination_max"],
        "no_memory_per_layer_non_abstention": all(
            row["non_abstention_rate"] <= thresholds["no_memory_per_layer_non_abstain_max"]
            for row in no_memory["per_layer"].values()),
    }
    gates["non_compressed_gates_passed"] = all(gates.values())
    return gates


def score_predictions(predictions_path: str | Path, *, split: str,
                      decode_fn: Callable, native_execution: bool = False,
                      dev_output: str | Path | None = None,
                      expected_attempt_id: str | None = None,
                      prepared: str | Path = PREPARED_DIR) -> dict:
    require(split in ("dev", "sealed_final"), "scoring", "split differs")
    prepared = Path(prepared)
    config = load_config(prepared=prepared)
    binding = validate_tokenizer_binding(config["tokenizer_binding"])
    if split == "sealed_final":
        require(dev_output is not None and expected_attempt_id is not None,
                "final_admission", "native development output and attempt are required")
        development = verify_development_evidence(
            Path(dev_output), decode_fn=decode_fn,
            expected_attempt_id=expected_attempt_id, require_native=True,
            prepared=prepared)
    else:
        development = None
    manifest = validate_prepared(
        config, defer_final_labels=split == "dev", prepared=prepared)
    cases, labels, condition_rows = _cases(prepared, split)
    rows = read_jsonl(predictions_path)
    by_key = {}
    for row in rows:
        key = (row.get("condition"), row.get("case_id"))
        require(key not in by_key, "prediction", f"duplicate prediction {key}")
        by_key[key] = row
    require(set(by_key) == set(condition_rows), "prediction", "prediction coverage differs")
    attempt_ids = set()
    classes = set()
    model_identities = set()
    reader_states = set()
    scored = []
    for key, condition_row in condition_rows.items():
        condition, case_id = key
        row = by_key[key]
        require(row.get("format") == PREDICTION_FORMAT, "prediction", "format differs")
        require(row.get("protocol_id") == PROTOCOL_ID, "prediction", "protocol differs")
        require(row.get("split") == split, "prediction", "split differs")
        legacy.assert_no_answer_shortcut(row)
        legacy.assert_routing_is_answer_free(row.get("routing_metadata", {}))
        legacy.assert_prompt_matches(condition_row, row)
        require(row.get("source_context_present") == condition_row["source_context_present"],
                "prediction", "source-context flag differs")
        require(row.get("source_kv_loaded") is False, "prediction", "source KV retained")
        require(row.get("prompt_source_spans") == 0, "prediction", "source spans retained")
        require(row.get("tokenizer_binding_sha256") == tokenizer_binding_sha(binding),
                "prediction", "tokenizer binding differs")
        require(row.get("decode_policy") == binding["decode_policy"],
                "prediction", "decode policy differs")
        require(row.get("evidence_class") in ("native_mps", "cpu_fixture"),
                "prediction", "evidence class differs")
        require(isinstance(row.get("attempt_id"), str) and row["attempt_id"],
                "prediction", "attempt identity missing")
        for name in ("model_identity", "reader_state_sha256"):
            require(isinstance(row.get(name), str) and len(row[name]) == 64,
                    "prediction", f"{name} missing")
        outputs = row.get("layer_outputs")
        require(isinstance(outputs, list)
                and [item.get("layer") for item in outputs] == list(LAYERS),
                "prediction", "layer group differs")
        layer_rows = [_layer_evidence(
            item, condition=condition, case_id=case_id,
            target=labels[case_id]["target"], binding=binding, decode_fn=decode_fn,
            decode_metadata={
                "tokenizer_binding_sha256": row["tokenizer_binding_sha256"],
                "decode_policy": row["decode_policy"],
            })
            for item in outputs]
        scored.append({
            "condition": condition, "case_id": case_id, "layers": layer_rows,
            "accepted": any(item["correct"] for item in layer_rows),
            "complete": all(item["correct"] for item in layer_rows),
            "group_abstained": all(item["unknown"] for item in layer_rows),
        })
        attempt_ids.add(row["attempt_id"])
        classes.add(row["evidence_class"])
        model_identities.add(row["model_identity"])
        reader_states.add(row["reader_state_sha256"])
    require(len(attempt_ids) == len(classes) == len(model_identities) == len(reader_states) == 1,
            "prediction", "attempt, evidence class, model or reader identity is inconsistent")
    if expected_attempt_id is not None:
        require(attempt_ids == {expected_attempt_id}, "prediction",
                "final and development attempts differ")
    metrics = {condition: _metric([row for row in scored if row["condition"] == condition])
               for condition in CONDITIONS}
    physical = _physical(scored, config, native_execution)
    gates = _gates(metrics, config)
    codec = read_json(PREPARED_DIR / "codec.json")
    compressed = [row for row in scored if row["condition"] == "compressed_record"]
    reconstruction = sum(
        normalize_answer(legacy.decode_compressed_literal(
            codec, labels[row["case_id"]]["compressed_code"]))
        == normalize_answer(labels[row["case_id"]]["target"]) for row in compressed) / len(compressed)
    require(reconstruction >= config["evaluation"]["gates"]["compressed_reconstruction_control_min"],
            "reconstruction_control", "host reconstruction control failed")
    evidence_class = next(iter(classes))
    admitted = gates["non_compressed_gates_passed"] and physical["complete"]
    if split == "sealed_final":
        admitted = admitted and development["native_execution_admissible"]
    report = {
        "format": SCORE_FORMAT,
        "protocol_id": PROTOCOL_ID,
        "split": split,
        "status": "native_compressed_channel_blocked",
        "native_computation": False,
        "answer_index_metrics_only": False,
        "config_sha256": sha256_file(CONFIG_PATH),
        "prepared_manifest_sha256": manifest["manifest_sha256"],
        "tokenizer_binding_sha256": tokenizer_binding_sha(binding),
        "attempt_id": next(iter(attempt_ids)),
        "evidence_class": evidence_class,
        "model_identity": next(iter(model_identities)),
        "reader_state_sha256": next(iter(reader_states)),
        "final_labels_opened": split == "sealed_final",
        "host_abstention_flags_used": False,
        "accepted_layer_substitute_for_complete_group": False,
        "accepted_layer_substitute_for_group_abstention": False,
        "metrics_source": "re_decoded_generated_text",
        "conditions": metrics,
        "per_case_outputs": scored,
        "physical_work": physical,
        "gates": gates,
        "host_compressed_reconstruction_accuracy": reconstruction,
        "native_compressed_gate_status": config["compressed_channel"]["native_gate_status"],
        "development_evidence_admissible": admitted,
        "native_execution_admissible": evidence_class == "native_mps" and admitted,
        "final_admission": (None if development is None else {
            "admitted": development["native_execution_admissible"],
            "independently_recomputed": True,
            "native_development_evidence": True,
            "dev_report_sha256": sha256_file(Path(dev_output) / "dev/score_report.json"),
            "dev_predictions_sha256": sha256_file(Path(dev_output) / "dev/predictions_all.jsonl"),
        }),
    }
    return report


def verify_development_evidence(output_root: Path, *, decode_fn: Callable,
                                expected_attempt_id: str,
                                require_native: bool = True,
                                prepared: str | Path = PREPARED_DIR) -> dict:
    dev = output_root / "dev"
    report_path = dev / "score_report.json"
    predictions_path = dev / "predictions_all.jsonl"
    stage_path = dev / "journal.json"
    for path in (report_path, predictions_path, stage_path,
                 output_root / "publication_journal.json", output_root / "work_ledger.json"):
        require(path.is_file(), "final_admission", f"missing development evidence {path}")
    report = read_json(report_path)
    stage = read_json(stage_path)
    require(stage.get("stage_complete") is True
            and stage.get("publication_stage_done") is True,
            "final_admission", "development stage is not completely published")
    require(stage.get("attempt_id") == expected_attempt_id
            == report.get("attempt_id"), "final_admission", "development attempt differs")
    if require_native:
        require(report.get("evidence_class") == "native_mps"
                and report.get("native_execution_admissible") is True,
                "final_admission", "development is not independently admitted native evidence")
    ledger = read_json(output_root / "work_ledger.json")
    require(ledger.get("attempt_id") == expected_attempt_id,
            "final_admission", "work ledger attempt differs")
    require(ledger.get("model_identity") == report.get("model_identity"),
            "final_admission", "model identity differs from development ledger")
    require(ledger.get("tokenizer_binding_sha256") == report.get("tokenizer_binding_sha256"),
            "final_admission", "tokenizer identity differs from development ledger")
    journal = publication.Journal(
        output_root, identity={
            "protocol_id": PROTOCOL_ID,
            "config_sha256": report["config_sha256"],
            "prepared_manifest_sha256": report["prepared_manifest_sha256"],
            "authorization_receipt_sha256": ledger["authorization_receipt_sha256"],
            "attempt_id": expected_attempt_id,
            "output_root": str(output_root.resolve()),
        })
    require(journal.stage_done("dev"), "final_admission",
            "development publication artifacts are missing or changed")
    published = {artifact.path for artifact in journal.artifacts("dev")}
    require({"dev/score_report.json", "dev/predictions_all.jsonl",
             "dev/admission.json", "dev/journal.json"} <= published,
            "final_admission", "development publication is incomplete")
    expected = score_predictions(
        predictions_path, split="dev", decode_fn=decode_fn,
        native_execution=True, prepared=prepared)
    require(canonical(expected) == canonical(report), "final_admission",
            "development report differs from independently recomputed evidence")
    return report

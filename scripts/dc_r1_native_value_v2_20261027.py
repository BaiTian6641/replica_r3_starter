"""DC-R1 v2: corrected native record-readout protocol preparation.

Version 2 amends the v1 preparation package without changing any v1 file or
piece of evidence.  It makes four corrections explicit:

* Abstention is a declared complete-layer-group outcome.  A case abstains only
  when every declared layer's actual re-decoded generated output is UNKNOWN.
  Any non-UNKNOWN layer output, including an invented answer, makes group
  abstention false.  Per-layer non-abstention and hallucination metrics are
  computed from generated/re-decoded text, never from a host abstention flag.
  "Accepted layer" accuracy is diagnostic only and cannot substitute for group
  abstention or complete-group recall.
* The compressed channel is declared precisely.  The native input is a codec id
  plus one integer code.  The host codec table is unavailable to the model.
  Host ``decode_compressed_literal`` is a separate reconstruction audit/control;
  it is not proof of a learned compressed readout.  Native compressed gates are
  reported as blocked until a separate native channel receipt exists.
* Physical work is accounted in separate counters: prompt generation requests,
  prefill forwards, decode forwards, generated tokens, layer readout streams and
  processed positions, broken down over five conditions and three layers across
  development and sealed-final splits.  The old v1 value 240 is preserved as
  incorrect historical request accounting, not silently raised.
* Sealed-final scoring has its own entry point and refuses to read final labels
  until a valid development score report has been admitted.

This module is preparation only: no torch import, model load, tokenizer load,
generation, optimizer step, gradient, download or native job exists here.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path
import sys
from typing import Any, Callable, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dc_protocol_common_20261027 import (  # noqa: E402
    ANSWER_SHORTCUT_FIELDS, EVALUATION_CASE_FORBIDDEN_FIELDS, ProtocolError,
    assert_no_answer_shortcut, assert_prompt_matches,
    assert_routing_is_answer_free, canonical, check_work_budget,
    check_work_budget_v2, find_forbidden_fields, is_work_counter,
    normalize_answer, preflight_receipt, read_json, read_jsonl, require,
    seal_manifest, sha256_file, sha256_json, tokenizer_binding_sha,
    validate_baseline_model, validate_guards, validate_manifest,
    validate_tokenizer_binding, verify_decoded_text, write_json, write_jsonl,
)

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_ID = "dc_r1_native_value_readout_v2"
CONFIG_FORMAT = "replica.dc_r1_native_value.protocol.v2"
MANIFEST_FORMAT = "replica.dc_r1_native_value.prepared.v2"
CASE_FORMAT = "replica.dc_r1_native_value.eval_case.v2"
CONDITION_FORMAT = "replica.dc_r1_native_value.condition.v2"
LABEL_FORMAT = "replica.dc_r1_native_value.label.v2"
PREDICTION_FORMAT = "replica.dc_r1_native_value.prediction.v2"
SCORE_FORMAT = "replica.dc_r1_native_value.score_report.v2"
CODEC_FORMAT = "replica.dc_r1_native_value.codec.v2"
VALIDATION_FORMAT = "replica.dc_r1_native_value.validation.v2"

CONFIG_PATH = ROOT / "configs" / "dc_r1_native_value_protocol_v2.json"
PREPARED_DIR = ROOT / "data" / "dc_r1_native_value_v2_20261027"
V1_PROTOCOL_ID = "dc_r1_native_value_readout_v1"
V1_CONFIG_PATH = ROOT / "configs" / "dc_r1_native_value_protocol_v1.json"
V1_PREPARED_DIR = ROOT / "data" / "dc_r1_native_value_20261027"
V1_CONFIG_SHA256 = "afe323a0cea96e38dd023a983b953a35bef2d762a1207599e591dd3d17c8d295"
V1_MANIFEST_SHA256 = "361eeebd57ba3ab1ca9cb8a751b10331778d3001b65aa05703c80a29e60d4918"

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

# Physical-work units.  This is the existing conservative sequential per-layer
# schedule: one generation request per case x condition x layer, with no shared
# or batched prefill.  MiniCPM generate_action's prefill forward produces the
# first token, so decode forwards are generated_tokens - active_requests.  The
# prefill-token ceiling is 128 per request (the v1 aggregate 30720) and the
# no-cache processed-position ceiling per request is
# T*L + T*(T-1)/2 for L<=128, T<=8, i.e. 1052.
PREFILL_TOKEN_CEILING_PER_REQUEST = 128
MAX_GENERATED_TOKENS_PER_STREAM = 8
PROCESSED_POSITIONS_CEILING_PER_REQUEST = (
    MAX_GENERATED_TOKENS_PER_STREAM * PREFILL_TOKEN_CEILING_PER_REQUEST
    + MAX_GENERATED_TOKENS_PER_STREAM
    * (MAX_GENERATED_TOKENS_PER_STREAM - 1) // 2
)
ABSTENTION_POLICY_ID = "all_declared_layers_group_abstention_v1"

# A v2 layer output may not carry a host-computed abstention/correctness field.
# Every reported abstention/hallucination outcome is recomputed from the actual
# generated ids and re-decoded text.
V2_LAYER_FORBIDDEN_FIELDS = frozenset(
    set(ANSWER_SHORTCUT_FIELDS)
    | {"abstain", "abstained", "group_abstained", "non_abstain",
       "non_abstention", "hallucinated", "hallucination",
       "accepted_layer_correct", "complete_group_correct"}
)

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
            "Deterministic 8-bit bucket codec.  The native compressed channel "
            "supplies only codec_id plus one integer code; the host codec table "
            "below is unavailable to the model.  A bucket has a canonical "
            "literal and one alternate member, so a code reconstructs only the "
            "canonical bucket literal.  Host reconstruction is an audit/control, "
            "not evidence of a learned compressed readout."),
        "bits": 8,
        "bucket_count": LOSSLESS_CODES,
        "native_input_fields": ["codec_id", "code"],
        "native_input_is_codec_id_plus_integer": True,
        "host_codec_table_visible_to_model": False,
        "host_decode_role": "reconstruction_audit_control_only",
        "host_decode_is_native_readout_evidence": False,
        "entries": entries,
    }


def _codec_entry(codec: Mapping[str, Any], code: int) -> dict:
    for entry in codec["entries"]:
        if entry["code"] == code:
            return entry
    raise ProtocolError("codec", f"unknown code {code}")


def decode_compressed_literal(codec: Mapping[str, Any], code: int) -> str:
    """Host-only reconstruction audit; not a native readout measurement."""
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
        evidence = {"kind": "source_text",
                    "source_text": record["source_text"]}
    elif condition == "selected_literal":
        prompt = (f"Use the selected literal record and answer with only the "
                  f"recorded phrase.\nRecord: {record['literal']}\n"
                  f"Question: {record['query']}")
        evidence = {"kind": "literal", "literal": record["literal"]}
    elif condition == "compressed_record":
        # Only codec_id and the integer code are visible.  No codec entry,
        # canonical literal, bucket member or reconstruction hint is supplied.
        prompt = (f"Use the selected compressed record and answer with only "
                  f"the recorded phrase.\nCodec: {codec['codec_id']}\n"
                  f"Code: {record['compressed_code']}\n"
                  f"Question: {record['query']}")
        evidence = {
            "kind": "compressed_code",
            "codec_id": codec["codec_id"],
            "code": record["compressed_code"],
            "native_input_fields": ["codec_id", "code"],
            "host_codec_table_visible_to_model": False,
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
    return {
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


def processed_positions_for_request(prompt_tokens: int,
                                    generated_tokens: int) -> int:
    """No-cache growing-prefix positions for one sequential request.

    Each generated token re-reads the prompt plus the already generated prefix,
    so positions are ``T*L + T*(T-1)/2`` for prompt length ``L`` and generated
    count ``T``.  Early stop simply lowers ``T`` and the result is a <= ceiling.
    """
    require(isinstance(prompt_tokens, int) and not isinstance(prompt_tokens, bool)
            and 0 <= prompt_tokens <= PREFILL_TOKEN_CEILING_PER_REQUEST,
            "workload", "prompt_tokens must be in the 128-token ceiling")
    require(isinstance(generated_tokens, int)
            and not isinstance(generated_tokens, bool)
            and 0 <= generated_tokens <= MAX_GENERATED_TOKENS_PER_STREAM,
            "workload", "generated_tokens must be in the 8-token ceiling")
    return (generated_tokens * prompt_tokens
            + generated_tokens * (generated_tokens - 1) // 2)


def sequential_request_work(requests: int, *,
                            generated_tokens: int | None = None,
                            active_requests: int | None = None,
                            prompt_tokens_total: int | None = None) -> dict:
    """Counters for a sequential per-layer schedule, including early stop.

    ``active_requests`` is the number of requests that emitted at least one
    token.  With no early stop it equals ``requests`` and the identity
    ``decode_forwards == generated_tokens - requests`` holds; with early stop
    some requests emit zero tokens, so ``decode_forwards`` is lower and the
    aggregate remains <= the full ceiling.
    """
    require(isinstance(requests, int) and not isinstance(requests, bool)
            and requests >= 0, "workload", "requests must be nonnegative")
    generated = (MAX_GENERATED_TOKENS_PER_STREAM * requests
                 if generated_tokens is None else generated_tokens)
    require(isinstance(generated, int) and not isinstance(generated, bool)
            and 0 <= generated <= MAX_GENERATED_TOKENS_PER_STREAM * requests,
            "workload", "generated_tokens exceeds the per-request ceiling")
    active = requests if active_requests is None else active_requests
    require(isinstance(active, int) and not isinstance(active, bool)
            and 0 <= active <= requests, "workload",
            "active_requests must be in [0, requests]")
    require(generated >= active, "workload",
            "every active request generates at least one token")
    prefill_forwards = requests
    decode_forwards = generated - active
    prefill_tokens = (requests * PREFILL_TOKEN_CEILING_PER_REQUEST
                      if prompt_tokens_total is None else prompt_tokens_total)
    require(isinstance(prefill_tokens, int)
            and not isinstance(prefill_tokens, bool)
            and 0 <= prefill_tokens
            <= requests * PREFILL_TOKEN_CEILING_PER_REQUEST,
            "workload", "prefill_tokens exceeds the per-request ceiling")
    return {
        "generation_requests": requests,
        "layer_readout_streams": requests,
        "prefill_forwards": prefill_forwards,
        "generated_tokens": generated,
        "decode_forwards": decode_forwards,
        "forward_calls": prefill_forwards + decode_forwards,
        "prefill_tokens": prefill_tokens,
        "processed_positions_ceiling": (
            requests * PROCESSED_POSITIONS_CEILING_PER_REQUEST),
    }


def _condition_work_breakdown() -> dict:
    splits = {}
    for split, cases in (("dev", DEV_CASES),
                         ("sealed_final", SEALED_CASES)):
        per_condition = {}
        for condition in CONDITIONS:
            requests = cases * len(LAYER_GROUP)
            work = sequential_request_work(requests)
            per_condition[condition] = {
                "generation_requests": work["generation_requests"],
                "prefill_forwards": work["prefill_forwards"],
                "prefill_tokens": work["prefill_tokens"],
                "layer_readout_streams": work["layer_readout_streams"],
                "generated_tokens": work["generated_tokens"],
                "decode_forwards": work["decode_forwards"],
                "forward_calls": work["forward_calls"],
                "processed_positions": (
                    requests * PROCESSED_POSITIONS_CEILING_PER_REQUEST),
            }
        splits[split] = per_condition
    return {
        "basis": (
            "one sequential per-layer generation request per case x condition "
            "x layer; MiniCPM generate_action's prefill forward produces the "
            "first token; decode forwards = generated_tokens - active_requests; "
            "processed positions = T*L + T*(T-1)/2 under the no-cache "
            "growing-prefix adapter"),
        "per_split_condition": splits,
    }


def _layer_work_breakdown() -> dict:
    splits = {}
    for split, cases in (("dev", DEV_CASES),
                         ("sealed_final", SEALED_CASES)):
        per_layer = {}
        for layer in LAYER_GROUP:
            requests = cases * len(CONDITIONS)
            work = sequential_request_work(requests)
            per_layer[str(layer)] = {
                "generation_requests": work["generation_requests"],
                "layer_readout_streams": work["layer_readout_streams"],
                "prefill_forwards": work["prefill_forwards"],
                "generated_tokens": work["generated_tokens"],
                "decode_forwards": work["decode_forwards"],
                "forward_calls": work["forward_calls"],
                "prefill_tokens": work["prefill_tokens"],
                "processed_positions": (
                    requests * PROCESSED_POSITIONS_CEILING_PER_REQUEST),
            }
        splits[split] = per_layer
    return {
        "basis": (
            "sequential per-layer requests; every request has its own "
            "prefill forward and decode forwards, so all counters are "
            "attributable per layer without a shared-prefill assumption"),
        "per_split_layer": splits,
    }


def workload_derivation() -> dict:
    splits = {}
    for split, cases in (("dev", DEV_CASES),
                         ("sealed_final", SEALED_CASES)):
        requests = cases * len(CONDITIONS) * len(LAYER_GROUP)
        work = sequential_request_work(requests)
        splits[split] = {
            "cases": cases,
            "conditions": len(CONDITIONS),
            "layers": len(LAYER_GROUP),
            "generation_requests": work["generation_requests"],
            "prefill_forwards": work["prefill_forwards"],
            "prefill_tokens_max": work["prefill_tokens"],
            "layer_readout_streams": work["layer_readout_streams"],
            "generated_tokens_max": work["generated_tokens"],
            "decode_forwards_max": work["decode_forwards"],
            "forward_calls_max": work["forward_calls"],
            "processed_positions_max": work["processed_positions_ceiling"],
        }
    total = {
        key: (splits["dev"][key] + splits["sealed_final"][key])
        for key in splits["dev"]
        if key not in {"cases", "conditions", "layers"}
    }
    total["cases"] = DEV_CASES + SEALED_CASES
    total["conditions"] = len(CONDITIONS)
    total["layers"] = len(LAYER_GROUP)
    return {
        "basis": (
            "read-only sequential per-layer generation for five declared "
            "conditions and three declared layers across 12 development and 4 "
            "sealed-final cases; no shared or batched prefill"),
        "counter_units": {
            "generation_requests": (
                "one sequential per-layer generation request per case x "
                "condition x layer"),
            "prefill_forwards": (
                "one prefill forward per request; generate_action's first "
                "forward produces the first generated token"),
            "decode_forwards": (
                "one forward per additional generated token: "
                "generated_tokens - active_requests, at most "
                "generated_tokens - generation_requests when every request "
                "emits at least one token"),
            "generated_tokens": (
                f"generated token ids, at most {MAX_GENERATED_TOKENS_PER_STREAM}"
                " per request"),
            "layer_readout_streams": (
                "one layer-specific readout stream per case x condition x "
                "layer; equal to generation_requests in this sequential "
                "schedule"),
            "processed_positions": (
                "no-cache growing-prefix positions T*L + T*(T-1)/2 per "
                "request; the per-request ceiling is "
                f"{PROCESSED_POSITIONS_CEILING_PER_REQUEST} for L<=128,T<=8"),
            "forward_calls": (
                "prefill forwards plus decode forwards; early stop only "
                "lowers this value"),
        },
        "splits": splits,
        "total": total,
        "conditions": _condition_work_breakdown(),
        "layers": _layer_work_breakdown(),
        "prefill_token_ceiling_per_request": PREFILL_TOKEN_CEILING_PER_REQUEST,
        "max_generated_tokens_per_request": MAX_GENERATED_TOKENS_PER_STREAM,
        "processed_positions_ceiling_per_request":
            PROCESSED_POSITIONS_CEILING_PER_REQUEST,
        "processed_positions_formula": "T*L + T*(T-1)/2",
        "early_stop_is_upper_bound": True,
        "prefill_shared_by_layer_streams": False,
        "batched_or_shared_prefill": False,
    }


def budget_schedule() -> dict:
    return {
        "mode": "sequential_per_layer_no_shared_prefill",
        "generation_requests_per_case_condition_layer": 1,
        "prefill_forward_produces_first_token": True,
        "prefill_token_ceiling_per_request":
            PREFILL_TOKEN_CEILING_PER_REQUEST,
        "max_generated_tokens_per_request":
            MAX_GENERATED_TOKENS_PER_STREAM,
        "decode_forwards_formula": "generated_tokens - active_requests",
        "forward_calls_formula": "prefill_forwards + decode_forwards",
        "processed_positions_formula": "T*L + T*(T-1)/2",
        "processed_positions_ceiling_per_request":
            PROCESSED_POSITIONS_CEILING_PER_REQUEST,
        "early_stop_is_upper_bound": True,
        "batched_or_shared_prefill": False,
    }


def expected_physical_limits() -> dict:
    total = workload_derivation()["total"]
    return {
        "max_model_loads": 1,
        "max_tokenizer_loads": 1,
        "max_optimizer_steps": 0,
        "max_backward_calls": 0,
        "max_generation_requests": total["generation_requests"],
        "max_prefill_forwards": total["prefill_forwards"],
        "max_decode_forwards": total["decode_forwards_max"],
        "max_layer_readout_streams": total["layer_readout_streams"],
        "max_forward_calls": total["forward_calls_max"],
        "max_generated_tokens": total["generated_tokens_max"],
        "max_prefill_tokens": total["prefill_tokens_max"],
        "max_processed_positions": total["processed_positions_max"],
        "max_checkpoint_writes": 0,
        "max_downloads": 0,
        "max_retries": 0,
        "max_workers": 1,
        "max_wall_seconds": 1800,
    }


def expected_v1_physical_limits() -> dict:
    """The exact v1 limits, preserved for lineage and not reused as v2 caps."""
    return {
        "max_model_loads": 1,
        "max_tokenizer_loads": 1,
        "max_optimizer_steps": 0,
        "max_backward_calls": 0,
        "max_forward_calls": 240,
        "max_generation_calls": 240,
        "max_generated_tokens": 1920,
        "max_prefill_tokens": 30720,
        "max_checkpoint_writes": 0,
        "max_downloads": 0,
        "max_retries": 0,
        "max_workers": 1,
        "max_wall_seconds": 1800,
    }


def expected_v1_gates() -> dict:
    return {
        "exact_context_accepted_min": 0.8,
        "selected_literal_accepted_min": 0.9,
        "selected_literal_complete_min": 0.7,
        "compressed_reconstruction_min": 0.75,
        "compressed_accepted_min": 0.6,
        "compressed_complete_min": 0.4,
        "no_memory_abstain_min": 0.95,
        "no_memory_non_abstain_max": 0.05,
        "wrong_value_correct_target_max": 0.1,
        "no_layer_acceptance_reporting": True,
    }


def build_fixture(config_path: str | Path = CONFIG_PATH,
                  output: str | Path = PREPARED_DIR) -> dict:
    config = read_json(config_path)
    validate_config(config)
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

    amendment = deepcopy(config["amendment"])
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
        "lineage": {
            "amendment_id": amendment["amendment_id"],
            "amends_protocol_id": amendment["amends_protocol_id"],
            "amends_config_path": amendment["amends_config_path"],
            "amends_config_sha256": amendment["amends_config_sha256"],
            "amends_manifest_sha256": amendment["amends_manifest_sha256"],
            "v1_fixture_root": amendment["v1_fixture_root"],
            "v1_preserved": True,
            "v2_prepared_root": str(root.relative_to(ROOT)),
        },
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
            "role": "host_reconstruction_audit_control_only",
        },
        "compressed_channel": {
            "native_input_fields": ["codec_id", "code"],
            "native_input_is_codec_id_plus_integer": True,
            "host_codec_table_visible_to_model": False,
            "host_decode_role": "reconstruction_audit_control_only",
            "host_decode_is_native_readout_evidence": False,
            "native_gate_status": "blocked_no_native_compressed_channel_receipt",
        },
        "source_groups": {
            "dev": sorted({row["source_group"] for row in dev_cases}),
            "sealed_final": sorted({row["source_group"]
                                    for row in final_cases}),
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
    require(codec.get("native_input_fields") == ["codec_id", "code"],
            "fixture", "native compressed input must be codec_id plus code")
    require(codec.get("host_codec_table_visible_to_model") is False,
            "fixture", "host codec table must be unavailable to the model")
    require(codec.get("host_decode_role")
            == "reconstruction_audit_control_only", "fixture",
            "host decode role differs")
    require(codec.get("host_decode_is_native_readout_evidence") is False,
            "fixture", "host decode must not count as native readout evidence")
    codes = [entry.get("code") for entry in codec.get("entries", [])]
    require(codes == list(range(LOSSLESS_CODES)), "fixture",
            "codec codes must be exactly 0..15")
    codec_literals: set[str] = set()
    for entry in codec["entries"]:
        require(entry.get("canonical_literal") in entry.get("members", []),
                "fixture", "codec canonical literal is not a member")
        require(len(set(entry.get("members", []))) == len(entry["members"]),
                "fixture", "codec bucket members repeat")
        codec_literals.update(entry["members"])

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
                    require(evidence.get("native_input_fields")
                            == ["codec_id", "code"], "fixture",
                            "compressed native input fields differ")
                    require(evidence.get("host_codec_table_visible_to_model")
                            is False, "fixture",
                            "compressed evidence exposes the host codec table")
                    require(label["target"].casefold()
                            not in row["prompt"].casefold(), "fixture",
                            "compressed prompt leaks the target literal")
                    for literal in codec_literals:
                        require(literal.casefold()
                                not in row["prompt"].casefold(), "fixture",
                                "compressed prompt leaks a codec entry")
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
        "format": VALIDATION_FORMAT,
        "protocol_id": PROTOCOL_ID,
        "status": "valid",
        "manifest_sha256": manifest["manifest_sha256"],
        "counts": manifest["counts"],
        "compressed_reconstruction": manifest["compressed_reconstruction"],
        "lineage": manifest["lineage"],
        "native_computation": False,
        "labels_read_for_construction_audit": bool(validate_labels),
        "heldout_opened": False,
    }


# ---------------------------------------------------------------------------
# Protocol config validation


def _validate_amendment(config: Mapping[str, Any]) -> dict:
    amendment = config.get("amendment")
    require(isinstance(amendment, Mapping), "amendment",
            "versioned amendment lineage is required")
    require(amendment.get("amendment_id")
            == "dc_r1_native_value_v2_amendment_20261027", "amendment",
            "amendment id differs")
    require(amendment.get("amends_protocol_id") == V1_PROTOCOL_ID,
            "amendment", "amended protocol id differs")
    require(amendment.get("amends_config_path")
            == str(V1_CONFIG_PATH.relative_to(ROOT)), "amendment",
            "v1 config path differs")
    require(amendment.get("v1_fixture_root")
            == str(V1_PREPARED_DIR.relative_to(ROOT)), "amendment",
            "v1 fixture root differs")
    require(amendment.get("v1_preserved") is True, "amendment",
            "v1 files and evidence must be preserved")
    require(amendment.get("amends_config_sha256") == V1_CONFIG_SHA256,
            "amendment", "declared v1 config hash differs")
    require(amendment.get("amends_manifest_sha256") == V1_MANIFEST_SHA256,
            "amendment", "declared v1 manifest hash differs")
    require(V1_CONFIG_PATH.is_file(), "amendment", "v1 config is missing")
    require(V1_PREPARED_DIR.joinpath("manifest.json").is_file(), "amendment",
            "v1 manifest is missing")
    require(sha256_file(V1_CONFIG_PATH) == V1_CONFIG_SHA256, "amendment",
            "v1 config bytes changed; historical lineage is broken")
    require(sha256_file(V1_PREPARED_DIR / "manifest.json") == V1_MANIFEST_SHA256,
            "amendment",
            "v1 manifest bytes changed; historical evidence is broken")
    supersedes = amendment.get("supersedes")
    require(isinstance(supersedes, list) and set(supersedes) == {
        "any_layer_abstention_acceptance",
        "conflated_forward_and_generation_request_accounting",
        "host_compressed_reconstruction_treated_as_native_readout_evidence",
    }, "amendment", "amendment supersession list differs")
    return dict(amendment)


def _validate_budget(config: Mapping[str, Any]) -> None:
    budget = config.get("budget", {})
    physical = budget.get("physical_limits", {})
    expected = expected_physical_limits()
    require(physical == expected, "budget",
            "corrected physical limits differ from the derived workload")
    historical = budget.get("historical_v1_limits")
    require(historical == expected_v1_physical_limits(), "budget",
            "v1 physical limits must be preserved exactly")
    require(budget.get("schedule") == budget_schedule(), "budget",
            "sequential per-layer work schedule differs")
    accounting = budget.get("historical_request_accounting", {})
    require(accounting.get("value") == 240, "budget",
            "the old 240 must be preserved as historical accounting")
    require(accounting.get("status")
            == "incorrect_historical_request_accounting_preserved_not_a_cap",
            "budget", "historical 240 status differs")
    require(accounting.get("corrected_generation_requests")
            == expected["max_generation_requests"], "budget",
            "historical correction must name corrected generation requests")
    require(accounting.get("corrected_layer_readout_streams")
            == expected["max_layer_readout_streams"], "budget",
            "historical correction must name corrected layer streams")
    require(accounting.get("corrected_forward_calls")
            == expected["max_forward_calls"], "budget",
            "historical correction must name corrected forward calls")
    require(accounting.get("corrected_prefill_forwards")
            == expected["max_prefill_forwards"], "budget",
            "historical correction must name corrected prefill forwards")
    require(accounting.get("corrected_decode_forwards")
            == expected["max_decode_forwards"], "budget",
            "historical correction must name corrected decode forwards")
    require(accounting.get("corrected_prefill_tokens")
            == expected["max_prefill_tokens"], "budget",
            "historical correction must name corrected prefill tokens")
    require(accounting.get("corrected_processed_positions")
            == expected["max_processed_positions"], "budget",
            "historical correction must name corrected processed positions")
    require(physical["max_layer_readout_streams"]
            == physical["max_generation_requests"], "budget",
            "sequential schedule must have streams equal to requests")
    require(physical["max_prefill_forwards"]
            == physical["max_generation_requests"], "budget",
            "every request must have exactly one prefill forward")
    require(physical["max_decode_forwards"]
            == physical["max_generated_tokens"]
            - physical["max_generation_requests"], "budget",
            "decode forwards must be generated tokens minus requests")
    require(physical["max_forward_calls"]
            == physical["max_prefill_forwards"]
            + physical["max_decode_forwards"], "budget",
            "forward calls must equal prefill plus decode forwards")
    require(physical["max_forward_calls"]
            == physical["max_generated_tokens"], "budget",
            "sequential full-activity forward calls must equal generated tokens")
    require(physical["max_prefill_tokens"]
            == physical["max_generation_requests"]
            * PREFILL_TOKEN_CEILING_PER_REQUEST, "budget",
            "prefill tokens must be 128 per request")
    require(physical["max_processed_positions"]
            == physical["max_generation_requests"]
            * PROCESSED_POSITIONS_CEILING_PER_REQUEST, "budget",
            "processed positions must use the 1052 per-request ceiling")
    require(budget.get("workload_derivation") == workload_derivation(),
            "budget", "detailed workload derivation differs")
    require(budget.get("condition_breakdown")
            == _condition_work_breakdown(), "budget",
            "condition work breakdown differs")
    require(budget.get("layer_breakdown") == _layer_work_breakdown(),
            "budget", "layer work breakdown differs")
    ceiling = budget.get("authorized_run_ceiling", {})
    required = {
        "generation_requests_total": expected["max_generation_requests"],
        "prefill_forwards_total": expected["max_prefill_forwards"],
        "decode_forwards_total": expected["max_decode_forwards"],
        "layer_readout_streams_total": expected["max_layer_readout_streams"],
        "forward_calls_total": expected["max_forward_calls"],
        "generated_tokens_total": expected["max_generated_tokens"],
        "prefill_tokens_total": expected["max_prefill_tokens"],
        "processed_positions_total": expected["max_processed_positions"],
        "backward_calls_total": 0,
        "optimizer_updates_total": 0,
        "model_loads": 1,
        "tokenizer_loads": 1,
        "checkpoint_writes": 0,
        "downloads": 0,
        "retries": 0,
    }
    for key, value in required.items():
        require(ceiling.get(key) == value, "budget",
                f"authorized_run_ceiling.{key} differs")
    require(budget.get("preparation", {}).get("native_jobs") == 0, "budget",
            "preparation must declare zero native jobs")
    require(physical["max_optimizer_steps"] == 0, "budget",
            "DC-R1 must declare zero optimizer steps")
    require(physical["max_backward_calls"] == 0, "budget",
            "DC-R1 must declare zero backward calls")
    # Compose the shared helper for the counters it recognizes.
    shared = check_work_budget({}, budget)
    require(shared["within_budget"] is True, "budget",
            "shared work-budget check failed")


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
    require(classification.get("generation") is False, "config",
            "no generation in preparation")
    require(classification.get("learned_retrieval") is False, "config",
            "learned retrieval remains closed")
    require(classification.get("storage_expansion") is False, "config",
            "storage expansion remains closed")
    validate_guards(config)
    validate_baseline_model(config)
    validate_tokenizer_binding(config.get("tokenizer_binding"))
    _validate_amendment(config)

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

    require(config.get("hypothesis", {}).get("id") == "DC-R1-H2", "config",
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
    require(
        readout.get("exact_literal_and_compressed_reconstruction_reported")
        is True, "readout_gate",
        "literal/compressed reconstruction must be reported")
    require(readout.get("thresholds_condition_independent") is True,
            "readout_gate",
            "thresholds may not depend on individual targets")

    anti = config.get("anti_shortcut", {})
    require(anti.get("answer_index_bypass_allowed") is False,
            "anti_shortcut", "answer-index bypass is forbidden")
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
    require(anti.get("host_abstention_flags_used") is False, "anti_shortcut",
            "abstention metrics must not use host flags")
    require(anti.get("target_in_condition_evidence") == [
        "exact_context", "selected_literal", "wrong_value"], "anti_shortcut",
        "target-bearing condition evidence declaration differs")

    abstention = config.get("abstention_policy", {})
    require(abstention.get("policy_id") == ABSTENTION_POLICY_ID,
            "abstention_policy", "abstention policy id differs")
    require(abstention.get("group_abstention_requires_all_layers") is True,
            "abstention_policy", "group abstention must require all layers")
    require(abstention.get("any_layer_abstention_acceptance") is False,
            "abstention_policy",
            "any-layer abstention must not be accepted")
    require(abstention.get("metrics_source") == "re_decoded_generated_text",
            "abstention_policy",
            "abstention metrics must use re-decoded generated text")
    require(abstention.get("host_abstention_flags_used") is False,
            "abstention_policy", "host abstention flags must not be used")
    require(abstention.get("per_layer_actual_output_metrics") is True,
            "abstention_policy", "per-layer actual-output metrics are required")
    require(
        abstention.get("accepted_layer_substitute_for_group_abstention")
        is False, "abstention_policy",
        "accepted layer cannot substitute for group abstention")
    require(abstention.get("accepted_layer_substitute_for_complete_group")
            is False, "abstention_policy",
            "accepted layer cannot substitute for complete group")
    require(abstention.get("complete_group_separate") is True,
            "abstention_policy", "complete group must remain separate")
    require(abstention.get("group_abstention_rule")
            == "group abstains iff every declared layer's actual re-decoded "
               "output is UNKNOWN; any non-UNKNOWN layer makes it false",
            "abstention_policy", "group abstention rule text differs")
    require(abstention.get("non_abstention_definition")
            == "case has at least one declared layer whose actual re-decoded "
               "output is not UNKNOWN",
            "abstention_policy", "non-abstention definition differs")
    require(abstention.get("hallucination_definition")
            == "case has at least one declared layer whose actual re-decoded "
               "output is neither UNKNOWN nor the exact target",
            "abstention_policy", "hallucination definition differs")

    compressed = config.get("compressed_channel", {})
    require(compressed.get("native_input_fields") == ["codec_id", "code"],
            "compressed_channel",
            "native compressed input must be codec_id plus integer code")
    require(compressed.get("native_input_is_codec_id_plus_integer") is True,
            "compressed_channel", "native input declaration differs")
    require(compressed.get("host_codec_table_visible_to_model") is False,
            "compressed_channel",
            "host codec table must be unavailable to the model")
    require(compressed.get("host_decode_compressed_literal_role")
            == "reconstruction_audit_control_only", "compressed_channel",
            "host decode role differs")
    require(compressed.get("host_decode_is_learned_readout_evidence")
            is False, "compressed_channel",
            "host decode must not be evidence of a learned readout")
    require(compressed.get("prompt_may_include_codec_entries") is False,
            "compressed_channel", "codec entries may not enter prompts")
    require(compressed.get("native_gate_status")
            == "blocked_no_native_compressed_channel_receipt",
            "compressed_channel", "native compressed gate status differs")
    require(compressed.get("native_gates")
            == ["compressed_accepted_native", "compressed_complete_native"],
            "compressed_channel", "native compressed gate list differs")
    require(compressed.get("gates_lowered") is False, "compressed_channel",
            "compressed gates may not be lowered")

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
    require(prepared.get("v1_fixture_root")
            == str(V1_PREPARED_DIR.relative_to(ROOT)), "prepared",
            "v1 fixture lineage differs")

    reconstruction = config.get("compressed_reconstruction", {})
    require(reconstruction.get("dev") == 0.75, "config",
            "dev compressed reconstruction declaration differs")
    require(reconstruction.get("sealed_final") == 0.75, "config",
            "sealed compressed reconstruction declaration differs")
    require(reconstruction.get("host_decode_role")
            == "reconstruction_audit_control_only", "config",
            "host reconstruction role differs")

    evaluation = config.get("evaluation", {})
    require(evaluation.get("generated_answer_scoring") is True, "evaluation",
            "generated-answer scoring is required")
    require(evaluation.get("tokenizer_and_decode_binding_required") is True,
            "evaluation", "tokenizer/decode binding is required")
    require(evaluation.get("sealed_final_one_shot") is True, "evaluation",
            "sealed final must be one-shot")
    require(evaluation.get("fit_then_dev_then_sealed_final") is True,
            "evaluation", "evaluation order differs")
    require(evaluation.get("sealed_final_requires_dev_admission") is True,
            "evaluation", "final scoring must require dev admission")
    require(evaluation.get("final_labels_opened_before_admission") is False,
            "evaluation", "final labels may not open before admission")
    require(evaluation.get("scoring_entry_points")
            == ["score_dev", "score_sealed_final"], "evaluation",
            "separate scoring entry points differ")
    require(evaluation.get("admission_requirements")
            == ["valid_dev_score_report", "config_sha256_match",
                "prepared_manifest_sha256_match",
                "development_evidence_admissible"], "evaluation",
            "final admission requirements differ")
    gates = evaluation.get("gates", {})
    expected_gates = {
        "exact_context_accepted_min": 0.8,
        "selected_literal_accepted_min": 0.9,
        "selected_literal_complete_min": 0.7,
        "compressed_reconstruction_control_min": 0.75,
        "compressed_accepted_native_min": 0.6,
        "compressed_complete_native_min": 0.4,
        "no_memory_group_abstain_min": 0.95,
        "no_memory_group_non_abstain_max": 0.05,
        "no_memory_per_layer_non_abstain_max": 0.05,
        "no_memory_hallucination_max": 0.05,
        "wrong_value_correct_target_max": 0.1,
        "per_layer_reporting_required": True,
    }
    require(gates == expected_gates, "evaluation",
            "v2 gate values differ from the declared no-lowering set")
    require(evaluation.get("historical_v1_gates") == expected_v1_gates(),
            "evaluation", "v1 gate values must be preserved exactly")
    require(gates["exact_context_accepted_min"]
            == expected_v1_gates()["exact_context_accepted_min"]
            and gates["selected_literal_accepted_min"]
            == expected_v1_gates()["selected_literal_accepted_min"]
            and gates["selected_literal_complete_min"]
            == expected_v1_gates()["selected_literal_complete_min"]
            and gates["compressed_reconstruction_control_min"]
            == expected_v1_gates()["compressed_reconstruction_min"]
            and gates["compressed_accepted_native_min"]
            == expected_v1_gates()["compressed_accepted_min"]
            and gates["compressed_complete_native_min"]
            == expected_v1_gates()["compressed_complete_min"]
            and gates["no_memory_group_abstain_min"]
            == expected_v1_gates()["no_memory_abstain_min"]
            and gates["no_memory_group_non_abstain_max"]
            == expected_v1_gates()["no_memory_non_abstain_max"]
            and gates["wrong_value_correct_target_max"]
            == expected_v1_gates()["wrong_value_correct_target_max"],
            "evaluation", "v2 gates may not lower v1 thresholds")

    _validate_budget(config)

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
    require(launch.get("native_compressed_readout_status")
            == "blocked_no_native_compressed_channel_receipt", "launch",
            "native compressed readout status differs")
    return {
        "status": "valid",
        "protocol_id": PROTOCOL_ID,
        "conditions": list(CONDITIONS),
        "layer_group": list(LAYER_GROUP),
        "abstention_policy": ABSTENTION_POLICY_ID,
        "final_admission_required": True,
    }


def validate_prepared(prepared: str | Path = PREPARED_DIR,
                      config_path: str | Path = CONFIG_PATH) -> dict:
    config = read_json(config_path)
    validate_config(config)
    return _validate_fixture(Path(prepared), config, validate_labels=True)


def check_physical_work(accounting: Mapping[str, Any],
                        config: Mapping[str, Any]) -> dict:
    """Fail closed with the shared v2 canonical work-counter checker.

    DCP1Protocol owns ``check_work_budget_v2`` and its counter registry; DC-R1
    v2 consumes them without editing the shared module.  All of
    generation_requests, prefill_forwards, decode_forwards,
    layer_readout_streams, generated_tokens, prefill_tokens, forward_calls and
    processed_positions are canonical counters there, so no local duplicate
    checker is retained.
    """
    require(isinstance(accounting, Mapping), "physical_work",
            "accounting must be an object")
    for name in accounting:
        require(is_work_counter(name), "physical_work",
                f"unrecognized v2 accounting counter {name!r}")
    return check_work_budget_v2(
        accounting, config["budget"], check="physical_work")


# ---------------------------------------------------------------------------
# Per-case readout scoring


def _manifest_hash(config: Mapping[str, Any], prepared: Path) -> str:
    """Read only manifest.json; used before final-label admission."""
    manifest = read_json(prepared / "manifest.json")
    require(manifest.get("format") == MANIFEST_FORMAT, "manifest",
            "manifest format differs")
    require(manifest.get("protocol_id") == PROTOCOL_ID, "manifest",
            "manifest protocol_id differs")
    declared = config.get("prepared", {}).get("manifest_sha256")
    require(declared in (None, manifest.get("manifest_sha256")), "prepared",
            "config does not bind this prepared manifest")
    return manifest["manifest_sha256"]


def _layer_actual(text: str, target: str, *, completion_valid: bool) -> dict:
    normalized = normalize_answer(text)
    correct = completion_valid and normalized == normalize_answer(target)
    abstained = completion_valid and normalized == "unknown"
    return {
        "normalized_output": normalized,
        "abstained": abstained,
        "correct": correct,
        "invented_non_target": (not abstained) and (not correct),
        "empty_generation": normalized == "",
        "completion_valid": completion_valid,
    }


def _layer_work(item: Mapping[str, Any], *, condition: str, case_id: str,
                layer: int) -> dict:
    ids = item.get("generated_ids")
    require(isinstance(ids, list) and ids
            and all(isinstance(token, int) and not isinstance(token, bool)
                    and token >= 0 for token in ids), "physical_work",
            "each layer stream must contain generated token IDs")
    work = item.get("work")
    require(isinstance(work, Mapping), "physical_work",
            "each layer stream must carry physical work")
    prompt_tokens = work.get("prefill_tokens")
    require(isinstance(prompt_tokens, int)
            and not isinstance(prompt_tokens, bool) and prompt_tokens > 0,
            "physical_work", "prefill_tokens must be a positive integer")
    count = len(ids)
    expected = {
        "generation_requests": 1,
        "generation_prefill_forwards": 1,
        "generation_decode_forwards": count - 1,
        "autoregressive_forwards": count,
        "forward_calls": count,
        "generated_tokens": count,
        "prefill_tokens": prompt_tokens,
        "processed_positions": (
            count * prompt_tokens + count * (count - 1) // 2),
        "layer_readout_streams": 1,
    }
    require(dict(work) == expected, "physical_work",
            f"invalid work for {condition}/{case_id}/layer:{layer}: "
            f"expected {expected}, got {dict(work)}")
    stopped = item.get("stopped") is True
    reason = item.get("stop_reason")
    require(isinstance(reason, str) and bool(reason), "physical_work",
            "each layer stream must carry stop evidence")
    stream_id = f"{condition}/{case_id}/layer:{layer}"
    require(item.get("physical_stream_id") == stream_id, "physical_work",
            "physical layer stream identity differs")
    return {
        "physical_stream_id": stream_id,
        "work": expected,
        "stopped": stopped,
        "stop_reason": reason,
        "completion_valid": stopped and reason not in {
            "truncated", "max_new_tokens", "length", "budget"},
    }


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
    found = find_forbidden_fields(row, V2_LAYER_FORBIDDEN_FIELDS)
    require(not found, "prediction",
            f"host-computed outcome fields are forbidden: {found}")
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
        found = find_forbidden_fields(item, V2_LAYER_FORBIDDEN_FIELDS)
        require(not found, "prediction",
                f"host-computed outcome fields are forbidden: {found}")
        generated = {
            "generated_ids": item.get("generated_ids"),
            "generated_text": item.get("generated_text"),
            "empty_generation": item.get("empty_generation"),
            "tokenizer_binding_sha256": row.get("tokenizer_binding_sha256"),
            "decode_policy": row.get("decode_policy"),
        }
        text = verify_decoded_text(generated, binding, decode_fn)
        evidence = _layer_work(
            item, condition=condition, case_id=case["case_id"],
            layer=item["layer"])
        actual = _layer_actual(
            text, target, completion_valid=evidence["completion_valid"])
        layer_rows.append({
            "layer": item["layer"],
            "generated_ids": list(item["generated_ids"]),
            "generated_text": text,
            **evidence,
            **actual,
        })
    group_abstained = all(item["abstained"] for item in layer_rows)
    accepted = any(item["correct"] for item in layer_rows)
    complete = all(item["correct"] for item in layer_rows)
    invented = [item for item in layer_rows
                if item["invented_non_target"]]
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
        "group_abstained": group_abstained,
        "group_non_abstained": not group_abstained,
        "any_layer_abstained": any(item["abstained"]
                                   for item in layer_rows),
        "per_layer_non_abstention": {
            str(item["layer"]): not item["abstained"]
            for item in layer_rows
        },
        "invented_layers": [item["layer"] for item in invented],
        "hallucinated": bool(invented),
        "accepted_layer_correct": accepted,
        "complete_group_correct": complete,
        "empty_generation_layers": [item["layer"] for item in layer_rows
                                    if item["empty_generation"]],
        "followed_wrong_literal": followed_wrong,
    }


def _work_evidence(rows: Sequence[Mapping[str, Any]], *,
                   config: Mapping[str, Any],
                   enforce_budget: bool = False) -> dict:
    layers = [item for row in rows for item in row["layers"]]
    require(bool(layers), "physical_work", "no layer work evidence")
    stream_ids = [item["physical_stream_id"] for item in layers]
    require(len(set(stream_ids)) == len(stream_ids), "physical_work",
            "duplicate physical layer stream identities")
    names = (
        "generation_requests", "generation_prefill_forwards",
        "generation_decode_forwards", "autoregressive_forwards",
        "forward_calls", "generated_tokens", "prefill_tokens",
        "processed_positions", "layer_readout_streams",
    )
    totals = {name: sum(item["work"][name] for item in layers)
              for name in names}
    totals.update({"model_loads": 1, "tokenizer_loads": 1,
                   "optimizer_steps": 0, "backward_calls": 0,
                   "checkpoint_writes": 0})
    require(totals["generation_requests"] == len(layers)
            == totals["generation_prefill_forwards"]
            == totals["layer_readout_streams"], "physical_work",
            "requests, prefill forwards and layer streams do not reconcile")
    require(totals["autoregressive_forwards"]
            == totals["generation_prefill_forwards"]
            + totals["generation_decode_forwards"]
            == totals["forward_calls"] == totals["generated_tokens"],
            "physical_work", "autoregressive forward identities differ")
    bounded = dict(totals)
    bounded["prefill_forwards"] = bounded.pop("generation_prefill_forwards")
    bounded["decode_forwards"] = bounded.pop("generation_decode_forwards")
    bounded.pop("autoregressive_forwards")
    budget_check = "not_applicable_non_native_preparation"
    if enforce_budget:
        check_physical_work(bounded, config=config)
        budget_check = "passed"
    return {
        "budget_check": budget_check,
        "complete": True,
        "totals": totals,
        "per_condition": {
            condition: {
                name: sum(item["work"][name]
                          for row in rows if row["condition"] == condition
                          for item in row["layers"])
                for name in names}
            for condition in CONDITIONS},
        "per_layer": {
            str(layer): {
                name: sum(item["work"][name]
                          for row in rows for item in row["layers"]
                          if item["layer"] == layer)
                for name in names}
            for layer in LAYER_GROUP},
    }


def _metric_block(rows: Sequence[Mapping[str, Any]],
                  layer_group: Sequence[int]) -> dict:
    count = len(rows)
    require(count > 0, "scoring", "empty metric block")
    layer_output_count = count * len(layer_group)
    return {
        "cases": count,
        "layer_outputs": layer_output_count,
        "group_abstention_rate": (
            sum(1 for row in rows if row["group_abstained"]) / count),
        "group_non_abstention_rate": (
            sum(1 for row in rows if row["group_non_abstained"]) / count),
        "any_layer_abstention_rate_informational_only": (
            sum(1 for row in rows if row["any_layer_abstained"]) / count),
        "accepted_layer_accuracy": (
            sum(1 for row in rows if row["accepted_layer_correct"]) / count),
        "complete_group_accuracy": (
            sum(1 for row in rows if row["complete_group_correct"]) / count),
        "hallucination_rate": (
            sum(1 for row in rows if row["hallucinated"]) / count),
        "layer_output_non_abstention_rate": (
            sum(1 for row in rows for item in row["layers"]
                if not item["abstained"]) / layer_output_count),
        "layer_output_hallucination_rate": (
            sum(1 for row in rows for item in row["layers"]
                if item["invented_non_target"]) / layer_output_count),
        "wrong_value_follow_rate": (
            sum(1 for row in rows if row["followed_wrong_literal"]) / count),
        "empty_generation_rate": (
            sum(1 for row in rows for item in row["layers"]
                if item["empty_generation"]) / layer_output_count),
        "per_layer_abstention_rate": {
            str(layer): (
                sum(1 for row in rows
                    for item in row["layers"] if item["layer"] == layer
                    and item["abstained"])
                / count)
            for layer in layer_group
        },
        "per_layer_non_abstention_rate": {
            str(layer): (
                sum(1 for row in rows
                    for item in row["layers"] if item["layer"] == layer
                    and not item["abstained"])
                / count)
            for layer in layer_group
        },
        "per_layer_hallucination_rate": {
            str(layer): (
                sum(1 for row in rows
                    for item in row["layers"] if item["layer"] == layer
                    and item["invented_non_target"])
                / count)
            for layer in layer_group
        },
        "per_layer_accuracy": {
            str(layer): (
                sum(1 for row in rows
                    for item in row["layers"] if item["layer"] == layer
                    and item["correct"])
                / count)
            for layer in layer_group
        },
    }


def _read_predictions(path: str | Path,
                      condition_rows: Mapping[tuple[str, str], dict],
                      cases: Mapping[str, Mapping[str, Any]],
                      labels: Mapping[str, Mapping[str, Any]],
                      binding: Mapping[str, Any],
                      decode_fn: Callable[[Sequence[int]], str]) -> list[dict]:
    seen: set[tuple[str, str]] = set()
    scored: list[dict] = []
    for row in read_jsonl(path):
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
    return scored


def _condition_rows(prepared: Path,
                    split_prefix: str) -> dict[tuple[str, str], dict]:
    rows: dict[tuple[str, str], dict] = {}
    for condition in CONDITIONS:
        path = prepared / f"{split_prefix}_conditions_{condition}.jsonl"
        for row in read_jsonl(path):
            key = (condition, row["case_id"])
            require(key not in rows, "scoring", f"duplicate condition row {key}")
            rows[key] = row
    return rows


def check_dev_report_boundary(dev_report_path: str | Path | None, *,
                              config_path: str | Path,
                              prepared_manifest_sha256: str) -> dict:
    """Check report identity before tokenizer work; this is not full admission."""
    require(dev_report_path is not None, "final_admission",
            "sealed-final scoring requires --dev-report")
    path = Path(dev_report_path)
    require(path.is_file(), "final_admission",
            f"development score report is missing: {path}")
    report = read_json(path)
    require(report.get("format") == SCORE_FORMAT, "final_admission",
            "development report format differs")
    require(report.get("protocol_id") == PROTOCOL_ID, "final_admission",
            "development report protocol differs")
    require(report.get("split") == "dev", "final_admission",
            "development report split differs")
    require(report.get("final_labels_opened") is False, "final_admission",
            "development report must not have opened final labels")
    require(report.get("host_abstention_flags_used") is False,
            "final_admission",
            "development report used host abstention flags")
    require(report.get("development_evidence_admissible") is True,
            "final_admission",
            "development evidence is not admissible")
    require(report.get("config_sha256") == sha256_file(config_path),
            "final_admission",
            "development report did not bind the exact v2 config")
    require(report.get("prepared_manifest_sha256")
            == prepared_manifest_sha256, "final_admission",
            "development report did not bind the exact v2 fixture")
    return {
        "boundary_validated": True,
        "admitted": False,
        "dev_report_path": str(path),
        "dev_report_sha256": sha256_file(path),
        "status": "development_boundary_validated_not_independently_admitted",
        "requirements": ["valid_dev_score_report", "config_sha256_match",
                         "prepared_manifest_sha256_match",
                         "development_evidence_admissible"],
    }


def admit_dev_report(dev_report_path: str | Path | None, *,
                     config_path: str | Path,
                     prepared_manifest_sha256: str,
                     dev_predictions_path: str | Path | None,
                     prepared: str | Path = PREPARED_DIR,
                     decode_fn: Callable[[Sequence[int]], str]) -> dict:
    """Independently recompute development evidence before final-label access."""
    boundary = check_dev_report_boundary(
        dev_report_path, config_path=config_path,
        prepared_manifest_sha256=prepared_manifest_sha256)
    require(dev_predictions_path is not None, "final_admission",
            "development predictions are required for independent admission")
    expected = score_dev(
        dev_predictions_path, prepared=prepared, config_path=config_path,
        decode_fn=decode_fn)
    actual = read_json(boundary["dev_report_path"])
    require(canonical(actual) == canonical(expected), "final_admission",
            "development report differs from independently recomputed evidence")
    return {
        "admitted": True,
        "dev_report_path": boundary["dev_report_path"],
        "dev_report_sha256": boundary["dev_report_sha256"],
        "dev_predictions_path": str(dev_predictions_path),
        "dev_predictions_sha256": sha256_file(dev_predictions_path),
        "independently_recomputed": True,
        "native_execution_admissible": False,
        "status": "development_evidence_admitted",
        "requirements": boundary["requirements"] + [
            "independent_development_recomputation"],
    }


def _gate(value: float, *, minimum: float | None = None,
          maximum: float | None = None) -> dict:
    passed = True
    result: dict[str, Any] = {"value": value}
    if minimum is not None:
        result["min"] = minimum
        passed = passed and value >= minimum
    if maximum is not None:
        result["max"] = maximum
        passed = passed and value <= maximum
    result["passed"] = passed
    return result


def _score_split(
        predictions_path: str | Path,
        *, split: str, prepared: str | Path,
        config_path: str | Path,
        decode_fn: Callable[[Sequence[int]], str],
        dev_report_path: str | Path | None = None,
        dev_predictions_path: str | Path | None = None,
        native_execution: bool = False) -> dict:
    require(split in ("dev", "sealed_final"), "scoring",
            f"unknown split {split!r}")
    config = read_json(config_path)
    validate_config(config)
    prepared_dir = Path(prepared)
    manifest_sha = _manifest_hash(config, prepared_dir)
    admission = None
    if split == "sealed_final":
        # Admission happens before validate_manifest hashes any final label
        # file and before any final label is parsed.
        admission = admit_dev_report(
            dev_report_path, config_path=config_path,
            prepared_manifest_sha256=manifest_sha,
            dev_predictions_path=dev_predictions_path,
            prepared=prepared_dir, decode_fn=decode_fn)
    validate_manifest(
        prepared_dir, config, expected_files=FILES,
        manifest_format=MANIFEST_FORMAT, protocol_id=PROTOCOL_ID)
    binding = validate_tokenizer_binding(config["tokenizer_binding"])
    require(callable(decode_fn), "scoring",
            "a sealed-tokenizer decode function is required")
    prefix = "dev" if split == "dev" else "sealed_final"
    labels_path = prepared_dir / (
        "sealed_labels_dev.jsonl" if split == "dev"
        else "sealed_labels_final.jsonl")
    cases = {row["case_id"]: row
             for row in read_jsonl(prepared_dir / f"{prefix}_cases.jsonl")}
    labels = {row["case_id"]: row for row in read_jsonl(labels_path)}
    require(set(cases) == set(labels), "scoring",
            "case/label coverage differs")
    condition_rows = _condition_rows(prepared_dir, prefix)
    require(set(condition_rows) == {
        (condition, case_id) for condition in CONDITIONS for case_id in cases},
        "scoring", "condition coverage differs")

    scored = _read_predictions(
        predictions_path, condition_rows, cases, labels, binding, decode_fn)
    physical_work = _work_evidence(
        scored, config=config, enforce_budget=native_execution)
    by_condition = {condition: [row for row in scored
                               if row["condition"] == condition]
                    for condition in CONDITIONS}
    metrics = {
        condition: _metric_block(rows, LAYER_GROUP)
        for condition, rows in by_condition.items()
    }
    codec = read_json(prepared_dir / "codec.json")
    reconstruction = sum(
        1 for label in labels.values()
        if decode_compressed_literal(codec, label["compressed_code"])
        == label["target"]) / len(labels)
    gates_cfg = config["evaluation"]["gates"]
    exact = metrics["exact_context"]
    selected = metrics["selected_literal"]
    compressed = metrics["compressed_record"]
    no_memory = metrics["no_memory"]
    wrong_value = metrics["wrong_value"]

    gates: dict[str, dict] = {
        "exact_context_accepted": _gate(
            exact["accepted_layer_accuracy"],
            minimum=gates_cfg["exact_context_accepted_min"]),
        "selected_literal_accepted": _gate(
            selected["accepted_layer_accuracy"],
            minimum=gates_cfg["selected_literal_accepted_min"]),
        "selected_literal_complete": _gate(
            selected["complete_group_accuracy"],
            minimum=gates_cfg["selected_literal_complete_min"]),
        "no_memory_group_abstain": _gate(
            no_memory["group_abstention_rate"],
            minimum=gates_cfg["no_memory_group_abstain_min"]),
        "no_memory_group_non_abstain": _gate(
            no_memory["group_non_abstention_rate"],
            maximum=gates_cfg["no_memory_group_non_abstain_max"]),
        "no_memory_per_layer_non_abstain": _gate(
            max(no_memory["per_layer_non_abstention_rate"].values()),
            maximum=gates_cfg["no_memory_per_layer_non_abstain_max"]),
        "no_memory_hallucination": _gate(
            no_memory["hallucination_rate"],
            maximum=gates_cfg["no_memory_hallucination_max"]),
    }
    reconstruction_gate = _gate(
        reconstruction,
        minimum=gates_cfg["compressed_reconstruction_control_min"])
    reconstruction_gate["evidence_role"] = (
        "host_reconstruction_audit_control_only")
    reconstruction_gate["native_readout_evidence"] = False
    gates["compressed_reconstruction_control"] = reconstruction_gate

    wrong_correct = sum(
        1 for row in by_condition["wrong_value"]
        for item in row["layers"] if item["correct"]) / (
            len(by_condition["wrong_value"]) * len(LAYER_GROUP))
    gates["wrong_value_correct_target"] = _gate(
        wrong_correct,
        maximum=gates_cfg["wrong_value_correct_target_max"])
    require(metrics["wrong_value"]["accepted_layer_accuracy"]
            <= gates_cfg["selected_literal_accepted_min"], "scoring",
            "wrong-value control unexpectedly passes the readout gate")

    blocked_reason = (
        "no external native compressed-channel receipt exists; the model sees "
        "only codec_id plus an integer and the host codec table is unavailable; "
        "host decode_compressed_literal is a reconstruction audit/control, not "
        "a learned-readout measurement")
    blocked_reason = ("no external native compressed-channel receipt exists; "
                      "the model sees only codec_id plus an integer and the "
                      "host codec table is unavailable; host "
                      "decode_compressed_literal is a reconstruction "
                      "audit/control, not a learned-readout measurement")
    for name, value in (
        ("compressed_accepted_native",
         compressed["accepted_layer_accuracy"]),
        ("compressed_complete_native",
         compressed["complete_group_accuracy"]),
    ):
        gates[name] = {
            "value": value,
            "min": (gates_cfg["compressed_accepted_native_min"]
                    if name == "compressed_accepted_native"
                    else gates_cfg["compressed_complete_native_min"]),
            "passed": False,
            "evaluable": False,
            "blocked": True,
            "evidence_status": "blocked_no_native_compressed_channel_receipt",
            "block_reason": blocked_reason,
            "reported_actual_output_metric": True,
            "not_a_host_reconstruction_gate": True,
        }

    non_native_gate_names = [name for name, gate in gates.items()
                             if not gate.get("blocked", False)]
    non_native_gates_passed = all(
        gates[name]["passed"] for name in non_native_gate_names)
    native_blocked = [name for name, gate in gates.items()
                      if gate.get("blocked", False)]
    readout_useful = all(gate["passed"] for gate in gates.values())
    if not non_native_gates_passed:
        status = "readout_not_useful"
    elif native_blocked:
        status = "native_compressed_channel_blocked"
    else:
        status = "readout_useful"

    reporting_complete = all(
        condition in metrics for condition in CONDITIONS
    ) and all(
        set(metrics[condition]["per_layer_accuracy"])
        == {str(layer) for layer in LAYER_GROUP}
        for condition in CONDITIONS
    )
    development_evidence_admissible = (
        split == "dev" and non_native_gates_passed and reporting_complete
        and reconstruction_gate["passed"] and physical_work["complete"])
    report = {
        "format": SCORE_FORMAT,
        "protocol_id": PROTOCOL_ID,
        "split": split,
        "status": status,
        "native_computation": False,
        "answer_index_used": False,
        "host_abstention_flags_used": False,
        "generated_output_metrics_only": True,
        "config_sha256": sha256_file(config_path),
        "prepared_manifest_sha256": manifest_sha,
        "tokenizer_binding_sha256": tokenizer_binding_sha(binding),
        "abstention_policy": {
            "policy_id": ABSTENTION_POLICY_ID,
            "group_abstention_requires_all_layers": True,
            "any_layer_abstention_acceptance": False,
            "metrics_source": "re_decoded_generated_text",
            "host_abstention_flags_used": False,
            "accepted_layer_substitute_for_group_abstention": False,
            "accepted_layer_substitute_for_complete_group": False,
            "complete_group_separate": True,
        },
        "conditions": metrics,
        "per_case_outputs": scored,
        "physical_work": physical_work,
        "reconstruction": {
            "exact_literal_available_rate": 1.0,
            "host_compressed_reconstruction_accuracy": reconstruction,
            "literal_minus_compressed": 1.0 - reconstruction,
            "evidence_role": "reconstruction_audit_control_only",
            "host_decode_is_native_readout_evidence": False,
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
        "compressed_channel": {
            "native_input_fields": ["codec_id", "code"],
            "native_input_is_codec_id_plus_integer": True,
            "host_codec_table_visible_to_model": False,
            "host_decode_is_native_readout_evidence": False,
            "native_compressed_readout_status":
                "blocked_no_native_compressed_channel_receipt",
            "native_compressed_gates": native_blocked,
            "blocked_reason": blocked_reason,
            "actual_generated_compressed_metrics": {
                "accepted_layer_accuracy":
                    compressed["accepted_layer_accuracy"],
                "complete_group_accuracy":
                    compressed["complete_group_accuracy"],
                "per_layer_accuracy":
                    compressed["per_layer_accuracy"],
                "group_abstention_rate":
                    compressed["group_abstention_rate"],
            },
        },
        "accepted_layer_vs_complete_group": {
            condition: {
                "accepted_layer_accuracy":
                    metrics[condition]["accepted_layer_accuracy"],
                "complete_group_accuracy":
                    metrics[condition]["complete_group_accuracy"],
                "gap": metrics[condition]["accepted_layer_accuracy"]
                - metrics[condition]["complete_group_accuracy"],
                "group_abstention_rate":
                    metrics[condition]["group_abstention_rate"],
                "group_non_abstention_rate":
                    metrics[condition]["group_non_abstention_rate"],
                "any_layer_abstention_rate_informational_only":
                    metrics[condition][
                        "any_layer_abstention_rate_informational_only"],
                "per_layer_accuracy":
                    metrics[condition]["per_layer_accuracy"],
                "per_layer_non_abstention_rate":
                    metrics[condition]["per_layer_non_abstention_rate"],
                "per_layer_hallucination_rate":
                    metrics[condition]["per_layer_hallucination_rate"],
            }
            for condition in CONDITIONS
        },
        "gates": gates,
        "blocked_native_gates": native_blocked,
        "non_native_gates_passed": non_native_gates_passed,
        "development_evidence_admissible": development_evidence_admissible,
        "readout_useful": readout_useful,
        "final_labels_opened": split == "sealed_final",
        "final_admission": admission,
        "learned_retrieval_interpretable": False,
        "learned_retrieval_status": "blocked_or_not_evaluated",
        "storage_expansion_allowed": False,
        "storage_expansion_status":
            "blocked_until_useful_readout_and_new_protocol",
        "controls_use_identical_scorer": True,
        "conditions_use_identical_scorer": True,
        "source_kv_loaded": False,
    }
    return report


def score_dev(
        predictions_path: str | Path, *,
        prepared: str | Path = PREPARED_DIR,
        config_path: str | Path = CONFIG_PATH,
        decode_fn: Callable[[Sequence[int]], str],
        native_execution: bool = False) -> dict:
    """Development scoring entry point; final labels are never opened."""
    return _score_split(
        predictions_path, split="dev", prepared=prepared,
        config_path=config_path, decode_fn=decode_fn,
        native_execution=native_execution)


def score_sealed_final(
        predictions_path: str | Path, *,
        dev_report_path: str | Path | None,
        dev_predictions_path: str | Path | None,
        prepared: str | Path = PREPARED_DIR,
        config_path: str | Path = CONFIG_PATH,
        decode_fn: Callable[[Sequence[int]], str],
        native_execution: bool = False) -> dict:
    """Sealed-final entry point; refuses before reading final labels."""
    return _score_split(
        predictions_path, split="sealed_final", prepared=prepared,
        config_path=config_path, decode_fn=decode_fn,
        dev_report_path=dev_report_path,
        dev_predictions_path=dev_predictions_path,
        native_execution=native_execution)


def score_r1_v2(
        predictions_path: str | Path, *, split: str,
        prepared: str | Path = PREPARED_DIR,
        decode_fn: Callable[[Sequence[int]], str],
        dev_report_path: str | Path | None = None,
        dev_predictions_path: str | Path | None = None,
        native_execution: bool = False) -> dict:
    if split == "dev":
        return score_dev(
            predictions_path, prepared=prepared, config_path=config_path,
            decode_fn=decode_fn)
    if split == "sealed_final":
        return score_sealed_final(
            predictions_path, dev_report_path=dev_report_path,
            dev_predictions_path=dev_predictions_path,
            prepared=prepared, config_path=config_path, decode_fn=decode_fn,
            native_execution=native_execution)
    raise ProtocolError("scoring", f"unknown split {split!r}")


# ---------------------------------------------------------------------------
# Review-only preflight


def preflight(config_path: str | Path = CONFIG_PATH,
              prepared: str | Path = PREPARED_DIR,
              *, launch: bool = False,
              authorization: str | Path | None = None) -> dict:
    config = read_json(config_path)
    validate_config(config)
    prepared_dir = Path(prepared)
    manifest = validate_manifest(
        prepared_dir, config, expected_files=FILES,
        manifest_format=MANIFEST_FORMAT, protocol_id=PROTOCOL_ID)
    receipt, exit_code = preflight_receipt(
        config_path=config_path,
        manifest_path=prepared_dir / "manifest.json",
        config=config,
        protocol_id=PROTOCOL_ID,
        manifest_sha256=manifest["manifest_sha256"],
        authorization_path=authorization,
        launch=launch)
    receipt["budget_readiness"] = {
        "physical_limits": deepcopy(config["budget"]["physical_limits"]),
        "derived_workload": workload_derivation()["total"],
        "historical_v1_limits": deepcopy(
            config["budget"]["historical_v1_limits"]),
        "historical_request_accounting": deepcopy(
            config["budget"]["historical_request_accounting"]),
        "zero_optimizer_and_backward": True,
        "native_jobs": 0,
    }
    receipt["readiness"] = {
        "preparation_only": True,
        "native_execution_allowed": False,
        "authorization_receipt_required": True,
        "authorization_receipt_present": authorization is not None,
        "native_compressed_channel_receipt_present": False,
        "native_compressed_gate_status":
            "blocked_no_native_compressed_channel_receipt",
        "final_labels_openable": False,
        "final_scoring_requires_dev_admission": True,
        "v1_files_preserved": True,
        "v1_config_sha256": V1_CONFIG_SHA256,
        "v1_manifest_sha256": V1_MANIFEST_SHA256,
    }
    receipt["amendment_id"] = config["amendment"]["amendment_id"]
    receipt["exit_code"] = exit_code
    return receipt

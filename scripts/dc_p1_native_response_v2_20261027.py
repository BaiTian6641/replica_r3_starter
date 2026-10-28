"""DC-P1 v2: corrected-accounting, matched-control, staged native response study.

What v2 changes relative to the preserved v1 package
(``configs/dc_p1_native_response_protocol_v1.json`` and
``data/dc_p1_native_response_20261027``, both untouched):

1. ``dc_p1_v2_a1_matched_control``: a matched trained control with
   ``answer_span_ce_weight = 0.0`` runs the identical objective, masks,
   teaching rows, optimizer, learning rate and one-step-per-event clock for the
   same 36 development events.  The v1 package compared 0.5 against only a
   no-update and a reset-P arm, so "more optimisation" and "the span term" were
   confounded.
2. ``dc_p1_v2_a2_work_unit_correction``: physical work is counted at real
   execution points (model/tokenizer loads, training forwards, prefill/decode/
   autoregressive forwards, processed text positions including no-cache growth,
   generated tokens, backwards, optimizer steps, parity probes, checkpoint
   writes, restart work, temporary storage) instead of one forward per
   generation request.  v1's ``forward_calls = 368`` is preserved in
   ``budget.historical_v1`` as *incorrect historical accounting*; it is never
   reused as a ceiling.
3. ``dc_p1_v2_a3_staged_admission``: fit -> development -> final are separate
   scoring entry points with independently sealed, digest-checked,
   tokenizer-backed receipts.  No development generation or label access
   happens before fit admission; no final teaching, generation or label access
   happens before development admission; the final shot is one-shot.
4. ``dc_p1_v2_a4_temporal_semantics``: post-correction ordinary queries resolve
   the value current at their query epoch, and historical queries name the
   historical event explicitly.  v1's ``paraphrase`` role implicitly asked for
   a superseded value after correction teaching.
5. ``dc_p1_v2_a5_objective_audit``: teaching responses are the answer tokens
   followed by the terminal EOS, so the answer-span-plus-EOS mask coincides
   with the full-response mask.  The audit therefore *declares* the treatment
   objective a global CE rescale of the identical full-response CE, and the
   claim is narrowed to that.
6. ``dc_p1_v2_a6_final_confirmation_only``: the sealed final stage is
   treatment-only (12 updates, 20 cases) confirmation evidence, not a
   0.5-versus-0.0 comparison.

This module is preparation only: it builds the deterministic fixture, derives
and validates the physical-work schedule from declared units, validates the
protocol and a pre-existing external authorization receipt, and refuses to
launch anything.  It imports no torch, loads no MiniCPM5, loads no tokenizer,
constructs no optimizer, performs no generation and writes no authorization.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import shlex
import sys
from typing import Any, Callable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from dc_protocol_common_20261027 import (  # noqa: E402
    AUTHORIZATION_FORMAT as COMMON_AUTHORIZATION_FORMAT,
    EVALUATION_CASE_FORBIDDEN_FIELDS, GUARD_BYTES, PREFLIGHT_FORMAT,
    ProtocolError, assert_routing_is_answer_free, assert_target_absent,
    canonical, check_work_budget_v2, find_forbidden_fields, read_json,
    read_jsonl, require, require_committed_bytes, seal_manifest, sha256_file,
    sha256_json, tokenizer_binding_sha, validate_baseline_model,
    validate_guards, validate_manifest, validate_tokenizer_binding,
    verify_sealed_record, write_json, write_jsonl,
)
from replica_lab.dc_response_objective import (  # noqa: E402
    CLAIM_GLOBAL_RESCALE, CLAIM_SPAN_SPECIFIC, PUBLIC_API as OBJECTIVE_API,
    TERMINAL_EOS_ID,
)
from replica_lab.dc_staged_evaluation import (  # noqa: E402
    PUBLIC_API as STAGED_API, RECEIPT_FORMAT as STAGE_RECEIPT_FORMAT,
    SCORE_REPORT_FORMAT, STAGE_DEVELOPMENT, STAGE_FINAL, STAGE_FIT,
    STAGE_ORDER,
)

PROTOCOL_ID = "dc_p1_native_response_v2"
CONFIG_FORMAT = "replica.dc_p1_native_response.protocol.v2"
MANIFEST_FORMAT = "replica.dc_p1_native_response.prepared.v2"
CASE_FORMAT = "replica.dc_p1_native_response_v2.eval_case.v1"
TEACHING_FORMAT = "replica.dc_p1_native_response_v2.teaching_case.v1"
LABEL_FORMAT = "replica.dc_p1_native_response_v2.label.v1"
PREDICTION_FORMAT = "replica.dc_p1_native_response_v2.prediction.v1"
VALIDATION_FORMAT = "replica.dc_p1_native_response_v2.validation.v1"
TOKEN_LENGTHS_FORMAT = "replica.dc_p1_native_response_v2.token_lengths.v1"
PREFLIGHT_FORMAT_V2 = PREFLIGHT_FORMAT.replace(".v1", ".v2")
AUTHORIZATION_FORMAT_V2 = "replica.dc_protocol.authorization.v2"

CONFIG_PATH = ROOT / "configs" / "dc_p1_native_response_protocol_v2.json"
PREPARED_DIR = ROOT / "data" / "dc_p1_native_response_v2_20261027"
TOKEN_LENGTHS_PATH = PREPARED_DIR / "token_lengths.v2.json"
V1_CONFIG_PATH = ROOT / "configs" / "dc_p1_native_response_protocol_v1.json"
V1_PREPARED_DIR = ROOT / "data" / "dc_p1_native_response_20261027"
AUTHORIZATION_PATH = (ROOT / "reports" / "dc_p1_native_response_v2_20261027"
                      / "authorization.json")

TRAIN_GROUPS = 12
SEALED_GROUPS = 4
FIT_ROLES = ("fit_primary", "fit_correction", "fit_abstention")
DEV_ROLES = ("post_correction_current", "new_instance_current",
             "historical_state", "abstention", "retention")
PRIMARY_ROLES = ("post_correction_current", "new_instance_current")
ARMS = ("treatment", "matched_trained", "no_update", "reset_P")
FINAL_ARMS = ("treatment",)
FILES = (
    "teaching.jsonl",
    "fit_cases.jsonl",
    "dev_cases.jsonl",
    "sealed_labels_fit.jsonl",
    "sealed_labels_dev.jsonl",
    "sealed_final_teaching.jsonl",
    "sealed_final_fit_cases.jsonl",
    "sealed_final_cases.jsonl",
    "sealed_final_labels_fit.jsonl",
    "sealed_labels_final.jsonl",
)
EVENTS = (
    ("primary", 0),
    ("correction", 1),
    ("abstention", 2),
)
RESOLUTION_CURRENT = "current_at_query_epoch"
RESOLUTION_HISTORICAL = "historical_explicit_event"
RESOLUTION_EXPLICIT = "explicit_event_reference"
RESOLUTION_STATELESS = "stateless_rule"
RESPONSE_LAYOUT = "answer_tokens_then_terminal_eos"
ANSWER_SPAN_RULE = "answer_content_offsets_plus_terminal_eos"
PARENT_CHECKPOINT_ID = "dcp1v2:parent:minicpm5-f9740005"
SYLLABLES = (
    "ka", "ro", "mi", "ta", "ne", "su", "vi", "lo", "za", "pe",
    "du", "fi", "go", "ha", "je", "qu", "we", "xi", "yo", "bu",
)


# ---------------------------------------------------------------------------
# Deterministic synthetic words


def _word(*parts: Any) -> str:
    digest = hashlib.sha256(
        canonical(list(parts)).encode("utf-8")).hexdigest()
    indices = [int(digest[index:index + 2], 16) % len(SYLLABLES)
               for index in range(0, 9, 2)]
    return "".join(SYLLABLES[index] for index in indices)


def _group_name(index: int, sealed: bool) -> str:
    return f"p1v2src_{'sealed' if sealed else 'dev'}_{index:02d}"


def _case_id(group: str, role: str) -> str:
    return f"{group}_{role}"


def _prompt_sha(prompt: str) -> str:
    return sha256_json({"prompt": prompt})


def _event_checkpoint(group: str, position: int) -> str:
    return f"dcp1v2:{group}:ev{position:02d}"


def _state_sha(state: Mapping[str, Any]) -> str:
    return sha256_json({key: state[key] for key in sorted(state)})


def _facts(group: str, *, sealed: bool) -> dict[str, str]:
    return {
        "group": group,
        "primary_entity": _word("p1v2-entity", group, "primary"),
        "secondary_entity": _word("p1v2-entity", group, "secondary"),
        "absent_entity": _word("p1v2-entity", group, "absent"),
        "primary_value": _word("p1v2-value", group, "primary"),
        "secondary_value": _word("p1v2-value", group, "secondary"),
        "corrected_value": _word("p1v2-value", group, "corrected"),
        "retention_word": _word("p1v2-retain", group, "word"),
    }


def _teaching_prompt(role: str, entity: str, position: int) -> str:
    if role == "primary":
        return f"At event {position}, which phrase is assigned to locker {entity}?"
    if role == "correction":
        return (f"After the correction at event {position}, which phrase belongs "
                f"to locker {entity}?")
    return f"At event {position}, what phrase is recorded for locker {entity}?"


def _current_prompt(entity: str, variant: int) -> str:
    return (
        f"Which phrase is current for locker {entity}?",
        f"State the phrase locker {entity} holds now.",
        f"Return the current phrase for locker {entity}.",
    )[variant % 3]


def _new_instance_prompt(entity: str, variant: int) -> str:
    return (
        f"Which phrase is assigned to locker {entity}?",
        f"Report the phrase held by locker {entity}.",
        f"Name the phrase for locker {entity}.",
    )[variant % 3]


def _historical_prompt(entity: str, variant: int) -> str:
    return (
        f"At event 0, which phrase did locker {entity} hold?",
        f"Before the correction, at event 0, what phrase was recorded for "
        f"locker {entity}?",
        f"Return the phrase locker {entity} held at event 0.",
    )[variant % 3]


def _abstention_prompt(entity: str, variant: int) -> str:
    return (
        f"Which phrase belongs to locker {entity}?",
        f"Report the phrase for locker {entity}.",
        f"What phrase is held by locker {entity}?",
    )[variant % 3]


def _retention_prompt(word: str) -> str:
    return (f"Reverse the letters of the word '{word}' and answer with only "
            "the reversed word.")


# ---------------------------------------------------------------------------
# Fixture construction


def _build_group(group: str, index: int, *, sealed: bool) -> dict[str, list[dict]]:
    facts = _facts(group, sealed=sealed)
    e1 = facts["primary_entity"]
    e2 = facts["secondary_entity"]
    e3 = facts["absent_entity"]
    v1 = facts["primary_value"]
    v2 = facts["secondary_value"]
    v3 = facts["corrected_value"]
    ordinal_base = index * len(EVENTS)
    states: list[dict[str, str]] = [
        {e1: v1},
        {e1: v3},
        {e1: v3, e2: v2, e3: "ABSENT"},
    ]
    sources = {
        "primary": f"Archive {group}: locker {e1} uses phrase {v1}.",
        "correction": (f"Registry {group}: locker {e1} used phrase {v1}. "
                       f"Correction: locker {e1} now uses phrase {v3}."),
        "abstention": (f"Registry {group}: locker {e2} uses phrase {v2}. "
                       f"No record exists for locker {e3}."),
    }
    targets = {"primary": v1, "correction": v3, "abstention": "UNKNOWN"}
    superseded = {"primary": None, "correction": v1, "abstention": None}
    teaching: list[dict] = []
    for position, (role, _) in enumerate(EVENTS):
        prompt = _teaching_prompt(role, e1 if role != "abstention" else e3,
                                  position)
        before = _event_checkpoint(group, position - 1) if position else (
            PARENT_CHECKPOINT_ID)
        teaching.append({
            "format": TEACHING_FORMAT,
            "case_id": _case_id(group, f"teach_{role}"),
            "split_role": "train",
            "role": role,
            "source_group": group,
            "event_position": position,
            "event_ordinal": ordinal_base + position,
            "query_epoch": position,
            "resolution_mode": RESOLUTION_EXPLICIT,
            "source": sources[role],
            "prompt": prompt,
            "prompt_sha256": _prompt_sha(prompt),
            "target": targets[role],
            "superseded_value": superseded[role],
            "response_layout": RESPONSE_LAYOUT,
            "answer_span_rule": ANSWER_SPAN_RULE,
            "eos_id": TERMINAL_EOS_ID,
            "state_after": states[position],
            "state_after_sha256": _state_sha(states[position]),
            "checkpoint_id_before": before,
            "checkpoint_id_after": _event_checkpoint(group, position),
            "source_context_allowed": True,
            "source_kv_allowed": True,
            "routing_metadata": {"mode": "fixed", "expert_id": 0},
        })

    fit_cases: list[dict] = []
    fit_labels: list[dict] = []
    fit_split = "final_fit" if sealed else "fit"
    for position, (role, _) in enumerate(EVENTS):
        teaching_row = teaching[position]
        case = {
            "format": CASE_FORMAT,
            "case_id": _case_id(group, FIT_ROLES[position]),
            "split_role": fit_split,
            "role": FIT_ROLES[position],
            "source_group": group,
            "event_position": position,
            "query_epoch": position,
            "resolution_mode": RESOLUTION_EXPLICIT,
            "historical_event_position": None,
            "prompt": teaching_row["prompt"],
            "prompt_sha256": teaching_row["prompt_sha256"],
            "inference_inputs": ["prompt"],
            "source_context_present": False,
            "source_kv_loaded": False,
        }
        fit_cases.append(case)
        fit_labels.append({
            "format": LABEL_FORMAT,
            "case_id": case["case_id"],
            "split_role": fit_split,
            "role": FIT_ROLES[position],
            "source_group": group,
            "target": targets[role],
            "query_epoch": position,
            "resolution_mode": RESOLUTION_EXPLICIT,
            "resolved_event_position": position,
            "state_sha256": _state_sha(states[position]),
            "event_checkpoint_id": _event_checkpoint(group, position),
        })

    variant = index % 3
    evaluation_split = "sealed_final" if sealed else "dev"
    specs = [
        ("post_correction_current", _current_prompt(e1, variant), v3,
         RESOLUTION_CURRENT, 2, 1, None),
        ("new_instance_current", _new_instance_prompt(e2, variant + 1), v2,
         RESOLUTION_CURRENT, 2, 2, None),
        ("historical_state", _historical_prompt(e1, variant + 2), v1,
         RESOLUTION_HISTORICAL, 0, 0, 0),
        ("abstention", _abstention_prompt(e3, variant + 2), "UNKNOWN",
         RESOLUTION_CURRENT, 2, 2, None),
        ("retention", _retention_prompt(facts["retention_word"]),
         facts["retention_word"][::-1], RESOLUTION_STATELESS, 2, None, None),
    ]
    dev_cases: list[dict] = []
    dev_labels: list[dict] = []
    for role, prompt, target, mode, epoch, resolved, historical in specs:
        case = {
            "format": CASE_FORMAT,
            "case_id": _case_id(group, role),
            "split_role": evaluation_split,
            "role": role,
            "source_group": group,
            "query_epoch": epoch,
            "resolution_mode": mode,
            "historical_event_position": historical,
            "prompt": prompt,
            "prompt_sha256": _prompt_sha(prompt),
            "inference_inputs": ["prompt"],
            "source_context_present": False,
            "source_kv_loaded": False,
        }
        dev_cases.append(case)
        label = {
            "format": LABEL_FORMAT,
            "case_id": case["case_id"],
            "split_role": evaluation_split,
            "role": role,
            "source_group": group,
            "target": target,
            "query_epoch": epoch,
            "resolution_mode": mode,
            "resolved_event_position": resolved,
            "state_sha256": (_state_sha(states[resolved])
                             if resolved is not None else None),
            "event_checkpoint_id": (_event_checkpoint(group, resolved)
                                    if resolved is not None else None),
        }
        if role == "retention":
            label["retention_input"] = facts["retention_word"]
            label["retention_rule"] = "reverse"
        if role == "historical_state":
            label["historical_event_position"] = 0
        if role == "post_correction_current":
            label["superseded_value"] = v1
            label["current_value_epoch"] = 2
        dev_labels.append(label)
    return {
        "teaching": teaching,
        "fit_cases": fit_cases,
        "fit_labels": fit_labels,
        "dev_cases": dev_cases,
        "dev_labels": dev_labels,
    }


def build_fixture(config_path: str | Path = CONFIG_PATH,
                  output: str | Path = PREPARED_DIR) -> dict:
    """Build a fresh deterministic v2 fixture; never overwrite evidence."""
    config = read_json(config_path)
    validate_config(config)
    require(config["prepared"]["groups"] == TRAIN_GROUPS, "build",
            "declared development group count differs")
    require(config["prepared"]["sealed_groups"] == SEALED_GROUPS, "build",
            "declared sealed group count differs")
    root = Path(output)
    require(not root.exists(), "build", f"output is not fresh: {root}")
    root.mkdir(parents=True)
    teaching: list[dict] = []
    fit_cases: list[dict] = []
    fit_labels: list[dict] = []
    dev_cases: list[dict] = []
    dev_labels: list[dict] = []
    for index in range(TRAIN_GROUPS):
        built = _build_group(_group_name(index, False), index, sealed=False)
        teaching.extend(built["teaching"])
        fit_cases.extend(built["fit_cases"])
        fit_labels.extend(built["fit_labels"])
        dev_cases.extend(built["dev_cases"])
        dev_labels.extend(built["dev_labels"])
    final_teaching: list[dict] = []
    final_fit_cases: list[dict] = []
    final_fit_labels: list[dict] = []
    final_cases: list[dict] = []
    final_labels: list[dict] = []
    for offset in range(SEALED_GROUPS):
        index = TRAIN_GROUPS + offset
        built = _build_group(_group_name(offset, True), index, sealed=True)
        # Sealed-group rows live only in sealed files and are never part of the
        # development teaching stream.
        final_teaching.extend(built["teaching"])
        final_fit_cases.extend(built["fit_cases"])
        final_fit_labels.extend(built["fit_labels"])
        final_cases.extend(built["dev_cases"])
        final_labels.extend(built["dev_labels"])
    files = {
        "teaching.jsonl": teaching,
        "fit_cases.jsonl": fit_cases,
        "dev_cases.jsonl": dev_cases,
        "sealed_labels_fit.jsonl": fit_labels,
        "sealed_labels_dev.jsonl": dev_labels,
        "sealed_final_teaching.jsonl": final_teaching,
        "sealed_final_fit_cases.jsonl": final_fit_cases,
        "sealed_final_cases.jsonl": final_cases,
        "sealed_final_labels_fit.jsonl": final_fit_labels,
        "sealed_labels_final.jsonl": final_labels,
    }
    for name, rows in files.items():
        write_jsonl(root / name, rows)
    body = {
        "format": MANIFEST_FORMAT,
        "protocol_id": PROTOCOL_ID,
        "declared_role": "deterministic_role_sealed_synthetic_text",
        "native_computation": False,
        "model_loaded": False,
        "tokenizer_loaded": False,
        "downloads": 0,
        "heldout_opened": False,
        "seed": config.get("seed"),
        "format_version": 2,
        "supersedes_manifest": "data/dc_p1_native_response_20261027/manifest.json",
        "temporal_semantics": {
            "events_per_group": len(EVENTS),
            "teaching_query_epoch": "event_position",
            "ordinary_queries": RESOLUTION_CURRENT,
            "historical_queries": RESOLUTION_HISTORICAL,
            "retention_queries": RESOLUTION_STATELESS,
            "post_correction_current_role": "post_correction_current",
            "historical_role": "historical_state",
        },
        "teaching_layout": {
            "response": RESPONSE_LAYOUT,
            "answer_span": ANSWER_SPAN_RULE,
            "eos_id": TERMINAL_EOS_ID,
            "masks_coincide_by_construction": True,
        },
        "fit_prompt_policy": "identical_to_teaching_prompt",
        "counts": {
            "teaching": len(teaching),
            "fit_cases": len(fit_cases),
            "dev_cases": len(dev_cases),
            "sealed_final_teaching": len(final_teaching),
            "sealed_final_fit_cases": len(final_fit_cases),
            "sealed_final_cases": len(final_cases),
            "labels_fit": len(fit_labels),
            "labels_dev": len(dev_labels),
            "sealed_final_labels_fit": len(final_fit_labels),
            "labels_final": len(final_labels),
        },
        "source_groups": {
            "development": sorted({row["source_group"] for row in teaching}),
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
    files = {name: read_jsonl(root / name) for name in FILES}
    teaching = files["teaching.jsonl"]
    fit_cases = files["fit_cases.jsonl"]
    dev_cases = files["dev_cases.jsonl"]
    final_teaching = files["sealed_final_teaching.jsonl"]
    final_fit_cases = files["sealed_final_fit_cases.jsonl"]
    final_cases = files["sealed_final_cases.jsonl"]
    fit_labels = files["sealed_labels_fit.jsonl"]
    dev_labels = files["sealed_labels_dev.jsonl"]
    final_fit_labels = files["sealed_final_labels_fit.jsonl"]
    final_labels = files["sealed_labels_final.jsonl"]

    require(len(teaching) == TRAIN_GROUPS * len(EVENTS), "fixture",
            "teaching count differs")
    require(len(fit_cases) == TRAIN_GROUPS * len(EVENTS), "fixture",
            "fit case count differs")
    require(len(dev_cases) == TRAIN_GROUPS * len(DEV_ROLES), "fixture",
            "development case count differs")
    require(len(final_teaching) == SEALED_GROUPS * len(EVENTS), "fixture",
            "sealed teaching count differs")
    require(len(final_fit_cases) == SEALED_GROUPS * len(EVENTS), "fixture",
            "sealed fit case count differs")
    require(len(final_cases) == SEALED_GROUPS * len(DEV_ROLES), "fixture",
            "sealed case count differs")
    for rows, labels in ((fit_cases, fit_labels),
                         (final_fit_cases, final_fit_labels),
                         (dev_cases, dev_labels),
                         (final_cases, final_labels)):
        require(len(rows) == len(labels), "fixture",
                "case/label coverage differs")

    for group_rows, sealed in ((teaching, False), (final_teaching, True)):
        by_group: dict[str, list[dict]] = {}
        for row in group_rows:
            require(row.get("format") == TEACHING_FORMAT, "fixture",
                    "teaching row format differs")
            require(row.get("split_role") == "train", "fixture",
                    "teaching split_role differs")
            require(row.get("role") in ("primary", "correction", "abstention"),
                    "fixture", "unknown teaching role")
            require(isinstance(row.get("source"), str) and row["source"],
                    "fixture", "teaching source is empty")
            require(row.get("source_context_allowed") is True, "fixture",
                    "teaching must fail closed without source context")
            require(row.get("source_kv_allowed") is True, "fixture",
                    "teaching source-KV policy differs")
            assert_routing_is_answer_free(row.get("routing_metadata"))
            require(row["prompt_sha256"] == _prompt_sha(row["prompt"]),
                    "fixture", "teaching prompt hash differs")
            require(row["response_layout"] == RESPONSE_LAYOUT, "fixture",
                    "teaching response layout differs")
            require(row["answer_span_rule"] == ANSWER_SPAN_RULE, "fixture",
                    "teaching answer-span rule differs")
            require(row["eos_id"] == TERMINAL_EOS_ID, "fixture",
                    "teaching terminal EOS differs")
            require(row["query_epoch"] == row["event_position"], "fixture",
                    "teaching query epoch must equal its event position")
            require(row["resolution_mode"] == RESOLUTION_EXPLICIT, "fixture",
                    "teaching rows must name the referenced event explicitly")
            require(row["checkpoint_id_after"]
                    == _event_checkpoint(row["source_group"],
                                     row["event_position"]), "fixture",
                    "teaching checkpoint identity differs")
            require(row["checkpoint_id_before"]
                    in {PARENT_CHECKPOINT_ID,
                        _event_checkpoint(row["source_group"],
                                          row["event_position"] - 1)},
                    "fixture", "teaching predecessor checkpoint differs")
            require(row["state_after_sha256"] == _state_sha(row["state_after"]),
                    "fixture", "teaching state digest differs")
            if row["role"] == "abstention":
                require(row["target"] == "UNKNOWN", "fixture",
                        "abstention teaching target must be UNKNOWN")
            if row["role"] == "correction":
                require(row["superseded_value"] != row["target"], "fixture",
                        "correction must change the value")
                require(row["target"] in row["state_after"].values(),
                        "fixture",
                        "correction state does not hold the corrected value")
                require(row["superseded_value"]
                        not in row["state_after"].values(), "fixture",
                        "correction state still holds the superseded value")
            if row["role"] == "primary":
                require(row["target"] in row["state_after"].values(),
                        "fixture", "primary state does not hold its value")
            assert_target_absent(row["prompt"], row["target"], check="fixture")
            by_group.setdefault(row["source_group"], []).append(row)
        for group, rows in by_group.items():
            rows.sort(key=lambda row: row["event_position"])
            require([row["event_position"] for row in rows]
                    == list(range(len(EVENTS))), "fixture",
                    f"event positions are not contiguous in {group}")
            require([row["event_ordinal"] for row in rows]
                    == sorted(row["event_ordinal"] for row in rows),
                    "fixture", f"event ordinals are not ordered in {group}")
            for earlier, later in zip(rows, rows[1:]):
                # State only grows or is superseded: every value that survives
                # in the later state is unchanged, and no later state resurrects
                # a superseded value under the same locker.
                for locker, value in later["state_after"].items():
                    if locker in earlier["state_after"]:
                        require(
                            value == earlier["state_after"][locker]
                            or later["role"] == "correction", "fixture",
                            f"state for {locker} changed without a correction")
            require(rows[1]["checkpoint_id_before"]
                    == rows[0]["checkpoint_id_after"], "fixture",
                    "event checkpoint chain is broken")
            require(rows[2]["checkpoint_id_before"]
                    == rows[1]["checkpoint_id_after"], "fixture",
                    "event checkpoint chain is broken")
        require(len(by_group) == (SEALED_GROUPS if sealed else TRAIN_GROUPS),
                "fixture", "group count differs")

    for rows, labels, split in ((fit_cases, fit_labels, "fit"),
                                (final_fit_cases, final_fit_labels, "final_fit"),
                                (dev_cases, dev_labels, "dev"),
                                (final_cases, final_labels, "sealed_final")):
        require({row["case_id"] for row in rows}
                == {row["case_id"] for row in labels}, "fixture",
                f"{split} case/label ids differ")
        for case in rows:
            require(case.get("format") == CASE_FORMAT, "fixture",
                    "evaluation case format differs")
            require(case.get("split_role") == split, "fixture",
                    f"case split_role differs in {split}")
            found = find_forbidden_fields(
                case, EVALUATION_CASE_FORBIDDEN_FIELDS)
            require(not found, "fixture",
                    f"evaluation case carries forbidden fields: {found}")
            require(case.get("inference_inputs") == ["prompt"], "fixture",
                    "evaluation must expose only the source-free prompt")
            require(case.get("source_context_present") is False, "fixture",
                    "evaluation case would retain source context")
            require(case.get("source_kv_loaded") is False, "fixture",
                    "evaluation case would retain source KV")
            require(case.get("resolution_mode") in (
                RESOLUTION_CURRENT, RESOLUTION_HISTORICAL,
                RESOLUTION_EXPLICIT, RESOLUTION_STATELESS), "fixture",
                "unknown resolution mode")
            require(isinstance(case.get("query_epoch"), int), "fixture",
                    "query_epoch is required")
            assert_routing_is_answer_free(case.get("routing_metadata", {}))
            require(case["prompt_sha256"] == _prompt_sha(case["prompt"]),
                    "fixture", "evaluation prompt hash differs")
            if case["resolution_mode"] == RESOLUTION_HISTORICAL:
                require(case["historical_event_position"] == 0, "fixture",
                        "historical cases must name their event explicitly")
                require("event 0" in case["prompt"], "fixture",
                        "historical prompt does not name the historical event")
            else:
                require(case.get("historical_event_position") is None,
                        "fixture",
                        "non-historical case declares a historical event")

    label_by_id = {row["case_id"]: row for row in
                   fit_labels + final_fit_labels + dev_labels + final_labels}
    case_by_id = {row["case_id"]: row for row in
                  fit_cases + final_fit_cases + dev_cases + final_cases}
    for case_id, case in case_by_id.items():
        label = label_by_id[case_id]
        assert_target_absent(case["prompt"], label["target"], check="fixture")
        require(label["source_group"] == case["source_group"], "fixture",
                "case/label source group differs")
        require(label["role"] == case["role"], "fixture",
                "case/label role differs")
        require(label["query_epoch"] == case["query_epoch"], "fixture",
                "case/label query epoch differs")
        require(label["resolution_mode"] == case["resolution_mode"], "fixture",
                "case/label resolution mode differs")
        if case["split_role"] in ("fit", "final_fit"):
            require(label["resolved_event_position"]
                    == case["event_position"], "fixture",
                    "fit label does not bind its teaching event")
        if label["role"] == "abstention":
            require(label["target"] == "UNKNOWN", "fixture",
                    "abstention target must be UNKNOWN")
        if label["role"] == "retention":
            require(label.get("retention_input")
                    and label["target"] == label["retention_input"][::-1]
                    and label["resolution_mode"] == RESOLUTION_STATELESS,
                    "fixture", "retention target is not the declared reversal")
        if label["role"] == "historical_state":
            require(label["resolved_event_position"] == 0, "fixture",
                    "historical label must resolve event 0")
            require(label.get("historical_event_position") == 0, "fixture",
                    "historical label must name event 0")
            require(label["target"] != label_by_id[
                case_id.replace("historical_state",
                                "post_correction_current")]["target"],
                "fixture",
                "historical and current values must differ after correction")
        if label["resolved_event_position"] is not None:
            require(label["event_checkpoint_id"]
                    == _event_checkpoint(case["source_group"],
                                         label["resolved_event_position"]),
                    "fixture", "label event checkpoint differs")
            require(isinstance(label["state_sha256"], str)
                    and len(label["state_sha256"]) == 64, "fixture",
                    "label state digest missing")

    train_groups = {row["source_group"] for row in teaching}
    dev_groups = {row["source_group"] for row in dev_cases}
    final_groups = {row["source_group"] for row in final_cases}
    final_teach_groups = {row["source_group"] for row in final_teaching}
    require(train_groups == dev_groups, "fixture",
            "teaching and development source groups differ")
    require(final_teach_groups == final_groups, "fixture",
            "sealed teaching/evaluation groups differ")
    require(not (train_groups & final_groups), "fixture",
            "sealed groups overlap development groups")
    require(len(final_groups) == SEALED_GROUPS, "fixture",
            "sealed group count differs")
    require(len(train_groups) == TRAIN_GROUPS, "fixture",
            "development group count differs")

    prompts = [row["prompt"] for row in
               teaching + final_teaching + dev_cases + final_cases]
    require(len(prompts) == len(set(prompts)), "fixture",
            "teaching/evaluation prompt is duplicated")
    require(len({row["case_id"] for row in
                 teaching + dev_cases + final_cases}) == len(
                     teaching) + len(dev_cases) + len(final_cases),
            "fixture", "case ids are not unique")
    require(len({row["case_id"] for row in
                 fit_cases + final_fit_cases}) == len(fit_cases)
            + len(final_fit_cases), "fixture", "fit case ids are not unique")
    for case in fit_cases + final_fit_cases:
        matches = [row for row in teaching + final_teaching
                   if row["source_group"] == case["source_group"]
                   and row["event_position"] == case["event_position"]]
        require(len(matches) == 1, "fixture",
                "a fit case does not bind exactly one teaching event")
        require(case["prompt"] == matches[0]["prompt"], "fixture",
                "the declared fit prompt policy is identical_to_teaching")
        require(case["prompt_sha256"] == matches[0]["prompt_sha256"],
                "fixture", "fit and teaching prompt hashes differ")

    if validate_labels:
        require(not any(row.get("answer_index") is not None
                        for row in fit_labels + final_fit_labels + dev_labels
                        + final_labels),
                "fixture", "labels must not use answer indexes")
    return {
        "format": VALIDATION_FORMAT,
        "protocol_id": PROTOCOL_ID,
        "status": "valid",
        "manifest_sha256": manifest["manifest_sha256"],
        "counts": manifest["counts"],
        "source_groups": manifest["source_groups"],
        "native_computation": False,
        "labels_read_for_construction_audit": bool(validate_labels),
        "heldout_opened": False,
        "fit_prompt_policy": "identical_to_teaching_prompt",
        "temporal_semantics": "current_at_query_epoch_and_explicit_historical_events",
    }


def validate_prepared(prepared: str | Path = PREPARED_DIR,
                      config_path: str | Path = CONFIG_PATH) -> dict:
    config = read_json(config_path)
    validate_config(config)
    report = _validate_fixture(Path(prepared), config, validate_labels=True)
    report["prompt_overlap_with_v1"] = _prompt_overlap_with_v1(
        Path(prepared))
    require(report["prompt_overlap_with_v1"] == 0, "lineage",
            "v2 prompts overlap the preserved v1 fixture")
    report["v1_fixture_preserved"] = V1_PREPARED_DIR.is_dir()
    return report


def _prompt_overlap_with_v1(prepared: Path) -> int:
    if not V1_PREPARED_DIR.is_dir():
        return 0
    v1_prompts: set[str] = set()
    for name in ("teaching.jsonl", "dev_cases.jsonl",
                 "sealed_final_cases.jsonl", "fit_cases.jsonl",
                 "sealed_final_fit_cases.jsonl"):
        path = V1_PREPARED_DIR / name
        if path.is_file():
            v1_prompts.update(row["prompt"] for row in read_jsonl(path))
    v2_prompts: set[str] = set()
    for name in FILES:
        v2_prompts.update(row.get("prompt", "") for row in
                          read_jsonl(prepared / name))
    return len(v1_prompts & v2_prompts)


# ---------------------------------------------------------------------------
# Work-unit derivation


def _arm_work_units(arm: Mapping[str, Any], generation: Mapping[str, Any],
                    *, final: bool = False,
                    per_call_positions: int | None = None,
                    teaching_tokens: int | None = None) -> dict[str, int]:
    tokens = int(generation["max_new_tokens_per_call"])
    if per_call_positions is None:
        per_call_positions = (
            tokens * int(generation["evaluation_prompt_tokens_max"])
            + tokens * (tokens - 1) // 2)
    if teaching_tokens is None:
        teaching_tokens = int(generation["teaching_tokens_per_event_max"])
    updates = int(arm["updates"])
    requests = int(arm.get("fit_generations", 0)) \
        + int(arm.get("development_generations", 0)) \
        + int(arm.get("cases_generations", 0))
    teaching_positions = updates * teaching_tokens
    generation_positions = requests * per_call_positions
    # Each teaching event has one atomic recovery write. The final per-event
    # checkpoint is also the retained arm artifact; only reset-P needs another
    # physical write after restoring the exact parent state.
    per_event_writes = updates
    artifact_writes = int(bool(
        updates and arm.get("restore_parent_before_evaluation")))
    retained_artifacts = int(arm.get("checkpoint_artifacts", 0))
    return {
        "updates": updates,
        "training_forwards": updates,
        "backward_calls": updates,
        "optimizer_steps": updates,
        "generation_requests": requests,
        "generation_prefill_forwards": requests,
        "generation_decode_forwards": requests * (tokens - 1),
        "autoregressive_forwards": requests * tokens,
        "generated_tokens": requests * tokens,
        "teaching_processed_positions": teaching_positions,
        "generation_processed_positions": generation_positions,
        "processed_text_positions": teaching_positions + generation_positions,
        "per_event_checkpoint_writes": per_event_writes,
        "artifact_checkpoint_writes": artifact_writes,
        "retained_checkpoint_artifacts": retained_artifacts,
        "checkpoint_writes": per_event_writes + artifact_writes,
        "final_stage": bool(final),
    }


def derive_work_schedule(config: Mapping[str, Any], *,
                         token_lengths: Mapping[str, Any] | None = None) -> dict:
    """Derive every physical-work ceiling from declared execution units.

    The formula set is closed: the caller supplies update counts, generation
    counts and per-call token caps, and this function produces the counters.
    ``token_lengths`` (from the sealed tokenizer step) replaces the declared
    prompt-length ceilings with measured maxima and must not exceed them.
    """
    generation = config["generation"]
    processes = config["budget"]["processes"]
    parity = config["budget"]["parity_probes"]
    storage = config["budget"]["storage"]
    wall = config["budget"]["wall"]
    tokens = int(generation["max_new_tokens_per_call"])
    prompt_tokens = int(generation["evaluation_prompt_tokens_max"])
    teaching_tokens = int(generation["teaching_tokens_per_event_max"])
    parity_tokens = int(generation["parity_probe_tokens_max"])
    per_call_positions = tokens * prompt_tokens + tokens * (tokens - 1) // 2

    measured: dict[str, Any] | None = None
    if token_lengths is not None:
        maxima = token_lengths.get("maxima", {})
        measured = {
            "evaluation_prompt_tokens": int(
                maxima["evaluation_prompt_tokens"]),
            "teaching_sequence_tokens": int(maxima["teaching_sequence_tokens"]),
            "retention_prompt_tokens": int(
                maxima.get("retention_prompt_tokens",
                           maxima["evaluation_prompt_tokens"])),
        }
        for key, ceiling in (
                ("evaluation_prompt_tokens", prompt_tokens),
                ("teaching_sequence_tokens", teaching_tokens),
                ("retention_prompt_tokens", prompt_tokens)):
            require(measured[key] <= ceiling, "token_lengths",
                    f"measured {key}={measured[key]} exceeds the declared "
                    f"ceiling {ceiling}")
        prompt_tokens = measured["evaluation_prompt_tokens"]
        teaching_tokens = measured["teaching_sequence_tokens"]
        parity_tokens = measured["evaluation_prompt_tokens"]
        per_call_positions = (tokens * prompt_tokens
                             + tokens * (tokens - 1) // 2)

    arms = config["controls"]["arms"]
    final_stage = config["controls"]["final_stage"]
    per_arm: dict[str, dict[str, int]] = {}
    work_options = {
        "per_call_positions": per_call_positions,
        "teaching_tokens": teaching_tokens,
    }
    for arm in arms:
        arm_id = arm["arm_id"]
        require(arm_id not in per_arm, "work_units",
                f"duplicate arm {arm_id!r}")
        per_arm[arm_id] = _arm_work_units(arm, generation, **work_options)
    per_arm["treatment_final"] = _arm_work_units(
        final_stage, generation, final=True, **work_options)

    total_arms = sum(per_arm[arm["arm_id"]]["updates"] for arm in arms)
    require(total_arms + per_arm["treatment_final"]["updates"]
            == config["budget"]["declared_total_updates"], "work_units",
            "development plus final update count differs from the declaration")

    def summed(name: str) -> int:
        return sum(block[name] for block in per_arm.values())

    parity_forwards = sum(int(value) for value in parity.values())
    recovery_reloads = int(processes.get("recovery_reloads", 0))
    model_loads = (
        int(processes["native_stage_processes"])
        * int(processes["model_loads_per_native_process"])
        + recovery_reloads * int(processes["model_loads_per_native_process"]))
    tokenizer_loads = (
        (int(processes["native_stage_processes"]) + recovery_reloads)
        * int(processes["tokenizer_loads_per_process"])
        + int(processes["scoring_processes"])
        * int(processes["tokenizer_loads_per_scoring_process"]))
    generation_requests = summed("generation_requests")
    training_forwards = summed("training_forwards")
    generated_tokens = summed("generated_tokens")
    artifact_writes = summed("artifact_checkpoint_writes")
    checkpoint_writes = summed("checkpoint_writes")
    teaching_positions = summed("teaching_processed_positions")
    generation_positions = summed("generation_processed_positions")
    parity_positions = parity_forwards * parity_tokens
    # A recovery reload may recompute the in-flight event's forward and
    # backward, but may never repeat a committed optimizer step.
    restart_training_forwards = recovery_reloads
    restart_backward_calls = recovery_reloads
    restart_optimizer_steps = 0
    restart_positions = recovery_reloads * teaching_tokens
    processed_positions = (teaching_positions + generation_positions
                           + parity_positions + restart_positions)
    retained_artifacts = summed("retained_checkpoint_artifacts")
    live_recovery = int(storage["live_recovery_checkpoints"])
    temp_overlap = int(storage["temp_overlap_blocks"])
    require(retained_artifacts == int(storage["retained_checkpoint_artifacts"]),
            "storage", "retained artifacts differ from the declared arm set")
    storage_bytes = (
        (retained_artifacts + live_recovery + temp_overlap)
        * int(storage["checkpoint_bytes_per_write"])
        + generated_tokens * int(storage["receipt_bytes_per_token"])
        + int(storage["journal_bytes_max"]))
    counters = {
        "model_loads": model_loads,
        "tokenizer_loads": tokenizer_loads,
        "training_forwards": training_forwards,
        "backward_calls": summed("backward_calls"),
        "optimizer_steps": summed("optimizer_steps"),
        "generation_requests": generation_requests,
        "generation_prefill_forwards": summed("generation_prefill_forwards"),
        "generation_decode_forwards": summed("generation_decode_forwards"),
        "autoregressive_forwards": summed("autoregressive_forwards"),
        "generated_tokens": generated_tokens,
        "teaching_processed_positions": teaching_positions,
        "generation_processed_positions": generation_positions,
        "parity_probe_forwards": parity_forwards,
        "parity_processed_positions": parity_positions,
        "restart_processed_text_positions": restart_positions,
        "processed_text_positions": processed_positions,
        "forward_calls": (training_forwards
                          + summed("autoregressive_forwards")
                          + parity_forwards + restart_training_forwards),
        "control_evaluations": 0,
        "per_event_checkpoint_writes": summed("per_event_checkpoint_writes"),
        "artifact_checkpoint_writes": artifact_writes,
        "checkpoint_writes": checkpoint_writes,
        "retained_checkpoint_artifacts": retained_artifacts,
        "live_recovery_checkpoints": live_recovery,
        "storage_temp_overlap_blocks": temp_overlap,
        "restarts": recovery_reloads,
        "recovery_reloads": recovery_reloads,
        "restart_model_loads": recovery_reloads,
        "restart_tokenizer_loads": recovery_reloads,
        "restart_training_forwards": restart_training_forwards,
        "restart_backward_calls": restart_backward_calls,
        "restart_optimizer_steps": restart_optimizer_steps,
        "temporary_storage_bytes": storage_bytes,
    }
    wall_seconds = (
        model_loads * int(wall["model_load_seconds"])
        + tokenizer_loads * int(wall["tokenizer_load_seconds"])
        + training_forwards * int(wall["teaching_event_seconds"])
        + restart_training_forwards * int(wall["teaching_event_seconds"])
        + generation_requests * int(wall["generation_request_seconds"])
        + parity_forwards * int(wall["parity_probe_seconds"])
        + checkpoint_writes * int(wall["checkpoint_write_seconds"])
        + int(processes["scoring_processes"])
        * int(wall["scoring_process_seconds"]))
    wall_seconds = math.ceil(wall_seconds * (1.0 + float(
        wall["overhead_fraction"])))
    counters["wall_seconds"] = wall_seconds
    per_arm_limits: dict[str, dict[str, int]] = {}
    for arm_id, block in per_arm.items():
        per_arm_limits[arm_id] = {
            f"max_{name}": block[name] for name in (
                "optimizer_steps", "training_forwards", "backward_calls",
                "generation_requests", "generation_prefill_forwards",
                "generation_decode_forwards", "autoregressive_forwards",
                "generated_tokens", "processed_text_positions",
                "per_event_checkpoint_writes", "artifact_checkpoint_writes",
                "checkpoint_writes")
        }
        per_arm_limits[arm_id]["arm_id"] = arm_id
    return {
        "format": "replica.dc_p1_native_response_v2.work_schedule.v1",
        "protocol_id": PROTOCOL_ID,
        "unit_formulas": {
            "per_generation_call_positions": (
                "tokens*evaluation_prompt_tokens_max + tokens*(tokens-1)/2 "
                "(prefill over the prompt, then one full no-cache forward per "
                "emitted token, so processed positions grow by one token each "
                "decode step)"),
            "prefill_forwards": "1 per generation request",
            "decode_forwards": "1 per emitted token after the first",
            "autoregressive_forwards": "tokens per generation request",
            "generated_tokens": "tokens per generation request",
            "teaching_processed_positions": (
                "teaching_events * teaching_tokens_per_event_max"),
            "training_forwards": "1 per teaching event",
            "backward_calls": "1 per teaching event",
            "optimizer_steps": "1 per teaching event (one step per event)",
            "parity_processed_positions": (
                "parity_probes * parity_probe_tokens_max"),
            "checkpoint_writes": (
                "per_teaching_event_atomic_writes + reset_parent_artifact_writes "
                "(120 + 1); the last per-event checkpoint is retained as the "
                "arm artifact without a duplicate final write"),
            "temporary_storage_bytes": (
                "(retained_checkpoint_artifacts + live_recovery_checkpoints + "
                "temp_overlap_blocks)*checkpoint_bytes_per_write + "
                "generated_tokens*receipt_bytes_per_token + journal_bytes_max"),
            "wall_seconds": (
                "ceil((model_loads*model_load_seconds + "
                "tokenizer_loads*tokenizer_load_seconds + "
                "training_forwards*teaching_event_seconds + "
                "restart_training_forwards*teaching_event_seconds + "
                "generation_requests*generation_request_seconds + "
                "parity_probes*parity_probe_seconds + "
                "checkpoint_writes*checkpoint_write_seconds + "
                "scoring_processes*scoring_process_seconds) * "
                "(1+overhead_fraction))"),
        },
        "inputs": {
            "max_new_tokens_per_call": tokens,
            "evaluation_prompt_tokens": prompt_tokens,
            "teaching_tokens_per_event": teaching_tokens,
            "parity_probe_tokens": parity_tokens,
            "per_call_positions": per_call_positions,
            "token_lengths_source": ("measured_sealed_tokenizer"
                                     if measured else
                                     "declared_ceiling"),
        },
        "measured_token_lengths": measured,
        "counters": counters,
        "per_arm": per_arm,
        "per_arm_limits": per_arm_limits,
        "development_updates": total_arms,
        "final_updates": per_arm["treatment_final"]["updates"],
        "declared_total_updates": (total_arms
                                   + per_arm["treatment_final"]["updates"]),
        "development_generation_requests": sum(
            per_arm[arm["arm_id"]]["generation_requests"] for arm in arms),
        "final_generation_requests": per_arm["treatment_final"][
            "generation_requests"],
    }


def validate_work_units(config: Mapping[str, Any], *,
                        token_lengths: Mapping[str, Any] | None = None) -> dict:
    """Require the declared ceilings to equal the derived schedule.

    With ``token_lengths`` the measured schedule must fit inside every declared
    ceiling, and it is returned instead of the ceiling schedule: measured
    lengths may only reduce positions, never raise a cap.
    """
    schedule = derive_work_schedule(config)
    budget = config["budget"]
    physical = budget.get("physical_limits")
    require(isinstance(physical, Mapping), "work_units",
            "budget.physical_limits is required")
    for name, value in schedule["counters"].items():
        key = f"max_{name}"
        require(key in physical, "work_units",
                f"physical_limits is missing {key}")
        require(physical[key] == value, "work_units",
                f"{key}={physical[key]} differs from the derived {value}")
    declared_arm_limits = budget.get("per_arm_limits")
    require(isinstance(declared_arm_limits, Mapping), "work_units",
            "budget.per_arm_limits is required")
    require(set(declared_arm_limits) == set(schedule["per_arm_limits"]),
            "work_units",
            "per-arm limit keys differ from the derived arm set")
    for arm_id, expected in schedule["per_arm_limits"].items():
        declared = declared_arm_limits[arm_id]
        for key, value in expected.items():
            if key == "arm_id":
                require(declared.get(key) == value, "work_units",
                        f"per_arm_limits[{arm_id}].arm_id differs")
                continue
            require(declared.get(key) == value, "work_units",
                    f"per_arm_limits[{arm_id}][{key}]={declared.get(key)} "
                    f"differs from the derived {value}")
    # The ceilings must accept exactly the derived run and refuse one more unit.
    check_work_budget_v2(schedule["counters"], budget)
    for arm_id in schedule["per_arm"]:
        block = schedule["per_arm"][arm_id]
        observed = {name: block[name] for name in (
            "optimizer_steps", "training_forwards", "backward_calls",
            "generation_requests", "processed_text_positions",
            "checkpoint_writes", "generated_tokens",
            "generation_prefill_forwards", "generation_decode_forwards",
            "autoregressive_forwards", "per_event_checkpoint_writes",
            "artifact_checkpoint_writes")}
        check_work_budget_v2(observed, budget, arm=arm_id)
    if token_lengths is None:
        return schedule
    measured = derive_work_schedule(config, token_lengths=token_lengths)
    check_work_budget_v2(measured["counters"], budget)
    for name, value in measured["counters"].items():
        require(value <= schedule["counters"][name], "work_units",
                f"measured {name}={value} exceeds the declared ceiling "
                f"{schedule['counters'][name]}")
    for arm_id, block in measured["per_arm_limits"].items():
        for key, value in block.items():
            if key == "arm_id":
                continue
            require(value <= schedule["per_arm_limits"][arm_id][key],
                    "work_units",
                    f"measured {arm_id}.{key}={value} exceeds the declared "
                    f"{schedule['per_arm_limits'][arm_id][key]}")
    return measured


def validate_token_lengths(config: Mapping[str, Any],
                           artifact: Mapping[str, Any],
                           *, prepared: str | Path = PREPARED_DIR) -> dict:
    """Validate the sealed tokenizer-derived prompt-length artifact."""
    require(artifact.get("format") == TOKEN_LENGTHS_FORMAT, "token_lengths",
            "token length artifact format differs")
    require(artifact.get("protocol_id") == PROTOCOL_ID, "token_lengths",
            "token length artifact protocol differs")
    require(artifact.get("tokenizer_binding_sha256")
            == tokenizer_binding_sha(config["tokenizer_binding"]),
            "token_lengths",
            "token length artifact was produced by another tokenizer binding")
    seal = verify_sealed_record(artifact, check="token_lengths")
    maxima = seal.get("maxima")
    require(isinstance(maxima, Mapping), "token_lengths",
            "token length maxima are required")
    for key in ("evaluation_prompt_tokens", "teaching_sequence_tokens"):
        require(isinstance(maxima.get(key), int) and maxima[key] > 0,
                "token_lengths", f"maxima.{key} must be a positive integer")
    prompts = seal.get("prompts")
    require(isinstance(prompts, Mapping), "token_lengths",
            "per-prompt token counts are required")
    prepared = Path(prepared)
    expected: set[str] = set()
    for name in FILES:
        for row in read_jsonl(prepared / name):
            if "prompt_sha256" in row:
                expected.add(row["prompt_sha256"])
    require(expected <= set(prompts), "token_lengths",
            "the token length artifact does not cover every sealed prompt")
    require(all(isinstance(value, int) and value > 0
                for value in prompts.values()), "token_lengths",
            "prompt token counts must be positive integers")
    require(maxima["evaluation_prompt_tokens"]
            <= config["generation"]["evaluation_prompt_tokens_max"],
            "token_lengths",
            "measured evaluation_prompt_tokens exceeds the declared ceiling")
    require(maxima["teaching_sequence_tokens"]
            <= config["generation"]["teaching_tokens_per_event_max"],
            "token_lengths",
            "measured teaching_sequence_tokens exceeds the declared ceiling")
    return seal


def validate_prepared_token_lengths(
        config_path: str | Path = CONFIG_PATH,
        prepared: str | Path = PREPARED_DIR) -> dict:
    config = read_json(config_path)
    validate_config(config)
    path = Path(prepared) / TOKEN_LENGTHS_PATH.name
    if not path.is_file():
        return {
            "status": "pending_tokenizer_step",
            "path": str(path),
            "command": ("python3 scripts/tokenize_dc_p1_v2_prompts_20261027.py "
                        f"--prepared {prepared}"),
            "declared_sha256": config["prepared"]["token_lengths"][
                "sha256"],
        }
    artifact = read_json(path)
    seal = validate_token_lengths(config, artifact, prepared=prepared)
    declared = config["prepared"]["token_lengths"]
    require(declared.get("sha256") == sha256_file(path), "token_lengths",
            "config does not bind the exact token length artifact bytes")
    schedule = validate_work_units(config, token_lengths=seal)
    return {
        "status": "sealed",
        "path": str(path),
        "artifact_sha256": sha256_file(path),
        "receipt_sha256": seal["receipt_sha256"],
        "measured_counters": schedule["counters"],
    }


# ---------------------------------------------------------------------------
# Protocol validation


def validate_config(config: Mapping[str, Any]) -> dict:
    require(config.get("format") == CONFIG_FORMAT, "config",
            "config format differs")
    require(config.get("protocol_id") == PROTOCOL_ID, "config",
            "protocol_id differs")
    require(config.get("study") == "DC-P1", "config", "study must be DC-P1")
    require(config.get("version") == 2, "config", "version must be 2")
    classification = config.get("classification", {})
    for key, expected in (("preparation_only", True),
                          ("native_execution_allowed", False),
                          ("native", False),
                          ("model_execution", False),
                          ("tokenizer_load", False),
                          ("generation", False),
                          ("optimizer_updates", 0),
                          ("gradients", 0),
                          ("checkpoints_written", 0),
                          ("downloads", 0),
                          ("heldout_opened", False),
                          ("promotion", False),
                          ("persistent_learning_claim", False)):
        require(classification.get(key) == expected, "config",
                f"classification.{key}={classification.get(key)!r} must be "
                f"{expected!r}")
    validate_guards(config)
    validate_baseline_model(config)
    validate_tokenizer_binding(config.get("tokenizer_binding"))

    lineage = config.get("lineage", {})
    require(lineage.get("supersedes") == "dc_p1_native_response_v1", "lineage",
            "v2 must supersede exactly dc_p1_native_response_v1")
    require(lineage.get("v1_preserved_unchanged") is True, "lineage",
            "the v1 package must stay unchanged")
    require(lineage.get("v1_evidence_retained") is True, "lineage",
            "v1 evidence must be retained")
    require(lineage.get("v1_config") == "configs/dc_p1_native_response_protocol_v1.json",
            "lineage", "v1 config path differs")
    require(lineage.get("v1_prepared_root")
            == "data/dc_p1_native_response_20261027", "lineage",
            "v1 prepared root differs")
    amendments = config.get("amendments")
    require(isinstance(amendments, list) and len(amendments) >= 6, "amendment",
            "v2 must declare at least the six versioned amendments")
    kinds = set()
    for amendment in amendments:
        require(isinstance(amendment, Mapping), "amendment",
                "amendment must be an object")
        for key in ("amendment_id", "kind", "reason", "declared_work_added",
                    "old_limit_preserved"):
            require(key in amendment, "amendment",
                    f"amendment is missing {key}")
        require(amendment["kind"] in (
            "added_matched_control", "corrected_unit", "added_stage_admission",
            "semantic_correction", "declared_claim_narrowing",
            "final_stage_scope"), "amendment",
            f"unknown amendment kind {amendment['kind']!r}")
        require(amendment["old_limit_preserved"] is True, "amendment",
                "an amendment must preserve the old limit")
        require(amendment.get("silent") is False, "amendment",
                "an amendment may not be silent")
        kinds.add(amendment["kind"])
    require(len(kinds) == len(amendments), "amendment",
            "amendment kinds must be distinct")
    require("added_matched_control" in kinds, "amendment",
            "the matched trained control amendment is required")
    require("corrected_unit" in kinds, "amendment",
            "the work-unit correction amendment is required")

    representation = config.get("representation", {})
    for key in ("synthetic_feature_injection", "answer_index_bypass",
                "source_context_at_scoring", "source_kv_at_scoring"):
        require(representation.get(key) is False, "representation",
                f"representation.{key} must be false")
    require(representation.get("selected_value_carrier") == "none",
            "representation", "a selected-value carrier would bypass learning")

    plastic = config.get("plastic_spec", {})
    require(plastic.get("frozen_backbone") is True, "plastic_spec",
            "the backbone must be frozen")
    require(plastic.get("delta_site") == "final_hidden_before_lm_head",
            "plastic_spec", "the selected-expert delta site differs")
    require(plastic.get("trainable_groups")
            == ["expert_0.gate", "expert_0.up", "expert_0.down"],
            "plastic_spec", "trainable groups differ")
    bank = plastic.get("bank", {})
    for key, expected in (("experts", 16), ("width", 256),
                          ("hidden_width", 1024),
                          ("optimizer", "NativeFusedAdamW"),
                          ("dtype", "FP32"), ("lr", 0.0001),
                          ("p_init", "zero_read_down")):
        require(bank.get(key) == expected, "plastic_spec",
                f"plastic_spec.bank.{key}={bank.get(key)!r} differs from "
                f"{expected!r}")
    features = plastic.get("features", {})
    require(features.get("sites") == [13, 27], "plastic_spec",
            "feature sites differ")
    require(features.get("feature_version")
            == "sites_masked_mean_projection.v1", "plastic_spec",
            "feature version differs")
    require(features.get("detached") is True
            and features.get("no_grad_feature_pass") is True, "plastic_spec",
            "site features must be detached")
    read_path = plastic.get("read_path", {})
    require(read_path.get("gain") == 0.1, "plastic_spec",
            "read gain differs")
    require(read_path.get("trainable") is False
            and read_path.get("initial_null") is True, "plastic_spec",
            "the read path must stay fixed and initially null")

    parity = config.get("baseline_parity", {})
    for key in ("no_extension_is_default",
                "disabled_extension_preserves_native_logits",
                "disabled_extension_preserves_native_greedy_tokens",
                "parity_required_before_teaching",
                "parity_required_after_reset_P"):
        require(parity.get(key) is True, "baseline_parity",
                f"baseline_parity.{key} must be true")
    require(parity.get("logit_rtol") == 0.0 and parity.get("logit_atol") == 0.0,
            "baseline_parity", "exact parity is required")

    objective = config.get("objective", {})
    encoding = config.get("generation", {}).get("prompt_encoding")
    require(isinstance(encoding, Mapping), "generation",
            "generation.prompt_encoding is required")
    require(encoding.get("chat_template") is True, "generation",
            "prompts must use the native chat template")
    require(encoding.get("add_generation_prompt") is True, "generation",
            "prompts must use a generation prompt")
    require(encoding.get("enable_thinking") is False, "generation",
            "thinking must be disabled")
    require(encoding.get("add_special_tokens") is False, "generation",
            "chat-template special tokens must be disabled")
    require(encoding.get("max_length") is None, "generation",
            "prompts may not be length-capped")

    require(objective.get("api") == OBJECTIVE_API, "objective",
            "objective API binding differs from the implemented module")
    require(objective.get("terminal_eos_id") == TERMINAL_EOS_ID, "objective",
            "terminal EOS id differs")
    require(objective.get("answer_span_mask")
            == "answer_content_plus_terminal_eos", "objective",
            "answer-span mask definition differs")
    require(objective.get("full_response_mask")
            == "all_supervised_response_tokens", "objective",
            "full-response mask definition differs")
    require(objective.get("masks_coincide_by_construction") is True, "objective",
            "the teaching layout makes the masks coincide")
    require(objective.get("declared_claim") == CLAIM_GLOBAL_RESCALE,
            "objective",
            "the coincident-mask treatment must be declared a global CE rescale")
    require(objective.get("span_specific_claim_allowed") is False, "objective",
            "a span-specific claim is not allowed for coincident masks")
    require(objective.get("audit_required") is True, "objective",
            "the mask/objective audit is mandatory")
    require(objective.get("audit_disagreement_action")
            == "refuse_and_declare", "objective",
            "a disagreeing audit must refuse admission")
    require(objective.get("answer_span_ce_weight_treatment") == 0.5,
            "objective", "treatment weight differs")
    require(objective.get("answer_span_ce_weight_matched") == 0.0, "objective",
            "matched weight differs")
    require(objective.get("narrowed_claim", {}).get("id")
            == "DC-P1-H1-v2-global-ce-rescale", "objective",
            "the narrowed claim identity differs")
    require(CLAIM_SPAN_SPECIFIC not in json.dumps(
        objective.get("narrowed_claim", {})), "objective",
        "the narrowed claim may not re-introduce a span-specific statement")

    recipe = config.get("recipe", {})
    baseline = recipe.get("baseline", {})
    treatment = recipe.get("treatment", {})
    required_baseline = {
        "objective": "mean_response_nll_plus_weighted_answer_span_ce",
        "steps_per_event": 1,
        "optimizer": "NativeFusedAdamW",
        "learning_rate": 0.0001,
        "gradient_norm_max": 1.0,
        "response_only_labels": True,
        "label_shift": 1,
        "trainable_groups": ["expert_0.gate", "expert_0.up", "expert_0.down"],
        "source_context_in_teaching_prompt": True,
        "source_kv_reused_at_evaluation": False,
        "answer_span_ce_weight": 0.0,
    }
    for key, expected in required_baseline.items():
        require(baseline.get(key) == expected, "recipe",
                f"baseline {key}={baseline.get(key)!r} differs from {expected!r}")
    require(treatment.get("steps_per_event") == 1, "recipe",
            "budget02 revival is forbidden; exactly one step per event")
    require(treatment.get("budget02_revival") is False, "recipe",
            "budget02 revival is forbidden")
    require(treatment.get("answer_span_ce_weight") == 0.5, "recipe",
            "treatment weight differs")
    require(treatment.get("unchanged_vs_matched")
            == sorted(required_baseline), "recipe",
            "treatment must enumerate exactly the fields it shares with the "
            "matched trained control")
    for key, expected in required_baseline.items():
        if key == "answer_span_ce_weight":
            continue
        require(treatment.get(key) == expected, "recipe",
                f"treatment silently changed {key}")
    require(treatment.get("primary_change_count") == 1, "recipe",
            "exactly one primary recipe change is required")
    changes = treatment.get("changes")
    require(isinstance(changes, list) and len(changes) == 1, "recipe",
            "exactly one primary recipe change is required")
    require(changes[0].get("field") == "loss.answer_span_ce_weight", "recipe",
            "primary change field differs")
    require(changes[0].get("from") == 0.0 and changes[0].get("to") == 0.5,
            "recipe", "primary change values differ")

    prepared = config.get("prepared", {})
    require(prepared.get("root") == str(PREPARED_DIR.relative_to(ROOT)),
            "prepared", "prepared root differs")
    require(prepared.get("groups") == TRAIN_GROUPS, "prepared",
            "development group count differs")
    require(prepared.get("sealed_groups") == SEALED_GROUPS, "prepared",
            "sealed group count differs")
    require(set(prepared.get("files", {})) == set(FILES), "prepared",
            "prepared file set differs")
    counts = prepared.get("counts", {})
    expected_counts = {
        "teaching": TRAIN_GROUPS * len(EVENTS),
        "fit_cases": TRAIN_GROUPS * len(EVENTS),
        "dev_cases": TRAIN_GROUPS * len(DEV_ROLES),
        "sealed_final_teaching": SEALED_GROUPS * len(EVENTS),
        "sealed_final_fit_cases": SEALED_GROUPS * len(EVENTS),
        "sealed_final_cases": SEALED_GROUPS * len(DEV_ROLES),
        "labels_fit": TRAIN_GROUPS * len(EVENTS),
        "labels_dev": TRAIN_GROUPS * len(DEV_ROLES),
        "sealed_final_labels_fit": SEALED_GROUPS * len(EVENTS),
        "labels_final": SEALED_GROUPS * len(DEV_ROLES),
    }
    for key, value in expected_counts.items():
        require(counts.get(key) == value, "prepared",
                f"prepared count {key}={counts.get(key)} differs from {value}")
    require(counts.get("optimizer_steps")
            == config["budget"]["declared_total_updates"], "prepared",
            "prepared optimizer steps differ from the declared total")
    require(counts.get("development_teaching_updates")
            == config["budget"]["declared_development_updates"], "prepared",
            "development update count differs")
    require(counts.get("final_teaching_updates")
            == config["budget"]["declared_final_updates"], "prepared",
            "final update count differs")
    require(prepared.get("fit_prompt_policy")
            == "identical_to_teaching_prompt", "prepared",
            "fit prompt policy differs")
    token_lengths = prepared.get("token_lengths")
    require(isinstance(token_lengths, Mapping), "prepared",
            "prepared.token_lengths binding is required")
    require(token_lengths.get("path")
            == "token_lengths.v2.json", "prepared",
            "token length artifact path differs")
    require(token_lengths.get("status") in ("pending_tokenizer_step", "sealed"),
            "prepared", "token length status differs")
    if token_lengths.get("status") == "sealed":
        require(isinstance(token_lengths.get("sha256"), str)
                and len(token_lengths["sha256"]) == 64, "prepared",
                "a sealed token length artifact needs a digest")
    else:
        require(token_lengths.get("sha256") is None, "prepared",
                "a pending tokenizer step cannot bind a digest")
        require(isinstance(token_lengths.get("command"), str)
                and "tokenize_dc_p1_v2_prompts" in token_lengths["command"],
                "prepared", "the pending step must name its command")

    controls = config.get("controls", {})
    arms = controls.get("arms", [])
    require([row.get("arm_id") for row in arms] == list(ARMS), "controls",
            "arm order/identity differs")
    require([row.get("answer_span_ce_weight") for row in arms]
            == [0.5, 0.0, 0.5, 0.5], "controls",
            "arm answer-span weights differ")
    require([row.get("updates") for row in arms] == [36, 36, 0, 36],
            "controls", "arm update counts differ")
    require([row.get("checkpoint_artifacts") for row in arms] == [1, 1, 0, 1],
            "controls", "arm checkpoint artifact counts differ")
    for arm in arms:
        require(arm.get("initialization") == "parent_checkpoint", "controls",
                "every arm must start from the same parent checkpoint")
        for key in ("source_context_at_evaluation", "source_kv_at_evaluation"):
            require(arm.get(key) is False, "controls",
                    f"{arm['arm_id']}.{key} must be false")
        require(arm.get("fit_generations") == TRAIN_GROUPS * len(EVENTS),
                "controls", "every arm needs the same fit evidence")
        require(arm.get("development_generations")
                == TRAIN_GROUPS * len(DEV_ROLES), "controls",
                "every arm needs the same development evidence")
        require(isinstance(arm.get("checkpoint_id"), str)
                and arm["checkpoint_id"].startswith("dcp1v2:arm:"), "controls",
                f"arm {arm['arm_id']} checkpoint identity differs")
    require(arms[1].get("role") == "matched_trained_control", "controls",
            "the second arm must be the matched trained control")
    require(arms[2].get("updates") == 0, "controls",
            "the no-update control must have zero updates")
    require(arms[3].get("restore_parent_before_evaluation") is True,
            "controls", "the reset-P arm must restore the parent")
    require(controls.get("identical_data_and_budget") is True, "controls",
            "all arms must share the teaching rows and the clock")
    require(controls.get("shared_teaching_exposure") is True, "controls",
            "arms must share teaching exposure")
    require(controls.get("matched_control_explicit") is True, "controls",
            "the matched control must be declared as an added arm")
    require(controls.get("budget_not_silently_raised") is True, "controls",
            "the added arm's work must be declared, not hidden")
    final_stage = controls.get("final_stage", {})
    require(final_stage.get("arms") == list(FINAL_ARMS), "controls",
            "the final stage must be treatment-only")
    require(final_stage.get("updates") == 12, "controls",
            "final teaching updates differ")
    require(final_stage.get("fit_cases") == SEALED_GROUPS * len(EVENTS),
            "controls", "final fit case count differs")
    require(final_stage.get("cases") == SEALED_GROUPS * len(DEV_ROLES),
            "controls", "final case count differs")
    require(final_stage.get("role") == "confirmation_only", "controls",
            "the final stage is confirmation only")
    require(final_stage.get("comparison") is None, "controls",
            "the final stage must not declare a 0.5-vs-0.0 comparison")
    require(final_stage.get("checkpoint_artifacts") == 1, "controls",
            "the final checkpoint artifact count differs")

    staging = config.get("staging", {})
    require(staging.get("order") == list(STAGE_ORDER), "staging",
            "stage order differs")
    require(staging.get("receipt_format") == STAGE_RECEIPT_FORMAT, "staging",
            "stage receipt format differs")
    require(staging.get("score_report_format") == SCORE_REPORT_FORMAT,
            "staging", "score report format differs")
    require(staging.get("api") == STAGED_API, "staging",
            "staged evaluation API binding differs")
    require(staging.get("fit_admission", {}).get("required_before")
            == "development_generation_and_labels", "staging",
            "fit admission must precede development generation and labels")
    require(staging.get("fit_admission", {}).get("requires")
            == "evidence_for_every_arm_plus_treatment_fit_gates", "staging",
            "fit admission requirements differ")
    require(staging.get("fit_admission", {}).get("on_failure")
            == "stop_with_evidence_and_leave_development_closed", "staging",
            "a failed fit gate must stop the study")
    require(staging.get("development_admission", {}).get("required_before")
            == "final_teaching_generation_and_labels", "staging",
            "development admission must precede final work")
    require(staging.get("development_admission", {}).get("requires")
            == "primary_comparison_vs_matched_0.0_and_reset_P_and_no_update",
            "staging", "development admission requirements differ")
    require(staging.get("final_admission", {}).get("one_shot") is True,
            "staging", "the final stage must be one-shot")
    require(staging.get("final_admission", {}).get("fresh_attempt_id") is True,
            "staging", "the final stage needs a fresh attempt id")
    require(staging.get("final_admission", {}).get("arms") == ["treatment"],
            "staging", "final admission is treatment-only")
    require(staging.get("label_access") == "vault_permit_required", "staging",
            "labels must be vault-gated")
    require(staging.get("tokenizer_backed_receipts") is True, "staging",
            "stage receipts must be tokenizer-backed")

    anti = config.get("anti_shortcut", {})
    for key in ("answer_index_bypass_allowed", "answer_bearing_routing_metadata",
                "target_in_prompt_allowed", "target_conditioned_thresholds",
                "globally_unique_prompt_ids_allowed_as_features",
                "candidate_answer_selection_allowed", "empty_generation_credit",
                "case_id_in_prompt", "state_digest_in_case_prompt"):
        require(anti.get(key) is False, "anti_shortcut",
                f"anti_shortcut.{key} must be false")
    for key in ("source_context_removed_at_scoring",
                "source_kv_removed_at_scoring", "per_case_scoring",
                "controls_use_identical_scorer", "corrupted_controls_must_fail",
                "historical_query_must_name_event",
                "ordinary_query_resolves_at_query_epoch"):
        require(anti.get(key) is True, "anti_shortcut",
                f"anti_shortcut.{key} must be true")

    evaluation = config.get("evaluation", {})
    require(evaluation.get("source_free") is True, "evaluation",
            "evaluation must be source-free")
    require(evaluation.get("generated_answer_scoring") is True, "evaluation",
            "generated-answer scoring is required")
    require(evaluation.get("tokenizer_and_decode_binding_required") is True,
            "evaluation", "tokenizer/decode binding is required")
    require(evaluation.get("staged_fit_development_final") is True,
            "evaluation", "staged evaluation is required")
    require(evaluation.get("sealed_final_one_shot") is True, "evaluation",
            "the sealed final stage must be one-shot")
    require(evaluation.get("sealed_final_requires_separate_teaching") is True,
            "evaluation", "the sealed final stage must be taught separately")
    require(evaluation.get("fit_roles") == list(FIT_ROLES), "evaluation",
            "fit roles differ")
    require(evaluation.get("development_roles") == list(DEV_ROLES),
            "evaluation", "development roles differ")
    require(evaluation.get("primary_roles") == list(PRIMARY_ROLES),
            "evaluation", "primary roles differ")
    require(evaluation.get("control_arms") == ["matched_trained", "no_update",
                                               "reset_P"], "evaluation",
            "control arm list differs")
    require(evaluation.get("paired_by_source_group") is True, "evaluation",
            "paired comparison is required")
    require(evaluation.get("target_conditioned_thresholds") is False,
            "evaluation", "thresholds may not depend on targets")
    gates = evaluation.get("gates", {})
    for key in ("fit_exact_min", "fit_nonempty_min",
                "post_correction_accuracy_min", "new_instance_accuracy_min",
                "historical_accuracy_min", "abstention_accuracy_min",
                "abstention_false_recall_max", "retention_accuracy_min",
                "retention_decline_max", "control_primary_accuracy_max",
                "post_correction_min_gain", "new_instance_min_gain",
                "paired_ci_low_min"):
        require(key in gates, "evaluation", f"missing gate {key}")

    budget = config.get("budget", {})
    require(budget.get("declared_development_updates") == 108, "budget",
            "development updates must be 108 (36+36+0+36)")
    require(budget.get("declared_final_updates") == 12, "budget",
            "final updates must be 12")
    require(budget.get("declared_total_updates") == 120, "budget",
            "the total declared optimizer step ceiling must be 120")
    historical = budget.get("historical_v1", {})
    require(historical.get("preserved") is True, "budget",
            "the v1 limits must be preserved")
    require(historical.get("forward_calls") == 368, "budget",
            "the historical 368 forward-call accounting must be retained")
    require(historical.get("accounting_status")
            == "incorrect_request_equals_forward_accounting", "budget",
            "368 must be declared incorrect historical accounting")
    require(historical.get("reused_as_a_ceiling") is False, "budget",
            "the historical 368 may not be reused as a ceiling")
    require(historical.get("optimizer_steps") == 48, "budget",
            "the v1 optimizer step declaration must be preserved")
    require(budget.get("amendment_work_declaration", {}).get(
        "matched_control_updates") == 36, "budget",
            "the matched control's added updates must be declared")
    require(budget.get("amendment_work_declaration", {}).get(
        "matched_control_generation_requests") == 96, "budget",
            "the matched control's added generations must be declared")
    schedule = validate_work_units(config)
    require(isinstance(budget.get("processes"), Mapping), "budget",
            "budget.processes is required")
    require(budget["processes"].get("native_stage_processes") == len(STAGE_ORDER),
            "budget", "one native stage process per stage is required")
    require(budget["processes"].get("scoring_processes") == len(STAGE_ORDER),
            "budget", "one independent scoring process per stage is required")
    require(budget["processes"].get("exclusive_worker") is True, "budget",
            "the exclusive worker policy is required")
    require(budget["processes"].get("model_loads_per_native_process") == 1,
            "budget", "one model load per native process is required")
    require(budget["processes"].get("tokenizer_loads_per_process") == 1,
            "budget", "one tokenizer load per process is required")
    require(budget["processes"].get("tokenizer_loads_per_scoring_process")
            == 1, "budget", "one tokenizer load per scoring process is "
            "required")
    require(budget["processes"].get("recovery_reloads") == 1, "budget",
            "exactly one fresh-process recovery reload is permitted")
    per_process = budget.get("per_process_limits")
    require(isinstance(per_process, Mapping), "budget",
            "budget.per_process_limits is required")
    for key, expected in (("max_model_loads", 1), ("max_tokenizer_loads", 1),
                          ("max_workers", 1), ("max_retries", 0),
                          ("max_downloads", 0),
                          ("max_optimizer_steps_repeated", 0)):
        require(per_process.get(key) == expected, "budget",
                f"per_process_limits.{key} must be {expected}")
    recovery = budget.get("recovery")
    require(isinstance(recovery, Mapping), "budget",
            "budget.recovery is required")
    require(recovery.get("granularity") == "per_teaching_event", "budget",
            "fresh-process recovery must be per teaching event")
    require(recovery.get("may_repeat_completed_optimizer_work") is False,
            "budget", "recovery may not repeat committed optimizer work")
    require(recovery.get("journal") == "append_only", "budget",
            "the recovery journal must be append-only")
    require(recovery.get("atomic_write_per_teaching_event") is True, "budget",
            "each teaching event needs one atomic durable write")
    require(recovery.get("reloads_max") == 1, "budget",
            "exactly one recovery reload is permitted")
    storage = budget.get("storage")
    require(isinstance(storage, Mapping), "budget",
            "budget.storage is required")
    require(storage.get("retained_checkpoint_artifacts") == 4, "budget",
            "exactly four arm checkpoint artifacts are retained")
    require(storage.get("live_recovery_checkpoints") == 1, "budget",
            "at most one live recovery checkpoint is retained")
    require(storage.get("temp_overlap_blocks") == 1, "budget",
            "one atomic-write overlap block is declared")
    require(budget.get("checkpoint_write_breakdown") == {
        "per_event_atomic_writes": schedule["counters"][
            "per_event_checkpoint_writes"],
        "reset_parent_artifact_writes": schedule["counters"][
            "artifact_checkpoint_writes"],
        "retained_arm_artifacts": schedule["counters"][
            "retained_checkpoint_artifacts"],
        "attempt_total": schedule["counters"]["checkpoint_writes"]}, "budget",
            "every physical checkpoint write must be declared")
    require(budget.get("historical_v1", {}).get(
        "physical_limits", {}).get("max_checkpoint_writes") == 4, "budget",
        "the v1 checkpoint write cap must stay preserved, not reused")
    require(budget.get("recovery_work_declaration", {}).get(
        "restart_optimizer_steps") == 0, "budget",
        "recovery may not add an optimizer step")
    require(budget.get("recovery_work_declaration", {}).get(
        "restart_training_forwards") == 1, "budget",
        "the in-flight event recomputation must be declared")
    launch_output = config.get("launch", {}).get("output_mode")
    require(isinstance(launch_output, Mapping), "launch",
            "launch.output_mode is required")
    require(launch_output.get("fit") == "fresh_only", "launch",
            "the initial fit process requires a fresh output root")
    require(launch_output.get("development") == "continue_same_attempt",
            "launch", "development continues the same attempt output")
    require(launch_output.get("final") == "continue_same_attempt", "launch",
            "final continues the same attempt output")
    require(launch_output.get("resume") == "continue_same_attempt", "launch",
            "resume continues the same attempt output")
    require(launch_output.get("continue_requires_attempt_journal") is True,
            "launch", "continuing an output root requires its attempt journal")

    launch = config.get("launch", {})
    require(launch.get("review_only") is True, "launch",
            "the launch declaration must be review-only")
    require(launch.get("native_execution") is False, "launch",
            "native execution must be false in this package")
    require(launch.get("runner_path")
            == "scripts/run_dc_p1_native_response_v2_20261027.py", "launch",
            "runner path differs from the declared native runner")
    require(launch.get("requires_authorization_receipt") is True, "launch",
            "a separate authorization receipt is required")
    require(launch.get("authorization_receipt_format")
            == AUTHORIZATION_FORMAT_V2, "launch",
            "the authorization receipt format differs")
    require(launch.get("authorization_receipt_present_at_declaration")
            is False, "launch",
            "no authorization receipt exists at declaration time")
    require(launch.get("fresh_output_path") is True, "launch",
            "a fresh output path is required")
    require(launch.get("on_missing_receipt") == "refuse_nonzero", "launch",
            "a missing authorization must refuse nonzero")
    require(launch.get("on_missing_runner") == "refuse_nonzero", "launch",
            "a missing runner must refuse nonzero")
    schema = launch.get("authorization_fields")
    require(isinstance(schema, list), "launch",
            "the authorization schema field list is required")
    for field in ("format", "protocol_id", "config_sha256",
                  "prepared_manifest_sha256", "protocol_commit",
                  "runner_path", "runner_sha256", "authorized_by",
                  "native_execution_allowed", "single_attempt",
                  "max_optimizer_steps", "max_generation_requests",
                  "receipt_sha256"):
        require(field in schema, "launch",
                f"authorization schema is missing {field}")
    require(config["launch"].get("preflight_review_command", "").endswith(
        "preflight_dc_p1_native_response_v2_20261027.py --review "
        "--receipt runs/dc_p1_native_response_v2_20261027/preflight_review01.json"),
        "launch", "the review command differs")
    require(config["launch"].get("preflight_launch_command", "").endswith(
        "preflight_dc_p1_native_response_v2_20261027.py --launch "
        "--receipt runs/dc_p1_native_response_v2_20261027/preflight_launch01.json "
        f"--authorization {AUTHORIZATION_PATH.relative_to(ROOT)}"), "launch",
        "the launch command differs")
    require(config["launch"].get("output_root")
            == "runs/dc_p1_native_response_v2_20261027/native01", "launch",
            "the native output root differs")

    integration = config.get("integration_only_fixture", {})
    require(integration.get("status") == "integration_only", "config",
            "the 22-class fixture must stay integration-only")
    require(integration.get("modified") is False, "config",
            "the 22-class fixture must not be modified")
    require(integration.get("used_as_parent") is False, "config",
            "the 22-class fixture cannot be a parent")
    require(integration.get("used_as_scoring_oracle") is False, "config",
            "the 22-class fixture cannot be a scoring oracle")
    require(integration.get("claims_allowed") == [], "config",
            "the 22-class fixture allows no claim here")
    require(config.get("limitations"), "config", "limitations are required")
    return {
        "status": "valid",
        "protocol_id": PROTOCOL_ID,
        "amendments": [row["amendment_id"] for row in amendments],
        "arms": list(ARMS),
        "declared_total_updates": budget["declared_total_updates"],
        "masks_coincide_by_construction": True,
        "declared_claim": CLAIM_GLOBAL_RESCALE,
    }


def validate_lineage(config: Mapping[str, Any],
                     v1_config_path: str | Path = V1_CONFIG_PATH) -> dict:
    """Require the preserved v1 declarations to appear unmodified in v2."""
    v1 = read_json(v1_config_path)
    historical = config["budget"]["historical_v1"]
    v1_limits = v1["budget"]["physical_limits"]
    for key in ("max_model_loads", "max_tokenizer_loads", "max_optimizer_steps",
                "max_backward_calls", "max_forward_calls",
                "max_generation_calls", "max_generated_tokens",
                "max_prefill_tokens", "max_checkpoint_writes",
                "max_downloads", "max_retries", "max_workers",
                "max_wall_seconds"):
        require(historical.get("physical_limits", {}).get(key)
                == v1_limits.get(key), "lineage",
                f"historical v1 limit {key} differs from the preserved config")
    require(historical.get("forward_calls") == v1_limits["max_forward_calls"],
            "lineage", "the historical forward-call number differs from v1")
    require(historical.get("optimizer_steps")
            == v1_limits["max_optimizer_steps"], "lineage",
            "the historical optimizer step number differs from v1")
    require(historical.get("generation_calls")
            == v1_limits["max_generation_calls"], "lineage",
            "the historical generation-call number differs from v1")
    require(v1["protocol_id"] == "dc_p1_native_response_v1", "lineage",
            "the preserved v1 protocol id differs")
    require(v1["classification"]["native_execution_allowed"] is False,
            "lineage", "the v1 package must stay non-native")
    require(config["protocol_id"] != v1["protocol_id"], "lineage",
            "v2 must be a distinct protocol id")
    return {
        "status": "preserved",
        "v1_protocol_id": v1["protocol_id"],
        "v1_forward_calls": v1_limits["max_forward_calls"],
        "v1_optimizer_steps": v1_limits["max_optimizer_steps"],
        "v1_generation_calls": v1_limits["max_generation_calls"],
        "v2_forward_accounting": "execution_point_counters",
    }


# ---------------------------------------------------------------------------
# Preflight


def preflight(config_path: str | Path = CONFIG_PATH,
              prepared: str | Path = PREPARED_DIR, *, launch: bool = False,
              authorization: str | Path | None = None,
              commit_verifier: Callable[[str, str], bytes | None] | None = None,
              repo_root: str | Path = ROOT) -> dict:
    """Review-ready v2 preflight.  Never launches native work itself."""
    config = read_json(config_path)
    validate_config(config)
    validate_lineage(config)
    manifest_path = Path(prepared) / "manifest.json"
    manifest = read_json(manifest_path)
    prepared_report = _validate_fixture(
        Path(prepared), config, validate_labels=False)
    token_lengths_path = Path(prepared) / TOKEN_LENGTHS_PATH.name
    token_lengths_status = "pending_tokenizer_step"
    token_artifact = None
    if token_lengths_path.is_file():
        token_artifact = validate_token_lengths(
            config, read_json(token_lengths_path), prepared=prepared)
        token_lengths_status = "sealed"
    schedule = validate_work_units(config, token_lengths=token_artifact)
    runner_path = config["launch"]["runner_path"]
    runner_file = Path(repo_root) / runner_path
    runner_present = runner_file.is_file()
    config_sha = sha256_file(config_path)
    receipt: dict[str, Any] = {
        "format": PREFLIGHT_FORMAT_V2,
        "protocol_id": PROTOCOL_ID,
        "status": "review_ready",
        "config_path": str(config_path),
        "config_sha256": config_sha,
        "prepared_manifest_path": str(manifest_path),
        "prepared_manifest_sha256": manifest["manifest_sha256"],
        "prepared_validation": {
            "status": prepared_report["status"],
            "counts": prepared_report["counts"],
            "labels_read": False,
        },
        "token_lengths_status": token_lengths_status,
        "token_lengths_path": str(token_lengths_path),
        "work_units": {
            "declared_total_updates": schedule["declared_total_updates"],
            "development_updates": schedule["development_updates"],
            "final_updates": schedule["final_updates"],
            "generation_requests": schedule["counters"][
                "generation_requests"],
            "ceilings": dict(sorted(
                config["budget"]["physical_limits"].items())),
            "measured_counters": (dict(sorted(schedule["counters"].items()))
                                  if token_artifact is not None else None),
            "token_lengths_source": schedule["inputs"]["token_lengths_source"],
        },
        "runner_path": runner_path,
        "runner_sha256": (sha256_file(runner_file) if runner_present else None),
        "runner_present": runner_present,
        "authorization_receipt_present": AUTHORIZATION_PATH.is_file(),
        "native_execution": False,
        "model_loaded": False,
        "tokenizer_loaded": False,
        "downloads": 0,
        "heldout_opened": False,
        "launch_requested": bool(launch),
        "launch_requirements": deepcopy(config["launch"]),
        "review_only": True,
    }
    if not launch:
        receipt["native_execution_status"] = "refused_preparation_only"
        receipt["exit_code"] = 0
        return receipt

    require(authorization is not None, "authorization",
            "a separate externally issued authorization receipt is required "
            "before native work")
    authorization_record = read_json(authorization)
    sealed = verify_sealed_record(authorization_record, check="authorization")
    require(sealed.get("format") == AUTHORIZATION_FORMAT_V2, "authorization",
            "authorization format differs")
    require(sealed.get("protocol_id") == PROTOCOL_ID, "authorization",
            "authorization protocol differs")
    require(sealed.get("config_sha256") == config_sha, "authorization",
            "authorization did not bind the exact config bytes")
    require(sealed.get("prepared_manifest_sha256")
            == manifest["manifest_sha256"], "authorization",
            "authorization did not bind the exact prepared fixture")
    require(sealed.get("runner_path") == runner_path, "authorization",
            "authorization bound a different runner path")
    require(sealed.get("native_execution_allowed") is True, "authorization",
            "native_execution_allowed is not true")
    require(sealed.get("single_attempt") is True, "authorization",
            "exactly one attempt must be authorized")
    require(isinstance(sealed.get("authorized_by"), str)
            and bool(sealed["authorized_by"].strip()), "authorization",
            "authorized_by must be a nonempty string")
    require(sealed.get("max_optimizer_steps")
            == schedule["declared_total_updates"], "authorization",
            "authorization ceiling differs from the derived optimizer steps")
    require(sealed.get("max_generation_requests")
            == schedule["counters"]["generation_requests"], "authorization",
            "authorization ceiling differs from the derived generation count")
    require(sealed.get("restart_allowance") == 1, "authorization",
            "restart allowance must be exactly one same-attempt recovery reload")
    protocol_commit = sealed.get("protocol_commit")
    require(isinstance(protocol_commit, str) and len(protocol_commit) == 40
            and all(character in "0123456789abcdef"
                    for character in protocol_commit), "authorization",
            "authorization must bind a full 40-hex protocol commit")
    require(runner_present, "authorization",
            f"the declared native runner is absent: {runner_path}")
    require(sealed.get("runner_sha256") == receipt["runner_sha256"],
            "authorization",
            "authorization did not bind the exact runner bytes")
    from replica_lab.dc_p1_native_runner import IMPLEMENTATION_RELATIVES
    for relative in IMPLEMENTATION_RELATIVES:
        require_committed_bytes(
            protocol_commit, relative,
            sha256_file(Path(repo_root) / relative), verifier=commit_verifier)
    for path, digest in (
            (str(config_path), config_sha),
            (str(manifest_path), sha256_file(manifest_path)),
            (str(Path(prepared) / "teaching.jsonl"),
             manifest["files"]["teaching.jsonl"])):
        relative = Path(path)
        try:
            relative = relative.resolve().relative_to(Path(repo_root).resolve())
        except ValueError:
            pass
        require_committed_bytes(protocol_commit, str(relative), digest,
                               verifier=commit_verifier)
    receipt["authorization_receipt_sha256"] = sha256_file(authorization)
    receipt["protocol_commit"] = protocol_commit
    receipt["status"] = "launch_admitted"
    receipt["native_execution"] = False
    receipt["native_execution_status"] = (
        "external authorization admitted; the native runner, not this "
        "preflight, performs the work")
    receipt["executor"] = runner_path
    report_root = Path(config["launch"]["output_root"]).parent
    receipt["next_command"] = shlex.join([
        "PYTHONDONTWRITEBYTECODE=1", "python3",
        "scripts/run_capacity_preflight.py",
        "--entry-point", "run_dc_p1_native_response_v2_20261027.py",
        "--report", str(report_root / "guard_fit01.json"), "--",
        "--stage", "fit", "--device", "mps",
        "--output", str(config["launch"]["output_root"]),
        "--authorization", str(authorization),
        "--protocol-commit", protocol_commit,
        "--report", str(report_root / "fit_result01.json")])
    receipt["exit_code"] = 0
    return receipt


# ---------------------------------------------------------------------------
# CLI (fixture build / validation / schedule inspection)


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--work-units", action="store_true")
    parser.add_argument("--output", type=Path, default=PREPARED_DIR)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    args = parser.parse_args(argv)
    try:
        if args.build:
            manifest = build_fixture(args.config, args.output)
            print(json.dumps({
                "status": "built",
                "path": str(args.output),
                "manifest_sha256": manifest["manifest_sha256"],
                "counts": manifest["counts"],
            }, indent=2, sort_keys=True))
        if args.validate:
            print(json.dumps(validate_prepared(args.output, args.config),
                             indent=2, sort_keys=True))
        if args.work_units:
            config = read_json(args.config)
            validate_config(config)
            artifact = None
            path = Path(args.output) / TOKEN_LENGTHS_PATH.name
            if path.is_file():
                artifact = validate_token_lengths(
                    config, read_json(path), prepared=args.output)
            print(json.dumps(derive_work_schedule(
                config, token_lengths=artifact), indent=2, sort_keys=True))
    except ProtocolError as error:
        print(json.dumps({"status": "refused", **error.as_dict()},
                        indent=2, sort_keys=True))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

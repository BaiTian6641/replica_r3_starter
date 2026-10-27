"""DC-P1: bounded native response-recipe learnability preparation.

The study asks one question: after the same native teaching exposures, does one
declared response-recipe change produce useful *source-free generated answers*
on fresh paraphrases and new instances, relative to an identical no-update arm
and a reset-P arm?  The ``source context`` and source KV cache are absent for
every scored generation.

This module is preparation only.  It builds a deterministic role-sealed text
fixture, validates the protocol, scores already-generated answers per case and
fails closed before native execution.  It does not import torch, load MiniCPM5,
load a tokenizer, construct an optimizer or run a native command.

The historical 22-class selected-expert fixture is retained strictly as an
integration/accounting fixture.  It is not a parent, a scoring oracle or
evidence for this study.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Callable, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dc_protocol_common_20261027 import (  # noqa: E402
    ANSWER_SHORTCUT_FIELDS, EVALUATION_CASE_FORBIDDEN_FIELDS,
    GUARD_BYTES, ProtocolError, assert_no_answer_shortcut,
    assert_prompt_matches, assert_routing_is_answer_free, assert_source_free_prediction,
    assert_target_absent, canonical, check_work_budget, find_forbidden_fields,
    normalize_answer, paired_bootstrap_ci, preflight_receipt, read_json,
    read_jsonl, require, seal_manifest, sha256_file, sha256_json,
    tokenizer_binding_sha, validate_baseline_model, validate_guards,
    validate_manifest, validate_tokenizer_binding, verify_decoded_text,
    write_json, write_jsonl,
)

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_ID = "dc_p1_native_response_v1"
CONFIG_FORMAT = "replica.dc_p1_native_response.protocol.v1"
MANIFEST_FORMAT = "replica.dc_p1_native_response.prepared.v1"
CASE_FORMAT = "replica.dc_p1_native_response.eval_case.v1"
TEACHING_FORMAT = "replica.dc_p1_native_response.teaching_case.v1"
LABEL_FORMAT = "replica.dc_p1_native_response.label.v1"
PREDICTION_FORMAT = "replica.dc_p1_native_response.prediction.v1"
SCORE_FORMAT = "replica.dc_p1_native_response.score_report.v1"

CONFIG_PATH = ROOT / "configs" / "dc_p1_native_response_protocol_v1.json"
PREPARED_DIR = ROOT / "data" / "dc_p1_native_response_20261027"
TRAIN_GROUPS = 12
SEALED_GROUPS = 4
TRAIN_ROLES = ("fit_primary", "fit_correction", "fit_abstention")
DEV_ROLES = ("paraphrase", "new_instance", "correction", "abstention",
             "retention")
FIT_ROLE_TO_TEACHING = {
    "fit_primary": "primary",
    "fit_correction": "correction",
    "fit_abstention": "abstention",
}
ARMS = ("treatment", "no_update", "reset_P")
PRIMARY_ROLES = ("paraphrase", "new_instance")
FILES = (
    "teaching.jsonl",
    "fit_cases.jsonl",
    "dev_cases.jsonl",
    "sealed_final_teaching.jsonl",
    "sealed_final_fit_cases.jsonl",
    "sealed_final_cases.jsonl",
    "sealed_labels_fit.jsonl",
    "sealed_labels_dev.jsonl",
    "sealed_final_labels_fit.jsonl",
    "sealed_labels_final.jsonl",
)
SYLLABLES = (
    "ka", "ro", "mi", "ta", "ne", "su", "vi", "lo", "za", "pe",
    "du", "fi", "go", "ha", "je", "qu", "we", "xi", "yo", "bu",
)
PRIMARY_CHANGE_FIELD = "loss.answer_span_ce_weight"
CHANGE_FROM = 0.0
CHANGE_TO = 0.5


# ---------------------------------------------------------------------------
# Deterministic synthetic words


def _word(*parts: Any) -> str:
    digest = hashlib.sha256(
        canonical(list(parts)).encode("utf-8")).hexdigest()
    indices = [int(digest[index:index + 2], 16) % len(SYLLABLES)
               for index in range(0, 9, 2)]
    return "".join(SYLLABLES[index] for index in indices)


def _group(index: int, sealed: bool = False) -> str:
    prefix = "sealed" if sealed else "dev"
    return f"p1src_{prefix}_{index:02d}"


def _case_id(group: str, role: str) -> str:
    return f"{group}_{role}"


def _prompt_sha(prompt: str) -> str:
    return sha256_json({"prompt": prompt})


def _paraphrase(group: str, entity: str, variant: int) -> str:
    return (
        f"State the phrase recorded for locker {entity}.",
        f"What phrase belongs to locker {entity}?",
        f"Return the phrase for locker {entity}.",
    )[variant % 3]


def _new_instance_prompt(entity: str, variant: int) -> str:
    return (
        f"Which phrase is assigned to locker {entity}?",
        f"Report the phrase held by locker {entity}.",
        f"Name the phrase for locker {entity}.",
    )[variant % 3]


def _correction_prompt(entity: str, variant: int) -> str:
    return (
        f"Which phrase is current for locker {entity}?",
        f"State the corrected phrase for locker {entity}.",
        f"After the update, which phrase does locker {entity} hold?",
    )[variant % 3]


def _abstention_prompt(entity: str, variant: int) -> str:
    return (
        f"Which phrase belongs to locker {entity}?",
        f"Report the phrase for locker {entity}.",
        f"Name the phrase for locker {entity}.",
    )[variant % 3]


def _fit_prompt(role: str, entity: str) -> str:
    if role == "primary":
        return f"Which phrase is assigned to locker {entity}?"
    if role == "correction":
        return f"After the correction, what phrase belongs to locker {entity}?"
    return f"What phrase is recorded for locker {entity}?"


def _retention_prompt(word: str) -> str:
    return (f"Reverse the letters of the word '{word}' and answer with only "
            "the reversed word.")


def _facts(group: str, index: int, sealed: bool) -> dict[str, str]:
    return {
        "group": group,
        "primary_entity": _word("p1-entity", group, "primary"),
        "secondary_entity": _word("p1-entity", group, "secondary"),
        "absent_entity": _word("p1-entity", group, "absent"),
        "primary_value": _word("p1-value", group, "primary"),
        "secondary_value": _word("p1-value", group, "secondary"),
        "corrected_value": _word("p1-value", group, "corrected"),
        "retention_word": _word("p1-retain", group, "word"),
    }


def _build_group(group: str, index: int, sealed: bool) -> tuple[
        list[dict], list[dict], list[dict], list[dict], list[dict]]:
    facts = _facts(group, index, sealed)
    e1, e2, e3 = (facts["primary_entity"], facts["secondary_entity"],
                  facts["absent_entity"])
    v1, v2, v3 = (facts["primary_value"], facts["secondary_value"],
                  facts["corrected_value"])
    primary_source = (
        f"Archive {group}: locker {e1} uses phrase {v1}. "
        f"Locker {e2} uses phrase {v2}.")
    correction_source = (
        f"Registry {group}: locker {e1} used phrase {v1}. "
        f"Correction: locker {e1} now uses phrase {v3}.")
    abstention_source = f"Registry {group}: locker {e2} uses phrase {v2}."
    teaching: list[dict] = []
    for role, source, prompt, target, superseded in (
        ("primary", primary_source, _fit_prompt("primary", e1), v1, None),
        ("correction", correction_source, _fit_prompt("correction", e1),
         v3, v1),
        ("abstention", abstention_source,
         _fit_prompt("abstention", e3), "UNKNOWN", None),
    ):
        teaching.append({
            "format": TEACHING_FORMAT,
            "case_id": _case_id(group, role),
            "split_role": "train",
            "role": role,
            "source_group": group,
            "source": source,
            "prompt": prompt,
            "prompt_sha256": _prompt_sha(prompt),
            "target": target,
            "superseded_value": superseded,
            "target_available_at": 1000 + index,
            "source_context_allowed": True,
            "source_kv_allowed": True,
            "routing_metadata": {"mode": "fixed", "expert_id": 0},
        })

    fit_cases: list[dict] = []
    fit_labels: list[dict] = []
    for role_teach, role_fit, target in (
        ("primary", "fit_primary", v1),
        ("correction", "fit_correction", v3),
        ("abstention", "fit_abstention", "UNKNOWN"),
    ):
        prompt = _fit_prompt(role_teach, e1 if role_teach != "abstention" else e3)
        case = {
            "format": CASE_FORMAT,
            "case_id": _case_id(group, role_fit),
            "split_role": "fit",
            "role": role_fit,
            "source_group": group,
            "prompt": prompt,
            "prompt_sha256": _prompt_sha(prompt),
            "inference_inputs": ["prompt"],
            "source_context_present": False,
            "source_kv_loaded": False,
        }
        fit_cases.append(case)
        fit_labels.append({
            "format": LABEL_FORMAT,
            "case_id": case["case_id"],
            "split_role": "fit",
            "role": role_fit,
            "source_group": group,
            "target": target,
        })

    variant = index % 3
    dev_specs = [
        ("paraphrase", _paraphrase(group, e1, variant), v1, {}),
        ("new_instance", _new_instance_prompt(e2, variant + 1), v2, {}),
        ("correction", _correction_prompt(e1, variant), v3, {"superseded_value": v1}),
        ("abstention", _abstention_prompt(e3, variant + 2), "UNKNOWN", {}),
        ("retention", _retention_prompt(facts["retention_word"]),
         facts["retention_word"][::-1],
         {"retention_input": facts["retention_word"], "retention_rule": "reverse"}),
    ]
    dev_cases: list[dict] = []
    dev_labels: list[dict] = []
    evaluation_split = "sealed_final" if sealed else "dev"
    for role, prompt, target, extra in dev_specs:
        case = {
            "format": CASE_FORMAT,
            "case_id": _case_id(group, role),
            "split_role": evaluation_split,
            "role": role,
            "source_group": group,
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
        }
        label.update(extra)
        dev_labels.append(label)
    return teaching, fit_cases, fit_labels, dev_cases, dev_labels


def build_fixture(config_path: str | Path, output: str | Path) -> dict:
    """Build a fresh deterministic fixture; never overwrite an existing tree."""
    config = read_json(config_path)
    require(config.get("protocol_id") == PROTOCOL_ID, "build",
            "config protocol_id differs")
    require(config.get("prepared", {}).get("groups") == TRAIN_GROUPS,
            "build", "declared training group count differs")
    require(config.get("prepared", {}).get("sealed_groups") == SEALED_GROUPS,
            "build", "declared sealed group count differs")
    root = Path(output)
    require(not root.exists(), "build", f"output is not fresh: {root}")
    root.mkdir(parents=True)

    teaching: list[dict] = []
    fit_cases: list[dict] = []
    fit_labels: list[dict] = []
    dev_cases: list[dict] = []
    dev_labels: list[dict] = []
    for index in range(TRAIN_GROUPS):
        group = _group(index)
        built = _build_group(group, index, sealed=False)
        teaching.extend(built[0])
        fit_cases.extend(built[1])
        fit_labels.extend(built[2])
        dev_cases.extend(built[3])
        dev_labels.extend(built[4])

    final_teaching: list[dict] = []
    final_fit_cases: list[dict] = []
    final_fit_labels: list[dict] = []
    final_cases: list[dict] = []
    final_labels: list[dict] = []
    for offset in range(SEALED_GROUPS):
        index = TRAIN_GROUPS + offset
        group = _group(offset, sealed=True)
        built = _build_group(group, index, sealed=True)
        # Sealed-group teaching and fit rows live only in sealed files.  They
        # are never part of the development training stream.
        final_teaching.extend(built[0])
        final_fit_cases.extend(built[1])
        final_fit_labels.extend(built[2])
        final_cases.extend(built[3])
        final_labels.extend(built[4])

    files = {
        "teaching.jsonl": teaching,
        "fit_cases.jsonl": fit_cases,
        "dev_cases.jsonl": dev_cases,
        "sealed_final_teaching.jsonl": final_teaching,
        "sealed_final_fit_cases.jsonl": final_fit_cases,
        "sealed_final_cases.jsonl": final_cases,
        "sealed_labels_fit.jsonl": fit_labels,
        "sealed_labels_dev.jsonl": dev_labels,
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
            "train_and_dev": sorted({row["source_group"] for row in teaching}),
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

    require(len(teaching) == TRAIN_GROUPS * 3, "fixture",
            "teaching count differs")
    require(len(fit_cases) == TRAIN_GROUPS * 3, "fixture",
            "fit case count differs")
    require(len(dev_cases) == TRAIN_GROUPS * len(DEV_ROLES), "fixture",
            "dev case count differs")
    require(len(final_teaching) == SEALED_GROUPS * 3, "fixture",
            "sealed final teaching count differs")
    require(len(final_fit_cases) == SEALED_GROUPS * 3, "fixture",
            "sealed final fit case count differs")
    require(len(final_cases) == SEALED_GROUPS * len(DEV_ROLES), "fixture",
            "sealed final case count differs")
    require(len(fit_labels) == len(fit_cases), "fixture",
            "fit label coverage differs")
    require(len(dev_labels) == len(dev_cases), "fixture",
            "dev label coverage differs")
    require(len(final_fit_labels) == len(final_fit_cases), "fixture",
            "sealed final fit label coverage differs")
    require(len(final_labels) == len(final_cases), "fixture",
            "sealed final label coverage differs")

    for row in teaching + final_teaching:
        require(row.get("format") == TEACHING_FORMAT, "fixture",
                "teaching row format differs")
        require(row.get("split_role") == "train", "fixture",
                "teaching row split_role differs")
        require(row.get("role") in ("primary", "correction", "abstention"),
                "fixture", "unknown teaching role")
        require(isinstance(row.get("source"), str) and row["source"],
                "fixture", "teaching source is empty")
        require(row.get("source_context_allowed") is True, "fixture",
                "teaching must fail closed without source context")
        require(row.get("source_kv_allowed") is True, "fixture",
                "teaching source KV policy differs")
        assert_routing_is_answer_free(row.get("routing_metadata"))
        require(row["prompt_sha256"] == _prompt_sha(row["prompt"]), "fixture",
                "teaching prompt hash differs")
        assert_target_absent(row["prompt"], row["target"], check="fixture")
        if row["role"] == "abstention":
            require(row["target"] == "UNKNOWN", "fixture",
                    "abstention teaching target must be UNKNOWN")
        if row["role"] == "correction":
            require(row.get("superseded_value")
                    and row["superseded_value"] != row["target"], "fixture",
                    "correction must change the value")

    for rows, labels in ((fit_cases, fit_labels),
                         (final_fit_cases, final_fit_labels)):
        require({row["case_id"] for row in rows}
                == {row["case_id"] for row in labels}, "fixture",
                "fit case/label ids differ")
        for case in rows:
            require(case.get("format") == CASE_FORMAT, "fixture",
                    "fit case format differs")
            require(case.get("split_role") == "fit", "fixture",
                    "fit case split_role differs")
            require(find_forbidden_fields(
                case, EVALUATION_CASE_FORBIDDEN_FIELDS) == [], "fixture",
                "fit case carries forbidden fields")
            require(case.get("inference_inputs") == ["prompt"], "fixture",
                    "fit evaluation must expose only the source-free prompt")
            require(case.get("source_context_present") is False, "fixture",
                    "fit case would retain source context")
            require(case.get("source_kv_loaded") is False, "fixture",
                    "fit case would retain source KV")
            assert_routing_is_answer_free(case.get("routing_metadata", {}))
            require(case["prompt_sha256"] == _prompt_sha(case["prompt"]),
                    "fixture", "fit prompt hash differs")
    for rows, labels, split in ((dev_cases, dev_labels, "dev"),
                                (final_cases, final_labels, "sealed_final")):
        require({row["case_id"] for row in rows}
                == {row["case_id"] for row in labels}, "fixture",
                f"{split} case/label ids differ")
        for case in rows:
            require(case.get("format") == CASE_FORMAT, "fixture",
                    "evaluation case format differs")
            require(case.get("split_role") == split, "fixture",
                    "evaluation split_role differs")
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
            assert_routing_is_answer_free(case.get("routing_metadata", {}))
            require(case["prompt_sha256"] == _prompt_sha(case["prompt"]),
                    "fixture", "evaluation prompt hash differs")

    label_by_id = {row["case_id"]: row for row in
                   fit_labels + final_fit_labels + dev_labels + final_labels}
    all_cases = fit_cases + final_fit_cases + dev_cases + final_cases
    for case in all_cases:
        label = label_by_id[case["case_id"]]
        assert_target_absent(case["prompt"], label["target"], check="fixture")
        require(label["source_group"] == case["source_group"], "fixture",
                "case/label source group differs")
        require(label["role"] == case["role"], "fixture",
                "case/label role differs")
        if label["role"] == "abstention":
            require(label["target"] == "UNKNOWN", "fixture",
                    "abstention target must be UNKNOWN")
        if label["role"] == "retention":
            require(label.get("retention_input")
                    and label["target"] == label["retention_input"][::-1],
                    "fixture", "retention target is not the declared reversal")
        if label["role"] == "correction":
            require(label.get("superseded_value")
                    and label["superseded_value"] != label["target"],
                    "fixture", "correction target did not change")
        require(label["target"].casefold() not in case["prompt"].casefold(),
                "fixture", "target leaked into evaluation prompt")

    train_groups = {row["source_group"] for row in teaching}
    dev_groups = {row["source_group"] for row in dev_cases}
    final_groups = {row["source_group"] for row in final_cases}
    final_teach_groups = {row["source_group"] for row in final_teaching}
    require(train_groups == dev_groups, "fixture",
            "training and development source groups differ")
    require(final_teach_groups == final_groups, "fixture",
            "sealed final teaching/evaluation groups differ")
    require(not (train_groups & final_groups), "fixture",
            "sealed final groups overlap training/development groups")
    require(len(final_groups) == SEALED_GROUPS, "fixture",
            "sealed final group count differs")
    require(len(train_groups) == TRAIN_GROUPS, "fixture",
            "training group count differs")

    # Prompt and source-group uniqueness prevents globally unique prompt
    # formats or random-ID shortcuts from acting as hidden identifiers.
    prompts = [row["prompt"] for row in
               teaching + final_teaching + dev_cases + final_cases]
    require(len(prompts) == len(set(prompts)), "fixture",
            "teaching/development prompt is duplicated")
    require(len({row["case_id"] for row in all_cases}) == len(all_cases),
            "fixture", "case ids are not unique")

    if validate_labels:
        # The construction audit is allowed to read every split, but the
        # manifest still declares held-out labels closed to runtime code.
        require(not any(row.get("answer_index") is not None
                        for row in fit_labels + final_fit_labels + dev_labels
                        + final_labels),
                "fixture", "labels must not use answer indexes")
    return {
        "format": "replica.dc_p1_native_response.validation.v1",
        "protocol_id": PROTOCOL_ID,
        "status": "valid",
        "manifest_sha256": manifest["manifest_sha256"],
        "counts": manifest["counts"],
        "source_groups": manifest["source_groups"],
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
    require(config.get("study") == "DC-P1", "config", "study must be DC-P1")
    classification = config.get("classification", {})
    require(classification.get("preparation_only") is True, "config",
            "preparation_only must be true")
    require(classification.get("native_execution_allowed") is False, "config",
            "native execution must be false before separate authorization")
    require(classification.get("model_execution") is False, "config",
            "no model execution in preparation")
    require(classification.get("tokenizer_load") is False, "config",
            "no tokenizer load in preparation")
    validate_guards(config)
    validate_baseline_model(config)
    validate_tokenizer_binding(config.get("tokenizer_binding"))
    representation = config.get("representation", {})
    require(representation.get("synthetic_feature_injection") is False,
            "representation", "synthetic feature injection is forbidden")
    require(representation.get("answer_index_bypass") is False,
            "representation", "answer-index bypass is forbidden")
    require(representation.get("selected_value_carrier") == "none",
            "representation", "a selected-value carrier would bypass learning")
    require(representation.get("source_context_at_scoring") is False,
            "representation", "source context must be absent at scoring")
    require(representation.get("source_kv_at_scoring") is False,
            "representation", "source KV must be absent at scoring")
    parity = config.get("baseline_parity", {})
    require(parity.get("no_extension_is_default") is True, "baseline_parity",
            "no-extension default is required")
    require(parity.get("disabled_extension_preserves_native_logits") is True,
            "baseline_parity", "native logit parity is required")
    require(parity.get("disabled_extension_preserves_native_greedy_tokens")
            is True, "baseline_parity", "native greedy parity is required")
    require(parity.get("logit_rtol") == 0.0 and parity.get("logit_atol") == 0.0,
            "baseline_parity", "exact baseline parity is required")
    require(parity.get("parity_required_before_teaching") is True,
            "baseline_parity", "parity must precede teaching")
    require(parity.get("parity_required_after_reset_P") is True,
            "baseline_parity", "reset-P parity is required")

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

    hypothesis = config.get("hypothesis", {})
    require(hypothesis.get("id") == "DC-P1-H1", "config",
            "hypothesis id differs")
    recipe = config.get("recipe", {})
    baseline = recipe.get("baseline", {})
    treatment = recipe.get("treatment", {})
    required_baseline = {
        "objective": "mean_response_nll",
        "steps_per_event": 1,
        "optimizer": "NativeFusedAdamW",
        "learning_rate": 0.0001,
        "gradient_norm_max": 1.0,
        "response_only_labels": True,
        "label_shift": 1,
        "trainable_groups": [
            "expert_0.gate", "expert_0.up", "expert_0.down"],
        "source_context_in_teaching_prompt": True,
        "source_kv_reused_at_evaluation": False,
    }
    for key, expected in required_baseline.items():
        require(baseline.get(key) == expected, "recipe",
                f"baseline {key}={baseline.get(key)!r} differs from {expected!r}")
    require(treatment.get("steps_per_event") == 1, "recipe",
            "budget02 revival is forbidden; exactly one step per event")
    require(treatment.get("budget02_revival") is False, "recipe",
            "budget02 revival is forbidden")
    changes = treatment.get("changes")
    require(isinstance(changes, list) and len(changes) == 1, "recipe",
            "exactly one primary recipe change is required")
    change = changes[0]
    require(change.get("field") == PRIMARY_CHANGE_FIELD, "recipe",
            "primary change field differs")
    require(change.get("from") == CHANGE_FROM, "recipe",
            "baseline answer-span weight differs")
    require(change.get("to") == CHANGE_TO, "recipe",
            "treatment answer-span weight differs")
    require(treatment.get("primary_change_count") == 1, "recipe",
            "primary_change_count must be one")
    require(treatment.get("unchanged_vs_baseline")
            == sorted(required_baseline), "recipe",
            "treatment must enumerate exactly the unchanged baseline fields")
    for key, expected in required_baseline.items():
        require(treatment.get(key) == expected, "recipe",
                f"treatment silently changed baseline field {key}")
    require(treatment.get("optimizer") == "NativeFusedAdamW", "recipe",
            "optimizer must remain unchanged")
    require(treatment.get("learning_rate") == 0.0001, "recipe",
            "learning rate must remain unchanged")
    require(treatment.get("steps_per_event") == 1, "recipe",
            "steps per event must remain one")
    require(treatment.get("source_kv_reused_at_evaluation") is False,
            "recipe", "evaluation source KV reuse is forbidden")
    require(treatment.get("source_context_removed_at_evaluation") is True,
            "recipe", "evaluation source context must be removed")

    anti = config.get("anti_shortcut", {})
    require(anti.get("answer_index_bypass_allowed") is False, "anti_shortcut",
            "answer-index bypass is forbidden")
    require(anti.get("answer_bearing_routing_metadata") is False,
            "anti_shortcut", "answer-bearing routing metadata is forbidden")
    require(anti.get("source_context_removed_at_scoring") is True,
            "anti_shortcut", "source context must be removed at scoring")
    require(anti.get("source_kv_removed_at_scoring") is True, "anti_shortcut",
            "source KV must be removed at scoring")
    require(anti.get("per_case_scoring") is True, "anti_shortcut",
            "per-case scoring is required")
    require(anti.get("controls_use_identical_scorer") is True,
            "anti_shortcut", "controls must use the identical scorer")
    require(anti.get("corrupted_controls_must_fail") is True, "anti_shortcut",
            "corrupted controls must fail")
    require(anti.get("target_in_prompt_allowed") is False, "anti_shortcut",
            "targets may not be placed in evaluation prompts")

    prepared = config.get("prepared", {})
    require(prepared.get("root") == str(PREPARED_DIR.relative_to(ROOT)),
            "prepared", "prepared root differs")
    require(prepared.get("groups") == TRAIN_GROUPS, "prepared",
            "training group count differs")
    require(prepared.get("sealed_groups") == SEALED_GROUPS, "prepared",
            "sealed group count differs")
    counts = prepared.get("counts", {})
    require(counts.get("teaching") == TRAIN_GROUPS * 3, "prepared",
            "teaching count differs")
    require(counts.get("fit_cases") == TRAIN_GROUPS * 3, "prepared",
            "fit case count differs")
    require(counts.get("dev_cases") == TRAIN_GROUPS * len(DEV_ROLES),
            "prepared", "dev case count differs")
    require(counts.get("sealed_final_teaching") == SEALED_GROUPS * 3,
            "prepared", "sealed final teaching count differs")
    require(counts.get("sealed_final_fit_cases") == SEALED_GROUPS * 3,
            "prepared", "sealed final fit count differs")
    require(counts.get("sealed_final_cases")
            == SEALED_GROUPS * len(DEV_ROLES), "prepared",
            "sealed final count differs")
    require(counts.get("optimizer_steps")
            == TRAIN_GROUPS * 3 + SEALED_GROUPS * 3, "prepared",
            "optimizer step count differs")
    require(prepared.get("manifest_sha256") is None
            or isinstance(prepared.get("manifest_sha256"), str),
            "prepared", "manifest digest must be null or a string")

    controls = config.get("controls", {})
    arms = controls.get("arms", [])
    require([row.get("arm_id") for row in arms] == list(ARMS), "controls",
            "control arm order/identity differs")
    require(arms[0].get("role") == "treatment", "controls",
            "first arm must be treatment")
    require(arms[1].get("updates") == 0, "controls",
            "identical no-update arm must have zero updates")
    require(arms[2].get("reset_before_evaluation") is True, "controls",
            "reset-P arm must reset before evaluation")
    require(controls.get("identical_data_and_budget") is True, "controls",
            "all arms must share data and budget")
    require(controls.get("fresh_paraphrase_required") is True, "controls",
            "fresh paraphrase gain is required")
    require(controls.get("new_instance_required") is True, "controls",
            "new-instance gain is required")

    evaluation = config.get("evaluation", {})
    require(evaluation.get("source_free") is True, "evaluation",
            "evaluation must be source-free")
    require(evaluation.get("generated_answer_scoring") is True, "evaluation",
            "generated-answer scoring is required")
    require(evaluation.get("tokenizer_and_decode_binding_required") is True,
            "evaluation", "tokenizer/decode binding is required")
    require(evaluation.get("fit_then_dev_then_sealed_final") is True,
            "evaluation", "evaluation order must be fit, dev, sealed final")
    require(evaluation.get("sealed_final_one_shot") is True, "evaluation",
            "sealed final must be one-shot")
    require(evaluation.get("sealed_final_requires_separate_teaching") is True,
            "evaluation", "sealed final must be taught separately")
    require(evaluation.get("sealed_final_teaching_updates")
            == SEALED_GROUPS * 3, "evaluation",
            "sealed final teaching updates differ")
    require(evaluation.get("target_conditioned_thresholds") is False,
            "evaluation", "thresholds may not depend on targets")
    gates = evaluation.get("gates", {})
    for key in ("fit_exact_min", "fit_nonempty_min",
                "fresh_paraphrase_min_gain", "new_instance_min_gain",
                "fresh_paraphrase_min_accuracy",
                "new_instance_min_accuracy", "correction_accuracy_min",
                "abstention_accuracy_min", "abstention_false_recall_max",
                "retention_accuracy_min", "retention_decline_max",
                "control_primary_accuracy_max", "paired_ci_low_min"):
        require(key in gates, "evaluation", f"missing gate {key}")

    budget = config.get("budget", {})
    check_work_budget({}, budget)
    physical = budget.get("physical_limits", {})
    teaching_updates = TRAIN_GROUPS * 3 + SEALED_GROUPS * 3
    require(physical.get("max_optimizer_steps")
            == teaching_updates, "budget",
            "optimizer step ceiling differs from teaching count")
    require(physical.get("max_backward_calls")
            == teaching_updates, "budget",
            "backward ceiling differs from teaching count")
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
        "primary_change": deepcopy(change),
        "unchanged_fields": sorted(required_baseline),
    }


def validate_prepared(prepared: str | Path = PREPARED_DIR,
                      config_path: str | Path = CONFIG_PATH) -> dict:
    config = read_json(config_path)
    validate_config(config)
    return _validate_fixture(Path(prepared), config, validate_labels=True)


# ---------------------------------------------------------------------------
# Per-case generated-answer scoring


def _accuracy(rows: Sequence[Mapping[str, Any]]) -> float | None:
    if not rows:
        return None
    return sum(1 for row in rows if row["correct"]) / len(rows)


def _role_rows(rows: Sequence[Mapping[str, Any]], role: str) -> list[dict]:
    return [row for row in rows if row["role"] == role]


def _group_accuracy(rows: Sequence[Mapping[str, Any]], role: str) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        if row["role"] == role:
            grouped.setdefault(row["source_group"], []).append(
                1.0 if row["correct"] else 0.0)
    return {group: sum(values) / len(values)
            for group, values in sorted(grouped.items())}


def _score_one(prediction: Mapping[str, Any], case: Mapping[str, Any],
               label: Mapping[str, Any], binding: Mapping[str, Any],
               decode_fn: Callable[[Sequence[int]], str]) -> dict:
    require(prediction.get("format") == PREDICTION_FORMAT, "prediction",
            "prediction format differs")
    require(prediction.get("protocol_id") == PROTOCOL_ID, "prediction",
            "prediction protocol differs")
    assert_no_answer_shortcut(prediction)
    assert_prompt_matches(case, prediction)
    assert_source_free_prediction(prediction)
    assert_routing_is_answer_free(prediction.get("routing_metadata", {}))
    text = verify_decoded_text(prediction, binding, decode_fn)
    target = label["target"]
    assert_target_absent(case["prompt"], target, check="target_leak")
    require(label["case_id"] == case["case_id"], "label",
            "label/case mismatch")
    correct = normalize_answer(text) == normalize_answer(target)
    abstained = normalize_answer(text) == "unknown"
    return {
        "case_id": case["case_id"],
        "arm_id": prediction["arm_id"],
        "split": case["split_role"],
        "role": case["role"],
        "source_group": case["source_group"],
        "target": target,
        "generated_text": text,
        "normalized_generated_text": normalize_answer(text),
        "correct": correct,
        "abstained": abstained,
        "empty_generation": text == "",
        "stop_reason": prediction.get("stop_reason"),
        "stopped": prediction.get("stopped") is True,
    }


def score_p1(
        predictions_path: str | Path,
        *,
        fit_cases_path: str | Path,
        fit_labels_path: str | Path,
        dev_cases_path: str | Path,
        dev_labels_path: str | Path,
        config_path: str | Path = CONFIG_PATH,
        decode_fn: Callable[[Sequence[int]], str]) -> dict:
    """Score already-generated DC-P1 answers.  No tokenizer is loaded here."""
    config = read_json(config_path)
    validate_config(config)
    binding = validate_tokenizer_binding(config["tokenizer_binding"])
    require(callable(decode_fn), "scoring",
            "a sealed-tokenizer decode function is required")
    cases: dict[str, dict] = {}
    labels: dict[str, dict] = {}
    for path in (fit_cases_path, dev_cases_path):
        for case in read_jsonl(path):
            require(case["case_id"] not in cases, "scoring",
                    f"duplicate case id {case['case_id']}")
            cases[case["case_id"]] = case
    for path in (fit_labels_path, dev_labels_path):
        for label in read_jsonl(path):
            require(label["case_id"] not in labels, "scoring",
                    f"duplicate label id {label['case_id']}")
            labels[label["case_id"]] = label
    require(set(cases) == set(labels), "scoring",
            "case/label coverage differs")

    seen: set[tuple[str, str]] = set()
    scored: list[dict] = []
    for row in read_jsonl(predictions_path):
        arm = row.get("arm_id")
        case_id = row.get("case_id")
        require(arm in ARMS, "scoring", f"unknown arm {arm!r}")
        require(case_id in cases, "scoring", f"unknown case {case_id!r}")
        key = (arm, case_id)
        require(key not in seen, "scoring", f"duplicate prediction {key}")
        seen.add(key)
        scored.append(_score_one(row, cases[case_id], labels[case_id],
                                 binding, decode_fn))
    expected = {(arm, case_id) for arm in ARMS for case_id in cases}
    missing = sorted(expected - seen)
    require(not missing, "scoring",
            f"missing predictions: {missing[:8]}")

    by_arm: dict[str, list[dict]] = {}
    for row in scored:
        by_arm.setdefault(row["arm_id"], []).append(row)

    def arm_metrics(arm: str) -> dict:
        rows = by_arm[arm]
        fit_rows = [row for row in rows if row["split"] == "fit"]
        dev_rows = [row for row in rows if row["split"] == "dev"]
        fit_metrics = {
            "cases": len(fit_rows),
            "exact_accuracy": _accuracy(fit_rows),
            "nonempty_rate": (
                sum(1 for row in fit_rows if not row["empty_generation"])
                / len(fit_rows) if fit_rows else None),
            "stop_rate": (
                sum(1 for row in fit_rows if row["stopped"])
                / len(fit_rows) if fit_rows else None),
            "by_role": {
                role: _accuracy(_role_rows(fit_rows, role))
                for role in ("fit_primary", "fit_correction", "fit_abstention")
            },
        }
        role_metrics = {}
        for role in DEV_ROLES:
            role_rows = _role_rows(dev_rows, role)
            role_metrics[role] = {
                "cases": len(role_rows),
                "exact_accuracy": _accuracy(role_rows),
                "abstain_rate": (
                    sum(1 for row in role_rows if row["abstained"])
                    / len(role_rows) if role_rows else None),
                "false_recall_rate": (
                    sum(1 for row in role_rows
                        if row["abstained"] is False
                        and normalize_answer(row["target"]) == "unknown")
                    / len(role_rows) if role_rows else None),
            }
        return {
            "arm_id": arm,
            "fit": fit_metrics,
            "dev": role_metrics,
            "all_predictions": len(rows),
        }

    metrics = {arm: arm_metrics(arm) for arm in ARMS}
    treatment = metrics["treatment"]

    def accuracy(arm: str, role: str) -> float:
        value = metrics[arm]["dev"][role]["exact_accuracy"]
        require(value is not None, "scoring", f"no rows for {arm}/{role}")
        return float(value)

    def gain_report(control: str, role: str) -> dict:
        treatment_groups = _group_accuracy(by_arm["treatment"], role)
        control_groups = _group_accuracy(by_arm[control], role)
        require(set(treatment_groups) == set(control_groups), "scoring",
                f"paired groups differ for {role}")
        diffs = [treatment_groups[group] - control_groups[group]
                 for group in sorted(treatment_groups)]
        ci = paired_bootstrap_ci(
            diffs, seed=config["seed"],
            resamples=config["evaluation"]["paired_bootstrap_resamples"])
        return {
            "role": role,
            "control": control,
            "paired_groups": len(diffs),
            "treatment_accuracy": accuracy("treatment", role),
            "control_accuracy": accuracy(control, role),
            "mean_gain": sum(diffs) / len(diffs),
            "ci_low": ci["low"],
            "ci_high": ci["high"],
            "ci": ci,
        }

    gains = {
        f"{role}_vs_{control}": gain_report(control, role)
        for role in PRIMARY_ROLES for control in ("no_update", "reset_P")
    }
    gates_cfg = config["evaluation"]["gates"]
    role_thresholds = {
        "paraphrase": gates_cfg["fresh_paraphrase_min_accuracy"],
        "new_instance": gates_cfg["new_instance_min_accuracy"],
        "correction": gates_cfg["correction_accuracy_min"],
        "abstention": gates_cfg["abstention_accuracy_min"],
    }
    gate_results: dict[str, dict] = {}
    for role, threshold in role_thresholds.items():
        value = accuracy("treatment", role)
        gate_results[f"treatment_{role}_accuracy"] = {
            "value": value, "min": threshold, "passed": value >= threshold}
    fit_exact = treatment["fit"]["exact_accuracy"]
    fit_nonempty = treatment["fit"]["nonempty_rate"]
    gate_results["treatment_fit_exact"] = {
        "value": fit_exact, "min": gates_cfg["fit_exact_min"],
        "passed": fit_exact is not None
        and fit_exact >= gates_cfg["fit_exact_min"]}
    gate_results["treatment_fit_nonempty"] = {
        "value": fit_nonempty, "min": gates_cfg["fit_nonempty_min"],
        "passed": fit_nonempty is not None
        and fit_nonempty >= gates_cfg["fit_nonempty_min"]}

    gain_keys = {
        "paraphrase": "fresh_paraphrase_min_gain",
        "new_instance": "new_instance_min_gain",
    }
    for role in PRIMARY_ROLES:
        gain_key = gain_keys[role]
        for control in ("no_update", "reset_P"):
            report = gains[f"{role}_vs_{control}"]
            gate_results[f"{role}_gain_vs_{control}"] = {
                "mean_gain": report["mean_gain"],
                "min_gain": gates_cfg[gain_key],
                "ci_low": report["ci_low"],
                "ci_low_min": gates_cfg["paired_ci_low_min"],
                "passed": (
                    report["mean_gain"] >= gates_cfg[gain_key]
                    and report["ci_low"] > gates_cfg["paired_ci_low_min"]),
            }

    control_max = gates_cfg["control_primary_accuracy_max"]
    for control in ("no_update", "reset_P"):
        for role in PRIMARY_ROLES:
            value = accuracy(control, role)
            gate_results[f"{control}_{role}_control_ceiling"] = {
                "value": value, "max": control_max,
                "passed": value <= control_max,
            }

    false_recall = treatment["dev"]["abstention"]["false_recall_rate"]
    gate_results["abstention_false_recall"] = {
        "value": false_recall,
        "max": gates_cfg["abstention_false_recall_max"],
        "passed": false_recall is not None
        and false_recall <= gates_cfg["abstention_false_recall_max"],
    }

    retention = accuracy("treatment", "retention")
    gate_results["retention_accuracy"] = {
        "value": retention, "min": gates_cfg["retention_accuracy_min"],
        "passed": retention >= gates_cfg["retention_accuracy_min"]}
    for control in ("no_update", "reset_P"):
        decline = accuracy(control, "retention") - retention
        gate_results[f"retention_decline_vs_{control}"] = {
            "value": decline, "max": gates_cfg["retention_decline_max"],
            "passed": decline <= gates_cfg["retention_decline_max"]}

    # A corrupted control that copies the answer index or leaks the target is
    # rejected by _score_one before reaching these gates.  These checks make
    # it impossible for an affected control to be silently re-scored with a
    # different policy.
    require(all(row["arm_id"] in ARMS for row in scored), "scoring",
            "unexpected arm reached the scorer")
    require(len({(row["arm_id"], row["case_id"]) for row in scored})
            == len(scored), "scoring", "scorer duplicated a prediction")

    passed = all(row["passed"] for row in gate_results.values())
    return {
        "format": SCORE_FORMAT,
        "protocol_id": PROTOCOL_ID,
        "status": "passed" if passed else "failed",
        "native_computation": False,
        "source_free": True,
        "answer_index_used": False,
        "tokenizer_binding_sha256": tokenizer_binding_sha(binding),
        "arms": metrics,
        "gains": gains,
        "gates": gate_results,
        "all_gates_passed": passed,
        "controls_use_identical_scorer": True,
        "target_conditioned_thresholds": False,
    }


# ---------------------------------------------------------------------------
# Build/validation/preflight CLI helpers


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

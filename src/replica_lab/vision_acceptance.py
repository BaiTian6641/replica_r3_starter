"""Fail-closed acceptance for a bounded vision-learning diagnostic.

This module evaluates evidence; it does not run a model, assign a label, or
promote an artifact.  A positive result requires all of the following:

* matched pre/post generated-answer improvement;
* explicit correct, wrong/sibling, shuffled and no-image control evidence;
* paired text-only retention without an undeclared decline; and
* a candidate-bound, zero-update fresh-process extension reload.

Missing evidence is ``inconclusive``.  Measured threshold failures are
``rejected``.  Neither result authorizes development, promotion or a retry.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
import math
from typing import Any


CONTROL_ALIASES = {
    "correct": ("correct",),
    "wrong": ("wrong", "sibling"),
    "shuffled": ("shuffled",),
    "noimage": ("noimage", "blank"),
}
EXPECTED_CONTROL_PROVENANCE = {
    "correct": frozenset(("correct_image",)),
    "wrong": frozenset(("wrong_image", "changed_sibling_image")),
    "shuffled": frozenset(("cross_pair_shuffled_image", "shuffled_image")),
    "noimage": frozenset(("no_image", "blank_image")),
}
THRESHOLD_NAMES = (
    "min_answer_accuracy_gain",
    "min_correct_margin_wrong",
    "min_correct_margin_shuffled",
    "min_correct_margin_noimage",
    "max_retention_decline",
)


class VisionAcceptanceError(ValueError):
    """Raised when the acceptance contract itself is malformed."""


def _mapping(value: Any) -> Mapping[str, Any] | None:
    return value if isinstance(value, Mapping) else None


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _case_ids(score: Mapping[str, Any]) -> tuple[str, ...] | None:
    value = score.get("case_ids")
    if isinstance(value, (list, tuple)) and value and all(
            isinstance(item, str) and item for item in value):
        return tuple(sorted(value))
    rows = score.get("rows")
    if isinstance(rows, list) and rows:
        identifiers = []
        for row in rows:
            if not isinstance(row, Mapping) or not isinstance(
                    row.get("case_id"), str) or not row["case_id"]:
                return None
            identifiers.append(row["case_id"])
        return tuple(sorted(set(identifiers)))
    return None


def _control(score: Mapping[str, Any], canonical: str) -> tuple[str, Mapping[str, Any]] | None:
    controls = _mapping(score.get("controls"))
    if controls is None:
        return None
    for name in CONTROL_ALIASES[canonical]:
        value = _mapping(controls.get(name))
        if value is not None:
            return name, value
    return None


def _paired(score: Mapping[str, Any], name: str) -> Mapping[str, Any] | None:
    paired = _mapping(score.get("paired"))
    return _mapping(paired.get(name)) if paired is not None else None


def _accuracy(score: Mapping[str, Any]) -> float | None:
    control = _control(score, "correct")
    if control is None:
        return None
    return _number(control[1].get("accuracy"))


def _thresholds(value: Mapping[str, Any]) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise VisionAcceptanceError("thresholds must be a mapping")
    result = {}
    for name in THRESHOLD_NAMES:
        number = _number(value.get(name))
        if number is None or number < 0:
            raise VisionAcceptanceError(f"invalid threshold: {name}")
        result[name] = number
    if result["min_answer_accuracy_gain"] <= 0:
        raise VisionAcceptanceError("answer improvement threshold must be positive")
    return result


def _bool_check(container: Mapping[str, Any], key: str, expected: bool,
                failures: list[str], missing: list[str], label: str) -> None:
    if key not in container:
        missing.append(f"{label}:{key}")
    elif container[key] is not expected:
        failures.append(f"{label}:{key}")


def evaluate_vision_acceptance(
        *, baseline_answer: Mapping[str, Any],
        candidate_answer: Mapping[str, Any],
        retention: Mapping[str, Any], reload_receipt: Mapping[str, Any],
        thresholds: Mapping[str, Any], candidate_report_sha256: str | None,
        required_reload_comparisons: Iterable[str] = (),
        expected_control_provenance: Mapping[str, Iterable[str]] | None = None,
        ) -> dict[str, Any]:
    """Return a complete, non-promoting acceptance assessment.

    The caller supplies the digest of the candidate report file.  The reload
    receipt must carry that same digest as ``run_report_sha256``; a bare reload
    success flag is therefore insufficient evidence.
    """
    limits = _thresholds(thresholds)
    failures: list[str] = []
    missing: list[str] = []
    metrics: dict[str, Any] = {}

    baseline = _mapping(baseline_answer)
    candidate = _mapping(candidate_answer)
    if baseline is None:
        missing.append("generated_answer:baseline")
        baseline = {}
    if candidate is None:
        missing.append("generated_answer:candidate")
        candidate = {}

    baseline_accuracy = _accuracy(baseline)
    candidate_accuracy = _accuracy(candidate)
    if baseline_accuracy is None:
        missing.append("generated_answer:baseline_accuracy")
    if candidate_accuracy is None:
        missing.append("generated_answer:candidate_accuracy")
    if baseline_accuracy is not None and candidate_accuracy is not None:
        gain = candidate_accuracy - baseline_accuracy
        metrics["baseline_answer_accuracy"] = baseline_accuracy
        metrics["candidate_answer_accuracy"] = candidate_accuracy
        metrics["answer_accuracy_gain"] = gain
        if gain < limits["min_answer_accuracy_gain"]:
            failures.append("generated_answer_improvement")

    baseline_ids = _case_ids(baseline)
    candidate_ids = _case_ids(candidate)
    if baseline_ids is None:
        missing.append("generated_answer:baseline_case_ids")
    if candidate_ids is None:
        missing.append("generated_answer:candidate_case_ids")
    if baseline_ids is not None and candidate_ids is not None:
        if baseline_ids != candidate_ids:
            failures.append("matched_answer_case_identity")
        metrics["matched_answer_cases"] = len(candidate_ids)

    correct = _control(candidate, "correct")
    if correct is None:
        missing.append("controls:correct")
    elif correct[1].get("available") is not True:
        failures.append("controls:correct_unavailable")

    provenance = _mapping(candidate.get("control_provenance"))
    if provenance is None:
        provenance = _mapping(candidate.get("image_roles"))
    expected = expected_control_provenance or EXPECTED_CONTROL_PROVENANCE
    margins: dict[str, float] = {}
    for canonical in ("wrong", "shuffled", "noimage"):
        control = _control(candidate, canonical)
        if control is None:
            missing.append(f"controls:{canonical}")
            continue
        name, values = control
        if values.get("available") is not True:
            failures.append(f"controls:{canonical}_unavailable")
        paired = _paired(candidate, name)
        margin = _number(paired.get("correct_margin")) if paired else None
        if margin is None:
            missing.append(f"controls:{canonical}_paired_margin")
        else:
            margins[canonical] = margin
            threshold = limits[f"min_correct_margin_{canonical}"]
            if margin < threshold:
                failures.append(f"correct_vs_{canonical}_margin")
        if provenance is None:
            missing.append("controls:provenance")
        else:
            role = provenance.get(name)
            allowed = frozenset(expected.get(canonical, ()))
            if role is None:
                missing.append(f"controls:{canonical}_provenance")
            elif not isinstance(role, str) or role not in allowed:
                failures.append(f"controls:{canonical}_provenance")
    if provenance is not None:
        correct_role = provenance.get("correct")
        if correct_role is None:
            missing.append("controls:correct_provenance")
        elif correct_role not in frozenset(expected.get("correct", ())):
            failures.append("controls:correct_provenance")
    metrics["correct_control_margins"] = margins

    retention_map = _mapping(retention)
    if retention_map is None:
        missing.append("text_retention:receipt")
        retention_map = {}
    text_only = retention_map.get("text_only") is True or (
        isinstance(retention_map.get("scope"), str)
        and "text retention only" in retention_map["scope"])
    if not text_only:
        missing.append("text_retention:text_only_scope")
    before = _mapping(retention_map.get("before"))
    after = _mapping(retention_map.get("after"))
    if before is None:
        missing.append("text_retention:before")
        before = {}
    if after is None:
        missing.append("text_retention:after")
        after = {}
    before_accuracy = _number(before.get("accuracy"))
    after_accuracy = _number(after.get("accuracy"))
    if before_accuracy is None:
        missing.append("text_retention:before_accuracy")
    if after_accuracy is None:
        missing.append("text_retention:after_accuracy")
    before_ids = _case_ids(before)
    after_ids = _case_ids(after)
    if before_ids is None:
        missing.append("text_retention:before_case_ids")
    if after_ids is None:
        missing.append("text_retention:after_case_ids")
    if before_ids is not None and after_ids is not None:
        if before_ids != after_ids:
            failures.append("text_retention:paired_case_identity")
        metrics["retention_cases"] = len(before_ids)
    before_cases = before.get("cases")
    after_cases = after.get("cases")
    if (not isinstance(before_cases, int) or isinstance(before_cases, bool)
            or before_cases <= 0):
        missing.append("text_retention:before_cases")
    if (not isinstance(after_cases, int) or isinstance(after_cases, bool)
            or after_cases != before_cases):
        failures.append("text_retention:case_count")
    if before_accuracy is not None and after_accuracy is not None:
        delta = after_accuracy - before_accuracy
        metrics["retention_before_accuracy"] = before_accuracy
        metrics["retention_after_accuracy"] = after_accuracy
        metrics["retention_accuracy_delta"] = delta
        if delta < -limits["max_retention_decline"]:
            failures.append("text_retention_decline")

    reload_map = _mapping(reload_receipt)
    if reload_map is None:
        missing.append("fresh_process_reload:receipt")
        reload_map = {}
    for key in ("fresh_process", "zero_updates", "extension_only", "base_unchanged"):
        _bool_check(reload_map, key, True, failures, missing,
                    "fresh_process_reload")
    _bool_check(reload_map, "base_saved", False, failures, missing,
                "fresh_process_reload")
    if reload_map.get("bitwise_equal") is not True:
        if "bitwise_equal" not in reload_map:
            missing.append("fresh_process_reload:bitwise_equal")
        else:
            failures.append("fresh_process_reload:bitwise_equal")
    comparisons = _mapping(reload_map.get("comparisons"))
    required = tuple(required_reload_comparisons)
    if not required:
        missing.append("fresh_process_reload:comparison_contract")
    if comparisons is None:
        missing.append("fresh_process_reload:comparisons")
        comparisons = {}
    for name in required:
        if name not in comparisons:
            missing.append(f"fresh_process_reload:comparison:{name}")
        elif comparisons[name] is not True:
            failures.append(f"fresh_process_reload:comparison:{name}")
    for name, equal in comparisons.items():
        if equal is not True:
            failures.append(f"fresh_process_reload:comparison:{name}")
    if not candidate_report_sha256:
        missing.append("fresh_process_reload:candidate_report_binding")
    elif reload_map.get("run_report_sha256") != candidate_report_sha256:
        failures.append("fresh_process_reload:candidate_report_binding")
    metrics["reload_comparisons"] = len(comparisons)
    metrics["fresh_process_reload_verified"] = not any(
        item.startswith("fresh_process_reload") for item in failures + missing)

    learned_failures = [item for item in failures + missing
                        if not item.startswith("text_retention")
                        and not item.startswith("fresh_process_reload")]
    engineering_failures = [item for item in failures + missing
                            if item.startswith("text_retention")
                            or item.startswith("fresh_process_reload")]
    status = ("inconclusive" if missing else
              "accepted" if not failures else "rejected")
    return {
        "format": "replica.vision_acceptance.assessment.v1",
        "status": status,
        "ok": status == "accepted",
        "diagnostic_only": True,
        "promotion": False,
        "development_authorized": False,
        "retry_authorized": False,
        "learned_evidence_ok": not learned_failures,
        "engineering_evidence_ok": not engineering_failures,
        "failures": failures,
        "missing_evidence": missing,
        "metrics": metrics,
        "claims": {
            "generated_answer_improvement_measured": "generated_answer_improvement" not in failures
            and "generated_answer:baseline_accuracy" not in missing
            and "generated_answer:candidate_accuracy" not in missing,
            "causal_image_dependence_measured": not any(
                item.startswith("controls:") or item.startswith("correct_vs_")
                for item in failures + missing),
            "text_retention_measured": not any(
                item.startswith("text_retention") for item in failures + missing),
            "fresh_process_reload_measured": metrics["fresh_process_reload_verified"],
            "native_execution": False,
            "calibration_or_generalization": False,
        },
    }


def answer_from_report(report: Mapping[str, Any], role: str = "dev") -> Mapping[str, Any]:
    """Extract a score mapping without inferring or manufacturing evidence."""
    if not isinstance(report, Mapping):
        raise VisionAcceptanceError("report must be a mapping")
    if isinstance(report.get("answer"), Mapping):
        return report["answer"]
    container = report.get(role)
    if isinstance(container, Mapping) and isinstance(container.get("answer"), Mapping):
        return container["answer"]
    raise VisionAcceptanceError(f"report has no {role}.answer score mapping")

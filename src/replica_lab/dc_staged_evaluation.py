"""Staged, sealed evaluation admission for the DC v2 studies.

DC-P1 v2 (and, by reuse, DC-R1 v2) may not look at a later stage's labels or
generation targets before an independent, digest-checked, tokenizer-backed
receipt admits the earlier stage:

    fit  ->  development  ->  final (one shot)

This module owns that gate.  It is standard-library only: it never imports
torch, never loads a model or a tokenizer, and never reads a fixture from a
hard-coded path.  Callers pass cases, labels, predictions and a
tokenizer-backed ``verify_decoded`` callback; the decode identity is therefore
part of the evidence rather than an assumption.

Nothing here creates an authorization, promotes a checkpoint or opens a
held-out split on its own.  A ``LabelVault`` refuses a stage unless a live
permit from :class:`StagedAdmission` is presented, and every refusal is recorded
by :class:`AccessSpy` so a test can prove later stages stayed closed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Callable, Mapping, Sequence

STAGE_FIT = "fit"
STAGE_DEVELOPMENT = "development"
STAGE_FINAL = "final"
STAGE_ORDER = (STAGE_FIT, STAGE_DEVELOPMENT, STAGE_FINAL)

RECEIPT_FORMAT = "replica.dc_stage_receipt.v2"
ADMISSION_FORMAT = "replica.dc_stage_admission.v2"
SCORE_REPORT_FORMAT = "replica.dc_stage_score_report.v2"
ACCESS_FORMAT = "replica.dc_stage_access_spy.v2"

# Gate names this module produces and consumes.  A protocol config must declare
# the matching thresholds; a missing threshold is a fail-closed error, never a
# default.
FIT_GATE_THRESHOLDS = {
    "treatment_fit_exact": "fit_exact_min",
    "treatment_fit_nonempty": "fit_nonempty_min",
}
DEV_ROLE_GATES = {
    "post_correction_current": "post_correction_accuracy_min",
    "new_instance_current": "new_instance_accuracy_min",
    "historical_state": "historical_accuracy_min",
    "abstention": "abstention_accuracy_min",
    "retention": "retention_accuracy_min",
}
DEV_GAIN_GATES = {
    "post_correction_current": "post_correction_min_gain",
    "new_instance_current": "new_instance_min_gain",
}
PRIMARY_ROLES = ("post_correction_current", "new_instance_current")
FINAL_ROLE_GATES = DEV_ROLE_GATES
ANSWER_TERM_GATES = ("all_arms_evidence", "treatment_arm_present",
                     "matched_control_weight_zero", "mask_audit_present",
                     "objective_claim_consistent", "decode_verified_per_arm")


class StageError(RuntimeError):
    """One fail-closed stage/admission violation."""

    def __init__(self, check: str, detail: str):
        super().__init__(f"{check}: {detail}")
        self.check = check
        self.detail = detail

    def as_dict(self) -> dict[str, str]:
        return {"check": self.check, "detail": self.detail}


def _require(condition: bool, check: str, detail: str) -> None:
    if not condition:
        raise StageError(check, detail)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


# ---------------------------------------------------------------------------
# Deterministic digests (stdlib only, identical rule to the protocol helper)


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def sha256_rows(rows: Sequence[Mapping[str, Any]]) -> str:
    return sha256_json([dict(row) for row in rows])


def is_hex64(value: Any) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def seal_record(body: Mapping[str, Any], *,
                digest_field: str = "receipt_sha256") -> dict:
    sealed = {key: value for key, value in body.items() if key != digest_field}
    sealed[digest_field] = sha256_json(sealed)
    return sealed


def verify_sealed_record(record: Any, *, digest_field: str = "receipt_sha256",
                        check: str = "seal") -> dict:
    _require(isinstance(record, Mapping), check, "record must be an object")
    digest = record.get(digest_field)
    _require(is_hex64(digest), check,
             f"{digest_field} must be a sealed sha256 digest")
    body = {key: value for key, value in record.items() if key != digest_field}
    _require(sha256_json(body) == digest, check,
             f"{digest_field} does not match the record body")
    return dict(record)


def load_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    _require(path.is_file(), "read", f"missing JSONL file: {path}")
    rows: list[dict] = []
    for number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        _require(isinstance(row, dict), "read",
                 f"{path}:{number} is not a JSON object")
        rows.append(row)
    return rows


def load_json(path: str | Path) -> dict:
    path = Path(path)
    _require(path.is_file(), "read", f"missing JSON file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), "read", f"{path} is not a JSON object")
    return value


def normalize_answer(value: Any) -> str:
    """strip_outer_whitespace_then_casefold_then_collapse_whitespace"""
    _require(isinstance(value, str), "answer", "answer must be a string")
    return " ".join(value.strip().casefold().split())


# ---------------------------------------------------------------------------
# Access spy and label vault


class AccessSpy:
    """Record every label/stage access attempt, granted or refused."""

    def __init__(self) -> None:
        self._events: list[dict[str, Any]] = []

    def record(self, access: str, *, stage: str, granted: bool,
               detail: str = "") -> None:
        _require(isinstance(access, str) and bool(access), "access",
                 "access name must be a nonempty string")
        self._events.append({
            "access": access,
            "stage": stage,
            "granted": bool(granted),
            "detail": detail,
        })

    @property
    def events(self) -> tuple[dict[str, Any], ...]:
        return tuple(dict(event) for event in self._events)

    @property
    def denied(self) -> tuple[dict[str, Any], ...]:
        return tuple(event for event in self._events if not event["granted"])

    @property
    def granted(self) -> tuple[dict[str, Any], ...]:
        return tuple(event for event in self._events if event["granted"])

    def count(self, *, access: str | None = None, stage: str | None = None,
              granted: bool | None = None) -> int:
        return sum(1 for event in self._events
                   if (access is None or event["access"] == access)
                   and (stage is None or event["stage"] == stage)
                   and (granted is None or event["granted"] is granted))

    def assert_clean(self) -> None:
        _require(not self.denied, "access_spy",
                 f"unauthorized access attempts were recorded: "
                 f"{[event['access'] for event in self.denied]}")

    def state(self) -> dict[str, Any]:
        return {
            "format": ACCESS_FORMAT,
            "granted": len(self.granted),
            "denied": len(self.denied),
            "stages_granted": sorted({event["stage"]
                                      for event in self.granted}),
            "events": list(self.events),
        }


@dataclass(frozen=True)
class AccessPermit:
    """A live, in-process permit to read exactly one stage's labels."""

    stage: str
    attempt_id: str
    receipt_sha256: str
    _nonce: object = field(default_factory=object, repr=False, compare=False)

    def as_dict(self) -> dict[str, str]:
        return {
            "stage": self.stage,
            "attempt_id": self.attempt_id,
            "receipt_sha256": self.receipt_sha256,
        }


class LabelVault:
    """Per-stage labels that only open with a live admission permit."""

    def __init__(self, labels_by_stage: Mapping[str, Any], *,
                 spy: AccessSpy | None = None) -> None:
        _require(isinstance(labels_by_stage, Mapping), "vault",
                 "labels_by_stage must be a mapping")
        self._spy = spy if spy is not None else AccessSpy()
        self._sources: dict[str, Any] = {}
        self._labels: dict[str, tuple[dict[str, Any], ...]] = {}
        for stage, source in labels_by_stage.items():
            _require(stage in STAGE_ORDER, "vault", f"unknown stage {stage!r}")
            self._sources[stage] = source
        for stage, source in self._sources.items():
            if not callable(source):
                self._load_rows(stage)
        self._opened: set[str] = set()

    def _load_rows(self, stage: str) -> tuple[dict[str, Any], ...]:
        if stage in self._labels:
            return self._labels[stage]
        _require(stage in self._sources, "vault", f"no labels for {stage!r}")
        source = self._sources[stage]
        rows = source() if callable(source) else source
        _require(isinstance(rows, Sequence) and not isinstance(rows, str),
                 "vault", f"labels for {stage!r} must be a sequence")
        ids = []
        copied = []
        for row in rows:
            _require(isinstance(row, Mapping), "vault",
                     "label rows must be mappings")
            case_id = row.get("case_id")
            _require(isinstance(case_id, str) and case_id, "vault",
                     "label row is missing case_id")
            _require(case_id not in ids, "vault",
                     f"duplicate label id {case_id!r} in {stage!r}")
            ids.append(case_id)
            copied.append(dict(row))
        self._labels[stage] = tuple(copied)
        return self._labels[stage]

    def _metadata(self, stage: str, permit: AccessPermit | None) -> tuple[
            dict[str, Any], ...]:
        if stage not in self._labels and callable(self._sources.get(stage)):
            return self.open(stage, permit=permit)
        return self._load_rows(stage)

    @property
    def spy(self) -> AccessSpy:
        return self._spy

    @property
    def stages(self) -> tuple[str, ...]:
        return tuple(stage for stage in STAGE_ORDER if stage in self._sources)

    @property
    def opened(self) -> tuple[str, ...]:
        return tuple(stage for stage in STAGE_ORDER if stage in self._opened)

    def stage_ids(self, stage: str, *, permit: AccessPermit | None = None
                  ) -> tuple[str, ...]:
        """Case ids; lazy sources require the same permit as label reads."""
        rows = self._metadata(stage, permit)
        return tuple(row["case_id"] for row in rows)

    def stage_digest(self, stage: str, *, permit: AccessPermit | None = None
                     ) -> str:
        """Digest of sealed labels; lazy sources require admission first."""
        return sha256_rows(self._metadata(stage, permit))

    def open(self, stage: str, *, permit: AccessPermit | None) -> tuple[
            dict[str, Any], ...]:
        try:
            _require(stage in self._sources, "label_access",
                     f"no sealed labels for stage {stage!r}")
            _require(permit is not None, "label_access",
                     f"stage {stage!r} labels require an admission permit")
            _require(isinstance(permit, AccessPermit), "label_access",
                     "permit must be an AccessPermit")
            _require(permit.stage == stage, "label_access",
                     f"permit for {permit.stage!r} cannot open {stage!r} labels")
            rows = self._load_rows(stage)
        except StageError as error:
            self._spy.record("labels", stage=stage, granted=False,
                             detail=error.detail)
            raise
        self._opened.add(stage)
        self._spy.record("labels", stage=stage, granted=True,
                         detail="stage label read")
        return tuple(dict(row) for row in rows)


# ---------------------------------------------------------------------------
# Receipts


@dataclass(frozen=True)
class ReceiptExpectation:
    """The exact identity every stage receipt must bind."""

    protocol_id: str
    config_sha256: str
    manifest_sha256: str
    protocol_commit: str
    tokenizer_binding_sha256: str
    decode_policy_sha256: str
    arms: tuple[str, ...]
    cases_expected: Mapping[str, int]
    receipt_format: str = RECEIPT_FORMAT

    def __post_init__(self) -> None:
        for name in ("protocol_id", "protocol_commit"):
            value = getattr(self, name)
            _require(isinstance(value, str) and bool(value), "expectation",
                     f"{name} must be a nonempty string")
        for name in ("config_sha256", "manifest_sha256",
                     "tokenizer_binding_sha256", "decode_policy_sha256"):
            _require(is_hex64(getattr(self, name)), "expectation",
                     f"{name} must be a sealed sha256 digest")
        _require(bool(self.arms) and all(isinstance(arm, str) and arm
                                        for arm in self.arms),
                 "expectation", "arms must be a nonempty string tuple")
        _require(len(set(self.arms)) == len(self.arms), "expectation",
                 "arms must be distinct")
        _require(isinstance(self.cases_expected, Mapping)
                 and set(self.cases_expected) == set(STAGE_ORDER),
                 "expectation",
                 "cases_expected must declare a count for every stage")

    def as_dict(self) -> dict[str, Any]:
        return {
            "protocol_id": self.protocol_id,
            "config_sha256": self.config_sha256,
            "manifest_sha256": self.manifest_sha256,
            "protocol_commit": self.protocol_commit,
            "tokenizer_binding_sha256": self.tokenizer_binding_sha256,
            "decode_policy_sha256": self.decode_policy_sha256,
            "arms": list(self.arms),
            "cases_expected": dict(self.cases_expected),
            "receipt_format": self.receipt_format,
        }


def build_stage_receipt(*, report: Mapping[str, Any],
                        expectation: ReceiptExpectation,
                        attempt_id: str) -> dict:
    """Seal one stage's score report into an independent receipt."""
    _require(isinstance(report, Mapping), "receipt", "report must be an object")
    stage = report.get("stage")
    _require(stage in STAGE_ORDER, "receipt", f"unknown report stage {stage!r}")
    _require(isinstance(attempt_id, str) and bool(attempt_id.strip()),
             "receipt", "attempt_id must be a nonempty string")
    arms = report.get("arms")
    _require(isinstance(arms, Mapping), "receipt", "report arms missing")
    expected_arms = (expectation.arms if stage != STAGE_FINAL
                     else (STAGE_FINAL_TREATMENT_ARM,))
    _require(set(arms) == set(expected_arms), "receipt",
             f"report arms {sorted(arms)} differ from expected "
             f"{sorted(expected_arms)}")
    seam: dict[str, Any] = {}
    for arm, metrics in arms.items():
        _require(isinstance(metrics, Mapping), "receipt",
                 f"arm {arm!r} metrics must be an object")
        seam[arm] = {
            "cases": metrics.get("cases"),
            "exact_accuracy": metrics.get("exact_accuracy"),
            "checkpoint_id": metrics.get("checkpoint_id"),
            "answer_span_ce_weight": metrics.get("answer_span_ce_weight"),
            "decode_verified": metrics.get("decode_verified"),
            "prediction_sha256": metrics.get("prediction_sha256"),
        }
    gates = report.get("gates")
    _require(isinstance(gates, Mapping) and bool(gates), "receipt",
             "report gates must be a nonempty object")
    body = {
        "format": expectation.receipt_format,
        "stage": stage,
        "protocol_id": expectation.protocol_id,
        "config_sha256": expectation.config_sha256,
        "manifest_sha256": expectation.manifest_sha256,
        "protocol_commit": expectation.protocol_commit,
        "tokenizer_binding_sha256": expectation.tokenizer_binding_sha256,
        "decode_policy_sha256": expectation.decode_policy_sha256,
        "attempt_id": attempt_id,
        "status": report.get("status"),
        "arms": seam,
        "cases": {"stage": stage, "count": report["cases"]["count"]},
        "evidence_counts": dict(report.get("cases", {})),
        "gates": json.loads(json.dumps(gates)),
        "objective_audit": json.loads(json.dumps(
            report.get("objective_audit"))) if report.get(
                "objective_audit") is not None else None,
        "evidence_class": report.get("evidence_class", "cpu_fixture"),
        "native_execution_admissible": bool(
            report.get("native_execution_admissible", False)),
        "score_report_sha256": sha256_json(
            {key: value for key, value in report.items()
             if key != "report_sha256"}),
    }
    return seal_record(body)


STAGE_FINAL_TREATMENT_ARM = "treatment"


def verify_stage_receipt(receipt: Any, *, stage: str,
                         expectation: ReceiptExpectation) -> dict:
    """Fail closed unless the receipt binds exactly this stage identity."""
    _require(stage in STAGE_ORDER, "receipt", f"unknown stage {stage!r}")
    sealed = verify_sealed_record(receipt, check="receipt")
    _require(sealed.get("format") == expectation.receipt_format, "receipt",
             "receipt format differs")
    _require(sealed.get("stage") == stage, "receipt",
             f"receipt stage {sealed.get('stage')!r} is not {stage!r}")
    for key, expected in (
            ("protocol_id", expectation.protocol_id),
            ("config_sha256", expectation.config_sha256),
            ("manifest_sha256", expectation.manifest_sha256),
            ("protocol_commit", expectation.protocol_commit),
            ("tokenizer_binding_sha256",
             expectation.tokenizer_binding_sha256),
            ("decode_policy_sha256", expectation.decode_policy_sha256)):
        _require(sealed.get(key) == expected, "receipt",
                 f"receipt {key} differs from the admitted identity")
    _require(sealed.get("status") in ("passed", "failed"), "receipt",
             "receipt status must be passed or failed")
    arms = sealed.get("arms")
    _require(isinstance(arms, Mapping), "receipt", "receipt arms missing")
    expected_arms = (expectation.arms if stage != STAGE_FINAL
                     else (STAGE_FINAL_TREATMENT_ARM,))
    _require(set(arms) == set(expected_arms), "receipt",
             f"receipt arms {sorted(arms)} differ from expected "
             f"{sorted(expected_arms)}")
    expected_cases = expectation.cases_expected[stage]
    _require(sealed.get("cases") == {"stage": stage,
                                     "count": expected_cases},
             "receipt",
             f"receipt case coverage differs from {expected_cases}")
    evidence_counts = sealed.get("evidence_counts")
    _require(isinstance(evidence_counts, Mapping), "receipt",
             "receipt prediction coverage is missing")
    _require(evidence_counts.get("stage") == stage
             and evidence_counts.get("count") == expected_cases
             and evidence_counts.get("arms") == len(expected_arms)
             and evidence_counts.get("predictions")
             == expected_cases * len(expected_arms), "receipt",
             "receipt raw prediction coverage differs from the stage schedule")
    for arm, row in arms.items():
        _require(row.get("cases") == expected_cases, "receipt",
                 f"arm {arm!r} case coverage differs from {expected_cases}")
        _require(is_hex64(row.get("prediction_sha256")), "receipt",
                 f"arm {arm!r} prediction digest is missing")
    gates = sealed.get("gates")
    _require(sealed.get("evidence_class") in ("native_mps", "cpu_fixture"),
             "receipt", "receipt evidence class differs")
    _require(isinstance(sealed.get("native_execution_admissible"), bool),
             "receipt", "native execution admissibility must be boolean")
    _require(isinstance(gates, Mapping) and bool(gates), "receipt",
             "receipt gates missing")
    for name, row in gates.items():
        _require(isinstance(row, Mapping)
                 and isinstance(row.get("passed"), bool), "receipt",
                 f"gate {name!r} must carry a boolean passed flag")
    if sealed["status"] == "passed":
        _require(all(row["passed"] for row in gates.values()), "receipt",
                 "a passed receipt carries a failed gate")
    attempt = sealed.get("attempt_id")
    _require(isinstance(attempt, str) and bool(attempt.strip()), "receipt",
             "receipt attempt_id must be a nonempty string")
    _require(is_hex64(sealed.get("score_report_sha256")), "receipt",
             "receipt must bind the score report digest")
    return sealed


def require_prior_report(report: Any, *, required_format: str,
                        protocol_id: str, stage: str | None = None,
                        statuses: Sequence[str] = ("passed",),
                        round_trip_checks: Sequence[str] = (),
                        check: str = "report_admission") -> dict:
    """Reusable "an earlier independent report must exist and pass" gate."""
    _require(isinstance(report, Mapping), check, "report must be an object")
    _require(report.get("format") == required_format, check,
             "report format differs")
    _require(report.get("protocol_id") == protocol_id, check,
             "report protocol differs")
    if stage is not None:
        _require(report.get("stage") == stage, check,
                 f"report stage {report.get('stage')!r} is not {stage!r}")
    _require(report.get("status") in tuple(statuses), check,
             f"report status {report.get('status')!r} is not admitted")
    attempt = report.get("attempt_id")
    _require(isinstance(attempt, str) and bool(attempt.strip()), check,
             "report attempt_id must be a nonempty string")
    for key in round_trip_checks:
        _require(key in report, check, f"report is missing {key}")
    if "report_sha256" in report:
        verify_sealed_record(report, digest_field="report_sha256", check=check)
    gates = report.get("gates")
    if gates is not None:
        _require(isinstance(gates, Mapping) and bool(gates), check,
                 "report gates must be a nonempty object")
        for name, row in gates.items():
            _require(isinstance(row, Mapping)
                     and isinstance(row.get("passed"), bool), check,
                     f"gate {name!r} must carry a boolean passed flag")
        if tuple(statuses) == ("passed",):
            _require(all(row["passed"] for row in gates.values()), check,
                     "a passed report carries a failed gate")
    return dict(report)


# ---------------------------------------------------------------------------
# Admission


def _required_gate_names(stage: str, *, arms: Sequence[str],
                         control_arms: Sequence[str]) -> tuple[str, ...]:
    if stage == STAGE_FIT:
        return tuple(FIT_GATE_THRESHOLDS) + ANSWER_TERM_GATES
    if stage == STAGE_DEVELOPMENT:
        names = list(ANSWER_TERM_GATES)
        for role in PRIMARY_ROLES:
            names.append(f"treatment_{role}_accuracy")
            for control in control_arms:
                names.append(f"{role}_gain_vs_{control}")
                names.append(f"{control}_{role}_control_ceiling")
        for role in ("historical_state", "abstention", "retention"):
            names.append(f"treatment_{role}_accuracy")
        names.append("abstention_false_recall")
        names.append("retention_not_degraded_vs_matched")
        return tuple(names)
    names = [f"treatment_final_{role}_accuracy"
             for role in FINAL_ROLE_GATES]
    names += ["final_fit_exact", "final_fit_nonempty",
              "final_confirmation_only", "mask_audit_present"]
    return tuple(names)


class StagedAdmission:
    """Ordered, one-shot stage admission backed by independent receipts."""

    def __init__(self, expectation: ReceiptExpectation, *,
                 spy: AccessSpy | None = None,
                 control_arms: Sequence[str] = (
                     "matched_trained", "no_update", "reset_P"),
                 receipt_verifier: Callable[..., dict] | None = None,
                 final_started: bool = False) -> None:
        self._expectation = expectation
        self._spy = spy if spy is not None else AccessSpy()
        self._control_arms = tuple(control_arms)
        self._verify = receipt_verifier or verify_stage_receipt
        self._admitted: dict[str, dict[str, Any]] = {}
        self._attempts: dict[str, str] = {}
        self._final_consumed = False
        self._final_started = bool(final_started)

    @property
    def spy(self) -> AccessSpy:
        return self._spy

    @property
    def fit_admitted(self) -> bool:
        return STAGE_FIT in self._admitted

    @property
    def development_admitted(self) -> bool:
        return STAGE_DEVELOPMENT in self._admitted

    @property
    def final_consumed(self) -> bool:
        return self._final_consumed
    @property
    def final_started(self) -> bool:
        return self._final_started


    @staticmethod
    def _coerce(receipt: Any) -> Any:
        """Accept a receipt mapping or a path to a sealed receipt file."""
        if isinstance(receipt, (str, Path)):
            return load_json(receipt)
        return receipt

    def _verify_stage(self, receipt: Any, stage: str) -> dict:
        sealed = self._verify(self._coerce(receipt), stage=stage,
                              expectation=self._expectation)
        attempt = sealed["attempt_id"]
        _require(attempt not in self._attempts.values(), "receipt",
                 f"attempt id {attempt!r} was already used for another stage")
        return sealed

    def _check_gates(self, sealed: Mapping[str, Any], stage: str) -> None:
        required = _required_gate_names(
            stage, arms=self._expectation.arms,
            control_arms=self._control_arms)
        gates = sealed["gates"]
        for name in required:
            _require(name in gates, "stage_gate",
                     f"{stage} receipt is missing required gate {name!r}")
        failed = [name for name in required if not gates[name]["passed"]]
        _require(not failed, "stage_gate",
                 f"{stage} gates did not pass: {failed}")
        _require(sealed["status"] == "passed", "stage_gate",
                 f"{stage} receipt status is {sealed['status']!r}")

    def admit_fit(self, receipt: Any) -> dict:
        sealed = self._verify_stage(receipt, STAGE_FIT)
        try:
            self._check_gates(sealed, STAGE_FIT)
            arms = sealed["arms"]
            for arm in self._expectation.arms:
                _require(arms[arm].get("cases") is not None
                         and arms[arm].get("decode_verified") is not None,
                         "fit_admission",
                         f"arm {arm!r} carries no fit evidence")
            treatment = arms[STAGE_FINAL_TREATMENT_ARM]
            _require(treatment.get("answer_span_ce_weight") == 0.5,
                     "fit_admission",
                     "the treatment arm must keep answer_span_ce_weight 0.5")
            matched = arms.get("matched_trained")
            _require(matched is not None
                     and matched.get("answer_span_ce_weight") == 0.0,
                     "fit_admission",
                     "the matched trained control must declare weight 0.0")
        except StageError as error:
            self._spy.record("admit_fit", stage=STAGE_FIT, granted=False,
                             detail=error.detail)
            raise
        self._admitted[STAGE_FIT] = sealed
        self._attempts[STAGE_FIT] = sealed["attempt_id"]
        self._spy.record("admit_fit", stage=STAGE_FIT, granted=True,
                         detail="fit evidence admitted")
        return sealed

    def admit_development(self, receipt: Any) -> dict:
        sealed = self._verify_stage(receipt, STAGE_DEVELOPMENT)
        try:
            _require(self.fit_admitted, "stage_order",
                     "development cannot be admitted before fit")
            self._check_gates(sealed, STAGE_DEVELOPMENT)
            arms = sealed["arms"]
            matched = arms.get("matched_trained")
            _require(matched is not None
                     and matched.get("answer_span_ce_weight") == 0.0,
                     "development_admission",
                     "the matched 0.0 control must appear in development")
            for control in self._control_arms:
                _require(control in arms, "development_admission",
                         f"control arm {control!r} is missing from development")
        except StageError as error:
            self._spy.record("admit_development", stage=STAGE_DEVELOPMENT,
                             granted=False, detail=error.detail)
            raise
        self._admitted[STAGE_DEVELOPMENT] = sealed
        self._attempts[STAGE_DEVELOPMENT] = sealed["attempt_id"]
        self._spy.record("admit_development", stage=STAGE_DEVELOPMENT,
                        granted=True,
                        detail="development comparison admitted")
        return sealed

    def admit_final(self, receipt: Any) -> dict:
        if self._final_consumed:
            error = StageError("final_one_shot",
                               "the final stage is one-shot and already consumed")
            self._spy.record("admit_final", stage=STAGE_FINAL, granted=False,
                             detail=error.detail)
            raise error
        sealed = self._verify_stage(receipt, STAGE_FINAL)
        try:
            _require(self.development_admitted, "stage_order",
                     "final cannot be admitted before development")
            _require(self._final_started, "final_one_shot",
                     "a final receipt without a started final attempt is "
                     "not admissible")
            _require(not self._final_consumed, "final_one_shot",
                     "the final stage is one-shot and already consumed")
            self._check_gates(sealed, STAGE_FINAL)
            _require(sealed["attempt_id"] not in self._attempts.values(),
                     "final_one_shot",
                     "final must use a fresh attempt id")
        except StageError as error:
            self._spy.record("admit_final", stage=STAGE_FINAL, granted=False,
                             detail=error.detail)
            raise
        self._admitted[STAGE_FINAL] = sealed
        self._attempts[STAGE_FINAL] = sealed["attempt_id"]
        self._final_consumed = True
        self._spy.record("admit_final", stage=STAGE_FINAL, granted=True,
                         detail="final confirmation admitted once")
        return sealed

    def require(self, stage: str) -> AccessPermit:
        """Open exactly the stage whose prerequisites are admitted.

        ``fit`` needs no predecessor; ``development`` needs fit admitted;
        ``final`` needs development admitted and can be started only once, so
        the one-shot rule bites before any final label is read.
        """
        _require(stage in STAGE_ORDER, "stage_order", f"unknown stage {stage!r}")
        if stage == STAGE_FIT:
            _require(not self._admitted, "stage_order",
                     "fit labels may only be opened before any admission; a "
                     "later fit label read would be a replay")
            self._spy.record("permit", stage=stage, granted=True,
                             detail="first-stage fit label read for its own "
                                    "receipt")
            return AccessPermit(
                stage=stage, attempt_id="bootstrap:fit",
                receipt_sha256=sha256_json({"bootstrap": stage,
                                            "protocol_id":
                                                self._expectation.protocol_id}))
        if stage == STAGE_DEVELOPMENT:
            granted = self.fit_admitted
            self._spy.record("permit", stage=stage, granted=granted,
                             detail="development label read")
            _require(granted, "stage_order",
                     "development labels require an admitted fit receipt")
            receipt = self._admitted[STAGE_FIT]
            return AccessPermit(stage=stage, attempt_id=receipt["attempt_id"],
                                receipt_sha256=receipt["receipt_sha256"])
        granted = self.development_admitted and not self._final_started
        self._spy.record("permit", stage=stage, granted=granted,
                         detail="final label read")
        _require(self.development_admitted, "stage_order",
                 "final work requires an admitted development receipt")
        _require(not self._final_started, "final_one_shot",
                 "the final stage has already been started; it is one-shot")
        self._final_started = True
        receipt = self._admitted[STAGE_DEVELOPMENT]
        return AccessPermit(stage=stage, attempt_id=receipt["attempt_id"],
                            receipt_sha256=receipt["receipt_sha256"])

    def require_closed(self, stage: str) -> None:
        """Assert a stage is still closed; used by ordering tests/guards."""
        _require(stage not in self._admitted, "stage_order",
                 f"stage {stage!r} is already admitted")
        self._spy.record("require_closed", stage=stage, granted=True,
                         detail="stage is closed as declared")

    def state(self) -> dict[str, Any]:
        return {
            "format": ADMISSION_FORMAT,
            "protocol_id": self._expectation.protocol_id,
            "stages": {
                stage: {
                    "admitted": stage in self._admitted,
                    "attempt_id": self._attempts.get(stage),
                    "receipt_sha256": (self._admitted.get(stage) or {}).get(
                        "receipt_sha256"),
                }
                for stage in STAGE_ORDER
            },
            "final_started": self._final_started,
            "final_consumed": self._final_consumed,
            "access": self._spy.state(),
        }


# ---------------------------------------------------------------------------
# Per-stage scoring entry points


def _role_rows(rows: Sequence[Mapping[str, Any]], role: str) -> list[dict]:
    return [row for row in rows if row["role"] == role]


def _accuracy(rows: Sequence[Mapping[str, Any]]) -> float | None:
    if not rows:
        return None
    return sum(1.0 for row in rows if row["correct"]) / len(rows)


def _group_accuracy(rows: Sequence[Mapping[str, Any]],
                    role: str) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        if row["role"] != role:
            continue
        grouped.setdefault(row["source_group"], []).append(
            1.0 if row["correct"] else 0.0)
    return {group: sum(values) / len(values)
            for group, values in sorted(grouped.items())}


def _paired_bootstrap_ci(differences: Sequence[float], *, seed: int,
                         resamples: int, alpha: float = 0.05) -> dict:
    _require(bool(differences), "paired_bootstrap", "no paired differences")
    count = len(differences)
    if count == 1:
        value = float(differences[0])
        return {"mean": value, "low": value, "high": value,
                "resamples": 1, "alpha": alpha}
    rng = random.Random(seed)
    samples = []
    for _ in range(resamples):
        total = 0.0
        for _ in range(count):
            total += float(differences[rng.randrange(count)])
        samples.append(total / count)
    samples.sort()
    low_index = max(0, int((alpha / 2.0) * resamples) - 1)
    high_index = min(resamples - 1, int((1.0 - alpha / 2.0) * resamples))
    return {
        "mean": sum(float(value) for value in differences) / count,
        "low": samples[low_index],
        "high": samples[high_index],
        "resamples": resamples,
        "alpha": alpha,
    }


def _case_index(cases: Sequence[Mapping[str, Any]],
                labels: Sequence[Mapping[str, Any]], *, stage: str) -> tuple[
        dict[str, dict], dict[str, dict]]:
    case_by_id: dict[str, dict] = {}
    for case in cases:
        case_id = case.get("case_id")
        _require(isinstance(case_id, str) and case_id, "scoring",
                 "case is missing case_id")
        _require(case_id not in case_by_id, "scoring",
                 f"duplicate case id {case_id!r}")
        expected_splits = {
            STAGE_FIT: {"fit"},
            STAGE_DEVELOPMENT: {"dev"},
            STAGE_FINAL: {"final_fit", "sealed_final"},
        }
        _require(case.get("split_role") in expected_splits[stage], "scoring",
                 f"case {case_id!r} split_role is not valid for {stage!r}")
        _require(case.get("inference_inputs") == ["prompt"], "scoring",
                 f"case {case_id!r} must expose only the prompt")
        case_by_id[case_id] = dict(case)
    label_by_id: dict[str, dict] = {}
    for label in labels:
        case_id = label.get("case_id")
        _require(isinstance(case_id, str) and case_id, "scoring",
                 "label is missing case_id")
        _require(case_id not in label_by_id, "scoring",
                 f"duplicate label id {case_id!r}")
        _require(label.get("split_role") in expected_splits[stage], "scoring",
                 f"label {case_id!r} split_role is not valid for {stage!r}")
        label_by_id[case_id] = dict(label)
    _require(set(case_by_id) == set(label_by_id), "scoring",
             "case and label coverage differ")
    return case_by_id, label_by_id


def _verify_prediction_identity(
        prediction: Mapping[str, Any], *, stage: str, protocol_id: str,
        arm_spec: Mapping[str, Any], binding: Mapping[str, Any],
        binding_sha256: str, case: Mapping[str, Any]) -> None:
    _require(prediction.get("protocol_id") == protocol_id, "prediction",
             "prediction protocol differs")
    _require(prediction.get("stage") == stage, "prediction",
             f"prediction stage is {prediction.get('stage')!r}, not {stage!r}")
    _require(prediction.get("arm_id") == arm_spec["arm_id"], "prediction",
             "prediction arm differs from the scored arm")
    _require(prediction.get("checkpoint_id")
             == arm_spec.get("checkpoint_id"), "prediction",
             f"arm {arm_spec['arm_id']!r} prediction bound to checkpoint "
             f"{prediction.get('checkpoint_id')!r}, expected "
             f"{arm_spec.get('checkpoint_id')!r}")
    _require(prediction.get("case_id") == case["case_id"], "prediction",
             "prediction case differs")
    _require(prediction.get("prompt") == case["prompt"], "case_binding",
             "prediction prompt differs from the sealed case")
    _require(prediction.get("prompt_sha256") == case["prompt_sha256"],
             "case_binding", "prediction prompt hash differs")
    _require(prediction.get("tokenizer_binding_sha256") == binding_sha256,
             "prediction",
             "prediction tokenizer binding differs from the sealed binding")
    _require(prediction.get("decode_policy") == binding["decode_policy"],
             "prediction",
             "prediction decode policy differs from the sealed policy")
    for key in ("query_epoch", "resolution_mode"):
        _require(prediction.get(key) == case.get(key), "temporal_binding",
                 f"prediction {key} differs from the sealed case")
    for key, expected in (("source_context_present", False),
                          ("source_kv_loaded", False),
                          ("source_prefill_tokens", 0),
                          ("source_kv_reused", False),
                          ("prompt_source_spans", 0)):
        _require(prediction.get(key) == expected, "source_free_evaluation",
                 f"{key}={prediction.get(key)!r}; expected {expected!r}")
    forbidden = ("answer", "answer_index", "gold", "gold_index", "target",
                 "target_index", "value", "value_index", "label",
                 "target_token", "answer_token")
    for key in forbidden:
        _require(key not in prediction, "answer_shortcut",
                 f"answer-bearing field {key!r} is forbidden in a prediction")
    routing = prediction.get("routing_metadata", {})
    _require(isinstance(routing, Mapping), "routing_metadata",
             "routing_metadata must be an object")
    for key in routing:
        _require(str(key).casefold() not in set(forbidden)
                 | {"world_answer", "record_answer", "expected", "correct"},
                 "routing_metadata",
                 f"answer-bearing routing field {key!r} is forbidden")


def _score_rows(*, stage: str, protocol_id: str, cases: Sequence[Mapping[str, Any]],
                labels: Sequence[Mapping[str, Any]],
                predictions: Sequence[Mapping[str, Any]],
                arm_specs: Sequence[Mapping[str, Any]],
                binding: Mapping[str, Any], binding_sha256: str,
                decode_fn: Callable[[Sequence[int]], str],
                verify_decoded: Callable[..., str],
                normalize: Callable[[str], str]) -> tuple[
        dict[str, list[dict]], dict[str, int]]:
    _require(callable(verify_decoded), "scoring",
             "a tokenizer-backed verify_decoded callback is required")
    _require(callable(decode_fn), "scoring", "a decode function is required")
    case_by_id, label_by_id = _case_index(cases, labels, stage=stage)
    specs = {spec["arm_id"]: dict(spec) for spec in arm_specs}
    _require(len(specs) == len(arm_specs), "scoring", "duplicate arm spec")
    seen: set[tuple[str, str]] = set()
    scored: dict[str, list[dict]] = {arm: [] for arm in specs}
    raw_predictions: dict[str, list[dict]] = {arm: [] for arm in specs}
    for prediction in predictions:
        arm = prediction.get("arm_id")
        _require(arm in specs, "scoring", f"unexpected arm {arm!r}")
        case_id = prediction.get("case_id")
        _require(case_id in case_by_id, "scoring",
                 f"unknown case {case_id!r}")
        key = (arm, case_id)
        _require(key not in seen, "scoring", f"duplicate prediction {key}")
        seen.add(key)
        case = case_by_id[case_id]
        label = label_by_id[case_id]
        _verify_prediction_identity(
            prediction, stage=stage, protocol_id=protocol_id,
            arm_spec=specs[arm], binding=binding,
            binding_sha256=binding_sha256, case=case)
        raw_predictions[arm].append(dict(prediction))
        text = verify_decoded(prediction, binding, decode_fn)
        target = label["target"]
        for key in ("query_epoch", "resolution_mode"):
            _require(label.get(key) == case.get(key), "temporal_binding",
                     f"label {key} differs from the sealed case")
        _require(prediction.get("resolved_event_position")
                 == label.get("resolved_event_position"), "temporal_binding",
                 "prediction resolved event differs from the sealed label")
        if case.get("resolution_mode") == "historical_explicit_event":
            _require(label.get("resolved_event_position")
                     == case.get("historical_event_position"),
                     "temporal_binding",
                     "historical query did not resolve its explicit event")
        elif case.get("resolution_mode") == "current_at_query_epoch":
            resolved = label.get("resolved_event_position")
            _require(_is_int(resolved) and resolved <= case["query_epoch"],
                     "temporal_binding",
                     "current query resolved a future or invalid event")
        normalized = normalize(text)
        expected = normalize(target)
        stopped = prediction.get("stopped") is True
        stop_reason = prediction.get("stop_reason")
        completion_valid = stopped and stop_reason not in {
            "truncated", "max_new_tokens", "length", "budget"}
        scored[arm].append({
            "case_id": case_id,
            "arm_id": arm,
            "role": case["role"],
            "source_group": case["source_group"],
            "target": target,
            "generated_text": text,
            "normalized_generated_text": normalized,
            "correct": completion_valid and normalized == expected,
            "abstained": normalized == "unknown",
            "empty_generation": text == "",
            "stopped": stopped,
            "stop_reason": stop_reason,
            "completion_valid": completion_valid,
            "query_epoch": case.get("query_epoch"),
            "resolution_mode": case.get("resolution_mode"),
            "resolved_event_position": label.get("resolved_event_position"),
            "checkpoint_id": prediction.get("checkpoint_id"),
        })
    expected_keys = {(arm, case_id) for arm in specs for case_id in case_by_id}
    missing = sorted(expected_keys - seen)
    _require(not missing, "scoring",
             f"missing predictions: {missing[:8]}"
             + ("" if len(missing) <= 8 else f" (+{len(missing) - 8} more)"))
    counts = {
        "stage": stage,
        "count": len(case_by_id),
        "arms": len(specs),
        "predictions": len(seen),
        "split_role_counts": {
            role: sum(1 for case in case_by_id.values()
                      if case["split_role"] == role)
            for role in sorted({case["split_role"]
                                for case in case_by_id.values()})},
        "prediction_sha256_by_arm": {
            arm: sha256_rows(raw_predictions[arm]) for arm in specs},
    }
    return scored, counts


def _arm_metrics(rows: Sequence[Mapping[str, Any]],
                 spec: Mapping[str, Any], *,
                 roles: Sequence[str]) -> dict[str, Any]:
    fit_rows = [row for row in rows if row["role"].startswith("fit_")]
    dev_rows = [row for row in rows if not row["role"].startswith("fit_")]
    return {
        "arm_id": spec["arm_id"],
        "cases": len(rows),
        "exact_accuracy": _accuracy(rows),
        "nonempty_rate": (
            sum(1.0 for row in rows if not row["empty_generation"]) / len(rows)
            if rows else None),
        "stop_rate": (
            sum(1.0 for row in rows if row["stopped"]) / len(rows)
            if rows else None),
        "decode_verified": len(rows),
        "checkpoint_id": spec.get("checkpoint_id"),
        "prediction_sha256": spec.get("prediction_sha256"),
        "answer_span_ce_weight": spec.get("answer_span_ce_weight"),
        "role_accuracy": {role: _accuracy(_role_rows(rows, role))
                          for role in roles},
        "fit": {
            "cases": len(fit_rows),
            "exact_accuracy": _accuracy(fit_rows),
            "nonempty_rate": (
                sum(1.0 for row in fit_rows
                    if not row["empty_generation"]) / len(fit_rows)
                if fit_rows else None),
        } if fit_rows else None,
    }


def _objective_gates(objective_audit: Mapping[str, Any] | None) -> dict:
    gates: dict[str, dict[str, Any]] = {}
    present = isinstance(objective_audit, Mapping)
    gates["mask_audit_present"] = {"value": present, "passed": present}
    if present:
        claim = objective_audit.get("claim")
        expected = objective_audit.get("expected_claim")
        gates["objective_claim_consistent"] = {
            "value": claim,
            "expected_claim": expected,
            "passed": (claim in ("span_specific_weighting",
                                 "global_ce_rescale")
                       and (expected is None or claim == expected)),
            "rescaled_only": objective_audit.get("rescaled_only"),
            "added_term_changes_learning_signal": objective_audit.get(
                "added_term_changes_learning_signal"),
        }
    else:
        gates["objective_claim_consistent"] = {"value": None, "passed": False}
    return gates


def score_fit_stage(
        *, protocol_id: str, cases: Sequence[Mapping[str, Any]],
        labels: Sequence[Mapping[str, Any]],
        predictions: Sequence[Mapping[str, Any]],
        arm_specs: Sequence[Mapping[str, Any]], binding: Mapping[str, Any],
        binding_sha256: str, decode_fn: Callable[[Sequence[int]], str],
        verify_decoded: Callable[..., str], gates: Mapping[str, Any],
        objective_audit: Mapping[str, Any] | None = None,
        normalize: Callable[[str], str] = normalize_answer) -> dict:
    """Fit-stage admission evidence: every arm, plus the treatment gates."""
    required = ("fit_exact_min", "fit_nonempty_min")
    for key in required:
        _require(key in gates, "scoring", f"missing fit gate threshold {key}")
    scored, counts = _score_rows(
        stage=STAGE_FIT, protocol_id=protocol_id, cases=cases, labels=labels,
        predictions=predictions, arm_specs=arm_specs, binding=binding,
        binding_sha256=binding_sha256, decode_fn=decode_fn,
        verify_decoded=verify_decoded, normalize=normalize)
    roles = ("fit_primary", "fit_correction", "fit_abstention")
    spec_by_arm = {
        spec["arm_id"]: {
            **spec,
            "prediction_sha256": counts["prediction_sha256_by_arm"][
                spec["arm_id"]],
        } for spec in arm_specs}
    metrics = {arm: _arm_metrics(scored[arm], spec_by_arm[arm], roles=roles)
               for arm in spec_by_arm}
    treatment = metrics.get("treatment")
    _require(treatment is not None, "scoring", "treatment arm is missing")
    gate_rows: dict[str, dict[str, Any]] = {}
    gate_rows["treatment_fit_exact"] = {
        "value": treatment["fit"]["exact_accuracy"],
        "min": gates["fit_exact_min"],
        "passed": treatment["fit"]["exact_accuracy"] is not None
        and treatment["fit"]["exact_accuracy"] >= gates["fit_exact_min"],
    }
    gate_rows["treatment_fit_nonempty"] = {
        "value": treatment["fit"]["nonempty_rate"],
        "min": gates["fit_nonempty_min"],
        "passed": treatment["fit"]["nonempty_rate"] is not None
        and treatment["fit"]["nonempty_rate"] >= gates["fit_nonempty_min"],
    }
    gate_rows["all_arms_evidence"] = {
        "value": sorted(metrics),
        "passed": all(metrics[arm]["decode_verified"] == counts["count"]
                      for arm in metrics),
    }
    gate_rows["treatment_arm_present"] = {
        "value": "treatment" in metrics, "passed": "treatment" in metrics}
    matched = metrics.get("matched_trained")
    gate_rows["matched_control_weight_zero"] = {
        "value": None if matched is None
        else matched["answer_span_ce_weight"],
        "passed": matched is not None
        and matched["answer_span_ce_weight"] == 0.0,
    }
    gate_rows["decode_verified_per_arm"] = {
        "value": {arm: metrics[arm]["decode_verified"] for arm in metrics},
        "passed": all(metrics[arm]["decode_verified"] == counts["count"]
                      for arm in metrics),
    }
    gate_rows.update(_objective_gates(objective_audit))
    passed = all(row["passed"] for row in gate_rows.values())
    return {
        "format": SCORE_REPORT_FORMAT,
        "stage": STAGE_FIT,
        "protocol_id": protocol_id,
        "status": "passed" if passed else "failed",
        "arms": metrics,
        "cases": counts,
        "gates": gate_rows,
        "objective_audit": dict(objective_audit) if objective_audit else None,
        "source_free": True,
        "answer_index_used": False,
        "tokenizer_binding_sha256": binding_sha256,
        "controls_use_identical_scorer": True,
    }


def score_development_stage(
        *, protocol_id: str, cases: Sequence[Mapping[str, Any]],
        labels: Sequence[Mapping[str, Any]],
        predictions: Sequence[Mapping[str, Any]],
        arm_specs: Sequence[Mapping[str, Any]], binding: Mapping[str, Any],
        binding_sha256: str, decode_fn: Callable[[Sequence[int]], str],
        verify_decoded: Callable[..., str], gates: Mapping[str, Any],
        seed: int, resamples: int = 2000,
        control_arms: Sequence[str] = ("matched_trained", "no_update",
                                       "reset_P"),
        objective_audit: Mapping[str, Any] | None = None,
        normalize: Callable[[str], str] = normalize_answer) -> dict:
    """Development comparison: treatment against the matched 0.0 control."""
    for key in list(DEV_ROLE_GATES.values()) + list(DEV_GAIN_GATES.values()) \
            + ["control_primary_accuracy_max", "abstention_false_recall_max",
               "retention_decline_max", "paired_ci_low_min"]:
        _require(key in gates, "scoring",
                 f"missing development gate threshold {key}")
    scored, counts = _score_rows(
        stage=STAGE_DEVELOPMENT, protocol_id=protocol_id, cases=cases,
        labels=labels, predictions=predictions, arm_specs=arm_specs,
        binding=binding, binding_sha256=binding_sha256, decode_fn=decode_fn,
        verify_decoded=verify_decoded, normalize=normalize)
    spec_by_arm = {
        spec["arm_id"]: {
            **spec,
            "prediction_sha256": counts["prediction_sha256_by_arm"][
                spec["arm_id"]],
        } for spec in arm_specs}
    roles = tuple(DEV_ROLE_GATES)
    metrics = {arm: _arm_metrics(rows, spec_by_arm[arm], roles=roles)
               for arm, rows in scored.items()}
    treatment = metrics["treatment"]
    gate_rows: dict[str, dict[str, Any]] = {}

    def arm_accuracy(arm: str, role: str) -> float:
        value = metrics[arm]["role_accuracy"][role]
        _require(value is not None, "scoring",
                 f"no {arm}/{role} rows")
        return float(value)

    for role, key in DEV_ROLE_GATES.items():
        value = arm_accuracy("treatment", role)
        gate_rows[f"treatment_{role}_accuracy"] = {
            "value": value, "min": gates[key], "passed": value >= gates[key]}

    for role in PRIMARY_ROLES:
        for control in control_arms:
            treatment_groups = _group_accuracy(scored["treatment"], role)
            control_groups = _group_accuracy(scored[control], role)
            _require(set(treatment_groups) == set(control_groups), "scoring",
                     f"paired groups differ for {role}/{control}")
            differences = [treatment_groups[group] - control_groups[group]
                           for group in sorted(treatment_groups)]
            ci = _paired_bootstrap_ci(differences, seed=seed,
                                      resamples=resamples)
            gate_rows[f"{role}_gain_vs_{control}"] = {
                "mean_gain": ci["mean"],
                "min_gain": gates[DEV_GAIN_GATES[role]],
                "ci_low": ci["low"],
                "ci_low_min": gates["paired_ci_low_min"],
                "passed": (ci["mean"] >= gates[DEV_GAIN_GATES[role]]
                           and ci["low"] > gates["paired_ci_low_min"]),
                "paired_groups": len(differences),
            }
            value = arm_accuracy(control, role)
            gate_rows[f"{control}_{role}_control_ceiling"] = {
                "value": value,
                "max": gates["control_primary_accuracy_max"],
                "passed": value <= gates["control_primary_accuracy_max"],
            }

    abstention_rows = _role_rows(scored["treatment"], "abstention")
    false_recall = (
        sum(1.0 for row in abstention_rows
            if not row["abstained"]
            and normalize(row["target"]) == "unknown") / len(abstention_rows)
        if abstention_rows else None)
    gate_rows["abstention_false_recall"] = {
        "value": false_recall,
        "max": gates["abstention_false_recall_max"],
        "passed": false_recall is not None
        and false_recall <= gates["abstention_false_recall_max"],
    }
    matched_retention = arm_accuracy("matched_trained", "retention")
    treatment_retention = arm_accuracy("treatment", "retention")
    decline = matched_retention - treatment_retention
    gate_rows["retention_not_degraded_vs_matched"] = {
        "treatment": treatment_retention,
        "matched_trained": matched_retention,
        "decline": decline,
        "max": gates["retention_decline_max"],
        "passed": decline <= gates["retention_decline_max"],
    }
    gate_rows["all_arms_evidence"] = {
        "value": sorted(metrics),
        "passed": all(metrics[arm]["decode_verified"] == counts["count"]
                      for arm in metrics),
    }
    gate_rows["treatment_arm_present"] = {
        "value": "treatment" in metrics, "passed": "treatment" in metrics}
    matched = metrics.get("matched_trained")
    gate_rows["matched_control_weight_zero"] = {
        "value": None if matched is None
        else matched["answer_span_ce_weight"],
        "passed": matched is not None
        and matched["answer_span_ce_weight"] == 0.0,
    }
    gate_rows["decode_verified_per_arm"] = {
        "value": {arm: metrics[arm]["decode_verified"] for arm in metrics},
        "passed": all(metrics[arm]["decode_verified"] == counts["count"]
                      for arm in metrics),
    }
    gate_rows.update(_objective_gates(objective_audit))
    passed = all(row["passed"] for row in gate_rows.values())
    return {
        "format": SCORE_REPORT_FORMAT,
        "stage": STAGE_DEVELOPMENT,
        "protocol_id": protocol_id,
        "status": "passed" if passed else "failed",
        "arms": metrics,
        "cases": counts,
        "gates": gate_rows,
        "objective_audit": dict(objective_audit) if objective_audit else None,
        "primary_roles": list(PRIMARY_ROLES),
        "control_arms": list(control_arms),
        "source_free": True,
        "answer_index_used": False,
        "tokenizer_binding_sha256": binding_sha256,
        "controls_use_identical_scorer": True,
        "paired_bootstrap_resamples": resamples,
    }


def score_final_stage(
        *, protocol_id: str, cases: Sequence[Mapping[str, Any]],
        labels: Sequence[Mapping[str, Any]],
        predictions: Sequence[Mapping[str, Any]],
        arm_specs: Sequence[Mapping[str, Any]], binding: Mapping[str, Any],
        binding_sha256: str, decode_fn: Callable[[Sequence[int]], str],
        verify_decoded: Callable[..., str], gates: Mapping[str, Any],
        objective_audit: Mapping[str, Any] | None = None,
        normalize: Callable[[str], str] = normalize_answer) -> dict:
    """One-shot final confirmation: treatment only, no 0.5-vs-0.0 claim."""
    _require(len(arm_specs) == 1
             and arm_specs[0].get("arm_id") == STAGE_FINAL_TREATMENT_ARM,
             "scoring",
             "the final stage is treatment-only confirmation evidence")
    for key in list(FINAL_ROLE_GATES.values()) + ["fit_exact_min",
                                                  "fit_nonempty_min"]:
        _require(key in gates, "scoring",
                 f"missing final gate threshold {key}")
    scored, counts = _score_rows(
        stage=STAGE_FINAL, protocol_id=protocol_id, cases=cases,
        labels=labels, predictions=predictions, arm_specs=arm_specs,
        binding=binding, binding_sha256=binding_sha256, decode_fn=decode_fn,
        verify_decoded=verify_decoded, normalize=normalize)
    _require(counts["split_role_counts"].get("final_fit", 0) > 0
             and counts["split_role_counts"].get("sealed_final", 0) > 0,
             "scoring", "final scoring requires final-fit and final cases")
    spec = {
        **arm_specs[0],
        "prediction_sha256": counts["prediction_sha256_by_arm"][
            arm_specs[0]["arm_id"]],
    }
    metrics = {spec["arm_id"]: _arm_metrics(
        scored[spec["arm_id"]], spec, roles=tuple(FINAL_ROLE_GATES))}
    rows = scored[spec["arm_id"]]
    gate_rows: dict[str, dict[str, Any]] = {}
    for role, key in FINAL_ROLE_GATES.items():
        value = _accuracy(_role_rows(rows, role))
        _require(value is not None, "scoring", f"no final {role} rows")
        gate_rows[f"treatment_final_{role}_accuracy"] = {
            "value": float(value), "min": gates[key],
            "passed": float(value) >= gates[key]}
    fit_rows = [row for row in rows if row["role"].startswith("fit_")]
    fit_exact = _accuracy(fit_rows)
    fit_nonempty = (sum(1.0 for row in fit_rows
                        if not row["empty_generation"]) / len(fit_rows)
                    if fit_rows else None)
    gate_rows["final_fit_exact"] = {
        "value": fit_exact, "min": gates["fit_exact_min"],
        "passed": fit_exact is not None
        and fit_exact >= gates["fit_exact_min"]}
    gate_rows["final_fit_nonempty"] = {
        "value": fit_nonempty, "min": gates["fit_nonempty_min"],
        "passed": fit_nonempty is not None
        and fit_nonempty >= gates["fit_nonempty_min"]}
    gate_rows["final_confirmation_only"] = {
        "arms": [spec["arm_id"]],
        "comparison": None,
        "passed": [spec["arm_id"]] == [STAGE_FINAL_TREATMENT_ARM],
    }
    gate_rows.update(_objective_gates(objective_audit))
    passed = all(row["passed"] for row in gate_rows.values())
    return {
        "format": SCORE_REPORT_FORMAT,
        "stage": STAGE_FINAL,
        "protocol_id": protocol_id,
        "status": "passed" if passed else "failed",
        "arms": metrics,
        "cases": counts,
        "gates": gate_rows,
        "objective_audit": dict(objective_audit) if objective_audit else None,
        "confirmation_only": True,
        "final_comparison": None,
        "source_free": True,
        "answer_index_used": False,
        "tokenizer_binding_sha256": binding_sha256,
        "controls_use_identical_scorer": True,
    }


PUBLIC_API = {
    "module": "replica_lab.dc_staged_evaluation",
    "stages": list(STAGE_ORDER),
    "fit": "score_fit_stage",
    "development": "score_development_stage",
    "final": "score_final_stage",
    "admission": "StagedAdmission",
    "vault": "LabelVault",
    "spy": "AccessSpy",
    "receipt": "build_stage_receipt",
    "verify_receipt": "verify_stage_receipt",
    "prior_report": "require_prior_report",
}

"""Executable DC-P1 v2 response-learning runner.

This module owns orchestration only: authorization binding, stage ordering,
fit/development/final execution, physical-work accounting, per-event durable
checkpoints, source-free generated-answer evaluation and publication.  It does
not modify the protocol, fixture, scorer or publication modules.

The native path lives in :class:`NativeP1Backend`.  It loads the sealed
MiniCPM5 source through :func:`replica_lab.architecture.loading.load_local_mps`
inside an exclusive worker, freezes the backbone, installs the declared fixed
plastic read path, trains exactly ``expert_0.gate/up/down`` with
``NativeFusedAdamW`` and generates greedy answers with no source context and no
KV cache.  The production CLI has no fake fallback.  CPU orchestration tests
inject an explicit fake backend through the ``backend`` keyword; the CLI never
sets that keyword.

Nothing here imports ``torch`` or a native model at module import time.
Authorization and commit binding are checked before the native backend is
constructed.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Callable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from dc_protocol_common_20261027 import (  # noqa: E402
    ProtocolError, assert_no_answer_shortcut,
    assert_prompt_matches, assert_source_free_prediction,
    check_work_budget_v2, is_hex64, read_json, read_jsonl, require,
    require_committed_bytes, sha256_file, sha256_json, tokenizer_binding_sha,
    validate_baseline_model, validate_guards, validate_tokenizer_binding,
    verify_decoded_text,
)
from replica_lab.learning_evidence import learning_success_evidence
from replica_lab import artifact_publication as publication

PROTOCOL_ID = "dc_p1_native_response_v2"
CONFIG_FORMATS = ("replica.dc_p1_native_response.protocol.v2",)
PREDICTION_FORMAT = "replica.dc_p1_native_response_v2.prediction.v1"
CHECKPOINT_FORMAT = "replica.dc_p1_native_response_v2.checkpoint.v1"
STAGE_JOURNAL_FORMAT = "replica.dc_p1_native_response_v2.stage_journal.v1"
AUTHORIZATION_FORMAT_V2 = "replica.dc_protocol.authorization.v2"
STAGES = ("fit", "development", "final")
ARMS = ("treatment", "matched_trained", "no_update", "reset_P")
FINAL_ARM = "treatment"
PARENT_ID = "dcp1v2:parent:minicpm5-f9740005"
RUNNER_RELATIVE = "scripts/run_dc_p1_native_response_v2_20261027.py"
IMPLEMENTATION_RELATIVES = (
    RUNNER_RELATIVE,
    "src/replica_lab/dc_p1_native_runner.py",
    "src/replica_lab/dc_p1_p_only_reader.py",
    "src/replica_lab/dc_response_objective.py",
    "src/replica_lab/dc_staged_evaluation.py",
    "src/replica_lab/artifact_publication.py",
    "src/replica_lab/learning_evidence.py",
    "scripts/dc_protocol_common_20261027.py",
)
COUNTER_NAMES = (
    "model_loads", "tokenizer_loads", "training_forwards", "backward_calls",
    "optimizer_steps", "generation_requests", "generation_prefill_forwards",
    "generation_decode_forwards", "autoregressive_forwards", "forward_calls",
    "generated_tokens", "teaching_processed_positions",
    "generation_processed_positions", "processed_text_positions",
    "parity_probe_forwards", "parity_processed_positions", "control_evaluations",
    "per_event_checkpoint_writes", "artifact_checkpoint_writes",
    "checkpoint_writes", "retained_checkpoint_artifacts",
    "live_recovery_checkpoints", "storage_temp_overlap_blocks", "restarts",
    "recovery_reloads", "restart_model_loads", "restart_tokenizer_loads",
    "restart_training_forwards", "restart_backward_calls",
    "restart_optimizer_steps", "restart_processed_text_positions",
    "temporary_storage_bytes",
)


class RunnerError(RuntimeError):
    """One runner, evidence or accounting invariant failed."""


class Refused(RunnerError):
    """A fail-closed pre-native refusal with a machine-readable reason."""

    def __init__(self, check: str, detail: str):
        super().__init__(f"{check}: {detail}")
        self.check = check
        self.detail = detail

    def as_dict(self) -> dict[str, str]:
        return {"check": self.check, "detail": self.detail}


class RunInterrupted(RuntimeError):
    """Expected bounded interruption; recovery is allowed without ambiguity."""

    def __init__(self, completed_updates: int):
        self.completed_updates = int(completed_updates)
        super().__init__(f"interrupted after {self.completed_updates} updates")


# ---------------------------------------------------------------------------
# Pure helpers


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def empty_counters() -> dict[str, int]:
    return {name: 0 for name in COUNTER_NAMES}

def _add(left: Mapping[str, int], right: Mapping[str, int]) -> dict[str, int]:
    result = dict(left)
    for name, value in right.items():
        require(name in COUNTER_NAMES, "counter", f"unknown counter {name!r}")
        require(isinstance(value, int) and not isinstance(value, bool)
                and value >= 0, "counter", f"{name} must be nonnegative")
        if name in {"temporary_storage_bytes", "live_storage_bytes",
                    "live_recovery_checkpoints", "storage_temp_overlap_blocks",
                    "retained_checkpoint_artifacts", "wall_seconds"}:
            result[name] = max(result.get(name, 0), value)
        else:
            result[name] = result.get(name, 0) + value
    result["forward_calls"] = (
        result.get("training_forwards", 0)
        + result.get("autoregressive_forwards", 0)
        + result.get("parity_probe_forwards", 0)
        + result.get("restart_training_forwards", 0))
    split_positions = (
        result.get("teaching_processed_positions", 0)
        + result.get("generation_processed_positions", 0)
        + result.get("parity_processed_positions", 0)
        + result.get("restart_processed_text_positions", 0))
    if split_positions:
        result["processed_text_positions"] = split_positions
    return result


def _max_merge(left: Mapping[str, int], right: Mapping[str, int]) -> dict[str, int]:
    result = dict(left)
    for name, value in right.items():
        require(name in COUNTER_NAMES, "counter", f"unknown counter {name!r}")
        result[name] = max(result.get(name, 0), int(value))
    return result


def _counters_from_mapping(value: Any, name: str) -> dict[str, int]:
    require(isinstance(value, Mapping), "journal", f"{name} must be an object")
    result = empty_counters()
    for key, raw in value.items():
        require(key in COUNTER_NAMES, "journal", f"unknown counter {key!r}")
        require(isinstance(raw, int) and not isinstance(raw, bool) and raw >= 0,
                "journal", f"{key} must be nonnegative")
        result[key] = int(raw)
    return result


def git_head() -> str:
    """Return the full HEAD commit of this worktree without network access."""
    import subprocess

    completed = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        check=False, capture_output=True, text=True, timeout=30)
    value = completed.stdout.strip() if completed.returncode == 0 else ""
    return value if (len(value) == 40 and all(
        character in "0123456789abcdef" for character in value)) else ""


def _require_clean_tracked_worktree() -> None:
    import subprocess

    completed = subprocess.run(
        ["git", "-C", str(ROOT), "status", "--porcelain=v1",
         "--untracked-files=no"], check=False, capture_output=True, text=True,
        timeout=30)
    require(completed.returncode == 0, "authorization",
            "could not inspect tracked worktree state")
    require(not completed.stdout.strip(), "authorization",
            "tracked worktree changes must be committed before native work")


def stage_root(output: Path, stage: str) -> Path:
    return output / stage


def _journal_path(output: Path, stage: str) -> Path:
    return stage_root(output, stage) / "journal.json"


def _publication_identity(context: "RunContext") -> dict[str, Any]:
    return {
        "protocol_id": PROTOCOL_ID,
        "config_sha256": context.config_sha256,
        "prepared_manifest_sha256": context.manifest_sha256,
        "protocol_commit": context.protocol_commit,
        "authorization_receipt_sha256": context.authorization_sha256,
    }


# ---------------------------------------------------------------------------
# Work accounting


@dataclass(frozen=True)
class WorkReservation:
    delta: dict[str, int]
    arm: str | None


class WorkMeter:
    """Cumulative + local counters with pre-call and post-call cap checks.

    Run-level counters are checked against ``budget.physical_limits``.  Each
    arm also owns a separate cumulative counter block checked against
    ``budget.per_arm_limits[arm]``; per-arm work is never inferred from the run
    total.
    """

    def __init__(self, budget: Mapping[str, Any], *,
                 cumulative: Mapping[str, int] | None = None,
                 local: Mapping[str, int] | None = None,
                 arm_counters: Mapping[str, Mapping[str, int]] | None = None,
                 enforce_budget: bool = True):
        self.budget = budget
        self.cumulative = _counters_from_mapping(
            cumulative or empty_counters(), "cumulative")
        self.local = _counters_from_mapping(local or empty_counters(), "local")
        self.arm_counters = {
            str(arm): _counters_from_mapping(values, f"arm:{arm}")
            for arm, values in (arm_counters or {}).items()}
        self.enforce_budget = bool(enforce_budget)

    def _arm(self, arm: str) -> dict[str, int]:
        return self.arm_counters.setdefault(arm, empty_counters())
    def reserve(self, delta: Mapping[str, int], *, arm: str | None = None
                ) -> WorkReservation:
        """Reserve exact maximum work before any native execution."""
        self.precheck(delta, arm=arm)
        return WorkReservation(dict(delta), arm)

    def reconcile(self, reservation: WorkReservation,
                  actual: Mapping[str, int]) -> dict:
        """Charge actual completed work immediately after execution."""
        for name, value in actual.items():
            require(name in reservation.delta, "work",
                    f"actual counter {name!r} was not reserved")
            require(value <= reservation.delta[name], "work",
                    f"actual {name}={value} exceeds reservation "
                    f"{reservation.delta[name]}")
        return self.charge(actual, arm=reservation.arm)

    def abort(self, reservation: WorkReservation, error: BaseException) -> None:
        """Record ambiguous in-flight work; a later resume must refuse it."""
        self.ambiguous_work.append({
            "delta": dict(reservation.delta), "arm": reservation.arm,
            "error": str(error),
        })

    @property
    def ambiguous_work(self) -> list[dict]:
        if not hasattr(self, "_ambiguous_work"):
            self._ambiguous_work = []
        return self._ambiguous_work

    def take_ambiguous_work(self) -> list[dict]:
        rows = list(self.ambiguous_work)
        self.ambiguous_work.clear()
        return rows


    def precheck(self, delta: Mapping[str, int], *, arm: str | None = None) -> None:
        """Fail before work that would cross any declared native ceiling."""
        if not self.enforce_budget:
            return
        check_work_budget_v2(_add(self.cumulative, delta), self.budget)
        check_work_budget_v2(_add(self.local, delta), self.budget)
        if arm is not None:
            check_work_budget_v2(_add(self._arm(arm), delta), self.budget,
                                 arm=arm)

    def charge(self, delta: Mapping[str, int], *, arm: str | None = None) -> dict:
        self.precheck(delta, arm=arm)
        self.cumulative = _add(self.cumulative, delta)
        self.local = _add(self.local, delta)
        if arm is not None:
            self.arm_counters[arm] = _add(self._arm(arm), delta)
        if not self.enforce_budget:
            return dict(delta)
        check_work_budget_v2(self.cumulative, self.budget)
        check_work_budget_v2(self.local, self.budget)
        if arm is not None:
            check_work_budget_v2(self.arm_counters[arm], self.budget, arm=arm)
        return dict(delta)

    def adopt(self, cumulative: Mapping[str, int],
              local: Mapping[str, int] | None = None,
              arm: str | None = None,
              arm_counters: Mapping[str, int] | None = None) -> None:
        """Accept durable counters after a crash; counters only increase."""
        self.cumulative = _max_merge(self.cumulative, cumulative)
        if local is not None:
            self.local = _max_merge(self.local, local)
        if arm is not None and arm_counters is not None:
            self.arm_counters[arm] = _max_merge(
                self._arm(arm), arm_counters)

    def report(self) -> dict[str, Any]:
        return {
            "cumulative": dict(sorted(self.cumulative.items())),
            "local": dict(sorted(self.local.items())),
            "per_arm": {arm: dict(sorted(values.items()))
                        for arm, values in sorted(self.arm_counters.items())},
            "pre_call_and_post_call": True,
        }


# ---------------------------------------------------------------------------
# Context and authorization


@dataclass(frozen=True)
class RunContext:
    config_path: Path
    config: dict
    config_sha256: str
    prepared: Path
    manifest: dict
    manifest_sha256: str
    authorization: dict
    authorization_sha256: str
    protocol_commit: str
    output: Path

    @property
    def binding(self) -> dict:
        return validate_tokenizer_binding(self.config["tokenizer_binding"])

    @property
    def binding_sha256(self) -> str:
        return tokenizer_binding_sha(self.binding)

    @property
    def decode_policy_sha256(self) -> str:
        return sha256_json(self.binding["decode_policy"])

    def trainable_groups(self) -> tuple[str, ...]:
        spec = self.config.get("plastic_spec", {})
        groups = spec.get("trainable_groups")
        if isinstance(groups, list) and groups:
            return tuple(str(item) for item in groups)
        return ("expert_0.gate", "expert_0.up", "expert_0.down")

    def arm_specs(self, stage: str = "fit") -> list[dict]:
        if stage == "final":
            final = self.config.get("controls", {}).get("final_stage", {})
            return [{
                "arm_id": FINAL_ARM,
                "role": final.get("role", "confirmation_only"),
                "answer_span_ce_weight": 0.5,
                "updates": int(final.get("updates", 12)),
                "initialization": "same_parent_checkpoint",
                "trainable_groups": list(self.trainable_groups()),
                "restore_parent_before_evaluation": False,
                "checkpoint_id": final.get(
                    "checkpoint_id", "dcp1v2:arm:treatment:final12"),
                "source_context_at_evaluation": False,
                "source_kv_at_evaluation": False,
            }]
        arms = self.config.get("controls", {}).get("arms")
        require(isinstance(arms, list) and arms, "config",
                "config.controls.arms is required")
        specs = [dict(arm) for arm in arms]
        require(tuple(spec["arm_id"] for spec in specs) == ARMS, "config",
                f"declared arm order must be {ARMS}")
        for spec in specs:
            require(spec.get("source_context_at_evaluation") is False,
                    "config", "source context must be removed at evaluation")
            require(spec.get("source_kv_at_evaluation") is False,
                    "config", "source KV must be removed at evaluation")
        return specs


def verify_authorization(*, config_path: str | Path, prepared: str | Path,
                         authorization_path: str | Path | None,
                         protocol_commit: str | None,
                         require_head: bool = False, stage: str = "fit",
                         commit_verifier: Callable[[str, str], bytes | None]
                         | None = None) -> tuple[
                             RunContext, dict[str, Any]]:
    """Verify external bindings without reading later-stage label bytes."""
    config_path = Path(config_path)
    prepared = Path(prepared)
    authorization_path = (None if authorization_path is None
                          else Path(authorization_path))
    config = read_json(config_path)
    require(config.get("protocol_id") == PROTOCOL_ID, "authorization",
            "config protocol_id differs from the runner")
    require(config.get("format") in CONFIG_FORMATS, "authorization",
            "config format differs")
    validate_guards(config)
    validate_baseline_model(config)
    validate_tokenizer_binding(config.get("tokenizer_binding"))
    manifest = read_json(prepared / "manifest.json")
    require(manifest.get("protocol_id") == PROTOCOL_ID, "authorization",
            "prepared manifest protocol differs")
    body = {key: value for key, value in manifest.items()
            if key != "manifest_sha256"}
    require(manifest.get("manifest_sha256") == sha256_json(body),
            "authorization", "prepared manifest self hash differs")
    files = manifest.get("files")
    require(isinstance(files, Mapping) and bool(files), "authorization",
            "prepared manifest files are missing")
    require(stage in STAGES, "authorization", f"unknown stage {stage!r}")
    allowed_files = {
        "fit": {"teaching.jsonl", "fit_cases.jsonl"},
        "development": {"teaching.jsonl", "fit_cases.jsonl", "dev_cases.jsonl"},
        "final": {"sealed_final_teaching.jsonl", "sealed_final_fit_cases.jsonl",
                  "sealed_final_cases.jsonl"},
    }[stage]
    for name in allowed_files:
        path = prepared / name
        require(path.is_file(), "authorization",
                f"prepared file is missing: {name}")
        require(sha256_file(path) == files[name], "authorization",
                f"prepared file bytes differ: {name}")
    require(config.get("prepared", {}).get("manifest_sha256")
            == manifest.get("manifest_sha256"), "authorization",
            "config does not bind the prepared manifest")
    token_lengths = config.get("prepared", {}).get("token_lengths")
    if isinstance(token_lengths, Mapping) and token_lengths.get(
            "required_before_native_execution") is True:
        length_path = prepared / str(token_lengths.get("path", ""))
        require(length_path.is_file(), "token_lengths",
                "the sealed token-length measurement is missing; native "
                "execution is refused")
        require(token_lengths.get("sha256") == sha256_file(length_path),
                "token_lengths",
                "token-length file bytes differ from the declared digest")
    config_sha = sha256_file(config_path)
    require(isinstance(protocol_commit, str) and len(protocol_commit) == 40
            and all(character in "0123456789abcdef"
                    for character in protocol_commit), "authorization",
            "a full 40-hex --protocol-commit is required before native work")
    if require_head:
        head = git_head()
        require(head == protocol_commit, "authorization",
                f"HEAD {head!r} differs from --protocol-commit "
                f"{protocol_commit!r}")
        _require_clean_tracked_worktree()
    require_committed_bytes(
        protocol_commit,
        config_path.resolve().relative_to(ROOT.resolve()).as_posix(),
        sha256_file(config_path), verifier=commit_verifier)
    require_committed_bytes(
        protocol_commit,
        (prepared / "manifest.json").resolve().relative_to(
            ROOT.resolve()).as_posix(), sha256_file(prepared / "manifest.json"),
        verifier=commit_verifier)
    for relative in IMPLEMENTATION_RELATIVES:
        require_committed_bytes(
            protocol_commit, relative, sha256_file(ROOT / relative),
            verifier=commit_verifier)
    require(authorization_path is not None, "authorization",
            "a separate external authorization receipt is required")
    authorization = read_json(authorization_path)
    require(authorization.get("format") == AUTHORIZATION_FORMAT_V2,
            "authorization", "authorization format differs")
    require(authorization.get("protocol_id") == PROTOCOL_ID, "authorization",
            "authorization protocol differs")
    require(authorization.get("config_sha256") == config_sha, "authorization",
            "authorization did not bind the exact config bytes")
    require(authorization.get("prepared_manifest_sha256")
            == manifest.get("manifest_sha256"), "authorization",
            "authorization did not bind the exact prepared manifest")
    require(authorization.get("protocol_commit") == protocol_commit,
            "authorization", "authorization protocol_commit differs")
    require(authorization.get("native_execution_allowed") is True,
            "authorization", "native_execution_allowed is not true")
    require(authorization.get("single_attempt") is True, "authorization",
            "single_attempt must be true")
    require(isinstance(authorization.get("authorized_by"), str)
            and bool(authorization["authorized_by"].strip()), "authorization",
            "authorized_by must be a nonempty string")
    require(is_hex64(authorization.get("receipt_sha256")), "authorization",
            "receipt_sha256 must be a sealed digest")
    require(sha256_json({key: value for key, value in authorization.items()
                         if key != "receipt_sha256"})
            == authorization["receipt_sha256"], "authorization",
            "authorization self digest differs")
    require(is_hex64(authorization.get("runner_sha256")), "authorization",
            "runner_sha256 is required")
    require(authorization.get("runner_path") == RUNNER_RELATIVE,
            "authorization", "authorization runner_path differs")
    require(authorization["runner_sha256"]
            == sha256_file(ROOT / RUNNER_RELATIVE), "authorization",
            "runner bytes differ from the receipt")
    require(authorization.get("max_optimizer_steps")
            == config["budget"]["declared_total_updates"], "authorization",
            "authorization optimizer-step ceiling differs from 120")
    require(authorization.get("max_generation_requests")
            == config["budget"]["physical_limits"]["max_generation_requests"],
            "authorization",
            "authorization generation-request ceiling differs from 416")
    require(authorization.get("restart_allowance") == 1, "authorization",
            "exactly one same-attempt recovery reload must be authorized")
    output = Path(config["launch"]["output_root"])
    if not output.is_absolute():
        output = ROOT / output
    require(authorization.get("output_root") == str(output), "authorization",
            "authorization output_root differs from the single attempt root")
    context = RunContext(
        config_path=config_path, config=config, config_sha256=config_sha,
        prepared=prepared, manifest=manifest,
        manifest_sha256=manifest["manifest_sha256"],
        authorization=authorization,
        authorization_sha256=sha256_file(authorization_path),
        protocol_commit=protocol_commit, output=output)
    return context, {
        "status": "authorization_verified",
        "authorization_receipt_sha256": context.authorization_sha256,
        "config_sha256": config_sha,
        "prepared_manifest_sha256": context.manifest_sha256,
        "protocol_commit": protocol_commit,
        "implementation_sha256": {
            relative: sha256_file(ROOT / relative)
            for relative in IMPLEMENTATION_RELATIVES},
    }


# ---------------------------------------------------------------------------
# Case/label loading and stage identity


EXPECTED_SPLIT_ROLES = {
    "fit": {"fit"},
    "development": {"dev"},
    "final": {"final_fit", "sealed_final"},
}


def _load_cases(prepared: Path, names: Sequence[str], stage: str) -> list[dict]:
    rows: list[dict] = []
    for name in names:
        for row in read_jsonl(prepared / name):
            require(row.get("split_role") in EXPECTED_SPLIT_ROLES[stage],
                    "prepared", f"{name}: split_role is not valid for {stage}")
            require(row.get("inference_inputs") == ["prompt"], "prepared",
                    f"{name}: evaluation case must expose only the prompt")
            require(row.get("source_context_present") is False
                    and row.get("source_kv_loaded") is False, "prepared",
                    "evaluation cases must be source-free")
            rows.append(dict(row))
    require(rows, "prepared", f"no {stage} cases")
    require(len({row["case_id"] for row in rows}) == len(rows), "prepared",
            "duplicate case id")
    return rows


def _load_labels(prepared: Path, names: Sequence[str], stage: str) -> list[dict]:
    rows: list[dict] = []
    for name in names:
        for row in read_jsonl(prepared / name):
            require(row.get("split_role") in EXPECTED_SPLIT_ROLES[stage],
                    "prepared", f"{name}: label split_role is not valid for {stage}")
            rows.append(dict(row))
    require(rows, "prepared", f"no {stage} labels")
    return rows


def stage_cases(context: RunContext, stage: str) -> list[dict]:
    files = context.config["prepared"]["files"]
    if stage == "fit":
        return _load_cases(context.prepared, [files["fit_cases.jsonl"]], "fit")
    if stage == "development":
        return _load_cases(context.prepared, [files["dev_cases.jsonl"]],
                           "development")
    return _load_cases(
        context.prepared,
        [files["sealed_final_fit_cases.jsonl"], files["sealed_final_cases.jsonl"]],
        "final")


def stage_labels(context: RunContext, stage: str) -> list[dict]:
    files = context.config["prepared"]["files"]
    if stage == "fit":
        return _load_labels(context.prepared, [files["sealed_labels_fit.jsonl"]],
                            "fit")
    if stage == "development":
        return _load_labels(context.prepared, [files["sealed_labels_dev.jsonl"]],
                            "development")
    return _load_labels(
        context.prepared,
        [files["sealed_final_labels_fit.jsonl"], files["sealed_labels_final.jsonl"]],
        "final")


def teaching_rows(context: RunContext, stage: str) -> list[dict]:
    files = context.config["prepared"]["files"]
    name = (files["teaching.jsonl"] if stage != "final"
            else files["sealed_final_teaching.jsonl"])
    rows = [dict(row) for row in read_jsonl(context.prepared / name)]
    rows.sort(key=lambda row: int(row["event_ordinal"]))
    require(len({row["event_ordinal"] for row in rows}) == len(rows),
            "prepared", "duplicate teaching event ordinal")
    return rows


def _expectation(context: RunContext) -> Any:
    from replica_lab.dc_p1_evidence import receipt_expectation

    return receipt_expectation(
        config=context.config, config_path=context.config_path,
        manifest=context.manifest, binding=context.binding,
        protocol_commit=context.protocol_commit)


# ---------------------------------------------------------------------------
# Durable runner journal (separate from the publication journal)


def _new_stage_journal(context: RunContext, stage: str, attempt_id: str,
                       prior: Mapping[str, int],
                       prior_local: Mapping[str, int],
                       prior_arms: Mapping[str, Mapping[str, int]]) -> dict:
    return {
        "format": STAGE_JOURNAL_FORMAT,
        "protocol_id": PROTOCOL_ID,
        "stage": stage,
        "attempt_id": attempt_id,
        "config_sha256": context.config_sha256,
        "manifest_sha256": context.manifest_sha256,
        "protocol_commit": context.protocol_commit,
        "authorization_receipt_sha256": context.authorization_sha256,
        "created_at_utc": _utc(),
        "prior_cumulative": dict(sorted(prior.items())),
        "prior_local": dict(sorted(prior_local.items())),
        "prior_arm_counters": {
            arm: dict(sorted(values.items()))
            for arm, values in sorted(prior_arms.items())},
        "cumulative": dict(sorted(prior.items())),
        "local": dict(sorted(empty_counters().items())),
        "arms": {},
        "arm_counters": {
            arm: dict(sorted(values.items()))
            for arm, values in sorted(prior_arms.items())},
        "parity": None,
        "objective_audit": None,
        "resumes_used": 0,
        "ambiguous_work": [],
        "stage_complete": False,
        "publication_stage_done": False,
        "result": None,
    }


def _load_stage_journal(path: Path) -> dict | None:
    return read_json(path) if path.is_file() else None


def _write_journal(context: RunContext, stage: str, journal: dict) -> None:
    from replica_lab import artifact_publication as publication

    journal["updated_at_utc"] = _utc()
    stage_root(context.output, stage).mkdir(parents=True, exist_ok=True)
    publication.write_json(context.output, f"{stage}/journal.json", journal)


def _previous_journal(context: RunContext, stage: str) -> tuple[
        dict[str, int], dict[str, int], dict[str, dict[str, int]]]:
    index = STAGES.index(stage)
    if index == 0:
        return empty_counters(), empty_counters(), {}
    previous = STAGES[index - 1]
    journal = _load_stage_journal(_journal_path(context.output, previous))
    require(journal is not None, "stage_order",
            f"{previous} journal is missing; {stage} is closed")
    require(journal.get("stage_complete") is True, "stage_order",
            f"{previous} stage is not complete; {stage} is closed")
    return (_counters_from_mapping(journal.get("cumulative", {}), "cumulative"),
            _counters_from_mapping(journal.get("local", {}), "local"),
            {arm: _counters_from_mapping(values, f"arm:{arm}")
             for arm, values in journal.get("arm_counters", {}).items()})


def attempt_id(context: RunContext, stage: str) -> str:
    return "dcp1v2:" + sha256_json({
        "protocol_id": PROTOCOL_ID,
        "stage": stage,
        "config_sha256": context.config_sha256,
        "manifest_sha256": context.manifest_sha256,
        "protocol_commit": context.protocol_commit,
        "output": str(context.output),
    })[:32]


def _require_resume_allowed(context: RunContext, journal: dict) -> None:
    allowance = context.authorization.get("restart_allowance", 0)
    require(isinstance(allowance, int) and not isinstance(allowance, bool)
            and allowance >= 1, "resume",
            "the authorization carries no restart_allowance; a resume that "
            "reloads the model is refused")
    used = int(journal.get("resumes_used", 0))
    require(used < allowance, "resume",
            f"restart allowance {allowance} is already consumed ({used})")


# ---------------------------------------------------------------------------
# Predictions, scoring and receipts


def prediction_row(*, context: RunContext, stage: str,
                   arm_spec: Mapping[str, Any], case: Mapping[str, Any],
                   result: Mapping[str, Any],
                   decode_fn: Callable[[Sequence[int]], str],
                   evidence_class: str) -> dict:
    binding = context.binding
    ids = [int(item) for item in result["generated_ids"]]
    text = result["generated_text"]
    row = {
        "format": PREDICTION_FORMAT,
        "protocol_id": PROTOCOL_ID,
        "stage": stage,
        "arm_id": arm_spec["arm_id"],
        "case_id": case["case_id"],
        "prompt": case["prompt"],
        "prompt_sha256": case["prompt_sha256"],
        "generated_ids": ids,
        "generated_text": text,
        "empty_generation": text == "",
        "tokenizer_binding_sha256": context.binding_sha256,
        "decode_policy": binding["decode_policy"],
        "source_context_present": False,
        "source_kv_loaded": False,
        "source_prefill_tokens": 0,
        "source_kv_reused": False,
        "prompt_source_spans": 0,
        "routing_metadata": {"mode": "fixed", "expert_id": 0},
        "checkpoint_id": arm_spec["checkpoint_id"],
        "evidence_class": evidence_class,
        "query_epoch": case.get("query_epoch"),
        "resolution_mode": case.get("resolution_mode"),
        "resolved_event_position": result.get("resolved_event_position"),
        "reader_state_sha256": result.get("reader_state_sha256"),
        "evaluation_owner": result.get("evaluation_owner"),
        "source_free_audit": result.get("source_free_audit"),
        "stopped": bool(result.get("stopped")),
        "stop_reason": result.get("stop_reason"),
        "generated_tokens": int(result.get("generated_tokens", len(ids))),
        "prefill_tokens": int(result.get("prefill_tokens", 0)),
        "prefill_forwards": int(result.get("prefill_forwards", 1)),
        "decode_forwards": int(result.get("decode_forwards", 0)),
        "processed_positions": int(result.get("processed_positions", 0)),
    }
    assert_no_answer_shortcut(row)
    assert_source_free_prediction(row)
    assert_prompt_matches(case, row)
    verify_decoded_text(row, binding, decode_fn)
    return row


def _score_report(*, context: RunContext, stage: str,
                  cases: Sequence[dict], labels: Sequence[dict],
                  predictions: Sequence[dict], arm_specs: Sequence[dict],
                  objective_audit: Mapping[str, Any] | None,
                  decode_fn: Callable[[Sequence[int]], str],
                  evidence_class: str) -> dict:
    from replica_lab import dc_staged_evaluation as staged

    binding = context.binding
    gates = context.config["evaluation"]["gates"]
    verify = (lambda prediction, binding_arg, decoder:
              verify_decoded_text(prediction, binding_arg, decoder))
    common = dict(
        protocol_id=PROTOCOL_ID, cases=cases, labels=labels,
        predictions=predictions, arm_specs=arm_specs, binding=binding,
        binding_sha256=context.binding_sha256, decode_fn=decode_fn,
        verify_decoded=verify, gates=gates, objective_audit=objective_audit)
    if stage == "fit":
        report = staged.score_fit_stage(**common)
    elif stage == "development":
        report = staged.score_development_stage(
            **common, seed=int(context.config["seed"]),
            resamples=int(context.config["evaluation"]
                          .get("paired_bootstrap_resamples", 2000)))
    else:
        report = staged.score_final_stage(**common)
    # The receipt verifier binds exactly {"stage","count"}; richer counts stay
    # in the score report under "scoring_counts".
    report["scoring_counts"] = dict(report.get("cases", {}))
    report["evidence_class"] = evidence_class
    report["native_execution_admissible"] = (
        evidence_class == "native_mps" and report["status"] == "passed")
    return report
def _receipt_and_admission(*, context: RunContext, stage: str, report: dict,
                           expectation: Any, attempt: str, publish: Any,
                           decode_fn: Callable[[Sequence[int]], str]) -> tuple[
                               dict, Any, Exception | None, dict | None]:
    from replica_lab import dc_p1_evidence as evidence
    from replica_lab import dc_staged_evaluation as staged

    receipt = staged.build_stage_receipt(
        report=report, expectation=expectation, attempt_id=attempt)
    staged.verify_stage_receipt(receipt, stage=stage,
                                expectation=expectation)
    publish.write_json(context.output, f"{stage}/receipt.json", receipt)
    evidence.verify_stage_evidence(
        stage=stage, output_root=context.output, prepared=context.prepared,
        config=context.config, config_path=context.config_path,
        manifest=context.manifest, protocol_commit=context.protocol_commit,
        binding=context.binding, decode_fn=decode_fn,
        expected_evidence_class=report["evidence_class"],
        expected_attempt_id=attempt, require_publication=False)
    admission = staged.StagedAdmission(expectation)
    if report.get("evidence_class") != "native_mps":
        return receipt, admission, Refused(
            "evidence_class", "CPU fixtures cannot admit native stages"), None
    try:
        if stage == "fit":
            admitted = admission.admit_fit(receipt)
        elif stage == "development":
            fit_receipt = read_json(
                stage_root(context.output, "fit") / "receipt.json")
            admission.admit_fit(fit_receipt)
            admitted = admission.admit_development(receipt)
        else:
            fit_receipt = read_json(
                stage_root(context.output, "fit") / "receipt.json")
            dev_receipt = read_json(
                stage_root(context.output, "development") / "receipt.json")
            admission.admit_fit(fit_receipt)
            admission.admit_development(dev_receipt)
            admission.require("final")
            admitted = admission.admit_final(receipt)
    except Exception as caught:
        return receipt, admission, caught, None
    return receipt, admission, None, admitted


def _prior_admission(context: RunContext, stage: str, *,
                     evidence_class: str = "native_mps",
                     decode_fn: Callable[[Sequence[int]], str] | None = None
                     ) -> Any:
    """Resolve and recompute earlier evidence; reads no current-stage labels."""
    from replica_lab import dc_p1_evidence as evidence
    from replica_lab import dc_staged_evaluation as staged

    admission = staged.StagedAdmission(_expectation(context))
    fit_evidence = evidence.verify_stage_evidence(
        stage="fit", output_root=context.output, prepared=context.prepared,
        config=context.config, config_path=context.config_path,
        manifest=context.manifest, protocol_commit=context.protocol_commit,
        binding=context.binding, decode_fn=decode_fn,
        expected_evidence_class=evidence_class,
        expected_attempt_id=attempt_id(context, "fit"),
        require_publication=True)
    admission.admit_fit(fit_evidence["receipt"])
    if stage == "final":
        development_evidence = evidence.verify_stage_evidence(
            stage="development", output_root=context.output,
            prepared=context.prepared, config=context.config,
            config_path=context.config_path, manifest=context.manifest,
            protocol_commit=context.protocol_commit, binding=context.binding,
            decode_fn=decode_fn, expected_evidence_class=evidence_class,
            expected_attempt_id=attempt_id(context, "development"),
            require_publication=True)
        admission.admit_development(development_evidence["receipt"])
    return admission


# ---------------------------------------------------------------------------
# Backends


class Backend:
    """Duck-typed execution backend shared by the native and fake paths."""

    def startup_work(self) -> dict[str, int]:
        return {"model_loads": 1, "tokenizer_loads": 1}

    def encode_teaching(self, row: Mapping[str, Any]) -> dict:
        raise NotImplementedError

    def encode_parity(self, case: Mapping[str, Any]) -> dict:
        raise NotImplementedError

    def train_step(self, encoding: Mapping[str, Any], *, weight: float,
                   arm: str) -> dict:
        raise NotImplementedError

    def estimate_evaluation_work(self, case: Mapping[str, Any]) -> dict:
        raise NotImplementedError

    def evaluate(self, case: Mapping[str, Any], *, arm: str,
                 prepared: Mapping[str, Any] | None = None) -> dict:
        raise NotImplementedError

    def parity_probe(self, example: Mapping[str, Any], *, enabled: bool) -> dict:
        raise NotImplementedError

    def reset_to_parent(self, *, clear_optimizer: bool = True) -> None:
        raise NotImplementedError

    def capture_checkpoint(self, meta: Mapping[str, Any]) -> dict:
        raise NotImplementedError

    @property
    def evidence_class(self) -> str:
        return "cpu_fixture"



    def restore_checkpoint(self, payload: Mapping[str, Any],
                           meta: Mapping[str, Any]) -> None:
        raise NotImplementedError

    def assert_backbone_unchanged(self) -> dict:
        raise NotImplementedError

    @property
    def decode_fn(self) -> Callable[[Sequence[int]], str]:
        raise NotImplementedError

    def close(self) -> None:
        return None


class NativeP1Backend(Backend):
    """Real MiniCPM5 execution: frozen backbone + selected expert read path."""

    def __init__(self, context: RunContext, stage: str):
        import torch
        from replica_lab.architecture.contracts import PlasticSpec
        from replica_lab.architecture.loading import (
            exclusive_mps_worker, load_local_mps)
        from replica_lab.nested_learning import ExpertBank
        from replica_lab.nested_text import initialize_null_plastic
        from replica_lab.profiling import RunProfiler

        self.torch = torch
        self.context = context
        self.stage = stage
        plastic = context.config["plastic_spec"]
        bank_cfg = plastic["bank"]
        features = plastic["features"]
        read_path = plastic["read_path"]
        generation = context.config["generation"]
        self.width = int(bank_cfg["width"])
        self.hidden_width = int(bank_cfg["hidden_width"])
        self.sites = tuple(int(site) for site in features["sites"])
        self.projection_seed = int(features["projection_seed"])
        self.read_gain = float(read_path["gain"])
        self.max_new = int(generation["max_new_tokens_per_call"])
        self.max_prompt = int(generation["evaluation_prompt_tokens_max"])
        self.max_teaching = int(generation["teaching_tokens_per_event_max"])
        self.max_parity = int(generation["parity_probe_tokens_max"])
        self.stops = tuple(int(item) for item in generation["stop_tokens"])
        self.eos_id = int(generation["terminal_eos_token"])
        self._worker = exclusive_mps_worker(ROOT)
        self._worker.__enter__()
        try:
            # The guard watches the attempt root, not a stage-only directory.
            profile_dir = Path(context.output)
            self.profiler = RunProfiler("mps", profile_dir).start()
            model_path = ROOT / context.config["baseline_model"]["model_path"]
            lock_path = ROOT / context.config["baseline_model"]["lock_path"]
            self.adapter, self.tokenizer, self.lock = load_local_mps(
                model_path, lock_path, profiler=self.profiler)
            spec = PlasticSpec(
                experts=int(bank_cfg["experts"]), width=self.width,
                hidden_width=self.hidden_width, sites=self.sites,
                route_mode="fixed", route_seed=0,
                projection_seed=self.projection_seed,
                read_gain=self.read_gain,
                feature_version=features.get(
                    "feature_version", "sites_masked_mean_projection.v1"),
                p_init=bank_cfg.get("p_init", "zero_read_down"))
            self.adapter.install_plastic(spec)
            self.plastic_spec = spec
            self.device = next(self.adapter.base.parameters()).device
            self.bank = ExpertBank(
                expert_count=1, width=self.width,
                hidden_width=self.hidden_width,
                seed=int(bank_cfg.get("seed", 0)),
                lr=float(bank_cfg.get("lr", 1e-4)),
                scope="dc_p1").to(self.device)
            initialize_null_plastic(self.bank)
            self.optimizer = self.bank.optimizer
            self.projection = self.adapter.plastic_ports[
                str(self.sites[0])].decoder.detach()
            self.decoder_digest = self.adapter.plastic_read_digest()
            self.parent_state = {
                name: parameter.detach().clone()
                for name, parameter in self.bank.expert(0).named_parameters()}
            self.parent_sha256 = self._state_sha(self.parent_state)
            self.base_identity = self.adapter.parameter_stamp(
                original_only=True)
            self.trainable_names = tuple(
                f"expert_0.{name}"
                for name, parameter in self.bank.expert(0).named_parameters()
                if parameter.requires_grad)
            self.profiler.mark(
                "model_loaded", parameters=self.adapter.parameter_audit(),
                plastic_spec=self.plastic_spec.to_dict())
            self._teaching_positions = {}
            for row in teaching_rows(context, stage):
                self._teaching_positions.setdefault(
                    row["source_group"], []).append(int(row["event_position"]))
            self._reader = None
        except BaseException:
            self._worker.__exit__(None, None, None)
            raise

    # -- lifecycle ---------------------------------------------------------

    def startup_work(self) -> dict[str, int]:
        return {"model_loads": 1, "tokenizer_loads": 1}

    def close(self) -> None:
        try:
            if getattr(self, "profiler", None) is not None:
                self.profiler.mark("stage_complete")
                self.profiler.stop()
        finally:
            self._worker.__exit__(None, None, None)

    # -- encoding ----------------------------------------------------------

    def _render(self, text: str) -> list[int]:
        rendered = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
        ids = self.tokenizer(rendered, add_special_tokens=False)["input_ids"]
        require(isinstance(ids, list) and ids, "tokenizer",
                "tokenizer returned no ids")
        return [int(item) for item in ids]

    def encode_teaching(self, row: Mapping[str, Any]) -> dict:
        prompt_ids = self._render(
            f"{row['source']}\n{row['prompt']}")
        target_ids = self.tokenizer(
            str(row["target"]), add_special_tokens=False)["input_ids"]
        require(isinstance(target_ids, list) and target_ids, "teaching",
                f"empty target tokenization for {row.get('case_id')}")
        target_ids = [int(item) for item in target_ids]
        response_ids = target_ids + [self.eos_id]
        input_tokens = len(prompt_ids) + len(response_ids) - 1
        require(input_tokens <= self.max_teaching, "teaching",
                f"teaching sequence {input_tokens} exceeds "
                f"{self.max_teaching}")
        return {
            "case_id": row["case_id"],
            "event_ordinal": int(row["event_ordinal"]),
            "prompt_ids": prompt_ids,
            "response_ids": response_ids,
            "answer_span": (0, len(target_ids) - 1),
            "input_tokens": input_tokens,
            "prompt_tokens": len(prompt_ids),
            "response_tokens": len(response_ids),
            "sequence_tokens": len(prompt_ids) + len(response_ids),
        }

    def encode_parity(self, case: Mapping[str, Any]) -> dict:
        ids = self._render(str(case["prompt"]))[: self.max_parity]
        require(ids, "parity", "empty parity prompt")
        return {"prompt_ids": ids}

    def _features(self, hidden_states: Sequence[Any], prompt_length: int) -> Any:
        pooled = []
        for site in self.sites:
            index = site + 1
            require(index < len(hidden_states), "sites",
                    f"hidden state {index} is absent for site {site}")
            value = hidden_states[index][:, :prompt_length, :].float()
            pooled.append(value.mean(dim=1) @ self.projection.T)
        return self.torch.stack(pooled, dim=0).mean(dim=0)

    def _native_forward(self, ids: Sequence[int]) -> tuple[Any, Any]:
        torch = self.torch
        tensor = torch.tensor([list(ids)], dtype=torch.long,
                              device=self.device)
        mask = torch.ones_like(tensor, dtype=torch.bool)
        with torch.no_grad():
            output = self.adapter.base.model(
                input_ids=tensor, attention_mask=mask, use_cache=False,
                output_hidden_states=True)
        return output.last_hidden_state, output.hidden_states

    def _logits(self, ids: Sequence[int], prompt_length: int, *,
                enabled: bool) -> Any:
        final, hidden_states = self._native_forward(ids)
        if enabled:
            features = self._features(hidden_states, prompt_length)
            slot = self.bank.expert(0)(features)
            port = self.adapter.plastic_ports[str(self.sites[0])]
            final = final + port(final, slot, enabled=True)
        return self.adapter.base.lm_head(final)

    # -- teaching ----------------------------------------------------------

    def train_step(self, encoding: Mapping[str, Any], *, weight: float,
                   arm: str) -> dict:
        from replica_lab.dc_response_objective import (
            answer_span_objective, audit_objective_signal, shift_labels)

        torch = self.torch
        prompt_ids = [int(item) for item in encoding["prompt_ids"]]
        response_ids = [int(item) for item in encoding["response_ids"]]
        shifted, masks = shift_labels(
            prompt_ids=prompt_ids, response_ids=response_ids,
            answer_span=tuple(encoding["answer_span"]),
            eos_id=self.eos_id)
        input_ids = prompt_ids + response_ids[:-1]
        final, hidden_states = self._native_forward(input_ids)
        features = self._features(hidden_states, len(prompt_ids))
        slot = self.bank.expert(0)(features)
        port = self.adapter.plastic_ports[str(self.sites[0])]
        final = final + port(final, slot, enabled=True)
        logits = self.adapter.base.lm_head(final)
        start = len(prompt_ids) - 1
        response_logits = logits[0, start:start + len(response_ids)]
        require(tuple(response_logits.shape) == (len(response_ids),
                                                 int(logits.shape[-1])),
                "objective", "response logits are misaligned")
        labels = torch.tensor(shifted, dtype=torch.long,
                              device=self.device)
        objective = answer_span_objective(
            response_logits, labels, masks, answer_span_ce_weight=weight)
        loss = objective["loss"]
        require(bool(torch.isfinite(loss)), "training", "loss is nonfinite")
        self.bank.zero_gradients()
        loss.backward()
        parameters = [parameter for _, parameter
                      in self.bank.expert(0).named_parameters()]
        for name, parameter in self.bank.expert(0).named_parameters():
            require(parameter.grad is not None, "gradient_scope",
                    f"{name} received no gradient")
            require(bool(torch.isfinite(parameter.grad).all()),
                    "gradient_scope", f"nonfinite gradient for {name}")
        for name, parameter in self.adapter.named_parameters():
            require(parameter.grad is None, "gradient_scope",
                    f"frozen backbone parameter {name} acquired a gradient")
        raw_gradient_norm = float(torch.sqrt(sum(
            (parameter.grad.detach() ** 2).sum()
            for parameter in parameters)).detach().cpu())
        clip_scale = min(1.0, 1.0 / max(raw_gradient_norm, 1e-12))
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        clip_scale_applied = 1.0 if raw_gradient_norm <= 1.0 else clip_scale
        before_step = {name: parameter.detach().clone()
                       for name, parameter in self.bank.expert(0).named_parameters()}
        self.optimizer.step()
        changed = [f"expert_0.{name}"
                   for name, parameter in self.bank.expert(0).named_parameters()
                   if not torch.equal(before_step[name], parameter)]
        self.bank.zero_gradients()
        audit = audit_objective_signal(
            masks=masks, answer_span_ce_weight=weight, logits=None,
            label_ids=None)
        return {
            "loss": float(loss.detach()),
            "full_response_ce": objective["full_response_ce"],
            "answer_span_ce": objective["answer_span_ce"],
            "applied_term": objective["applied_term"],
            "audit": audit,
            "raw_gradient_norm": raw_gradient_norm,
            "clip_scale": clip_scale_applied,
            "changed_trainable_names": changed,
            "physical_work": {
                "training_forwards": 1,
                "backward_calls": 1,
                "optimizer_steps": 1,
                "teaching_processed_positions": len(input_ids),
            },
        }

    # -- evaluation --------------------------------------------------------

    def _p_only_reader(self):
        from replica_lab.dc_p1_p_only_reader import POnlyReader
        current = self._state_sha({
            name: parameter.detach().clone()
            for name, parameter in self.bank.expert(0).named_parameters()})
        if self._reader is None or self._reader.reader_state_sha256 != current:
            self._reader = POnlyReader(self)
        return self._reader


    def estimate_evaluation_work(self, case: Mapping[str, Any]) -> dict:
        return self._p_only_reader().estimate_evaluation_work(case)

    def evaluate(self, case: Mapping[str, Any], *, arm: str,
                 prepared: Mapping[str, Any] | None = None) -> dict:
        return self._p_only_reader().evaluate(case, arm=arm, prepared=prepared)

    def parity_probe(self, example: Mapping[str, Any], *,
                     enabled: bool) -> dict:
        ids = [int(item) for item in example["prompt_ids"]][: self.max_parity]
        require(ids, "parity", "empty parity prompt")
        with self.torch.no_grad():
            logits = self._logits(ids, len(ids), enabled=enabled)
            last = logits[0, -1].detach().to(
                "cpu", self.torch.float32).contiguous()
        digest = hashlib.sha256(last.numpy().tobytes()).hexdigest()
        return {
            "logits_sha256": digest,
            "greedy_ids": [int(last.argmax().item())],
            "processed_positions": len(ids),
        }

    # -- state -------------------------------------------------------------

    def _state_sha(self, state: Mapping[str, Any]) -> str:
        digest = hashlib.sha256()
        for name in sorted(state):
            value = state[name]
            payload = value.detach().to("cpu", self.torch.float32).contiguous()
            digest.update(name.encode())
            digest.update(json.dumps(list(payload.shape)).encode())
            digest.update(payload.numpy().tobytes())
        return digest.hexdigest()

    def _to_cpu(self, value: Any) -> Any:
        torch = self.torch
        if isinstance(value, torch.Tensor):
            return value.detach().to("cpu").clone()
        if isinstance(value, Mapping):
            return {key: self._to_cpu(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._to_cpu(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self._to_cpu(item) for item in value)
        return value

    def _to_device(self, value: Any) -> Any:
        torch = self.torch
        if isinstance(value, torch.Tensor):
            return value.to(self.device)
        if isinstance(value, Mapping):
            return {key: self._to_device(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._to_device(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self._to_device(item) for item in value)
        return value

    def reset_to_parent(self, *, clear_optimizer: bool = True) -> None:
        with self.torch.no_grad():
            for name, parameter in self.bank.expert(0).named_parameters():
                parameter.copy_(self.parent_state[name])
        self.bank.zero_gradients()
        if clear_optimizer:
            self.optimizer.state.clear()
        self.bank.expert_steps = [0]

    def capture_checkpoint(self, meta: Mapping[str, Any]) -> dict:
        expert_state = {
            name: parameter.detach().to("cpu").clone()
            for name, parameter in self.bank.expert(0).named_parameters()}
        optimizer_state = self._to_cpu(self.optimizer.state_dict())
        payload = {
            "format": CHECKPOINT_FORMAT,
            "protocol_id": PROTOCOL_ID,
            "meta": dict(meta),
            "expert_state": expert_state,
            "optimizer_state": optimizer_state,
            "trainable_names": list(self.trainable_names),
            "plastic_spec_sha256": sha256_json(self.plastic_spec.to_dict()),
            "decoder_digest": self.decoder_digest,
            "parent_state_sha256": self.parent_sha256,
            "extension_state_sha256": self._state_sha(expert_state),
            "backbone_saved": False,
            "base_saved": False,
        }
        payload["_estimated_bytes"] = self._payload_bytes(payload)
        return payload

    def _payload_bytes(self, value: Any) -> int:
        torch = self.torch
        if isinstance(value, torch.Tensor):
            return int(value.numel() * value.element_size())
        if isinstance(value, Mapping):
            return sum(self._payload_bytes(item) for item in value.values())
        if isinstance(value, (list, tuple)):
            return sum(self._payload_bytes(item) for item in value)
        return 0

    def _expected_names(self) -> set[str]:
        return {name for name, _ in self.bank.expert(0).named_parameters()}

    def restore_checkpoint(self, payload: Mapping[str, Any],
                           meta: Mapping[str, Any]) -> None:
        require(payload.get("format") == CHECKPOINT_FORMAT, "checkpoint",
                "checkpoint format differs")
        stored = payload.get("meta", {})
        require(isinstance(stored, Mapping), "checkpoint", "meta is missing")
        for key in ("arm_id", "checkpoint_id"):
            if key in meta:
                require(stored.get(key) == meta[key], "checkpoint",
                        f"checkpoint {key} differs")
        require(stored.get("parent_id", PARENT_ID) == PARENT_ID,
                "checkpoint", "checkpoint parent differs")
        require(payload.get("backbone_saved") is False
                and payload.get("base_saved") is False, "checkpoint",
                "a checkpoint must never carry backbone weights")
        state = payload.get("expert_state")
        require(isinstance(state, Mapping)
                and set(state) == self._expected_names(), "checkpoint",
                "checkpoint expert tensor set differs")
        with self.torch.no_grad():
            for name, parameter in self.bank.expert(0).named_parameters():
                parameter.copy_(self._to_device(state[name]))
        optimizer_state = self._to_device(payload.get("optimizer_state"))
        self.optimizer.load_state_dict(optimizer_state)
        self.bank.expert_steps = [0 if stored.get("restored") else int(
            stored.get("updates_completed", 0))]
        require(self._state_sha(state) == payload.get("extension_state_sha256"),
                "checkpoint", "extension state digest differs")
        self.bank.zero_gradients()

    def assert_backbone_unchanged(self) -> dict:
        current = self.adapter.parameter_stamp(original_only=True)
        require(current == self.base_identity, "backbone",
                "native backbone identity changed")
        trainable = [(name, parameter) for name, parameter
                     in self.adapter.named_parameters()
                     if parameter.requires_grad]
        require(not trainable, "backbone",
                f"frozen adapter parameters became trainable: "
                f"{[name for name, _ in trainable]}")
        names = tuple(name for name, parameter
                      in self.bank.expert(0).named_parameters()
                      if parameter.requires_grad)
        require(names == ("gate", "up", "down"), "gradient_scope",
                f"unexpected trainable expert tensors: {names}")
        require(self.adapter.plastic_read_digest() == self.decoder_digest,
                "read_path", "fixed decoder changed")
        return {
            "backbone_unchanged": True,
            "adapter_trainable_parameters": 0,
            "trainable_expert_parameters": 3,
            "trainable_groups": list(self.trainable_names),
            "decoder_digest": self.decoder_digest,
        }

    @property
    def evidence_class(self) -> str:
        return "native_mps"


    def decode_fn(self) -> Callable[[Sequence[int]], str]:
        def decode(ids: Sequence[int]) -> str:
            return self.tokenizer.decode(
                list(ids), skip_special_tokens=True,
                clean_up_tokenization_spaces=False)
        return decode


def build_native_backend(context: RunContext, stage: str) -> Backend:
    """Construct the real native backend; this is the only torch import path."""
    return NativeP1Backend(context, stage)


# ---------------------------------------------------------------------------
# Stage execution


@dataclass
class StageState:
    context: RunContext
    stage: str
    attempt: str
    meter: WorkMeter
    journal: dict

    def save(self) -> None:
        self.journal["cumulative"] = dict(sorted(self.meter.cumulative.items()))
        self.journal["local"] = dict(sorted(self.meter.local.items()))
        self.journal["arm_counters"] = {
            arm: dict(sorted(values.items()))
            for arm, values in sorted(self.meter.arm_counters.items())}
        ambiguous = self.meter.take_ambiguous_work()
        if ambiguous:
            self.journal.setdefault("ambiguous_work", []).extend(ambiguous)
        _write_journal(self.context, self.stage, self.journal)


def _stage_setup(context: RunContext, stage: str, *,
                 resume: bool, enforce_budget: bool = True) -> tuple[StageState, bool]:
    output = context.output
    fit_dir = stage_root(output, "fit")
    if stage == "fit":
        if resume:
            require(output.exists(), "resume", "no existing attempt to resume")
        else:
            require(not output.exists() or not any(output.iterdir()),
                    "fresh_output", f"fit output is not fresh: {output}")
        output.mkdir(parents=True, exist_ok=True)
    else:
        require(output.exists(), "stage_order",
                f"{stage} requires the attempt output {output}")
        require(_load_stage_journal(_journal_path(output, "fit")) is not None,
                "stage_order", "fit journal is missing")
    prior, prior_local, prior_arms = _previous_journal(context, stage)
    journal = _load_stage_journal(_journal_path(output, stage))
    if journal is not None:
        for key, expected in (
                ("protocol_id", PROTOCOL_ID),
                ("config_sha256", context.config_sha256),
                ("manifest_sha256", context.manifest_sha256),
                ("protocol_commit", context.protocol_commit)):
            require(journal.get(key) == expected, "resume",
                    f"stage journal {key} differs")
        require(not journal.get("ambiguous_work"), "resume",
                "ambiguous in-flight native work blocks recovery")
        if journal.get("stage_complete") is True:
            if resume and journal.get("publication_stage_done") is True \
                    and publication.published_stage_done(
                        output, stage, identity=_publication_identity(context)):
                state = StageState(
                    context=context, stage=stage,
                    attempt=journal["attempt_id"],
                    meter=WorkMeter(context.config["budget"],
                                    cumulative=journal.get("cumulative"),
                                    local=journal.get("local")),
                    journal=journal)
                return state, True
            if not resume:
                raise RunnerError(
                    f"{stage} is already complete at {output}; use --resume to "
                    "inspect it or a fresh authorized attempt")
            journal["stage_complete"] = False
            journal["publication_stage_done"] = False
            journal["status"] = "in_progress"
        if not resume:
            raise RunnerError(
                f"{stage} journal at {output} is incomplete; use --resume")
        _require_resume_allowed(context, journal)
        journal["resumes_used"] = int(journal.get("resumes_used", 0)) + 1
    else:
        require(not resume or stage == "fit", "resume",
                f"no {stage} journal to resume")
        journal = _new_stage_journal(
            context, stage, attempt_id(context, stage), prior, prior_local,
            prior_arms)
    meter = WorkMeter(context.config["budget"],
                      cumulative=journal.get("cumulative") or prior,
                      local=journal.get("local") or empty_counters(),
                      arm_counters=journal.get("arm_counters") or {},
                      enforce_budget=enforce_budget)
    state = StageState(context=context, stage=stage,
                       attempt=journal["attempt_id"], meter=meter,
                       journal=journal)

    if resume:
        meter.charge({
            "restarts": 1,
            "recovery_reloads": 1,
            "restart_model_loads": 1,
            "restart_tokenizer_loads": 1,
        })
    state.save()
    return state, False
def _construct_backend(state: StageState, stage: str,
                       backend: Backend | None) -> Backend:
    if backend is not None:
        reservation = state.meter.reserve(backend.startup_work())
        try:
            work = backend.startup_work()
        except RunInterrupted:
            state.save()
            raise
        except BaseException as error:
            state.meter.abort(reservation, error)
            state.save()
            raise
        state.meter.reconcile(reservation, work)
        state.save()
        return backend
    reservation = state.meter.reserve({"model_loads": 1, "tokenizer_loads": 1})
    try:
        native = build_native_backend(state.context, stage)
    except BaseException as error:
        state.meter.abort(reservation, error)
        state.save()
        raise
    state.meter.reconcile(reservation, native.startup_work())
    state.save()
    return native



def _startup(state: StageState, backend: Backend) -> None:
    state.meter.charge(backend.startup_work())
    state.save()


def _teaching_audit(encodings: Sequence[Mapping[str, Any]],
                    weight: float) -> dict:
    from replica_lab.dc_response_objective import audit_teaching_rows

    rows = [{
        "input_ids": [int(item) for item in encoding["prompt_ids"]]
        + [int(item) for item in encoding["response_ids"]],
        "response_ids": [int(item) for item in encoding["response_ids"]],
        "answer_span": list(encoding["answer_span"]),
    } for encoding in encodings]
    return audit_teaching_rows(rows, answer_span_ce_weight=weight)


def _arm_journal(journal: dict, arm: str) -> dict:
    return journal.setdefault("arms", {}).setdefault(arm, {
        "events_completed": [],
        "optimizer_attempts": [],
        "updates_completed": 0,
        "no_op_optimizer_steps": 0,
        "checkpoint": None,
        "trained": False,
        "restored": False,
        "complete": False,
        "generated_cases": [],
    })


def checkpoint_relative(stage: str, arm: str) -> str:
    return f"{stage}/checkpoints/{arm}.pt"

def _validate_checkpoint(payload: Mapping[str, Any], arm_spec: Mapping[str, Any],
                         teaching: Sequence[Mapping[str, Any]],
                         backend: Backend, *, require_complete: bool = True) -> None:
    stored = payload.get("meta", {})
    require(isinstance(stored, Mapping), "checkpoint", "checkpoint meta is missing")
    require(stored.get("answer_span_ce_weight")
            == float(arm_spec.get("answer_span_ce_weight", 0.0)), "checkpoint",
            "checkpoint objective weight differs")
    require(stored.get("parent_state_sha256")
            == getattr(backend, "parent_sha256", "cpu_fixture_parent"),
            "checkpoint", "checkpoint parent state differs")
    completed = [int(item) for item in stored.get("events_completed", [])]
    require(completed == list(range(len(completed))), "checkpoint",
            "checkpoint event history is not a prefix")
    require(int(stored.get("updates_completed", -1)) == len(completed),
            "checkpoint", "checkpoint update count differs from event history")
    if require_complete:
        require(completed == list(range(int(arm_spec.get("updates", 0)))),
                "checkpoint", "checkpoint is not the declared complete arm")
    require([int(teaching[index]["event_ordinal"]) for index in completed]
            == list(stored.get("event_ids", [])), "checkpoint",
            "checkpoint event IDs differ")
    require([sha256_json(teaching[index]) for index in completed]
            == list(stored.get("event_digests", [])), "checkpoint",
            "checkpoint event content digests differ")
    require(stored.get("checkpoint_id_before")
            == teaching[0]["checkpoint_id_before"], "checkpoint",
            "checkpoint initial event identity differs")
    require(stored.get("checkpoint_id_after")
            == teaching[completed[-1]]["checkpoint_id_after"], "checkpoint",
            "checkpoint final event identity differs")
    if stored.get("restored") is True:
        require(payload.get("extension_state_sha256")
                == getattr(backend, "parent_sha256", "cpu_fixture_parent"),
                "checkpoint", "reset checkpoint did not restore exact parent")


def _write_arm_checkpoint(state: StageState, publish: Any, relative: str,
                          payload: Mapping[str, Any], arm_key: str | None,
                          *, kind: str):
    writes = []
    reservation = state.meter.reserve({
        "checkpoint_writes": 1,
        "per_event_checkpoint_writes" if kind == "per_event"
        else "artifact_checkpoint_writes": 1,
        "temporary_storage_bytes": int(payload.get("_estimated_bytes", 0)),
    }, arm=arm_key)
    try:
        artifact = publish.write_torch(
            state.context.output, relative, payload, reuse_identical=True,
            on_write=lambda: writes.append(1))
    except BaseException as error:
        state.meter.abort(reservation, error)
        state.save()
        raise
    actual = {
        "checkpoint_writes": len(writes),
        "per_event_checkpoint_writes" if kind == "per_event"
        else "artifact_checkpoint_writes": len(writes),
        "temporary_storage_bytes": int(artifact.bytes) if writes else 0,
    }
    state.meter.reconcile(reservation, actual)
    state.save()
    return artifact




def _train_arm(state: StageState, backend: Backend,
               arm_spec: Mapping[str, Any], teaching: Sequence[dict], *,
               resume: bool, publish: Any, check_arm: bool,
               budget_arm: str | None = None) -> dict:
    arm = arm_spec["arm_id"]
    arm_key = (budget_arm or arm) if check_arm else None
    record = _arm_journal(state.journal, arm)
    updates = int(arm_spec.get("updates", 0))
    weight = float(arm_spec.get("answer_span_ce_weight", 0.0))
    require(weight in (0.0, 0.5), "config",
            "v2 arms use answer_span_ce_weight 0.0 or 0.5")
    encodings = [backend.encode_teaching(row) for row in teaching[:updates]]
    require(len(encodings) == updates, "config",
            f"arm {arm} teaching count differs from declared updates")
    relative = checkpoint_relative(state.stage, arm)
    if record.get("checkpoint") and int(record.get("updates_completed", 0)) > 0:
        payload = publish.load_torch(state.context.output, relative)
        _validate_checkpoint(payload, arm_spec, teaching, backend,
                             require_complete=bool(record.get("complete")))
        backend.restore_checkpoint(
            payload, {"arm_id": arm, "stage": state.stage,
                      "checkpoint_id": arm_spec["checkpoint_id"]})
        stored = payload.get("meta", {})
        state.meter.adopt(
            _counters_from_mapping(stored.get("counters", {}), "checkpoint"),
            _counters_from_mapping(stored.get("local_counters", {}),
                                   "checkpoint-local"),
            arm=arm_key,
            arm_counters=_counters_from_mapping(
                stored.get("arm_counters", {}), "checkpoint-arm"))
        if record.get("complete"):
            completed = [int(item) for item in record.get("events_completed", [])]
            require(completed == list(range(updates))
                    and int(record.get("updates_completed", -1)) == updates,
                    "resume", f"arm {arm} completed an incorrect update sequence")
            return record
    else:
        require(not record.get("updates_completed"),
                "resume",
                f"arm {arm} journals optimizer work without a checkpoint")
        backend.reset_to_parent(clear_optimizer=True)
        record["events_completed"] = []
        record["updates_completed"] = 0
        record["trained"] = False
        record["restored"] = False
    if updates == 0:
        record["complete"] = True
        record["learning_evidence"] = learning_success_evidence(
            declared_updates=updates, execution_complete=True,
            optimizer_steps=0, changed_trainable_names=[])
        state.save()
        return record
    completed = [int(item) for item in record.get("events_completed", [])]
    require(completed == list(range(len(completed))), "resume",
            f"arm {arm} event journal is not a prefix")
    restart_index = len(completed)
    for index in range(len(completed), len(encodings)):
        encoding = encodings[index]
        restart = bool(resume and index == restart_index
                       and index in record.get("optimizer_attempts", []))
        work = ({
            "training_forwards": 1,
            "backward_calls": 1,
            "optimizer_steps": 1,
            "teaching_processed_positions": int(encoding["input_tokens"]),
            "restart_training_forwards": 1,
            "restart_backward_calls": 1,
            "restart_optimizer_steps": 1,
            "restart_processed_text_positions": int(
                encoding["input_tokens"]),
        } if restart else {
            "training_forwards": 1, "backward_calls": 1,
            "optimizer_steps": 1,
            "teaching_processed_positions": int(encoding["input_tokens"]),
        })
        reservation = state.meter.reserve(work, arm=arm_key)
        attempts = record.setdefault("optimizer_attempts", [])
        if index not in attempts:
            attempts.append(index)
            state.save()
        try:
            result = backend.train_step(encoding, weight=weight, arm=arm)
        except RunInterrupted:
            state.save()
            raise
        except BaseException as error:
            state.meter.abort(reservation, error)
            state.save()
            raise
        actual = result.get("physical_work", work)
        changed = result.get("changed_trainable_names")
        if restart and "training_forwards" in actual:
            actual = dict(actual)
            actual.update({
                "restart_training_forwards": int(actual["training_forwards"]),
                "restart_backward_calls": int(actual["backward_calls"]),
                "restart_optimizer_steps": int(actual["optimizer_steps"]),
                "restart_processed_text_positions": int(
                    actual["teaching_processed_positions"]),
            })
        # A no-op still performed physical optimizer work. Charge it before
        # rejecting the learning event so a later resume is an explicit retry.
        state.meter.reconcile(reservation, actual)
        if isinstance(changed, (list, tuple)) and not changed:
            record["no_op_optimizer_steps"] = int(
                record.get("no_op_optimizer_steps", 0)) + 1
        state.save()
        require(isinstance(changed, (list, tuple)), "learning",
                f"arm {arm} did not report changed trainable tensors")
        require(bool(changed), "learning",
                f"arm {arm} optimizer step changed no trainable tensor")
        require(all(isinstance(name, str) and name for name in changed)
                and set(changed) <= set(state.context.trainable_groups()),
                "learning", f"arm {arm} reported an undeclared trainable tensor")
        record["changed_trainable_names"] = sorted(
            set(record.get("changed_trainable_names", [])) | set(changed))
        state.save()
        completed.append(index)
        record["events_completed"] = list(completed)
        record["updates_completed"] = len(completed)
        record["trained"] = True
        record["last_loss"] = result.get("loss")
        payload = backend.capture_checkpoint({
            "stage": state.stage, "arm_id": arm,
            "checkpoint_id": arm_spec["checkpoint_id"],
            "events_completed": list(completed),
            "updates_completed": len(completed),
            "counters": state.meter.cumulative,
            "local_counters": state.meter.local,
            "arm_counters": state.meter._arm(arm_key) if arm_key else {},
            "parent_id": PARENT_ID,
            "changed_trainable_names": list(record["changed_trainable_names"]),
            "no_op_optimizer_steps": int(record.get("no_op_optimizer_steps", 0)),
            "answer_span_ce_weight": weight,
            "parent_state_sha256": getattr(
                backend, "parent_sha256", "cpu_fixture_parent"),
            "event_ids": [int(teaching[index]["event_ordinal"])
                          for index in completed],
            "event_digests": [sha256_json(teaching[index])
                              for index in completed],
            "checkpoint_id_before": teaching[0]["checkpoint_id_before"],
            "checkpoint_id_after": teaching[completed[-1]]["checkpoint_id_after"],
        })
        artifact = _write_arm_checkpoint(
            state, publish, relative, payload, arm_key, kind="per_event")
        record["checkpoint"] = artifact.as_dict()
        state.save()
    if arm_spec.get("restore_parent_before_evaluation"):
        backend.reset_to_parent(clear_optimizer=True)
        record["restored"] = True
        payload = backend.capture_checkpoint({
            "stage": state.stage, "arm_id": arm,
            "checkpoint_id": arm_spec["checkpoint_id"],
            "events_completed": list(record["events_completed"]),
            "updates_completed": int(record.get("updates_completed", 0)),
            "restored": True,
            "counters": state.meter.cumulative,
            "local_counters": state.meter.local,
            "arm_counters": state.meter._arm(arm_key) if arm_key else {},
            "parent_id": PARENT_ID,
            "changed_trainable_names": list(record["changed_trainable_names"]),
            "no_op_optimizer_steps": int(record.get("no_op_optimizer_steps", 0)),
            "answer_span_ce_weight": weight,
            "parent_state_sha256": getattr(
                backend, "parent_sha256", "cpu_fixture_parent"),
            "event_ids": [int(teaching[index]["event_ordinal"])
                          for index in completed],
            "event_digests": [sha256_json(teaching[index])
                              for index in completed],
            "checkpoint_id_before": teaching[0]["checkpoint_id_before"],
            "checkpoint_id_after": teaching[completed[-1]]["checkpoint_id_after"],
        })
        artifact = _write_arm_checkpoint(
            state, publish, relative, payload, arm_key, kind="artifact")
        record["checkpoint"] = artifact.as_dict()
    record["complete"] = True
    counters = state.meter._arm(arm_key) if arm_key else state.meter.local
    record["learning_evidence"] = learning_success_evidence(
        declared_updates=updates, execution_complete=record["complete"],
        optimizer_steps=int(counters.get("optimizer_steps", 0)),
        retry_optimizer_steps=int(counters.get("restart_optimizer_steps", 0)),
        no_op_optimizer_steps=int(record.get("no_op_optimizer_steps", 0)),
        changed_trainable_names=record.get("changed_trainable_names", []),
        restored_parent=bool(arm_spec.get("restore_parent_before_evaluation")))
    require(completed == list(range(updates))
            and int(record.get("updates_completed", -1)) == updates,
            "resume", f"arm {arm} completed an incorrect update sequence")
    state.meter.charge({"retained_checkpoint_artifacts": (
        int(state.meter.cumulative.get("retained_checkpoint_artifacts", 0)) + 1)})
    state.save()
    return record


def _generate_arm(state: StageState, backend: Backend,
                  arm_spec: Mapping[str, Any], cases: Sequence[dict], *,
                  resume: bool, publish: Any, check_arm: bool,
                  evidence_class: str, budget_arm: str | None = None) -> list[dict]:
    arm = arm_spec["arm_id"]
    arm_key = (budget_arm or arm) if check_arm else None
    record = _arm_journal(state.journal, arm)
    relative = f"{state.stage}/predictions_{arm}.jsonl"
    existing: list[dict] = []
    if (state.context.output / relative).is_file():
        existing = read_jsonl(state.context.output / relative)
    done = [str(row["case_id"]) for row in existing]
    recorded = [str(item) for item in record.get("generated_cases", [])]
    require(not recorded or recorded == done[:len(recorded)], "resume",
            f"arm {arm} prediction journal differs from the published file")
    by_case = {case["case_id"]: case for case in cases}
    require(set(done) <= set(by_case), "resume",
            f"arm {arm} predicted an unknown case")
    rows = list(existing)
    for case in cases:
        if case["case_id"] in done:
            continue
        plan = backend.estimate_evaluation_work(case)
        reservation = state.meter.reserve(plan["work"], arm=arm_key)
        try:
            result = backend.evaluate(case, arm=arm, prepared=plan)
        except RunInterrupted:
            state.save()
            raise
        except BaseException as error:
            state.meter.abort(reservation, error)
            state.save()
            raise
        state.meter.reconcile(reservation, {
            "generation_requests": 1,
            "generation_prefill_forwards": 1,
            "generation_decode_forwards": int(result["decode_forwards"]),
            "autoregressive_forwards": 1 + int(result["decode_forwards"]),
            "generated_tokens": int(result["generated_tokens"]),
            "generation_processed_positions": int(
                result["processed_positions"]),
        })
        state.save()
        row = prediction_row(
            context=state.context, stage=state.stage, arm_spec=arm_spec,
            case=case, result=result, decode_fn=backend.decode_fn,
            evidence_class=evidence_class)
        rows.append(row)
        done.append(case["case_id"])
        record["generated_cases"] = list(done)
        publish.write_jsonl(state.context.output, relative, rows)
        state.save()
    publish.write_jsonl(state.context.output, relative, rows)
    record["generated_cases"] = [row["case_id"] for row in rows]
    state.save()
    return rows


def _probe(state: StageState, backend: Backend,
           example: Mapping[str, Any], *, enabled: bool) -> dict:
    reservation = state.meter.reserve({
        "parity_probe_forwards": 1,
        "parity_processed_positions": len(example["prompt_ids"])})
    try:
        result = backend.parity_probe(example, enabled=enabled)
    except RunInterrupted:
        state.save()
        raise
    except BaseException as error:
        state.meter.abort(reservation, error)
        state.save()
        raise
    state.meter.reconcile(reservation, {
        "parity_probe_forwards": 1,
        "parity_processed_positions": int(result["processed_positions"]),
    })
    state.save()
    return result


def _parity(state: StageState, backend: Backend, example: Mapping[str, Any]) -> dict:
    if state.journal.get("parity") is not None:
        return state.journal["parity"]
    baseline = _probe(state, backend, example, enabled=False)
    extension = _probe(state, backend, example, enabled=True)
    report = {
        "format": "replica.dc_p1_native_response_v2.parity.v1",
        "baseline_logits_sha256": baseline["logits_sha256"],
        "extension_logits_sha256": extension["logits_sha256"],
        "baseline_tokens": baseline["greedy_ids"],
        "extension_tokens": extension["greedy_ids"],
        "logits_equal": baseline["logits_sha256"] == extension["logits_sha256"],
        "greedy_equal": baseline["greedy_ids"] == extension["greedy_ids"],
        "before_teaching": True,
        "after_reset": None,
    }
    require(report["logits_equal"] and report["greedy_equal"], "parity",
            "disabled and zero-read paths do not match before teaching")
    state.journal["parity"] = report
    state.save()
    return report


def _parity_after_reset(state: StageState, backend: Backend,
                        example: Mapping[str, Any]) -> dict:
    parity = state.journal.get("parity")
    require(isinstance(parity, Mapping), "parity",
            "after-reset parity has no before-teaching baseline")
    if parity.get("after_reset") is not None:
        return parity["after_reset"]
    restored = _probe(state, backend, example, enabled=True)
    after = {
        "logits_sha256": restored["logits_sha256"],
        "tokens": restored["greedy_ids"],
        "matches_parent": (
            restored["logits_sha256"] == parity["baseline_logits_sha256"]
            and restored["greedy_ids"] == parity["baseline_tokens"]),
    }
    require(after["matches_parent"], "parity",
            "reset_P does not restore exact parent behavior")
    parity["after_reset"] = after
    state.save()
    return after


# ---------------------------------------------------------------------------
# Publication


def _artifact_cap(context: RunContext) -> int | None:
    budget = context.config.get("budget", {})
    for holder in (budget.get("authorized_run_ceiling", {}),
                   budget.get("physical_limits", {})):
        for key in ("artifact_bytes_max", "max_artifact_bytes"):
            value = holder.get(key)
            if isinstance(value, int) and value > 0:
                return value
    return None


class _MeteredPublication:
    """Artifact module facade that observes every atomic write."""

    def __init__(self, module: Any, meter: Any) -> None:
        self._module = module
        self.meter = meter

    def __getattr__(self, name: str) -> Any:
        return getattr(self._module, name)

    def write_json(self, *args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("meter", self.meter)
        return self._module.write_json(*args, **kwargs)

    def write_jsonl(self, *args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("meter", self.meter)
        return self._module.write_jsonl(*args, **kwargs)

    def write_torch(self, *args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("meter", self.meter)
        return self._module.write_torch(*args, **kwargs)


def _publication(context: RunContext):
    from replica_lab import artifact_publication as publication

    meter = publication.LiveByteMeter(context.output, label="dc_p1_v2")
    meter.observe()
    journal = publication.Journal(
        context.output, identity=_publication_identity(context),
        cap_bytes=_artifact_cap(context), meter=meter)
    return _MeteredPublication(publication, meter), journal


def _publish_stage(*, context: RunContext, stage: str,
                   relatives: Sequence[str], publication, journal) -> list:
    artifacts = []
    for relative in relatives:
        path = context.output / relative
        if path.is_file():
            artifacts.append(publication.Artifact.of(context.output, relative))
    journal.record(stage, artifacts)
    meter = publication.meter
    meter.observe()
    journal.record_peak(meter.report())
    cap = _artifact_cap(context)
    if cap is not None:
        measured = publication.total_bytes(context.output)
        if measured > cap:
            publication.write_overflow_evidence(
                context.output, context=f"dc_p1_v2:{stage}",
                cap_bytes=cap, measured_bytes=measured,
                files=publication.inventory(context.output))
            raise RunnerError(
                f"artifact budget exceeded: {measured} > {cap}")
    return artifacts


def _stage_relatives(context: RunContext, stage: str,
                     arms: Sequence[Mapping[str, Any]],
                     extra: Sequence[str] = ()) -> list[str]:
    # The runner journal is deliberately not a publication artifact: it can
    # change after a stage is published and must never invalidate a sealed
    # publication record by self-reference.
    relatives = [f"{stage}/receipt.json"]
    for arm in arms:
        relatives.append(f"{stage}/predictions_{arm['arm_id']}.jsonl")
        if int(arm.get("updates", 0)) > 0:
            relatives.append(checkpoint_relative(stage, arm["arm_id"]))
    report = ("fit/fit_report.json" if stage == "fit"
              else f"{stage}/{stage}_report.json")
    relatives.append(report)
    if (context.output / stage / "admission.json").is_file():
        relatives.append(f"{stage}/admission.json")
    relatives.extend(extra)
    return relatives


# ---------------------------------------------------------------------------
# Public stage entry points


def _with_output(context: RunContext, output: str | Path) -> RunContext:
    target = Path(output)
    if not target.is_absolute():
        target = ROOT / target
    # The canonical guard output is the stage directory.  Accept either the
    # stage directory or the attempt root and always store the attempt root.
    if target.name in STAGES:
        target = target.parent
    return RunContext(
        config_path=context.config_path, config=context.config,
        config_sha256=context.config_sha256, prepared=context.prepared,
        manifest=context.manifest, manifest_sha256=context.manifest_sha256,
        authorization=context.authorization,
        authorization_sha256=context.authorization_sha256,
        protocol_commit=context.protocol_commit, output=target)


def run_fit(*, config_path: str | Path, prepared: str | Path,
            authorization_path: str | Path | None,
            protocol_commit: str | None, output: str | Path,
            resume: bool = False, backend: Backend | None = None,
            require_head: bool = True,
            commit_verifier: Callable[[str, str], bytes | None]
            | None = None) -> dict:
    context, verified = verify_authorization(
        config_path=config_path, prepared=prepared,
        authorization_path=authorization_path,
        protocol_commit=protocol_commit, require_head=require_head,
        stage="fit", commit_verifier=commit_verifier)
    return _run_stage("fit", _with_output(context, output), verified,
                      resume=resume, backend=backend)


def run_development(*, config_path: str | Path, prepared: str | Path,
                    authorization_path: str | Path | None,
                    protocol_commit: str | None, output: str | Path,
                    resume: bool = False, backend: Backend | None = None,
                    require_head: bool = True,
                    commit_verifier: Callable[[str, str], bytes | None]
                    | None = None) -> dict:
    context, verified = verify_authorization(
        config_path=config_path, prepared=prepared,
        authorization_path=authorization_path,
        protocol_commit=protocol_commit, require_head=require_head,
        stage="development", commit_verifier=commit_verifier)
    return _run_stage("development", _with_output(context, output), verified,
                      resume=resume, backend=backend)


def run_final(*, config_path: str | Path, prepared: str | Path,
              authorization_path: str | Path | None,
              protocol_commit: str | None, output: str | Path,
              resume: bool = False, backend: Backend | None = None,
              require_head: bool = True,
              commit_verifier: Callable[[str, str], bytes | None]
              | None = None) -> dict:
    context, verified = verify_authorization(
        config_path=config_path, prepared=prepared,
        authorization_path=authorization_path,
        protocol_commit=protocol_commit, require_head=require_head,
        stage="final", commit_verifier=commit_verifier)
    return _run_stage("final", _with_output(context, output), verified,
                      resume=resume, backend=backend)


def run_stage(stage: str, **kwargs: Any) -> dict:
    require(stage in STAGES, "stage", f"unknown stage {stage!r}")
    if stage == "fit":
        return run_fit(**kwargs)
    if stage == "development":
        return run_development(**kwargs)
    return run_final(**kwargs)


def _run_stage(stage: str, context: RunContext, verified: Mapping[str, Any],
               *, resume: bool, backend: Backend | None) -> dict:
    if stage == "fit":
        return _run_fit(context, verified, resume=resume, backend=backend)
    if stage == "development":
        return _run_development(context, verified, resume=resume,
                                backend=backend)
    return _run_final(context, verified, resume=resume, backend=backend)


def _already_complete(state: StageState, verified: Mapping[str, Any]) -> dict:
    result = dict(state.journal.get("result") or {})
    result.setdefault("status", state.journal.get("status", "complete"))
    result.setdefault("stage", state.stage)
    result["assertion"] = verified
    result["already_complete"] = True
    result["counters"] = state.meter.report()
    return result


def _run_fit(context: RunContext, verified: Mapping[str, Any], *,
             resume: bool, backend: Backend | None) -> dict:
    execution_class = "native_mps" if backend is None else "cpu_fixture"
    state, complete = _stage_setup(context, "fit", resume=resume,
                                   enforce_budget=backend is None)
    if complete:
        return _already_complete(state, verified)
    publication, publication_journal = _publication(context)
    backend = _construct_backend(state, "fit", backend)
    try:
        teaching = teaching_rows(context, "fit")
        arms = context.arm_specs("fit")
        cases = stage_cases(context, "fit")
        encodings = [backend.encode_teaching(row) for row in teaching]
        state.journal["objective_audit"] = _teaching_audit(encodings, 0.5)
        parity_example = backend.encode_parity(cases[0])
        _parity(state, backend, parity_example)
        predictions: list[dict] = []
        for arm_spec in arms:
            _train_arm(state, backend, arm_spec, teaching, resume=resume,
                       publish=publication, check_arm=True)
            if arm_spec.get("restore_parent_before_evaluation"):
                _parity_after_reset(state, backend, parity_example)
            predictions.extend(_generate_arm(
                state, backend, arm_spec, cases, resume=resume,
                publish=publication, check_arm=True,
                evidence_class=execution_class))
        labels = stage_labels(context, "fit")
        report = _score_report(
            context=context, stage="fit", cases=cases, labels=labels,
            predictions=predictions, arm_specs=arms,
            objective_audit=state.journal["objective_audit"],
            decode_fn=backend.decode_fn,
            evidence_class=execution_class)
        report["learning_evidence"] = {
            spec["arm_id"]: dict(state.journal["arms"][spec["arm_id"]].get(
                "learning_evidence", {})) for spec in arms}
        report["attempt_id"] = state.attempt
        publication.write_json(context.output, "fit/fit_report.json", report)
        receipt, admission, error, admitted = _receipt_and_admission(
            context=context, stage="fit", report=report,
            expectation=_expectation(context), attempt=state.attempt,
            publish=publication, decode_fn=backend.decode_fn)
        if admitted is not None:
            publication.write_json(context.output, "fit/admission.json",
                                   admission.state())
        state.journal.update({
            "stage_complete": False, "publication_stage_done": False,
            "status": report["status"], "admitted": admitted is not None,
            "admission_error": None if error is None else str(error),
            "access": admission.spy.state(),
        })
        result = {
            "status": report["status"], "stage": "fit",
            "assertion": verified, "attempt_id": state.attempt,
            "admitted": admitted is not None,
            "admission_error": None if error is None else str(error),
            "report_path": str(stage_root(context.output, "fit")
                               / "fit_report.json"),
            "receipt_path": str(stage_root(context.output, "fit")
                                / "receipt.json"),
            "counters": state.meter.report(),
        }
        state.journal["result"] = result
        state.save()
        artifacts = _publish_stage(
            context=context, stage="fit",
            relatives=_stage_relatives(context, "fit", arms),
            publication=publication, journal=publication_journal)
        result["artifacts"] = [artifact.as_dict() for artifact in artifacts]
        result["backbone"] = backend.assert_backbone_unchanged()
        state.journal.update({
            "stage_complete": True, "publication_stage_done": True})
        state.journal["result"] = result
        state.save()
        return result
    finally:
        backend.close()


def _load_fit_checkpoint(state: StageState, backend: Backend,
                         arm_spec: Mapping[str, Any], publication: Any) -> None:
    arm = arm_spec["arm_id"]
    relative = checkpoint_relative("fit", arm)
    if int(arm_spec.get("updates", 0)) == 0:
        backend.reset_to_parent(clear_optimizer=True)
        return
    require((state.context.output / relative).is_file(), "checkpoint",
            f"fit checkpoint for arm {arm!r} is missing")
    payload = publication.load_torch(state.context.output, relative)
    _validate_checkpoint(payload, arm_spec, teaching_rows(state.context, "fit"),
                         backend)
    backend.restore_checkpoint(
        payload, {"arm_id": arm, "stage": "fit",
                  "checkpoint_id": arm_spec["checkpoint_id"]})


def _run_development(context: RunContext, verified: Mapping[str, Any], *,
                     resume: bool, backend: Backend | None) -> dict:
    execution_class = "native_mps" if backend is None else "cpu_fixture"
    # Fit admission is verified before any native import or dev label read.
    _prior_admission(
        context, "development", evidence_class=execution_class,
        decode_fn=backend.decode_fn if backend is not None else None)
    state, complete = _stage_setup(
        context, "development", resume=resume,
        enforce_budget=backend is None)
    if complete:
        return _already_complete(state, verified)
    publication, publication_journal = _publication(context)
    backend = _construct_backend(state, "development", backend)
    try:
        arms = context.arm_specs("fit")
        cases = stage_cases(context, "development")
        fit_encodings = [
            backend.encode_teaching(row)
            for row in teaching_rows(context, "fit")]
        state.journal["objective_audit"] = _teaching_audit(fit_encodings, 0.5)
        predictions: list[dict] = []
        for arm_spec in arms:
            _load_fit_checkpoint(state, backend, arm_spec, publication)
            predictions.extend(_generate_arm(
                state, backend, arm_spec, cases, resume=resume,
                publish=publication, check_arm=True,
                evidence_class=execution_class))
        labels = stage_labels(context, "development")
        report = _score_report(
            context=context, stage="development", cases=cases, labels=labels,
            predictions=predictions, arm_specs=arms,
            objective_audit=state.journal["objective_audit"],
            decode_fn=backend.decode_fn,
            evidence_class=execution_class)
        report["attempt_id"] = state.attempt
        publication.write_json(context.output,
                                   "development/development_report.json",
                                   report)
        receipt, admission, error, admitted = _receipt_and_admission(
            context=context, stage="development", report=report,
            expectation=_expectation(context), attempt=state.attempt,
            publish=publication, decode_fn=backend.decode_fn)
        if admitted is not None:
            publication.write_json(context.output,
                                   "development/admission.json",
                                   admission.state())
        state.journal.update({
            "stage_complete": False, "publication_stage_done": False,
            "status": report["status"], "admitted": admitted is not None,
            "admission_error": None if error is None else str(error),
            "access": admission.spy.state(),
        })
        result = {
            "status": report["status"], "stage": "development",
            "assertion": verified, "attempt_id": state.attempt,
            "admitted": admitted is not None,
            "admission_error": None if error is None else str(error),
            "report_path": str(stage_root(context.output, "development")
                               / "development_report.json"),
            "receipt_path": str(stage_root(context.output, "development")
                                / "receipt.json"),
            "counters": state.meter.report(),
        }
        state.journal["result"] = result
        state.save()
        artifacts = _publish_stage(
            context=context, stage="development",
            relatives=_stage_relatives(context, "development", arms),
            publication=publication, journal=publication_journal)
        result["artifacts"] = [artifact.as_dict() for artifact in artifacts]
        result["backbone"] = backend.assert_backbone_unchanged()
        state.journal.update({
            "stage_complete": True, "publication_stage_done": True})
        state.journal["result"] = result
        state.save()
        return result
    finally:
        backend.close()


def _run_final(context: RunContext, verified: Mapping[str, Any], *,
               resume: bool, backend: Backend | None) -> dict:
    execution_class = "native_mps" if backend is None else "cpu_fixture"
    admission = _prior_admission(
        context, "final", evidence_class=execution_class,
        decode_fn=backend.decode_fn if backend is not None else None)
    state, complete = _stage_setup(
        context, "final", resume=resume,
        enforce_budget=backend is None)
    if complete:
        return _already_complete(state, verified)
    admission.require("final")
    state.journal["final_started"] = True
    state.save()
    publication, publication_journal = _publication(context)
    backend = _construct_backend(state, "final", backend)
    try:
        teaching = teaching_rows(context, "final")
        arms = context.arm_specs("final")
        arm_spec = arms[0]
        cases = stage_cases(context, "final")
        encodings = [backend.encode_teaching(row) for row in teaching]
        objective_audit = _teaching_audit(encodings, 0.5)
        state.journal["objective_audit"] = objective_audit
        _train_arm(state, backend, arm_spec, teaching, resume=resume,
                   publish=publication, check_arm=True,
                   budget_arm="treatment_final")
        predictions = _generate_arm(
            state, backend, arm_spec, cases, resume=resume,
            publish=publication, check_arm=True,
            evidence_class=execution_class, budget_arm="treatment_final")
        labels = stage_labels(context, "final")
        report = _score_report(
            context=context, stage="final", cases=cases, labels=labels,
            predictions=predictions, arm_specs=arms,
            objective_audit=objective_audit, decode_fn=backend.decode_fn,
            evidence_class=execution_class)
        report["attempt_id"] = state.attempt
        report["confirmation_only"] = True
        publication.write_json(context.output, "final/final_report.json",
                                   report)
        receipt, admission, error, admitted = _receipt_and_admission(
            context=context, stage="final", report=report,
            expectation=_expectation(context), attempt=state.attempt,
            publish=publication, decode_fn=backend.decode_fn)
        if admitted is not None:
            publication.write_json(context.output, "final/admission.json",
                                   admission.state())
        state.journal.update({
            "stage_complete": False, "publication_stage_done": False,
            "status": report["status"], "admitted": admitted is not None,
            "admission_error": None if error is None else str(error),
            "access": admission.spy.state(),
        })
        result = {
            "status": report["status"], "stage": "final",
            "assertion": verified, "attempt_id": state.attempt,
            "admitted": admitted is not None,
            "admission_error": None if error is None else str(error),
            "report_path": str(stage_root(context.output, "final")
                               / "final_report.json"),
            "receipt_path": str(stage_root(context.output, "final")
                                / "receipt.json"),
            "counters": state.meter.report(),
        }
        state.journal["result"] = result
        state.save()
        publication.meter.observe()
        artifact_budget = {
            "format": "replica.dc_p1_native_response_v2.artifact_budget.v1",
            "cap_bytes": _artifact_cap(context),
            "measured_bytes": publication.total_bytes(context.output),
            "peak": publication.meter.report(),
            "counters": state.meter.report(),
        }
        publication.write_json(context.output, "artifact_budget.json",
                                   artifact_budget)
        artifacts = _publish_stage(
            context=context, stage="final",
            relatives=_stage_relatives(context, "final", arms,
                                       extra=["artifact_budget.json"]),
            publication=publication, journal=publication_journal)
        result["artifacts"] = [artifact.as_dict() for artifact in artifacts]
        result["publication"] = publication_journal.summary()
        result["backbone"] = backend.assert_backbone_unchanged()
        state.journal.update({
            "stage_complete": True, "publication_stage_done": True})
        state.journal["result"] = result
        state.save()
        return result
    finally:
        backend.close()


PUBLIC_API = {
    "module": "replica_lab.dc_p1_native_runner",
    "run_fit": "run_fit",
    "run_development": "run_development",
    "run_final": "run_final",
    "run_stage": "run_stage",
    "verify_authorization": "verify_authorization",
    "WorkMeter": "WorkMeter",
    "NativeP1Backend": "NativeP1Backend",
}

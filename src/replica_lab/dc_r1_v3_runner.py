"""DC-R1 v3 authorized staged runner and cumulative work ledger."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Callable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from dc_protocol_common_20261027 import (
    ProtocolError, read_json, read_jsonl, require, require_committed_bytes,
    sha256_file, sha256_json, tokenizer_binding_sha, validate_manifest,
    validate_tokenizer_binding,
)
from replica_lab import artifact_publication as publication
from replica_lab import dc_r1_v3_scoring as scoring
from replica_lab.dc_r1_v3_backend import NativeR1Backend

PROTOCOL_ID = scoring.PROTOCOL_ID
AUTHORIZATION_FORMAT = "replica.dc_protocol.authorization.v3"
PREDICTION_FORMAT = scoring.PREDICTION_FORMAT
JOURNAL_FORMAT = "replica.dc_r1_native_value.stage_journal.v3"
LEDGER_FORMAT = "replica.dc_r1_native_value.work_ledger.v3"
STAGES = ("dev", "sealed_final")
CONDITIONS = scoring.CONDITIONS
LAYERS = scoring.LAYERS
IMPLEMENTATION_RELATIVES = (
    "configs/dc_r1_native_value_protocol_v3.json",
    "data/dc_r1_native_value_v2_20261027/manifest.json",
    "scripts/dc_r1_native_value_v3_20261028.py",
    "scripts/preflight_dc_r1_native_value_v3_20261028.py",
    "scripts/run_dc_r1_native_value_v3_20261028.py",
    "src/replica_lab/artifact_publication.py",
    "src/replica_lab/dc_r1_v3_backend.py",
    "src/replica_lab/dc_r1_v3_scoring.py",
    "src/replica_lab/dc_r1_v3_runner.py",
    "src/replica_lab/model_identity.py",
    "src/replica_lab/native_readout.py",
)


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class RunContext:
    root: Path
    config_path: Path
    config: dict
    config_sha256: str
    prepared: Path
    manifest: dict
    manifest_sha256: str
    authorization: dict
    authorization_sha256: str
    output: Path
    attempt_id: str
    protocol_commit: str
    binding: dict


class AttemptRegistry:
    """Single-use receipt registry outside all output directories."""

    def __init__(self, authorization_path: Path) -> None:
        self.path = authorization_path.with_suffix(authorization_path.suffix + ".uses.json")

    def register(self, context: RunContext) -> None:
        payload = {
            "format": "replica.dc_r1_native_value.authorization_use.v1",
            "authorization_receipt_sha256": context.authorization_sha256,
            "attempt_id": context.attempt_id,
            "output_root": str(context.output),
            "protocol_commit": context.protocol_commit,
            "created_at_utc": _utc(),
        }
        if self.path.exists():
            previous = read_json(self.path)
            for key in ("authorization_receipt_sha256", "attempt_id", "output_root",
                        "protocol_commit"):
                require(previous.get(key) == payload[key], "single_attempt",
                        f"authorization was already used with different {key}")
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(payload, stream, indent=2, sort_keys=True)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
        except BaseException:
            try:
                self.path.unlink()
            except OSError:
                pass
            raise


def verify_authorization(*, config_path: str | Path, prepared: str | Path,
                        authorization_path: str | Path, output: str | Path,
                        require_head: bool = True,
                        commit_verifier: Callable | None = None,
                        register: bool = True) -> tuple[RunContext, dict]:
    config_path = Path(config_path)
    prepared = Path(prepared)
    authorization_path = Path(authorization_path)
    output = Path(output)
    if not output.is_absolute():
        output = ROOT / output
    output = output.resolve()
    config = scoring.load_config(config_path, prepared=prepared)
    manifest = scoring.validate_prepared(
        config, defer_final_labels=True, prepared=prepared)
    config_sha = sha256_file(config_path)
    authorization = read_json(authorization_path)
    require(authorization.get("format") == AUTHORIZATION_FORMAT,
            "authorization", "authorization format differs")
    require(authorization.get("protocol_id") == PROTOCOL_ID,
            "authorization", "authorization protocol differs")
    require(authorization.get("config_sha256") == config_sha,
            "authorization", "config bytes differ")
    require(authorization.get("prepared_manifest_sha256")
            == manifest["manifest_sha256"], "authorization", "fixture bytes differ")
    commit = authorization.get("protocol_commit")
    require(isinstance(commit, str) and len(commit) == 40
            and all(character in "0123456789abcdef" for character in commit),
            "authorization", "protocol_commit must be full 40-hex")
    if require_head:
        actual = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
            capture_output=True, check=True).stdout.strip()
        require(actual == commit, "authorization", "protocol commit is not HEAD")
        dirty = subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=no"],
            cwd=ROOT, text=True, capture_output=True, check=True).stdout.strip()
        require(not dirty, "authorization", "tracked worktree changes are not allowed")
    for relative in IMPLEMENTATION_RELATIVES:
        require_committed_bytes(
            commit, relative, sha256_file(ROOT / relative), verifier=commit_verifier)
    attempt_id = authorization.get("attempt_id")
    require(isinstance(attempt_id, str) and attempt_id.strip(),
            "authorization", "attempt_id is required")
    require(authorization.get("output_root") == str(output),
            "authorization", "authorization output root differs")
    require(authorization.get("native_execution_allowed") is True,
            "authorization", "native execution is not allowed")
    require(authorization.get("single_attempt") is True,
            "authorization", "single_attempt must be true")
    require(isinstance(authorization.get("authorized_by"), str)
            and authorization["authorized_by"].strip(), "authorization",
            "authorized_by is required")
    restart_allowance = authorization.get("restart_allowance", 0)
    require(isinstance(restart_allowance, int) and not isinstance(restart_allowance, bool)
            and 0 <= restart_allowance <= int(config["schedule"]["recovery_max"]),
            "authorization", "restart allowance exceeds declared recovery")
    require(sha256_json({key: value for key, value in authorization.items()
                         if key != "receipt_sha256"})
            == authorization.get("receipt_sha256"), "authorization",
            "authorization self digest differs")
    context = RunContext(
        root=ROOT, config_path=config_path, config=config, config_sha256=config_sha,
        prepared=prepared, manifest=manifest, manifest_sha256=manifest["manifest_sha256"],
        authorization=authorization, authorization_sha256=sha256_file(authorization_path),
        output=output, attempt_id=attempt_id, protocol_commit=commit,
        binding=validate_tokenizer_binding(config["tokenizer_binding"]))
    if register:
        AttemptRegistry(authorization_path).register(context)
    return context, context.binding


class WorkLedger:
    def __init__(self, context: RunContext, *, clock: Callable[[], float] = time.monotonic) -> None:
        self.context = context
        self.clock = clock
        self.path = context.output / "work_ledger.json"
        self.meter = publication.LiveByteMeter(
            context.output, cap_bytes=int(context.config["budget"]["physical_limits"]["max_artifact_bytes"]),
            label="dc_r1_v3")
        if self.path.exists():
            payload = read_json(self.path)
            for key, expected in (("attempt_id", context.attempt_id),
                                  ("output_root", str(context.output)),
                                  ("config_sha256", context.config_sha256),
                                  ("prepared_manifest_sha256", context.manifest_sha256),
                                  ("authorization_receipt_sha256", context.authorization_sha256)):
                require(payload.get(key) == expected, "work_ledger",
                        f"ledger {key} differs")
        else:
            payload = {
                "format": LEDGER_FORMAT,
                "attempt_id": context.attempt_id,
                "output_root": str(context.output),
                "config_sha256": context.config_sha256,
                "prepared_manifest_sha256": context.manifest_sha256,
                "authorization_receipt_sha256": context.authorization_sha256,
                "protocol_commit": context.protocol_commit,
                "model_identity": None,
                "tokenizer_binding_sha256": tokenizer_binding_sha(context.binding),
                "created_at_utc": _utc(),
                "elapsed_seconds": 0.0,
                "cumulative": {},
                "recovery": {},
                "failed": {},
            }
        self.payload = payload
        self.cumulative = dict(payload.get("cumulative", {}))
        self.recovery = dict(payload.get("recovery", {}))
        self.failed = dict(payload.get("failed", {}))
        self.elapsed = float(payload.get("elapsed_seconds", 0.0))
        self.last_clock = self.clock()
        self.model_identity = payload.get("model_identity")
        self.tokenizer_binding_sha256 = payload["tokenizer_binding_sha256"]
        self.save()

    def _tick(self) -> None:
        now = self.clock()
        self.elapsed += max(0.0, now - self.last_clock)
        self.last_clock = now
        limit = self.context.config["budget"]["physical_limits"]["max_wall_seconds"]
        require(self.elapsed <= limit, "wall_time",
                f"elapsed {self.elapsed:.3f}s exceeds {limit}s")

    def save(self) -> None:
        self._tick()
        self.payload.update({
            "elapsed_seconds": self.elapsed,
            "cumulative": dict(sorted(self.cumulative.items())),
            "recovery": dict(sorted(self.recovery.items())),
            "failed": dict(sorted(self.failed.items())),
            "model_identity": self.model_identity,
            "tokenizer_binding_sha256": self.tokenizer_binding_sha256,
        })
        publication.write_json(self.context.output, "work_ledger.json",
                               self.payload, meter=self.meter)
        self.meter.observe()

    def _check(self, work: Mapping[str, int], *, recovery: bool = False,
               failed: bool = False) -> None:
        self._tick()
        limits = self.context.config["budget"]["physical_limits"]
        for name, value in work.items():
            require(isinstance(value, int) and not isinstance(value, bool) and value >= 0,
                    "work", f"invalid counter {name}")
            budget_name = {
                "generation_prefill_forwards": "prefill_forwards",
                "generation_decode_forwards": "decode_forwards",
                "autoregressive_forwards": "forward_calls",
            }.get(name, name)
            require(isinstance(limits.get(f"max_{budget_name}"), int), "work",
                    f"missing ceiling for {name}")
            if failed:
                current = self.failed
            elif recovery:
                current = self.recovery
            else:
                current = self.cumulative
            require(current.get(name, 0) + value <= limits[f"max_{budget_name}"], "work",
                    f"{name} exceeds declared ceiling {limits[f'max_{budget_name}']}")

    def reserve(self, work: Mapping[str, int], *, recovery: bool = False) -> dict[str, int]:
        self._check(work, recovery=recovery)
        return dict(work)

    def charge(self, work: Mapping[str, int], *, recovery: bool = False) -> None:
        self._check(work, recovery=recovery)
        target = self.recovery if recovery else self.cumulative
        for name, value in work.items():
            target[name] = target.get(name, 0) + value
        self.save()

    def charge_failed(self, work: Mapping[str, int], *, recovery: bool = False) -> None:
        self._check(work, failed=True)
        for name, value in work.items():
            self.failed[name] = self.failed.get(name, 0) + value
        if recovery:
            recovery_work = {
                "recovery_forward_calls": work.get("failed_forward_calls", 0),
                "recovery_processed_positions": work.get("failed_processed_positions", 0),
            }
            self._check(recovery_work, recovery=True)
            for name, value in recovery_work.items():
                self.recovery[name] = self.recovery.get(name, 0) + value
        self.save()

    def set_model_identity(self, identity: str) -> None:
        if self.model_identity is not None:
            require(self.model_identity == identity, "model_identity",
                    "model identity differs across stages")
        self.model_identity = identity
        self.save()

    def report(self) -> dict:
        return {
            "cumulative": dict(sorted(self.cumulative.items())),
            "recovery": dict(sorted(self.recovery.items())),
            "failed": dict(sorted(self.failed.items())),
            "elapsed_seconds": self.elapsed,
            "artifact": self.meter.report(),
        }


def _stage_path(context: RunContext, stage: str) -> Path:
    return context.output / stage / "journal.json"


def _new_stage(context: RunContext, stage: str) -> dict:
    return {
        "format": JOURNAL_FORMAT, "stage": stage,
        "attempt_id": context.attempt_id,
        "config_sha256": context.config_sha256,
        "prepared_manifest_sha256": context.manifest_sha256,
        "authorization_receipt_sha256": context.authorization_sha256,
        "stage_complete": False, "publication_stage_done": False,
        "status": "in_progress", "admitted": False,
        "predictions": {}, "trajectories": [], "attempted": [],
        "last_attempt": None, "resumes_used": 0, "restart_charged": False,
        "result": None, "error": None,
    }


def _save_stage(context: RunContext, stage: str, journal: dict,
                ledger: WorkLedger) -> None:
    journal["cumulative"] = ledger.report()
    publication.write_json(context.output, f"{stage}/journal.json", journal,
                           meter=ledger.meter)


def _stage_setup(context: RunContext, stage: str, *, resume: bool,
                 ledger: WorkLedger) -> tuple[dict, bool]:
    require(stage in STAGES, "stage", "unknown stage")
    path = _stage_path(context, stage)
    if path.exists():
        journal = read_json(path)
        for key, expected in (("attempt_id", context.attempt_id),
                              ("config_sha256", context.config_sha256),
                              ("prepared_manifest_sha256", context.manifest_sha256),
                              ("authorization_receipt_sha256", context.authorization_sha256)):
            require(journal.get(key) == expected, "resume", f"stage journal {key} differs")
        if journal.get("stage_complete") is True:
            require(resume, "resume", f"{stage} already complete")
            if (journal.get("publication_stage_done") is True
                    and publication.published_stage_done(
                        context.output, stage, identity={
                            "protocol_id": PROTOCOL_ID,
                            "config_sha256": context.config_sha256,
                            "prepared_manifest_sha256": context.manifest_sha256,
                            "authorization_receipt_sha256": context.authorization_sha256,
                            "attempt_id": context.attempt_id,
                            "output_root": str(context.output.resolve()),
                        })):
                return journal, True
            journal["stage_complete"] = False
            journal["publication_stage_done"] = False
            journal["status"] = "in_progress"
            journal["error"] = None
        require(resume, "resume", f"{stage} is incomplete; use --resume")
        require(int(journal.get("resumes_used", 0)) < int(
            context.authorization.get("restart_allowance", 0)),
            "resume", "no restart allowance remains")
        journal["resumes_used"] = int(journal.get("resumes_used", 0)) + 1
        journal["status"] = "in_progress"
        journal["error"] = None
        return journal, False
    require(not resume, "resume", f"no {stage} journal to resume")
    if stage == "dev" and context.output.exists():
        require(not any(item.name != "work_ledger.json"
                        for item in context.output.iterdir()), "fresh_output",
                "development output is not fresh")
    journal = _new_stage(context, stage)
    _save_stage(context, stage, journal, ledger)
    return journal, False


def build_native_backend(context: RunContext, stage: str):
    return NativeR1Backend(context, stage)


def _load_backend(context: RunContext, stage: str, injected: Any,
                  ledger: WorkLedger):
    expected = {"model_loads": 1, "tokenizer_loads": 1}
    ledger.reserve(expected)
    ledger.charge(expected)
    backend = injected if injected is not None else build_native_backend(context, stage)
    require(backend.startup_work() == expected, "work", "backend startup work differs")
    ledger.set_model_identity(backend.model_identity)
    return backend


def _load_final_decoder(context: RunContext, injected: Any,
                        ledger: WorkLedger):
    if injected is not None:
        return injected.decode_fn
    ledger.reserve({"tokenizer_loads": 1})
    ledger.charge({"tokenizer_loads": 1})
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        str(context.root / "checkpoints/minicpm5_source"), local_files_only=True,
        trust_remote_code=False, revision=context.binding["revision"])
    policy = context.binding["decode_policy"]
    return lambda ids: tokenizer.decode(
        list(ids), skip_special_tokens=policy["skip_special_tokens"],
        clean_up_tokenization_spaces=policy["clean_up_tokenization_spaces"])


def _condition_rows(prepared: Path, split: str) -> tuple[dict, dict]:
    prefix = "dev" if split == "dev" else "sealed_final"
    cases = read_jsonl(prepared / f"{prefix}_cases.jsonl")
    conditions = {}
    for condition in CONDITIONS:
        for row in read_jsonl(prepared / f"{prefix}_conditions_{condition}.jsonl"):
            conditions[(condition, row["case_id"])] = row
    return cases, conditions


def _work_from_result(result: Mapping[str, Any]) -> dict[str, int]:
    return {
        "generation_requests": 1,
        "generation_prefill_forwards": 1,
        "generation_decode_forwards": int(result["decode_forwards"]),
        "autoregressive_forwards": int(result["autoregressive_forwards"]),
        "forward_calls": int(result["forward_calls"]),
        "generated_tokens": int(result["generated_tokens"]),
        "prefill_tokens": int(result["prefill_tokens"]),
        "processed_positions": int(result["processed_positions"]),
        "layer_readout_streams": 1,
    }


def _recovery_work(work: Mapping[str, int]) -> dict[str, int]:
    return {
        "recovery_generation_requests": work["generation_requests"],
        "recovery_prefill_forwards": work["generation_prefill_forwards"],
        "recovery_forward_calls": work["forward_calls"],
        "recovery_generated_tokens": work["generated_tokens"],
        "recovery_prefill_tokens": work["prefill_tokens"],
        "recovery_processed_positions": work["processed_positions"],
        "recovery_layer_readout_streams": work["layer_readout_streams"],
    }


def _step_work(step: Mapping[str, Any], *, prompt_tokens: int,
               recovery: bool) -> dict[str, int]:
    first = int(step["raw_index"]) == 0
    work = {
        "generation_requests": int(first),
        "generation_prefill_forwards": int(first),
        "generation_decode_forwards": int(not first),
        "autoregressive_forwards": 1,
        "forward_calls": 1,
        "generated_tokens": 1,
        "prefill_tokens": prompt_tokens if first else 0,
        "processed_positions": int(step["processed_positions"]),
        "layer_readout_streams": int(first),
    }
    return _recovery_work(work) if recovery else work


def _prediction_row(context: RunContext, stage: str, condition: str,
                    case: Mapping[str, Any], results: Sequence[Mapping[str, Any]],
                    evidence_class: str) -> dict:
    return {
        "format": PREDICTION_FORMAT, "protocol_id": PROTOCOL_ID, "split": stage,
        "condition": condition, "case_id": case["case_id"],
        "prompt": case["prompt"], "prompt_sha256": case["prompt_sha256"],
        "source_context_present": case["source_context_present"],
        "source_kv_loaded": False, "prompt_source_spans": 0,
        "routing_metadata": {"mode": "fixed", "expert_id": 0},
        "tokenizer_binding_sha256": tokenizer_binding_sha(context.binding),
        "decode_policy": context.binding["decode_policy"],
        "evidence_class": evidence_class,
        "attempt_id": context.attempt_id,
        "model_identity": results[0]["model_identity"],
        "reader_state_sha256": results[0]["reader_state_sha256"],
        "layer_outputs": [{
            "layer": int(result["layer"]),
            "raw_generated_ids": list(result["raw_generated_ids"]),
            "terminal_stop_ids": list(result["terminal_stop_ids"]),
            "generated_ids": list(result["generated_ids"]),
            "generated_text": result["generated_text"],
            "empty_generation": bool(result["empty_generation"]),
            "stopped": bool(result["stopped"]),
            "stop_reason": result["stop_reason"],
            "physical_stream_id": f"{condition}/{case['case_id']}/layer:{result['layer']}",
            "work": _work_from_result(result),
        } for result in results],
    }


def _generate(context: RunContext, stage: str, backend: Any,
              ledger: WorkLedger, journal: dict, evidence_class: str) -> dict:
    cases, conditions = _condition_rows(context.prepared, stage)
    predictions = {condition: [dict(row) for row in journal["predictions"].get(condition, [])]
                   for condition in CONDITIONS}
    completed = {(condition, row["case_id"])
                 for condition, rows in predictions.items() for row in rows}
    attempted = set(journal.get("attempted", []))
    for condition in CONDITIONS:
        for case in cases:
            key = (condition, case["case_id"])
            if key in completed:
                continue
            condition_row = conditions[key]
            results = []
            for layer in LAYERS:
                stream_key = f"{condition}/{case['case_id']}/layer:{layer}"
                recovery = stream_key in attempted
                plan = backend.estimate_evaluation_work(condition_row)
                work = plan["work"]
                ledger.reserve(_recovery_work(work) if recovery else work,
                               recovery=recovery)
                journal["last_attempt"] = {
                    "condition": condition, "case_id": case["case_id"], "layer": layer,
                    "recovery": recovery,
                }
                attempted.add(stream_key)
                journal["attempted"] = sorted(attempted)
                _save_stage(context, stage, journal, ledger)
                charged = []

                def on_step(step):
                    measured = _step_work(
                        step, prompt_tokens=int(work["prefill_tokens"]), recovery=recovery)
                    ledger.charge(measured, recovery=recovery)
                    charged.append(measured)
                    journal["trajectories"].append({
                        "stream": stream_key, "recovery": recovery, **dict(step)})
                    _save_stage(context, stage, journal, ledger)

                def on_error(error):
                    failed = {
                        "failed_forward_calls": int(error["forward_calls"]),
                        "failed_processed_positions": int(error["processed_positions"]),
                    }
                    ledger.charge_failed(failed, recovery=recovery)
                    journal["trajectories"].append({
                        "stream": stream_key, "recovery": recovery, **dict(error)})
                    _save_stage(context, stage, journal, ledger)

                result = backend.evaluate(
                    condition_row, layer=layer, prepared=plan,
                    on_step=on_step, on_error=on_error)
                expected = _work_from_result(result)
                if not recovery:
                    totals = {name: sum(item[name] for item in charged)
                              for name in expected}
                    require(totals == expected, "work",
                            f"charged work differs for {stream_key}: {totals} != {expected}")
                results.append(result)
            predictions[condition].append(_prediction_row(
                context, stage, condition, condition_row, results, evidence_class))
            completed.add(key)
            journal["predictions"] = predictions
            _save_stage(context, stage, journal, ledger)
    return predictions


def _write_predictions(context: RunContext, stage: str, predictions: Mapping[str, Any],
                       ledger: WorkLedger) -> None:
    for condition, rows in predictions.items():
        publication.write_jsonl(context.output, f"{stage}/predictions_{condition}.jsonl",
                                rows, meter=ledger.meter)
    rows = [row for condition in CONDITIONS for row in predictions[condition]]
    publication.write_jsonl(context.output, f"{stage}/predictions_all.jsonl",
                            rows, meter=ledger.meter)


def _publish(context: RunContext, stage: str, journal: dict,
             ledger: WorkLedger, report: dict, admitted: bool,
             backend: Any) -> dict:
    publication.write_json(context.output, f"{stage}/score_report.json", report,
                           meter=ledger.meter)
    admission = {
        "attempt_id": context.attempt_id,
        "evidence_class": report["evidence_class"],
        "native_execution_admissible": admitted,
        "gate_admitted": report["development_evidence_admissible"],
        "status": report["status"],
    }
    publication.write_json(context.output, f"{stage}/admission.json", admission,
                           meter=ledger.meter)
    backbone = backend.assert_backbone_unchanged()
    journal.update({
        "stage_complete": True, "publication_stage_done": True,
        "status": report["status"], "admitted": admitted,
        "result": {
            "status": report["status"], "stage": stage,
            "attempt_id": context.attempt_id, "admitted": admitted,
            "report_path": str(context.output / stage / "score_report.json"),
            "counters": ledger.report(), "backbone": backbone,
        },
    })
    _save_stage(context, stage, journal, ledger)
    pub = publication.Journal(
        context.output, identity={
            "protocol_id": PROTOCOL_ID,
            "config_sha256": context.config_sha256,
            "prepared_manifest_sha256": context.manifest_sha256,
            "authorization_receipt_sha256": context.authorization_sha256,
            "attempt_id": context.attempt_id,
            "output_root": str(context.output.resolve()),
        }, cap_bytes=int(context.config["budget"]["physical_limits"]["max_artifact_bytes"]),
        meter=ledger.meter)
    artifacts = [publication.Artifact.of(context.output, f"{stage}/score_report.json"),
                 publication.Artifact.of(context.output, f"{stage}/admission.json"),
                 publication.Artifact.of(context.output, f"{stage}/journal.json"),
                 publication.Artifact.of(context.output, f"{stage}/predictions_all.jsonl")]
    artifacts.extend(publication.Artifact.of(
        context.output, f"{stage}/predictions_{condition}.jsonl") for condition in CONDITIONS)
    pub.record(stage, artifacts)
    ledger.meter.observe()
    pub.record_peak(ledger.meter.report())
    return dict(journal["result"])


def _run(context: RunContext, stage: str, *, resume: bool,
         injected_backend: Any, clock: Callable[[], float]) -> dict:
    ledger = WorkLedger(context, clock=clock)
    journal, complete = _stage_setup(context, stage, resume=resume, ledger=ledger)
    if complete:
        result = dict(journal["result"])
        result["already_complete"] = True
        return result
    if journal["resumes_used"] and not journal.get("restart_charged"):
        ledger.charge({"restarts": 1})
        journal["restart_charged"] = True
        _save_stage(context, stage, journal, ledger)
    evidence_class = "native_mps" if injected_backend is None else "cpu_fixture"
    backend = None
    try:
        if stage == "sealed_final":
            decode_fn = _load_final_decoder(context, injected_backend, ledger)
            scoring.verify_development_evidence(
                context.output, decode_fn=decode_fn,
                expected_attempt_id=context.attempt_id, require_native=True,
                prepared=context.prepared)
        backend = _load_backend(context, stage, injected_backend, ledger)
        journal["model_identity"] = backend.model_identity
        _save_stage(context, stage, journal, ledger)
        predictions = _generate(context, stage, backend, ledger, journal, evidence_class)
        _write_predictions(context, stage, predictions, ledger)
        report = scoring.score_predictions(
            context.output / stage / "predictions_all.jsonl", split=stage,
            decode_fn=backend.decode_fn, native_execution=True,
            dev_output=context.output if stage == "sealed_final" else None,
            expected_attempt_id=context.attempt_id if stage == "sealed_final" else None,
            prepared=context.prepared)
        admitted = evidence_class == "native_mps" and report["native_execution_admissible"]
        return _publish(context, stage, journal, ledger, report, admitted, backend)
    except BaseException as error:
        journal.update({
            "stage_complete": False, "publication_stage_done": False,
            "status": "failed", "admitted": False, "error": str(error),
        })
        try:
            _save_stage(context, stage, journal, ledger)
        finally:
            ledger.save()
        raise
    finally:
        if backend is not None:
            backend.close()


def run_stage(stage: str, *, config_path: str | Path, prepared: str | Path,
              authorization_path: str | Path, output: str | Path,
              resume: bool = False, backend: Any = None,
              require_head: bool = True, commit_verifier: Callable | None = None,
              clock: Callable[[], float] = time.monotonic) -> dict:
    context, _ = verify_authorization(
        config_path=config_path, prepared=prepared,
        authorization_path=authorization_path, output=output,
        require_head=require_head, commit_verifier=commit_verifier)
    require(stage in STAGES, "stage", "unknown stage")
    return _run(context, stage, resume=resume, injected_backend=backend, clock=clock)


def run_dev(**kwargs: Any) -> dict:
    return run_stage("dev", **kwargs)


def run_sealed_final(**kwargs: Any) -> dict:
    return run_stage("sealed_final", **kwargs)

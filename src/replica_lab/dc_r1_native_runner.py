"""DC-R1 native readout runner with staged evidence and artifact admission."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any, Callable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from dc_protocol_common_20261027 import (
    ProtocolError, read_json, read_jsonl, require, sha256_file, sha256_json,
    tokenizer_binding_sha, validate_baseline_model, validate_guards,
    validate_manifest, validate_tokenizer_binding,
)
from replica_lab import artifact_publication as publication

PROTOCOL_ID = "dc_r1_native_value_readout_v2"
AUTHORIZATION_FORMAT = "replica.dc_protocol.authorization.v1"
PREDICTION_FORMAT = "replica.dc_r1_native_value.prediction.v2"
JOURNAL_FORMAT = "replica.dc_r1_native_value.stage_journal.v1"
STAGES = ("dev", "sealed_final")
LAYERS = (13, 27, 41)
CONDITIONS = ("exact_context", "selected_literal", "compressed_record",
              "wrong_value", "no_memory")


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _attempt_id(context: "RunContext", stage: str) -> str:
    return "dcr1v2:" + sha256_json({
        "protocol_id": PROTOCOL_ID, "stage": stage,
        "config_sha256": context.config_sha256,
        "manifest_sha256": context.manifest_sha256,
        "output": str(context.output),
    })[:32]


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


def verify_authorization(*, config_path: str | Path, prepared: str | Path,
                        authorization_path: str | Path,
                        output: str | Path | None = None) -> tuple[RunContext, dict]:
    config_path = Path(config_path)
    prepared = Path(prepared)
    authorization_path = Path(authorization_path)
    config = read_json(config_path)
    require(config.get("protocol_id") == PROTOCOL_ID, "authorization",
            "config protocol differs")
    require(config.get("format") == "replica.dc_r1_native_value.protocol.v2",
            "authorization", "config format differs")
    validate_guards(config)
    validate_baseline_model(config)
    binding = validate_tokenizer_binding(config["tokenizer_binding"])
    manifest = read_json(prepared / "manifest.json")
    require(manifest.get("protocol_id") == PROTOCOL_ID, "authorization",
            "prepared manifest protocol differs")
    body = {key: value for key, value in manifest.items()
            if key != "manifest_sha256"}
    require(manifest["manifest_sha256"] == sha256_json(body), "authorization",
            "prepared manifest self hash differs")
    for name, digest in manifest["files"].items():
        path = prepared / name
        require(path.is_file(), "authorization", f"missing prepared file {path}")
        if name != "sealed_labels_final.jsonl":
            require(sha256_file(path) == digest, "authorization",
                    f"prepared file bytes differ: {name}")
    config_sha = sha256_file(config_path)
    require(config["prepared"]["manifest_sha256"] == manifest["manifest_sha256"],
            "authorization", "config does not bind the prepared manifest")
    authorization = read_json(authorization_path)
    require(authorization.get("format") == AUTHORIZATION_FORMAT,
            "authorization", "authorization format differs")
    require(authorization.get("protocol_id") == PROTOCOL_ID,
            "authorization", "authorization protocol differs")
    require(authorization.get("config_sha256") == config_sha,
            "authorization", "authorization did not bind exact config bytes")
    require(authorization.get("prepared_manifest_sha256")
            == manifest["manifest_sha256"], "authorization",
            "authorization did not bind exact prepared fixture")
    require(authorization.get("native_execution_allowed") is True,
            "authorization", "native_execution_allowed is not true")
    require(authorization.get("single_attempt") is True,
            "authorization", "single_attempt must be true")
    require(isinstance(authorization.get("authorized_by"), str)
            and bool(authorization["authorized_by"].strip()), "authorization",
            "authorized_by must be a nonempty string")
    require(sha256_json({key: value for key, value in authorization.items()
                         if key != "receipt_sha256"})
            == authorization.get("receipt_sha256"), "authorization",
            "authorization self digest differs")
    target = Path(output) if output is not None else Path(config["launch"]["output_root"])
    if not target.is_absolute():
        target = ROOT / target
    return RunContext(
        root=ROOT, config_path=config_path, config=config,
        config_sha256=config_sha, prepared=prepared, manifest=manifest,
        manifest_sha256=manifest["manifest_sha256"], authorization=authorization,
        authorization_sha256=sha256_file(authorization_path), output=target), binding


class WorkMeter:
    def __init__(self, budget: Mapping[str, Any]) -> None:
        self.budget = budget
        self.cumulative: dict[str, int] = {}

    def _check(self, work: Mapping[str, int]) -> None:
        for name, value in work.items():
            require(isinstance(value, int) and not isinstance(value, bool)
                    and value >= 0, "work", f"invalid counter {name}")
            budget_name = {
                "generation_prefill_forwards": "prefill_forwards",
                "generation_decode_forwards": "decode_forwards",
                "autoregressive_forwards": "forward_calls",
            }.get(name, name)
            ceiling = self.budget.get(f"max_{budget_name}")
            require(ceiling is not None, "work", f"missing ceiling for {name}")
            require(self.cumulative.get(name, 0) + value <= ceiling, "work",
                    f"{name} exceeds declared ceiling {ceiling}")

    def reserve(self, work: Mapping[str, int]) -> dict[str, int]:
        self._check(work)
        return dict(work)

    def charge(self, work: Mapping[str, int]) -> None:
        self._check(work)
        for name, value in work.items():
            self.cumulative[name] = self.cumulative.get(name, 0) + value

    def report(self) -> dict[str, int]:
        return dict(sorted(self.cumulative.items()))


def _stage_root(context: RunContext, stage: str) -> Path:
    return context.output / stage


def _journal_path(context: RunContext, stage: str) -> Path:
    return _stage_root(context, stage) / "journal.json"


def _new_journal(context: RunContext, stage: str) -> dict:
    return {
        "format": JOURNAL_FORMAT,
        "stage": stage,
        "attempt_id": _attempt_id(context, stage),
        "config_sha256": context.config_sha256,
        "manifest_sha256": context.manifest_sha256,
        "authorization_receipt_sha256": context.authorization_sha256,
        "created_at_utc": _utc(),
        "stage_complete": False,
        "publication_stage_done": False,
        "status": "incomplete",
        "admitted": False,
        "cumulative": {},
        "result": None,
    }


def _save_journal(context: RunContext, stage: str, journal: dict,
                  meter: WorkMeter) -> None:
    journal["cumulative"] = meter.report()
    journal["updated_at_utc"] = _utc()
    path = _journal_path(context, stage)
    path.parent.mkdir(parents=True, exist_ok=True)
    publication.write_json(path.parent, path.name, journal)


def _stage_setup(context: RunContext, stage: str, *, resume: bool,
                 budget: Mapping[str, Any]) -> tuple[dict, WorkMeter, bool]:
    require(stage in STAGES, "stage", f"unknown stage {stage}")
    path = _journal_path(context, stage)
    if stage == "dev" and context.output.exists() and not path.exists():
        require(not any(context.output.iterdir()), "launch",
                "development output root is not fresh")
    if path.is_file():
        journal = read_json(path)
        for key, expected in (
                ("config_sha256", context.config_sha256),
                ("manifest_sha256", context.manifest_sha256),
                ("authorization_receipt_sha256", context.authorization_sha256),
                ("attempt_id", _attempt_id(context, stage))):
            require(journal.get(key) == expected, "resume",
                    f"stage journal {key} differs")
        if journal.get("stage_complete") is True:
            require(resume, "resume", f"{stage} is already complete")
            return journal, WorkMeter(budget), True
        require(resume, "resume", f"{stage} is incomplete; use --resume")
        require(journal.get("resumes_used", 0) < int(
            context.authorization.get("restart_allowance", 0)),
            "resume", "no restart allowance remains")
        journal["resumes_used"] = int(journal.get("resumes_used", 0)) + 1
    else:
        require(not resume, "resume", f"no {stage} journal to resume")
        journal = _new_journal(context, stage)
    return journal, WorkMeter(budget), False


def build_native_backend(context: RunContext, stage: str):
    from replica_lab.dc_r1_native_backend import NativeR1Backend
    return NativeR1Backend(context, stage)


def _tokenizer_decode(binding: Mapping[str, Any], tokenizer_dir: Path):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_dir), local_files_only=True, trust_remote_code=False,
        revision=binding["revision"])
    policy = binding["decode_policy"]

    def decode(ids: Sequence[int]) -> str:
        return tokenizer.decode(
            list(ids), skip_special_tokens=policy["skip_special_tokens"],
            clean_up_tokenization_spaces=policy["clean_up_tokenization_spaces"])
    return decode


def _condition_rows(prepared: Path, split_prefix: str) -> dict[tuple[str, str], dict]:
    rows: dict[tuple[str, str], dict] = {}
    for condition in CONDITIONS:
        path = prepared / f"{split_prefix}_conditions_{condition}.jsonl"
        for row in read_jsonl(path):
            key = (condition, row["case_id"])
            require(key not in rows, "prepared", f"duplicate condition row {key}")
            rows[key] = row
    require(rows, "prepared", f"no condition rows for {split_prefix}")
    return rows


def _cases(prepared: Path, split_prefix: str) -> list[dict]:
    rows = read_jsonl(prepared / f"{split_prefix}_cases.jsonl")
    require(rows, "prepared", f"no cases for {split_prefix}")
    return rows


def _prediction_row(*, context: RunContext, stage: str, condition: str,
                    case: Mapping[str, Any],
                    layer_results: Sequence[Mapping[str, Any]],
                    evidence_class: str) -> dict:
    binding = context.config["tokenizer_binding"]
    outputs = []
    for layer_result in layer_results:
        work = {
            "generation_requests": 1,
            "generation_prefill_forwards": 1,
            "generation_decode_forwards": int(layer_result["decode_forwards"]),
            "autoregressive_forwards": int(layer_result["autoregressive_forwards"]),
            "forward_calls": int(layer_result["forward_calls"]),
            "generated_tokens": int(layer_result["generated_tokens"]),
            "prefill_tokens": int(layer_result["prefill_tokens"]),
            "processed_positions": int(layer_result["processed_positions"]),
            "layer_readout_streams": 1,
        }
        outputs.append({
            "layer": int(layer_result["layer"]),
            "generated_ids": list(layer_result["generated_ids"]),
            "generated_text": layer_result["generated_text"],
            "empty_generation": bool(layer_result["empty_generation"]),
            "stopped": bool(layer_result["stopped"]),
            "stop_reason": layer_result["stop_reason"],
            "physical_stream_id": f"{condition}/{case['case_id']}/layer:{layer_result['layer']}",
            "work": work,
        })
    return {
        "format": PREDICTION_FORMAT,
        "protocol_id": PROTOCOL_ID,
        "split": stage,
        "condition": condition,
        "case_id": case["case_id"],
        "prompt": case["prompt"],
        "prompt_sha256": case["prompt_sha256"],
        "source_context_present": case["source_context_present"],
        "source_kv_loaded": False,
        "prompt_source_spans": 0,
        "routing_metadata": {"mode": "fixed", "expert_id": 0},
        "tokenizer_binding_sha256": tokenizer_binding_sha(binding),
        "decode_policy": binding["decode_policy"],
        "evidence_class": evidence_class,
        "reader_state_sha256": layer_results[0]["reader_state_sha256"],
        "evaluation_owner": layer_results[0]["evaluation_owner"],
        "layer_outputs": outputs,
    }


def _generate(context: RunContext, stage: str, backend: Any,
              meter: WorkMeter, evidence_class: str,
              journal: dict | None = None) -> dict[str, list[dict]]:
    prefix = "dev" if stage == "dev" else "sealed_final"
    conditions = _condition_rows(context.prepared, prefix)
    cases = _cases(context.prepared, prefix)
    saved = journal.get("predictions", {}) if journal is not None else {}
    predictions = {
        condition: [dict(row) for row in saved.get(condition, [])]
        for condition in CONDITIONS}
    completed = {
        (condition, row["case_id"])
        for condition, rows in predictions.items() for row in rows}
    for condition in CONDITIONS:
        for case in cases:
            condition_row = conditions[(condition, case["case_id"])]
            key = (condition, case["case_id"])
            if key in completed:
                continue
            layer_results = []
            for layer in LAYERS:
                if journal is not None:
                    journal["last_attempt"] = {
                        "condition": condition, "case_id": case["case_id"],
                        "layer": layer,
                    }
                plan = backend.estimate_evaluation_work(condition_row)
                reservation = meter.reserve(plan["work"])
                result = backend.evaluate(condition_row, layer=layer, prepared=plan)
                if journal is not None:
                    journal.setdefault("trajectories", []).append({
                        "condition": condition, "case_id": case["case_id"],
                        "layer": layer, "result": dict(result),
                    })
                actual = {
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
                require(set(actual) <= set(reservation), "work",
                        "actual work contains an unreserved counter")
                for name, value in actual.items():
                    require(value <= reservation[name], "work",
                            f"actual {name} exceeds reservation")
                meter.charge(actual)
                if journal is not None:
                    journal["predictions"] = predictions
                    _save_journal(context, stage, journal, meter)
                layer_results.append(result)
            predictions[condition].append(_prediction_row(
                context=context, stage=stage, condition=condition,
                case=condition_row, layer_results=layer_results,
                evidence_class=evidence_class))
            completed.add(key)
            if journal is not None:
                journal["predictions"] = predictions
                _save_journal(context, stage, journal, meter)
    return predictions


def _write_predictions(context: RunContext, stage: str,
                       predictions: Mapping[str, Sequence[Mapping[str, Any]]],
                       meter: publication.LiveByteMeter) -> None:
    for condition, rows in predictions.items():
        publication.write_jsonl(
            context.output, f"{stage}/predictions_{condition}.jsonl", rows,
            meter=meter)
    all_rows = [row for condition in CONDITIONS for row in predictions[condition]]
    publication.write_jsonl(
        context.output, f"{stage}/predictions_all.jsonl", all_rows, meter=meter)


def _publish_stage(context: RunContext, stage: str,
                   artifacts: Sequence[publication.Artifact],
                   journal: publication.Journal,
                   meter: publication.LiveByteMeter) -> None:
    journal.record(stage, artifacts)
    meter.observe()
    journal.record_peak(meter.report())
    cap = int(context.config["budget"]["authorized_run_ceiling"]["artifact_bytes_max"])
    measured = publication.total_bytes(context.output)
    if measured > cap:
        publication.write_overflow_evidence(
            context.output, context=f"dc_r1_v2:{stage}", cap_bytes=cap,
            measured_bytes=measured, files=publication.inventory(context.output))
        raise ProtocolError("artifact", f"artifact budget exceeded: {measured} > {cap}")


def _prior_dev_boundary(context: RunContext, decode_fn: Callable) -> None:
    from dc_r1_native_value_v2_20261027 import admit_dev_report
    from dc_r1_native_value_v2_20261027 import check_dev_report_boundary
    report_path = _stage_root(context, "dev") / "score_report.json"
    check_dev_report_boundary(
        report_path, config_path=context.config_path,
        prepared_manifest_sha256=context.manifest_sha256)
    predictions_path = _stage_root(context, "dev") / "predictions_all.jsonl"
    require(predictions_path.is_file(), "final_admission",
            "development predictions are missing")
    admit_dev_report(
        report_path, config_path=context.config_path,
        prepared_manifest_sha256=context.manifest_sha256,
        dev_predictions_path=predictions_path, decode_fn=decode_fn,
        native_execution=True)
    from dc_r1_native_value_v2_20261027 import FILES, MANIFEST_FORMAT
    validate_manifest(
        context.prepared, context.config, expected_files=FILES,
        manifest_format=MANIFEST_FORMAT, protocol_id=PROTOCOL_ID)


def _score(context: RunContext, stage: str, evidence_class: str,
           decode_fn: Callable) -> dict:
    from dc_r1_native_value_v2_20261027 import score_r1_v2
    predictions_path = _stage_root(context, stage) / "predictions_all.jsonl"
    require(predictions_path.is_file(), "scoring",
            f"missing prediction stream: {predictions_path}")
    if stage == "sealed_final":
        report = score_r1_v2(
            predictions_path, split="sealed_final", prepared=context.prepared,
            decode_fn=decode_fn, config_path=context.config_path,
            dev_report_path=_stage_root(context, "dev") / "score_report.json",
            dev_predictions_path=_stage_root(context, "dev") / "predictions_all.jsonl",
            native_execution=True)
    else:
        report = score_r1_v2(
            predictions_path, split="dev", prepared=context.prepared,
            decode_fn=decode_fn, config_path=context.config_path,
            native_execution=True)
    return report


def _run(context: RunContext, stage: str, *, resume: bool, backend: Any,
         binding: Mapping[str, Any]) -> dict:
    journal, meter, complete = _stage_setup(
        context, stage, resume=resume,
        budget=context.config["budget"]["physical_limits"])
    if complete:
        result = dict(journal["result"])
        result["already_complete"] = True
        return result
    for name, value in journal.get("cumulative", {}).items():
        require(isinstance(value, int) and not isinstance(value, bool) and value >= 0,
                "resume", f"invalid persisted counter {name}")
        meter.cumulative[name] = value
    journal.update({
        "stage_complete": False, "publication_stage_done": False,
        "status": "in_progress", "admitted": False, "result": None,
        "error": None,
    })
    injected_backend = backend
    evidence_class = "native_mps" if injected_backend is None else "cpu_fixture"
    backend = None
    try:
        _save_journal(context, stage, journal, meter)
        if stage == "sealed_final":
            decode_fn = (injected_backend.decode_fn if injected_backend is not None
                         else _tokenizer_decode(binding, ROOT / "checkpoints/minicpm5_source"))
            _prior_dev_boundary(context, decode_fn)
        backend = (injected_backend if injected_backend is not None
                   else build_native_backend(context, stage))
        meter.charge(backend.startup_work())
        _save_journal(context, stage, journal, meter)
        publication_meter = publication.LiveByteMeter(context.output, label="dc_r1_v2")
        publication_meter.observe()
        publication_journal = publication.Journal(
            context.output, identity={
                "protocol_id": PROTOCOL_ID,
                "config_sha256": context.config_sha256,
                "prepared_manifest_sha256": context.manifest_sha256,
                "authorization_receipt_sha256": context.authorization_sha256,
            }, cap_bytes=int(context.config["budget"]["authorized_run_ceiling"]["artifact_bytes_max"]),
            meter=publication_meter)
        predictions = _generate(
            context, stage, backend, meter, evidence_class, journal=journal)
        _write_predictions(context, stage, predictions, publication_meter)
        report = _score(context, stage, evidence_class, backend.decode_fn)
        report_path = _stage_root(context, stage) / "score_report.json"
        publication.write_json(report_path.parent, report_path.name, report,
                                meter=publication_meter)
        gate_admitted = (report["development_evidence_admissible"]
                         if stage == "dev" else report["final_admission"]["admitted"])
        admitted = evidence_class == "native_mps" and bool(gate_admitted)
        publication.write_json(
            _stage_root(context, stage), "admission.json",
            {"admitted": admitted, "gate_admitted": bool(gate_admitted),
             "native_execution_admissible": admitted,
             "attempt_id": _attempt_id(context, stage),
             "status": report["status"], "evidence_class": evidence_class},
            meter=publication_meter)
        artifacts = [publication.Artifact.of(
            context.output, f"{stage}/score_report.json")]
        artifacts.extend(publication.Artifact.of(
            context.output, f"{stage}/predictions_{condition}.jsonl")
            for condition in CONDITIONS)
        artifacts.append(publication.Artifact.of(
            context.output, f"{stage}/predictions_all.jsonl"))
        artifacts.append(publication.Artifact.of(
            context.output, f"{stage}/admission.json"))
        _publish_stage(context, stage, artifacts, publication_journal,
                       publication_meter)
        backbone_assertion = backend.assert_backbone_unchanged()
        journal.update({
            "stage_complete": True,
            "publication_stage_done": True,
            "status": report["status"],
            "admitted": admitted,
            "result": {
                "status": report["status"],
                "stage": stage,
                "attempt_id": _attempt_id(context, stage),
                "admitted": admitted,
                "report_path": str(report_path),
                "counters": meter.report(),
                "backbone": backbone_assertion,
            },
        })
        _save_journal(context, stage, journal, meter)
        return dict(journal["result"])
    except BaseException as error:
        journal.update({
            "stage_complete": False, "publication_stage_done": False,
            "status": "failed", "admitted": False, "error": str(error),
        })
        try:
            _save_journal(context, stage, journal, meter)
        except Exception as persistence_error:
            journal["journal_persistence_error"] = str(persistence_error)
        raise
    finally:
        if backend is not None:
            backend.close()


def run_stage(stage: str, *, config_path: str | Path, prepared: str | Path,
              authorization_path: str | Path, output: str | Path,
              resume: bool = False, backend: Any = None) -> dict:
    context, binding = verify_authorization(
        config_path=config_path, prepared=prepared,
        authorization_path=authorization_path, output=output)
    require(stage in STAGES, "stage", f"unknown stage {stage}")
    return _run(context, stage, resume=resume, backend=backend, binding=binding)


def run_dev(**kwargs: Any) -> dict:
    return run_stage("dev", **kwargs)


def run_sealed_final(**kwargs: Any) -> dict:
    return run_stage("sealed_final", **kwargs)

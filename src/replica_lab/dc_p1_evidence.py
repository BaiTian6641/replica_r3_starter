"""Artifact-backed DC-P1 evidence admission and tokenizer-only decoding.

Prerequisite stages are re-scored from raw cases, labels, predictions and the
sealed decode policy.  The module never imports a model.  Checkpoint loading is
CPU-only and uses ``weights_only=True``.  No current-stage label file is opened
by this module unless its stage is explicitly requested by the caller after the
admission boundary.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from dc_protocol_common_20261027 import (
    ProtocolError, normalize_answer, read_json, read_jsonl, require, sha256_file,
    sha256_json, tokenizer_binding_sha, validate_tokenizer_directory,
    verify_decoded_text,
)
from replica_lab import dc_staged_evaluation as staged
from replica_lab.learning_evidence import learning_success_evidence

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_ID = "dc_p1_native_response_v2"
CHECKPOINT_FORMAT = "replica.dc_p1_native_response_v2.checkpoint.v1"
NATIVE_CLASS = "native_mps"
CPU_CLASS = "cpu_fixture"
ZERO_DIGEST = "0" * 64

STAGE_FILES = {
    staged.STAGE_FIT: (("fit_cases.jsonl", "sealed_labels_fit.jsonl"),),
    staged.STAGE_DEVELOPMENT: (("dev_cases.jsonl", "sealed_labels_dev.jsonl"),),
    staged.STAGE_FINAL: (("sealed_final_fit_cases.jsonl",
                          "sealed_final_labels_fit.jsonl"),
                         ("sealed_final_cases.jsonl", "sealed_labels_final.jsonl")),
}
REPORT_FILES = {
    staged.STAGE_FIT: "fit_report.json",
    staged.STAGE_DEVELOPMENT: "development_report.json",
    staged.STAGE_FINAL: "final_report.json",
}
LATE_LABEL_FILES = {
    staged.STAGE_FIT: frozenset(),
    staged.STAGE_DEVELOPMENT: frozenset({"sealed_labels_dev.jsonl"}),
    staged.STAGE_FINAL: frozenset({"sealed_labels_final.jsonl",
                                   "sealed_final_labels_fit.jsonl"}),
}


def arm_specs(config: Mapping[str, Any], stage: str) -> list[dict]:
    groups = list(config["plastic_spec"]["trainable_groups"])
    if stage == staged.STAGE_FINAL:
        final = config["controls"]["final_stage"]
        return [{
            "arm_id": "treatment",
            "checkpoint_id": final["checkpoint_id"],
            "answer_span_ce_weight": config["objective"][
                "answer_span_ce_weight_treatment"],
            "updates": int(final["updates"]),
            "trainable_groups": groups,
        }]
    return [{
        "arm_id": arm["arm_id"],
        "checkpoint_id": arm["checkpoint_id"],
        "answer_span_ce_weight": arm["answer_span_ce_weight"],
        "updates": int(arm["updates"]),
        "restore_parent_before_evaluation": bool(
            arm.get("restore_parent_before_evaluation")),
        "trainable_groups": list(arm.get("trainable_groups", groups)),
    } for arm in config["controls"]["arms"]]


def receipt_expectation(*, config: Mapping[str, Any], config_path: str | Path,
                        manifest: Mapping[str, Any], binding: Mapping[str, Any],
                        protocol_commit: str) -> staged.ReceiptExpectation:
    counts = config["prepared"]["counts"]
    return staged.ReceiptExpectation(
        protocol_id=PROTOCOL_ID,
        config_sha256=sha256_file(config_path),
        manifest_sha256=manifest["manifest_sha256"],
        protocol_commit=protocol_commit,
        tokenizer_binding_sha256=tokenizer_binding_sha(binding),
        decode_policy_sha256=sha256_json(binding["decode_policy"]),
        arms=tuple(arm["arm_id"] for arm in config["controls"]["arms"]),
        cases_expected={
            staged.STAGE_FIT: counts["fit_cases"],
            staged.STAGE_DEVELOPMENT: counts["dev_cases"],
            staged.STAGE_FINAL: (counts["sealed_final_fit_cases"]
                                 + counts["sealed_final_cases"]),
        },
    )


def tokenizer_decode(binding: Mapping[str, Any], tokenizer_dir: str | Path,
                     ) -> Callable[[Sequence[int]], str]:
    """Load only the sealed tokenizer; no model or native backend is created."""
    validate_tokenizer_directory(binding, tokenizer_dir)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_dir), local_files_only=True, trust_remote_code=False,
        revision=binding["revision"])
    policy = binding["decode_policy"]

    def decode(ids: Sequence[int]) -> str:
        return tokenizer.decode(
            list(ids), skip_special_tokens=policy["skip_special_tokens"],
            clean_up_tokenization_spaces=policy[
                "clean_up_tokenization_spaces"])

    return decode


def _digest(value: Any, name: str) -> str:
    require(isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdef" for character in value)
            and value != ZERO_DIGEST, "evidence_digest",
            f"{name} must be a nonzero sha256 digest")
    return value


def _verified_rows(path: Path, manifest: Mapping[str, Any], *,
                   spy: Any = None, stage: str | None = None,
                   kind: str = "evidence") -> list[dict]:
    name = path.name
    require(name in manifest["files"], "artifact",
            f"prepared artifact is not manifest-bound: {name}")
    require(path.is_file(), "artifact", f"missing prepared artifact: {path}")
    require(sha256_file(path) == manifest["files"][name], "artifact",
            f"prepared artifact content differs: {name}")
    rows = read_jsonl(path)
    if spy is not None and stage is not None:
        spy.record("parse", stage=stage, granted=True,
                   detail=f"{kind}:{name}")
    return rows


def load_stage_rows(prepared: str | Path, stage: str,
                    manifest: Mapping[str, Any], *, spy: Any = None,
                    ) -> tuple[list[dict], list[dict]]:
    cases: list[dict] = []
    labels: list[dict] = []
    for cases_name, labels_name in STAGE_FILES[stage]:
        cases.extend(_verified_rows(
            Path(prepared) / cases_name, manifest, spy=spy, stage=stage,
            kind="cases"))
        labels.extend(_verified_rows(
            Path(prepared) / labels_name, manifest, spy=spy, stage=stage,
            kind="labels"))
    return cases, labels


def _default_paths(output_root: Path, stage: str,
                   specs: Sequence[Mapping[str, Any]]):
    checkpoint_stage = (staged.STAGE_FINAL if stage == staged.STAGE_FINAL
                        else staged.STAGE_FIT)
    report = output_root / stage / REPORT_FILES[stage]
    receipt = output_root / stage / "receipt.json"
    predictions = {
        spec["arm_id"]: output_root / stage / f"predictions_{spec['arm_id']}.jsonl"
        for spec in specs
    }
    checkpoints = {
        spec["arm_id"]: output_root / checkpoint_stage / "checkpoints" /
        f"{spec['arm_id']}.pt"
        for spec in specs if int(spec.get("updates", 0)) > 0
    }
    return report, receipt, predictions, checkpoints, checkpoint_stage


def _verify_publication(*, output_root: Path, stage: str,
                        checkpoint_stage: str,
                        expected_paths: Sequence[Path],
                        config_sha256: str, manifest_sha256: str,
                        protocol_commit: str, receipt_attempt_id: str,
                        report_attempt_id: Any) -> int:
    from replica_lab import artifact_publication as publication

    journal_path = output_root / "publication_journal.json"
    require(journal_path.is_file(), "publication",
            f"missing publication journal: {journal_path}")
    journal = read_json(journal_path)
    require(journal.get("format") == publication.JOURNAL_FORMAT,
            "publication", "publication journal format differs")
    identity = journal.get("identity", {})
    for key, expected in (("protocol_id", PROTOCOL_ID),
                          ("config_sha256", config_sha256),
                          ("prepared_manifest_sha256", manifest_sha256),
                          ("protocol_commit", protocol_commit)):
        require(identity.get(key) == expected, "publication",
                f"publication identity {key} differs")
    _digest(identity.get("authorization_receipt_sha256"),
            "publication authorization digest")
    stages = journal.get("stages")
    require(isinstance(stages, list), "publication",
            "publication stage list is missing")
    expected_stages = {stage, checkpoint_stage}
    verified = 0
    for current in expected_stages:
        entries = [entry for entry in stages if entry.get("stage") == current]
        require(len(entries) == 1, "publication",
                f"publication stage {current!r} is missing or duplicated")
        artifacts = [publication.Artifact.from_mapping(row)
                     for row in entries[0].get("artifacts", [])]
        require(artifacts, "publication",
                f"publication stage {current!r} has no artifacts")
        require(publication.verify_all(output_root, artifacts),
                "publication", f"publication artifacts changed for {current}")
        recorded = {artifact.path for artifact in artifacts}
        for path in expected_paths:
            if current == checkpoint_stage and checkpoint_stage != stage:
                relevant = path.is_relative_to(output_root / checkpoint_stage)
            else:
                relevant = path.is_relative_to(output_root / stage)
            if relevant:
                relative = path.relative_to(output_root).as_posix()
                require(relative in recorded, "publication",
                        f"artifact is not publication-bound: {relative}")
        stage_journal = read_json(output_root / current / "journal.json")
        require(stage_journal.get("stage_complete") is True
                and stage_journal.get("publication_stage_done") is True,
                "publication", f"stage {current} is not completely published")
        if current == stage:
            require(stage_journal.get("attempt_id") == receipt_attempt_id,
                    "stage_linkage", "stage journal attempt differs from receipt")
            require(report_attempt_id == stage_journal["attempt_id"],
                    "stage_linkage", "score report and stage journal attempts differ")
        verified += len(artifacts)
    return verified


def _tensor_state_sha256(state: Mapping[str, Any]) -> str:
    import torch

    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name]
        require(isinstance(value, torch.Tensor), "checkpoint",
                f"checkpoint tensor {name} is not a tensor")
        payload = value.detach().to("cpu", torch.float32).contiguous()
        digest.update(name.encode())
        digest.update(json.dumps(list(payload.shape)).encode())
        digest.update(payload.numpy().tobytes())
    return digest.hexdigest()


def _verify_checkpoints(*, output_root: Path, checkpoint_stage: str,
                        checkpoint_paths: Mapping[str, Path],
                        specs: Sequence[Mapping[str, Any]],
                        evidence_class: str,
                        predictions: Mapping[str, Sequence[Mapping[str, Any]]],
                        prepared: Path, manifest: Mapping[str, Any]) -> dict:
    from replica_lab import artifact_publication as publication

    states: dict[str, dict[str, str]] = {}
    for spec in specs:
        arm = spec["arm_id"]
        if arm not in checkpoint_paths:
            continue
        relative = checkpoint_paths[arm].relative_to(output_root).as_posix()
        payload = publication.load_torch(output_root, relative)
        require(payload.get("format") == CHECKPOINT_FORMAT, "checkpoint",
                f"{arm}: checkpoint format differs")
        require(payload.get("protocol_id") == PROTOCOL_ID, "checkpoint",
                f"{arm}: checkpoint protocol differs")
        require(payload.get("backbone_saved") is False
                and payload.get("base_saved") is False, "checkpoint",
                f"{arm}: checkpoint contains backbone weights")
        stored = payload.get("meta")
        require(isinstance(stored, Mapping), "checkpoint",
                f"{arm}: checkpoint metadata is missing")
        require(stored.get("arm_id") == arm
                and stored.get("checkpoint_id") == spec["checkpoint_id"],
                "checkpoint", f"{arm}: checkpoint identity differs")
        require(stored.get("answer_span_ce_weight")
                == spec["answer_span_ce_weight"], "checkpoint",
                f"{arm}: checkpoint objective weight differs")
        parent = _digest(stored.get("parent_state_sha256"),
                         f"{arm} parent state")
        require(payload.get("parent_state_sha256") == parent, "checkpoint",
                f"{arm}: parent digest disagrees with metadata")
        extension = _digest(payload.get("extension_state_sha256"),
                            f"{arm} extension state")
        if evidence_class == NATIVE_CLASS:
            expert_state = payload.get("expert_state")
            require(isinstance(expert_state, Mapping)
                    and set(expert_state) == {"gate", "up", "down"},
                    "checkpoint", f"{arm}: expert tensor set differs")
            require(_tensor_state_sha256(expert_state) == extension,
                    "checkpoint", f"{arm}: extension tensor digest differs")
            require(payload.get("trainable_names") == ["gate", "up", "down"],
                    "checkpoint", f"{arm}: trainable tensor names differ")
        completed = stored.get("events_completed")
        updates = int(spec["updates"])
        require(completed == list(range(updates)), "checkpoint",
                f"{arm}: checkpoint is not the declared complete event prefix")
        require(stored.get("updates_completed") == updates, "checkpoint",
                f"{arm}: checkpoint update count differs")
        teaching_name = ("sealed_final_teaching.jsonl"
                         if checkpoint_stage == staged.STAGE_FINAL
                         else "teaching.jsonl")
        teaching = _verified_rows(prepared / teaching_name, manifest)
        require([int(teaching[index]["event_ordinal"]) for index in completed]
                == list(stored.get("event_ids", [])), "checkpoint",
                f"{arm}: checkpoint event IDs differ")
        require([sha256_json(teaching[index]) for index in completed]
                == list(stored.get("event_digests", [])), "checkpoint",
                f"{arm}: checkpoint event content digests differ")
        require(stored.get("checkpoint_id_before")
                == teaching[0]["checkpoint_id_before"], "checkpoint",
                f"{arm}: checkpoint initial event identity differs")
        require(stored.get("checkpoint_id_after")
                == teaching[completed[-1]]["checkpoint_id_after"],
                "checkpoint", f"{arm}: checkpoint final event identity differs")
        restored = stored.get("restored") is True
        if restored:
            require(extension == parent, "checkpoint",
                    f"{arm}: reset checkpoint did not restore exact parent")
        states[arm] = {"parent": parent, "extension": extension,
                       "restored": restored}
    parent_states = {state["parent"] for state in states.values()}
    require(len(parent_states) == 1, "checkpoint",
            "checkpoint parent identities differ across arms")
    parent = next(iter(parent_states))
    if evidence_class == NATIVE_CLASS:
        for spec in specs:
            arm = spec["arm_id"]
            state = states.get(arm)
            expected = parent if state is None or state["restored"] else state["extension"]
            for row in predictions[arm]:
                require(row.get("reader_state_sha256") == expected,
                        "checkpoint_linkage",
                        f"{arm}: prediction reader state does not match checkpoint")
    return states




def _verify_learning_evidence(*, report: Mapping[str, Any],
                              specs: Sequence[Mapping[str, Any]],
                              checkpoint_paths: Mapping[str, Path],
                              output_root: Path) -> None:
    """Recompute the per-arm learning sidecar from checkpoint metadata."""
    from replica_lab import artifact_publication as publication

    actual = report.get("learning_evidence")
    if not isinstance(actual, Mapping):
        raise ProtocolError("learning_evidence", "fit learning sidecar must be an object")
    expected_arms = {spec["arm_id"] for spec in specs}
    if set(actual) != expected_arms:
        raise ProtocolError("learning_evidence", "fit learning arm set differs")
    for spec in specs:
        arm = spec["arm_id"]
        updates = int(spec.get("updates", 0))
        if updates == 0:
            expected = learning_success_evidence(
                declared_updates=0, execution_complete=True,
                optimizer_steps=0, changed_trainable_names=[])
        else:
            relative = checkpoint_paths[arm].relative_to(output_root).as_posix()
            payload = publication.load_torch(output_root, relative)
            stored = payload.get("meta", {})
            counters = stored.get("arm_counters", {})
            changed = stored.get("changed_trainable_names", [])
            require(isinstance(changed, list)
                    and all(isinstance(name, str) and name for name in changed)
                    and set(changed) <= set(spec["trainable_groups"]),
                    "learning_evidence", f"{arm}: changed tensor set differs")
            expected = learning_success_evidence(
                declared_updates=updates, execution_complete=True,
                optimizer_steps=int(counters.get("optimizer_steps", 0)),
                retry_optimizer_steps=int(
                    counters.get("restart_optimizer_steps", 0)),
                no_op_optimizer_steps=int(stored.get("no_op_optimizer_steps", 0)),
                changed_trainable_names=changed,
                restored_parent=stored.get("restored") is True)
            require(expected["learning_success"] is True, "learning_evidence",
                    f"{arm}: declared learning did not change trainable tensors")
        if actual.get(arm) != expected:
            raise ProtocolError(
                "learning_evidence", f"{arm}: learning sidecar differs from checkpoint")


def _recompute_report(*, stage: str, cases: Sequence[Mapping[str, Any]],
                      labels: Sequence[Mapping[str, Any]],
                      predictions: Sequence[Mapping[str, Any]],
                      specs: Sequence[Mapping[str, Any]],
                      config: Mapping[str, Any], binding: Mapping[str, Any],
                      decode_fn: Callable[[Sequence[int]], str],
                      original: Mapping[str, Any]) -> dict:
    binding_sha = tokenizer_binding_sha(binding)
    common = dict(
        protocol_id=PROTOCOL_ID, cases=cases, labels=labels,
        predictions=predictions, arm_specs=specs, binding=binding,
        binding_sha256=binding_sha, decode_fn=decode_fn,
        verify_decoded=lambda prediction, bound, decoder:
            verify_decoded_text(prediction, bound, decoder),
        gates=config["evaluation"]["gates"],
        objective_audit=original.get("objective_audit"),
        normalize=normalize_answer,
    )
    if stage == staged.STAGE_FIT:
        report = staged.score_fit_stage(**common)
    elif stage == staged.STAGE_DEVELOPMENT:
        report = staged.score_development_stage(
            **common, seed=int(config["seed"]),
            resamples=int(config["evaluation"]["paired_bootstrap_resamples"]),
            control_arms=tuple(config["evaluation"]["control_arms"]))
    else:
        report = staged.score_final_stage(**common)
    report["scoring_counts"] = dict(report["cases"])
    for key in ("scoring_counts", "evidence_class", "native_execution_admissible",
                "attempt_id", "confirmation_only", "learning_evidence"):
        if key in original:
            report[key] = original[key]
    require(set(report) == set(original), "score_recompute",
            f"score report fields differ: {sorted(set(report) ^ set(original))}")
    require(report == dict(original), "score_recompute",
            "score report metrics differ from underlying evidence")
    return report


def verify_stage_evidence(*, stage: str, output_root: str | Path,
                          prepared: str | Path, config: Mapping[str, Any],
                          config_path: str | Path, manifest: Mapping[str, Any],
                          protocol_commit: str, binding: Mapping[str, Any],
                          decode_fn: Callable[[Sequence[int]], str] | None = None,
                          expected_evidence_class: str | None = None,
                          expected_attempt_id: str | None = None,
                          report_path: str | Path | None = None,
                          receipt_path: str | Path | None = None,
                          prediction_paths: Mapping[str, str | Path] | None = None,
                          checkpoint_paths: Mapping[str, str | Path] | None = None,
                          require_publication: bool = False,
                          spy: Any = None) -> dict:
    """Resolve and independently re-score one stage's durable evidence."""
    output_root = Path(output_root)
    prepared = Path(prepared)
    specs = arm_specs(config, stage)
    defaults = _default_paths(output_root, stage, specs)
    report_path = Path(report_path) if report_path else defaults[0]
    receipt_path = Path(receipt_path) if receipt_path else defaults[1]
    prediction_paths = ({arm: Path(path) for arm, path in prediction_paths.items()}
                        if prediction_paths is not None else defaults[2])
    checkpoint_paths = ({arm: Path(path) for arm, path in checkpoint_paths.items()}
                        if checkpoint_paths is not None else defaults[3])
    checkpoint_stage = defaults[4]
    require(set(prediction_paths) == {spec["arm_id"] for spec in specs},
            "artifact", "prediction artifact arm set differs")
    require(set(checkpoint_paths) == {
        spec["arm_id"] for spec in specs if int(spec["updates"]) > 0},
        "artifact", "checkpoint artifact arm set differs")
    for path in (report_path, receipt_path, *prediction_paths.values(),
                 *checkpoint_paths.values()):
        require(path.is_file(), "artifact", f"missing evidence artifact: {path}")
    report = read_json(report_path)
    receipt = read_json(receipt_path)
    expectation = receipt_expectation(
        config=config, config_path=config_path, manifest=manifest,
        binding=binding, protocol_commit=protocol_commit)
    staged.verify_stage_receipt(receipt, stage=stage, expectation=expectation)
    require(report.get("format") == staged.SCORE_REPORT_FORMAT,
            "score_report", "score report format differs")
    require(report.get("stage") == stage
            and report.get("protocol_id") == PROTOCOL_ID, "score_report",
            "score report stage or protocol differs")
    expected_report_digest = sha256_json({
        key: value for key, value in report.items() if key != "report_sha256"})
    require(receipt["score_report_sha256"] == expected_report_digest,
            "score_report", "receipt does not bind the actual score report")
    require(receipt["evidence_class"] == report.get("evidence_class"),
            "evidence_class", "receipt and report execution classes differ")
    require(receipt["native_execution_admissible"]
            == report.get("native_execution_admissible"),
            "evidence_class", "receipt and report native admissibility differ")
    if expected_evidence_class is not None:
        require(report.get("evidence_class") == expected_evidence_class,
                "evidence_class",
                f"expected {expected_evidence_class}, got {report.get('evidence_class')}")
    if expected_attempt_id is not None:
        require(receipt["attempt_id"] == expected_attempt_id,
                "stage_linkage", "receipt attempt differs from the current attempt")
        require(report.get("attempt_id") == expected_attempt_id,
                "stage_linkage", "score report attempt differs from the receipt")
    cases, labels = load_stage_rows(prepared, stage, manifest, spy=spy)
    predictions: dict[str, list[dict]] = {
        spec["arm_id"]: [] for spec in specs}
    for path in dict.fromkeys(prediction_paths.values()):
        for row in read_jsonl(path):
            arm = row.get("arm_id")
            require(arm in predictions, "scoring",
                    f"unexpected arm {arm!r}")
            predictions[arm].append(row)
    for spec in specs:
        arm = spec["arm_id"]
        for row in predictions[arm]:
            require(row.get("evidence_class") == report["evidence_class"],
                    "evidence_class", f"{arm}: prediction execution class differs")
            require(row.get("checkpoint_id") == spec["checkpoint_id"],
                    "checkpoint_linkage", f"{arm}: prediction checkpoint differs")
            _digest(row.get("reader_state_sha256"), f"{arm} reader state")
    flat = [row for spec in specs for row in predictions[spec["arm_id"]]]
    checkpoint_states = _verify_checkpoints(
        output_root=output_root, checkpoint_stage=checkpoint_stage,
        checkpoint_paths=checkpoint_paths, specs=specs,
        evidence_class=report["evidence_class"], predictions=predictions,
        prepared=prepared, manifest=manifest)
    if stage == staged.STAGE_FIT:
        _verify_learning_evidence(
            report=report, specs=specs, checkpoint_paths=checkpoint_paths,
            output_root=output_root)
    if decode_fn is None:
        decode_fn = tokenizer_decode(binding, ROOT / "checkpoints/minicpm5_source")
    recomputed = _recompute_report(
        stage=stage, cases=cases, labels=labels, predictions=flat,
        specs=specs, config=config, binding=binding, decode_fn=decode_fn,
        original=report)
    published = 0
    if require_publication:
        published = _verify_publication(
            output_root=output_root, stage=stage,
            checkpoint_stage=checkpoint_stage,
            expected_paths=[report_path, receipt_path,
                            *prediction_paths.values(), *checkpoint_paths.values()],
            config_sha256=expectation.config_sha256,
            manifest_sha256=expectation.manifest_sha256,
            protocol_commit=protocol_commit,
            receipt_attempt_id=receipt["attempt_id"],
            report_attempt_id=report.get("attempt_id"))
    return {
        "receipt": receipt,
        "report": report,
        "recomputed": recomputed,
        "predictions": predictions,
        "checkpoint_states": checkpoint_states,
        "published_artifacts_verified": published,
        "cases": cases,
        "labels": labels,
        "expectation": expectation,
    }

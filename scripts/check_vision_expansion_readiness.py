#!/usr/bin/env python3
"""Check whether the reviewed vision expansion may be prepared.

The check is deliberately read-only apart from its own readiness report.  It
verifies the preserved 9/21/8/24 partition identity and independently replays
the positive diagnostic assessment from its referenced artifacts.  A positive
result permits only a new preparation protocol; native training and label
access remain closed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import validate_vision_acceptance as acceptance  # noqa: E402

DEFAULT_CONFIG = ROOT / "configs/vision_expansion_readiness_v1.json"
DEFAULT_REPORT = ROOT / "reports/vision_expansion_readiness_20261028/readiness.json"


class ReadinessError(ValueError):
    """Raised for malformed readiness inputs."""


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ReadinessError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _partition_checks(config: Mapping[str, Any], root: Path) -> dict[str, Any]:
    inputs = config["inputs"]
    identity_path = _resolve(root, inputs["partition_identity"])
    receipt_path = _resolve(root, inputs["admission_receipt"])
    specs_path = _resolve(root, inputs["synthetic_scene_specs"])
    identity = _read(identity_path)
    receipt = _read(receipt_path)
    specs = _read(specs_path)
    expected = config["expected_partitions"]
    counts = identity.get("counts", {})
    partitions = identity.get("partitions", {})
    violations: list[str] = []
    for name in ("admitted", "exclusions", "exposed"):
        rows = partitions.get(name)
        if not isinstance(rows, list) or len(rows) != expected[name]:
            violations.append(f"partition_count:{name}")
        if counts.get(name) != expected[name]:
            violations.append(f"declared_count:{name}")
    scenes = specs.get("scenes")
    if (not isinstance(scenes, list)
            or len(scenes) != expected["diagnostic_eval_scenes"]):
        violations.append("partition_count:diagnostic_eval_scenes")
    if counts.get("diagnostic_eval_scenes") != expected["diagnostic_eval_scenes"]:
        violations.append("declared_count:diagnostic_eval_scenes")
    if specs.get("format") != "replica.synthetic_scene_specs.v1":
        violations.append("synthetic_scene_format")

    admitted = partitions.get("admitted") or []
    admitted_ids = [row.get("child_row_id") for row in admitted
                    if isinstance(row, Mapping)]
    if receipt.get("decision") != "admit_training":
        violations.append("admission_receipt_decision")
    if receipt.get("admitted_row_ids") != admitted_ids:
        violations.append("admission_receipt_row_identity")
    if identity.get("admission", {}).get("receipt_sha256") != receipt.get(
            "receipt_sha256"):
        violations.append("admission_receipt_digest_binding")
    if identity.get("dedup", {}).get("admitted_collisions") != []:
        violations.append("admitted_dedup_collision")
    for key in ("admitted_vs_eval_collisions", "admitted_vs_exposed_collisions"):
        if identity.get("dedup", {}).get(key) != []:
            violations.append(key)

    groups: dict[str, set[str]] = {}
    for name in ("admitted", "exclusions", "exposed"):
        rows = partitions.get(name) or []
        values: set[str] = set()
        for row in rows:
            if not isinstance(row, Mapping) or not isinstance(
                    row.get("group_key"), str) or not row["group_key"]:
                violations.append(f"source_group_missing:{name}")
                continue
            group = row["group_key"]
            if group in values:
                violations.append(f"source_group_duplicate:{name}")
            values.add(group)
            if name == "admitted":
                if row.get("role") != config["source_policy"]["admitted_role"]:
                    violations.append("admitted_role")
                source = row.get("source")
                if not isinstance(source, Mapping) or source.get("split") != (
                        config["source_policy"]["admitted_source_split"]):
                    violations.append("admitted_source_split")
                if row.get("quarantined") is not False:
                    violations.append("admitted_quarantine")
                terms = row.get("license_terms")
                if not isinstance(terms, Mapping) or terms.get(
                        "research_use_only") is not True:
                    violations.append("admitted_research_terms")
        groups[name] = values
    for left, right in (("admitted", "exclusions"),
                        ("admitted", "exposed")):
        if groups[left] & groups[right]:
            violations.append(f"source_group_overlap:{left}:{right}")
    if not groups["exposed"] <= groups["exclusions"]:
        violations.append("exposed_not_in_exclusion_ledger")
    return {
        "ok": not violations,
        "violations": violations,
        "counts": {name: counts.get(name) for name in expected},
        "source_group_counts": {name: len(values)
                                for name, values in groups.items()},
        "admitted_row_ids": admitted_ids,
        "partition_identity_sha256": _sha256(identity_path),
        "admission_receipt_sha256": _sha256(receipt_path),
        "synthetic_scene_specs_sha256": _sha256(specs_path),
        "source_policy": config["source_policy"],
    }


def _positive_diagnostic(config: Mapping[str, Any], root: Path) -> dict[str, Any]:
    inputs = config["inputs"]
    path = _resolve(root, inputs["positive_diagnostic"])
    acceptance_path = _resolve(root, inputs["acceptance_protocol"])
    if not path.is_file():
        return {"ok": False, "present": False,
                "violations": ["positive_diagnostic_missing"],
                "path": str(path)}
    assessment = _read(path)
    violations: list[str] = []
    if assessment.get("format") != "replica.vision_acceptance.assessment.v1":
        violations.append("positive_diagnostic_format")
    if assessment.get("status") != "accepted" or assessment.get("ok") is not True:
        violations.append("positive_diagnostic_not_accepted")
    for key in ("learned_evidence_ok", "engineering_evidence_ok"):
        if assessment.get(key) is not True:
            violations.append(f"positive_diagnostic_{key}")
    for key in ("generated_answer_improvement_measured",
                "causal_image_dependence_measured", "text_retention_measured",
                "fresh_process_reload_measured"):
        if assessment.get("claims", {}).get(key) is not True:
            violations.append(f"positive_diagnostic_claim:{key}")
    evidence = assessment.get("evidence", {})
    if evidence.get("candidate_device") != "mps":
        violations.append("positive_diagnostic_not_mps")
    if evidence.get("candidate_cpu_fallback_enabled") is not False:
        violations.append("positive_diagnostic_cpu_fallback")
    input_records = assessment.get("inputs")
    paths: dict[str, Path] = {}
    if not isinstance(input_records, Mapping):
        violations.append("positive_diagnostic_inputs")
        input_records = {}
    for name in ("baseline", "candidate", "reload"):
        record = input_records.get(name)
        if not isinstance(record, Mapping) or not isinstance(
                record.get("path"), str) or not isinstance(
                record.get("sha256"), str):
            violations.append(f"positive_diagnostic_input:{name}")
            continue
        artifact = _resolve(root, record["path"])
        paths[name] = artifact
        if not artifact.is_file() or _sha256(artifact) != record["sha256"]:
            violations.append(f"positive_diagnostic_input_hash:{name}")
    if len(paths) == 3 and acceptance_path.is_file():
        try:
            replay = acceptance.assess(
                baseline_path=paths["baseline"], candidate_path=paths["candidate"],
                reload_path=paths["reload"], config_path=acceptance_path,
                role=str(assessment.get("role", "dev")))
            for key in ("status", "ok", "failures", "missing_evidence",
                        "metrics", "claims"):
                if replay[key] != assessment.get(key):
                    violations.append(f"positive_diagnostic_replay:{key}")
            if replay["inputs"]["candidate"]["sha256"] != (
                    assessment["inputs"]["candidate"]["sha256"]):
                violations.append("positive_diagnostic_candidate_binding")
        except (OSError, ValueError, json.JSONDecodeError) as error:
            violations.append(f"positive_diagnostic_replay_error:{error}")
    else:
        violations.append("positive_diagnostic_acceptance_protocol")
    return {
        "ok": not violations,
        "present": True,
        "path": str(path),
        "sha256": _sha256(path),
        "status": assessment.get("status"),
        "candidate_device": evidence.get("candidate_device"),
        "candidate_cpu_fallback_enabled": evidence.get(
            "candidate_cpu_fallback_enabled"),
        "violations": violations,
    }


def assess(*, root: str | Path = ROOT,
           config_path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    root_path = Path(root).resolve()
    config_path = Path(config_path).resolve()
    config = _read(config_path)
    if config.get("format") != "replica.vision_expansion_readiness.v1":
        raise ReadinessError("expansion readiness format differs")
    partitions = _partition_checks(config, root_path)
    diagnostic = _positive_diagnostic(config, root_path)
    allowed = partitions["ok"] and diagnostic["ok"]
    policy = config["policy"]
    return {
        "format": "replica.vision_expansion_readiness.assessment.v1",
        "name": config["name"],
        "status": (policy["accepted_diagnostic_status"] if allowed
                   else policy["missing_positive_diagnostic_status"]),
        "expansion_preparation_allowed": allowed,
        "native_execution": False,
        "training_authorized": False,
        "development_authorized": False,
        "heldout_authorized": False,
        "promotion": False,
        "retry_authorized": False,
        "model_loads": 0,
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "partitions": partitions,
        "positive_diagnostic": diagnostic,
        "expansion_scope": config["expansion_scope"],
        "refusals": [*partitions["violations"], *diagnostic["violations"]],
        "next_step": ("predeclare a bounded expansion protocol; no native command"
                      if allowed else
                      "provide an accepted MPS diagnostic with complete evidence"),
        "note": ("Readiness only. The 9/21/8/24 partitions are preserved; "
                 "no rows, sources, labels or native jobs are created."),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["verify"])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args(argv)
    try:
        report = assess(config_path=args.config)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n",
                               encoding="utf-8")
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"expansion readiness refused: {error}", file=sys.stderr)
        return 2
    print(json.dumps({
        "status": report["status"],
        "expansion_preparation_allowed": report["expansion_preparation_allowed"],
        "native_execution": False,
        "model_loads": 0,
        "refusals": report["refusals"],
        "output": str(args.output),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

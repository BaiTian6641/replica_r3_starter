#!/usr/bin/env python3
"""Evaluate a vision diagnostic's learned and engineering acceptance evidence.

This command reads JSON reports only.  It never loads a model, opens a label
role, executes native code, writes an authorization receipt or promotes an
artifact.  The candidate report's SHA-256 is supplied to the evaluator and
must match the fresh-process reload receipt's ``run_report_sha256``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from replica_lab.vision_acceptance import (  # noqa: E402
    VisionAcceptanceError, answer_from_report, evaluate_vision_acceptance,
)

DEFAULT_CONFIG = ROOT / "configs/vision_diagnostic_acceptance_v1.json"


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise VisionAcceptanceError(f"JSON report must be an object: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _answer_with_provenance(report: dict[str, Any], role: str) -> dict[str, Any]:
    answer = dict(answer_from_report(report, role))
    if "control_provenance" not in answer and "image_roles" not in answer:
        container = report.get(role)
        provenance = report.get("control_provenance")
        if provenance is None and isinstance(container, dict):
            provenance = container.get("control_provenance")
        if provenance is not None:
            answer["control_provenance"] = provenance
    return answer


def assess(*, baseline_path: Path, candidate_path: Path, reload_path: Path,
           config_path: Path = DEFAULT_CONFIG, role: str = "dev") -> dict[str, Any]:
    config = _read(config_path)
    thresholds = config.get("thresholds")
    required = config.get("required_reload_comparisons")
    if not isinstance(required, list) or not all(
            isinstance(item, str) and item for item in required):
        raise VisionAcceptanceError("required_reload_comparisons must be a string list")
    baseline = _read(baseline_path)
    candidate = _read(candidate_path)
    reload_receipt = _read(reload_path)
    candidate_digest = _sha256(candidate_path)
    assessment = evaluate_vision_acceptance(
        baseline_answer=answer_from_report(baseline, role),
        candidate_answer=_answer_with_provenance(candidate, role),
        retention=candidate.get("retention"),
        reload_receipt=reload_receipt,
        thresholds=thresholds,
        candidate_report_sha256=candidate_digest,
        required_reload_comparisons=required,
        expected_control_provenance=config.get("control_provenance"),
    )
    assessment.update({
        "name": config.get("name", "vision_diagnostic_acceptance_v1"),
        "role": role,
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "inputs": {
            "baseline": {"path": str(baseline_path),
                         "sha256": _sha256(baseline_path)},
            "candidate": {"path": str(candidate_path),
                          "sha256": candidate_digest},
            "reload": {"path": str(reload_path),
                       "sha256": _sha256(reload_path)},
        },
        "evidence": {
            "candidate_device": candidate.get("device"),
            "candidate_cpu_fallback_enabled": candidate.get(
                "cpu_fallback_enabled"),
            "candidate_native_execution": candidate.get("native_execution"),
        },
        "native_execution": False,
        "model_loads": 0,
        "authorization_created": False,
        "note": ("Assessment only. Historical reports are not rewritten; "
                 "accepted diagnostic evidence does not authorize development, "
                 "held-out access, promotion or retry."),
    })
    return assessment


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-report", type=Path, required=True)
    parser.add_argument("--candidate-report", type=Path, required=True)
    parser.add_argument("--reload-report", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--role", default="dev")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f"fresh output is required: {args.output}")
    try:
        report = assess(
            baseline_path=args.baseline_report,
            candidate_path=args.candidate_report,
            reload_path=args.reload_report,
            config_path=args.config,
            role=args.role,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n",
                               encoding="utf-8")
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"vision acceptance refused: {error}", file=sys.stderr)
        return 2
    print(json.dumps({
        "status": report["status"],
        "ok": report["ok"],
        "promotion": False,
        "native_execution": False,
        "output": str(args.output),
        "failures": report["failures"],
        "missing_evidence": report["missing_evidence"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

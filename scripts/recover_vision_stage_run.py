#!/usr/bin/env python3
"""Recover a stage run report from completed artifacts without retraining."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_vision_stage_diagnostic import (  # noqa: E402
    _audit_receipt, _forgetting, _read_jsonl, _sha, _verify_prepared)


def _retention(protocol, path, label):
    from run_vision_curriculum import load_retention
    from replica_lab.vision_qualification import score_retention
    prompts, labels = load_retention(protocol)
    rows = _read_jsonl(path)
    scored = score_retention(
        [{"case_id": row["case_id"], "group_id": row["family"],
          "family": row["family"], "split": row["split"],
          "prompt": row["prompt"]} for row in prompts],
        [{"case_id": case_id, "answer": answer}
         for case_id, answer in labels.items()],
        [{"case_id": row["case_id"],
          "generated_text": row["generated_text"]} for row in rows])
    return {"label": label, "rows": rows, "score": scored}


def _extension(protocol, path):
    from replica_lab.vision_qualification import validate_extension_inventory
    result = validate_extension_inventory(
        path, expected_names=sorted(
            name for names in json.loads((path / "manifest.json").read_text())["components"].values()
            for name in names),
        vision_pins=protocol["vision"])
    result["receipt_sha256"] = _sha(path / "manifest.json")
    return {key: result[key] for key in (
        "format", "receipt_sha256", "weights_sha256", "manifest_digest",
        "tensor_names", "base_saved", "components")}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--failure", type=Path, required=True)
    args = parser.parse_args(argv)
    protocol_path = args.protocol.resolve()
    protocol, manifest, implementation = _verify_prepared(protocol_path)
    run = args.run
    failure_path = args.failure
    failure = json.loads(failure_path.read_text())
    if failure.get("error") != "'max_caption_f1_forgetting_drop'":
        parser.error("Recovery is restricted to the known finalization-key failure")
    original_protocol = run / "protocol.json"
    if not original_protocol.is_file():
        parser.error("Original executed protocol copy is missing")
    backup = run / "protocol_attempt05.json"
    if not backup.exists():
        backup.write_bytes(original_protocol.read_bytes())
    original_protocol_sha = _sha(original_protocol)
    trace_a = _read_jsonl(run / "stage_a_updates.jsonl")
    trace_b = _read_jsonl(run / "stage_b_updates.jsonl")
    if len(trace_a) != 16 or len(trace_b) != 16:
        parser.error("Recovery requires exactly 16 updates per stage")
    dev_a = json.loads((run / "stage_a_dev_score.json").read_text())
    dev_b = json.loads((run / "stage_b_dev_score.json").read_text())
    audit_b = json.loads((run / "audit_after_stage_b_score.json").read_text())
    retention_pre = _retention(protocol, run / "retention_pre.jsonl", "before_updates")
    retention_a = _retention(protocol, run / "retention_stage_a.jsonl", "after_stage_a")
    retention_b = _retention(protocol, run / "retention_stage_b.jsonl", "after_stage_b")
    stage_a_extension = _extension(protocol, run / "stage_a/extension")
    stage_b_extension = _extension(protocol, run / "stage_b/extension")
    artifacts = {}
    for name in ("stage_a_updates.jsonl", "stage_b_updates.jsonl",
                 "stage_a_dev_score.json", "stage_b_dev_score.json",
                 "audit_after_stage_b_score.json", "retention_pre.jsonl",
                 "retention_stage_a.jsonl", "retention_stage_b.jsonl",
                 "stage_a/extension/manifest.json", "stage_b/extension/manifest.json"):
        artifacts[name] = _sha(run / name)
    original_protocol.write_bytes(protocol_path.read_bytes())
    report = {
        "format": "replica.vision_stage_diagnostic.run.v1",
        "status": "completed", "disposition": "pending_reload",
        "protocol_sha256": _sha(protocol_path),
        "implementation_sha256": implementation,
        "data_manifest_sha256": protocol["data"]["manifest"]["sha256"],
        "data_seals": protocol["data"]["seals"],
        "source_revision": protocol["model"]["revision"],
        "device": "mps", "cpu_fallback_enabled": False,
        "initialization": {"seed": protocol["train"]["init_seed"],
                           "warm_start": None},
        "trainable_scope": "vision_projector.", "trainable_tensors": 6,
        "optimizer": {
            **protocol["train"],
            "optimizer_class": "torch.optim.AdamW",
            "implementation": "native_fused",
            "state_audit_recovered": False,
            "audit_unavailable_reason": "Original process stopped during final report assembly",
        },
        "backward_only_probe": {"status": "not_recovered"},
        "stage_a": {"updates": trace_a, "extension": stage_a_extension,
                    "retention": retention_a, "development": dev_a},
        "stage_b": {"updates": trace_b, "extension": stage_b_extension,
                    "retention": retention_b, "development": dev_b,
                    "audit": audit_b},
        "retention_before": retention_pre,
        "forgetting": _forgetting(dev_a, dev_b, retention_a, retention_b,
                                  protocol["evaluation"]),
        "audit_receipt": _audit_receipt(protocol, run, create=False),
        "total_updates": len(trace_a) + len(trace_b),
        "total_examples": sum(len(row["case_ids"]) for row in trace_a + trace_b),
        "base_saved": False, "promotion": False,
        "profile": failure["profile"],
        "recovery": {
            "original_failure_sha256": _sha(failure_path),
            "original_protocol_sha256": original_protocol_sha,
            "recovery_script_sha256": _sha(Path(__file__)),
            "native_updates_rerun": 0,
            "optimizer_state_audit_recovered": False,
            "artifact_sha256": artifacts,
            "scope": "offline assembly from completed native artifacts; no quality claim",
        },
    }
    (run / "run_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": "recovered", "updates": report["total_updates"],
                      "examples": report["total_examples"],
                      "report": str(run / "run_report.json")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

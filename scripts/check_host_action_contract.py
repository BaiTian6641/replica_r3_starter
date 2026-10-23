#!/usr/bin/env python3
"""Run actual pure-tool host fixtures; no neural model, acquisition or learning."""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from replica_lab.host_action_contract import (
    VERSION, HostActionSession, HostBinding, HostObservation, HostSnapshot, digest,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="fresh evidence directory")
    parser.add_argument("--model-lock", type=Path, default=Path("configs/minicpm5_sealed_lock.local.json"))
    args = parser.parse_args()
    model_lock_hash = hashlib.sha256(args.model_lock.read_bytes()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=False)
    contract_path = Path(__file__).resolve().parents[1] / "src/replica_lab/host_action_contract.py"
    policy_hash = digest({"contract_sha256": hashlib.sha256(contract_path.read_bytes()).hexdigest(),
                          "fixture_policy_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    binding = HostBinding(model_lock_hash, policy_hash,
                          "assistant_for_source", "fictional-subject", "replica-a", "fixture-agent",
                          "Compute INPUT XOR MASK using observed registers.",
                          "local_evaluation", 10, ("input", "mask"))
    snapshot = HostSnapshot(1, (HostObservation.make("input", "INPUT=37"),
                                HostObservation.make("mask", "MASK=12")))

    def verify(action_id="verify", epoch=1, sources=None):
        return {"action_id": action_id, "kind": "verify", "epoch": epoch,
                "source_ids": ["input", "mask"] if sources is None else sources}

    def answer(receipt_id, action_id="answer", epoch=1):
        return {"action_id": action_id, "kind": "answer", "epoch": epoch,
                "receipt_id": receipt_id, "value": 41}

    checks = {}
    session = HostActionSession(binding, snapshot)
    verified = session.submit(verify())
    receipt_id = verified["detail"]["receipt_id"]
    released = session.submit(answer(receipt_id))
    checks["actual_success_and_matching_answer"] = (
        verified["detail"]["receipt"]["value"] == 41 and released["status"] == "ok")
    checks["duplicate_does_not_spend_tool_budget"] = (
        session.submit(verify()) == verified and session.tool_calls == 1)
    session.write_journal(args.output / "success_before_restart.jsonl")
    restarted = HostActionSession.read_journal(
        args.output / "success_before_restart.jsonl", current_snapshot=snapshot,
        expected_binding=binding, trusted_head=session.head)
    checks["restart_preserves_goal_receipt_and_spent_budget"] = (
        restarted.binding.goal == binding.goal and restarted.receipts == session.receipts
        and restarted.tool_calls == session.tool_calls and restarted.head == session.head)
    restarted.update_snapshot(HostSnapshot(2, (snapshot.observations[0],)))
    checks["revocation_precedes_duplicate_answer_and_receipt_release"] = all(
        restarted.submit(proposal)["detail"].get("reason") == "cached_action_invalidated"
        for proposal in (verify(), answer(receipt_id)))
    checks["old_receipt_cannot_authorize_new_epoch_answer"] = (
        restarted.submit(answer(receipt_id, "answer-current", 2))["status"] == "denied")
    restarted.write_journal(args.output / "success_after_invalidation.jsonl")

    failures = {}
    for name, text, expected in (("missing", "INPUT=37", "missing_register"),
                                 ("malformed", "INPUT=256\nMASK=12", "malformed_register")):
        case = HostActionSession(binding, HostSnapshot(1, (HostObservation.make("input", text),)))
        result = case.submit(verify(sources=["input"]))
        failures[name] = result
        checks[f"actual_{name}_tool_failure_recorded"] = (
            result["status"] == "tool_error" and result["detail"]["receipt"]["value"] is None
            and result["detail"]["receipt"]["error"] == expected and case.tool_calls == 1)
        case.write_journal(args.output / f"{name}_tool_failure.jsonl")
    forged = HostActionSession(binding, snapshot)
    try:
        forged.submit({**verify(), "receipt": {"value": 41}})
        checks["model_forged_receipt_rejected"] = False
    except ValueError:
        checks["model_forged_receipt_rejected"] = forged.tool_calls == 0
    checks["model_value_alone_cannot_release_answer"] = (
        forged.submit(answer("a" * 64))["detail"]["reason"] == "current_actual_receipt_required")
    wrong_scope = HostActionSession(binding, HostSnapshot(1, (
        snapshot.observations[0], replace(snapshot.observations[1], subject_id="other-subject"))))
    checks["wrong_scope_rejected_before_actual_tool"] = (
        wrong_scope.submit(verify())["detail"]["reason"] == "source_not_eligible"
        and wrong_scope.tool_calls == 0)
    bounded = HostActionSession(replace(binding, max_decisions=2, max_tool_calls=1), snapshot)
    bounded.submit(verify())
    checks["tool_budget_denial_recorded"] = (
        bounded.submit(verify("verify-second"))["detail"]["reason"] == "tool_budget")
    checks["decision_budget_denial_recorded"] = (
        bounded.submit(verify("verify-third"))["detail"]["reason"] == "decision_budget")
    bounded.write_journal(args.output / "budget_failures.jsonl")
    files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(args.output.iterdir()) if p.is_file()}
    report = {
        "schema_version": VERSION, "kind": "deterministic_early_S3_1_host_contract_preparation",
        "status": "passed" if all(checks.values()) else "failed", "checks": checks,
        "binding": asdict(binding), "actual_success": verified, "actual_failures": failures,
        "files_sha256": files,
        "implementation_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in (
                Path(__file__), Path(__file__).resolve().parents[1] / "src/replica_lab/host_action_contract.py")},
        "model_loaded": False, "model_actions_evaluated": False,
        "neural_learning_evaluated": False, "selected_architecture_integrated": False,
        "S3_1_complete": False,
        "limitations": ["Host input/authentication and revocation synchronization are caller-owned.",
                        "Restart uses a caller-retained trusted head; external trust-store persistence is absent.",
                        "Binding is a declaration; its local lock hash does not execute or attest a model.",
                        "Historical source bytes remain in journals; no erasure or neural unlearning.",
                        "Pure local tool only; no concurrent effects, streaming or neural cache."]}
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "checks": len(checks),
                      "report": str(args.output / "report.json"), "model_loaded": False}))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Offline end-to-end software demonstration, NOT an LLM or learning-quality run.

A fixed symbolic compiler identifies an XOR rule from permitted examples.
It tests record/restart/candidate/publication plumbing. Its rule grammar and
curriculum are hand-written; it cannot support a claim of model self-improvement.
"""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from replica_lab.contracts import Record, fingerprint
from replica_lab.memory import EvidenceStore
from replica_lab.learning import ManifestRegistry, GatePolicy, evaluate_candidate


def oracle(x, key, bits=10):
    return sum(((((x >> b) & 1) + ((key >> b) & 1)) % 2) << b for b in range(bits))


def compile_rule(support):
    candidates = [k for k in range(1024) if all((x ^ k) == y for x, y in support)]
    if len(candidates) != 1:
        raise ValueError("support does not identify a unique allowed rule")
    return candidates[0]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists() and any(a.output.iterdir()):
        raise FileExistsError("choose a new run directory")
    a.output.mkdir(parents=True, exist_ok=True)
    store = EvidenceStore(a.output / "events.sqlite")
    registry = ManifestRegistry(a.output / "registry")
    initial = {"base_revision": "fixture-fixed-grammar-v1", "identity_version": "fictional-source-v1",
               "skills": {}}
    initial_eval = {"decision": "accept", "kind": "initial_fixture_state",
                    "candidate_digest": fingerprint(initial), "parent_manifest_id": None}
    first = registry.publish(initial, initial_eval)
    active = first
    cycles = []
    # Promotion/test inputs never appear in the compiler's support.
    query_inputs = list(range(256, 856))
    for cycle, key in enumerate((17, 53, 91), 1):
        support = [(x, oracle(x, key)) for x in range(12)]
        event = Record(f"experience-{cycle}", "fixture-person", "replica-a", "fixture-agent",
                       "agent_experience", cycle, json.dumps({"support": support}),
                       f"fixture-environment-{cycle}", synthetic=True)
        store.put(event, authorized_subject="fixture-person", authorized_branch="replica-a")
        fitted = compile_rule(support)
        payload = {**initial, "skills": {**active["payload"]["skills"], f"family-{cycle}": fitted}}
        before = [float(x == oracle(x, key)) for x in query_inputs]
        after = [float((x ^ fitted) == oracle(x, key)) for x in query_inputs]
        # An already learned task or trivial identity task is a retention fixture.
        retention_before = [1.0] * len(query_inputs)
        retention_after = [1.0] * len(query_inputs)
        evaluation = evaluate_candidate(before, after, retention_before, retention_after,
            {"source_scope": True, "causal_support": True, "base_unchanged": True},
            GatePolicy(bootstrap_draws=200))
        evaluation.update(candidate_digest=fingerprint(payload), parent_manifest_id=active["manifest_id"])
        active = registry.publish(payload, evaluation, expected_parent=active["manifest_id"])
        cycles.append({"cycle": cycle, "compiler": "fixed_exhaustive_xor_grammar",
                       "evaluation": evaluation, "manifest_id": active["manifest_id"]})
    bad = evaluate_candidate([1.0]*600, [0.0]*600, [1.0]*600, [0.0]*600,
                             {"source_scope": True}, GatePolicy(bootstrap_draws=200))
    assert bad["decision"] == "reject"
    store.close()
    restarted = EvidenceStore(a.output / "events.sqlite")
    count = len(restarted.visible(subject_id="fixture-person", branch_id="replica-a", as_of=3))
    restarted.close()
    reloaded = ManifestRegistry(a.output / "registry")
    restart_equal = reloaded.active()["manifest_id"] == active["manifest_id"]
    rolled = reloaded.rollback(first["manifest_id"])
    report = {"kind": "deterministic_contract_demo", "model_weights_loaded": False,
              "model_learning_evaluated": False, "autonomous_curriculum_evaluated": False,
              "generalization_scope": "new integer inputs in a fixed XOR grammar only",
              "cycles": cycles, "harmful_candidate_decision": bad["decision"],
              "persisted_events_after_restart": count, "restart_manifest_equal": restart_equal,
              "rollback_verified": rolled["manifest_id"] == first["manifest_id"],
              "limitation": "Verifier and candidate share one process; production isolation is not implemented."}
    (a.output / "contract_demo.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"report": str(a.output / "contract_demo.json"),
                      "cycles": len(cycles), "restart": restart_equal,
                      "harmful_candidate": bad["decision"],
                      "model_learning_evaluated": False}, indent=2))


if __name__ == "__main__":
    main()

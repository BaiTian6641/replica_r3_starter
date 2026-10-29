#!/usr/bin/env python3
"""Materialize the independent, owned synthetic persistent-learner v2 fixture."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import os
import tempfile

FORMAT = "replica.spark_persistent_learner.fixture.v2"
PROTOCOL_ID = "spark_persistent_learner_v2"
SOURCE = "owned_synthetic_text_rules_v2_independent_worlds_with_exact_verifiable_outcomes"
SEED = 20261028


def _json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _record(record_id: str, text: str, world_id: str, role: str, epoch: int,
            *, supersedes: str | None = None) -> dict:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return {
        "available_at": epoch,
        "durable_ref": {"locator": f"data/spark_persistent_learner_v2/observations.jsonl#{record_id}",
                        "text_sha256": digest},
        "provenance": {"fixture_format": FORMAT, "record_id": record_id, "role": role,
                       "source": SOURCE, "text_sha256": digest, "world_id": world_id},
        "record_id": record_id, "role": role, "source_epoch": epoch,
        "supersedes": supersedes, "text": text, "text_sha256": digest,
        "valid_from": epoch, "valid_until": None, "world_id": world_id,
    }


def build() -> dict[str, object]:
    observations = [
        _record("fit-zeta-open", "For the zeta beacon, two short pulses followed by a steady light means open.", "fit_zeta", "fit", 1),
        _record("fit-eta-rule", "For eta requests, an amber tag means archive; every other visible tag means review.", "fit_eta", "fit", 2),
        _record("fit-zeta-correction", "For the zeta beacon, the same two short pulses followed by a steady light now mean closed after the maintenance revision.", "fit_zeta", "fit", 3, supersedes="fit-zeta-open"),
        _record("fit-zeta-legacy", "For a legacy zeta beacon, one short pulse means standby.", "fit_zeta", "fit", 4),
        _record("dev-theta-rule", "For the theta valve, one long pulse followed by two short pulses means ready.", "dev_theta", "development", 5),
        _record("dev-kappa-rule", "For kappa requests, a red tag means hold.", "dev_kappa", "development", 6),
        _record("dev-kappa-revocation", "The kappa red-tag rule is revoked; do not infer hold from an absent tag.", "dev_kappa", "development", 7, supersedes="dev-kappa-rule"),
    ]
    filler = [
        "The south warehouse lamp is green during normal operation.",
        "The west corridor is inspected every Tuesday morning.",
        "The archive trolley is returned to the lower shelf after use.",
        "The review desk is cleared before the weekly briefing.",
        "The maintenance notebook records the revision date in its header.",
        "The steady light on the auxiliary lamp indicates normal power.",
    ]
    observations.extend(_record(f"dev-filler2-{i}", text, "dev_filler2", "development", 8)
                        for i, text in enumerate(filler, 1))
    fit_events = [
        {"answer": "open", "event_id": "fit-01", "expected": "open", "kind": "new_rule",
         "observation_record_ids": ["fit-zeta-open"], "procedure_id": None,
         "question": "A zeta beacon emits two short pulses and then a steady light. What is the gate state?",
         "sequence": 1, "task_id": None, "world_id": "fit_zeta"},
        {"answer": "review", "event_id": "fit-02", "expected": "review", "kind": "similar_looking_different_procedure",
         "observation_record_ids": ["fit-eta-rule"], "procedure_id": "eta_procedure",
         "question": "An eta request has a green tag. Which queue receives it?", "sequence": 2,
         "task_id": "fit-eta-task", "world_id": "fit_eta"},
        {"answer": "closed", "event_id": "fit-03", "expected": "closed", "kind": "correction",
         "observation_record_ids": ["fit-zeta-correction"], "procedure_id": None,
         "question": "After the maintenance revision, a zeta beacon emits two short pulses and then a steady light. What is the gate state?",
         "sequence": 3, "task_id": None, "world_id": "fit_zeta"},
        {"answer": "standby", "event_id": "fit-04", "expected": "standby", "kind": "older_rule_retention",
         "observation_record_ids": ["fit-zeta-legacy"], "procedure_id": "zeta_procedure",
         "question": "For a legacy zeta beacon, one short pulse is observed. What state is that?",
         "sequence": 4, "task_id": None, "world_id": "fit_zeta"},
    ]
    cases = [
        ("dev-01", "ready", "fresh_world_new_instance", "theta_procedure", "The theta valve sends one long pulse and then two short pulses. What is its state?", ["new_information", "new_procedure_instances"], ["dev-theta-rule"], "dev_theta", False),
        ("dev-02", "ready", "fresh_world_paraphrase", "theta_procedure", "A theta valve shows a long pulse followed by two brief pulses. What state is it in?", ["new_procedure_instances"], ["dev-theta-rule"], "dev_theta", False),
        ("dev-03", "unknown", "similar_looking_interference", "theta_procedure", "The theta valve sends two short pulses followed by one long pulse. What is its state?", ["interference", "unknowns"], ["dev-theta-rule"], "dev_theta", False),
        ("dev-04", "unknown", "unknown_absent_tag", "kappa_procedure", "A kappa request has no visible tag. What should happen?", ["unknowns"], ["dev-kappa-revocation"], "dev_kappa", False),
        ("dev-05", "unknown", "revocation", "kappa_procedure", "A kappa request has a red tag. What should happen?", ["revocations"], ["dev-kappa-revocation"], "dev_kappa", False),
        ("dev-06", "unknown", "revocation_paraphrase", "kappa_procedure", "A kappa request shows no tag at all. What is the outcome?", ["revocations", "unknowns"], ["dev-kappa-revocation"], "dev_kappa", False),
        ("dev-07", "closed", "correction", "zeta_procedure", "A zeta beacon emits two short pulses followed by a steady light. What is the gate state?", ["corrections"], ["fit-zeta-correction"], "fit_zeta", False),
        ("dev-08", "closed", "correction_paraphrase", "zeta_procedure", "After the maintenance revision, a zeta beacon shows two short pulses then a steady light. What is the gate state?", ["corrections"], ["fit-zeta-correction"], "fit_zeta", False),
        ("dev-09", "standby", "older_rule_retention", "zeta_procedure", "For a legacy zeta beacon, one short pulse is observed. What state is that?", ["delayed_recall", "restart"], ["fit-zeta-legacy"], "fit_zeta", True),
        ("dev-10", "review", "similar_looking_different_procedure", "eta_procedure", "An eta request has a green tag. Which queue receives it?", ["new_procedure_instances"], ["fit-eta-rule"], "fit_eta", False),
        ("dev-11", "archive", "exception", "eta_procedure", "An eta request has an amber tag. Which queue receives it?", ["exceptions"], ["fit-eta-rule"], "fit_eta", False),
        ("dev-12", "review", "delayed_recall_capacity_pressure", "eta_procedure", "An eta request carries a yellow tag. Which queue receives it?", ["delayed_recall", "capacity_pressure"], ["fit-eta-rule"], "fit_eta", False),
    ]
    development = []
    labels = []
    for case_id, expected, kind, procedure, question, streams, support, world, retention in cases:
        development.append({"case_id": case_id, "kind": kind, "older_rule_retention": retention,
                            "procedure_id": procedure, "query_time": 10, "question": question,
                            "streams": streams, "support_record_ids": support, "world_id": world})
        labels.append({"case_id": case_id, "expected": expected})
    procedures = {"format": "replica.spark_persistent_learner.procedures.v2", "procedures": [
        {"applicability": ["zeta beacon pulse sequence"], "exceptions": ["maintenance revision changes the mapping"],
         "failure_lesson": "do not reuse the pre-maintenance mapping after the correction",
         "outcome": "verify the observed gate state against the current zeta rule", "outcome_checker": "fixture-rule-table",
         "outcome_independently_checked": True, "prerequisites": ["beacon is visible", "pulse sequence ends"],
         "procedure_id": "zeta_procedure", "source_record_ids": ["fit-zeta-correction", "fit-zeta-legacy"], "world_id": "fit_zeta"},
        {"applicability": ["eta request routing"], "exceptions": ["amber tag routes archive"],
         "failure_lesson": "an absent tag cannot be routed as an amber exception", "outcome": "route the request to the queue named by the tag rule",
         "outcome_checker": "fixture-rule-table", "outcome_independently_checked": True,
         "prerequisites": ["tag is visible", "request is eta"], "procedure_id": "eta_procedure",
         "source_record_ids": ["fit-eta-rule"], "world_id": "fit_eta"},
        {"applicability": ["theta valve pulse sequence"], "exceptions": ["pattern order differs"],
         "failure_lesson": "two short pulses before one long pulse are not the ready pattern",
         "outcome": "verify the theta valve state from the ordered pulses", "outcome_checker": "fixture-rule-table",
         "outcome_independently_checked": True, "prerequisites": ["pulse counts are ordered"],
         "procedure_id": "theta_procedure", "source_record_ids": ["dev-theta-rule"], "world_id": "dev_theta"},
        {"applicability": ["kappa request tag inspection"], "exceptions": ["red-tag rule revoked", "tag absent"],
         "failure_lesson": "a revoked rule cannot supply a held state", "outcome": "treat an absent or revoked tag as unknown",
         "outcome_checker": "fixture-rule-table", "outcome_independently_checked": True, "prerequisites": [],
         "procedure_id": "kappa_procedure", "source_record_ids": ["dev-kappa-revocation"], "world_id": "dev_kappa"},
    ]}
    tasks = {"format": "replica.spark_persistent_learner.tasks.v2", "tasks": [{
        "observed_completions": [{"checker": "fixture-rule-table", "evidence_record_ids": ["fit-eta-rule"],
                                   "observation": "the green tag routes to review", "observed_at": 2,
                                   "subgoal_id": "route_queue"}],
        "subgoals": [{"description": "inspect the eta tag", "subgoal_id": "inspect_tag"},
                     {"description": "route to the named queue", "subgoal_id": "route_queue"}],
        "task_id": "fit-eta-task", "world_id": "fit_eta"}]}
    worlds = [
        {"kind": "new_rule_correction_and_older_rule", "role": "fit", "rule_family": "zeta_beacon", "world_id": "fit_zeta"},
        {"kind": "similar_looking_different_procedure", "role": "fit", "rule_family": "eta_tag_routing", "world_id": "fit_eta"},
        {"kind": "fresh_world_new_rule", "role": "development", "rule_family": "theta_valve", "world_id": "dev_theta"},
        {"kind": "unknown_and_revocation", "role": "development", "rule_family": "kappa_tag_revocation", "world_id": "dev_kappa"},
        {"kind": "unrelated_capacity_pressure_records", "role": "development", "rule_family": "capacity_pressure_filler_v2", "world_id": "dev_filler2"},
    ]
    parity = [
        {"kind": "null_no_memory", "memory_source": "none", "paired_request_id": None, "procedure_id": None,
         "question": "A zeta beacon emits two short pulses and then a steady light. What is the gate state?", "request_id": "parity-01"},
        {"kind": "trained_p_read", "memory_source": "P", "paired_request_id": None, "procedure_id": "zeta_procedure",
         "question": "For a legacy zeta beacon, one short pulse is observed. What state is that?", "request_id": "parity-02"},
        {"kind": "restart_p_only", "memory_source": "P", "paired_request_id": "parity-02", "procedure_id": "zeta_procedure",
         "question": "For a legacy zeta beacon, one short pulse is observed. What state is that?", "request_id": "restart-parity-01"},
    ]
    files = {
        "observations.jsonl": "".join(_json(row) + "\n" for row in observations),
        "fit_events.jsonl": "".join(_json(row) + "\n" for row in fit_events),
        "development.jsonl": "".join(_json(row) + "\n" for row in development),
        "development_labels.jsonl": "".join(_json(row) + "\n" for row in labels),
        "parity_requests.jsonl": "".join(_json(row) + "\n" for row in parity),
        "procedures.json": json.dumps(procedures, indent=2, sort_keys=True) + "\n",
        "tasks.json": json.dumps(tasks, indent=2, sort_keys=True) + "\n",
        "worlds.json": json.dumps({"format": "replica.spark_persistent_learner.worlds.v2", "worlds": worlds}, indent=2, sort_keys=True) + "\n",
    }
    manifest = {
        "counts": {"arms": 11, "development_cases": len(development), "development_labels": len(labels),
                   "fit_events": len(fit_events), "observations": len(observations), "parity_requests": 2,
                   "procedures": len(procedures["procedures"]), "restart_parity_requests": 1, "task_workspaces": 1},
        "created_utc": "2026-10-28T00:00:00+00:00", "development_labels_closed_until_fit_receipt": True,
        "files": {name: {"bytes": len(content.encode("utf-8")), "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest()}
                  for name, content in sorted(files.items())},
        "format": FORMAT, "license": {"status": "owned_synthetic_project_generated", "external_text": False,
                                       "external_sources": [], "approval_required_for_identity_records": True},
        "protocol_id": PROTOCOL_ID, "required_streams": ["new_information", "new_procedure_instances", "corrections", "exceptions", "revocations", "unknowns", "similar_looking_different_procedures", "delayed_recall", "interference", "restart", "capacity_pressure"],
        "roles": {"development": "fresh_world_paraphrase_new_instance_and_retention_selection", "fit": "chronological_teaching_and_support_only", "sealed_final": "closed_not_materialized_not_read"},
        "sealed_final": {"materialized": False, "opened": False, "path": None}, "seed": SEED,
        "source_lineage": SOURCE, "split_unit": "independent_world_and_rule_family_before_paraphrase_generation",
        "version": 2, "worlds": worlds,
    }
    return {"files": files, "manifest": manifest}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/spark_persistent_learner_v2"))
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing fixture: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = build()
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        for name, content in payload["files"].items():
            (temporary / name).write_text(content, encoding="utf-8")
        (temporary / "manifest.json").write_text(json.dumps(payload["manifest"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, output)
    except Exception:
        # Preserve a failed publication for inspection; never silently replace a prior fixture.
        raise
    print(json.dumps({"output": str(output), "manifest_sha256": hashlib.sha256((output / "manifest.json").read_bytes()).hexdigest(),
                      "counts": payload["manifest"]["counts"], "sealed_final_opened": False}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

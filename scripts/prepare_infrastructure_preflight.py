#!/usr/bin/env python3
"""Audit the public synthetic release and lock a small original-model dev probe.

Tokenization uses the local source only. No model is loaded, no role is trained,
and the inspected calibration/promotion/audit fixtures are explicitly exposed.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from make_infrastructure_slice import validate_release
from train_text import encode_row
from replica_lab.contracts import canonical_json
from replica_lab.episodes import export_sft, input_view, verify_episode
from replica_lab.infrastructure_scoring import score_completion
from replica_lab.infrastructure_solver import FAMILIES, independent_check, solve_episode, solve_visible


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def relative(path):
    return str(Path(path).resolve().relative_to(ROOT))


def load_release(root):
    manifest = validate_release(root)
    sources = {record["source_id"]: record for record in json.loads((root / "source_manifest.json").read_text())}
    payloads = []
    for role in ("train", "dev", "calibration", "promotion", "audit"):
        folder = "canonical" if role in {"train", "dev"} else "evaluator_only"
        payloads.extend(lines(root / folder / f"{role}_episodes.jsonl"))
    return manifest, sources, payloads


def select_cases(payloads, sources):
    """Freeze one query from each of four distinct worlds/family, before inference."""
    selected = []
    for family in FAMILIES:
        for scenario in ("late_correction", "late_observation_ask", "late_observation_unknown", "tombstone"):
            eligible = [p for p in payloads if p["split"] == "dev"
                        and sources[p["sources"][0]["source_id"]]["family"] == family
                        and sources[p["sources"][0]["source_id"]]["scenario"] == scenario]
            worlds = sorted({p["group_id"] for p in eligible})
            if not worlds:
                raise ValueError("missing fixed development stratum")
            queries = sorted((p for p in eligible if p["group_id"] == worlds[0]), key=lambda p: p["query"]["at"])
            if len(queries) != 2:
                raise ValueError("expected exactly two queries/world")
            selected.append(queries[1 if scenario in {"late_correction", "late_observation_ask"} else 0])
    if len(selected) != 16 or len({p["group_id"] for p in selected}) != 16:
        raise ValueError("preflight must contain 16 distinct development worlds")
    if Counter(p["target"]["action"] for p in selected) != {"answer": 8, "ask": 4, "unknown": 4}:
        raise ValueError("preflight action balance differs from declaration")
    return selected


def summarize(values):
    values = sorted(values)
    if not values:
        raise ValueError("empty accounting category")
    return {"count": len(values), "sum": sum(values), "min": values[0],
            "mean": sum(values) / len(values), "p50": values[(len(values)-1)//2],
            "p95": values[int((len(values)-1)*.95)], "max": values[-1]}


def token_lengths(row, tokenizer):
    # Measurement allows a larger bound; admission below still rejects >512.
    batch = encode_row(row, tokenizer, 16384)
    full, prompt, support = (batch[key].numel() for key in ("input_ids", "prompt_ids", "memory_ids"))
    return {"native_full": full, "native_prompt": prompt, "source_writer": support,
            "supervised_answer_plus_eos": full-prompt, "answer": full-prompt-1, "eos": 1}


def controls(dev, sources, root):
    """Declarative semantic controls, not additional model generations."""
    expected = {p["id"]: solve_episode(p, root, sources[p["sources"][0]["source_id"]]) for p in dev}
    constants = sorted({x["response"] for x in expected.values()} | {"0", "UNKNOWN", "STOP"})
    scores = []
    for response in constants:
        correct = sum(score_completion(response, expected[p["id"]],
                     family=sources[p["sources"][0]["source_id"]]["family"])["task_correct"] for p in dev)
        scores.append({"response": response, "correct": correct, "total": len(dev), "rate": correct/len(dev)})
    missing = []
    for p in dev:
        family = sources[p["sources"][0]["source_id"]]["family"]
        visible = input_view(p, purpose="local_evaluation")["memory_texts"]
        semantic = [text.split(" | ")[-1] for text in visible]
        rule_only = [text for text in semantic if text.startswith("RULE ")]
        gold = solve_visible(family, p["query"]["prompt"], rule_only)
        missing.append({"id": p["id"], "family": family, "removed_fact_count": len(semantic)-len(rule_only),
                        "expected": gold, "original_expected": expected[p["id"]],
                        "answer_to_abstention": expected[p["id"]]["action"] == "answer" and gold["action"] != "answer"})
    return {"schema_version": "replica.infrastructure-controls.v2", "role": "dev", "examples": len(dev),
            "kind": "CPU semantic evaluator controls; no native generation", "constant_scores": scores,
            "best_constant": max(scores, key=lambda row: row["correct"]),
            "constant_perfect_fit_rejected": all(row["correct"] < len(dev) for row in scores),
            "missing_fact_controls": missing,
            "missing_fact_scope": "Unauthenticated semantic view only, no altered raw-source receipt; not a measured model evidence-sensitivity result.",
            "shuffled_evidence_model_condition": "pending", "retention_and_calibration": "pending"}


def prepare(release, output, tokenizer, model_path, model_lock):
    if output.exists():
        raise ValueError("use a fresh preflight directory")
    manifest, sources, payloads = load_release(release)
    selected = select_cases(payloads, sources)
    selected_ids = {p["id"] for p in selected}
    solver_hash = sha(ROOT / "src/replica_lab/infrastructure_solver.py")
    rows, case_records, length_records = {}, {}, []
    review_groups = defaultdict(list)
    for p in payloads:
        source = sources[p["sources"][0]["source_id"]]
        truth = solve_episode(p, release, source)
        receipt = verify_episode(p, verifier_id="replica.infrastructure.raw-source-solver.v2", verifier_sha256=solver_hash,
                                 verifier=lambda value, s=source: independent_check(value, release, s))
        if not receipt.passed:
            raise ValueError("independent raw-source rejection")
        row = export_sft(p, purpose="local_evaluation", verification=receipt)
        lengths = token_lengths(row, tokenizer)
        length_records.append({"id": p["id"], "role": p["split"], "family": source["family"], **lengths})
        review_groups[(source["family"], source["scenario"], p["split"])].append(p["id"])
        if p["id"] in selected_ids:
            rows[p["id"]] = row
            case_records[p["id"]] = {"id": p["id"], "world_id": p["group_id"], "family": source["family"],
                "scenario": source["scenario"], "expected_action": truth["action"], "expected_response": truth["response"]}
    overlong = [row["id"] for row in length_records if row["native_full"] > 512 or row["source_writer"] > 512]
    if overlong:
        raise ValueError(f"{len(overlong)} rows exceed declared512-token admission: {overlong[:8]}")
    review = []
    for family in FAMILIES:
        for scenario in ("late_correction", "late_observation_ask", "late_observation_unknown", "tombstone"):
            # Two worlds and both queries per role: 20 episodes/critical stratum.
            chosen = [i for role in ("train", "dev", "calibration", "promotion", "audit")
                      for i in sorted(review_groups[(family, scenario, role)])[:4]]
            if len(chosen) != 20:
                raise ValueError("insufficient operational review sample")
            review.append({"family": family, "scenario": scenario, "episode_ids": chosen,
                           "episodes": 20, "worlds": 10, "independent_raw_source_checks_passed": True})
    output.mkdir(parents=True)
    data_path = output / "dev_cases.jsonl"
    data_path.write_text("".join(canonical_json(rows[p["id"]])+"\n" for p in selected))
    metrics = ("native_full", "native_prompt", "source_writer", "supervised_answer_plus_eos", "answer", "eos")
    summaries = {"overall": {m: summarize([r[m] for r in length_records]) for m in metrics}, "by_role": {}, "by_family": {}}
    for dimension in ("role", "family"):
        for value in sorted({r[dimension] for r in length_records}):
            summaries[f"by_{dimension}"][value] = {m: summarize([r[m] for r in length_records if r[dimension] == value]) for m in metrics}
    write(output / "token_accounting.json", {"schema_version": "replica.infrastructure-token-accounting.v2",
        "release_manifest_sha256": sha(release/"manifest.json"), "tokenizer_path": relative(model_path),
        "model_lock_sha256": sha(model_lock), "admission_max_length": 512, "truncated_examples": 0,
        "overlong_ids": overlong, "admitted": len(length_records), "summary": summaries, "examples": length_records,
        "scope": "Native original chat template, thinking disabled; support is a separate additional writer view, not added to native tokens. All roles inspected public synthetic fixtures."})
    write(output / "stratified_review.json", {"kind": "Automated independent raw-source semantic and chronology review, not a human annotation study",
        "all_target_checks": len(payloads), "all_passed": True, "critical_strata": review,
        "sample_episodes": sum(s["episodes"] for s in review), "sample_worlds": sum(s["worlds"] for s in review),
        "selection": "First two sorted world IDs and both queries, per role/family/scenario; within-world queries correlated",
        "scope": manifest["generalization_scope"], "exposure": manifest["audit_exposure"]})
    write(output / "constant_controls.json", controls([p for p in payloads if p["split"] == "dev"], sources, release))
    code = ["scripts/prepare_infrastructure_preflight.py", "scripts/make_infrastructure_slice.py",
            "src/replica_lab/infrastructure_solver.py", "src/replica_lab/infrastructure_scoring.py",
            "src/replica_lab/episodes.py", "src/replica_lab/contracts.py", "scripts/train_text.py",
            "scripts/evaluate_text.py", "scripts/run_generation_preflight.py", "configs/infrastructure_slice_v2_design.json",
            "src/replica_lab/model_adapter.py", "src/replica_lab/generation.py", "src/replica_lab/neural.py",
            "src/replica_lab/evaluation.py", "scripts/score_infrastructure_generations.py",
            "src/replica_lab/capacity_guard.py", "scripts/validate_d2_experiment.py"]
    spec = {"schema_version": "replica.infrastructure_preflight.v2", "declared_at_utc": datetime.now(timezone.utc).isoformat(),
        "release_path": relative(release), "data_path": relative(data_path), "data_sha256": sha(data_path),
        "release_manifest_sha256": sha(release/"manifest.json"), "source_manifest_sha256": sha(release/"source_manifest.json"),
        "selected_cases": [case_records[p["id"]] for p in selected], "max_new_tokens": 32, "max_length": 512,
        "seed": 17, "decoding": "greedy", "role": "dev", "device": "mps", "dtype": "float32",
        "cpu_fallback": False, "model_path": relative(model_path), "model_lock": relative(model_lock),
        "model_lock_sha256": sha(model_lock), "implementation_sha256": {name: sha(ROOT/name) for name in code},
        "scope": "Frozen original B1 evidence-matched descriptive dev preflight, 16 distinct worlds/4 families; all mechanisms shared, no training or promotion. Raw backend's legacy XOR descriptions superseded by release-aware rescoring.",
        "selection_rule": "First sorted dev world per family/scenario: laterquery for correction/lateask, earlierquery for lateunknown/tombstone; 8answer/4ask/4unknown."}
    write(output / "evaluation_spec.json", spec)
    return {"examples": len(payloads), "selected": len(selected), "max_full_tokens": summaries["overall"]["native_full"]["max"],
            "output": relative(output), "release_manifest_sha256": spec["release_manifest_sha256"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, default=ROOT/"checkpoints/minicpm5_source")
    parser.add_argument("--model-lock", type=Path, default=ROOT/"configs/minicpm5_sealed_lock.local.json")
    args = parser.parse_args()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True, trust_remote_code=False)
    print(json.dumps(prepare(args.release, args.output, tokenizer, args.model_path, args.model_lock)))


if __name__ == "__main__":
    main()

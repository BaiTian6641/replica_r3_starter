#!/usr/bin/env python3
"""Reconcile known-known/known-unknown pair evidence from saved V13/V14 artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from replica_lab.intrinsic_memory_budget_v12 import atomic

CONTROLS = ("contextual", "row_permutation", "shuffled", "wrong", "zero", "absent", "exact_text")
V14_CONTROLS = ("contextual", "shuffled", "wrong", "zero", "absent", "exact_text")


def load(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v13-run", required=True, type=Path)
    parser.add_argument("--v14-run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("pair analysis receipt must be fresh")
    v13_root = args.v13_run
    v14_root = args.v14_run
    pairs = []
    for index in range(16):
        v13 = load(v13_root / "pairs" / f"pair_{index:02d}.json")
        v14 = load(v14_root / "fit_probes_512.json")
        row_a, row_b = v14["rows"][index * 2:index * 2 + 2]
        expected = v13["expected_answers"]
        category = "known_known" if all(answer != "unknown" for answer in expected) else "known_unknown"
        v13_sides = []
        for side in (0, 1):
            controls = {}
            for control in CONTROLS:
                output = v13["control_outputs"][side][v13["controls"].index(control)]
                controls[control] = {
                    "raw_answer": output["raw_answer"], "correct": output["correct"],
                    "terminated": output["terminated"], "truncated": output["truncated"],
                    "expected_minus_other_logit": output["expected_minus_other_logit"],
                    "first_token_correct": output["first_token_correct"],
                }
            v13_sides.append({"world_id": v13["world_ids"][side], "expected_answer": expected[side],
                              "controls": controls})
        v14_sides = []
        for row in (row_a, row_b):
            controls = {}
            for control in V14_CONTROLS:
                output = row["controls"][control]
                controls[control] = {
                    "raw_answer": output["raw_answer"], "correct": output["correct"],
                    "terminated": output["terminated"], "truncated": output["truncated"],
                    "first_token_correct": output["first_token_correct"],
                }
            v14_sides.append({"world_id": row["world_id"],
                              "expected_answer": expected[len(v14_sides)], "controls": controls,
                              "round_trip_exact": row["round_trip_exact"]})
        v14_contextual = [side["controls"]["contextual"]["raw_answer"] for side in v14_sides]
        pairs.append({
            "pair_index": index, "category": category, "expected_answers": expected,
            "both_correct_v13": v13["both_contextual_correct"],
            "both_correct_v14": all(side["controls"]["contextual"]["correct"] for side in v14_sides),
            "identical_answers_v14": v14_contextual[0] == v14_contextual[1],
            "v13_sides": v13_sides, "v14_sides": v14_sides,
        })
    summary = {}
    for category in ("known_known", "known_unknown"):
        selected = [pair for pair in pairs if pair["category"] == category]
        summary[category] = {
            "pairs": len(selected),
            "both_correct_v13": sum(pair["both_correct_v13"] for pair in selected),
            "both_correct_v14": sum(pair["both_correct_v14"] for pair in selected),
            "identical_answer_failures_v14": sum(
                not pair["both_correct_v14"] and pair["identical_answers_v14"] for pair in selected),
        }
    v14_probes = load(v14_root / "fit_probes_512.json")
    for control in V14_CONTROLS:
        observed = sum(row["controls"][control]["correct"] for row in v14_probes["rows"])
        if observed != v14_probes["correct"][control]:
            raise ValueError("V14 control total differs from raw rows")
    result = {
        "format": "replica.intrinsic_memory.pair_category_reconciliation.v15",
        "v13_run": str(v13_root), "v14_run": str(v14_root),
        "v13_packet_sha256": "1f8fa0100c4243ff3bcf11ce05ed5e9cc95c9d9f4be74dfd2cb05a1a85a78b07",
        "v14_packet_sha256": "074bee9bdc0b8f77f892ac491b3465e4fcb56c022bf7a5cca17af381751af7ff",
        "summary": summary,
        "pairs": pairs,
        "margin_evidence": {
            "v13": "Signed expected-versus-counterfactual answer-token logit margins for every control.",
            "v14": "Raw answers and first-token correctness preserved; full answer-token margins were not preserved and are not inferred.",
            "multi_token_answers": "Use complete per-step sequence likelihood, not a first-token or max-vocabulary delta. Current color answers are single-token.",
        },
        "development_labels_used": False,
        "interpretation": "Known-known and known-unknown categories are computed from expected answers, not from the review snapshot or generated answers.",
    }
    atomic(args.output, result)
    print(json.dumps({"summary": summary, "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())

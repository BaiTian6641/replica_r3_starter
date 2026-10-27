#!/usr/bin/env python3
"""Replay preserved NL1 dev04 outputs through scoring v2 (no model execution).

This produces a corrected measurement of existing raw outputs.  It does not
rewrite dev03/dev04, infer missing stop metadata, or open sealed-final data.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from replica_lab.nested_text_scoring import (ANSWER_GRAMMAR_VERSION,
                                             SCORING_VERSION, score_response)

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "runs/nested_text_nl1_20261026/p_only_dev04/gates/dev.json"

REPLAY_SCORING_VERSION = "replica.nested_text.scoring.v2.1.1"
REPLAY_FORMAT = "replica.nested_text.scoring_replay.v2.1"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    source = json.loads(INPUT.read_text())
    groups = []
    for name, score in source["scores"].items():
        groups.append((f"taught:{name}", score))
    for name, score in source["preteach_scores"].items():
        groups.append((f"preteach:{name}", score))
    for name in ("shuffled_route", "reset_P", "spine_absent",
                 "wrong_world_P", "wrong_world_on_primary"):
        groups.append((f"control:{name}", source["controls"][name]))
    candidate_answers = sorted({
        row["expected"]
        for score in source["scores"].values() for row in score["results"]
    })
    summaries = []
    replay_rows = []
    for label, score in groups:
        rows = []
        reasons = Counter()
        for item in score["results"]:
            output = item.get("output", "")
            scored = score_response(
                item["expected"], output, candidate_answers=candidate_answers,
                truncated=False)
            reasons[scored.reason] += 1
            row = {
                "id": item["id"], "label": label,
                "expected": item["expected"], "output": output,
                "old_exact_correct": bool(item["correct"]),
                "correct": scored.correct, "reason": scored.reason,
                "prompt_sha256": item.get("prompt_sha256"),
            }
            rows.append(row)
            replay_rows.append(row)
        correct = sum(row["correct"] for row in rows)
        summaries.append({
            "label": label, "count": len(rows), "correct": correct,
            "old_exact_correct": score["correct"],
            "accuracy": correct / len(rows) if rows else 0.0,
            "reason_counts": dict(sorted(reasons.items())),
            "results": rows,
        })
    by_label = {entry["label"]: entry for entry in summaries}
    preteach = by_label["preteach:legacy_retention"]
    taught = by_label["taught:legacy_retention"]
    legacy = {
        "preteach_correct": preteach["correct"],
        "preteach_count": preteach["count"],
        "preteach_accuracy": preteach["accuracy"],
        "taught_correct": taught["correct"],
        "taught_count": taught["count"],
        "taught_accuracy": taught["accuracy"],
        "old_exact_preteach_correct": preteach["old_exact_correct"],
        "old_exact_taught_correct": taught["old_exact_correct"],
        "retention_decline": max(0.0, preteach["accuracy"] - taught["accuracy"]),
        "scope": "Eight legacy rows under derived scoring v2.1.1; no model run and no new training result",
    }
    result = {
        "format": REPLAY_FORMAT,
        "scoring_version": REPLAY_SCORING_VERSION,
        "scorer_contract_version": SCORING_VERSION,
        "answer_grammar_version": ANSWER_GRAMMAR_VERSION,
        "historical_source_scoring_version": "replica.nested_text.scoring.v2",
        "input": str(INPUT.relative_to(ROOT)),
        "input_sha256": hashlib.sha256(INPUT.read_bytes()).hexdigest(),
        "history_rewrite": False,
        "native_model_executed": False,
        "sealed_final_read": False,
        "raw_stop_metadata_available": False,
        "truncation_policy": "No global cap-hit inference; visibly incomplete text is rejected, while native v2 runs additionally record actual stop reasons",
        "groups": summaries,
        "legacy_retention": legacy,
        "claims": {
            "corrected_measurement_of_preserved_outputs": True,
            "new_learned_capability": False,
            "heldout_evaluation": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({
        "output": str(args.output), "scoring_attempts": len(replay_rows),
        "scoring_version": REPLAY_SCORING_VERSION,
        "answer_grammar_version": ANSWER_GRAMMAR_VERSION,
        "retention_decline": legacy["retention_decline"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

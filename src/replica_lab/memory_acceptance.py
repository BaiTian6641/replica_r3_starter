"""Versioned acceptance-aware memory metrics; raw rankings remain diagnostics."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from .vision_curriculum import clustered_bootstrap

VERSION = "memory.acceptance.v2"
RECEIPT_FORMAT = "replica.memory_acceptance.metrics.v2"

VALUE_ARMS = frozenset({"exact_key_value", "query_key_value"})


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _rate(numerator, denominator):
    return numerator / denominator if denominator else None


def _classify(row):
    norm = row.get("readout_norm")
    _require(isinstance(norm, (int, float)) and norm >= 0,
             "Readout norm must be nonnegative")
    zero = norm == 0
    top = row.get("top_cosine")
    second = row.get("second_cosine")
    tied = second is not None and top == second
    usable = not zero and not tied
    accepted = bool(row.get("accepted")) and usable
    top1 = bool(row.get("top1"))
    return {
        "raw_top1": top1,
        "accepted": accepted,
        "accepted_correct": accepted and top1,
        "accepted_wrong": accepted and not top1,
        "zero_or_tied": zero or tied,
        "zero_or_tied_top1": (zero or tied) and top1,
        "kind": row["kind"],
        "target_role": row.get("target_role"),
        "variant": row["variant"],
        "episode_id": row["episode_id"],
        "world_id": row["world_id"],
        "state": row["state"],
        "case_id": row["case_id"],
        "arm": row["arm"],
        "layer": row["layer"],
    }


def _query_row(rows):
    accepted_correct = all(row["accepted_correct"] for row in rows)
    accepted = any(row["accepted"] for row in rows)
    first = rows[0]
    return {
        "raw_top1": any(row["raw_top1"] for row in rows),
        "raw_top1_all": all(row["raw_top1"] for row in rows),
        "accepted": accepted,
        "accepted_correct": accepted_correct,
        "accepted_wrong": accepted and not accepted_correct,
        "zero_or_tied_top1": any(row["zero_or_tied_top1"] for row in rows),
        "kind": first["kind"],
        "target_role": first["target_role"],
        "variant": first["variant"],
        "episode_id": first["episode_id"],
        "world_id": first["world_id"],
        "state": first["state"],
        "case_id": first["case_id"],
        "arm": first["arm"],
        "layers": tuple(sorted(row["layer"] for row in rows)),
    }


def _block(rows):
    rows = list(rows)
    n = len(rows)
    accepted = sum(row["accepted"] for row in rows)
    correct = sum(row["accepted_correct"] for row in rows)
    unknown = [row for row in rows if row["kind"] == "unknown"]
    unknown_false = sum(row["accepted_wrong"] for row in unknown)
    fact_n = sum(row["kind"] == "fact" for row in rows)
    return {
        "n": n,
        "fact_n": fact_n,
        "raw_top1": sum(row["raw_top1"] for row in rows),
        "raw_top1_rate": _rate(sum(row["raw_top1"] for row in rows), n),
        "raw_top1_all": sum(row.get("raw_top1_all", row["raw_top1"]) for row in rows),
        "raw_top1_all_rate": _rate(
            sum(row.get("raw_top1_all", row["raw_top1"]) for row in rows), n),
        "accepted_correct": correct,
        "accepted_correct_rate": _rate(correct, fact_n),
        "accepted": accepted,
        "accepted_precision": _rate(correct, accepted),
        "accepted_wrong": sum(row["accepted_wrong"] for row in rows),
        "acceptance_coverage": _rate(accepted, n),
        "abstention_rate": _rate(n - accepted, n),
        "zero_or_tied_top1": sum(row["zero_or_tied_top1"] for row in rows),
        "unknown_n": len(unknown),
        "unknown_false_recall": unknown_false,
        "unknown_false_recall_rate": _rate(unknown_false, len(unknown)),
    }


def _groups(rows, key):
    grouped = {}
    for row in rows:
        grouped.setdefault(key(row), []).append(row)
    return grouped


def _uncertainty(query_rows, *, seed, resamples, confidence):
    by_world = _groups(query_rows, lambda row: row["world_id"])
    result = {}
    values = {}
    for metric in ("accepted_correct_rate", "accepted_precision", "acceptance_coverage"):
        for world, rows in by_world.items():
            block = _block(rows)
            value = block[metric]
            if value is not None:
                values[world] = [value]
        if values:
            result[metric] = clustered_bootstrap(
                values, resamples=resamples, confidence=confidence, seed=seed)
        else:
            result[metric] = None
    result["unit"] = "world"
    result["layer_readouts_are_not_independent"] = True
    return result


def acceptance_metrics(rankings, *, bootstrap_seed=17, resamples=2000,
                       confidence=0.95):
    """Compute readout, query, role, layer, episode and world metrics."""
    rows = [_classify(row) for row in rankings if row["arm"] in VALUE_ARMS]
    keys = [(row["variant"], row["episode_id"], row["state"], row["case_id"],
             row["arm"], row["layer"]) for row in rows]
    _require(len(keys) == len(set(keys)), "Duplicate ranking row identity")
    query_rows = [_query_row(rows) for rows in _groups(
        rows, lambda row: (row["variant"], row["episode_id"], row["state"],
                           row["case_id"], row["arm"])).values()]
    result = {}
    for variant in sorted({row["variant"] for row in rows}):
        layer_rows = [row for row in rows if row["variant"] == variant]
        queries = [row for row in query_rows if row["variant"] == variant]
        result[variant] = {
            "readout": _block(layer_rows),
            "query": _block(queries),
            "correction": _block(row for row in queries
                                 if row["target_role"] == "correction_target"),
            "unaffected": _block(row for row in queries
                                 if row["target_role"] == "retained"),
            "unknown": _block(row for row in queries if row["kind"] == "unknown"),
            "by_layer": {
                str(layer): _block(row for row in layer_rows if row["layer"] == layer)
                for layer in sorted({row["layer"] for row in layer_rows})},
            "by_episode": {
                episode: _block(row for row in queries if row["episode_id"] == episode)
                for episode in sorted({row["episode_id"] for row in queries})},
            "by_world": {
                world: _block(row for row in queries if row["world_id"] == world)
                for world in sorted({row["world_id"] for row in queries})},
            "counts": {
                "layer_readouts": len(layer_rows),
                "queries": len(queries),
                "episodes": len({row["episode_id"] for row in queries}),
                "independent_worlds": len({row["world_id"] for row in queries}),
            },
            "uncertainty": _uncertainty(
                queries, seed=bootstrap_seed, resamples=resamples,
                confidence=confidence),
        }
    return result


def evaluate_acceptance(metrics, config):
    """Apply only the explicitly versioned acceptance gates to a candidate."""
    _require(config.get("metric_version") == VERSION,
             "Acceptance consumer must declare memory.acceptance.v2")
    candidate = config["candidate_variant"]
    oracle = config["oracle_variant"]
    _require(candidate in metrics and oracle in metrics,
             "Candidate or oracle metrics are missing")
    query = metrics[candidate]["query"]
    correction = metrics[candidate]["correction"]
    unaffected = metrics[candidate]["unaffected"]
    unknown = metrics[candidate]["unknown"]
    thresholds = config["gates"]
    for name, block in (("candidate", query), ("correction", correction),
                        ("unaffected", unaffected), ("unknown", unknown),
                        ("oracle", metrics[oracle]["query"])):
        _require(block["n"] > 0 and (name not in {"candidate", "oracle"} or block["fact_n"] > 0),
                 f"Acceptance coverage is missing: {name}")
    checks = {
        "accepted_correct_rate": (query["accepted_correct_rate"]
                                  >= thresholds["accepted_correct_rate_min"]),
        "accepted_precision": (query["accepted_precision"] is not None
                               and query["accepted_precision"]
                               >= thresholds["accepted_precision_min"]),
        "correction": (correction["accepted_correct_rate"]
                       >= thresholds["correction_accepted_correct_rate_min"]),
        "unaffected": (unaffected["accepted_correct_rate"]
                       >= thresholds["unaffected_accepted_correct_rate_min"]),
        "unknown_false_recall": (unknown["unknown_false_recall_rate"]
                                 <= thresholds["unknown_false_recall_max"]),
        "oracle_raw_bound": (metrics[oracle]["query"]["raw_top1_rate"]
                             >= thresholds["oracle_raw_top1_rate_min"]),
    }
    return {
        "decision": "accepted" if all(checks.values()) else "rejected",
        "checks": checks,
        "accepted_correct_rate_min": thresholds["accepted_correct_rate_min"],
        "accepted_precision_min": thresholds["accepted_precision_min"],
        "correction_accepted_correct_rate_min": thresholds["correction_accepted_correct_rate_min"],
        "unaffected_accepted_correct_rate_min": thresholds["unaffected_accepted_correct_rate_min"],
        "unknown_false_recall_max": thresholds["unknown_false_recall_max"],
        "oracle_raw_top1_rate_min": thresholds["oracle_raw_top1_rate_min"],
        "acceptance_coverage_policy": "report_only",
    }


def rescore_ranking_file(path, config, *, generation_path=None):
    """Rescore immutable rankings into a new versioned receipt."""
    path = Path(path)
    original = path.read_bytes()
    rankings = [json.loads(line) for line in original.decode().splitlines() if line]
    _require(rankings, "Ranking artifact is empty")
    metrics = acceptance_metrics(
        rankings, bootstrap_seed=config["bootstrap"]["seed"],
        resamples=config["bootstrap"]["resamples"],
        confidence=config["bootstrap"]["confidence"])
    verdict = evaluate_acceptance(metrics, config)
    generation = {"status": "not_measured", "accuracy": None}
    if generation_path is not None:
        generation_path = Path(generation_path)
        answers = [json.loads(line) for line in
                   generation_path.read_text().splitlines() if line]
        by_arm = {}
        for arm in sorted({row["arm"] for row in answers}):
            rows = [row for row in answers if row["arm"] == arm]
            by_arm[arm] = {"n": len(rows), "correct": sum(row["correct"] for row in rows),
                           "accuracy": _rate(sum(row["correct"] for row in rows), len(rows))}
        generation = {"status": "historical_reference_only",
                      "source_sha256": _sha(generation_path), "by_arm": by_arm,
                      "accuracy": None,
                      "note": "Generation is not vector-readout evidence and is excluded from acceptance gates"}
    receipt = {
        "format": RECEIPT_FORMAT,
        "metric_version": VERSION,
        "source": {"path": str(path), "sha256": _sha(path), "rows": len(rankings),
                   "original_bytes_unchanged": original == path.read_bytes()},
        "candidate_variant": config["candidate_variant"],
        "oracle_variant": config["oracle_variant"],
        "metrics": metrics,
        "verdict": verdict,
        "generation": generation,
        "claims": {
            "raw_rankings_are_not_accepted_retrievals": True,
            "zero_and_ambiguous_tied_readouts_receive_no_credit": True,
            "layers_are_not_independent_for_uncertainty": True,
            "hybrid_retrieval_and_learned_memory_are_separate": True,
            "generated_answer_claims_are_separate": True,
        },
    }
    return receipt

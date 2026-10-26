#!/usr/bin/env python3
"""Offline data/exposure/power arithmetic; no training, media or network access."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "configs/convincing_result_data_budget_v1.json"


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_inputs(folder):
    manifest = read(folder / "input_manifest.json")
    for item in manifest["inputs"]:
        path = ROOT / item["path"]
        if path.stat().st_size != item["bytes"] or sha(path) != item["sha256"]:
            raise ValueError(f"Review input changed: {item['path']}")
    return len(manifest["inputs"])


def episode_work(e):
    if e["teaching_events"] < 2 * e["slow_clock_events"]:
        raise ValueError("Episode must exercise two slow-clock boundaries")
    steps = (e["teaching_events"] * e["fast_steps_per_event"]
             + e["teaching_events"] // e["slow_clock_events"] * e["slow_steps_per_boundary"])
    observe = e["teaching_events"] * e["observation_positions_per_event"]
    inner = steps * e["inner_positions_per_step_including_replay"]
    query = (e["future_queries"] + e["retention_queries"]) * e["positions_per_query"]
    return {"inner_optimizer_steps": steps, "observation_positions": observe,
            "inner_positions": inner, "outer_query_positions": query,
            "total_positions": observe + inner + query}


def power_design(p):
    normal = NormalDist()
    za = normal.inv_cdf(1 - p["family_alpha"] / (2 * p["primary_comparisons"]))
    zb = normal.inv_cdf(p["power"])
    rows = [{"effect": effect, "independent_groups": [math.ceil((za + zb)**2 * q / effect**2)
             for q in p["paired_discordance"]]} for effect in p["effects"]]
    # One-sided noninferiority, conservatively Bonferroni over the same family size.
    zni = normal.inv_cdf(1 - p["family_alpha"] / p["primary_comparisons"])
    retention = [math.ceil((zni + zb)**2 * q / p["retention_noninferiority_margin"]**2)
                 for q in p["retention_discordance"]]
    return {"superiority": rows, "retention_noninferiority_groups": retention,
            "pilot_200_detectable_effect": [(za + zb) * math.sqrt(q / 200)
                                             for q in p["paired_discordance"]],
            "zero_errors_95pct_upper_below_1pct_groups": math.ceil(math.log(0.05) / math.log(0.99))}


def scaled_cost(compute, positions):
    factor = positions / 1_000_000_000
    result = []
    for hw in compute["hardware"]:
        source = next(x for x in hw["scenarios"] if x["positions"] == 1_000_000_000)
        row = {"hardware": hw["id"], "gpus": hw["gpus"], "measured": False,
               "gpu_hours": [x * factor for x in source["gpu_hours"]],
               "wall_hours": [x * factor for x in source["wall_hours"]]}
        if "gpu_compute_USD" in source:
            row["gpu_compute_USD"] = [x * factor for x in source["gpu_compute_USD"]]
        result.append(row)
    return result


def calculate(c):
    if c["average_exposures"] <= 0 or c["positions_per_outer_update"] <= 0:
        raise ValueError("Positive exposures and update size required")
    folder = ROOT / c["evidence_directory"]
    count = verify_inputs(folder)
    prior = read(ROOT / c["compute_estimate"])
    e, a, campaign = c["episode"], c["audit"], c["campaign"]
    ep = episode_work(e)
    equivalence = (campaign["full_seeds"]
                   + campaign["additional_preparation_controls"] * campaign["seeds_per_control"]
                   * campaign["continuation_fraction"]
                   + sum(campaign["extra_scaling_curve_fractions"]))
    visits = a["factorial_arms"] * a["seeds"] * (
        a["canonical_worlds"] + a["additional_orders"] * a["worlds_per_additional_order"])
    extra_query = a["extra_query_variants"] * a["queries_per_variant"] * e["positions_per_query"]
    audit_no_p = ep["observation_positions"] + ep["outer_query_positions"] + extra_query
    audit_p_fraction = a["p_update_arms"] / a["factorial_arms"]
    if not 0 <= audit_p_fraction <= 1:
        raise ValueError("Invalid fraction of audit arms with P updates")
    models = []
    for model in c["models"]:
        buckets = []
        for b in c["ordinary_records"]:
            records = b["minicpm_records"] * model["scale"]
            row = {"bucket": b["bucket"], "distinct_record_target": records,
                   "record_visits": records * c["average_exposures"],
                   "processed_positions": records * c["average_exposures"] * b["mean_processed_positions"]}
            if "questions_per_asset" in b:
                row["distinct_asset_target_if_assumption_holds"] = records / b["questions_per_asset"]
                if "mean_asset_seconds" in b:
                    row["unique_media_hours_if_assumption_holds"] = records / b["questions_per_asset"] * b["mean_asset_seconds"] / 3600
            buckets.append(row)
        episodes = e["minicpm_unique_episodes"] * model["scale"]
        episode_visits = episodes * c["average_exposures"]
        ordinary = sum(x["processed_positions"] for x in buckets)
        total = ordinary + episode_visits * ep["total_positions"]
        outer = ordinary + episode_visits * (ep["observation_positions"] + ep["outer_query_positions"])
        for b in buckets:
            b["position_fraction"] = b["processed_positions"] / total
        s = c["storage_illustration"]
        by_bucket = {b["bucket"]: b for b in buckets}
        storage = (by_bucket["image_region_document"]["distinct_asset_target_if_assumption_holds"] * s["image_bytes_per_asset"]
                   + by_bucket["audio"]["unique_media_hours_if_assumption_holds"] * 3600
                   * s["audio_sample_rate"] * s["audio_bytes_per_sample_mono"]
                   + by_bucket["temporal_av"]["unique_media_hours_if_assumption_holds"] * 3600
                   * s["video_bits_per_second"] / 8)
        compute = next(x for x in prior["models"] if x["model"]["id"] == model["id"])
        models.append({**model, "ordinary_buckets": buckets,
                       "ordinary_record_target": sum(x["distinct_record_target"] for x in buckets),
                       "unique_episode_target": episodes, "episode_visits": episode_visits,
                       "unique_teaching_transactions": episodes * e["teaching_events"],
                       "episode_positions": episode_visits * ep["total_positions"],
                       "episode_position_fraction": episode_visits * ep["total_positions"] / total,
                       "total_processed_positions": total,
                       "outer_position_budget": outer,
                       "outer_updates_if_packing_contract_holds": math.ceil(outer / c["positions_per_outer_update"]),
                       "full_batch_work_equivalents": total / c["positions_per_outer_update"],
                       "separate_inner_p_steps": episode_visits * ep["inner_optimizer_steps"],
                       "illustrative_media_storage_GiB": storage / 2**30,
                       "minimum_research_training_positions": total * equivalence,
                       "single_schedule_conditional_compute": scaled_cost(compute, total),
                       "minimum_training_campaign_conditional_compute": scaled_cost(compute, total * equivalence),
                       "judgment_range_conditional_compute": [scaled_cost(compute, n) for n in model["judgment_position_range"]]})
    return {"format": "replica.convincing_result_data_estimate.v1",
            "status": c["status"], "verified_input_count": count,
            "episode": ep, "power": power_design(c["power"]),
            "campaign_full_schedule_equivalents": equivalence,
            "factorial_audit": {"episode_visits": visits,
                                "positions_per_p_enabled_visit": ep["total_positions"] + extra_query,
                                "positions_per_p_disabled_visit": audit_no_p,
                                "fixed_exposure_processed_positions": visits * (audit_no_p + audit_p_fraction * ep["inner_positions"]),
                                "equal_total_work_position_allowance": visits * (ep["total_positions"] + extra_query),
                                "gpu_hours": None,
                                "note": "Needs measured per-inner-step and forward-only pricing. Excludes ordinary modality/legacy audits, additional preparation contrasts and policy evolution."},
            "models": models,
            "compute_limits": "Inherited short-context mixed-precision delivery assumptions and routine elapsed allowances; no new device measurement. Fixed qualification, explicit factorial audit, teachers, acquisition, longer contexts and encoder-heavy reforecast are additional. M2 executable budget remains null."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = calculate(read(args.config))
    result["config_sha256"] = sha(args.config)
    result["calculator_sha256"] = sha(Path(__file__))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as f:
        json.dump(result, f, indent=2, sort_keys=True)
        f.write("\n")


if __name__ == "__main__":
    main()

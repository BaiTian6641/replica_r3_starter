#!/usr/bin/env python3
"""Prepare the Stage-2 9-Site Full-Attention Backbone & Placement Packet (V20).

Creates:
- configs/spark_backbone_9site_v20.json (`arm: "full_attention_nine"`,
  `reader_regime: "matched"`, `sites: [3, 7, 11, 15, 19, 23, 27, 31, 35]`,
  `launch_authorized: false`, gated behind Stage-1 V19 qualification)
- reports/spark_backbone_9site_v20/native_packet.json
- reports/spark_backbone_9site_v20/authorization.template.json
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from prepare_spark_alignment_v19 import (
    AUTHORIZATION_FORMAT,
    OUTER_WALL_SECONDS,
    PACKET_FORMAT,
    derive_v19_budget,
    derive_v19_parent_plan,
)
from replica_lab.intrinsic_memory_alignment_v19 import (
    ALIGNMENT_PROTOCOL,
    DEFAULT_TOP_K,
    alignment_protocol_contract,
)
from replica_lab.intrinsic_memory_budget_v12 import atomic
from replica_lab.intrinsic_memory_execution_v5 import hash_file
from replica_lab.intrinsic_memory_paired_protocol_v18 import totals

NINE_FULL_ATTENTION_SITES = [3, 7, 11, 15, 19, 23, 27, 31, 35]


def prepare_stage2_9site() -> dict:
    report_dir = ROOT / "reports/spark_backbone_9site_v20"
    config_path = ROOT / "configs/spark_backbone_9site_v20.json"
    report_dir.mkdir(parents=True, exist_ok=True)

    stage1_protocol = json.loads(
        (ROOT / "configs/spark_producer_reader_alignment_v19.json").read_text()
    )
    stage1_packet_path = ROOT / "reports/spark_alignment_v19/native_packet.json"
    stage1_result_path = (
        ROOT
        / "reports/spark_alignment_v19/native_bf16_alignment_attempt01/result.json"
    )
    stage1_result = json.loads(stage1_result_path.read_text())

    contract = alignment_protocol_contract(
        top_k=DEFAULT_TOP_K, steps=128, paired_joint=True
    )
    contract["sites"] = list(NINE_FULL_ATTENTION_SITES)
    contract["reader_regime"] = "matched"
    contract["placements_compared"] = [
        "block_input_query_to_post_block_residual",
        "block_input_query_to_pre_mlp_residual",
    ]

    budget = derive_v19_budget(steps_per_arm=128)
    parent_budget = derive_v19_parent_plan(budget)
    worker_totals = totals(budget)
    combined_totals = {
        key: val + totals(parent_budget)[key]
        for key, val in worker_totals.items()
    }
    output_root = "reports/spark_backbone_9site_v20/native_bf16_9site_attempt01"
    attempt_marker = "runs/spark_backbone_9site_v20/attempts.jsonl"

    protocol = {
        **stage1_protocol,
        "format": ALIGNMENT_PROTOCOL,
        "status": "prepared_stage2_bound_to_completed_stage1_v19",
        "launch_authorized": False,
        "stage": 2,
        "arm": "full_attention_nine",
        "reader_regime": "matched",
        "sites": list(NINE_FULL_ATTENTION_SITES),
        "intrinsic_placement": "block_input_query_to_pre_mlp_residual",
        "stage1_v19_predecessor_outcome": {
            "packet_path": str(stage1_packet_path.relative_to(ROOT)),
            "packet_sha256": hash_file(stage1_packet_path),
            "result_path": str(stage1_result_path.relative_to(ROOT)),
            "result_sha256": hash_file(stage1_result_path),
            "ce_teacher_forced_complete_correct": stage1_result["ce_final"][
                "teacher_forced_complete_correct"
            ],
            "ce_known_known_pair_both_correct_count": stage1_result["ce_final"][
                "known_known_pair_both_correct_count"
            ],
            "ce_positive_pair_margin_worlds": stage1_result["ce_final"][
                "positive_pair_margin_worlds"
            ],
            "ce_mean_final_pair_minimum_margin": stage1_result["ce_final"][
                "mean_final_pair_minimum_margin"
            ],
            "ce_diagnostic_regime": stage1_result["ce_final"][
                "diagnostic_regime"
            ],
            "kl_teacher_forced_complete_correct": stage1_result["kl_final"][
                "teacher_forced_complete_correct"
            ],
            "kl_diagnostic_regime": stage1_result["kl_final"][
                "diagnostic_regime"
            ],
            "scientific_classification": (
                "ce_7_of_8_complete_correct_3_of_4_pairs_solved_kl_decoder_interface_bottleneck_at_2_sites"
            ),
        },
        "alignment_contract": contract,
        "budget": budget,
        "parent_budget": parent_budget,
        "total_reservations": worker_totals,
        "total_with_preflight": combined_totals,
        "wall_seconds": OUTER_WALL_SECONDS,
        "output_root": output_root,
        "attempt_marker": attempt_marker,
    }
    atomic(config_path, protocol)
    protocol_sha = hash_file(config_path)

    packet_path = report_dir / "native_packet.json"
    packet = {
        "format": PACKET_FORMAT,
        "status": "prepared_stage2_gated_on_stage1_v19_qualification",
        "launch_authorized": False,
        "experiment_version": 20,
        "stage": 2,
        "arm": "full_attention_nine",
        "reader_regime": "matched",
        "attempts": 1,
        "retries": 0,
        "native_workers": 1,
        "cpu_fallback": False,
        "output_root": output_root,
        "attempt_marker": attempt_marker,
        "protocol": {
            "path": str(config_path.relative_to(ROOT)),
            "sha256": protocol_sha,
        },
    }
    atomic(packet_path, packet)
    packet_sha = hash_file(packet_path)

    auth_template_path = report_dir / "authorization.template.json"
    atomic(
        auth_template_path,
        {
            "format": AUTHORIZATION_FORMAT,
            "decision": "not_authorized",
            "stage": 2,
            "requires_stage1_v19_qualification": True,
            "packet_sha256": packet_sha,
            "protocol_sha256": protocol_sha,
            "checkpoint_manifest_sha256": protocol["checkpoint"]["manifest_sha256"],
            "fit_data_sha256": protocol["data"]["fit"]["sha256"],
            "diagnostic_fit_data_sha256": protocol["data"]["diagnostic_fit"]["sha256"],
            "output_root": output_root,
            "attempts": 1,
            "retries": 0,
            "human_approval_reference": None,
            "human_approval_text": None,
        },
    )
    return {
        "stage": 2,
        "packet_path": str(packet_path.relative_to(ROOT)),
        "packet_sha256": packet_sha,
        "protocol_path": str(config_path.relative_to(ROOT)),
        "protocol_sha256": protocol_sha,
        "sites": NINE_FULL_ATTENTION_SITES,
        "reader_regime": "matched",
        "placement": protocol["intrinsic_placement"],
    }


def prepare_stage2_9site_independent() -> dict:
    report_dir = ROOT / "reports/spark_backbone_9site_v20"
    config_path = ROOT / "configs/spark_backbone_9site_independent_v20.json"
    report_dir.mkdir(parents=True, exist_ok=True)

    stage1_protocol = json.loads(
        (ROOT / "configs/spark_producer_reader_alignment_v19.json").read_text()
    )
    stage1_packet_path = ROOT / "reports/spark_alignment_v19/native_packet.json"
    stage1_result_path = (
        ROOT
        / "reports/spark_alignment_v19/native_bf16_alignment_attempt01/result.json"
    )
    stage1_result = json.loads(stage1_result_path.read_text())

    stage2_matched_packet_path = (
        ROOT / "reports/spark_backbone_9site_v20/native_packet.json"
    )
    stage2_matched_result_path = (
        ROOT
        / "reports/spark_backbone_9site_v20/native_bf16_9site_attempt01/result.json"
    )
    stage2_matched_result = json.loads(stage2_matched_result_path.read_text())

    contract = alignment_protocol_contract(
        top_k=DEFAULT_TOP_K, steps=128, paired_joint=True
    )
    contract["sites"] = list(NINE_FULL_ATTENTION_SITES)
    contract["reader_regime"] = "independent"
    contract["placements_compared"] = [
        "block_input_query_to_post_block_residual",
        "block_input_query_to_pre_mlp_residual",
    ]

    budget = derive_v19_budget(steps_per_arm=128)
    parent_budget = derive_v19_parent_plan(budget)
    worker_totals = totals(budget)
    combined_totals = {
        key: val + totals(parent_budget)[key]
        for key, val in worker_totals.items()
    }
    output_root = (
        "reports/spark_backbone_9site_v20/native_bf16_9site_independent_attempt01"
    )
    attempt_marker = "runs/spark_backbone_9site_v20/attempts_independent.jsonl"

    protocol = {
        **stage1_protocol,
        "format": ALIGNMENT_PROTOCOL,
        "status": "prepared_stage2b_independent_bound_to_stage1_v19_and_stage2a_matched",
        "launch_authorized": False,
        "stage": 2,
        "arm": "full_attention_nine",
        "reader_regime": "independent",
        "sites": list(NINE_FULL_ATTENTION_SITES),
        "intrinsic_placement": "block_input_query_to_post_block_residual",
        "zero_shock_step0_parity_with_v16_2site": True,
        "stage1_v19_predecessor_outcome": {
            "packet_path": str(stage1_packet_path.relative_to(ROOT)),
            "packet_sha256": hash_file(stage1_packet_path),
            "result_path": str(stage1_result_path.relative_to(ROOT)),
            "result_sha256": hash_file(stage1_result_path),
            "ce_teacher_forced_complete_correct": stage1_result["ce_final"][
                "teacher_forced_complete_correct"
            ],
            "ce_known_known_pair_both_correct_count": stage1_result["ce_final"][
                "known_known_pair_both_correct_count"
            ],
            "ce_positive_pair_margin_worlds": stage1_result["ce_final"][
                "positive_pair_margin_worlds"
            ],
            "ce_mean_final_pair_minimum_margin": stage1_result["ce_final"][
                "mean_final_pair_minimum_margin"
            ],
            "ce_diagnostic_regime": stage1_result["ce_final"][
                "diagnostic_regime"
            ],
            "kl_teacher_forced_complete_correct": stage1_result["kl_final"][
                "teacher_forced_complete_correct"
            ],
            "kl_diagnostic_regime": stage1_result["kl_final"][
                "diagnostic_regime"
            ],
            "scientific_classification": (
                "ce_7_of_8_complete_correct_3_of_4_pairs_solved_kl_decoder_interface_bottleneck_at_2_sites"
            ),
        },
        "stage2a_matched_predecessor_outcome": {
            "packet_path": str(stage2_matched_packet_path.relative_to(ROOT)),
            "packet_sha256": hash_file(stage2_matched_packet_path),
            "result_path": str(stage2_matched_result_path.relative_to(ROOT)),
            "result_sha256": hash_file(stage2_matched_result_path),
            "ce_mean_final_pair_minimum_margin": stage2_matched_result[
                "ce_final"
            ]["mean_final_pair_minimum_margin"],
            "kl_mean_final_pair_minimum_margin": stage2_matched_result[
                "kl_final"
            ]["mean_final_pair_minimum_margin"],
            "ce_minus_kl_pair_minimum_margin": stage2_matched_result[
                "comparison"
            ]["ce_minus_kl_pair_minimum_margin"],
            "scientific_classification": (
                "depth_grouped_matched_weight_tying_caused_step0_broadcast_shock_and_cross_depth_gradient_cancellation"
            ),
        },
        "alignment_contract": contract,
        "budget": budget,
        "parent_budget": parent_budget,
        "total_reservations": worker_totals,
        "total_with_preflight": combined_totals,
        "wall_seconds": OUTER_WALL_SECONDS,
        "output_root": output_root,
        "attempt_marker": attempt_marker,
    }
    atomic(config_path, protocol)
    protocol_sha = hash_file(config_path)

    packet_path = report_dir / "native_packet_independent.json"
    packet = {
        "format": PACKET_FORMAT,
        "status": "prepared_stage2b_independent_9site_post_block_zero_shock",
        "launch_authorized": False,
        "experiment_version": 20,
        "stage": 2,
        "arm": "full_attention_nine",
        "reader_regime": "independent",
        "attempts": 1,
        "retries": 0,
        "native_workers": 1,
        "cpu_fallback": False,
        "output_root": output_root,
        "attempt_marker": attempt_marker,
        "protocol": {
            "path": str(config_path.relative_to(ROOT)),
            "sha256": protocol_sha,
        },
    }
    atomic(packet_path, packet)
    packet_sha = hash_file(packet_path)

    auth_template_path = report_dir / "authorization_independent.template.json"
    atomic(
        auth_template_path,
        {
            "format": AUTHORIZATION_FORMAT,
            "decision": "not_authorized",
            "stage": 2,
            "requires_stage1_v19_qualification": True,
            "packet_sha256": packet_sha,
            "protocol_sha256": protocol_sha,
            "checkpoint_manifest_sha256": protocol["checkpoint"]["manifest_sha256"],
            "fit_data_sha256": protocol["data"]["fit"]["sha256"],
            "diagnostic_fit_data_sha256": protocol["data"]["diagnostic_fit"]["sha256"],
            "output_root": output_root,
            "attempts": 1,
            "retries": 0,
            "human_approval_reference": None,
            "human_approval_text": None,
        },
    )
    return {
        "stage": 2,
        "packet_path": str(packet_path.relative_to(ROOT)),
        "packet_sha256": packet_sha,
        "protocol_path": str(config_path.relative_to(ROOT)),
        "protocol_sha256": protocol_sha,
        "sites": NINE_FULL_ATTENTION_SITES,
        "reader_regime": "independent",
        "placement": protocol["intrinsic_placement"],
        "zero_shock_step0_parity_with_v16_2site": True,
    }


def prepare_stage2_9site_independent_attempt02() -> dict:
    report_dir = ROOT / "reports/spark_backbone_9site_v20"
    config_path = ROOT / "configs/spark_backbone_9site_independent_v20_attempt02.json"
    report_dir.mkdir(parents=True, exist_ok=True)

    prev_protocol = json.loads(
        (ROOT / "configs/spark_backbone_9site_independent_v20.json").read_text()
    )
    attempt01_receipt = (
        ROOT
        / "reports/spark_backbone_9site_v20/native_bf16_9site_independent_attempt01/supervision/receipt.json"
    )
    attempt01_stderr = (
        ROOT
        / "reports/spark_backbone_9site_v20/native_bf16_9site_independent_attempt01/supervision/child.stderr"
    )

    output_root = (
        "reports/spark_backbone_9site_v20/native_bf16_9site_independent_attempt02"
    )
    attempt_marker = "runs/spark_backbone_9site_v20/attempts_independent_attempt02.jsonl"

    protocol = {
        **prev_protocol,
        "status": "prepared_stage2b_independent_attempt02_after_load_stage_requires_grad_fix",
        "launch_authorized": False,
        "output_root": output_root,
        "attempt_marker": attempt_marker,
        "stage2b_attempt01_load_failure_provenance": {
            "packet_path": "reports/spark_backbone_9site_v20/native_packet_independent.json",
            "packet_sha256": hash_file(
                ROOT / "reports/spark_backbone_9site_v20/native_packet_independent.json"
            ),
            "supervision_receipt_path": str(attempt01_receipt.relative_to(ROOT)),
            "supervision_receipt_sha256": hash_file(attempt01_receipt),
            "child_stderr_path": str(attempt01_stderr.relative_to(ROOT)),
            "child_stderr_sha256": hash_file(attempt01_stderr),
            "stage_failed": "load",
            "forwards_executed": 0,
            "backwards_executed": 0,
            "updates_executed": 0,
            "root_cause": (
                "_upgrade_to_independent_reader set p.requires_grad_(True) on cloned ports "
                "before path.assert_optimizer_ownership() while donor ports 3 and 7 still "
                "had requires_grad=False during load, triggering ContextualMemoryError("
                "'partial phi freezing is not a declared training phase')"
            ),
            "fix_applied": (
                "Match donor_requires_grad during _upgrade_to_independent_reader so all "
                "modules share uniform requires_grad=False during assert_optimizer_ownership() "
                "and are uniformly enabled together immediately afterward."
            ),
        },
    }
    atomic(config_path, protocol)
    protocol_sha = hash_file(config_path)

    packet_path = report_dir / "native_packet_independent_attempt02.json"
    packet = {
        "format": PACKET_FORMAT,
        "status": "prepared_stage2b_independent_9site_post_block_zero_shock_attempt02",
        "launch_authorized": False,
        "experiment_version": 20,
        "stage": 2,
        "arm": "full_attention_nine",
        "reader_regime": "independent",
        "attempts": 1,
        "retries": 0,
        "native_workers": 1,
        "cpu_fallback": False,
        "output_root": output_root,
        "attempt_marker": attempt_marker,
        "protocol": {
            "path": str(config_path.relative_to(ROOT)),
            "sha256": protocol_sha,
        },
    }
    atomic(packet_path, packet)
    packet_sha = hash_file(packet_path)

    auth_template_path = (
        report_dir / "authorization_independent_attempt02.template.json"
    )
    atomic(
        auth_template_path,
        {
            "format": AUTHORIZATION_FORMAT,
            "decision": "not_authorized",
            "stage": 2,
            "requires_stage1_v19_qualification": True,
            "packet_sha256": packet_sha,
            "protocol_sha256": protocol_sha,
            "checkpoint_manifest_sha256": protocol["checkpoint"]["manifest_sha256"],
            "fit_data_sha256": protocol["data"]["fit"]["sha256"],
            "diagnostic_fit_data_sha256": protocol["data"]["diagnostic_fit"]["sha256"],
            "output_root": output_root,
            "attempts": 1,
            "retries": 0,
            "human_approval_reference": None,
            "human_approval_text": None,
        },
    )
    return {
        "stage": 2,
        "packet_path": str(packet_path.relative_to(ROOT)),
        "packet_sha256": packet_sha,
        "protocol_path": str(config_path.relative_to(ROOT)),
        "protocol_sha256": protocol_sha,
        "sites": NINE_FULL_ATTENTION_SITES,
        "reader_regime": "independent",
        "placement": protocol["intrinsic_placement"],
        "zero_shock_step0_parity_with_v16_2site": True,
    }


if __name__ == "__main__":
    print(
        json.dumps(
            prepare_stage2_9site_independent_attempt02(), indent=2, sort_keys=True
        )
    )

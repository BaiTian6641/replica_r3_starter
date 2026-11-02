#!/usr/bin/env python3
"""Prepare the V19 fresh bounded paired-joint fitting & matched CE-vs-top-64-KL alignment packet.

Creates:
- data/intrinsic_memory_alignment_v19/fit.json (4 fresh known-known pairs = 8 worlds)
- data/intrinsic_memory_alignment_v19/diagnostic_fit.json (4 disjoint diagnostic pairs = 8 worlds)
- configs/spark_producer_reader_alignment_v19.json
- reports/spark_alignment_v19/native_packet.json (launch_authorized: false)
- reports/spark_alignment_v19/authorization.template.json
- reports/spark_alignment_v19/cpu_validation.json
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from replica_lab.intrinsic_memory_alignment_v19 import (
    ALIGNMENT_PROTOCOL,
    DEFAULT_TOP_K,
    alignment_protocol_contract,
    evaluate_exact_text_competence,
    run_alignment_comparison,
    validate_alignment_pairs_v19,
)
from replica_lab.intrinsic_memory_budget_v12 import COUNTERS, atomic, sha_json, validate_plan
from replica_lab.intrinsic_memory_coverage_v12 import ENVIRONMENT_CLOSURE_BYTES, SOURCE_FILE_BYTES
from replica_lab.intrinsic_memory_cpu_fixture_v18 import make_engine
from replica_lab.intrinsic_memory_execution_v5 import bound_json, hash_file
from replica_lab.intrinsic_memory_paired_protocol_v18 import (
    CHECKPOINT_READ_BOUND,
    METADATA_READ_BOUND,
    totals,
)

PACKET_FORMAT = "replica.intrinsic_memory.alignment_launch_packet.v19"
AUTHORIZATION_FORMAT = "replica.intrinsic_memory.alignment_authorization.v19"
OUTER_WALL_SECONDS = 3300

FIT_PAIRS = (
    ("Aurelia", "cobalt", "red", "blue"),
    ("Bramwell", "vermilion", "green", "gold"),
    ("Calderon", "saffron", "blue", "green"),
    ("Dorian", "obsidian", "gold", "red"),
)

DIAGNOSTIC_PAIRS = (
    ("Evander", "cerulean", "red", "green"),
    ("Fiora", "chartreuse", "blue", "gold"),
    ("Gideon", "amaranth", "green", "red"),
    ("Helena", "ultramarine", "gold", "blue"),
)


def _build_corpus_rows(
    pair_specs: tuple[tuple[str, str, str, str], ...],
    *,
    prefix: str,
    split_name: str,
    seed_base: int,
) -> list[dict]:
    rows: list[dict] = []
    for pair_idx, (entity,relay, ans0, ans1) in enumerate(pair_specs):
        family = f"{prefix}_decision_{pair_idx:02d}"
        question = f"What is the recorded color of {entity}?"
        for side, answer in enumerate((ans0, ans1)):
            wid = f"{prefix}_{2 * pair_idx + side:02d}"
            text = (
                f"Archive ledger entry for {entity}: relay {relay} confirms "
                f"the recorded color of {entity} is {answer}."
            )
            rows.append(
                {
                    "world_id": wid,
                    "split": split_name,
                    "family": family,
                    "group_id": pair_idx,
                    "variant": "base" if side == 0 else "counterfactual",
                    "entity": entity,
                    "relay": relay,
                    "text": text,
                    "question": question,
                    "answer": answer,
                    "alternative_answer": ans1 if side == 0 else ans0,
                    "seed": seed_base + 2 * pair_idx + side,
                }
            )
    return validate_alignment_pairs_v19(rows)


def _stage(wall: int, **counts: int) -> dict:
    return {**{key: 0 for key in COUNTERS}, "wall_seconds": wall, **counts}


def derive_v19_budget(*, steps_per_arm: int = 128) -> dict:
    # 16 worlds (8 fit + 8 disjoint diagnostic):
    # - Exact-text competence check: 16 worlds * 2 candidates = 32 forwards
    # - Top-64 teacher extraction on 8 fitting worlds = 8 forwards
    # - Per arm (2 arms):
    #   - Baseline + terminal evaluation on 16 worlds * 6 forwards/world = 192 forwards/arm (384 total)
    #   - Paired-joint training: steps_per_arm * 2 worlds/step = 256 forwards/arm (512 total)
    #   - Frozen source feature forwards: 16 (cap 48)
    eval_forwards = 32 + 8 + 2 * 2 * 16 * 6
    train_forwards = 2 * steps_per_arm * 2
    source_forwards = 48
    total_forwards = eval_forwards + train_forwards + source_forwards
    total_backwards = 2 * steps_per_arm * 2
    total_updates = 2 * steps_per_arm
    plan = {
        "admission": _stage(
            180,
            artifact_read_bytes=(
                SOURCE_FILE_BYTES
                + ENVIRONMENT_CLOSURE_BYTES
                + CHECKPOINT_READ_BOUND
                + METADATA_READ_BOUND
            ),
            artifact_bytes=4 * 1024**2,
            writes=8,
        ),
        "load": _stage(
            240,
            model_loads=1,
            restart_restores=1,
            writes=18,
            artifact_bytes=1024**2,
            artifact_read_bytes=(
                SOURCE_FILE_BYTES + CHECKPOINT_READ_BOUND + METADATA_READ_BOUND
            ),
        ),
        "diagnose": _stage(
            3000,
            forwards=total_forwards,
            backwards=total_backwards,
            updates=total_updates,
            input_tokens=(eval_forwards + train_forwards) * 256 + 16 * 512,
            source_encodes=16,
            source_feature_forwards=source_forwards,
            tokenizer_calls=16,
            prompt_tokenizer_calls=48,
            source_token_bytes=16 * 256,
            cache_misses=source_forwards,
            cache_evictions=source_forwards,
            cache_recomputations=source_forwards,
            artifact_read_bytes=16 * 1024**2,
            writes=6400,
            artifact_bytes=64 * 1024**2,
        ),
        "finalize": _stage(
            60,
            writes=4,
            artifact_bytes=2 * 1024**2,
        ),
    }
    validate_plan(plan)
    return plan


def derive_v19_parent_plan(worker_plan: dict) -> dict:
    plan = {
        "admission": worker_plan["admission"],
        "supervise": _stage(
            OUTER_WALL_SECONDS,
            writes=32,
            artifact_bytes=4 * 1024**2,
            artifact_read_bytes=16 * 1024**2,
        ),
    }
    validate_plan(plan)
    return plan


def prepare() -> dict:
    data_dir = ROOT / "data/intrinsic_memory_alignment_v19"
    report_dir = ROOT / "reports/spark_alignment_v19"
    config_path = ROOT / "configs/spark_producer_reader_alignment_v19.json"
    data_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    fit_rows = _build_corpus_rows(
        FIT_PAIRS, prefix="v19_align_fit", split_name="fit", seed_base=1900
    )
    diag_rows = _build_corpus_rows(
        DIAGNOSTIC_PAIRS,
        prefix="v19_align_diag",
        split_name="diagnostic_fit",
        seed_base=1950,
    )

    fit_path = data_dir / "fit.json"
    diag_path = data_dir / "diagnostic_fit.json"
    atomic(
        fit_path,
        {
            "format": "replica.intrinsic_memory.alignment_corpus.v19",
            "split": "fit",
            "pairs": 4,
            "worlds": 8,
            "disjoint_from_v11": True,
            "disjoint_from_sealed_v19_eval": True,
            "rows": fit_rows,
        },
    )
    atomic(
        diag_path,
        {
            "format": "replica.intrinsic_memory.alignment_corpus.v19",
            "split": "diagnostic_fit",
            "pairs": 4,
            "worlds": 8,
            "disjoint_from_fit": True,
            "disjoint_from_v11": True,
            "disjoint_from_sealed_v19_eval": True,
            "rows": diag_rows,
        },
    )

    parent_v18 = json.loads(
        (ROOT / "configs/spark_paired_update_v18_2_preparation04.json").read_text()
    )
    v18_native_result_path = (
        ROOT
        / "reports/spark_repairs_v18/preparation_attempt04/native_bf16_comparison_attempt01/result.json"
    )
    v18_native_result = json.loads(v18_native_result_path.read_text())

    contract = alignment_protocol_contract(
        top_k=DEFAULT_TOP_K, steps=128, paired_joint=True
    )
    budget = derive_v19_budget(steps_per_arm=128)
    parent_budget = derive_v19_parent_plan(budget)
    worker_totals = totals(budget)
    combined_totals = {
        key: val + totals(parent_budget)[key]
        for key, val in worker_totals.items()
    }
    output_root = "reports/spark_alignment_v19/native_bf16_alignment_attempt01"
    attempt_marker = "runs/spark_alignment_v19/attempts.jsonl"

    protocol = {
        "format": ALIGNMENT_PROTOCOL,
        "status": "prepared_conditional_not_selected_or_authorized",
        "launch_authorized": False,
        "arm": "two",
        "reader_regime": "independent",
        "predecessor_v18_outcome": {
            "packet_sha256": v18_native_result["packet_sha256"],
            "result_path": str(v18_native_result_path.relative_to(ROOT)),
            "result_sha256": hash_file(v18_native_result_path),
            "sequential_mean_final_pair_minimum_margin": v18_native_result[
                "summary"
            ]["endpoints"]["sequential"]["mean_final_pair_minimum_margin"],
            "paired_joint_mean_final_pair_minimum_margin": v18_native_result[
                "summary"
            ]["endpoints"]["paired_joint"]["mean_final_pair_minimum_margin"],
            "joint_minus_sequential": v18_native_result["summary"][
                "joint_minus_sequential"
            ],
            "relative_conditions_passed": v18_native_result["summary"][
                "relative_conditions_passed"
            ],
            "paired_joint_teacher_forced_complete_correct": v18_native_result[
                "summary"
            ]["endpoints"]["paired_joint"]["teacher_forced_complete_correct"],
            "sequential_teacher_forced_complete_correct": v18_native_result[
                "summary"
            ]["endpoints"]["sequential"]["teacher_forced_complete_correct"],
            "scientific_classification": (
                "relative_paired_joint_improvement_with_inadequate_absolute_recall"
            ),
        },
        "source_repo": parent_v18["source_repo"],
        "source_revision": parent_v18["source_revision"],
        "source": parent_v18["source"],
        "source_execution": parent_v18["source_execution"],
        "source_features": parent_v18["source_features"],
        "source_feature_binding": parent_v18["source_feature_binding"],
        "generation": parent_v18["generation"],
        "environment": parent_v18["environment"],
        "environment_closure": parent_v18["environment_closure"],
        "slot_policy_v12": parent_v18["slot_policy_v12"],
        "intrinsic_placement": parent_v18["intrinsic_placement"],
        "intrinsic_reader_key_mode": "content",
        "sites": [3, 7],
        "checkpoint": parent_v18["diagnostic"]["checkpoint"],
        "checkpoint_components": parent_v18["diagnostic"]["checkpoint_components"],
        "alignment_contract": contract,
        "training": {
            "optimizer": "NativeFusedAdamW",
            "lr": 0.0001,
            "seed": 17,
            "steps_per_arm": 128,
            "total_parameter_updates": 256,
            "paired_joint_batch_size": 2,
            "joint_coefficients": [0.5, 0.5],
            "inherited_v16_moments": True,
        },
        "data": {
            "fit": {
                "path": str(fit_path.relative_to(ROOT)),
                "sha256": hash_file(fit_path),
                "pairs": 4,
                "worlds": 8,
            },
            "diagnostic_fit": {
                "path": str(diag_path.relative_to(ROOT)),
                "sha256": hash_file(diag_path),
                "pairs": 4,
                "worlds": 8,
                "used_in_optimizer_steps": False,
            },
            "development_labels_used": False,
            "sealed_evaluation_labels_opened": False,
        },
        "budget": budget,
        "parent_budget": parent_budget,
        "total_reservations": worker_totals,
        "total_with_preflight": combined_totals,
        "wall_seconds": OUTER_WALL_SECONDS,
        "resource_limits": parent_v18["arms"]["two"]["resource_limits"],
        "output_root": output_root,
        "attempt_marker": attempt_marker,
    }
    atomic(config_path, protocol)
    protocol_sha = hash_file(config_path)

    packet_path = report_dir / "native_packet.json"
    packet = {
        "format": PACKET_FORMAT,
        "status": "prepared_conditional_not_selected_or_authorized",
        "launch_authorized": False,
        "experiment_version": 19,
        "arm": "two",
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

    auth_template_path = report_dir / "authorization.template.json"
    atomic(
        auth_template_path,
        {
            "format": AUTHORIZATION_FORMAT,
            "decision": "not_authorized",
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

    # Run deterministic CPU validation on the fresh V19 corpus + disjoint diagnostic split
    with tempfile.TemporaryDirectory() as tmp:
        engine = make_engine(Path(tmp))
        cpu_run = run_alignment_comparison(
            engine,
            fit_rows,
            diagnostic_rows=diag_rows,
            top_k=DEFAULT_TOP_K,
            steps=4,
            paired_joint=True,
            require_competence=False,
        )

    cpu_val_path = report_dir / "cpu_validation.json"
    cpu_receipt = {
        "format": "replica.intrinsic_memory.alignment_cpu_validation.v19",
        "packet_sha256": packet_sha,
        "protocol_sha256": protocol_sha,
        "fit_data_sha256": protocol["data"]["fit"]["sha256"],
        "diagnostic_fit_data_sha256": protocol["data"]["diagnostic_fit"]["sha256"],
        "native_model_loads": 0,
        "exact_text_competence_checked": True,
        "disjoint_diagnostic_evaluated": (
            "disjoint_diagnostic" in cpu_run["arms"]["direct_answer_plus_eos_ce"]
        ),
        "shuffled_probe_controls_verified": (
            "slot_shuffled_probe_accuracy"
            in cpu_run["arms"]["direct_answer_plus_eos_ce"]["final"]
        ),
        "state_restoration_equal": [
            all(item["equal"].values()) for item in cpu_run["state_restoration"]
        ],
        "comparison": cpu_run["comparison"],
        "ce_final_summary": cpu_run["arms"]["direct_answer_plus_eos_ce"]["final"],
        "kl_final_summary": cpu_run["arms"][f"top_{DEFAULT_TOP_K}_teacher_kl"]["final"],
    }
    atomic(cpu_val_path, cpu_receipt)
    return {
        "packet_path": str(packet_path.relative_to(ROOT)),
        "packet_sha256": packet_sha,
        "protocol_path": str(config_path.relative_to(ROOT)),
        "protocol_sha256": protocol_sha,
        "fit_data_sha256": protocol["data"]["fit"]["sha256"],
        "diagnostic_fit_data_sha256": protocol["data"]["diagnostic_fit"]["sha256"],
        "cpu_validation_path": str(cpu_val_path.relative_to(ROOT)),
        "cpu_validation_sha256": hash_file(cpu_val_path),
    }


if __name__ == "__main__":
    print(json.dumps(prepare(), indent=2, sort_keys=True))

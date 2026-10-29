#!/usr/bin/env python3
"""Hash-pin the existing SparkX research inputs and verify a clean checkout.

The bundle is a *manifest*, not a copy. It never duplicates model weights,
media assets or other large payloads, and it never rewrites an input file.
Every required input is identified by repository-relative path, role, byte
size and SHA-256 digest.

A clean checkout has every required input tracked by git, no unstaged input
changes, and worktree bytes matching the manifest. Untracked inputs and
unstaged modifications are refused by default; intentional staged changes are
allowed because they belong to the commit being prepared. ``--allow-untracked``
is an explicit development-only override and records ``clean_checkout_ready:
false``.

Commands:
  python3 scripts/prepare_sparkx_research_bundle.py --write
  python3 scripts/prepare_sparkx_research_bundle.py --check
  python3 scripts/prepare_sparkx_research_bundle.py --check --allow-untracked
  python3 scripts/prepare_sparkx_research_bundle.py --profile final-product --write
  python3 scripts/prepare_sparkx_research_bundle.py --profile final-product --check

The historical first-tranche profile is the default and its 117-input manifest
is never rewritten by the final-product profile. The final-product profile is
additive: it pins current acceptance, calibration, continuation, vision-curve,
final-verification and clean-export evidence, plus the clean-export links.

Exit codes: 0 verified, 1 verification failure, 2 missing/unreadable input
or manifest. No network access, no model or dataset loader, no native job.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

FORMAT = "replica.sparkx_research_bundle.v1"
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = "reports/sparkx_first_tranche_20261026/research_bundle_manifest.json"
DEFAULT_FINAL_PRODUCT_MANIFEST = (
    "reports/sparkx_final_product_20261026/final_product_bundle_manifest.json"
)
FINAL_PRODUCT_FORMAT = "replica.sparkx_final_product_bundle.v1"
FINAL_PRODUCT_GENERATED_OUTPUT = "reports/sparkx_final_product_20261026/verification_v2.json"

# Ordered (path, role) pairs. Paths are repository-relative; the manifest is
# deterministic because entries are sorted by path before serialization.
INPUT_ROLES: list[tuple[str, str]] = [
    (
        "reports/spark_preparation_20261025/run_01/headers/"
        "official_bf16_control/tensor_index.json",
        "sealed BF16 source tensor index used for the backbone parameter inventory",
    ),
    (
        "reports/spark_preparation_20261025/run_01/headers/"
        "official_fp8_adaptation_base/tensor_index.json",
        "sealed mixed-dtype FP8 adaptation tensor index (source storage only)",
    ),
    (
        "reports/fused_adamw_20261025/sustained01.guard.json",
        "repository driver-guard policy values",
    ),
    (
        "docs/architecture/21_SparkX_2_6_4B_Consolidated_Architecture_and_MiniCPM_Validation.md",
        "declared encoder counts, revisions, projectors and memory-module scope",
    ),
    (
        "configs/sparkx_residency_components_v1.json",
        "versioned machine-readable residency component inventory",
    ),
    (
        "reports/sparkx_residency_20261026/residency_v2_markdown_legacy.json",
        "preserved prose-derived residency v2 legacy receipt",
    ),
    (
        "configs/sparkx_2_6_4b_nested_learning_v1.json",
        "revision 4.5 nested-learning research specification",
    ),
    (
        "configs/sparkx_nested_learning_sources_v1.json",
        "declared content-addressed nested source cache and provisioning policy",
    ),
    (
        "configs/sparkx_2_6_4b_intrinsic_learning_spine_v1.json",
        "intrinsic memory and optional spine research configuration",
    ),
    (
        "docs/architecture/22_State_Mechanism_and_RSI_Contracts.md",
        "state ownership, transaction and bounded-RSI contracts",
    ),
    (
        "docs/architecture/README.md",
        "architecture reading order and revision 4.5 entry point",
    ),
    (
        "docs/research/sparkx_nested_learning_20261026/README.md",
        "nested-learning reading guide and evidence boundary",
    ),
    (
        "docs/research/sparkx_nested_learning_20261026/Nested_Learning_Contract.md",
        "nested-learning algorithm and integration contract",
    ),
    (
        "docs/research/sparkx_nested_learning_20261026/Validation_and_Open_Issues.md",
        "NL0-NL7 validation ladder and open issue ledger",
    ),
    (
        "docs/research/sparkx_nested_learning_20261026/Research_Review.md",
        "nested-learning primary-source comparison and limits",
    ),
    (
        "docs/research/sparkx_intrinsic_learning_spine_20261026/README.md",
        "intrinsic-memory background and revision status",
    ),
    (
        "docs/research/sparkx_intrinsic_learning_spine_20261026/Architecture_and_Spine.md",
        "intrinsic memory and optional spine design",
    ),
    (
        "docs/research/sparkx_intrinsic_learning_spine_20261026/Teaching_and_Validation.md",
        "intrinsic learning validation requirements",
    ),
    (
        "docs/research/sparkx_intrinsic_learning_spine_20261026/Research_Review.md",
        "intrinsic learning research review",
    ),
    (
        "reports/sparkx_nested_learning_20261026/README.md",
        "nested-learning audit report and non-claims",
    ),
    (
        "reports/sparkx_nested_learning_20261026/evidence.json",
        "nested-learning source manifest and provenance",
    ),
    (
        "reports/sparkx_nested_learning_20261026/analytical_costs.json",
        "nested-learning analytical payload ledger",
    ),
    (
        "reports/sparkx_nested_learning_20261026/calculate_costs.py",
        "nested-learning analytical cost generator",
    ),
    (
        "reports/sparkx_nested_learning_20261026/verify.py",
        "nested-learning document and source-payload verifier",
    ),
    (
        "scripts/provision_sparkx_nested_sources.py",
        "offline nested source cache provisioner",
    ),
    (
        "reports/sparkx_nested_learning_20261026/verification.json",
        "current strict nested source verification receipt",
    ),
    (
        "reports/sparkx_nested_learning_20261026/verification_absolute_legacy.json",
        "preserved absolute-path nested verification legacy receipt",
    ),
    (
        "reports/sparkx_nested_learning_20261026/repository_snapshot.json",
        "nested-learning repository observation and snapshot bindings",
    ),
    (
        "reports/sparkx_nested_learning_20261026/architecture_v4_4_snapshot.txt",
        "preserved revision 4.4 architecture snapshot",
    ),
    (
        "reports/sparkx_nested_learning_20261026/state_contract_snapshot.txt",
        "preserved pre-revision state contract snapshot",
    ),
    *[
        (
            f"reports/sparkx_nested_learning_20261026/input_{index}_snapshot.txt",
            "preserved nested-learning input snapshot",
        )
        for index in range(2, 10)
    ],
    (
        "reports/sparkx_residency_20261026/calculate_residency.py",
        "deterministic residency ledger generator",
    ),
    (
        "scripts/prepare_sparkx_research_bundle.py",
        "bundle manifest generator/verifier (this file's hash is pinned separately)",
    ),
    (
        "scripts/prepare_sparkx_first_tranche.py",
        "bounded first-tranche admission implementation",
    ),
    (
        "reports/sparkx_residency_20261026/residency_v2.json",
        "current analytical residency receipt (v2)",
    ),
    (
        "docs/experiments/20261026_Spark_Residency_Reconciliation.md",
        "residency reconciliation note: 18-byte formula, dtype inventory and gates",
    ),
    (
        "docs/datasets/20261026_SparkX_Dataset_Catalog.json",
        "source pins, licenses, split metadata and priorities",
    ),
    (
        "docs/datasets/20261026_SparkX_Dataset_Plan.md",
        "dataset strategy and intake constraints",
    ),
    (
        "docs/datasets/20261026_MiniCPM_Dataset_Validation.md",
        "MiniCPM validation plan referenced by the dataset package",
    ),
    (
        "docs/datasets/20261024_Source_Catalog.json",
        "earlier source catalog retained as a reference",
    ),
    (
        "reports/sparkx_dataset_hunt_20261026/evidence.json",
        "per-source evidence records and hashed provenance",
    ),
    (
        "reports/sparkx_dataset_hunt_20261026/hf_metadata.json",
        "pinned Hugging Face metadata snapshot",
    ),
    (
        "reports/sparkx_dataset_hunt_20261026/README.md",
        "dataset research report and limitations",
    ),
    (
        "reports/sparkx_dataset_hunt_20261026/documentation_validation.json",
        "prior documentation validation record retained as an input",
    ),
    (
        "reports/sparkx_dataset_hunt_20261026/repository_snapshot.json",
        "repository snapshot pins for the research package",
    ),
    (
        "reports/sparkx_dataset_hunt_20261026/clotho_metadata_audit.json",
        "Clotho provenance CSV count audit",
    ),
    (
        "reports/sparkx_dataset_validation_20261026/validation.json",
        "metadata-only validation report",
    ),
    (
        "reports/sparkx_dataset_validation_20261026/intake_manifest.json",
        "prospective metadata-only intake record",
    ),
    (
        "configs/dataset_registry.json",
        "runtime dataset registry (quarantined, read-only for this bundle)",
    ),
    (
        "configs/sparkx_first_tranche_v1.json",
        "bounded first-tranche protocol configuration",
    ),
    (
        "docs/08_Dataset_Acquisition_and_Curriculum.md",
        "acquisition guide retained with its known snapshot drift",
    ),
    (
        "docs/experiments/20261026_Spark_Dataset_Preparation.md",
        "dataset preparation, provisioning and tranche documentation",
    ),
    (
        "reports/sparkx_first_tranche_20261026/synthetic_assets/assets.jsonl",
        "owned synthetic asset manifest for first-tranche protocol replay",
    ),
    *[
        (
            f"reports/sparkx_first_tranche_20261026/synthetic_assets/{name}",
            "owned synthetic fixture bytes; not external media",
        )
        for name in (
            "synthetic_scene_0001.json",
            "synthetic_scene_0002.json",
            "synthetic_audio_0001.json",
            "synthetic_audio_0002.json",
            "synthetic_av_0001.json",
            "synthetic_av_0002.json",
        )
    ],
    (
        "configs/real_image_tranche_v1.json",
        "predeclared bounded public-domain/CC0 real-image acquisition protocol",
    ),
    (
        "scripts/prepare_real_image_tranche.py",
        "bounded real-image downloader and offline QA verifier",
    ),
    (
        "tests/test_real_image_tranche.py",
        "real-image admission, failure-atomicity and drift-verification contracts",
    ),
    (
        "docs/experiments/20261026_Real_Image_Tranche.md",
        "real-image rights, provenance, QA and admission report",
    ),
    (
        "reports/real_image_tranche_20261026/attempt03/manifest.json",
        "content-addressed real-image source and derivative manifest",
    ),
    (
        "reports/real_image_tranche_20261026/attempt03/acquisition.json",
        "stable seven-request real-image acquisition receipt",
    ),
    (
        "reports/real_image_tranche_20261026/attempt03/verification.json",
        "stable offline real-image recomputation and drift-check receipt",
    ),
    (
        "reports/real_image_tranche_20261026/attempt03/report.json",
        "combined real-image acquisition and offline verification receipt",
    ),
    (
        "reports/real_image_tranche_20261026/independent_review.json",
        "independent real-image rights, hash and split review",
    ),
    *[
        (
            f"reports/real_image_tranche_20261026/attempt03/assets/{asset_id}.jpg",
            "bounded 960px public-domain/CC0 real-image derivative",
        )
        for asset_id in (
            "apollo17_blue_marble",
            "apollo11_buzz_flag",
            "loc_migrant_mother",
            "rijksmuseum_night_watch",
            "met_neptune_columns",
            "usgs_mameyes_landslide",
            "nara_lbj_oath_1963",
        )
    ],
    (
        "configs/nested_learning_nl0_v1.json",
        "NL0 standalone software/math qualification contract",
    ),
    (
        "src/replica_lab/nested_learning.py",
        "NL0 plastic, recurrence, observation and persistence implementation",
    ),
    (
        "tests/test_nested_learning.py",
        "NL0 numerical, causality, transaction and rollback contracts",
    ),
    (
        "docs/experiments/20261026_Nested_Learning_NL0.md",
        "NL0 implementation, independent review and qualification boundaries",
    ),
    (
        "reports/nested_learning_nl0_20261026/qualification.json",
        "NL0 software/math qualification receipt",
    ),
    (
        "reports/nested_learning_nl0_20261026/tests_final.xml",
        "NL0 final 28-test JUnit evidence",
    ),
    (
        "reports/nested_learning_nl0_20261026/tests.xml",
        "preserved NL0 status-only failed JUnit evidence",
    ),
    (
        "reports/spark_qualification_20261026/selected_runs.json",
        "portable selected-run digest receipt for the official BF16 and dequantized FP8 qualifications",
    ),
    (
        "docs/experiments/20261026_Spark_Precision_and_Provisioning.md",
        "three precision identities, frozen-base admission, storage ledger and H200 provisioning boundary",
    ),
    (
        "reports/spark_qualification_20261026/summary.json",
        "Spark2_5 qualification summary: stages, resources, storage and preserved failures",
    ),
    (
        "reports/spark_qualification_20261026/comparison.json",
        "paired BF16/derivative smoke comparison receipt (no capability claim)",
    ),
    (
        "reports/spark_weight_acquisition_20261026/receipt_01.json",
        "official BF16 and FP8 weight-acquisition and offline re-verification receipt",
    ),
    (
        "configs/spark_qualification_v1.json",
        "frozen source, stage, resource and storage pins for Spark2_5 qualification",
    ),
    (
        "configs/spark_weight_acquisition_v1.json",
        "official snapshot acquisition plan pins and budget",
    ),
    (
        "src/replica_lab/architecture/spark_adapter.py",
        "Spark2_5 frozen-base adapter, optimizer audit and extension-only export",
    ),
    (
        "scripts/qualify_spark2_5.py",
        "Spark2_5 stage runner and fresh-process reload comparison",
    ),
    (
        "scripts/dequantize_spark_fp8.py",
        "offline FP8 validator and single dequantized BF16 derivative writer",
    ),
    (
        "tests/test_spark2_5_adapter.py",
        "synthetic adapter, optimizer, export/reload and runner-contract tests",
    ),
    (
        "tests/test_dequantize_spark_fp8.py",
        "synthetic FP8 inventory, scale, dequantization and identity tests",
    ),
    (
        "reports/spark_qualification_20261025/backend_audit.json",
        "MPS native-FP8 refusal evidence and blocked stage ledger",
    ),
    (
        "configs/sparkx_2_6_4b_final_product_v5.json",
        "v5 final-product declarative profile and inventory reconciliation",
    ),
    (
        "docs/architecture/23_SparkX_2_6_4B_Final_Product_Blueprint.md",
        "v5 final-product blueprint and Spark precision identity",
    ),
    (
        "reports/sparkx_final_product_20261026/README.md",
        "v5 consolidation guide and repository evidence scope",
    ),
    (
        "reports/sparkx_final_product_20261026/verification.json",
        "v5 documentation and arithmetic verification receipt",
    ),
    (
        "reports/sparkx_final_product_20261026/analytical_costs.json",
        "v5 analytical payload and compute ledger",
    ),
    (
        "reports/sparkx_final_product_20261026/requirements_traceability.json",
        "v5 requirement-to-evidence traceability map",
    ),
    (
        "reports/sparkx_final_product_20261026/evidence.json",
        "v5 source and method evidence bindings",
    ),
    (
        "reports/sparkx_final_product_20261026/repository_snapshot.json",
        "historical 40-input snapshot and worktree drift record",
    ),
    (
        "reports/sparkx_final_product_20261026/calculate_costs.py",
        "v5 deterministic analytical cost calculator",
    ),
    (
        "reports/sparkx_final_product_20261026/verify.py",
        "v5 documentation and arithmetic verifier",
    ),
    (
        "reports/sparkx_final_product_20261026/architecture_v4_5_before.txt",
        "pre-edit architecture text snapshot",
    ),
    (
        "reports/sparkx_final_product_20261026/state_contract_before.txt",
        "pre-edit state contract text snapshot",
    ),
    (
        "reports/sparkx_final_product_20261026/architecture_index_before.txt",
        "pre-edit architecture index text snapshot",
    ),
    (
        "reports/runs_cleanup_20261026.json",
        "preserved failed cleanup inspection receipt",
    ),
    (
        "reports/runs_cleanup_20261026_verified.json",
        "verified cleanup journal and evidence preservation receipt",
    ),
]

# Additive current final-product evidence.  This profile is independent of the
# historical 117-input manifest above; it never rewrites that manifest and it
# requires only tracked repository evidence, never a temporary source cache.
FINAL_PRODUCT_INPUT_ROLES: list[tuple[str, str]] = [
    (
        "docs/architecture/23_SparkX_2_6_4B_Final_Product_Blueprint.md",
        "version-5 final-product blueprint with reconciled current evidence",
    ),
    (
        "docs/experiments/20261026_Memory_Acceptance_V4.md",
        "current v4 memory-acceptance protocol and rejected verdict",
    ),
    (
        "docs/experiments/20261026_Memory_Acceptance_Calibration.md",
        "development-only calibration protocol and rejection disposition",
    ),
    (
        "docs/experiments/20261026_Nested_Text_Independent_Continuation.md",
        "exact plastic continuation evidence and 16-expert scope limit",
    ),
    (
        "docs/experiments/20261026_Vision_Learnability_Curve.md",
        "latest vision learnability curve and terminal decision",
    ),
    (
        "configs/sparkx_2_6_4b_final_product_v5.json",
        "v5 declarative profile and inventory reconciliation",
    ),
    (
        "configs/memory_acceptance_v4.json",
        "v4 memory-acceptance metric configuration",
    ),
    (
        "configs/memory_acceptance_calibration_v1.json",
        "development acceptance/address calibration configuration",
    ),
    (
        "configs/nested_text_continuation_comparison_v1.json",
        "16-expert independent continuation protocol",
    ),
    (
        "configs/vision_learnability_curve_v1.json",
        "vision learnability curve protocol",
    ),
    (
        "reports/sparkx_residency_20261026/residency_v2.json",
        "current residency receipt consumed by the analytical calculator",
    ),
    (
        "reports/sparkx_nested_learning_20261026/analytical_costs.json",
        "prior nested-learning payload profile consumed by the calculator",
    ),
    (
        "reports/sparkx_final_product_20261026/README.md",
        "current consolidation guide with reconciled evidence",
    ),
    (
        "reports/sparkx_final_product_20261026/verify.py",
        "portable tracked-evidence verifier",
    ),
    (
        "reports/sparkx_final_product_20261026/analytical_costs.json",
        "v5 analytical payload and compute ledger",
    ),
    (
        "reports/sparkx_final_product_20261026/calculate_costs.py",
        "deterministic analytical calculator",
    ),
    (
        "reports/sparkx_final_product_20261026/requirements_traceability.json",
        "requirement-to-evidence traceability map",
    ),
    (
        "reports/sparkx_final_product_20261026/evidence.json",
        "public-source and method evidence bindings",
    ),
    (
        "reports/sparkx_final_product_20261026/portable_evidence.json",
        "portable required-evidence and source-pin inventory",
    ),
    (
        "reports/sparkx_final_product_20261026/reviewed_input_changes.json",
        "explicitly reviewed snapshot and historical-manifest changes",
    ),
    (
        "reports/sparkx_final_product_20261026/repository_snapshot.json",
        "historical 40-input snapshot and worktree drift record",
    ),
    (
        "reports/sparkx_final_product_20261026/verification.json",
        "historical v5 verification receipt, preserved",
    ),
    (
        "reports/memory_acceptance_20261026/causal_fresh_restart02_acceptance_v4.json",
        "sole current authoritative v4 memory-acceptance receipt",
    ),
    (
        "reports/memory_acceptance_20261026/receipt_manifest_v4.restart02.json",
        "v4 receipt manifest selecting the current receipt",
    ),
    (
        "reports/memory_acceptance_20261026/causal_fresh_restart02_acceptance_v3.json",
        "historical v3 acceptable receipt, never accepted as v4",
    ),
    (
        "reports/memory_acceptance_20261026/receipt_manifest_v3.json",
        "historical v3 receipt manifest",
    ),
    (
        "reports/memory_acceptance_20261026/calibration_dev01.json",
        "development-only calibration rejection receipt",
    ),
    (
        "reports/nested_text_continuation_comparison_20261026/comparison.json",
        "exact independent plastic continuation comparison",
    ),
    (
        "reports/nested_text_continuation_comparison_20261026/continuous_a.guard.json",
        "continuous_a guard receipt",
    ),
    (
        "reports/nested_text_continuation_comparison_20261026/continuous_b.guard.json",
        "continuous_b guard receipt",
    ),
    (
        "reports/nested_text_continuation_comparison_20261026/resume.guard.json",
        "resume guard receipt",
    ),
    (
        "reports/vision_learnability_curve_v1_20261026/summary.json",
        "latest vision curve summary and terminal decision",
    ),
    (
        "reports/vision_learnability_curve_v1_20261026_attempt04/guard.json",
        "completed vision curve attempt-04 guard receipt",
    ),
    (
        "reports/vision_learnability_curve_v1_20261026/guard.json",
        "historical failed vision curve guard receipt, preserved",
    ),
    (
        "reports/final_verification_20261027/summary.json",
        "final verification and current clean-clone summary",
    ),
    (
        "reports/final_verification_20261027/tests.xml",
        "final workspace JUnit receipt with preserved historical failures",
    ),
    (
        "reports/final_verification_20261027/clean_export/tests.xml",
        "current clean-clone JUnit receipt",
    ),
    (
        "reports/clean_export_20261026_final/summary.json",
        "historical clean-clone verification at f25670a",
    ),
    (
        "reports/clean_export_20261026_final/tests_final.xml",
        "historical clean-clone final JUnit receipt",
    ),
    (
        "reports/sparkx_first_tranche_20261026/research_bundle_manifest.json",
        "reviewed current 117-input first-tranche manifest; the pre-v5.1-residency bytes are preserved separately as research_bundle_manifest_pre_sparkx_v5_1_residency.json",
    ),
    (
        "reports/spark_preparation_20261025/run_01/assets/official_bf16_control/config.json",
        "tracked pinned Spark configuration",
    ),
    (
        "reports/spark_preparation_20261025/run_01/assets/official_bf16_control/modeling_spark.py",
        "tracked pinned Spark modeling code",
    ),
    (
        "docs/architecture/18_MiniCPM5_Vision_Training_and_Continual_Learning.md",
        "v5.1 MiniCPM direction pointer with historical V0-V4 material",
    ),
    (
        "docs/architecture/21_SparkX_2_6_4B_Consolidated_Architecture_and_MiniCPM_Validation.md",
        "v5.1-reviewed revision-4.5 consolidated plan with historical receipts",
    ),
    (
        "docs/architecture/22_State_Mechanism_and_RSI_Contracts.md",
        "v5.1-reviewed state, transaction and RSI contracts with pending Mkv namespace",
    ),
    (
        "docs/architecture/24_SparkX_2_6_v5_1_Refinement_Plan.md",
        "v5.1 refinement, residency inventory and staged learning plan",
    ),
    (
        "docs/architecture/25_MiniCPM5_Direction_Correction_and_Learning_Capacity_Plan.md",
        "v5.1 MiniCPM direction-correction and capacity program",
    ),
    (
        "docs/architecture/README.md",
        "v5.1 architecture index",
    ),
    (
        "docs/research/sparkx_nested_learning_20261026/README.md",
        "v5.1 nested-learning direction pointer",
    ),
    (
        "configs/sparkx_2_6_v5_1_direction_design.json",
        "v5.1 declarative design, residency inventory and staging gates",
    ),
    (
        "reports/sparkx_v5_1_review_20261027/README.md",
        "v5.1 review report and verification scope",
    ),
    (
        "reports/sparkx_v5_1_review_20261027/migration_manifest.json",
        "v5.1 reviewed old/new identity and arithmetic migration manifest",
    ),
    (
        "reports/sparkx_v5_1_review_20261027/native_gates.json",
        "v5.1 unresolved native-gate register",
    ),
    (
        "reports/sparkx_v5_1_review_20261027/deliverables.json",
        "v5.1 authored-output hash pins",
    ),
    (
        "reports/sparkx_v5_1_review_20261027/analytical_capacity.json",
        "v5.1 analytical capacity and residency ledger",
    ),
    (
        "reports/sparkx_v5_1_review_20261027/calculate_capacity.py",
        "v5.1 analytical calculator",
    ),
    (
        "reports/sparkx_v5_1_review_20261027/verify.py",
        "v5.1 portable document, arithmetic and migration verifier",
    ),
    (
        "reports/sparkx_v5_1_review_20261027/repository_snapshot.json",
        "v5.1 preserved-input snapshot",
    ),
    (
        "reports/sparkx_v5_1_review_20261027/evidence.json",
        "v5.1 primary-source manifest",
    ),
    (
        "reports/sparkx_v5_1_review_20261027/claim_dispositions.json",
        "v5.1 C1-C16 disposition register",
    ),
    (
        "reports/sparkx_v5_1_review_20261027/attachment_reference_inventory.json",
        "v5.1 attachment reference inventory",
    ),
    (
        "reports/sparkx_v5_1_review_20261027/inputs/attachment_v5_1_draft.txt",
        "preserved v5.1 draft source material",
    ),
    (
        "reports/sparkx_v5_1_review_20261027/inputs/00_23_SparkX_2_6_4B_Final_Product_Blueprint.txt",
        "preserved pre-change blueprint v5.0 snapshot",
    ),
    (
        "reports/sparkx_v5_1_review_20261027/inputs/01_18_MiniCPM5_Vision_Training_and_Continual_Learning.txt",
        "preserved pre-change MiniCPM vision snapshot",
    ),
    (
        "reports/sparkx_v5_1_review_20261027/inputs/02_22_State_Mechanism_and_RSI_Contracts.txt",
        "preserved pre-change state/RSI snapshot",
    ),
    (
        "reports/sparkx_v5_1_review_20261027/inputs/03_README.txt",
        "preserved pre-change architecture index snapshot",
    ),
    (
        "reports/sparkx_v5_1_review_20261027/inputs/04_21_SparkX_2_6_4B_Consolidated_Architecture_and_MiniCPM_Validation.txt",
        "preserved pre-change revision-4.5 snapshot",
    ),
    (
        "reports/sparkx_v5_1_review_20261027/inputs/05_README.txt",
        "preserved pre-change nested-learning guide snapshot",
    ),
    (
        "reports/sparkx_v5_1_review_20261027/inputs/06_20261027_Remediation_and_Next_Steps.txt",
        "preserved remediation experiment snapshot",
    ),
    (
        "reports/sparkx_v5_1_review_20261027/inputs/07_20261026_Memory_Bank_Readout_Hypothesis.txt",
        "preserved exact-bank readout snapshot",
    ),
    (
        "reports/sparkx_v5_1_review_20261027/inputs/08_20261026_Nested_Text_Response_Diagnostic_Budget02.txt",
        "preserved response-diagnostic snapshot",
    ),
    (
        "reports/sparkx_v5_1_review_20261027/inputs/09_20261026_Nested_Text_Independent_Continuation.txt",
        "preserved independent-continuation snapshot",
    ),
    (
        "reports/sparkx_v5_1_review_20261027/inputs/10_sparkx_2_6_4b_final_product_v5.txt",
        "preserved historical v5 configuration snapshot",
    ),
    (
        "tests/test_sparkx_v5_1_review.py",
        "v5.1 clean-checkout and tamper tests",
    ),
    (
        "tests/test_sparkx_final_product.py",
        "portable verifier, evidence-tamper and manifest tests",
    ),
    (
        "src/replica_lab/architecture/spine.py",
        "typed Spark spine packets, causal readers and E/P memory provenance",
    ),
    (
        "src/replica_lab/architecture/spine_checkpoint.py",
        "strict extension-only spine and paired E/P checkpoint recovery",
    ),
    (
        "src/replica_lab/architecture/spine_memory.py",
        "bounded evidence bank, selected P reader and source-free intervention",
    ),
    (
        "src/replica_lab/model_identity.py",
        "content-based model identity used by native spine export and reload",
    ),
    (
        "tests/test_spine.py",
        "CPU spine native influence, evidence, P and fresh-restore contracts",
    ),
    (
        "docs/experiments/20261028_Spark_Spine_Memory_Integration.md",
        "implemented spine scope, verification and explicit native limits",
    ),
    (
        "docs/experiments/20261028_Spark_Text_Memory_Milestone.md",
        "Spark text-memory implementation, negative behavioral evidence and next diagnosis",
    ),
    (
        "src/replica_lab/architecture/spine_text.py",
        "versioned text producer, causal query adapter and response supervision",
    ),
    (
        "src/replica_lab/spark_text_memory_runner.py",
        "bounded Spark text fit, development, restart and work-accounting runner",
    ),
    (
        "scripts/run_spark_text_memory_v1.py",
        "Spark text-memory protocol command entry point",
    ),
    (
        "scripts/preflight_spark_text_memory_v1.py",
        "Spark text-memory review and backend-authorization preflight",
    ),
    (
        "scripts/run_capacity_preflight.py",
        "exclusive native capacity watchdog with Spark text runner registration",
    ),
    (
        "configs/spark_text_memory_protocol_v1.json",
        "versioned Spark text-memory protocol and native execution boundaries",
    ),
    (
        "configs/architecture_implementation_status_v1.json",
        "machine-readable Spark-first construction order and deferred gates",
    ),
    (
        "docs/architecture/26_Spark_Text_Memory_Milestone.md",
        "active Spark text-memory architecture and roadmap",
    ),
    (
        "tests/test_spine_text.py",
        "tiny text producer, combined E+P and response-gradient contracts",
    ),
    (
        "tests/test_spark_text_memory_protocol.py",
        "protocol work derivation and backend authorization tests",
    ),
    (
        "tests/test_spark_text_memory_runner.py",
        "CPU fit, development, restart, P update and publication tests",
    ),
    (
        "data/spark_text_memory_v1/manifest.json",
        "hash-bound Spark text fixture manifest and sealed-final closure",
    ),
    (
        "data/spark_text_memory_v1/observations.jsonl",
        "owned text observations and durable provenance",
    ),
    (
        "data/spark_text_memory_v1/fit_events.jsonl",
        "chronological fit teaching events",
    ),
    (
        "data/spark_text_memory_v1/development.jsonl",
        "independent development query and retention cases",
    ),
    (
        "data/spark_text_memory_v1/development_labels.jsonl",
        "development labels gated by fit receipt",
    ),
    (
        "data/spark_text_memory_v1/parity_requests.jsonl",
        "pre/post-restart parity requests",
    ),
    (
        "data/spark_text_memory_v1/procedures.json",
        "verified reusable procedures and failure lessons",
    ),
    (
        "data/spark_text_memory_v1/tasks.json",
        "task subgoals and independently observed completion",
    ),
    (
        "data/spark_text_memory_v1/worlds.json",
        "independent world and rule-family split lineage",
    ),
]

FINAL_PRODUCT_CLEAN_EXPORT_LINKS = [
    {
        "path": "reports/final_verification_20261027/summary.json",
        "role": "current final verification and clean-clone summary",
        "status": "current",
    },
    {
        "path": "reports/final_verification_20261027/clean_export/tests.xml",
        "role": "current clean-clone JUnit receipt",
        "status": "current",
    },
    {
        "path": "reports/clean_export_20261026_final/summary.json",
        "role": "historical clean-clone summary at f25670a",
        "status": "historical",
    },
    {
        "path": "reports/clean_export_20261026_final/tests_final.xml",
        "role": "historical clean-clone final JUnit receipt",
        "status": "historical",
    },
]

# Additive 28 October persistent-learner review; historical profile unchanged.
FINAL_PRODUCT_INPUT_ROLES.extend([
    ('docs/architecture/27_Spark_X2_6_4B_Architecture_Research_Review.md', 'Spark native source and geometry research audit'),
    ('docs/architecture/28_Spark_Persistent_Learner_and_Tiered_Memory.md', 'implemented v2 architecture and native/learned boundaries'),
    ('docs/architecture/16_Loop_Agent_Instructions.md', 'current bounded worker instructions'),
    ('configs/spark_persistent_learner_tiered_design_v1.json', 'design-only expanded memory profiles and tier limits'),
    ('reports/spark_persistent_learner_review_20261028/README.md', 'independent review and research disposition'),
    ('reports/spark_persistent_learner_review_20261028/baseline.json', 'reviewed source and prior-document identities'),
    ('reports/spark_persistent_learner_review_20261028/review_checks.json', 'independent focused review and baseline verification summary'),
    ('reports/spark_persistent_learner_review_20261028/focused_snapshot.xml', 'preserved focused snapshot successes and failures'),
    ('reports/spark_persistent_learner_review_20261028/focused_snapshot_output.txt', 'raw focused review output'),
    ('reports/spark_persistent_learner_review_20261028/prefix_probe.json', 'tiny non-native decoder prefix perturbation'),
    ('reports/spark_persistent_learner_review_20261028/calculate_capacity.py', 'model-free memory capacity arithmetic'),
    ('reports/spark_persistent_learner_review_20261028/capacity_arithmetic.json', 'expanded profile arithmetic without native fit claim'),
    ('configs/spark_persistent_learner_protocol_v2.json', 'separately versioned P1 direct-CE and tiered protocol'),
    ('reports/spark_persistent_learner_v2/README.md', 'v2 implementation evidence and native runbook'),
    ('reports/spark_persistent_learner_v2/review02.json', 'preparation-only v2 source and budget review'),
    ('docs/experiments/20261029_Spark_Scoped_Readout_and_Recovery.md', 'repaired scoped-readout evidence, failure analysis and native/learned boundaries'),
    ('reports/spark_persistent_learner_v2/native_backend_plan.md', 'exact native-FP8 blocker and distinct derivative backend plan'),
    ('reports/spark_persistent_learner_v2/native_experiment_packet.json', 'bounded P1 native experiment packet with full-context and exact-I/O budgets'),
    ('reports/spark_persistent_learner_v2/review_final02.json', 'repaired protocol no-launch admission review'),
    ('reports/spark_persistent_learner_v2/readout_pressure04.json', 'real-file cache pressure, cold revalidation and correction provenance receipt'),
    ('reports/spark_persistent_learner_v2/control_baseline.json', 'pre-fix trained-reader leakage reproduction'),
    ('reports/spark_persistent_learner_v2/control_views_after.json', 'post-fix isolated arm reader and P-state evidence'),
    ('reports/spark_persistent_learner_v2/recovery_baseline.json', 'recovery reproduction baseline snapshot'),
])

FINAL_PRODUCT_INPUT_ROLES.extend([
    ('docs/experiments/20261029_Spark_Memory_Native_Readout03.md', 'completed placement-corrected S0-S1 outcome, measured limits and gated successors'),
    ('reports/intrinsic_memory_v3/native_readout03/result.json', 'third authorized native attempt: competence and useful-readout outcomes separated'),
    ('reports/intrinsic_memory_v3/native_readout03/cpu_successor_verification.json', 'clean CPU wall-supervisor and explicit-EOS candidate checks; unused by native attempt'),
    ('configs/contextual_completion_objective_protocol_v1.json', 'CPU completion-objective candidate with separate EOS and decoder-stop identities'),
    ('docs/experiments/20261029_Spark_Memory_Integrated_Access.md', 'combined contextual producer/store/reader CPU integration and native placement outcome'),
    ('docs/experiments/20261029_Spark_Memory_Formation_and_Replica_Successors.md', 'formation and tier-valid Replica cognitive-core successors; not implemented learning claims'),
    ('configs/contextual_memory_layer_training_protocol_v4.json', 'separate combined 2/9/36 contextual training and complete checkpoint contract'),
    ('configs/intrinsic_memory_density_protocol_v5.json', 'per-arm density feasibility preparation and independent authorization budgets'),
    ('reports/intrinsic_memory_density_v5/native_comparison_packet.json', 'non-launching 2/9/36 feasibility review packet'),
    ('reports/intrinsic_memory_density_v5/README.md', 'density controls and native runner admission gaps'),
    ('reports/intrinsic_memory_v3/native_readout02/result.json', 'second authorized attempt loaded model but failed first calibration forward placement'),
    ('configs/spark_persistent_learner_derivative_bf16_placement_protocol_v2.json', 'separate frozen-base device placement execution binding'),
    ('reports/intrinsic_memory_v3/native_experiment_s0_s1_placement_v3.json', 'executed placement-corrected S0-S1 packet; single-attempt authorization consumed'),
    ('docs/experiments/20261029_Spark_Memory_Native_Environment.md', 'authorized native load failure and corrected environment-bound successor'),
    ('configs/spark_intrinsic_execution_environment_v1.json', 'qualified interpreter dependency and critical runtime file bindings'),
    ('reports/intrinsic_memory_v3/native_readout01/result.json', 'first authorized S0-S1 attempt stopped before competence'),
    ('reports/intrinsic_memory_v3/native_experiment_s0_s1_environment_v2.json', 'historical environment-bound attempt02 command; authorization consumed'),
    ('docs/experiments/20261029_Spark_Memory_Layer_Access.md', 'all-layer intrinsic memory target and separately versioned CPU candidate'),
    ('configs/intrinsic_memory_layer_access_protocol_v4.json', 'CPU 2/9/36 layer-access candidate bindings; native density unqualified'),
    ('reports/intrinsic_memory_layout_v4/analytical_costs.json', 'actual reader formula and meta tensor parameter inventory; not native fit'),
    ('reports/intrinsic_memory_layout_v4/README.md', 'configurable intrinsic reader CPU evidence and native boundaries'),
    ('docs/experiments/20261029_Spark_Memory_S0_S1.md', 'current Spark-only foundation and intrinsic recall experiment with separate successor gates'),
    ('docs/research/spark_memory_20261029/Research_Crosscheck.md', 'dated primary-source mechanism and schedule distinctions'),
    ('reports/intrinsic_memory_v3/evidence_ledger_20261029.json', 'dated intrinsic-memory implementation and measured evidence status'),
    ('reports/intrinsic_memory_v3/README.md', 'intrinsic memory ownership, measured separation and current native packet'),
    ('configs/intrinsic_memory_readout_protocol_v3.json', 'intrinsic recall independent of the external spine'),
    ('reports/contextual_memory_v2/README.md', 'contextual CPU evidence and gated native experiment runbook'),
    ('configs/contextual_readout_protocol_v2.json', 'fixed contextual readout experiment and pass criteria'),
    ('configs/spark_competence_protocol_v1.json', 'separate frozen-original pre-fit competence gate'),
    ('configs/spark_derivative_execution_manifest_v1.json', 'actual model/configuration executable provenance'),
    ('configs/spark_persistent_learner_derivative_bf16_protocol_v1.json', 'distinct derivative execution and authorization binding'),
])

PROFILES = {
    "historical": {
        "manifest": DEFAULT_MANIFEST,
        "roles": INPUT_ROLES,
        "format": FORMAT,
        "profile": "historical",
        "clean_export_links": [],
        "generated_output": None,
    },
    "final-product": {
        "manifest": DEFAULT_FINAL_PRODUCT_MANIFEST,
        "roles": FINAL_PRODUCT_INPUT_ROLES,
        "format": FINAL_PRODUCT_FORMAT,
        "profile": "final-product",
        "clean_export_links": FINAL_PRODUCT_CLEAN_EXPORT_LINKS,
        "generated_output": FINAL_PRODUCT_GENERATED_OUTPUT,
    },
}

# Large payloads that must never be copied into this bundle. They are either
# absent or separately provisioned; their hashes are not required here.
NOT_COPIED = [
    "checkpoints/** (model weights)",
    "data/** (bulk media, annotations and caches)",
    "runs/** (ignored native run evidence)",
    "reports/**/*.safetensors",
    "**/*.pt (fresh-process reload tensors)",
]


class BundleError(RuntimeError):
    """A required input is missing, unreadable, untracked or hash-mismatched."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_tracked(root: Path, relative: str) -> bool:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "ls-files", "--error-unmatch", relative],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except OSError:
        return False
    return result.returncode == 0


def _git_worktree_matches_index(root: Path, relative: str) -> bool:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "diff", "--quiet", "--", relative],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except OSError:
        return False
    return result.returncode == 0


def _entry(root: Path, relative: str, role: str, allow_untracked: bool) -> dict:
    path = root / relative
    if not path.is_file():
        raise BundleError(f"missing required input: {relative}")
    try:
        digest = sha256_file(path)
        size = path.stat().st_size
    except OSError as error:  # pragma: no cover - permission/IO failure
        raise BundleError(f"unreadable required input: {relative}: {error}") from error
    tracked = _git_tracked(root, relative)
    matches_index = _git_worktree_matches_index(root, relative)
    if not tracked and not allow_untracked:
        raise BundleError(
            f"untracked required input: {relative} (commit it, or use "
            "--allow-untracked for a development-only manifest)"
        )
    if tracked and not matches_index and not allow_untracked:
        raise BundleError(
            f"unstaged required input change: {relative} (stage it for the "
            "intended commit or preserve it outside the clean bundle)"
        )
    return {
        "path": relative,
        "role": role,
        "bytes": size,
        "sha256": digest,
        "tracked_in_git": tracked,
        "worktree_matches_index": matches_index,
    }


def _canonical_digest(entries: list[dict]) -> str:
    canonical = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def build_manifest(
    root: Path,
    allow_untracked: bool = False,
    *,
    profile: str = "historical",
) -> dict:
    root = Path(root)
    if profile not in PROFILES:
        raise BundleError(f"unknown bundle profile: {profile}")
    settings = PROFILES[profile]
    roles = settings["roles"]
    entries = sorted(
        (_entry(root, relative, role, allow_untracked) for relative, role in roles),
        key=lambda entry: entry["path"],
    )
    untracked = [entry["path"] for entry in entries if not entry["tracked_in_git"]]
    dirty = [entry["path"] for entry in entries if not entry["worktree_matches_index"]]
    by_path = {entry["path"]: entry for entry in entries}
    clean_export_links = []
    for link in settings["clean_export_links"]:
        entry = by_path[link["path"]]
        clean_export_links.append(
            {
                "path": entry["path"],
                "role": link["role"],
                "status": link["status"],
                "bytes": entry["bytes"],
                "sha256": entry["sha256"],
            }
        )
    manifest = {
        "format": settings["format"],
        "kind": "hashed_manifest_not_an_asset_copy",
        "generator": "scripts/prepare_sparkx_research_bundle.py",
        "root_policy": (
            "manifest entry paths are repository-relative; historical receipt "
            "contents may quote the origin workspace and are not rewritten"
        ),
        "input_count": len(entries),
        "total_input_bytes": sum(entry["bytes"] for entry in entries),
        "bundle_digest": _canonical_digest(entries),
        "clean_checkout_ready": not untracked and not dirty,
        "untracked_inputs": untracked,
        "unstaged_modified_inputs": dirty,
        "tracking_enforced": not allow_untracked,
        "inputs": entries,
        "not_copied": NOT_COPIED,
        "provisioning": {
            "write_command": (
                "python3 scripts/prepare_sparkx_research_bundle.py --write"
                if profile == "historical"
                else (
                    "python3 scripts/prepare_sparkx_research_bundle.py "
                    f"--profile {profile} --write"
                )
            ),
            "check_command": (
                "python3 scripts/prepare_sparkx_research_bundle.py --check"
                if profile == "historical"
                else (
                    "python3 scripts/prepare_sparkx_research_bundle.py "
                    f"--profile {profile} --check"
                )
            ),

            "development_override": (
                "python3 scripts/prepare_sparkx_research_bundle.py --check "
                "--allow-untracked"
            ),
            "named_skip": (
                "tests that require this bundle should call "
                "pytest.importorskip or skip with the --check command in the reason; "
                "do not claim clean replay from an unprovisioned checkout"
            ),
            "network_required": False,
            "large_payload_copied": False,
        },
    }
    if profile != "historical":
        manifest["profile"] = settings["profile"]
        manifest["clean_export_links"] = clean_export_links
        manifest["generated_output"] = settings["generated_output"]
    return manifest


def verify_manifest(
    root: Path,
    manifest: dict,
    *,
    allow_untracked: bool = False,
    require_tracking: bool = True,
    profile: str | None = None,
) -> dict:
    root = Path(root)
    effective_profile = manifest.get("profile", profile or "historical")
    if effective_profile not in PROFILES:
        raise BundleError(f"unknown bundle profile: {effective_profile}")
    settings = PROFILES[effective_profile]
    if manifest.get("format") != settings["format"]:
        raise BundleError(f"unsupported manifest format: {manifest.get('format')!r}")
    declared = manifest.get("inputs")
    if not isinstance(declared, list) or not declared:
        raise BundleError("manifest has no inputs")

    failures: list[dict] = []
    entries: list[dict] = []
    reviewed_current_updates: list[dict] = []
    seen_paths = set()
    expected_roles = dict(settings["roles"])
    reviewed_updates: dict[str, dict] = {}
    if effective_profile == "historical":
        overlay_path = (
            root
            / "reports/sparkx_final_product_20261026/reviewed_input_changes.json"
        )
        if overlay_path.is_file():
            try:
                overlay = json.loads(overlay_path.read_text(encoding="utf-8"))
                reviewed_updates = {
                    update["path"]: update
                    for update in overlay.get("historical_manifest_updates", [])
                    if isinstance(update, dict) and update.get("path")
                }
            except (json.JSONDecodeError, OSError):
                reviewed_updates = {}
    for record in declared:
        relative = record.get("path") if isinstance(record, dict) else None
        if not isinstance(relative, str) or not relative:
            failures.append({"path": relative, "reason": "invalid_manifest_entry"})
            continue
        if relative.startswith("/") or ".." in Path(relative).parts:
            failures.append({"path": relative, "reason": "non_relative_path"})
            continue
        if relative in seen_paths:
            failures.append({"path": relative, "reason": "duplicate_manifest_path"})
            continue
        if relative not in expected_roles:
            failures.append({"path": relative, "reason": "unexpected_manifest_input"})
            continue
        if record.get("role") != expected_roles[relative]:
            failures.append({"path": relative, "reason": "role_mismatch"})
            continue
        seen_paths.add(relative)
        path = root / relative
        if not path.is_file():
            failures.append({"path": relative, "reason": "missing"})
            continue
        size = path.stat().st_size
        digest = sha256_file(path)
        tracked = _git_tracked(root, relative)
        matches_index = _git_worktree_matches_index(root, relative)
        reviewed = reviewed_updates.get(relative)
        reviewed_ok = (
            reviewed is not None
            and reviewed.get("prior_sha256") == record.get("sha256")
            and reviewed.get("current_sha256") == digest
        )
        if size != record.get("bytes") and not reviewed_ok:
            failures.append(
                {"path": relative, "reason": "size_mismatch", "expected": record.get("bytes"), "actual": size}
            )
        if digest != record.get("sha256"):
            if reviewed_ok:
                reviewed_current_updates.append(
                    {
                        "path": relative,
                        "prior_sha256": record.get("sha256"),
                        "current_sha256": digest,
                        "reviewed_in": "reports/sparkx_final_product_20261026/reviewed_input_changes.json",
                    }
                )
            else:
                failures.append(
                    {"path": relative, "reason": "sha256_mismatch", "expected": record.get("sha256"), "actual": digest}
                )
        if require_tracking and not allow_untracked and not tracked:
            failures.append({"path": relative, "reason": "untracked_in_git"})
        if require_tracking and not allow_untracked and not matches_index:
            failures.append({"path": relative, "reason": "unstaged_modification"})
        entries.append(
            {
                "path": relative,
                "role": record.get("role"),
                "bytes": size,
                "sha256": digest,
                "tracked_in_git": tracked,
                "worktree_matches_index": matches_index,
            }
        )

    if require_tracking and not allow_untracked and not manifest.get("clean_checkout_ready"):
        failures.append(
            {
                "path": "<bundle>",
                "reason": "manifest_not_clean_checkout_ready",
                "detail": "regenerate the manifest after committing every listed input",
            }
        )

    entries.sort(key=lambda entry: entry["path"])
    digest_entries = [
        {
            "path": record.get("path"),
            "role": record.get("role"),
            "bytes": record.get("bytes"),
            "sha256": record.get("sha256"),
            "tracked_in_git": record.get("tracked_in_git"),
            "worktree_matches_index": record.get("worktree_matches_index"),
        }
        for record in declared
        if isinstance(record, dict) and isinstance(record.get("path"), str)
    ]
    recomputed = _canonical_digest(
        sorted(digest_entries, key=lambda entry: entry["path"])
    )
    if recomputed != manifest.get("bundle_digest"):
        failures.append(
            {
                "path": "<bundle>",
                "reason": "bundle_digest_mismatch",
                "expected": manifest.get("bundle_digest"),
                "actual": recomputed,
            }
        )

    missing = sorted(
        relative for relative, _role in settings["roles"] if relative not in seen_paths
    )
    if missing:
        failures.append({"path": "<bundle>", "reason": "missing_manifest_inputs", "paths": missing})

    declared_links = manifest.get("clean_export_links", [])
    if effective_profile == "final-product":
        expected_links = {link["path"]: link for link in settings["clean_export_links"]}
        entries_by_path = {entry["path"]: entry for entry in entries}
        seen_links = set()
        for link in declared_links:
            relative = link.get("path") if isinstance(link, dict) else None
            expected = expected_links.get(relative)
            if expected is None:
                failures.append(
                    {"path": relative, "reason": "unexpected_clean_export_link"}
                )
                continue
            seen_links.add(relative)
            entry = entries_by_path.get(relative)
            if entry is None:
                failures.append(
                    {"path": relative, "reason": "clean_export_link_not_in_inputs"}
                )
                continue
            if (
                link.get("sha256") != entry["sha256"]
                or link.get("bytes") != entry["bytes"]
                or link.get("role") != expected["role"]
                or link.get("status") != expected["status"]
            ):
                failures.append(
                    {"path": relative, "reason": "clean_export_link_mismatch"}
                )
        for relative in sorted(set(expected_links) - seen_links):
            failures.append(
                {"path": relative, "reason": "missing_clean_export_link"}
            )
        generated = settings.get("generated_output")
        if generated and generated in entries_by_path:
            failures.append(
                {"path": generated, "reason": "generated_receipt_must_not_be_hashed"}
            )

    return {
        "format": "replica.sparkx_research_bundle_verification.v1",
        "profile": effective_profile,
        "verified": not failures,
        "clean_checkout_ready": not failures and bool(manifest.get("clean_checkout_ready")),
        "tracking_enforced": require_tracking and not allow_untracked,
        "input_count": len(entries),
        "total_input_bytes": sum(entry["bytes"] for entry in entries),
        "bundle_digest": recomputed,
        "reviewed_current_updates": reviewed_current_updates,
        "failures": failures,
        "network_used": False,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILES),
        default="historical",
        help="historical first-tranche profile (default) or additive final-product profile",
    )
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--write", action="store_true", help="write the manifest")
    action.add_argument("--check", action="store_true", help="verify the manifest")
    parser.add_argument(
        "--allow-untracked",
        action="store_true",
        help=(
            "development override: accept inputs that are not tracked by git. "
            "The manifest records clean_checkout_ready=false. Never use this to "
            "claim a clean-checkout replay."
        ),
    )
    args = parser.parse_args(argv)
    root = args.root.resolve()
    manifest_path = args.manifest or (root / PROFILES[args.profile]["manifest"])

    try:
        if args.write:
            manifest = build_manifest(
                root,
                allow_untracked=args.allow_untracked,
                profile=args.profile,
            )
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            print(
                f"Wrote {manifest_path}: profile={args.profile}, "
                f"{manifest['input_count']} inputs, "
                f"digest {manifest['bundle_digest']}, "
                f"clean_checkout_ready={manifest['clean_checkout_ready']}"
            )
            return 0

        if not manifest_path.is_file():
            raise BundleError(f"manifest missing: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        report = verify_manifest(
            root,
            manifest,
            allow_untracked=args.allow_untracked,
            require_tracking=True,
            profile=args.profile,
        )
    except (BundleError, json.JSONDecodeError, OSError) as error:
        print(f"bundle error: {error}", file=sys.stderr)
        return 2

    if not report["verified"]:
        print(json.dumps(report, indent=2))
        print(
            "bundle verification failed; fix the named inputs or, for development "
            "only, re-run with --allow-untracked",
            file=sys.stderr,
        )
        return 1

    print(
        f"Bundle verified: {report['input_count']} inputs, "
        f"{report['total_input_bytes']} bytes, "
        f"digest {report['bundle_digest']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

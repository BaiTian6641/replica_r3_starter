#!/usr/bin/env python3
"""Analytical residency lower bounds for full-scope Spark X2.6 4B training.

Standard library only. Every total is recomputed from the pinned official BF16
safetensors tensor index; no weights, assets, tokenizers, model code or native
run is required, downloaded or executed. All byte figures are analytical lower
bounds over the declared tensors and declared per-parameter byte rates. They are
not a native memory fit, a measured allocator peak, a throughput prediction, a
convergence claim or an admission decision.

Usage:  python3 reports/sparkx_residency_20261026/calculate_residency.py
Writes: reports/sparkx_residency_20261026/residency.json
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent

GIB = 2**30
MIB = 2**20

HEADER_PATH = "reports/spark_preparation_20261025/run_01/headers/official_bf16_control/tensor_index.json"
MINICPM_RUN_PATH = "runs/fused_adamw_20261025/sustained01/run_report.json"
MINICPM_GUARD_PATH = "reports/fused_adamw_20261025/sustained01.guard.json"
ARCHITECTURE_DOC_PATH = (
    "docs/architecture/21_SparkX_2_6_4B_Consolidated_Architecture_and_MiniCPM_Validation.md"
)
PRODUCT_PLAN_PATH = "configs/sparkx_2_6_4b_product_plan_v4.json"
SIGLIP_MANIFEST_PATH = (
    "checkpoints/siglip2_base_patch16_224_source/source_manifest.json"
)
CALCULATOR_PATH = "reports/sparkx_residency_20261026/calculate_residency.py"
RECEIPT_PATH = "reports/sparkx_residency_20261026/residency.json"

# Repository guard policy: 50 GiB sampled driver limit, unchanged since 23 Oct 2026.
# The committed receipt below is used to supply the policy value; it is asserted
# to equal this constant rather than trusted blindly.
DRIVER_GUARD_GIB = 50

DTYPE_BYTES = {"BF16": 2, "FP16": 2, "F32": 4, "FP32": 4, "F8_E4M3": 1, "F8_E5M2": 1}

EXCLUSIONS = [
    "activations, attention workspace and retained graphs, including activation checkpointing peaks",
    "full logits and the vocabulary projection over 131,072 rows",
    "gradient clipping reduction buffers and temporary copies",
    "optimizer temporaries: fused-kernel scratch, foreach lists, step/scalar buffers, block scales and dequantization buffers",
    "allocator, caching-allocator fragmentation, compiler and MPS driver overhead",
    "data loaders, raw media, replay buffers, KV cache and long-context state",
    "per-session or per-worker replication of any of the above",
    "checkpoint serialization peaks and export copies",
]

COMBINED_EXCLUSIONS = [
    "spatial/temporal bridges and the SigLIP2 projector/resampler: width 768 -> 1024 and 1024 -> 2560 stages are not counted here",
    "lexical/core encoder, vocabulary extensions, native memory machinery and spine attachments",
    "fast/expert plastic state from the revision 4.2 and 4.3 ledgers, including update staging and replay",
    "encoder streaming transients, feature queues and per-camera/per-microphone session state",
    "every item excluded from the backbone-only figures, which applies unchanged",
]

HISTORICAL_NOTE = (
    "Historical/unselected. These rows are the expressions from the earlier "
    "reports/sparkx_2_6_product_refinement_20261025/estimate_costs.py inventory. "
    "The selected optimizer is now NativeFusedAdamW (torch.optim.AdamW(fused=True)); "
    "Muon/int8 was never measured on Spark and is not a candidate here. They are "
    "reported only so the superseded numbers stay traceable."
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(rel: str) -> dict:
    path = ROOT / rel
    if not path.is_file():
        raise SystemExit(f"required evidence file missing: {rel}")
    return json.loads(path.read_text())


def elements(shape):
    return math.prod(shape)


def audit_header(header: dict) -> dict:
    tensors = header["tensors"]
    shards = header["shard_headers"]

    names = [t["name"] for t in tensors]
    duplicates = sorted({n for n, c in Counter(names).items() if c > 1})
    dtypes = Counter(t["dtype"] for t in tensors)
    unknown = sorted(d for d in dtypes if d not in DTYPE_BYTES)

    parameters = sum(elements(t["shape"]) for t in tensors)
    payload = sum(t["data_offsets"][1] - t["data_offsets"][0] for t in tensors)
    shard_payload = sum(s["payload_bytes"] for s in shards)
    shard_tensors = sum(s["tensor_count"] for s in shards)

    offset_mismatches = [
        t["name"]
        for t in tensors
        if t["data_offsets"][1] - t["data_offsets"][0]
        != elements(t["shape"]) * DTYPE_BYTES.get(t["dtype"], -1)
    ]
    shard_names = sorted({t["shard"] for t in tensors})
    head_tensors = [t["name"] for t in tensors if "lm_head" in t["name"]]
    embedding_tensors = [t["name"] for t in tensors if "embedding.weight" in t["name"]]

    checks = {
        "all_tensor_dtype_supported": not unknown,
        "all_tensors_bf16": set(dtypes) == {"BF16"},
        "no_duplicate_tensor_names": not duplicates,
        "offsets_match_shape_and_dtype": not offset_mismatches,
        "tensor_count_matches_shards": len(tensors) == shard_tensors,
        "payload_matches_shards": payload == shard_payload,
        "payload_equals_parameters_times_two": payload == parameters * 2,
        "shards_referenced_are_indexed": sorted(s["filename"] for s in shards) == shard_names,
        "language_head_not_serialized_separately": not head_tensors,
        "single_embedding_matrix_present": len(embedding_tensors) == 1,
    }
    failed = sorted(k for k, ok in checks.items() if not ok)
    if failed:
        raise SystemExit(f"tensor index audit failed: {failed}")

    return {
        "tensors": len(tensors),
        "shards": len(shards),
        "unique_parameters": parameters,
        "serialized_payload_bytes": payload,
        "dtype_element_counts": dict(sorted(dtypes.items())),
        "embedding_tensors": embedding_tensors,
        "language_head": (
            "not serialized as a separate tensor; the language head is tied to "
            f"{embedding_tensors[0]}, so the tied head adds no parameters and no bytes"
        ),
        "checks": checks,
        "auxiliary_parameters_historical_definition": sum(
            elements(t["shape"])
            for t in tensors
            if len(t["shape"]) != 2 or t["name"] == "model.embedding.weight"
        ),
    }


def encoder_scope() -> dict:
    """Parse the revision's declared encoder pair and validate it against sources.

    The declaration in the architecture document is scope metadata for the current
    revision: it is not a tensor inventory of the encoders and not a native
    measurement. Its parameter total is required to equal the sum of the two
    component declarations and its rounded GiB claim is required to match the
    16-byte recomputation.
    """
    doc_text = (ROOT / ARCHITECTURE_DOC_PATH).read_text()
    declaration = re.search(
        r"SigLIP2/SPEAR pair alone adds ([\d,]+) parameters and roughly "
        r"([\d.]+) GiB",
        doc_text,
    )
    if declaration is None:
        raise SystemExit("encoder declaration sentence not found in the architecture document")
    doc_revision = re.search(
        r"`google/siglip2-base-patch16-224` revision `([0-9a-f]{40})`", doc_text
    )
    if doc_revision is None:
        raise SystemExit("SigLIP2 revision pin not found in the architecture document")
    doc_audio = re.search(
        r"spear-base-speech-audio-v2`[^\n]*official metadata ([\d,]+) parameters",
        doc_text,
    )
    if doc_audio is None:
        raise SystemExit("SPEAR parameter declaration not found in the architecture document")
    doc_audio_parameters = int(doc_audio.group(1).replace(",", ""))

    declared_parameters = int(declaration.group(1).replace(",", ""))
    declared_GiB = float(declaration.group(2))
    declared_line = doc_text[: declaration.start()].count("\n") + 1

    plan = load_json(PRODUCT_PLAN_PATH)["vision"]
    plan_audio = load_json(PRODUCT_PLAN_PATH)["audio"]["candidate"]
    manifest = load_json(SIGLIP_MANIFEST_PATH)
    inventory = manifest["tensor_inventory"]

    vision_parameters = plan["encoder_parameters_observed"]
    audio_parameters = plan_audio["parameters_metadata"]
    combined = vision_parameters + audio_parameters
    recomputed_16_GiB = combined * 16 / GIB

    checks = {
        "declaration_equals_component_sum": declared_parameters == combined,
        "declaration_GiB_matches_16_byte_recomputation": abs(
            declared_GiB - recomputed_16_GiB
        )
        <= 0.01,
        "product_plan_matches_sealed_vision_inventory": vision_parameters
        == inventory["vision_parameter_elements"],
        "product_plan_matches_sealed_vision_revision": plan["revision"]
        == manifest["revision"],
        "architecture_doc_matches_sealed_vision_revision": doc_revision.group(1)
        == manifest["revision"],
        "component_revisions_are_pinned": all(
            re.fullmatch(r"[0-9a-f]{40}", value)
            for value in (plan["revision"], plan_audio["revision"])
        ),
        "architecture_doc_matches_audio_metadata": doc_audio_parameters
        == audio_parameters,
    }
    failed = sorted(k for k, ok in checks.items() if not ok)
    if failed:
        raise SystemExit(f"encoder scope declaration failed validation: {failed}")

    return {
        "role": (
            "declared scope metadata for the current revision; not a fresh encoder "
            "tensor inventory, not a compiled model, and not a native measurement"
        ),
        "declaration": {
            "path": ARCHITECTURE_DOC_PATH,
            "sha256": sha256_file(ROOT / ARCHITECTURE_DOC_PATH),
            "line": declared_line,
            "text": declaration.group(0),
            "declared_parameters": declared_parameters,
            "declared_GiB_at_16_bytes": declared_GiB,
            "recomputed_GiB_at_16_bytes": round(recomputed_16_GiB, 6),
        },
        "components": [
            {
                "name": "SigLIP2 base patch16-224 vision encoder",
                "repo_id": plan["repo_id"],
                "revision": plan["revision"],
                "parameters": vision_parameters,
                "evidence": {
                    "path": PRODUCT_PLAN_PATH,
                    "field": "vision.encoder_parameters_observed",
                    "sha256": sha256_file(ROOT / PRODUCT_PLAN_PATH),
                },
                "sealed_inventory": {
                    "path": SIGLIP_MANIFEST_PATH,
                    "field": "tensor_inventory.vision_parameter_elements",
                    "sha256": sha256_file(ROOT / SIGLIP_MANIFEST_PATH),
                    "seal_digest": manifest.get("seal_digest"),
                    "vision_tensor_count": inventory.get("vision_tensor_count"),
                    "note": (
                        "sealed source manifest reports the vision tower only "
                        f"({inventory.get('vision_tensor_count')} of "
                        f"{inventory.get('tensor_count')} tensors); the text tower and "
                        "the paired projector are not part of this declared scope"
                    ),
                },
            },
            {
                "name": "SPEAR base speech/audio v2 encoder",
                "repo_id": plan_audio["repo_id"],
                "revision": plan_audio["revision"],
                "parameters": audio_parameters,
                "evidence": {
                    "path": PRODUCT_PLAN_PATH,
                    "field": "audio.candidate.parameters_metadata",
                    "sha256": sha256_file(ROOT / PRODUCT_PLAN_PATH),
                    "note": (
                        "official repository metadata count; no sealed tensor index "
                        "for this candidate exists in the repository"
                    ),
                },
            },
        ],
        "combined_parameters": combined,
        "checks": checks,
    }


def bounds(parameters: int) -> dict:
    return {
        "fp8_weights_only": parameters * 1,
        "bf16_weights_only": parameters * 2,
        "fp32_weights_only": parameters * 4,
    }


def state_scenario(key, label, bytes_per_parameter, formula, note, parameters, guard):
    total = parameters * bytes_per_parameter
    headroom = guard - total
    return {
        "id": key,
        "label": label,
        "bytes_per_parameter": bytes_per_parameter,
        "formula": formula,
        "total_bytes": total,
        "total_GiB": round(total / GIB, 6),
        "below_driver_guard_lower_bound": total < guard,
        "headroom_to_driver_guard_bytes": headroom,
        "headroom_to_driver_guard_GiB": round(headroom / GIB, 6),
        "status": "analytical_lower_bound_not_admitted",
        "note": note,
    }


def calculate() -> dict:
    header = load_json(HEADER_PATH)
    minicpm_run = load_json(MINICPM_RUN_PATH)
    minicpm_guard = load_json(MINICPM_GUARD_PATH)

    audit = audit_header(header)
    parameters = audit["unique_parameters"]

    guard = minicpm_guard["guard"]["policy"]["driver_limit_bytes"]
    if guard != DRIVER_GUARD_GIB * GIB:
        raise SystemExit(
            f"driver guard policy mismatch: {guard} != {DRIVER_GUARD_GIB} GiB"
        )

    weight_bounds = bounds(parameters)

    training_state = [
        state_scenario(
            "fp32_weight_gradient_two_fp32_moments",
            "FP32 weights, FP32 gradients, two FP32 Adam moments",
            16,
            "4 + 4 + 4 + 4 bytes/parameter",
            "Full-precision AdamW over the entire 4,112,079,360-parameter scope, "
            "matching the currently selected NativeFusedAdamW state dtype. "
            "Exceeds the driver guard on the parameter/state inventory alone.",
            parameters,
            guard,
        ),
        state_scenario(
            "bf16_weight_gradient_two_fp32_moments_no_master",
            "BF16 weights, BF16 gradients, two FP32 Adam moments, no FP32 master copy",
            12,
            "2 + 2 + 4 + 4 bytes/parameter",
            "Mixed-precision AdamW without an FP32 master weight. BF16 gradients "
            "are an optimistic assumption: autocast/AMP normally accumulates FP32 "
            "gradients, which moves this row toward 14 bytes/parameter. The "
            "remaining headroom is not evidence of fit.",
            parameters,
            guard,
        ),
        state_scenario(
            "bf16_weight_fp32_gradient_two_fp32_moments_no_master",
            "BF16 stored weights, FP32 gradients, two FP32 Adam moments, no master copy",
            14,
            "2 + 4 + 4 + 4 bytes/parameter",
            "The common autocast/AMP shape: stored BF16 weights with FP32 "
            "accumulated gradients. Also outside the guard on the parameter and "
            "state inventory alone, before any activation.",
            parameters,
            guard,
        ),
        state_scenario(
            "bf16_weight_gradient_two_fp32_moments_fp32_master",
            "BF16 stored weights plus FP32 master weights, FP32 gradients, two FP32 Adam moments",
            16,
            "2 + 4 + 4 + 4 + 4 bytes/parameter",
            "Mixed precision with an explicit FP32 master copy, as required for "
            "faithful AdamW updates from BF16 storage. Byte-identical to the FP32 "
            "row and therefore also outside the guard before any activations.",
            parameters,
            guard,
        ),
    ]

    auxiliary = audit["auxiliary_parameters_historical_definition"]
    matrix = parameters - auxiliary
    muon_bytes = 9 * parameters + 7 * auxiliary
    historical = [
        {
            "id": "historical_bf16_weight_gradient_two_int8_moments_no_master",
            "label": "Historical: BF16 weights and gradients with two int8 quantized moments",
            "bytes_per_parameter": 6,
            "formula": "2 + 2 + 1 + 1 bytes/parameter",
            "total_bytes": parameters * 6,
            "total_GiB": round(parameters * 6 / GIB, 6),
            "status": "historical_unselected",
            "note": HISTORICAL_NOTE + " Quantization scales, block metadata and the "
            "dequantize/requantize workspace are excluded, so even this row is a "
            "lower bound for an implementation that does not exist here.",
        },
        {
            "id": "historical_fp32_muon_int8_momentum_with_auxiliary_fp32_adam",
            "label": "Historical: FP32 Muon with int8 momentum plus auxiliary FP32 Adam",
            "bytes_per_parameter": None,
            "formula": "(9 * parameters + 7 * auxiliary_parameters) bytes",
            "matrix_parameters": matrix,
            "auxiliary_parameters": auxiliary,
            "total_bytes": muon_bytes,
            "total_GiB": round(muon_bytes / GIB, 6),
            "status": "historical_unselected",
            "note": HISTORICAL_NOTE + " The auxiliary set is the historical "
            "assumption 'any non-2D tensor plus model.embedding.weight'; "
            "orthogonalization scratch, Newton-Schulz temporaries, block scales "
            "and cross-rank reductions are excluded. The 7-byte auxiliary term "
            "was inherited without a recorded per-parameter decomposition.",
        },
    ]
    for row in historical:
        row["below_driver_guard_lower_bound"] = row["total_bytes"] < guard
        row["headroom_to_driver_guard_bytes"] = guard - row["total_bytes"]
        row["headroom_to_driver_guard_GiB"] = round((guard - row["total_bytes"]) / GIB, 6)

    audit_block = minicpm_run["optimizer_audit"]
    minicpm_parameters = minicpm_run["trainable_parameters"]
    minicpm_state_bytes = audit_block["state_tensor_bytes"]
    minicpm_peak = minicpm_guard["guard"]["peak_driver_bytes"]
    minicpm_16 = minicpm_parameters * 16
    cross_check = {
        "role": "cross_check_of_optimizer_state_accounting_only_not_a_Spark_fit",
        "model": "MiniCPM5 sealed source, native sustained fused AdamW probe",
        "sources": [MINICPM_RUN_PATH, MINICPM_GUARD_PATH],
        "trainable_parameters": minicpm_parameters,
        "optimizer_class": audit_block["optimizer_class"],
        "optimizer_implementation": audit_block["implementation"],
        "fused": audit_block["fused"],
        "parameter_dtype": audit_block["parameter_dtype"],
        "state_tensor_bytes": minicpm_state_bytes,
        "state_bytes_by_dtype": audit_block["state_bytes_by_dtype"],
        "measured_bytes_per_parameter": round(minicpm_state_bytes / minicpm_parameters, 9),
        "two_moment_fp32_expectation_bytes": minicpm_parameters * 8,
        "excess_over_two_moment_expectation_bytes": minicpm_state_bytes
        - minicpm_parameters * 8,
        "interpretation": (
            "Measured optimizer state is two FP32 moments per trainable parameter "
            "(8 bytes/parameter) plus "
            f"{minicpm_state_bytes - minicpm_parameters * 8} bytes of non-parameter "
            "scalars. It confirms that the selected AdamW keeps no master copy, "
            "no int8 moments and no extra per-parameter tensors, which is why the "
            "12-byte and 16-byte Spark rows are the correct accounting shapes. It "
            "says nothing about Spark: the model, sequence length, activation "
            "profile and clip path differ."
        ),
        "measured_peak_driver_bytes": minicpm_peak,
        "excluded_overhead_exercised_at_that_peak_bytes": minicpm_peak - minicpm_16,
        "excluded_overhead_fraction_of_16_byte_bound": round(
            (minicpm_peak - minicpm_16) / minicpm_16, 6
        ),
        "excluded_overhead_note": (
            "On MiniCPM5 the sampled process peak exceeded its 16-byte/parameter "
            "parameter/state inventory by the figure above at physical microbatch 1 "
            "with activation checkpointing. That delta covers activations, logits, "
            "clip and allocator behaviour for a 2.52B-parameter model and must not "
            "be scaled, transferred or subtracted from the Spark guard."
        ),
    }

    encoder = encoder_scope()
    encoder_parameters = encoder["combined_parameters"]
    combined_parameters = parameters + encoder_parameters
    combined_state = [
        state_scenario(
            "combined_bf16_weight_gradient_two_fp32_moments_no_master",
            "Backbone plus declared encoder pair, BF16 weight and gradient, two FP32 moments, no master",
            12,
            "2 + 2 + 4 + 4 bytes/parameter",
            "Declared revision scope at the most optimistic accounting rate. The "
            "encoder increment is declared metadata, not a sealed tensor inventory, "
            "so this row is a lower bound over an assumption.",
            combined_parameters,
            guard,
        ),
        state_scenario(
            "combined_bf16_weight_fp32_gradient_two_fp32_moments_no_master",
            "Backbone plus declared encoder pair, BF16 stored weight, FP32 gradient, two FP32 moments, no master",
            14,
            "2 + 4 + 4 + 4 bytes/parameter",
            "Autocast-shaped mixed precision over the declared revision scope.",
            combined_parameters,
            guard,
        ),
        state_scenario(
            "combined_bf16_weight_gradient_two_fp32_moments_fp32_master",
            "Backbone plus declared encoder pair, BF16 stored weights with FP32 masters, FP32 gradients, two FP32 moments",
            16,
            "2 + 4 + 4 + 4 + 4 bytes/parameter",
            "Mixed precision with explicit FP32 masters over the declared revision "
            "scope; byte-identical to all-FP32 accounting, which the current fused "
            "AdamW selection makes the realistic full-scope shape.",
            combined_parameters,
            guard,
        ),
    ]
    for row in combined_state:
        increment = encoder_parameters * row["bytes_per_parameter"]
        row["encoder_increment_bytes"] = increment
        row["encoder_increment_GiB"] = round(increment / GIB, 6)
        row["declared_parameters"] = combined_parameters

    combined_over_guard = [
        row["id"] for row in combined_state if not row["below_driver_guard_lower_bound"]
    ]
    combined_under_guard = [
        row["id"] for row in combined_state if row["below_driver_guard_lower_bound"]
    ]

    over_guard = [s["id"] for s in training_state if not s["below_driver_guard_lower_bound"]]

    under_guard = [s["id"] for s in training_state if s["below_driver_guard_lower_bound"]]

    return {
        "format": "replica.sparkx_residency.v1",
        "kind": "analytical_lower_bounds_not_native_measurements_or_admission",
        "regeneration": "deterministic; re-running the calculator reproduces this file byte for byte",
        "calculator": {
            "path": CALCULATOR_PATH,
            "sha256": sha256_file(ROOT / CALCULATOR_PATH),
            "language": "python3 standard library only",
            "invocation": f"python3 {CALCULATOR_PATH}",
        },
        "receipt": {"path": RECEIPT_PATH, "note": "this file; written by the calculator"},
        "sources": [
            {
                "path": HEADER_PATH,
                "role": "official BF16 safetensors tensor index for the pinned checkpoint",
                "sha256": sha256_file(ROOT / HEADER_PATH),
                "bytes": (ROOT / HEADER_PATH).stat().st_size,
                "repo_id": header.get("repo_id"),
                "revision": header.get("revision"),
                "storage": header.get("storage"),
                "schema": header.get("schema"),
            },
            {
                "path": MINICPM_RUN_PATH,
                "role": "MiniCPM5 measured fused AdamW optimizer-state and parameter counts",
                "sha256": sha256_file(ROOT / MINICPM_RUN_PATH),
                "bytes": (ROOT / MINICPM_RUN_PATH).stat().st_size,
            },
            {
                "path": MINICPM_GUARD_PATH,
                "role": "driver guard policy value and sampled peak driver bytes",
                "sha256": sha256_file(ROOT / MINICPM_GUARD_PATH),
                "bytes": (ROOT / MINICPM_GUARD_PATH).stat().st_size,
            },
            {
                "path": ARCHITECTURE_DOC_PATH,
                "role": "current revision declaration of the SigLIP2/SPEAR encoder pair and its training scope",
                "sha256": sha256_file(ROOT / ARCHITECTURE_DOC_PATH),
                "bytes": (ROOT / ARCHITECTURE_DOC_PATH).stat().st_size,
            },
            {
                "path": PRODUCT_PLAN_PATH,
                "role": "structured vision/audio component declarations and pinned revisions",
                "sha256": sha256_file(ROOT / PRODUCT_PLAN_PATH),
                "bytes": (ROOT / PRODUCT_PLAN_PATH).stat().st_size,
            },
            {
                "path": SIGLIP_MANIFEST_PATH,
                "role": "sealed SigLIP2 vision tensor inventory and revision pin",
                "sha256": sha256_file(ROOT / SIGLIP_MANIFEST_PATH),
                "bytes": (ROOT / SIGLIP_MANIFEST_PATH).stat().st_size,
            },
        ],
        "driver_guard": {
            "driver_limit_bytes": guard,
            "driver_limit_GiB": guard / GIB,
            "policy": "50 GiB sampled driver stop, unchanged repository policy",
            "scope": "the guard bounds the native worker process, not the model alone",
            "companion_limits": {
                "sustained_swap_growth_limit_bytes": minicpm_guard["guard"]["policy"][
                    "swap_growth_limit_bytes"
                ],
                "minimum_free_percent": minicpm_guard["guard"]["policy"][
                    "minimum_free_percent"
                ],
                "note": "neither companion limit is evaluated by this calculator",
            },
        },
        "header_audit": audit,
        "weight_payload_lower_bounds": {
            "scope": f"all {parameters} unique parameters of the serialized checkpoint",
            "fp8_weights_only": {
                "bytes": weight_bounds["fp8_weights_only"],
                "GiB": round(weight_bounds["fp8_weights_only"] / GIB, 6),
                "formula": "parameters * 1 byte",
                "note": "8-bit storage of every parameter; per-block scales, padding "
                "and dequantization buffers are excluded",
            },
            "bf16_weights_only": {
                "bytes": weight_bounds["bf16_weights_only"],
                "GiB": round(weight_bounds["bf16_weights_only"] / GIB, 6),
                "formula": "parameters * 2 bytes",
                "note": "equals the serialized BF16 payload of the indexed shards",
            },
            "fp32_weights_only": {
                "bytes": weight_bounds["fp32_weights_only"],
                "GiB": round(weight_bounds["fp32_weights_only"] / GIB, 6),
                "formula": "parameters * 4 bytes",
                "note": "materialized FP32 parameters; the selected optimizer's "
                "measured state is FP32 and the model weights are FP32 in training",
            },
        },
        "training_state_scenarios": training_state,
        "encoder_scope": encoder,
        "combined_scope": {
            "scope": (
                f"backbone {parameters} + declared encoder pair {encoder_parameters} = "
                f"{combined_parameters} parameters"
            ),
            "backbone_parameters": parameters,
            "encoder_parameters": encoder_parameters,
            "combined_parameters": combined_parameters,
            "weight_payload_lower_bounds": {
                "fp8_weights_only": {
                    "bytes": combined_parameters * 1,
                    "GiB": round(combined_parameters * 1 / GIB, 6),
                    "formula": "combined_parameters * 1 byte",
                },
                "bf16_weights_only": {
                    "bytes": combined_parameters * 2,
                    "GiB": round(combined_parameters * 2 / GIB, 6),
                    "formula": "combined_parameters * 2 bytes",
                },
                "fp32_weights_only": {
                    "bytes": combined_parameters * 4,
                    "GiB": round(combined_parameters * 4 / GIB, 6),
                    "formula": "combined_parameters * 4 bytes",
                },
            },
            "training_state_scenarios": combined_state,
            "excluded_beyond_backbone_only": COMBINED_EXCLUSIONS,
            "interpretation": {
                "over_driver_guard_on_lower_bound_alone": combined_over_guard,
                "under_driver_guard_on_lower_bound_alone": combined_under_guard,
                "full_scope_local_admission": "not_admitted",
                "admission_statement": (
                    "The combined revision scope is not admitted locally either. "
                    "Two of its three accounting rates already exceed the guard on "
                    "the parameter and state inventory alone, and the remaining "
                    "12-byte rate leaves a headroom far smaller than the excluded "
                    "bridge, projector, fast/expert state, activation, logits, KV "
                    "and media budgets. The encoder total is a declared scope "
                    "figure, so even its parameter count is an assumption rather "
                    "than a sealed inventory."
                ),
            },
        },
        "historical_unselected_scenarios": historical,
        "minicpm_measured_cross_check": cross_check,
        "excluded_from_every_figure": EXCLUSIONS,
        "interpretation": {
            "over_driver_guard_on_lower_bound_alone": over_guard,
            "under_driver_guard_on_lower_bound_alone": under_guard,
            "full_scope_local_admission": "not_admitted",
            "admission_statement": (
                "No full-scope configuration is admitted. Any row at or above the "
                "guard is excluded outright, because the excluded terms can only "
                "add bytes. Any row below the guard remains unqualified: a lower "
                "bound below a ceiling is not evidence that the measured peak "
                "stays below it, and no Spark training run exists to measure that "
                "peak. No figure here is a native fit, allocator peak, throughput "
                "or convergence result."
            ),
            "scope_note": (
                "The training_state_scenarios rows cover the backbone only. The "
                "current revision trains the backbone plus the declared SigLIP2/SPEAR "
                "encoder pair, so combined_scope is the applicable full-scope "
                "inventory."
            ),
            "rounding": "byte counts are exact integers; GiB values are bytes / 2**30 rounded to 6 decimals",
        },
        "non_claims": [
            "no Spark weight, tokenizer, dataset or other asset was downloaded or inspected beyond the committed tensor index header and the sealed SigLIP2 source manifest",
            "the encoder pair total comes from a declarative scope statement validated against component declarations, not from a fresh encoder tensor index and not from a compiled or executed multimodal model",
            "no native Spark training, export, adapter, multimodal or inference run was performed or implied",
            "the Spark adapter remains design_only_not_implemented; FP8/BF16 source preparation is offline-only",
            "dataset admission, the source registry and optimizer/adapter status are unchanged by this receipt",
        ],
    }


def main() -> None:
    receipt = calculate()
    OUT.mkdir(parents=True, exist_ok=True)
    (ROOT / RECEIPT_PATH).write_text(json.dumps(receipt, indent=2) + "\n")
    print(
        "Wrote "
        f"{RECEIPT_PATH}: {receipt['header_audit']['unique_parameters']} parameters, "
        f"{receipt['header_audit']['tensors']} tensors, lower bounds only; "
        "no models or native jobs executed."
    )


if __name__ == "__main__":
    main()

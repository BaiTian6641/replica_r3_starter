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
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent

GIB = 2**30
MIB = 2**20

HEADER_PATH = "reports/spark_preparation_20261025/run_01/headers/official_bf16_control/tensor_index.json"
MINICPM_RUN_PATH = "runs/fused_adamw_20261025/sustained01/run_report.json"
MINICPM_GUARD_PATH = "reports/fused_adamw_20261025/sustained01.guard.json"
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
            "rounding": "byte counts are exact integers; GiB values are bytes / 2**30 rounded to 6 decimals",
        },
        "non_claims": [
            "no Spark weight, tokenizer, dataset or other asset was downloaded or inspected beyond the committed tensor index header",
            "no native Spark training, export, adapter or inference run was performed or implied",
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

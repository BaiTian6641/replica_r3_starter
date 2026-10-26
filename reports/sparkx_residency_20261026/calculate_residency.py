#!/usr/bin/env python3
"""Deterministic analytical residency ledger for the SparkX 2.6 4B revision.

Standard library only. All figures are recomputed from committed evidence:
the sealed BF16 tensor index, the sealed mixed-dtype FP8 adaptation index, the
repository driver-guard policy and the versioned machine-readable component
inventory at configs/sparkx_residency_components_v1.json. The architecture
document is cross-checked against that inventory but is never an arithmetic
source. No weights, loaders, tokenizer, dataset, model code or native job is
required, downloaded or executed.

The ledger deliberately separates:

* source artifact storage (BF16 checkpoint, mixed-dtype FP8 checkpoint),
* training storage (model copies, gradients, masters, moments),
* memory-module state (bank tensors, dynamic fast/knowledge state),
* analytical activation / logit / temporary-buffer budgets,
* restore and export peaks.

The corrected five-copy mixed-precision rate is 18 bytes/parameter:
BF16 weights 2 + FP32 master 4 + FP32 gradients 4 + two FP32 moments 4+4.
A separate all-FP32 rate is 16 bytes/parameter. That is a different storage
plan, not the same one with different rounding. FP8 artifact storage is a
third, source-only account and does not establish FP8 training capability.

Usage:
  python3 reports/sparkx_residency_20261026/calculate_residency.py
  python3 reports/sparkx_residency_20261026/calculate_residency.py --check

Writes (default): reports/sparkx_residency_20261026/residency_v2.json
The historical v1 receipt and prose-derived v2 legacy receipt are never modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent

GIB = 2**30
MIB = 2**20
KIB = 2**10

BF16_HEADER_PATH = (
    "reports/spark_preparation_20261025/run_01/headers/"
    "official_bf16_control/tensor_index.json"
)
FP8_HEADER_PATH = (
    "reports/spark_preparation_20261025/run_01/headers/"
    "official_fp8_adaptation_base/tensor_index.json"
)
GUARD_PATH = "reports/fused_adamw_20261025/sustained01.guard.json"
COMPONENTS_PATH = "configs/sparkx_residency_components_v1.json"
ARCHITECTURE_PATH = (
    "docs/architecture/21_SparkX_2_6_4B_Consolidated_Architecture_and_MiniCPM_Validation.md"
)
CALCULATOR_PATH = "reports/sparkx_residency_20261026/calculate_residency.py"
RECEIPT_PATH = "reports/sparkx_residency_20261026/residency_v2.json"
SUPERSEDED_RECEIPT_PATH = "reports/sparkx_residency_20261026/residency.json"
LEGACY_MARKDOWN_RECEIPT_PATH = (
    "reports/sparkx_residency_20261026/residency_v2_markdown_legacy.json"
)

DRIVER_GUARD_GIB = 50  # repository policy, asserted against the committed guard

DTYPE_BYTES = {
    "BF16": 2,
    "F16": 2,
    "F32": 4,
    "FP32": 4,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
}

# The four candidate training storage plans. Only the first two are asserted
# as live options; the others keep the optimistic bounds traceable. The key
# correction in this revision is that a coexisting BF16 weight and FP32 master
# cost 2 + 4, so the full AdamW copy set is 18 bytes/parameter, never 16.
TRAINING_PLANS = [
    {
        "id": "all_fp32",
        "label": "FP32 weights, FP32 gradients, two FP32 Adam moments",
        "copies": [
            ("weights_fp32", "FP32", 1),
            ("gradients_fp32", "FP32", 1),
            ("moment1_fp32", "FP32", 1),
            ("moment2_fp32", "FP32", 1),
        ],
        "note": (
            "The FP32 weight tensor is itself the master. This is 16 "
            "bytes/parameter: 4 + 4 + 4 + 4."
        ),
    },
    {
        "id": "bf16_weights_fp32_master_fp32_gradients_two_fp32_moments",
        "label": (
            "BF16 stored weights plus FP32 master weights, FP32 gradients, "
            "two FP32 Adam moments"
        ),
        "copies": [
            ("weights_bf16", "BF16", 1),
            ("master_weights_fp32", "FP32", 1),
            ("gradients_fp32", "FP32", 1),
            ("moment1_fp32", "FP32", 1),
            ("moment2_fp32", "FP32", 1),
        ],
        "note": (
            "The five-copy mixed-precision shape required for faithful AdamW "
            "updates from BF16 storage. It is 18 bytes/parameter: "
            "2 + 4 + 4 + 4 + 4. It must not be called 16."
        ),
    },
    {
        "id": "bf16_weights_fp32_gradients_two_fp32_moments_no_master",
        "label": (
            "BF16 stored weights, FP32 accumulated gradients, two FP32 Adam "
            "moments, no FP32 master copy"
        ),
        "copies": [
            ("weights_bf16", "BF16", 1),
            ("gradients_fp32", "FP32", 1),
            ("moment1_fp32", "FP32", 1),
            ("moment2_fp32", "FP32", 1),
        ],
        "note": "14 bytes/parameter: 2 + 4 + 4 + 4. The update has no master.",
    },
    {
        "id": "bf16_weights_bf16_gradients_two_fp32_moments_no_master",
        "label": (
            "BF16 stored weights and BF16 gradients, two FP32 Adam moments, "
            "no FP32 master copy"
        ),
        "copies": [
            ("weights_bf16", "BF16", 1),
            ("gradients_bf16", "BF16", 1),
            ("moment1_fp32", "FP32", 1),
            ("moment2_fp32", "FP32", 1),
        ],
        "note": (
            "12 bytes/parameter: 2 + 2 + 4 + 4. BF16 gradient accumulation is "
            "an optimistic lower bound; autocast usually accumulates FP32."
        ),
    },
]

# The component inventory owns every architecture width and formula used by the
# fixed-parameter account. These identifiers make missing or unknown entries a
# hard failure instead of an accidental fallback.
COMPONENT_INVENTORY_SCHEMA = "replica.sparkx_residency_components.v1"
COMPONENT_INVENTORY_KIND = "declared_analytical_component_inventory_not_measured"
COMPONENT_IDS = (
    "language_backbone",
    "vision_encoder",
    "audio_encoder",
    "vision_projector",
    "audio_projector",
    "memory_query_output",
    "memory_reader_matrices",
    "phrase_encoder",
)
COMPONENT_WIDTH_KEYS = (
    "backbone",
    "vision_encoder",
    "audio_encoder",
    "bridge",
    "phrase",
    "reader",
)

LAYERS = 36
FFN_WIDTH = 10240
VOCAB = 131072
QUERY_HEADS = 16
KV_HEADS = 4
HEAD_DIM = 256
VISION_PATCHES = 196
AUDIO_TOKENS = 40
MICROBATCH_POSITIONS = 256  # declared product-plan cap, not a measured batch
MEMORY_BANK_RECORDS = 4096
MEMORY_BANK_ELEMENTS_PER_RECORD = 896
SUSTAINED_SWAP_LIMIT = 512 * MIB
MINIMUM_FREE_PERCENT = 15

EXCLUSIONS = [
    "allocator fragmentation, caching-allocator slack, compiler workspace and MPS driver overhead",
    "autograd graph metadata, Python object overhead and per-tensor bookkeeping",
    "per-worker or per-session replication of model, optimizer, activation or memory state",
    "checkpoint I/O buffers beyond the declared export staging copy",
    "raw media, tokenized corpora, replay buffers and data-loader prefetch",
    "per-camera/per-microphone encoder streaming queues beyond the declared temporary budget",
    "teacher models, evaluation harnesses and any training-only auxiliary network not listed in the component inventory",
]


class ResidencyError(RuntimeError):
    """Raised when committed evidence contradicts the declared arithmetic."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(root: Path, relative: str) -> dict:
    path = root / relative
    if not path.is_file():
        raise ResidencyError(f"required committed input missing: {relative}")
    return json.loads(path.read_text(encoding="utf-8"))


def elements(shape) -> int:
    return math.prod(int(value) for value in shape)


def gib(value: int) -> float:
    return round(value / GIB, 6)


def mib(value: int) -> float:
    return round(value / MIB, 6)


def _fail(scope: str, failures) -> None:
    actual = sorted({failure for failure in failures if failure})
    if actual:
        raise ResidencyError(f"{scope} failed validation: {actual}")


def audit_bf16_header(header: dict) -> dict:
    """Audit the sealed BF16 checkpoint and return its parameter inventory."""
    tensors = header["tensors"]
    shards = header["shard_headers"]
    names = [tensor["name"] for tensor in tensors]
    duplicates = sorted(name for name, count in Counter(names).items() if count > 1)
    dtype_counts = Counter(tensor["dtype"] for tensor in tensors)
    unknown = sorted(dtype for dtype in dtype_counts if dtype not in DTYPE_BYTES)

    parameters = sum(elements(tensor["shape"]) for tensor in tensors)
    payload = sum(t["data_offsets"][1] - t["data_offsets"][0] for t in tensors)
    shard_payload = sum(shard["payload_bytes"] for shard in shards)
    shard_tensors = sum(shard["tensor_count"] for shard in shards)
    offset_mismatches = [
        tensor["name"]
        for tensor in tensors
        if tensor["data_offsets"][1] - tensor["data_offsets"][0]
        != elements(tensor["shape"]) * DTYPE_BYTES.get(tensor["dtype"], -1)
    ]
    embedding = [t["name"] for t in tensors if t["name"] == "model.embedding.weight"]
    head = [t["name"] for t in tensors if "lm_head" in t["name"]]

    checks = {
        "all_tensor_dtypes_known": not unknown,
        "all_tensors_bf16": set(dtype_counts) == {"BF16"},
        "no_duplicate_tensor_names": not duplicates,
        "offsets_match_shape_and_dtype": not offset_mismatches,
        "tensor_count_matches_shards": len(tensors) == shard_tensors,
        "payload_matches_shards": payload == shard_payload,
        "payload_equals_parameters_times_two": payload == parameters * 2,
        "shards_referenced_are_indexed": sorted(s["filename"] for s in shards)
        == sorted({t["shard"] for t in tensors}),
        "language_head_not_serialized_separately": not head,
        "single_tied_embedding_present": embedding
        == ["model.embedding.weight"],
    }
    _fail("BF16 tensor index audit", [k for k, ok in checks.items() if not ok])

    return {
        "role": "sealed BF16 source checkpoint; the only parameter inventory treated as complete",
        "tensor_count": len(tensors),
        "shard_count": len(shards),
        "unique_parameter_elements": parameters,
        "serialized_payload_bytes": payload,
        "dtype_tensor_counts": dict(sorted(dtype_counts.items())),
        "dtype_payload_bytes": {"BF16": payload},
        "checks": checks,
        "tied_embedding": embedding[0],
    }


def audit_fp8_header(header: dict) -> dict:
    """Audit the sealed mixed-dtype FP8 adaptation checkpoint.

    This artifact is *not* an all-FP8 weight dump. It stores 180 F8_E4M3
    matrices, their 180 BF16 scale tensors, a BF16 tied embedding, BF16 layer
    norms and BF16 attention gates. The audit therefore supports an explicit
    per-dtype artifact inventory instead of the superseded hypothetical
    "parameters x 1 byte" FP8 row.
    """
    tensors = header["tensors"]
    shards = header["shard_headers"]
    names = [tensor["name"] for tensor in tensors]
    duplicates = sorted(name for name, count in Counter(names).items() if count > 1)
    dtype_counts = Counter(tensor["dtype"] for tensor in tensors)
    unknown = sorted(dtype for dtype in dtype_counts if dtype not in DTYPE_BYTES)

    payload_by_dtype = Counter()
    tensor_count_by_dtype = Counter()
    parameter_elements_by_dtype = Counter()
    for tensor in tensors:
        size = tensor["data_offsets"][1] - tensor["data_offsets"][0]
        payload_by_dtype[tensor["dtype"]] += size
        tensor_count_by_dtype[tensor["dtype"]] += 1
        parameter_elements_by_dtype[tensor["dtype"]] += elements(tensor["shape"])

    payload = sum(payload_by_dtype.values())
    shard_payload = sum(shard["payload_bytes"] for shard in shards)
    shard_tensors = sum(shard["tensor_count"] for shard in shards)
    offset_mismatches = [
        tensor["name"]
        for tensor in tensors
        if tensor["data_offsets"][1] - tensor["data_offsets"][0]
        != elements(tensor["shape"]) * DTYPE_BYTES.get(tensor["dtype"], -1)
    ]

    f8_tensors = [t for t in tensors if t["dtype"] == "F8_E4M3"]
    scale_tensors = [t for t in tensors if t["name"].endswith("weight_scale")]
    scale_names = {t["name"] for t in scale_tensors}
    missing_scales = sorted(
        tensor["name"]
        for tensor in f8_tensors
        if tensor["name"].replace(".weight", ".weight_scale") not in scale_names
    )
    orphan_scales = sorted(
        name
        for name in scale_names
        if name.replace(".weight_scale", ".weight") not in set(names)
    )
    scales_are_bf16 = {t["dtype"] for t in scale_tensors} == {"BF16"}

    bf16_non_scale = [t for t in tensors if t["dtype"] == "BF16" and t not in scale_tensors]
    embedding = [t["name"] for t in bf16_non_scale if t["name"] == "model.embedding.weight"]
    final_norm = [t["name"] for t in bf16_non_scale if t["name"] == "model.norm.weight"]
    layer_norms = [
        t["name"]
        for t in bf16_non_scale
        if ".input_layernorm.weight" in t["name"]
        or ".post_attention_layernorm.weight" in t["name"]
    ]
    gates = [t["name"] for t in bf16_non_scale if ".self_attn.g_proj.weight" in t["name"]]
    accounted = (
        len(embedding)
        + len(final_norm)
        + len(layer_norms)
        + len(gates)
        + len(scale_tensors)
    )

    checks = {
        "all_tensor_dtypes_known": not unknown,
        "mixed_dtype_artifact": set(dtype_counts) == {"BF16", "F8_E4M3"},
        "no_duplicate_tensor_names": not duplicates,
        "offsets_match_shape_and_dtype": not offset_mismatches,
        "tensor_count_matches_shards": len(tensors) == shard_tensors,
        "payload_matches_shards": payload == shard_payload,
        "f8_matrices_have_bf16_scales": not missing_scales,
        "no_orphan_scale_tensors": not orphan_scales,
        "scale_tensors_are_bf16": scales_are_bf16,
        "one_scale_per_f8_matrix": len(scale_tensors) == len(f8_tensors),
        "tied_embedding_is_bf16": embedding == ["model.embedding.weight"],
        "final_norm_is_bf16": final_norm == ["model.norm.weight"],
        "all_layer_norms_are_bf16": len(layer_norms) == LAYERS * 2,
        "all_attention_gates_are_bf16": len(gates) == LAYERS,
        "bf16_tensors_accounted": accounted == len(bf16_non_scale) + len(scale_tensors),
    }
    scale_payload = sum(
        t["data_offsets"][1] - t["data_offsets"][0] for t in scale_tensors
    )

    def payload_bytes_of(tensor_list):
        return sum(t["data_offsets"][1] - t["data_offsets"][0] for t in tensor_list)

    embedding_payload = payload_bytes_of(
        [t for t in bf16_non_scale if t["name"] == "model.embedding.weight"]
    )
    norm_payload = payload_bytes_of(
        [
            t
            for t in bf16_non_scale
            if t["name"] == "model.norm.weight"
            or "layernorm.weight" in t["name"]
        ]
    )
    gate_payload = payload_bytes_of(
        [t for t in bf16_non_scale if ".self_attn.g_proj.weight" in t["name"]]
    )
    bf16_payload = payload_by_dtype["BF16"]
    checks["bf16_category_payloads_sum"] = (
        embedding_payload + norm_payload + gate_payload + scale_payload
        == bf16_payload
    )
    _fail("FP8 tensor index audit", [k for k, ok in checks.items() if not ok])

    return {
        "role": (
            "sealed mixed-dtype FP8 adaptation artifact; source storage only, "
            "not an FP8 training or backend capability record"
        ),
        "tensor_count": len(tensors),
        "shard_count": len(shards),
        "serialized_payload_bytes": payload,
        "dtype_tensor_counts": dict(sorted(tensor_count_by_dtype.items())),
        "dtype_payload_bytes": dict(sorted(payload_by_dtype.items())),
        "dtype_parameter_elements": dict(
            sorted(parameter_elements_by_dtype.items())
        ),
        "f8_matrix_count": len(f8_tensors),
        "scale_tensor_count": len(scale_tensors),
        "scale_payload_bytes": scale_payload,
        "bf16_category_payload_bytes": {
            "tied_embedding": embedding_payload,
            "layer_and_final_norms": norm_payload,
            "attention_gates": gate_payload,
            "scale_tensors": scale_payload,
        },
        "checks": checks,
    }


def _positive_int(value, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ResidencyError(f"{label} must be a positive integer")
    return value


def _require_exact_keys(mapping: dict, expected: set[str], label: str) -> None:
    missing = sorted(expected - set(mapping))
    unknown = sorted(set(mapping) - expected)
    if missing or unknown:
        details = []
        if missing:
            details.append(f"missing {missing}")
        if unknown:
            details.append(f"unknown {unknown}")
        raise ResidencyError(f"{label}: " + "; ".join(details))


def _width(widths: dict, reference: str, label: str) -> int:
    if not isinstance(reference, str) or reference not in widths:
        raise ResidencyError(f"{label} references unknown width: {reference!r}")
    return _positive_int(widths[reference], f"width {reference}")


def component_parameters(component: dict, widths: dict) -> int:
    """Compute one component from its versioned inventory descriptor."""
    component_id = component.get("id", "<unknown>")
    source = component.get("source")
    if not isinstance(source, dict):
        raise ResidencyError(f"{component_id} has no source descriptor")
    kind = source.get("kind")
    if kind == "bf16_header":
        raise ResidencyError(
            f"{component_id} is backed by the sealed BF16 header, not an inline count"
        )
    if kind in {"declared_encoder_count", "declared_count"}:
        if "parameters" not in component:
            raise ResidencyError(f"{component_id} is missing its declared count")
        return _positive_int(component["parameters"], f"{component_id} parameters")
    if kind != "formula":
        raise ResidencyError(f"{component_id} has unknown source kind: {kind!r}")

    formula = source.get("formula")
    if not isinstance(formula, dict):
        raise ResidencyError(f"{component_id} has no formula descriptor")
    formula_kind = formula.get("kind")
    if formula_kind == "two_stage_affine_with_bias":
        _require_exact_keys(
            formula,
            {"kind", "source_width", "bridge_width", "target_width"},
            f"{component_id} formula",
        )
        return _projector_parameters(
            _width(widths, formula["source_width"], f"{component_id} source"),
            _width(widths, formula["bridge_width"], f"{component_id} bridge"),
            _width(widths, formula["target_width"], f"{component_id} target"),
        )
    if formula_kind == "qkvo_reader_matrices":
        _require_exact_keys(
            formula,
            {"kind", "sites", "projections_per_site", "hidden_width", "head_width"},
            f"{component_id} formula",
        )
        sites = _positive_int(formula["sites"], f"{component_id} sites")
        projections = _positive_int(
            formula["projections_per_site"], f"{component_id} projections"
        )
        return (
            sites
            * projections
            * _width(widths, formula["hidden_width"], f"{component_id} hidden")
            * _width(widths, formula["head_width"], f"{component_id} head")
        )
    if formula_kind == "compact_phrase_encoder":
        _require_exact_keys(
            formula,
            {"kind", "layers", "matrix_terms_per_layer", "width", "hidden_width"},
            f"{component_id} formula",
        )
        layers = _positive_int(formula["layers"], f"{component_id} layers")
        terms = _positive_int(
            formula["matrix_terms_per_layer"], f"{component_id} matrix terms"
        )
        width = _width(widths, formula["width"], f"{component_id} width")
        hidden = _width(widths, formula["hidden_width"], f"{component_id} hidden")
        return layers * terms * width**2 + width * hidden
    raise ResidencyError(f"{component_id} has unknown formula kind: {formula_kind!r}")


def _formula_expression(component: dict, widths: dict) -> str:
    formula = component["source"]["formula"]
    kind = formula["kind"]
    if kind == "two_stage_affine_with_bias":
        source = _width(widths, formula["source_width"], "formula source")
        bridge = _width(widths, formula["bridge_width"], "formula bridge")
        target = _width(widths, formula["target_width"], "formula target")
        return f"{source}*{bridge}+{bridge} + {bridge}*{target}+{target}"
    if kind == "qkvo_reader_matrices":
        hidden = _width(widths, formula["hidden_width"], "formula hidden")
        head = _width(widths, formula["head_width"], "formula head")
        return f"sites * {formula['projections_per_site']} * {hidden} * {head}"
    if kind == "compact_phrase_encoder":
        width = _width(widths, formula["width"], "formula width")
        hidden = _width(widths, formula["hidden_width"], "formula hidden")
        return (
            f"{formula['layers']}*{formula['matrix_terms_per_layer']}*"
            f"{width}^2 + {width}*{hidden}"
        )
    raise ResidencyError(f"unknown formula kind: {kind!r}")


def validate_component_inventory(inventory: dict, bf16: dict | None = None) -> None:
    """Fail closed on missing, unknown or internally inconsistent components."""
    if not isinstance(inventory, dict):
        raise ResidencyError("component inventory must be a JSON object")

    failures: list[str] = []

    def check(condition: bool, name: str) -> None:
        if not condition:
            failures.append(name)

    check(
        inventory.get("schema_version") == COMPONENT_INVENTORY_SCHEMA,
        "schema_version",
    )
    check(inventory.get("kind") == COMPONENT_INVENTORY_KIND, "kind")
    check(bool(inventory.get("status")), "status")

    documentation = inventory.get("documentation")
    check(isinstance(documentation, dict), "documentation")
    if isinstance(documentation, dict):
        check(documentation.get("path") == ARCHITECTURE_PATH, "documentation.path")
        check(bool(documentation.get("role")), "documentation.role")

    widths = inventory.get("widths")
    check(isinstance(widths, dict), "widths")
    if isinstance(widths, dict):
        check(set(widths) == set(COMPONENT_WIDTH_KEYS), "widths.keys")
        for key in COMPONENT_WIDTH_KEYS:
            if key in widths:
                check(
                    type(widths[key]) is int and widths[key] > 0,
                    f"widths.{key}",
                )

    consistency = inventory.get("consistency")
    consistency_keys = {
        "backbone_parameters",
        "encoder_pair_parameters",
        "integrated_total_parameters",
    }
    check(isinstance(consistency, dict), "consistency")
    if isinstance(consistency, dict):
        check(set(consistency) == consistency_keys, "consistency.keys")
        for key in consistency_keys:
            if key in consistency:
                check(
                    type(consistency[key]) is int and consistency[key] > 0,
                    f"consistency.{key}",
                )

    components = inventory.get("components")
    check(isinstance(components, list), "components")
    if not isinstance(components, list):
        _fail("component inventory", failures)
        return

    ids = [
        component.get("id")
        for component in components
        if isinstance(component, dict)
    ]
    check(ids == list(COMPONENT_IDS), "component_ids_exact")
    check(len(ids) == len(set(ids)), "component_ids_unique")

    for component in components:
        if not isinstance(component, dict):
            failures.append("component_object")
            continue
        component_id = component.get("id", "<unknown>")
        source = component.get("source")
        check(
            isinstance(component.get("label"), str) and bool(component["label"]),
            f"{component_id}.label",
        )
        check(type(component.get("trainable")) is bool, f"{component_id}.trainable")
        if not isinstance(source, dict):
            failures.append(f"{component_id}.source")
            continue
        kind = source.get("kind")

        if component_id == "language_backbone":
            check(
                set(component) == {"id", "label", "trainable", "source"},
                f"{component_id}.keys",
            )
            check(kind == "bf16_header", f"{component_id}.source.kind")
            check(
                set(source) == {"kind", "path", "field", "tensor_count"},
                f"{component_id}.source.keys",
            )
            check(
                source.get("path") == BF16_HEADER_PATH,
                f"{component_id}.source.path",
            )
            check(source.get("field") == "sum(shape)", f"{component_id}.source.field")
            check(
                type(source.get("tensor_count")) is int
                and source.get("tensor_count", 0) > 0,
                f"{component_id}.source.tensor_count",
            )
        elif component_id in {"vision_encoder", "audio_encoder"}:
            check(
                set(component)
                == {"id", "label", "trainable", "parameters", "source"},
                f"{component_id}.keys",
            )
            check(kind == "declared_encoder_count", f"{component_id}.source.kind")
            check(
                set(source) == {"kind", "repo_id", "revision"},
                f"{component_id}.source.keys",
            )
            check(
                isinstance(source.get("repo_id"), str) and bool(source["repo_id"]),
                f"{component_id}.source.repo_id",
            )
            check(
                bool(re.fullmatch(r"[0-9a-f]{40}", str(source.get("revision", "")))),
                f"{component_id}.source.revision",
            )
        elif component_id == "memory_query_output":
            check(
                set(component)
                == {"id", "label", "trainable", "parameters", "source"},
                f"{component_id}.keys",
            )
            check(kind == "declared_count", f"{component_id}.source.kind")
            check(set(source) == {"kind"}, f"{component_id}.source.keys")
        else:
            check(
                set(component) == {"id", "label", "trainable", "source"},
                f"{component_id}.keys",
            )
            check(kind == "formula", f"{component_id}.source.kind")
            check(
                set(source) == {"kind", "formula"},
                f"{component_id}.source.keys",
            )

        try:
            if component_id == "language_backbone":
                if bf16 is not None:
                    _positive_int(
                        bf16.get("unique_parameter_elements"),
                        "backbone parameter count",
                    )
            else:
                component_parameters(
                    component, widths if isinstance(widths, dict) else {}
                )
        except ResidencyError as error:
            failures.append(f"{component_id}.parameters: {error}")

    if isinstance(widths, dict) and isinstance(consistency, dict):
        computed = {}
        try:
            for component in components:
                component_id = component["id"]
                if component_id == "language_backbone":
                    computed[component_id] = (
                        consistency["backbone_parameters"]
                        if bf16 is None
                        else bf16["unique_parameter_elements"]
                    )
                else:
                    computed[component_id] = component_parameters(component, widths)
        except (KeyError, TypeError, ResidencyError):
            failures.append("component_consistency_computation")
        else:
            if {"vision_encoder", "audio_encoder"} <= set(computed):
                check(
                    computed["vision_encoder"] + computed["audio_encoder"]
                    == consistency.get("encoder_pair_parameters"),
                    "encoder_pair_parameters",
                )
            if len(computed) == len(COMPONENT_IDS):
                check(
                    sum(computed.values())
                    == consistency.get("integrated_total_parameters"),
                    "integrated_total_parameters",
                )
            if bf16 is not None and "language_backbone" in computed:
                check(
                    computed["language_backbone"]
                    == consistency.get("backbone_parameters"),
                    "backbone_parameters",
                )
                check(
                    consistency.get("backbone_parameters")
                    == bf16.get("unique_parameter_elements"),
                    "backbone_header_parameters",
                )
                backbone_source = next(
                    component["source"]
                    for component in components
                    if component.get("id") == "language_backbone"
                )
                check(
                    backbone_source.get("tensor_count") == bf16.get("tensor_count"),
                    "backbone_header_tensor_count",
                )

    _fail("component inventory", failures)


def load_component_inventory(root: Path) -> dict:
    inventory = load_json(root, COMPONENTS_PATH)
    validate_component_inventory(inventory)
    return inventory


def validate_architecture_documentation(
    document: str,
    inventory: dict,
    backbone_parameters: int,
    integrated_total_parameters: int,
) -> dict:
    """Cross-check documentation prose against the machine inventory.

    Only presence checks run here. No documented number is parsed out and used
    as an arithmetic source; every component value comes from the inventory.
    """
    widths = inventory["widths"]
    components = {component["id"]: component for component in inventory["components"]}
    vision = components["vision_encoder"]
    audio = components["audio_encoder"]
    query = components["memory_query_output"]

    checks = {
        "backbone_unique_parameter_count_documented": (
            f"**{backbone_parameters:,} unique parameters**" in document
        ),
        "backbone_f2_parameter_count_documented": (
            f"**All {backbone_parameters:,} backbone parameters**" in document
        ),
        "vision_repository_documented": (
            f"`{vision['source']['repo_id']}`" in document
        ),
        "vision_revision_pinned": (
            f"`{vision['source']['revision']}`" in document
        ),
        "vision_parameter_count_documented": (
            f"{vision['parameters']:,} encoder parameters" in document
        ),
        "audio_repository_documented": (
            f"`{audio['source']['repo_id']}`" in document
        ),
        "audio_revision_pinned": (
            f"Pin SPEAR v2 at `{audio['source']['revision']}`" in document
        ),
        "audio_parameter_count_documented": (
            f"official metadata {audio['parameters']:,} parameters" in document
        ),
        "query_output_count_documented": (
            "Two Spark query/output projections alone add "
            f"{query['parameters']:,} weights" in document
        ),
        "integrated_total_documented": (
            f"{integrated_total_parameters:,}-parameter integrated inventory"
            in document
        ),
        "vision_projector_target_width_documented": (
            f"width {widths['backbone']}" in document
            and "spatial/temporal bridges" in document
        ),
        "audio_projector_width_documented": (
            f"width-{widths['audio_encoder']} output" in document
            and "spatial/temporal bridges" in document
        ),
        "memory_reader_width_documented": (
            f"Width-{widths['reader']} Q/K/V/O reader matrices" in document
        ),
        "two_port_reader_baseline_documented": "two-port baseline" in document,
        "phrase_encoder_width_documented": (
            f"width {widths['phrase']}, two layers" in document
        ),
        "phrase_encoder_rows_documented": (
            f"{widths['phrase']} active phrase rows" in document
        ),
    }
    _fail(
        "architecture documentation validation",
        [name for name, matched in checks.items() if not matched],
    )
    return {
        "checks": checks,
        "scope": (
            "presence cross-check only; every component value is owned by "
            "the versioned component inventory"
        ),
    }


def _projector_parameters(source_width: int, bridge_width: int, target_width: int) -> int:
    """Two affine stages with bias: source->bridge and bridge->target."""
    return (
        source_width * bridge_width
        + bridge_width
        + bridge_width * target_width
        + target_width
    )


def parameter_inventory(bf16: dict, inventory: dict) -> dict:
    """Explicit fixed-parameter inventory driven by the versioned JSON owner."""
    validate_component_inventory(inventory, bf16=bf16)
    widths = inventory["widths"]
    rows = []
    for component in inventory["components"]:
        component_id = component["id"]
        source = component["source"]
        source_kind = source["kind"]
        if component_id == "language_backbone":
            parameters = bf16["unique_parameter_elements"]
            evidence = {
                "path": source["path"],
                "field": source["field"],
                "tensor_count": bf16["tensor_count"],
                "kind": "sealed_tensor_index",
                "inventory_path": COMPONENTS_PATH,
            }
        elif source_kind == "declared_encoder_count":
            parameters = _positive_int(
                component["parameters"], f"{component_id} parameters"
            )
            evidence = {
                "path": COMPONENTS_PATH,
                "kind": "declared_encoder_count",
                "repo_id": source["repo_id"],
                "revision": source["revision"],
            }
        elif source_kind == "declared_count":
            parameters = _positive_int(
                component["parameters"], f"{component_id} parameters"
            )
            evidence = {
                "path": COMPONENTS_PATH,
                "kind": "declared_analytical_projection_count",
            }
        else:
            parameters = component_parameters(component, widths)
            formula_kind = source["formula"]["kind"]
            evidence = {
                "path": COMPONENTS_PATH,
                "kind": (
                    "explicit_two_stage_bias_formula"
                    if formula_kind == "two_stage_affine_with_bias"
                    else "explicit_formula_not_sealed"
                ),
                "formula": _formula_expression(component, widths),
            }
        rows.append(
            {
                "id": component_id,
                "label": component["label"],
                "parameters": parameters,
                "trainable": component["trainable"],
                "evidence": evidence,
            }
        )

    total = sum(row["parameters"] for row in rows)
    encoder_parameters = sum(
        row["parameters"]
        for row in rows
        if row["id"] in {"vision_encoder", "audio_encoder"}
    )
    return {
        "scope": "fixed trainable parameters; dynamic memory state is a separate account",
        "components": rows,
        "total_parameters": total,
        "backbone_parameters": bf16["unique_parameter_elements"],
        "declared_encoder_parameters": encoder_parameters,
        "notes": [
            "The two encoders are declared counts in the versioned component inventory and are cross-checked against the architecture document.",
            "Projector, reader and phrase counts are formula descriptors in the versioned component inventory, not measured modules.",
            "Dynamic fast/knowledge state and the bank tensor are excluded here and priced below.",
        ],
    }


def dynamic_memory_state() -> dict:
    """Explicit dynamic-memory state, not fixed trainable parameters."""
    bank_bytes = (
        MEMORY_BANK_RECORDS
        * MEMORY_BANK_ELEMENTS_PER_RECORD
        * DTYPE_BYTES["BF16"]
    )
    metadata_bytes = 2 * MIB
    f_parameters = (6 * MIB) // DTYPE_BYTES["FP32"]
    a_parameters = int(0.1875 * MIB) // DTYPE_BYTES["FP32"]
    p_parameters = 12_582_912
    p_bytes = p_parameters * DTYPE_BYTES["FP32"]
    p_moments = p_parameters * 2 * DTYPE_BYTES["FP32"]
    staging = 18_375 * KIB  # declared update-staging budget from the intrinsic ledger
    workspace = 36 * KIB
    return {
        "bank_tensor": {
            "records": MEMORY_BANK_RECORDS,
            "elements_per_record": MEMORY_BANK_ELEMENTS_PER_RECORD,
            "dtype": "BF16",
            "bytes": bank_bytes,
            "MiB": mib(bank_bytes),
        },
        "bank_metadata_cap_bytes": metadata_bytes,
        "fast_associative_F": {
            "parameters": f_parameters,
            "dtype": "FP32",
            "bytes": f_parameters * DTYPE_BYTES["FP32"],
        },
        "adaptive_addressing_A": {
            "parameters": a_parameters,
            "dtype": "FP32",
            "bytes": a_parameters * DTYPE_BYTES["FP32"],
        },
        "plastic_knowledge_P": {
            "parameters": p_parameters,
            "weight_bytes_fp32": p_bytes,
            "two_moment_bytes_fp32": p_moments,
        },
        "workspace_bytes": workspace,
        "update_staging_bytes": staging,
        "live_bytes": (
            bank_bytes
            + metadata_bytes
            + f_parameters * DTYPE_BYTES["FP32"]
            + a_parameters * DTYPE_BYTES["FP32"]
            + p_bytes
            + p_moments
            + workspace
        ),
        "update_peak_bytes": (
            bank_bytes
            + metadata_bytes
            + f_parameters * DTYPE_BYTES["FP32"]
            + a_parameters * DTYPE_BYTES["FP32"]
            + p_bytes
            + p_moments
            + workspace
            + staging
        ),
        "note": (
            "Dynamic state is not part of the 18-byte fixed-parameter rate. "
            "It is charged at its declared dtypes and update staging; replay "
            "and raw-source storage are excluded."
        ),
    }


def training_plan_row(plan: dict, parameters: int) -> dict:
    copies = [
        {"copy": name, "dtype": dtype, "count": count, "bytes": DTYPE_BYTES[dtype] * count}
        for name, dtype, count in plan["copies"]
    ]
    bytes_per_parameter = sum(copy["bytes"] for copy in copies)
    return {
        "id": plan["id"],
        "label": plan["label"],
        "copies": copies,
        "dtype_breakdown": {
            dtype: sum(copy["bytes"] for copy in copies if copy["dtype"] == dtype)
            for dtype in sorted({copy["dtype"] for copy in copies})
        },
        "bytes_per_parameter": bytes_per_parameter,
        "formula": " + ".join(f"{copy['bytes']}" for copy in copies) + " bytes/parameter",
        "parameters": parameters,
        "total_bytes": parameters * bytes_per_parameter,
        "total_GiB": gib(parameters * bytes_per_parameter),
        "note": plan["note"],
    }


def activation_inventory(widths: dict) -> dict:
    """Declared analytical activation/logit/temporary model.

    These are byte formulas over the declared workload, not measured peaks.
    They are included so the ledger is not a weights-only lower bound.
    """
    checkpoint_inputs = (
        LAYERS * MICROBATCH_POSITIONS * widths["backbone"] * DTYPE_BYTES["BF16"]
    )
    attention_workspace = (
        2
        * MICROBATCH_POSITIONS
        * QUERY_HEADS
        * HEAD_DIM
        * DTYPE_BYTES["FP32"]
    )
    mlp_workspace = (
        2 * MICROBATCH_POSITIONS * FFN_WIDTH * DTYPE_BYTES["FP32"]
    )
    vision_workspace = (
        3 * VISION_PATCHES * widths["vision_encoder"] * DTYPE_BYTES["FP32"]
    )
    audio_workspace = 3 * AUDIO_TOKENS * widths["audio_encoder"] * DTYPE_BYTES["FP32"]
    activations = (
        checkpoint_inputs
        + attention_workspace
        + mlp_workspace
        + vision_workspace
        + audio_workspace
    )
    logits = (
        3 * MICROBATCH_POSITIONS * VOCAB * DTYPE_BYTES["FP32"]
    )
    return {
        "workload": {
            "microbatch_positions": MICROBATCH_POSITIONS,
            "hidden_width": widths["backbone"],
            "ffn_width": FFN_WIDTH,
            "layers": LAYERS,
            "query_heads": QUERY_HEADS,
            "head_dim": HEAD_DIM,
            "vision_patches": VISION_PATCHES,
            "audio_tokens": AUDIO_TOKENS,
            "vocab": VOCAB,
        },
        "items": [
            {
                "id": "checkpointed_layer_inputs",
                "bytes": checkpoint_inputs,
                "formula": "layers * positions * hidden * 2",
            },
            {
                "id": "attention_recompute_workspace",
                "bytes": attention_workspace,
                "formula": "2 * positions * query_heads * head_dim * 4",
            },
            {
                "id": "mlp_recompute_workspace",
                "bytes": mlp_workspace,
                "formula": "2 * positions * ffn_width * 4",
            },
            {
                "id": "vision_feature_workspace",
                "bytes": vision_workspace,
                "formula": "3 * patches * vision_width * 4",
            },
            {
                "id": "audio_feature_workspace",
                "bytes": audio_workspace,
                "formula": "3 * audio_tokens * audio_width * 4",
            },
        ],
        "activation_bytes": activations,
        "activation_MiB": mib(activations),
        "logits": {
            "bytes": logits,
            "formula": "3 * positions * vocab * 4 (logits, softmax buffer, gradient)",
            "MiB": mib(logits),
        },
        "role": "analytical estimate with activation checkpointing; not a measured allocator peak",
    }


def temporary_buffers(parameters: int) -> dict:
    clip = parameters * DTYPE_BYTES["FP32"]
    fused_optimizer_scratch = 256 * MIB
    feature_queue = 64 * MIB
    total = clip + fused_optimizer_scratch + feature_queue
    return {
        "items": [
            {
                "id": "gradient_clip_reduction",
                "bytes": clip,
                "formula": "parameters * 4",
                "kind": "analytical",
            },
            {
                "id": "fused_optimizer_scratch",
                "bytes": fused_optimizer_scratch,
                "formula": "256 MiB declared budget",
                "kind": "analytical_declared_budget",
            },
            {
                "id": "encoder_feature_queue",
                "bytes": feature_queue,
                "formula": "64 MiB declared budget",
                "kind": "analytical_declared_budget",
            },
        ],
        "total_bytes": total,
        "total_MiB": mib(total),
        "role": "explicit temporary-buffer budget; allocator slack and driver overhead remain excluded",
    }


def phase_peaks(parameters: int, source_bf16_bytes: int, source_fp8_bytes: int) -> dict:
    """Restore and export peaks over the declared fixed parameter count."""
    bf16_weights = parameters * DTYPE_BYTES["BF16"]
    fp32_master = parameters * DTYPE_BYTES["FP32"]
    fp32_gradients = parameters * DTYPE_BYTES["FP32"]
    fp32_moments = parameters * 2 * DTYPE_BYTES["FP32"]
    fp8_export = parameters * DTYPE_BYTES["F8_E4M3"]
    bf16_export = bf16_weights

    restore_bf16 = source_bf16_bytes + bf16_weights + fp32_master
    restore_fp8 = source_fp8_bytes + bf16_weights + fp32_master
    export = fp32_master + bf16_export + fp8_export
    training_live = (
        bf16_weights + fp32_master + fp32_gradients + fp32_moments
    )
    return {
        "restore_from_bf16_source": {
            "bytes": restore_bf16,
            "GiB": gib(restore_bf16),
            "formula": "BF16 artifact + materialized BF16 weights + FP32 master",
            "status": "analytical; loader and dequantization unqualified",
        },
        "restore_from_fp8_source": {
            "bytes": restore_fp8,
            "GiB": gib(restore_fp8),
            "formula": "mixed-dtype FP8 artifact + BF16 weights + FP32 master",
            "status": "analytical; FP8 loader, dequantization and backend unqualified",
        },
        "export_bf16_and_fp8": {
            "bytes": export,
            "GiB": gib(export),
            "formula": "FP32 master + BF16 export copy + FP8 staging copy",
            "status": "analytical; serialization and backend export unqualified",
        },
        "training_live_parameter_state": {
            "bytes": training_live,
            "GiB": gib(training_live),
            "formula": "18-byte/parameter five-copy plan",
        },
    }


def build_ledger(root: Path, inventory: dict | None = None) -> dict:
    bf16_header = load_json(root, BF16_HEADER_PATH)
    fp8_header = load_json(root, FP8_HEADER_PATH)
    guard_receipt = load_json(root, GUARD_PATH)
    if inventory is None:
        inventory = load_component_inventory(root)
    else:
        validate_component_inventory(inventory)

    documentation_path = inventory["documentation"]["path"]
    documentation_file = root / documentation_path
    if not documentation_file.is_file():
        raise ResidencyError(
            f"required architecture documentation missing: {documentation_path}"
        )
    architecture_text = documentation_file.read_text(encoding="utf-8")
    inventory_sha256 = sha256_file(root / COMPONENTS_PATH)
    documentation_sha256 = sha256_file(documentation_file)

    bf16 = audit_bf16_header(bf16_header)
    fp8 = audit_fp8_header(fp8_header)
    parameter_rows = parameter_inventory(bf16, inventory)
    parameters = parameter_rows["total_parameters"]
    documentation_validation = validate_architecture_documentation(
        architecture_text,
        inventory,
        bf16["unique_parameter_elements"],
        parameters,
    )
    widths = inventory["widths"]
    dynamic = dynamic_memory_state()

    guard = guard_receipt["guard"]["policy"]["driver_limit_bytes"]
    if guard != DRIVER_GUARD_GIB * GIB:
        raise ResidencyError(f"driver guard policy mismatch: {guard} != {DRIVER_GUARD_GIB} GiB")
    if guard_receipt["guard"]["policy"]["swap_growth_limit_bytes"] != SUSTAINED_SWAP_LIMIT:
        raise ResidencyError("sustained swap-growth policy changed; review before recomputing")
    if guard_receipt["guard"]["policy"]["minimum_free_percent"] != MINIMUM_FREE_PERCENT:
        raise ResidencyError("minimum free-percent policy changed; review before recomputing")

    plans = [training_plan_row(plan, parameters) for plan in TRAINING_PLANS]
    plan_by_id = {plan["id"]: plan for plan in plans}
    corrected = plan_by_id[
        "bf16_weights_fp32_master_fp32_gradients_two_fp32_moments"
    ]
    all_fp32 = plan_by_id["all_fp32"]
    _fail(
        "training plan arithmetic",
        [
            "five_copy_rate_is_18" if corrected["bytes_per_parameter"] != 18 else "",
            "five_copy_formula_is_2_4_4_4_4"
            if corrected["formula"] != "2 + 4 + 4 + 4 + 4 bytes/parameter"
            else "",
            "all_fp32_rate_is_16" if all_fp32["bytes_per_parameter"] != 16 else "",
        ],
    )

    activations = activation_inventory(widths)
    temporary = temporary_buffers(parameters)
    peaks = phase_peaks(
        parameters,
        bf16["serialized_payload_bytes"],
        fp8["serialized_payload_bytes"],
    )

    projected_training = (
        corrected["total_bytes"]
        + dynamic["update_peak_bytes"]
        + activations["activation_bytes"]
        + activations["logits"]["bytes"]
        + temporary["total_bytes"]
    )
    projected_training_with_restore = projected_training + peaks["restore_from_bf16_source"]["bytes"]
    projected_training_with_export = projected_training + peaks["export_bf16_and_fp8"]["bytes"]

    return {
        "format": "replica.sparkx_residency.v2",
        "kind": "analytical_dtype_explicit_residency_ledger_not_native_measurement",
        "receipt_version": 2,
        "regeneration": "deterministic; re-running the calculator reproduces this file byte for byte",
        "calculator": {
            "path": CALCULATOR_PATH,
            "sha256": sha256_file(root / CALCULATOR_PATH),
            "language": "python3 standard library only",
            "invocation": f"python3 {CALCULATOR_PATH}",
        },
        "supersession": {
            "supersedes": SUPERSEDED_RECEIPT_PATH,
            "superseded_sha256": (
                sha256_file(root / SUPERSEDED_RECEIPT_PATH)
                if (root / SUPERSEDED_RECEIPT_PATH).is_file()
                else None
            ),
            "historical_receipt_untouched": True,
            "legacy_markdown_receipt": LEGACY_MARKDOWN_RECEIPT_PATH,
            "legacy_markdown_sha256": (
                sha256_file(root / LEGACY_MARKDOWN_RECEIPT_PATH)
                if (root / LEGACY_MARKDOWN_RECEIPT_PATH).is_file()
                else None
            ),
            "legacy_markdown_preserved": True,
            "migration": (
                "component counts and formulas moved from Markdown sentences "
                "to the versioned machine-readable component inventory"
            ),
            "reason": (
                "v1 labelled the five-copy BF16-weight + FP32-master row as "
                "16 bytes/parameter although its own formula was 2+4+4+4+4 = 18. "
                "v1 also treated FP8 as parameters x 1 byte and omitted "
                "activations, logits, temporary buffers and phase peaks."
            ),
        },
        "driver_guard": {
            "driver_limit_bytes": guard,
            "driver_limit_GiB": guard / GIB,
            "policy": "50 GiB sampled driver stop, unchanged repository policy",
            "scope": "the guard bounds the native worker process, not the model alone",
            "companion_limits": {
                "sustained_swap_growth_limit_bytes": SUSTAINED_SWAP_LIMIT,
                "minimum_free_percent": MINIMUM_FREE_PERCENT,
                "note": "neither companion limit is evaluated by this calculator",
            },
        },
        "source_storage": {
            "role": "artifact bytes on disk; not resident training state and not a capability claim",
            "bf16_source": {
                "path": BF16_HEADER_PATH,
                "repo_id": bf16_header.get("repo_id"),
                "revision": bf16_header.get("revision"),
                "storage": bf16_header.get("storage"),
                "payload_bytes": bf16["serialized_payload_bytes"],
                "dtype_payload_bytes": {"BF16": bf16["serialized_payload_bytes"]},
                "unique_parameter_elements": bf16["unique_parameter_elements"],
                "note": "sealed BF16 source; full-training reference artifact",
            },
            "fp8_source": {
                "path": FP8_HEADER_PATH,
                "repo_id": fp8_header.get("repo_id"),
                "revision": fp8_header.get("revision"),
                "storage": fp8_header.get("storage"),
                "payload_bytes": fp8["serialized_payload_bytes"],
                "dtype_payload_bytes": fp8["dtype_payload_bytes"],
                "dtype_tensor_counts": fp8["dtype_tensor_counts"],
                "f8_matrix_count": fp8["f8_matrix_count"],
                "scale_tensor_count": fp8["scale_tensor_count"],
                "scale_payload_bytes": fp8["scale_payload_bytes"],
                "bf16_category_payload_bytes": fp8["bf16_category_payload_bytes"],
                "note": (
                    "mixed dtype, not parameters x 1 byte: F8_E4M3 matrices plus "
                    "BF16 scales, tied embedding, norms and attention gates. It is "
                    "kept separate from BF16 source storage and from any training "
                    "or backend capability. FP8 loader, dequantization and training "
                    "support remain unqualified."
                ),
            },
            "fp8_payload_is_not_one_byte_per_parameter": True,
            "fp8_training_capability": "unqualified",
        },
        "bf16_header_audit": bf16,
        "fp8_header_audit": fp8,
        "component_inventory": {
            "path": COMPONENTS_PATH,
            "sha256": inventory_sha256,
            "schema_version": inventory["schema_version"],
            "kind": inventory["kind"],
            "status": inventory["status"],
            "role": "sole arithmetic owner for declared component counts and widths",
        },
        "documentation_validation": {
            "path": documentation_path,
            "sha256": documentation_sha256,
            "against_inventory": {
                "path": COMPONENTS_PATH,
                "sha256": inventory_sha256,
            },
            "checks": documentation_validation["checks"],
            "scope": documentation_validation["scope"],
        },
        "parameter_inventory": parameter_rows,
        "training_storage_plans": plans,
        "corrected_critical_rate": {
            "plan": corrected["id"],
            "bytes_per_parameter": 18,
            "formula": "2 + 4 + 4 + 4 + 4",
            "statement": (
                "BF16 weights + FP32 master weights + FP32 gradients + two FP32 "
                "moments coexist at 18 bytes/parameter. The 16-byte figure is the "
                "separate all-FP32 plan and must not be used for the five-copy plan."
            ),
        },
        "dynamic_memory_state": dynamic,
        "activations_and_logits": activations,
        "temporary_buffers": temporary,
        "phase_peaks": peaks,
        "projected_full_scope_training": {
            "scope": parameter_rows["scope"],
            "fixed_parameters": parameters,
            "training_plan": corrected["id"],
            "parameter_state_bytes": corrected["total_bytes"],
            "dynamic_state_bytes": dynamic["update_peak_bytes"],
            "activation_bytes": activations["activation_bytes"],
            "logits_bytes": activations["logits"]["bytes"],
            "temporary_buffer_bytes": temporary["total_bytes"],
            "resident_training_bytes": projected_training,
            "resident_training_GiB": gib(projected_training),
            "with_bf16_restore_peak_bytes": projected_training_with_restore,
            "with_bf16_restore_peak_GiB": gib(projected_training_with_restore),
            "with_export_peak_bytes": projected_training_with_export,
            "with_export_peak_GiB": gib(projected_training_with_export),
            "over_driver_guard_on_resident_training": projected_training >= guard,
            "headroom_to_driver_guard_bytes": guard - projected_training,
            "headroom_to_driver_guard_GiB": gib(guard - projected_training),
            "status": "analytical_not_admitted",
            "note": (
                "Parameter state, dynamic state, analytical activations, logits "
                "and temporary buffers are summed. Allocator and driver overhead "
                "are still excluded, so the true peak can only be larger."
            ),
        },
        "excluded_from_every_figure": EXCLUSIONS,
        "gates": {
            "loader": "unqualified",
            "dequantization": "unqualified",
            "backend": "unqualified",
            "resource": "unqualified",
            "native_execution": "not_run",
            "admission": "not_admitted",
            "statement": (
                "No loader, dequantizer, FP8/BF16 backend, MPS resource gate or "
                "native training run is qualified by this receipt. A value below "
                "the driver guard is not a measured fit."
            ),
        },
        "provisioning": {
            "required_inputs_tracked": [
                BF16_HEADER_PATH,
                FP8_HEADER_PATH,
                GUARD_PATH,
                COMPONENTS_PATH,
                ARCHITECTURE_PATH,
            ],
            "unprovisioned_local_inputs": [],
            "bundle_check_command": "python3 scripts/prepare_sparkx_research_bundle.py --check",
            "clean_checkout_replay": (
                "all residency inputs, including the versioned component inventory, "
                "are repository-tracked; this receipt pins the inventory SHA-256 "
                "and the bundle check verifies the sealed inputs listed in its "
                "manifest"
            ),
            "native_execution_required": False,
        },
        "non_claims": [
            "no Spark weight payload, tokenizer, dataset, media asset, model code or native runtime was downloaded, loaded or executed",
            "encoder parameter counts are declared in the versioned component inventory and cross-checked against the architecture document, not a fresh sealed encoder tensor index",
            "projector, reader and phrase figures are formula descriptors in the versioned component inventory; activation, logit and temporary-buffer figures are analytical formulas, not measurements",
            "the mixed-dtype FP8 source artifact does not demonstrate FP8 loading, dequantization, full backprop or backend support",
            "no MPS memory fit, allocator peak, throughput, convergence or training admission is established",
        ],
    }


def calculate(root: Path | None = None) -> dict:
    return build_ledger(Path(root) if root is not None else ROOT)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--check",
        action="store_true",
        help="recompute and compare against the committed receipt without writing",
    )
    args = parser.parse_args(argv)

    receipt = calculate(args.root)
    output = args.output or (args.root / RECEIPT_PATH)
    payload = json.dumps(receipt, indent=2) + "\n"
    if args.check:
        if not output.is_file():
            raise SystemExit(f"receipt missing for --check: {output}")
        if output.read_text(encoding="utf-8") != payload:
            raise SystemExit(f"receipt is stale or non-deterministic: {output}")
        print(f"Receipt verified: {output}")
        return 0

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(payload, encoding="utf-8")
    print(
        "Wrote "
        f"{output}: {receipt['parameter_inventory']['total_parameters']} fixed "
        f"parameters, corrected rate "
        f"{receipt['corrected_critical_rate']['bytes_per_parameter']} B/param, "
        f"resident training {receipt['projected_full_scope_training']['resident_training_GiB']} GiB "
        "(analytical only; no native job executed)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

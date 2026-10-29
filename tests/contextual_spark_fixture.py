"""Compatibility re-export of the contextual fixture now owned by the package.

Production runners must import ``replica_lab.contextual_spark_fixture`` rather
than depending on ``tests/``.  This thin module keeps existing test imports
working without carrying a second implementation.
"""
from replica_lab.contextual_spark_fixture import (
    DEFAULT_LAYERS, ENGINEERING_LABEL, LATE_SITE_MIN_LAYERS,
    NATIVE_SEALED_LAYERS, ContextualBlock, ContextualCausalAttention,
    ContextualSparkCausalLM, ContextualSparkModel, build_contextual_adapter,
    build_contextual_path, contextual_config_raw, ids_and_mask, labels_from_ids,
    open_memory_ports, populate_contextual_evidence, prefix_ids,
    token_mixing_probe,
)

__all__ = [
    "DEFAULT_LAYERS", "ENGINEERING_LABEL", "LATE_SITE_MIN_LAYERS",
    "NATIVE_SEALED_LAYERS", "ContextualBlock", "ContextualCausalAttention",
    "ContextualSparkCausalLM", "ContextualSparkModel",
    "build_contextual_adapter", "build_contextual_path",
    "contextual_config_raw", "ids_and_mask", "labels_from_ids",
    "open_memory_ports", "populate_contextual_evidence", "prefix_ids",
    "token_mixing_probe",
]

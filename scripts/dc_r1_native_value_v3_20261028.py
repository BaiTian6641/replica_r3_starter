"""Public DC-R1 v3 protocol and review entry points."""
from __future__ import annotations

from replica_lab.dc_r1_v3_scoring import (  # noqa: F401
    CONDITIONS, LAYERS, PREDICTION_FORMAT, PROTOCOL_ID, SCORE_FORMAT,
    load_config, score_predictions, validate_prepared, verify_development_evidence,
)
from preflight_dc_r1_native_value_v3_20261027 import preflight, workload  # noqa: F401

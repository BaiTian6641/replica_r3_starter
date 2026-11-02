#!/usr/bin/env python3
"""Prepare attempt04 with a refreshed executable implementation closure."""
from __future__ import annotations

import prepare_spark_interference_v17_attempt03 as base


def prepare():
    base.CONFIG = "configs/spark_interference_diagnostic_v17_attempt04.json"
    base.OUTPUT = "reports/intrinsic_memory_interference_v17/native_bf16_interference_attempt04"
    base.IMPLEMENTATION = "reports/intrinsic_memory_interference_v17/implementation_attempt04.json"
    base.PACKET_PATH = "reports/intrinsic_memory_interference_v17/native_packet_attempt04.json"
    base.AUTHORIZATION = "reports/intrinsic_memory_interference_v17/authorization_attempt04.json"
    base.PREPARATION = "reports/intrinsic_memory_interference_v17/preparation_attempt04.json"
    return base.prepare()


if __name__ == "__main__":
    print(prepare())

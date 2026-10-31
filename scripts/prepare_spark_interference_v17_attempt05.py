#!/usr/bin/env python3
"""Prepare attempt05 with all diagnostic and accounting repairs."""
from __future__ import annotations

import prepare_spark_interference_v17_attempt03 as base


def prepare():
    base.CONFIG = "configs/spark_interference_diagnostic_v17_attempt05.json"
    base.OUTPUT = "reports/intrinsic_memory_interference_v17/native_bf16_interference_attempt05"
    base.IMPLEMENTATION = "reports/intrinsic_memory_interference_v17/implementation_attempt05.json"
    base.PACKET_PATH = "reports/intrinsic_memory_interference_v17/native_packet_attempt05.json"
    base.AUTHORIZATION = "reports/intrinsic_memory_interference_v17/authorization_attempt05.json"
    base.PREPARATION = "reports/intrinsic_memory_interference_v17/preparation_attempt05.json"
    return base.prepare()


if __name__ == "__main__":
    print(prepare())

#!/usr/bin/env python3
"""Retain bounded synthetic CPU reproduction and comparison artifacts."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from replica_lab.intrinsic_memory_cpu_fixture_v18 import make_engine, fixture_rows
from replica_lab.intrinsic_memory_interference_v17 import run_pair
from replica_lab.intrinsic_memory_paired_update_v18 import run_comparison
from replica_lab.intrinsic_memory_budget_v12 import atomic, audit_io


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    receipts = {}
    for name in ("stale_slot_reproduction", "paired_comparison"):
        engine = make_engine(args.output / name)
        start = time.monotonic()
        try:
            result = (run_pair(engine, *fixture_rows()[:2], 0, engine.output)
                      if name == "stale_slot_reproduction" else run_comparison(engine, fixture_rows()))
            snapshot = engine.ledger.snapshot()
            receipt = {"scope": "independent_synthetic_CPU_only", "wall_seconds": time.monotonic()-start,
                       "reserved": dict(snapshot["reserved"]), "audit_io": audit_io(engine.ledger),
                       "path_cost": dict(engine.path.cost), "source_cost": dict(engine.path.feature_provider.cost),
                       "includes_one_fixture_moment_initialization_update": True,
                       "two_step_cpu_float64_reduction": True,
                       "native_model_loads": 0, "development_access": False}
            if name == "stale_slot_reproduction":
                receipt["live_minus_frozen_margins"] = [
                    live["margin"]["total_log_likelihood"] - frozen["margin"]["total_log_likelihood"]
                    for arm in result["arms"] for live, frozen in zip(arm["after"], arm["frozen_slot_intervention"])]
            else:
                receipt["endpoints"] = result["endpoints"]
                receipt["relative_conditions_passed"] = result["relative_conditions_passed"]
            raw = (json.dumps(receipt, indent=2, sort_keys=True)+"\n").encode()
            with engine.measured("cpu_validation_receipt", writes=1, artifact_bytes=len(raw)):
                atomic(engine.output / "validation.json", receipt)
            engine.ledger.leave("fixture", status="completed")
            receipts[name] = receipt
        finally:
            engine.path.close()
    print(json.dumps(receipts, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Measure real bounded tiered readout, cache pressure and cold revalidation."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile

from replica_lab.tiered_memory import (
    EvidenceRecordDraft, MemoryScope, TierCeilings, TieredMemoryStore,
)
from replica_lab.tiered_readout import (
    ReadoutBudget, read_scoped_memory, readout_cache_state, resume_scoped_readout,
)


def values(text: str) -> tuple[tuple[float, ...], ...]:
    payload = hashlib.shake_256(text.encode()).digest(4 * 256)
    return tuple(tuple((byte - 127.5) / 127.5 for byte in payload[index * 256:(index + 1) * 256])
                 for index in range(4))


def draft(record_id: str, text: str, epoch: int) -> EvidenceRecordDraft:
    digest = hashlib.sha256(text.encode()).hexdigest()
    return EvidenceRecordDraft(
        record_id=record_id, text=text, source_id="pressure-fixture",
        source_epoch=epoch, observed_at=epoch, event_at=epoch,
        valid_from=epoch, authenticity="source_report", verification_state="verified_source",
        values=values(text), value_producer_id="pressure_synthetic_values",
        value_producer_version="v1",
        source_lineage=(json.dumps({"source": "pressure_fixture", "record_id": record_id,
                                    "text_sha256": digest}),))


def latency_summary(samples: list[int]) -> dict[str, int | None]:
    values = sorted(int(value) for value in samples)
    if not values:
        return {"count": 0, "p50_ns": None, "p95_ns": None, "max_ns": None}
    return {
        "count": len(values),
        "p50_ns": values[(50 * len(values) + 99) // 100 - 1],
        "p95_ns": values[(95 * len(values) + 99) // 100 - 1],
        "max_ns": values[-1],
    }

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing receipt: {args.output}")
    root = Path(tempfile.mkdtemp(prefix="replica-readout-pressure-"))
    ceilings = TierCeilings(
        hot_bytes=64 * 1024**2, warm_bytes=64 * 1024**2, cold_bytes=64 * 1024**2,
        identity_reserved_bytes=64 * 1024, replay_reserved_bytes=256 * 1024,
        minimum_volume_free_bytes=0, workspace_quota_bytes=1024**2,
        artifact_budget_bytes=1024**2)
    scope = MemoryScope(subject_id="user_a", branch_id="main", purpose="assistant", as_of=1000)
    budget = ReadoutBudget(max_records=2, max_value_slots=8, cache_bytes=10000,
                           cold_wait_seconds=0)
    with TieredMemoryStore(root, ceilings=ceilings, enforce_volume_free=False) as store:
        store.admit_cluster(
            "warm", scope=scope, placement="warm",
            records=[draft("warm-alpha", "alpha warm payload", 1),
                     draft("warm-beta", "beta warm payload", 1),
                     draft("warm-gamma", "gamma warm payload", 1)])
        store.admit_cluster("cold", scope=scope, placement="cold",
                            records=[draft("cold-rule", "cold original rule", 1)])
        first = read_scoped_memory(store, "alpha warm", scope, budget=budget)
        second = read_scoped_memory(store, "alpha warm", scope, budget=budget)
        third = read_scoped_memory(store, "beta warm", scope, budget=budget)
        fourth = read_scoped_memory(store, "gamma warm", scope, budget=budget)
        cold_pending = read_scoped_memory(store, "cold original", scope, budget=budget)
        corrected_text = "cold corrected rule"
        correction_provenance = {
            "record_id": "cold-rule-corrected",
            "source": "pressure-correction",
            "text_sha256": hashlib.sha256(corrected_text.encode()).hexdigest(),
            "producer_revision": "pressure-neural-producer-v1",
            "representation_version": "pressure-representation-v1",
            "codec_version": "pressure-codec-v1",
            "payload_type": "observation",
        }
        correction = store.correct_record(
            "cold-rule", correction_id="cold-rule-corrected", text=corrected_text,
            source_id="pressure-correction", source_epoch=2, observed_at=2,
            scope=scope, authenticity="source_report", verification_state="verified_source",
            source_provenance=correction_provenance)
        invalidated = resume_scoped_readout(store, cold_pending.pending_cold[0].ticket_id,
                                            scope=scope, budget=budget)
        corrected = read_scoped_memory(store, "cold corrected", scope, budget=budget)
        corrected_ready = resume_scoped_readout(store, corrected.pending_cold[0].ticket_id,
                                                scope=scope, budget=budget)
        report = {
            "format": "replica.tiered_readout.pressure.v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "root": str(root),
            "budget": budget.document(),
            "warm_first": first.document(),
            "warm_second_cached": second.document(),
            "warm_third_eviction": third.document(),
            "warm_fourth_eviction": fourth.document(),
            "cold_pending": cold_pending.document(),
            "correction": correction.document(),
            "cold_invalidated_after_io": invalidated.document(),
            "cold_corrected": corrected.document(),
            "cold_corrected_ready": corrected_ready.document(),
            "cache_state": readout_cache_state(store),
            "latency": {
                "wall_clock": True,
                "warm_readout": latency_summary([
                    first.metrics.total_ns, second.metrics.total_ns,
                    third.metrics.total_ns, fourth.metrics.total_ns]),
                "cold_readout": latency_summary([
                    cold_pending.metrics.total_ns, invalidated.metrics.total_ns,
                    corrected_ready.metrics.total_ns]),
                "end_to_end_answer_cost": "not_measured_by_this_probe",
            },
            "placement_labels_not_residency": True,
            "gpu_resident": False,
            "artificial_delay": False,
            "larger_bank_allocated": False,
            "cache_is_in_process_copy": True,
            "native_execution": False,
            "semantic_compression_claimed": False,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"output": str(args.output),
                          "warm_cache_hits": second.metrics.cache_hits,
                          "warm_cache_evictions": fourth.metrics.cache_evictions,
                          "cold_pending": bool(cold_pending.pending_cold),
                          "cold_invalidated": invalidated.new_decision_boundary,
                          "corrected_record_ids": corrected_ready.record_ids}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

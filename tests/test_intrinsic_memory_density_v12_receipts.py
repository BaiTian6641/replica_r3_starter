"""Coherent receipt generation refuses overwrite and binds actual pilot values."""
import json

import pytest

from replica_lab.intrinsic_memory_density_protocols_v12 import (
    canonical_digest, load_protocols,
)
from replica_lab.intrinsic_memory_density_receipts_v12 import (
    PROTOCOL_SUMMARY, REPORT_MANIFEST, VERIFICATION_RESULTS,
    ReceiptError, write_new_json, write_repair_receipts,
)


def inputs(tmp_path):
    protocols = load_protocols("configs")
    digest = canonical_digest(protocols["preregistered_measured_total_cost"])
    cost = {
        "format": "test.pilot", "total_cost_protocol_sha256": digest,
        "target_cost_ns": 123.5, "selected_updates": {"two": 2, "all": 1},
        "matched_arms": ["two", "all"], "unmatched_arms": [],
        "scored_labels_read": False, "cost_metric": "measured_wall",
        "repeat": 3, "aggregate": "median_of_repeats",
        "declared_common_updates": 2, "declared_updates_cap": 8,
        "out_of_tolerance_control": {"outside": True},
    }
    cpu = {
        "format": "test.profile",
        "protocol_digests": {name: canonical_digest(protocol)
                             for name, protocol in protocols.items()},
        "arms": {"two": {
            "sites": [3, 7], "physical_reader_groups": 2, "reader_invocations": 2,
            "parameter_audit": {"reader_parameters": 272418,
                                "intended_parameters": 887090,
                                "optimizer_parameter_tensors": 43},
            "post_update": {"logits_sha256": "a" * 64,
                            "logits_changed_from_pre_update": True,
                            "reader_invocations": 2},
            "no_update_control": {"logits_sha256": "b" * 64,
                                  "unchanged_from_pre_update": True},
        }},
        "matched_arms": {},
        "fresh_process_continuation": {}, "matched_fresh_process_continuation": {},
    }
    cpu_path, cost_path = tmp_path / "cpu.json", tmp_path / "cost.json"
    write_new_json(cpu_path, cpu)
    write_new_json(cost_path, cost)
    return cpu_path, cost_path, cost


def test_receipts_reproduce_actual_pilot_target_and_refuse_overwrite(tmp_path):
    cpu, cost_path, cost = inputs(tmp_path)
    destination = tmp_path / "repair"
    manifest = write_repair_receipts(
        destination, cpu_summary_path=cpu, cost_pilot_path=cost_path)
    protocol = json.loads((destination / PROTOCOL_SUMMARY).read_text())
    verification = json.loads((destination / VERIFICATION_RESULTS).read_text())
    assert protocol["cost_pilot"]["target_cost_ns"] == cost["target_cost_ns"]
    assert verification["measured_total_cost_pilot"]["target_cost_ns"] == (
        cost["target_cost_ns"])
    assert verification["measured_total_cost_pilot"]["selected_updates"] == (
        cost["selected_updates"])
    assert verification["measured_total_cost_pilot"]["out_of_tolerance_control"] == (
        cost["out_of_tolerance_control"])
    assert manifest["outputs"][PROTOCOL_SUMMARY]
    assert json.loads((destination / REPORT_MANIFEST).read_text()) == manifest
    second = tmp_path / "repair-again"
    write_repair_receipts(second, cpu_summary_path=cpu, cost_pilot_path=cost_path)
    assert (destination / PROTOCOL_SUMMARY).read_bytes() == (
        second / PROTOCOL_SUMMARY).read_bytes()
    assert (destination / VERIFICATION_RESULTS).read_bytes() == (
        second / VERIFICATION_RESULTS).read_bytes()
    with pytest.raises(FileExistsError):
        write_repair_receipts(destination, cpu_summary_path=cpu,
                              cost_pilot_path=cost_path)


def test_receipts_refuse_a_pilot_with_a_different_protocol_digest(tmp_path):
    cpu, cost_path, cost = inputs(tmp_path)
    changed = dict(cost)
    changed["total_cost_protocol_sha256"] = "0" * 64
    write_new_json(tmp_path / "changed.json", changed)
    with pytest.raises(ReceiptError):
        write_repair_receipts(
            tmp_path / "repair", cpu_summary_path=cpu,
            cost_pilot_path=tmp_path / "changed.json")

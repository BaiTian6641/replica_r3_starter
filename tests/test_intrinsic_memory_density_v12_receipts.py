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
        "out_of_tolerance_control": {
            "deliberate_out_of_tolerance_observed": True,
            "charged_to_arm": "all", "control_cost_ns": 10,
            "arithmetic_control": {"outside_declared_tolerance": True},
        },
    }
    def row(sites, groups, reader_parameters, token):
        return {
            "sites": sites, "physical_reader_groups": groups,
            "reader_invocations": len(sites),
            "parameter_audit": {"reader_parameters": reader_parameters,
                                "intended_parameters": 887090,
                                "optimizer_parameter_tensors": 43,
                                "reader_groups": {"state_tensors": 22,
                                                  "alias_duplicated_state_tensors": 0}},
            "post_update": {"logits_sha256": token * 64,
                            "logits_changed_from_pre_update": True,
                            "reader_invocations": len(sites)},
            "no_update_control": {"logits_sha256": "b" * 64,
                                  "unchanged_from_pre_update": True},
            "feature_contract": {
                "input_policy": "canonical_span_only_no_template_no_special_tokens",
                "output_policy": "last_decoder_hidden_after_final_norm",
                "original_updates": 0,
            },
        }
    cpu = {
        "format": "test.profile",
        "protocol_digests": {name: canonical_digest(protocol)
                             for name, protocol in protocols.items()},
        "arms": {"two": row([3, 7], 2, 272418, "a"),
                 "all": row(list(range(36)), 36, 4903524, "c")},
        "matched_arms": {"two": row([3, 7], 2, 272418, "d"),
                         "all": row(list(range(36)), 2, 272418, "e")},
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


def test_partial_write_leaves_no_final_path_and_retry_succeeds(monkeypatch, tmp_path):
    import os
    from replica_lab.intrinsic_memory_density_receipts_v12 import write_new_json
    destination = tmp_path / "receipt.json"
    real_fsync = os.fsync

    def interrupted(descriptor):
        raise RuntimeError("interrupted")

    monkeypatch.setattr(os, "fsync", interrupted)
    with pytest.raises(RuntimeError):
        write_new_json(destination, {"complete": True})
    assert not destination.exists()
    monkeypatch.setattr(os, "fsync", real_fsync)
    write_new_json(destination, {"complete": True})
    assert json.loads(destination.read_text()) == {"complete": True}


@pytest.mark.parametrize("damage", ("no_update", "post_update", "feature_contract", "control_charge"))
def test_receipts_enforce_controls_and_feature_contract(tmp_path, damage):
    cpu, cost_path, cost = inputs(tmp_path)
    if damage in ("no_update", "post_update", "feature_contract"):
        value = json.loads(cpu.read_text())
        row = value["arms"]["two"]
        if damage == "no_update":
            row["no_update_control"]["unchanged_from_pre_update"] = False
        elif damage == "post_update":
            row["post_update"]["logits_changed_from_pre_update"] = False
        else:
            row.pop("feature_contract")
        write_new_json(tmp_path / "damaged-cpu.json", value)
        cpu = tmp_path / "damaged-cpu.json"
    else:
        value = json.loads(cost_path.read_text())
        value["out_of_tolerance_control"]["control_cost_ns"] = 0
        write_new_json(tmp_path / "damaged-cost.json", value)
        cost_path = tmp_path / "damaged-cost.json"
    with pytest.raises(ReceiptError):
        write_repair_receipts(tmp_path / ("repair-" + damage),
                              cpu_summary_path=cpu, cost_pilot_path=cost_path)


def test_directory_fsync_failure_removes_final_and_retry_succeeds(monkeypatch, tmp_path):
    from replica_lab.intrinsic_memory_density_receipts_v12 import write_new_json
    destination = tmp_path / "receipt.json"
    def interrupted(directory):
        raise RuntimeError("directory fsync interrupted")
    monkeypatch.setattr("replica_lab.intrinsic_memory_density_receipts_v12._fsync_directory", interrupted)
    with pytest.raises(RuntimeError):
        write_new_json(destination, {"complete": True})
    assert not destination.exists()
    monkeypatch.undo()
    write_new_json(destination, {"complete": True})
    assert json.loads(destination.read_text()) == {"complete": True}


def test_post_link_temp_unlink_failure_keeps_successful_final(monkeypatch, tmp_path):
    from replica_lab.intrinsic_memory_density_receipts_v12 import write_new_json
    import os
    destination = tmp_path / "receipt.json"
    real_unlink = os.unlink

    def fail_temporary(path):
        if str(path).endswith(".tmp"):
            raise OSError("temp unlink interrupted")
        return real_unlink(path)

    monkeypatch.setattr(os, "unlink", fail_temporary)
    write_new_json(destination, {"complete": True})
    assert json.loads(destination.read_text()) == {"complete": True}


def test_navigation_generator_is_consistent_and_deterministic(tmp_path):
    from replica_lab.intrinsic_memory_density_receipts_v12 import write_current_navigation
    root = tmp_path / "reports"
    canonical = root / "coordinator04"
    old = root / "repair01"
    canonical.mkdir(parents=True)
    old.mkdir()
    for directory in (canonical, old):
        for name in ("cpu_engineering_summary.json", "cost_pilot_summary.json",
                     "protocol_summary.json", "verification_results.json",
                     "report_manifest.json"):
            (directory / name).write_text('{"x": 1}\n')
    first = write_current_navigation(
        root, "coordinator04", generator="test", x6_test="test::x6")
    current = json.loads((root / "CURRENT.json").read_text())
    noncanonical = json.loads((root / "NONCANONICAL.json").read_text())
    assert current == first
    assert current["canonical_destination"] == (
        "reports/intrinsic_memory_density_v12/coordinator04")
    assert noncanonical["canonical_destination"] == current["canonical_destination"]
    assert [row["path"] for row in noncanonical["noncanonical_destinations"]] == [
        "reports/intrinsic_memory_density_v12",
        "reports/intrinsic_memory_density_v12/repair01"]
    assert (old / "NONCANONICAL.json").is_file()
    second = write_current_navigation(
        root, "coordinator04", generator="test", x6_test="test::x6")
    assert second == first
    assert (root / "CURRENT.json").read_text() == json.dumps(
        first, indent=2, sort_keys=True) + "\n"


def test_receipt_set_failure_removes_partial_outputs_and_retries(monkeypatch, tmp_path):
    import replica_lab.intrinsic_memory_density_receipts_v12 as receipts
    cpu, cost_path, cost = inputs(tmp_path)
    destination = tmp_path / "repair-set"
    real_write = receipts.write_new_json
    calls = {"count": 0}

    def failing_write(path, value):
        calls["count"] += 1
        if calls["count"] == 2:
            raise RuntimeError("second receipt interrupted")
        return real_write(path, value)

    monkeypatch.setattr(receipts, "write_new_json", failing_write)
    with pytest.raises(RuntimeError):
        receipts.write_repair_receipts(
            destination, cpu_summary_path=cpu, cost_pilot_path=cost_path)
    assert not (destination / PROTOCOL_SUMMARY).exists()
    monkeypatch.setattr(receipts, "write_new_json", real_write)
    receipts.write_repair_receipts(
        destination, cpu_summary_path=cpu, cost_pilot_path=cost_path)
    assert (destination / PROTOCOL_SUMMARY).is_file()

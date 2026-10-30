"""Coherent receipt generation refuses overwrite and binds actual pilot values."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from replica_lab.intrinsic_memory_density_protocols_v12 import (
    canonical_digest, load_protocols,
)
from replica_lab.intrinsic_memory_density_receipts_v12 import (
    PROTOCOL_SUMMARY, REPORT_MANIFEST, VERIFICATION_RESULTS,
    ReceiptError, _portable, write_new_json, write_repair_receipts,
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
    second = tmp_path / "other" / "repair"
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
    import replica_lab.intrinsic_memory_density_receipts_v12 as receipts
    destination = tmp_path / "receipt.json"
    real_fsync = receipts._durable_fsync

    def interrupted(descriptor):
        raise RuntimeError("interrupted")

    monkeypatch.setattr(receipts, "_durable_fsync", interrupted)
    with pytest.raises(RuntimeError):
        receipts.write_new_json(destination, {"complete": True})
    assert not destination.exists()
    monkeypatch.setattr(receipts, "_durable_fsync", real_fsync)
    receipts.write_new_json(destination, {"complete": True})
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
    assert current["canonical_destination"] == _portable(canonical)
    assert noncanonical["canonical_destination"] == current["canonical_destination"]
    assert [row["path"] for row in noncanonical["noncanonical_destinations"]] == [
        _portable(root), _portable(old)]
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


def test_navigation_transaction_rolls_back_on_directory_fsync_failure(monkeypatch, tmp_path):
    import replica_lab.intrinsic_memory_density_receipts_v12 as receipts
    root = tmp_path / "reports"
    canonical = root / "coordinator06"
    old = root / "repair01"
    canonical.mkdir(parents=True)
    old.mkdir()
    for directory in (canonical, old):
        for name in ("cpu_engineering_summary.json", "cost_pilot_summary.json",
                     "protocol_summary.json", "verification_results.json",
                     "report_manifest.json"):
            (directory / name).write_text('{"x": 1}\n')
    (root / "CURRENT.json").write_text('{"old": true}\n')
    (root / "NONCANONICAL.json").write_text('{"old": true}\n')
    real_fsync = receipts._fsync_directory
    calls = {"count": 0}

    def failing(directory):
        calls["count"] += 1
        if calls["count"] == 1:
            raise OSError("directory fsync failed")
        return real_fsync(directory)

    monkeypatch.setattr(receipts, "_fsync_directory", failing)
    with pytest.raises(OSError):
        receipts.write_current_navigation(
            root, "coordinator06", generator="test", x6_test="test::x6")
    assert (root / "CURRENT.json").read_text() == '{"old": true}\n'
    assert (root / "NONCANONICAL.json").read_text() == '{"old": true}\n'
    monkeypatch.setattr(receipts, "_fsync_directory", real_fsync)
    receipts.write_current_navigation(
        root, "coordinator06", generator="test", x6_test="test::x6")
    assert json.loads((root / "CURRENT.json").read_text())["canonical_destination"] == (
        receipts._portable(canonical))


def test_navigation_refuses_custom_filename_schema(tmp_path):
    from replica_lab.intrinsic_memory_density_receipts_v12 import write_current_navigation
    root = tmp_path / "reports"
    canonical = root / "custom"
    canonical.mkdir(parents=True)
    (canonical / "profile.json").write_text("{}\n")
    with pytest.raises(ReceiptError, match="schema filenames"):
        write_current_navigation(root, "custom", generator="test", x6_test="test")


def test_navigation_refuses_corrupt_backup_before_mutation(tmp_path):
    import replica_lab.intrinsic_memory_density_receipts_v12 as receipts
    root = tmp_path / "reports"
    canonical = root / "coordinator08"
    canonical.mkdir(parents=True)
    for name in ("cpu_engineering_summary.json", "cost_pilot_summary.json",
                 "protocol_summary.json", "verification_results.json",
                 "report_manifest.json"):
        (canonical / name).write_text('{"x": 1}\n')
    (root / "CURRENT.json").write_text("not-json\n")
    before = (root / "CURRENT.json").read_bytes()
    with pytest.raises(ReceiptError, match="corrupt receipt backup"):
        receipts.write_current_navigation(
            root, "coordinator08", generator="test", x6_test="test")
    assert (root / "CURRENT.json").read_bytes() == before


def test_navigation_incomplete_rollback_preserves_original_with_notes(monkeypatch, tmp_path):
    import replica_lab.intrinsic_memory_density_receipts_v12 as receipts
    root = tmp_path / "reports"
    canonical = root / "coordinator08"
    canonical.mkdir(parents=True)
    for name in ("cpu_engineering_summary.json", "cost_pilot_summary.json",
                 "protocol_summary.json", "verification_results.json",
                 "report_manifest.json"):
        (canonical / name).write_text('{"x": 1}\n')
    (root / "CURRENT.json").write_text('{"old": true}\n')
    (root / "NONCANONICAL.json").write_text('{"old": true}\n')
    real_fsync = receipts._fsync_directory
    calls = {"count": 0}

    def failing(directory):
        calls["count"] += 1
        if calls["count"] == 1:
            raise OSError("directory fsync failed")
        return real_fsync(directory)

    monkeypatch.setattr(receipts, "_fsync_directory", failing)
    monkeypatch.setattr(receipts, "_restore_backup",
                        lambda path, raw: (_ for _ in ()).throw(OSError("restore failed")))
    with pytest.raises(ReceiptError) as error:
        receipts.write_current_navigation(
            root, "coordinator08", generator="test", x6_test="test")
    assert isinstance(error.value.__cause__, OSError)
    assert error.value.notes
    assert not (root / "CURRENT.json").exists()
    assert not (root / "NONCANONICAL.json").exists()


def test_darwin_durability_uses_fullfsync(monkeypatch, tmp_path):
    import fcntl
    import sys
    import replica_lab.intrinsic_memory_density_receipts_v12 as receipts
    if sys.platform != "darwin":
        assert receipts.durability_label() == "posix_fsync"
        return
    calls = []
    monkeypatch.setattr(fcntl, "fcntl", lambda descriptor, request: calls.append(request))
    descriptor = (tmp_path / "fd").open("w")
    try:
        assert receipts._durable_fsync(descriptor.fileno()) == "darwin_F_FULLFSYNC"
    finally:
        descriptor.close()
    assert calls == [getattr(fcntl, "F_FULLFSYNC", 51)]


def test_current_is_single_validated_authority_pointer(tmp_path):
    import replica_lab.intrinsic_memory_density_receipts_v12 as receipts
    root = tmp_path / "reports"
    canonical = root / "coordinator08"
    canonical.mkdir(parents=True)
    for name in ("cpu_engineering_summary.json", "cost_pilot_summary.json",
                 "protocol_summary.json", "verification_results.json",
                 "report_manifest.json"):
        (canonical / name).write_text('{"x": 1}\n')
    current = receipts.write_current_navigation(
        root, "coordinator08", generator="test", x6_test="test")
    assert receipts.read_current_navigation(root) == current
    (canonical / "protocol_summary.json").write_text('{"x": 2}\n')
    with pytest.raises(ReceiptError, match="hash mismatch"):
        receipts.read_current_navigation(root)
    (root / "CURRENT.json").write_text("not-json\n")
    with pytest.raises(ReceiptError, match="invalid current navigation"):
        receipts.read_current_navigation(root)


def test_navigation_crash_between_renames_resolves_to_valid_current(tmp_path):
    import replica_lab.intrinsic_memory_density_receipts_v12 as receipts
    root = tmp_path / "reports"
    old = root / "old"
    new = root / "new"
    for directory in (old, new):
        directory.mkdir(parents=True)
        for name in ("cpu_engineering_summary.json", "cost_pilot_summary.json",
                     "protocol_summary.json", "verification_results.json",
                     "report_manifest.json"):
            (directory / name).write_text('{"x": 1}\n')
    receipts.write_current_navigation(
        root, "old", generator="test", x6_test="test")
    child = (
        "import os, sys\n"
        "sys.path.insert(0, sys.argv[1])\n"
        "import replica_lab.intrinsic_memory_density_receipts_v12 as r\n"
        "real = os.replace\n"
        "state = {'n': 0}\n"
        "def crash(src, dst):\n"
        "    state['n'] += 1\n"
        "    real(src, dst)\n"
        "    if state['n'] == 1: os._exit(17)\n"
        "r.os.replace = crash\n"
        "r.write_current_navigation(sys.argv[2], 'new', generator='test', x6_test='test')\n")
    result = subprocess.run(
        [sys.executable, "-c", child, str(Path("src").resolve()), str(root)],
        cwd=Path.cwd(), capture_output=True, text=True, timeout=60)
    assert result.returncode == 17, result.stderr
    current = receipts.read_current_navigation(root)
    assert current["canonical_destination"].endswith("/old")
    assert (root / "CURRENT.json").is_file()
    assert (root / "NONCANONICAL.json").is_file()


def test_final_unlink_failure_quarantines_before_raising(monkeypatch, tmp_path):
    import replica_lab.intrinsic_memory_density_receipts_v12 as receipts
    destination = tmp_path / "receipt.json"
    real_unlink = receipts.Path.unlink

    def failing_unlink(self, *args, **kwargs):
        if self == destination:
            raise OSError("final unlink failed")
        return real_unlink(self, *args, **kwargs)

    def failing_fsync(directory):
        raise OSError("directory fsync failed")

    monkeypatch.setattr(receipts.Path, "unlink", failing_unlink)
    monkeypatch.setattr(receipts, "_fsync_directory", failing_fsync)
    with pytest.raises(ReceiptError) as error:
        receipts.write_new_json(destination, {"complete": True})
    assert error.value.notes
    assert not destination.exists()


def test_receipt_set_rollback_failure_is_reported(monkeypatch, tmp_path):
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
    monkeypatch.setattr(receipts, "_remove_final", lambda path: False)
    with pytest.raises(ReceiptError) as error:
        receipts.write_repair_receipts(
            destination, cpu_summary_path=cpu, cost_pilot_path=cost_path)
    assert isinstance(error.value.__cause__, RuntimeError)
    assert error.value.notes


def test_current_missing_fails_closed_not_via_noncanonical(tmp_path):
    from replica_lab.intrinsic_memory_density_receipts_v12 import read_current_navigation
    root = tmp_path / "reports"
    root.mkdir()
    (root / "NONCANONICAL.json").write_text('{"canonical_destination": "elsewhere"}\n')
    with pytest.raises(ReceiptError, match="CURRENT navigation receipt is missing"):
        read_current_navigation(root)


def test_repair_cycle_derives_from_canonical_directory(tmp_path):
    cpu, cost_path, cost = inputs(tmp_path)
    destination = tmp_path / "coordinator11"
    manifest = write_repair_receipts(
        destination, cpu_summary_path=cpu, cost_pilot_path=cost_path)
    protocol = json.loads((destination / PROTOCOL_SUMMARY).read_text())
    verification = json.loads((destination / VERIFICATION_RESULTS).read_text())
    assert manifest["repair_cycle"] == "coordinator11"
    assert protocol["repair_cycle"] == "coordinator11"
    assert verification["repair_cycle"] == "coordinator11"
    assert "repair03" not in json.dumps(verification)
    with pytest.raises(ReceiptError, match="canonical directory identity"):
        write_repair_receipts(
            tmp_path / "coordinator12", cpu_summary_path=cpu,
            cost_pilot_path=cost_path, repair_cycle="repair03")

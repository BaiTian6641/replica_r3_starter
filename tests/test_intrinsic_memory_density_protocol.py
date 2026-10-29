"""Portable review preparation checks: no weights or qualified venv required."""
import copy
import hashlib
import json
from pathlib import Path

import pytest

import validate_spark_memory_density_v5 as density


ROOT = Path(__file__).resolve().parents[1]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


@pytest.fixture
def review(tmp_path):
    protocol = json.loads((ROOT / density.PROTOCOL).read_text())
    packet = json.loads((ROOT / density.PACKET).read_text())
    bindings = {}
    for index, name in enumerate(protocol["bindings"]):
        path = f"inputs/bound-{index}.txt"
        (tmp_path / "inputs").mkdir(exist_ok=True)
        (tmp_path / path).write_text(f"synthetic {name}\n")
        bindings[name] = {"path": path, "sha256": density.digest(tmp_path, path)}
    protocol["bindings"] = bindings
    for arm in protocol["arms"].values():
        arm["bindings"] = copy.deepcopy(bindings)
    reseal(tmp_path, protocol, packet)
    return tmp_path, protocol, packet


def reseal(root, protocol, packet):
    for name, arm in protocol["arms"].items():
        packet["arms"][name]["arm_sha256"] = hashlib.sha256(
            json.dumps(arm, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        for key in ("output_root", "guard_report"):
            packet["arms"][name][key] = arm[key]
    write(root / density.PROTOCOL, protocol)
    packet["protocol_sha256"] = density.digest(root, density.PROTOCOL)


def test_portable_synthetic_preparation(review):
    root, protocol, packet = review
    result = density.validate(protocol, packet, root=root)
    assert result["status"] == "review_packet_consistent_native_not_admitted"
    assert result["launch_allowed"] is False
    assert result["source_weights_read"] is False
    assert result["model_imported"] is False
    assert result["per_arm_budgets"]["generation_requests_max"] == 80
    assert result["comparison_budget_sum"]["forward_calls_max"] == 7785


@pytest.mark.parametrize("operation", ["changed", "missing", "symlink"])
def test_bound_dependency_refused(review, operation):
    root, protocol, packet = review
    path = root / next(iter(protocol["bindings"].values()))["path"]
    if operation == "changed":
        path.write_text("substitution\n")
    elif operation == "missing":
        path.unlink()
    else:
        content = path.read_bytes()
        path.unlink()
        target = root / "substitute.txt"
        target.write_bytes(content)
        path.symlink_to(target)
    with pytest.raises(ValueError):
        density.validate(protocol, packet, root=root)


@pytest.mark.parametrize("field", ["authorization", "launch_allowed", "native_execution"])
def test_review_does_not_authorize_a_native_attempt(review, field):
    root, protocol, packet = review
    packet[field] = True
    with pytest.raises(ValueError, match="cannot claim native admission"):
        density.validate(protocol, packet, root=root)


@pytest.mark.parametrize("key", ["forward_calls_max", "model_loads_max", "generated_tokens_max",
                                 "optimizer_steps_max", "artifact_bytes_max"])
def test_each_arm_has_explicit_unchanged_budget(review, key):
    root, protocol, packet = review
    protocol["arms"]["all"]["budgets"][key] += 1
    reseal(root, protocol, packet)
    with pytest.raises(ValueError, match="work budget differs"):
        density.validate(protocol, packet, root=root)


def test_wrong_arm_exposure_rejected(review):
    root, protocol, packet = review
    protocol["arms"]["all"]["shared_sha256"] = "0" * 64
    reseal(root, protocol, packet)
    with pytest.raises(ValueError, match="shared-contract"):
        density.validate(protocol, packet, root=root)


def test_v3_authorization_cannot_be_inherited(review):
    root, protocol, packet = review
    protocol["arms"]["two"]["authorization"]["inherited_from_v3"] = True
    reseal(root, protocol, packet)
    with pytest.raises(ValueError, match="cannot inherit v3"):
        density.validate(protocol, packet, root=root)


@pytest.mark.parametrize("stage,key", [("competence", "correct_min"),
    ("primary", "contextual_correct_min"), ("chronology", "revoked_false_recall_max")])
def test_acceptance_cannot_be_weakened(review, stage, key):
    root, protocol, packet = review
    protocol["scoring"][stage][key] += 1
    reseal(root, protocol, packet)
    with pytest.raises(ValueError, match="gate differs"):
        density.validate(protocol, packet, root=root)


@pytest.mark.parametrize("path", ["../escape", "/absolute", "src/replica_lab/runtime_pilot.py"])
def test_unsafe_or_protected_input_is_refused_before_read(tmp_path, path):
    with pytest.raises(ValueError):
        density.digest(tmp_path, path)


def test_budget_charges_full_growing_prefix_and_added_chronology():
    values = density.budgets()
    assert values["processed_text_positions_max"] == 80 * sum(range(2048, 2080)) + 35 * 2048
    assert values["forward_calls_max"] == 80 * 32 + 32 + 1 + 2
    assert values["P_updates"] == 0
    assert values["native_retries"] == 0


def test_independent_output_roots_required(review):
    root, protocol, packet = review
    protocol["arms"]["all"]["output_root"] = protocol["arms"]["two"]["output_root"]
    reseal(root, protocol, packet)
    with pytest.raises(ValueError, match="share an output"):
        density.validate(protocol, packet, root=root)

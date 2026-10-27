"""Synthetic CPU tests for the independent nested-text continuation receipt.

The fixture writes the declared artifact shape, including full canonical state
leaves (parameters, moments, dedup, chunks, event fingerprints, route and
personality).  It never loads MiniCPM, runs MPS, builds a learner or executes a
native job.  The comparator is exercised as a pure CPU tool.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
COMPARATOR = ROOT / "scripts" / "compare_nested_text_continuation.py"
CONFIG_PATH = ROOT / "configs" / "nested_text_continuation_comparison_v1.json"
RUN_FORMAT = "replica.nested_text.run.v1"
CONFIG_FORMAT = "replica.nested_text.nl1.v1"
P_ONLY_FORMAT = "replica.nested_learning.text_p_only.v1"
STATE_FORMAT = "replica.nested_continuation.state.v1"
EXPECTED_IDS = tuple(
    [f"teach-{index:04d}" for index in range(24)]
    + [f"correction-{index:04d}" for index in range(4)])
PARENT_STATE_INDEX = 27
RESUME_EVENT_INDEX = 28
CONTINUOUS_EVENTS = 28
PARENT_EVENTS = 27
LEGACY_BEFORE = "11" * 32
LEGACY_AFTER = "22" * 32
SECTIONS = (
    "controller", "controller.bank", "controller.bank.optimizer",
    "controller.clock", "controller.dedup", "controller.observation",
    "controller.revision", "controller.personality", "text", "text.protocol",
    "text.route", "text.feature_producer", "text.read_path",
    "text.personality", "text.event_fingerprints",
)
_PROTOCOL_PAYLOAD = {
    "world_uid": "world_alpha",
    "scope": "world_alpha/text",
    "expert_count": 16,
    "width": 256,
    "hidden_width": 1024,
    "hidden_size": 2048,
    "feature_sites": [13, 27],
    "target_projection_seed": 20261026,
    "target_projection_sha256":
        "db20117e3c41fa9a036d973559ff4eb93ed6dfc86969b0467800b476230aa340",
    "read_gain": 0.1,
    "read_path_sha256":
        "db20117e3c41fa9a036d973559ff4eb93ed6dfc86969b0467800b476230aa340",
    "route_mode": "fixed",
    "route_seed": 0,
    "theta_revision": "f97400052a43d642bbc6e9975e2397e3ae6a6b52",
    "theta_lock_sha256":
        "5eb451cea8f2dced48775502de2c243f72d84e996926de929f15896610bb068b",
    "parameter_stamp_sha256":
        "104e6b860af2cc722e193faec35bcc5bdec0cf20b719979f9e0d6d70704c582b",
    "tokenizer_sha256":
        "ec3bf8fa9d9778f35e2d450d7301f5ddc0e6994e7a14785b2e021605109e801f",
    "p_init": "zero_read_down",
    "steps_per_event": 1,
    "total_step_cap": 4,
    "clock_period": 16,
    "gradient_norm_max": 1.0,
}


def sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_file(path: Path) -> str:
    return sha(Path(path).read_bytes())


def sha_json(value) -> str:
    return sha(json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False).encode())


PROTOCOL_SHA = sha_json(_PROTOCOL_PAYLOAD)
PROTOCOL_BODY = {**_PROTOCOL_PAYLOAD, "protocol_sha256": PROTOCOL_SHA}


def _kind(value) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    raise TypeError(value)


def _put(leaves: dict, path: str, value) -> None:
    leaves[path] = {"kind": _kind(value), "value": value}


def _tensor(seed: str) -> dict:
    return {"kind": "tensor", "dtype": "torch.float32",
            "shape": [1024, 256], "sha256": sha(seed.encode())}


def section_hashes(leaves: dict) -> dict:
    result = {}
    for section in SECTIONS:
        selected = {path: value for path, value in leaves.items()
                    if path == section or path.startswith(section + ".")
                    or path.startswith(section + "[")}
        if selected:
            result[section] = sha_json(selected)
    return result


def canonical(event_index: int, *, tag: str = "shared") -> dict:
    """Build a structurally complete canonical state for one event count."""
    leaves: dict = {}
    ids = EXPECTED_IDS[:event_index]
    _put(leaves, "controller.revision", event_index)
    _put(leaves, "controller.clock.eligible_events", event_index)
    _put(leaves, "controller.clock_period", PROTOCOL_BODY["clock_period"])
    _put(leaves, "controller.total_step_cap", PROTOCOL_BODY["total_step_cap"])
    _put(leaves, "controller.max_steps", 2)
    _put(leaves, "controller.gradient_norm_max",
         PROTOCOL_BODY["gradient_norm_max"])
    _put(leaves, "controller.observation.format_version", 1)
    _put(leaves, "controller.observation.latest_sealed_at", float(event_index))
    for position, event_id in enumerate(ids):
        _put(leaves,
             f"controller.observation.observations[{position}].event_id", event_id)
        _put(leaves,
             f"controller.observation.observations[{position}].scope",
             "world_alpha/text")
        _put(leaves,
             f"controller.observation.observations[{position}].eligible", True)
        _put(leaves,
             f"controller.observation.chunks[{position}].chunk_id",
             f"chunk-{position + 1}")
        _put(leaves,
             f"controller.observation.chunks[{position}].fingerprint",
             sha(f"chunk:{tag}:{event_id}".encode()))
        _put(leaves,
             f"controller.observation.chunks[{position}].pre_revision", position)
        _put(leaves,
             f"controller.observation.chunks[{position}].post_revision",
             position + 1)
        _put(leaves,
             f"controller.observation.chunks[{position}].transition_count", 1)
        _put(leaves,
             f"controller.observation.chunks[{position}].sealed_at",
             float(position + 1))
        _put(leaves,
             f"controller.observation.chunks[{position}].aggregate_key[0]", 0.1)
        _put(leaves,
             f"controller.observation.chunks[{position}].aggregate_value[0]", 0.2)
        _put(leaves, f"controller.dedup.records[{position}].event_id", event_id)
        fingerprint = sha(f"tx:{tag}:{event_id}".encode())
        _put(leaves, f"controller.dedup.records[{position}].fingerprint",
             fingerprint)
        _put(leaves,
             f"controller.dedup.records[{position}].outcome.event_id", event_id)
        _put(leaves,
             f"controller.dedup.records[{position}].outcome.fingerprint",
             fingerprint)
        _put(leaves,
             f"controller.dedup.records[{position}].outcome.status", "committed")
        _put(leaves,
             f"controller.dedup.records[{position}].outcome.expert_id",
             (position + 1) % 16)
        _put(leaves,
             f"controller.dedup.records[{position}].outcome.revision",
             position + 1)
        _put(leaves,
             f"controller.dedup.records[{position}].outcome.eligible_events",
             position + 1)
        _put(leaves,
             f"controller.dedup.records[{position}].outcome.steps", 1)
        _put(leaves,
             f"controller.dedup.records[{position}].outcome.loss",
             100.0 + position + 1)
    _put(leaves, "controller.dedup.format_version", 1)
    _put(leaves, "controller.dedup.identity", "event_id/content_sha256")
    _put(leaves, "controller.bank.expert_count", 16)
    _put(leaves, "controller.bank.width", 256)
    _put(leaves, "controller.bank.hidden_width", 1024)
    _put(leaves, "controller.bank.seed", 0)
    for index in range(48):
        expert, name = divmod(index, 3)
        _put(leaves, f"controller.bank.parameters[{index}][0]",
             f"expert_{expert}.{('gate', 'up', 'down')[name]}")
        leaves[f"controller.bank.parameters[{index}][1]"] = _tensor(
            f"param:{tag}:{index}")
    for index in range(3):
        leaves[f"controller.bank.optimizer.state.{index}.exp_avg"] = _tensor(
            f"moment:{tag}:{index}")
        leaves[f"controller.bank.optimizer.state.{index}.exp_avg_sq"] = _tensor(
            f"moment2:{tag}:{index}")
        _put(leaves, f"controller.bank.optimizer.state.{index}.step", 1)
    _put(leaves, "controller.personality.mode", "declared")
    _put(leaves, "controller.personality.version", 1)
    _put(leaves, "controller.personality.subject_sha", "ab" * 32)
    _put(leaves, "controller.personality.snapshot_digest", "cd" * 32)
    _put(leaves, "text.personality.mode", "declared")
    _put(leaves, "text.personality.version", 1)
    _put(leaves, "text.personality.subject_sha", "ab" * 32)
    _put(leaves, "text.personality.snapshot_digest", "cd" * 32)
    for key, value in PROTOCOL_BODY.items():
        if isinstance(value, list):
            for position, item in enumerate(value):
                _put(leaves, f"text.protocol.{key}[{position}]", item)
        else:
            _put(leaves, f"text.protocol.{key}", value)
    _put(leaves, "text.protocol.protocol_sha256",
         PROTOCOL_BODY["protocol_sha256"])
    _put(leaves, "text.feature_producer.hidden_size",
         PROTOCOL_BODY["hidden_size"])
    _put(leaves, "text.feature_producer.width", PROTOCOL_BODY["width"])
    _put(leaves, "text.feature_producer.target_projection_seed",
         PROTOCOL_BODY["target_projection_seed"])
    _put(leaves, "text.feature_producer.target_projection_sha256",
         PROTOCOL_BODY["target_projection_sha256"])
    for position, site in enumerate(PROTOCOL_BODY["feature_sites"]):
        _put(leaves, f"text.feature_producer.sites[{position}]", site)
    _put(leaves, "text.read_path.gain", PROTOCOL_BODY["read_gain"])
    _put(leaves, "text.read_path.decoder_sha256",
         PROTOCOL_BODY["read_path_sha256"])
    _put(leaves, "text.read_path.version",
         "fixed_projection_sites_masked_mean.v1")
    _put(leaves, "text.route.mode", PROTOCOL_BODY["route_mode"])
    _put(leaves, "text.route.seed", PROTOCOL_BODY["route_seed"])
    _put(leaves, "text.route.expert_count", PROTOCOL_BODY["expert_count"])
    for position in range(PROTOCOL_BODY["expert_count"]):
        _put(leaves, f"text.route.permutation[{position}]", position)
    for event_id in ids:
        _put(leaves, f"text.event_fingerprints.{event_id}",
             sha(f"eventfingerprint:{tag}:{event_id}".encode()))
    counts = {
        "parameter_tensors": 48,
        "parameter_elements": 12582912,
        "optimizer_state_entries": 3,
        "observation_count": event_index,
        "chunk_count": event_index,
        "event_fingerprint_count": event_index,
        "leaf_count": len(leaves),
    }
    return {
        "format": STATE_FORMAT,
        "sha256": sha_json(leaves),
        "sections": section_hashes(leaves),
        "leaves": leaves,
        "counts": counts,
    }


def outcome(event_id: str, index: int, *, tag: str = "shared") -> dict:
    fingerprint = sha(f"tx:{tag}:{event_id}".encode())
    expert = index % 16
    return {
        "status": "committed",
        "event_id": event_id,
        "expert_id": expert,
        "route_mode": "fixed",
        "feature_sha256": sha(f"feature:{tag}:{event_id}".encode()),
        "target_sha256": sha(f"target:{tag}:{event_id}".encode()),
        "retried": False,
        "transaction": {
            "status": "committed", "event_id": event_id,
            "fingerprint": fingerprint, "expert_id": expert,
            "revision": index, "eligible_events": index, "steps": 1,
            "loss": 100.0 + index,
        },
        "chunk": {
            "status": "committed", "chunk_id": f"chunk-{index}",
            "fingerprint": sha(f"chunk:{tag}:{event_id}".encode()),
            "scope": "world_alpha/text", "sealed_at": float(index),
            "observation_ids": [f"observation-{index}"],
            "reader_ids": ["13", "27"],
            "pre_revision": index - 1, "post_revision": index,
            "transition_count": 1,
            "aggregate_key": [0.1, 0.2], "aggregate_value": [0.3, 0.4],
        },
        "gradient_norm_before_clipping": 0.25 + index * 0.01,
    }


def parameter_audit() -> dict:
    names = [f"expert_{expert}.{name}"
             for expert in range(16)
             for name in ("gate", "up", "down")]
    return {
        "theta_stamp_unchanged": True,
        "plastic_parameters": names,
        "plastic_parameter_count": len(names),
        "read_path_trainable_parameters": 0,
        "expert_steps": [2] * 14 + [0, 0],
    }


def guard(runtime: dict) -> dict:
    return {
        "status": "completed_within_declared_sampled_limits",
        "child_exit_code": 0,
        "guard": {
            "policy": {
                "driver_limit_bytes": runtime["max_driver_bytes"],
                "swap_growth_limit_bytes": runtime["max_swap_growth_bytes"],
                "minimum_free_percent": runtime["minimum_free_percent"],
                "sustained_host_samples": runtime["sustained_host_samples"],
            },
            "host_samples": 10,
            "driver_samples": 20,
            "peak_driver_bytes": 1000000,
            "peak_swap_growth_bytes": 0,
            "minimum_observed_free_percent": 50.0,
            "stop_reason": None,
        },
    }


def read_json(path: Path):
    return json.loads(Path(path).read_text())


def write_json(path: Path, value) -> None:
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def read_rows(path: Path) -> list:
    return [json.loads(line) for line in Path(path).read_text().splitlines()
            if line.strip()]


def write_rows(path: Path, rows: list) -> None:
    Path(path).write_text("".join(
        json.dumps(row, sort_keys=True) + "\n" for row in rows))


def corrupt_state(state: dict, path: str, value: dict) -> None:
    leaves = state["leaves"]
    leaves[path] = value
    state["sections"] = section_hashes(leaves)
    state["sha256"] = sha_json(leaves)


def build(tmp_path: Path) -> dict:
    """Write one exact-pass artifact set under ``tmp_path``."""
    protocol_body = copy.deepcopy(PROTOCOL_BODY)
    parent = tmp_path / "parent.pt"
    parent.write_bytes(b"parent-checkpoint-v1\n")
    config = copy.deepcopy(read_json(CONFIG_PATH))
    config["format"] = CONFIG_FORMAT
    config["implementation_paths"] = [
        "scripts/compare_nested_text_continuation.py",
        "src/replica_lab/nested_continuation.py",
    ]
    comparison = config["comparison"]
    comparison.update({
        "parent_checkpoint": str(parent),
        "parent_checkpoint_sha256": sha_file(parent),
        "parent_checkpoint_bytes": parent.stat().st_size,
        "parent_event_count": PARENT_EVENTS,
        "boundary_event": "correction-0003",
        "continuous_events": CONTINUOUS_EVENTS,
        "continuous_arms": 2,
        "resume_arms": 1,
        "protocol_sha256": protocol_body["protocol_sha256"],
        "legacy_before_state": LEGACY_BEFORE,
        "legacy_after_state": LEGACY_AFTER,
        "canonical_state_format": STATE_FORMAT,
        "require_full_dedup_and_coordinator": True,
        "require_gradient_norms": True,
        "require_exact_aa": True,
        "require_exact_ab": True,
        "require_guard_receipts": True,
        "allow_in_process_reference_verdict": False,
        "allow_numeric_envelope": False,
        "stop_on_any_mismatch": True,
        "heldout_opened": False,
    })
    protocol_path = tmp_path / "protocol.json"
    write_json(protocol_path, config)
    config_sha = sha_file(protocol_path)
    runtime = config["runtime"]
    implementation = {
        relative: sha_file(ROOT / relative)
        for relative in config["implementation_paths"]}
    manifest_sha = config["episodes"]["manifest_sha256"]

    states = {index: canonical(index)
              for index in range(1, CONTINUOUS_EVENTS + 1)}
    transactions = [outcome(event_id, index)
                    for index, event_id in enumerate(EXPECTED_IDS, 1)]
    audits = parameter_audit()

    arms = {}
    for label, checkpoint_bytes in (
            ("arm-a", b"checkpoint-arm-a\n"),
            ("arm-b", b"checkpoint-arm-b\n")):
        directory = tmp_path / label
        directory.mkdir()
        (directory / "p_only").mkdir()
        checkpoint = directory / "full_checkpoint.pt"
        checkpoint.write_bytes(checkpoint_bytes)
        checkpoint_sha = sha_file(checkpoint)
        p_only = directory / "p_only" / "state.pt"
        p_only.write_bytes(b"p-only-shared\n")
        p_only_sha = sha_file(p_only)
        write_json(directory / "p_only" / "manifest.json", {
            "format": P_ONLY_FORMAT,
            "bytes": p_only.stat().st_size,
            "sha256": p_only_sha,
            "keys": ["expert_0.gate"],
            "forbidden_absent": True,
            "source_inputs": 0,
        })
        history = []
        for index, event_id in enumerate(EXPECTED_IDS, 1):
            legacy = (LEGACY_BEFORE if index == PARENT_STATE_INDEX
                      else LEGACY_AFTER if index == CONTINUOUS_EVENTS
                      else sha(f"legacy:{event_id}".encode()))
            history.append({
                "event_id": event_id,
                "event_index": index,
                "transaction": transactions[index - 1],
                "legacy_state_digest": legacy,
                "canonical_state": states[index],
            })
        write_rows(directory / "state_history.jsonl", history)
        write_rows(directory / "transactions.jsonl", transactions)
        write_json(directory / "final_state.json", states[CONTINUOUS_EVENTS])
        write_json(directory / "parameter_audit.json", audits)
        write_json(directory / "protocol.json", protocol_body)
        write_json(directory / "cost.json", {
            "events": CONTINUOUS_EVENTS,
            "eligible_events": CONTINUOUS_EVENTS,
            "optimizer_steps": CONTINUOUS_EVENTS,
            "teacher_requests": CONTINUOUS_EVENTS,
            "feature_vectors": 2 * CONTINUOUS_EVENTS,
            "response_tokens": 100,
            "processed_positions": 100,
            "model_loads": 1,
            "tokenizer_loads": 1,
            "wall_seconds": 10.0,
            "update_seconds": 5.0,
            "canonical_state_seconds": 1.0,
            "duty_cycle": 0.5,
            "protocol_sha256": protocol_body["protocol_sha256"],
            "peak_driver_bytes": 1000000,
            "checkpoint_bytes": checkpoint.stat().st_size,
            "p_only_bytes": p_only.stat().st_size,
        })
        write_json(directory / "run.json", {
            "format": RUN_FORMAT, "mode": "teach", "status": "completed",
            "world_uid": protocol_body["world_uid"],
            "protocol_sha256": protocol_body["protocol_sha256"],
            "episodes_manifest_sha256": manifest_sha,
            "config_sha256": config_sha,
            "implementation_sha256": implementation,
            "theta_stamp_unchanged": True,
            "parameter_audit": "parameter_audit.json",
            "updates": CONTINUOUS_EVENTS,
            "learning_events": CONTINUOUS_EVENTS,
            "state_history": "state_history.jsonl",
            "final_state": "final_state.json",
            "canonical_state_sha256": states[CONTINUOUS_EVENTS]["sha256"],
            "legacy_state_digest": LEGACY_AFTER,
            "checkpoint_sha256": checkpoint_sha,
            "checkpoint_bytes": checkpoint.stat().st_size,
            "p_only_sha256": p_only_sha,
            "checkpoint": str(checkpoint),
            "p_only": str(p_only),
            "descriptive_only": True,
            "promotion": False,
            "non_claims": ["synthetic fixture only"],
        })
        arms[label] = directory

    resume = tmp_path / "resume"
    resume.mkdir()
    before, after = states[PARENT_STATE_INDEX], states[CONTINUOUS_EVENTS]
    write_json(resume / "before_state.json", before)
    write_json(resume / "after_state.json", after)
    write_rows(resume / "state_history.jsonl", [{
        "event_id": EXPECTED_IDS[-1],
        "event_index": CONTINUOUS_EVENTS,
        "transaction": transactions[-1],
        "legacy_state_digest": LEGACY_AFTER,
        "canonical_state": after,
    }])
    write_json(resume / "parameter_audit.json", audits)
    implementation_map = dict(implementation)
    write_json(resume / "resume.json", {
        "format": RUN_FORMAT, "mode": "resume",
        "status": "completed_pending_independent_comparison",
        "next_event": "correction-0003",
        "before_state": LEGACY_BEFORE, "after_state": LEGACY_AFTER,
        "before_canonical_sha256": before["sha256"],
        "after_canonical_sha256": after["sha256"],
        "before_state_file": "before_state.json",
        "after_state_file": "after_state.json",
        "state_history": "state_history.jsonl",
        "checkpoint": str(parent),
        "checkpoint_sha256": sha_file(parent),
        "checkpoint_bytes": parent.stat().st_size,
        "config_sha256": config_sha,
        "implementation_sha256": implementation_map,
        "protocol_sha256": protocol_body["protocol_sha256"],
        "episodes_manifest_sha256": manifest_sha,
        "resumed_outcome": transactions[-1],
        "transaction_retried": False,
        "restart_equal": None,
        "descriptive_only": True,
    })
    write_json(resume / "run.json", {
        "format": RUN_FORMAT, "mode": "resume",
        "status": "completed_pending_independent_comparison",
        "restart_equal": None,
        "before_state": LEGACY_BEFORE, "after_state": LEGACY_AFTER,
        "canonical_state_sha256": after["sha256"],
        "theta_stamp_unchanged": True,
        "parameter_audit": "parameter_audit.json",
        "checkpoint_sha256": sha_file(parent),
        "state_history": "state_history.jsonl",
        "final_state": "after_state.json",
        "config_sha256": config_sha,
        "implementation_sha256": implementation_map,
    })

    guards = {}
    for label in ("guard-a", "guard-b", "guard-resume"):
        path = tmp_path / f"{label}.json"
        write_json(path, guard(runtime))
        guards[label] = path
    return {
        "protocol": protocol_path,
        "continuous_a": arms["arm-a"],
        "continuous_b": arms["arm-b"],
        "resume": resume,
        "guard_a": guards["guard-a"],
        "guard_b": guards["guard-b"],
        "guard_resume": guards["guard-resume"],
        "parent": parent,
        "config": config,
        "protocol_body": protocol_body,
        "config_sha": config_sha,
        "manifest_sha": manifest_sha,
        "before": before,
        "after": after,
        "transactions": transactions,
        "outdir": tmp_path,
    }


def compare(module, fixture: dict) -> dict:
    return module.compare_artifacts(
        protocol=fixture["protocol"],
        continuous_a=fixture["continuous_a"],
        continuous_b=fixture["continuous_b"],
        resume=fixture["resume"],
        guard_a=fixture["guard_a"],
        guard_b=fixture["guard_b"],
        guard_resume=fixture["guard_resume"])


@pytest.fixture(scope="module")
def comparator():
    spec = importlib.util.spec_from_file_location(
        "compare_nested_text_continuation", COMPARATOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_exact_pass_cli_and_determinism(tmp_path, comparator, monkeypatch):
    fixture = build(tmp_path)
    first = tmp_path / "receipt-a.json"
    monkeypatch.setattr(sys, "argv", [
        "compare",
        "--protocol", str(fixture["protocol"]),
        "--continuous-a", str(fixture["continuous_a"]),
        "--continuous-b", str(fixture["continuous_b"]),
        "--resume", str(fixture["resume"]),
        "--guard-a", str(fixture["guard_a"]),
        "--guard-b", str(fixture["guard_b"]),
        "--guard-resume", str(fixture["guard_resume"]),
        "--output", str(first),
    ])
    assert comparator.main() == 0
    receipt = read_json(first)
    assert receipt["status"] == "exact_match"
    assert receipt["exact_status"] == "passed"
    assert receipt["variation_status"] == "none"
    assert receipt["resource_status"] == "within_declared_limits"
    assert receipt["earliest_divergence"] is None
    assert receipt["failures"] == []
    assert receipt["events"]["aa_states_compared"] >= CONTINUOUS_EVENTS
    assert receipt["events"]["boundary_event"] == "correction-0003"
    assert receipt["events"]["boundary_event_index"] == RESUME_EVENT_INDEX
    assert receipt["events"]["parent_state_index"] == PARENT_STATE_INDEX
    assert receipt["comparisons"]["boundary"]["continuous_a_before_sha256"] == \
        receipt["comparisons"]["boundary"]["resume_before_sha256"]
    assert receipt["comparisons"]["p_only_equal"] is True
    assert receipt["comparisons"]["checkpoint_bytes_equal"] is True
    assert receipt["claims"]["in_process_reference_verdict"] is False
    second = tmp_path / "receipt-b.json"
    monkeypatch.setattr(sys, "argv", sys.argv[:-1] + [str(second)])
    assert comparator.main() == 0
    assert first.read_bytes() == second.read_bytes()


def test_canonical_mismatch_localizes_first_event(tmp_path, comparator):
    fixture = build(tmp_path)
    rows = read_rows(fixture["continuous_b"] / "state_history.jsonl")
    corrupt_state(rows[6]["canonical_state"],
                  "controller.bank.parameters[0][1]",
                  _tensor("corrupted-parameter"))
    write_rows(fixture["continuous_b"] / "state_history.jsonl", rows)
    receipt = compare(comparator, fixture)
    assert receipt["status"] == "failed"
    assert receipt["exact_status"] == "failed"
    assert receipt["variation_status"] == "variation_detected"
    divergence = receipt["earliest_divergence"]
    assert divergence["event_index"] == 7
    assert divergence["event_id"] == "teach-0006"
    assert divergence["stage"] == "aa_canonical"
    assert divergence["path"] == "controller.bank.parameters[0][1]"
    assert any("aa_canonical" in failure for failure in receipt["failures"])


def test_resume_boundary_mismatch_is_ab_failure(tmp_path, comparator):
    fixture = build(tmp_path)
    before = read_json(fixture["resume"] / "before_state.json")
    corrupt_state(before, "controller.bank.parameters[1][1]",
                  _tensor("resume-corruption"))
    write_json(fixture["resume"] / "before_state.json", before)
    manifest = read_json(fixture["resume"] / "resume.json")
    manifest["before_canonical_sha256"] = before["sha256"]
    write_json(fixture["resume"] / "resume.json", manifest)
    receipt = compare(comparator, fixture)
    assert receipt["status"] == "failed"
    assert receipt["exact_status"] == "failed"
    divergence = receipt["earliest_divergence"]
    assert divergence["event_index"] == PARENT_STATE_INDEX
    assert divergence["stage"] == "boundary"
    assert any("boundary" in failure for failure in receipt["failures"])


def test_guard_stop_fails_closed(tmp_path, comparator):
    fixture = build(tmp_path)
    report = read_json(fixture["guard_b"])
    report["guard"]["stop_reason"] = "sustained_host_swap_growth"
    report["guard"]["peak_swap_growth_bytes"] = (
        fixture["config"]["runtime"]["max_swap_growth_bytes"] + 1)
    write_json(fixture["guard_b"], report)
    receipt = compare(comparator, fixture)
    assert receipt["status"] == "failed"
    assert receipt["exact_status"] == "failed"
    assert receipt["resource_status"] == "exceeded"
    assert any("guard-b" in failure for failure in receipt["failures"])


def test_missing_event_27_boundary_fails(tmp_path, comparator):
    fixture = build(tmp_path)
    history = read_rows(fixture["continuous_a"] / "state_history.jsonl")
    transactions = read_rows(fixture["continuous_a"] / "transactions.jsonl")
    history.pop(PARENT_STATE_INDEX - 1)
    transactions.pop(PARENT_STATE_INDEX - 1)
    write_rows(fixture["continuous_a"] / "state_history.jsonl", history)
    write_rows(fixture["continuous_a"] / "transactions.jsonl", transactions)
    receipt = compare(comparator, fixture)
    assert receipt["status"] == "failed"
    assert receipt["exact_status"] == "failed"
    assert receipt["earliest_divergence"]["event_index"] == PARENT_STATE_INDEX
    assert any("event_sequence" in failure for failure in receipt["failures"])
    assert any("boundary" in failure for failure in receipt["failures"])


def test_implementation_drift_fails(tmp_path, comparator):
    fixture = build(tmp_path)
    run = read_json(fixture["continuous_b"] / "run.json")
    first_path = sorted(run["implementation_sha256"])[0]
    run["implementation_sha256"][first_path] = "0" * 64
    write_json(fixture["continuous_b"] / "run.json", run)
    receipt = compare(comparator, fixture)
    assert receipt["status"] == "failed"
    assert receipt["exact_status"] == "failed"
    assert any("implementation" in failure for failure in receipt["failures"])


def test_in_process_reference_verdict_is_rejected(tmp_path, comparator):
    fixture = build(tmp_path)
    for name in ("run.json", "resume.json"):
        path = fixture["resume"] / name
        report = read_json(path)
        report["restart_equal"] = True
        report["status"] = "completed"
        report["reference_after_state"] = LEGACY_AFTER
        write_json(path, report)
    receipt = compare(comparator, fixture)
    assert receipt["status"] == "failed"
    assert receipt["exact_status"] == "failed"
    assert receipt["variation_status"] == "none"
    assert any("in_process_reference_verdict" in failure
               for failure in receipt["failures"])


def test_p_only_cross_arm_mismatch_fails(tmp_path, comparator):
    fixture = build(tmp_path)
    p_only = fixture["continuous_b"] / "p_only" / "state.pt"
    p_only.write_bytes(b"different-p-only-export\n")
    digest = sha_file(p_only)
    manifest = read_json(fixture["continuous_b"] / "p_only" / "manifest.json")
    manifest["bytes"] = p_only.stat().st_size
    manifest["sha256"] = digest
    write_json(fixture["continuous_b"] / "p_only" / "manifest.json", manifest)
    run = read_json(fixture["continuous_b"] / "run.json")
    run["p_only_sha256"] = digest
    write_json(fixture["continuous_b"] / "run.json", run)
    cost = read_json(fixture["continuous_b"] / "cost.json")
    cost["p_only_bytes"] = p_only.stat().st_size
    write_json(fixture["continuous_b"] / "cost.json", cost)
    receipt = compare(comparator, fixture)
    assert receipt["status"] == "failed"
    assert receipt["exact_status"] == "failed"
    assert any("p_only" in failure for failure in receipt["failures"])


def test_checkpoint_bytes_are_not_a_cross_arm_gate(tmp_path, comparator):
    fixture = build(tmp_path)
    checkpoint = fixture["continuous_b"] / "full_checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint-b-with-different-bytes-and-length\n")
    digest = sha_file(checkpoint)
    run = read_json(fixture["continuous_b"] / "run.json")
    run["checkpoint_sha256"] = digest
    run["checkpoint_bytes"] = checkpoint.stat().st_size
    write_json(fixture["continuous_b"] / "run.json", run)
    cost = read_json(fixture["continuous_b"] / "cost.json")
    cost["checkpoint_bytes"] = checkpoint.stat().st_size
    write_json(fixture["continuous_b"] / "cost.json", cost)
    receipt = compare(comparator, fixture)
    assert receipt["status"] == "exact_match"
    assert receipt["comparisons"]["checkpoint_bytes_equal"] is False


def test_gradient_norm_is_required_and_localized(tmp_path, comparator):
    fixture = build(tmp_path)
    rows = read_rows(fixture["continuous_b"] / "state_history.jsonl")
    transactions = read_rows(fixture["continuous_b"] / "transactions.jsonl")
    del rows[9]["transaction"]["gradient_norm_before_clipping"]
    del transactions[9]["gradient_norm_before_clipping"]
    write_rows(fixture["continuous_b"] / "state_history.jsonl", rows)
    write_rows(fixture["continuous_b"] / "transactions.jsonl", transactions)
    receipt = compare(comparator, fixture)
    assert receipt["status"] == "failed"
    assert receipt["earliest_divergence"]["event_index"] == 10
    assert any("gradient_norm_before_clipping" in failure
               for failure in receipt["failures"])


def test_frozen_theta_audit_is_required(tmp_path, comparator):
    fixture = build(tmp_path)
    audit = read_json(fixture["continuous_b"] / "parameter_audit.json")
    audit["theta_stamp_unchanged"] = False
    write_json(fixture["continuous_b"] / "parameter_audit.json", audit)
    receipt = compare(comparator, fixture)
    assert receipt["status"] == "failed"
    assert receipt["exact_status"] == "failed"
    assert any("theta_audit" in failure for failure in receipt["failures"])


def test_same_arm_twice_is_not_two_independent_arms(tmp_path, comparator):
    fixture = build(tmp_path)
    receipt = comparator.compare_artifacts(
        protocol=fixture["protocol"],
        continuous_a=fixture["continuous_a"],
        continuous_b=fixture["continuous_a"],
        resume=fixture["resume"],
        guard_a=fixture["guard_a"],
        guard_b=fixture["guard_b"],
        guard_resume=fixture["guard_resume"])
    assert receipt["status"] == "failed"
    assert receipt["exact_status"] == "failed"
    assert any("distinct directories" in failure
               for failure in receipt["failures"])


def test_comparator_has_no_in_process_learner_reference():
    source = COMPARATOR.read_text()
    assert "run_nested_text" not in source
    assert "NestedTextLearner" not in source
    assert "canonical_state(learner" not in source
    assert "from replica_lab.nested_continuation import FORMAT, compare_states" \
        in source

"""CPU contracts for causal query refresh and restart without re-encoding."""
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from architecture_helpers import tiny_adapter
from replica_lab.memory_causal_refresh import (
    CAUSAL, CHECKPOINT_FORMAT, ORACLE, PROTOCOL_FORMAT, SHUFFLED,
    load_policy_checkpoint, run_causal_refresh)
from replica_lab.memory_matched import derive_matched_labels, generate_matched_episodes
from replica_lab.memory_update_dynamics import DEFAULT_THRESHOLDS
from test_memory_diagnosis import SPEC, Tokenizer


def _fixture():
    episodes = []
    for episode in generate_matched_episodes(
            seed=7, families_per_split={"train": 1, "dev": 1, "heldout": 1},
            complexities=("single", "pair", "correction", "interference"), variants=2):
        if episode["complexity"] not in {item["complexity"] for item in episodes}:
            episodes.append(episode)
    labels = derive_matched_labels(episodes)
    thresholds = dict(DEFAULT_THRESHOLDS)
    thresholds.update({"causal_top1_floor": 0.85,
                       "oracle_exact_top1_floor": 0.85})
    protocol = {"format": PROTOCOL_FORMAT, "layers": (0, 2),
                "thresholds": thresholds, "protocol_sha256": "a" * 64}
    return episodes, labels, protocol, {"weights_sha256": "b" * 64}


ROOT = Path(__file__).resolve().parents[1]


def test_native_protocol_is_bounded_frozen_and_explicitly_hybrid():
    protocol = json.loads((ROOT / "configs/memory_causal_refresh_v1.json").read_text())
    assert protocol["budget"]["optimizer_updates"] == 0
    assert protocol["budget"]["native_diagnostic_jobs"] == 2
    assert protocol["budget"]["policy_checkpoints_saved"] == 1
    assert protocol["resource_policy"]["driver_limit_bytes"] == 50 * 1024**3
    assert protocol["resource_policy"]["cpu_fallback"] is False
    assert protocol["numeric_policy"]["torch_use_deterministic_algorithms"] is True
    assert protocol["numeric_policy"]["mps_fast_math"] == "disabled"
    assert protocol["numeric_policy"]["dtype"] == "float32"
    assert protocol["policy"]["forbidden_inputs"] == [
        "gold targets", "future queries", "undeclared source rereading",
        "evaluator expected answers"]
    assert protocol["reused_generation_controls"]["rerun"] is False
    assert "not learned-memory quality" in protocol["reused_generation_controls"]["classification"]
    assert all((ROOT / name).is_file() for name in protocol["implementation_paths"])


def test_native_launch_refuses_to_run_without_capacity_guard(monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "causal_native", ROOT / "scripts/run_memory_causal_refresh.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.delenv("REPLICA_CAPACITY_GUARD", raising=False)
    with pytest.raises(RuntimeError, match="capacity guard"):
        module.execute(SimpleNamespace())


def test_causal_refresh_restart_and_gold_independence(tmp_path):
    episodes, labels, protocol, receipt = _fixture()
    first_model = tiny_adapter(fast_memory=SPEC).eval()
    first = run_causal_refresh(
        first_model, Tokenizer(), protocol, output_dir=tmp_path / "first",
        episodes=episodes, labels_by_episode=labels,
        writer_extension_receipt=receipt, device="cpu")
    assert first["contexts_encoded"] > 0
    assert first["old_observations_reencoded"] is False
    assert first["optimizer_updates"] == 0
    assert first["classification"]["causal_refresh"].startswith("hybrid")
    checkpoint = tmp_path / "first/policy_checkpoint.pt"
    contexts = load_policy_checkpoint(
        checkpoint, protocol_sha256=protocol["protocol_sha256"],
        writer_sha256=receipt["weights_sha256"])
    assert contexts
    with pytest.raises(ValueError, match="writer differs"):
        load_policy_checkpoint(
            checkpoint, protocol_sha256=protocol["protocol_sha256"],
            writer_sha256="c" * 64)

    second_model = tiny_adapter(fast_memory=SPEC).eval()
    resumed = run_causal_refresh(
        second_model, Tokenizer(), protocol, output_dir=tmp_path / "resumed",
        episodes=episodes, labels_by_episode=labels,
        writer_extension_receipt=receipt, device="cpu",
        resume_checkpoint=checkpoint)
    assert resumed["resumed"] is True
    assert resumed["contexts_encoded"] == 0
    assert resumed["old_observations_reencoded"] is False
    assert resumed["raw"] == first["raw"]
    assert resumed["metrics"] == first["metrics"]

    decisions = [json.loads(line) for line in
                 (tmp_path / "first/decisions.jsonl").read_text().splitlines()]
    assert decisions
    assert all(row["used_future_queries"] is False for row in decisions)
    assert all(row["reread_source"] is False for row in decisions)
    assert any(row["variant"] == ORACLE and row["used_gold_target"] is True
               for row in decisions)
    assert all(row["used_gold_target"] is False for row in decisions
               if row["variant"] in (CAUSAL, SHUFFLED))

    # Changing evaluator target identities cannot alter the causal policy's
    # decision trace; it is free to alter only oracle/evaluator-side behavior.
    changed = copy.deepcopy(labels)
    for episode_labels in changed.values():
        for state_labels in episode_labels.values():
            alternate = state_labels["live_entries"][0]["record_id"]
            for target in state_labels["targets"].values():
                if target["kind"] == "fact":
                    target["target_entry_id"] = alternate
    run_causal_refresh(
        tiny_adapter(fast_memory=SPEC).eval(), Tokenizer(), protocol,
        output_dir=tmp_path / "changed-labels", episodes=episodes,
        labels_by_episode=changed, writer_extension_receipt=receipt, device="cpu")
    changed_decisions = [json.loads(line) for line in
                         (tmp_path / "changed-labels/decisions.jsonl").read_text().splitlines()]
    for variant in (CAUSAL, SHUFFLED):
        assert [row for row in changed_decisions if row["variant"] == variant] == \
            [row for row in decisions if row["variant"] == variant]

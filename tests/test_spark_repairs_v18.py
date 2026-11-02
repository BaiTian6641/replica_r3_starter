"""Independent CPU regression evidence for the V17 qualification and V18.2."""
import copy
import json
from dataclasses import replace

import pytest
import torch

from replica_lab.contextual_memory import ContextualMemoryError
from replica_lab.intrinsic_memory_cpu_fixture_v18 import fixture_rows, make_engine
from replica_lab.intrinsic_memory_interference_v17 import run_pair
from replica_lab.intrinsic_memory_measurement_v18 import (
    _cpu_float64,
    capture_state,
    module_statistics,
    restore_state,
    tensor_statistics,
    verify_state,
)
from replica_lab.intrinsic_memory_paired_update_v18 import run_comparison, validate_pairs
from replica_lab.intrinsic_memory_probe_v18 import probe
from replica_lab.intrinsic_memory_acceptance_v15 import chronological_correction, expect_invalidation
from replica_lab.intrinsic_memory_correction_v18 import publish_correction, fresh_revocation_check
from replica_lab.contextual_v12_fixture import canonical_span, open_v12_store, scope
from test_intrinsic_acceptance_v15 import compound_row, COMPOUND_FAMILIES
from test_intrinsic_archive_migration_v16 import make_release
from replica_lab.intrinsic_memory_archive_migration_v16 import inventory_archive


def test_nonzero_producer_stale_slot_reproduction_and_measured_restoration(tmp_path):
    engine = make_engine(tmp_path)
    try:
        state = capture_state(engine.path)
        result = run_pair(engine, *fixture_rows()[:2], 0, tmp_path)
        verify_state(engine.path, state)
        assert all(check["equal"]["optimizer_steps"] for check in result["state_restoration"])
        assert all(arm["actual_adamw_displacement"]["producer"]["l2"] > 0 for arm in result["arms"])
        assert all(arm["per_module_gradients"]["reader"]["l2"] > 0 for arm in result["arms"])
        deltas = [abs(live["margin"]["total_log_likelihood"] - frozen["margin"]["total_log_likelihood"])
                  for arm in result["arms"] for live, frozen in zip(arm["after"], arm["frozen_slot_intervention"])]
        assert min(deltas) > 1e-5
        assert all("every_sequence_position" in row["temporal_scope"]
                   for row in result["interventions"] if row["control"].startswith("wrong_read"))
        for component in ("parameters", "optimizer", "moments"):
            damaged = copy.deepcopy(state)
            if component == "parameters":
                next(iter(damaged[component].values())).add_(1)
            elif component == "optimizer":
                next(iter(damaged[component]["state"].values()))["step"] += 1
            else:
                entry = next(iter(damaged["optimizer"]["state"].values()))
                next(value for key, value in entry.items() if key != "step" and torch.is_tensor(value)).add_(1)
            with pytest.raises(ValueError, match="restoration mismatch"):
                verify_state(engine.path, damaged)
        torch.rand(3)
        with pytest.raises(ValueError, match="restoration mismatch"):
            verify_state(engine.path, state)
        restore_state(engine.path, state)
    finally:
        engine.path.close()


def test_v18_full_trajectory_exposure_step_counts_and_live_slots(tmp_path):
    engine = make_engine(tmp_path)
    try:
        before = dict(engine.ledger.snapshot()["reserved"])
        result = run_comparison(engine, list(reversed(fixture_rows())))
        counts = engine.ledger.snapshot()["reserved"]
        assert counts["forwards"] - before["forwards"] == 248
        assert counts["backwards"] - before["backwards"] == 16
        assert counts["updates"] - before["updates"] == 12
        assert counts["source_feature_forwards"] - before["source_feature_forwards"] == 8
        assert counts["prompt_tokenizer_calls"] - before["prompt_tokenizer_calls"] == 16
        assert result["optimizer_steps"] == {"sequential": 8, "paired_joint": 4}
        for arm, steps in result["optimizer_steps"].items():
            receipt = json.loads((tmp_path / (arm + ".json")).read_text())
            assert len(receipt["trajectory"]) == steps + 1
            assert sorted(w for step in receipt["updates"] for w in step["world_ids"]) == sorted(r["world_id"] for r in fixture_rows())
            for previous, current in zip(receipt["trajectory"], receipt["trajectory"][1:]):
                assert len(current["worlds"]) == 8
                assert all(a["slot_digest"] != b["slot_digest"] for a, b in zip(previous["worlds"], current["worlds"]))
            for step in receipt["updates"]:
                assert step["loss_coefficients"] == ([1] if arm == "sequential" else [.5, .5])
                assert step["actual_adamw_displacement"]["producer"]["l2"] > 0
        for damage in ("answer", "family", "question", "world_id"):
            rows = fixture_rows(); rows[0][damage] = "invalid"
            with pytest.raises(ValueError):
                validate_pairs(rows)
    finally:
        engine.path.close()


def test_joint_loss_normalizes_each_world_with_unequal_answer_lengths(tmp_path):
    engine = make_engine(tmp_path)
    original = engine.backend.answer_ids
    engine.backend.answer_ids = lambda answer: [9, 11, 12] if answer == "blue" else original(answer)
    try:
        run_comparison(engine, fixture_rows())
        joint = json.loads((tmp_path / "paired_joint.json").read_text())
        for step in joint["updates"]:
            assert sorted(step["supervised_tokens"]) == [2, 4]
            assert step["optimized_loss"] == pytest.approx(sum(step["per_world_mean_CE"])/2)
            assert step["loss_coefficients"] == [.5, .5]
        for row in joint["trajectory"][0]["worlds"]:
            assert row["expected"]["supervised_tokens"] + row["alternative"]["supervised_tokens"] == 6
    finally:
        engine.path.close()


def test_read_override_replaces_every_sequence_position(tmp_path):
    from replica_lab.intrinsic_memory_interference_v17 import _candidate
    from replica_lab.intrinsic_memory_counterfactual_v13 import _source
    engine = make_engine(tmp_path)
    observed = []
    hook = engine.path.reader.ports["3"].output_projection.register_forward_hook(
        lambda module, args, output: observed.append(args[0].detach().clone()))
    try:
        row = fixture_rows()[0]
        batch, _, sc = _source(engine.path, engine, row)
        replacement = torch.arange(256, dtype=torch.float32)
        _candidate(engine, row, batch, sc, row["answer"], prompt=engine.tokens(row),
                   override_site=3, override_read=replacement)
        assert observed[-1].shape[1] > 1
        assert torch.equal(observed[-1], replacement.view(1, 1, -1).expand_as(observed[-1]))
    finally:
        hook.remove(); engine.path.close()


@pytest.mark.parametrize("where", ["working", "snapshot"])
def test_execution_snapshot_rejects_source_tampering(tmp_path, where):
    from replica_lab.intrinsic_memory_execution_v5 import binding, local_closure
    from replica_lab.intrinsic_memory_paired_protocol_v18 import validate_implementation_snapshot
    source = "fixture.py"
    (tmp_path / source).write_text("VALUE = 1\n")
    snapshot_root = tmp_path / "snapshot"
    snapshot_root.mkdir()
    (snapshot_root / source).write_text("VALUE = 1\n")
    files = local_closure(tmp_path, [source])
    protocol = {}
    for key, value in {
        "implementation": {"entries": [source], "files": files},
        "source_snapshot": {"root": "snapshot", "files": files, "source_bytes": 10},
    }.items():
        name = key + ".json"
        (tmp_path / name).write_text(json.dumps(value))
        protocol[key] = binding(tmp_path, name)
    validate_implementation_snapshot(tmp_path, protocol)
    target = (tmp_path if where == "working" else snapshot_root) / source
    target.write_text("VALUE = 2\n")
    with pytest.raises(ValueError, match="closure differs|snapshot differs"):
        validate_implementation_snapshot(tmp_path, protocol)


def test_closure_includes_relative_and_absolute_package_submodule_imports(tmp_path):
    from replica_lab.intrinsic_memory_execution_v5 import local_closure
    files = {"src/replica_lab/__init__.py": "",
        "src/replica_lab/nested/__init__.py": "",
        "src/replica_lab/nested/entry.py": "from .. import artifact_publication as publication\nfrom replica_lab import helper\n",
        "src/replica_lab/artifact_publication.py": "VALUE = 1\n",
        "src/replica_lab/helper.py": "VALUE = 2\n"}
    for name, value in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value)
    assert set(local_closure(tmp_path, ["src/replica_lab/nested/entry.py"])) == set(files)


@pytest.mark.parametrize("damage", ["budget", "key_mode", "steps", "endpoint", "development"])
def test_protocol_rejects_comparison_and_accounting_drift(damage):
    from replica_lab.intrinsic_memory_paired_protocol_v18 import (
        PROTOCOL, diagnostic_plan, parent_plan, totals, budget_derivation, validate_protocol, training_contract)
    plan = diagnostic_plan()
    protocol = {"format": PROTOCOL, "protocol_revision": 2, "intrinsic_reader_key_mode": "content",
        "development_labels_used": False, "parameter_updates": 12,
        "source_repo": "XHToken/Spark-X2.6-4B", "source_revision": "66e3ae7bee54e0a523d692cb8ef841f35409807b",
        "training": training_contract(), "data": {"fit": {}},
        "intrinsic_placement": "block_input_query_to_post_block_residual",
        "diagnostic": {"pair_indices": [0, 1, 4, 5], "updates": 12, "development": False,
                       "generated_tokens": 0, "intrinsic_reader_key_mode": "content", "output_root": "out"},
        "comparison": {"arms": ["sequential", "paired_joint"], "steps": [8, 4],
            "primary": "mean_final_pair_minimum_total_margin", "guard": "worlds_below_own_baseline",
            "loss": "per_world_mean_answer_plus_EOS_CE", "joint_coefficients": [.5, .5],
            "primary_strictly_greater": True, "guard_no_greater": True},
        "arms": {"two": {"sites": [3, 7], "budget": plan, "total_reservations": totals(plan), "wall_seconds": 1800,
            "output_root": "out", "resource_limits": {"driver_bytes": 50*1024**3, "swap_growth_bytes": 512*1024**2,
            "minimum_host_free_percent": 15, "sustained_host_samples": 3, "telemetry_timeout_seconds": 30}}},
        "total_with_preflight": {key: value + totals(parent_plan())[key] for key, value in totals(plan).items()},
        "budget_derivation": budget_derivation()}
    validate_protocol(protocol)
    if damage == "budget":
        protocol["arms"]["two"]["budget"]["diagnose"]["forwards"] -= 1
    elif damage == "key_mode":
        protocol["intrinsic_reader_key_mode"] = "ordered"
    elif damage == "steps":
        protocol["comparison"]["steps"] = [8, 8]
    elif damage == "endpoint":
        protocol["comparison"]["primary"] = "best_prefix"
    else:
        protocol["data"]["development"] = {}
    with pytest.raises(ValueError):
        validate_protocol(protocol)


@pytest.mark.parametrize("order", [[0, 1, 2, 3, 4], [4, 0, 3, 2, 1], [1, 3, 4]])
def test_probe_original_groups_singletons_odd_counts_and_controls(order):
    features = torch.tensor([[1., 0.], [0., 1.], [2., 0.], [0., 2.], [3., 0.]])[order]
    answers = [["red", "blue", "red", "blue", "red"][i] for i in order]
    identities = [{"world_id": str(i), "pair_index": [8, 8, 17, 17, 25][i]} for i in order]
    for solver in ("primal", "dual"):
        baseline = probe(features, answers, identities=identities, solver=solver)
        shuffled = list(reversed(range(len(answers))))
        control = probe(features, answers, identities=identities, solver=solver, shuffled=shuffled)
        assert [r["world_id"] for r in baseline] == [str(i) for i in order]
        assert all(r["held_out_group"] not in r["training_groups"] for r in baseline)
        assert [r["actual"] for r in control] == [answers[i] for i in shuffled]
        with pytest.raises(ValueError, match="permutation"):
            probe(features, answers, identities=identities, shuffled=[0]*len(answers))


@pytest.mark.parametrize("unknown", [False, True])
@pytest.mark.parametrize("family", COMPOUND_FAMILIES)
def test_repeated_correction_complete_current_view_and_fresh_revocation(tmp_path, unknown, family):
    engine = make_engine(tmp_path)
    store = open_v12_store(tmp_path / "store")
    sc = scope("correction")
    try:
        row = compound_row(family, unknown=unknown)
        original = canonical_span("Independent original observation.", record_id="original")[0]
        engine.path.admit(store, original, scope=sc)
        old = engine.path.read(store, "observation", scope=sc)
        for color in ("green", "gold", "blue"):
            row = chronological_correction(row, color)
            span, publication = publish_correction(engine, store, old, row, sc)
            expect_invalidation("stale_snapshot", lambda: engine.path.validate(old, sc))
            fresh = engine.path.read(store, "B", scope=sc)
            assert "".join(item.text for item in fresh.spans) == publication["current_text"]
            assert f"B is {color}" in publication["current_text"]
            assert len(row["text"]) > len(publication["current_text"])
            assert (tmp_path / publication["history_path"]).is_file()
            old = fresh
        corrupted = replace(old, slots=old.slots.clone())
        corrupted.slots[0, 0] += 1
        with pytest.raises(ContextualMemoryError, match="payload digest"):
            expect_invalidation("stale_snapshot", lambda: engine.path.validate(corrupted, sc))
        store.revoke_record(span.record_id, reason="fixture", scope=sc)
        expect_invalidation("stale_snapshot", lambda: engine.path.validate(old, sc))
        assert fresh_revocation_check(store, "B", sc, [span.record_id])["selected_record_ids"] == []
    finally:
        store.close(); engine.path.close()


@pytest.mark.parametrize("container", [[], None, "bad", 7])
@pytest.mark.parametrize("location", ["manifest", "records", "record", "scope", "pointer"])
def test_malformed_archive_owner_quarantined_healthy_inventory_continues(tmp_path, container, location):
    bad = tmp_path / "stores/bad/artifacts/contextual_v12/releases/scope/00000001"
    good = tmp_path / "stores/good/artifacts/contextual_v12/releases/scope/00000001"
    make_release(good, 1, b"good")
    manifest = make_release(bad, 1, b"bad")
    target = bad / "manifest.json"
    if location == "manifest":
        manifest = container
    elif location == "record":
        manifest["records"]["record"] = container
    elif location == "pointer":
        target = bad.parent / "current.json"; manifest = container
    else:
        manifest[location] = container
    target.write_text(json.dumps(manifest))
    result = inventory_archive(tmp_path)
    assert result["current_pointer_valid_generations"] == 1
    assert len(result["quarantined_generations"]) == 1
    if location == "pointer":
        assert result["rows"][0]["final"]["generation"] is None


def test_cold_post_io_digest_failure_never_counts_as_authority(tmp_path, monkeypatch):
    from replica_lab.contextual_memory_v5 import ContextualReadFailure, AuthorityReadFailure
    engine = make_engine(tmp_path)
    store = open_v12_store(tmp_path / "store")
    sc = scope("cold")
    try:
        engine.path.admit(store, canonical_span("Synthetic cold badge.", record_id="cold")[0],
                          scope=sc, placement="cold")
        pending = engine.path.read(store, "badge", scope=sc)
        monkeypatch.setattr(store, "_validate_after_io", lambda *a: ("failed", "payload_digest_mismatch", ()))
        with pytest.raises(ContextualReadFailure) as caught:
            expect_invalidation("stale_snapshot", lambda: engine.path.resume(store, pending, scope=sc, wait_seconds=1))
        assert not isinstance(caught.value, AuthorityReadFailure)
        assert "operational" in str(caught.value)
    finally:
        store.close(); engine.path.close()


def test_corrupt_tensor_is_operational_even_with_changed_producer(tmp_path):
    from replica_lab.contextual_memory import AuthorityRejection
    engine = make_engine(tmp_path)
    store = open_v12_store(tmp_path / "store")
    sc = scope("mixed_failure")
    try:
        engine.path.admit(store, canonical_span("Synthetic badge.", record_id="corrupt")[0], scope=sc)
        old = engine.path.read(store, "badge", scope=sc)
        bad = replace(old, slots=old.slots.clone(), producer_version="another-release")
        bad.slots[0, 0] += 1
        with pytest.raises(ContextualMemoryError, match="payload digest") as caught:
            engine.path.validate(bad, sc)
        assert not isinstance(caught.value, AuthorityRejection)
    finally:
        store.close(); engine.path.close()


def test_correction_oversize_current_view_fails_before_publication(tmp_path):
    engine = make_engine(tmp_path)
    store = open_v12_store(tmp_path / "store")
    sc = scope("oversize")
    try:
        engine.path.admit(store, canonical_span("Original.", record_id="old")[0], scope=sc)
        old = engine.path.read(store, "Original", scope=sc)
        row = compound_row(COMPOUND_FAMILIES[0]); row["facts"]["anchor"] = "A"*200
        row["query_entity"] = "A"*200
        corrected = chronological_correction(row, "green")
        with pytest.raises(ContextualMemoryError, match="complete correction current view"):
            publish_correction(engine, store, old, corrected, sc)
        assert engine.path.validate(old, sc)
        assert not (tmp_path / "correction_history").exists()
    finally:
        store.close(); engine.path.close()


def test_preflight_rejection_precedes_attempt_and_model_loading(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from contextlib import nullcontext
    import run_spark_paired_update_v18 as runner
    from replica_lab.intrinsic_memory_paired_protocol_v18 import diagnostic_plan
    packet_path = tmp_path / "packet.json"
    packet_path.write_text(json.dumps({"protocol": {}, "output_root": "out", "attempt_marker": "attempt.jsonl"}))
    protocol = {"arms": {"two": {"budget": diagnostic_plan()}}}
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "bound_json", lambda *a: protocol)
    monkeypatch.setattr(runner, "validate_authorization", lambda *a, **k: None)
    monkeypatch.setattr(runner, "exclusive_worker", nullcontext)
    def refused(*a, **k):
        raise ValueError("pair/checkpoint/closure/budget preflight refused")
    monkeypatch.setattr(runner, "validate_bindings", refused)
    def forbidden(*a, **k):
        pytest.fail("native load or attempt happened before validation")
    monkeypatch.setattr(runner, "load_native", forbidden)
    monkeypatch.setattr(runner, "record_attempt", forbidden)
    with pytest.raises(ValueError, match="preflight refused"):
        runner.launch(SimpleNamespace(packet=packet_path, authorization=tmp_path / "authorization.json"))
    assert not (tmp_path / "attempt.jsonl").exists()


def test_cpu_and_mps_float64_reduction_and_probe_parity():
    from types import SimpleNamespace
    from replica_lab.optimizers import NativeFusedAdamW

    devices = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
    results = {}
    for device in devices:
        torch.manual_seed(17)
        layer_cpu = torch.nn.Linear(64, 64)
        x_cpu = torch.randn(4, 64)
        layer = copy.deepcopy(layer_cpu).to(device)
        x = x_cpu.to(device)
        path = SimpleNamespace(modules=lambda l=layer: {"p": l})
        optimizer = NativeFusedAdamW(
            [
                {
                    "params": list(layer.parameters()),
                    "param_names": [name for name, _ in layer.named_parameters()],
                    "scope": "new",
                    "group_name": "new_adamw_fused",
                    "algorithm": "adamw_fused",
                    "lr": 1e-4,
                    "weight_decay": 0.01,
                    "betas": (0.9, 0.999),
                    "eps": 1e-8,
                }
            ]
        )
        optimizer.zero_grad(set_to_none=True)
        (layer(x).pow(2).sum() * 10.0).backward()
        grad_stats = module_statistics(path)
        before_stats = tensor_statistics(list(layer.parameters()))
        before_tensors = {
            f"p.{name}": parameter.detach().clone()
            for name, parameter in layer.named_parameters()
        }
        optimizer.step()
        after_stats = module_statistics(path, before=before_tensors)
        feats = x.detach()
        probe_out = probe(
            feats,
            ["red", "blue", "red", "blue"],
            identities=[
                {"world_id": str(i), "pair_index": i // 2} for i in range(4)
            ],
        )
        converted = _cpu_float64(layer.weight)
        assert converted.device.type == "cpu"
        assert converted.dtype == torch.float64
        assert grad_stats["p"]["finite"] is True
        assert before_stats["finite"] is True
        assert after_stats["p"]["finite"] is True
        assert grad_stats["p"]["l2"] > 0.0
        assert after_stats["p"]["l2"] > 0.0
        results[device] = (grad_stats["p"], before_stats, after_stats["p"], probe_out)
        bad = torch.ones(8, device=device, dtype=torch.float32)
        bad[0] = float("nan")
        assert tensor_statistics([bad])["finite"] is False

    if "mps" in results:
        cpu_grad, cpu_before, cpu_after, cpu_probe = results["cpu"]
        mps_grad, mps_before, mps_after, mps_probe = results["mps"]
        assert mps_grad["l2"] == pytest.approx(cpu_grad["l2"], rel=1e-5, abs=1e-5)
        assert mps_grad["max_abs"] == pytest.approx(cpu_grad["max_abs"], rel=1e-5, abs=1e-5)
        assert mps_before["l2"] == pytest.approx(cpu_before["l2"], rel=1e-6, abs=1e-6)
        assert mps_after["l2"] == pytest.approx(cpu_after["l2"], rel=1e-5, abs=1e-6)
        assert mps_probe == cpu_probe


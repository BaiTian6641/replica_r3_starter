"""Adapters to the existing core, exact task runtime, workspace and identity."""
from __future__ import annotations

import time

import pytest
import torch

from e3_runtime_helpers import (ScriptedProposer, make_case, make_runtime,
                                run_test_action, write_action)

from replica_lab.architecture.contracts import CoreSpec, CoreState
from replica_lab.replica.identity import (IdentityContext, IdentityStore,
                                          PrincipleHypothesis, SourceEpisode)
from replica_lab.replica.reader import IdentityReader
from replica_lab.runtime.workspace_store import WorkspaceStore
from replica_lab.successors.adapters import (CoreWorkspaceAdapter, IdentityAdapter,
                                             TaskRuntimeFormationAdapter,
                                             formation_authority)
from replica_lab.successors.formation import (FormationStore, Observation,
                                              InteractionProposal)
from replica_lab.successors.tiers import TieredBlobStore, TierLimits

SPEC = CoreSpec(width=16, slots=4, steps=2, heads=4, task_features=24,
                ffn_hidden=32, control_hidden=20)
DIGEST = "a" * 64


def tensors():
    generator = torch.Generator().manual_seed(5)
    return (torch.randn(1, 3, SPEC.width, generator=generator),
            torch.ones(1, 3, dtype=torch.bool),
            torch.randn(1, 4, SPEC.width, generator=generator),
            torch.ones(1, 4, dtype=torch.bool),
            torch.zeros(1, SPEC.task_features))


def formation_store(root) -> FormationStore:
    return FormationStore(root, subject_id="subject-1", source_epochs=(0,),
                          producer_versions={"runtime": "v1", "core": "test"},
                          representation_version="rep-v1", codec_id="codec-v1")


def test_core_gradient_identity_null_parity_and_workspace_binding(tmp_path):
    sense, sense_valid, memory, memory_valid, task = tensors()
    adapter = CoreWorkspaceAdapter(SPEC, producer="core-test")
    training = adapter.build(sense, sense_valid, memory, memory_valid, task,
                             task_digest=DIGEST, revision=1, training=True)
    training.workspace.square().sum().backward()
    assert adapter.core.slot_queries.grad is not None
    assert bool(adapter.core.slot_queries.grad.abs().sum() > 0)
    inference = adapter.build(sense, sense_valid, memory, memory_valid, task,
                              task_digest=DIGEST, revision=2)
    assert not inference.workspace.requires_grad
    store = WorkspaceStore(tmp_path / "workspace")
    adapter.save(store, inference)
    loaded = adapter.load(store, expected_task_digest=DIGEST, expected_revision=2)
    assert loaded is not None and loaded.stamp() == inference.stamp()
    adapter.reset_task(store)
    assert store.load(expected_task_digest=DIGEST, expected_revision=2) is None

    reader = IdentityReader(width=SPEC.width, heads=SPEC.heads)
    identity_adapter = CoreWorkspaceAdapter(SPEC, identity_reader=reader)
    slots = torch.randn(1, 8, SPEC.width)
    valid = torch.ones(1, 8, dtype=torch.bool)
    identity_adapter.previous = None
    without = identity_adapter.build(sense, sense_valid, memory, memory_valid, task,
                                     task_digest=DIGEST, revision=3)
    identity_adapter.previous = None
    disabled = identity_adapter.build(
        sense, sense_valid, memory, memory_valid, task, task_digest=DIGEST,
        revision=3, identity_slots=slots, identity_valid=valid,
        identity_enabled=False)
    identity_adapter.previous = None
    untrained = identity_adapter.build(
        sense, sense_valid, memory, memory_valid, task, task_digest=DIGEST,
        revision=3, identity_slots=slots, identity_valid=valid)
    assert torch.equal(without.workspace, disabled.workspace)
    assert torch.equal(without.workspace, untrained.workspace)
    assert disabled.identity_digest is None
    assert untrained.identity_digest is not None


def test_task_runtime_formation_enabled_and_disabled(tmp_path):
    conditions = {}
    for enabled in (True, False):
        root = tmp_path / f"task-{'enabled' if enabled else 'disabled'}"
        root.mkdir()
        tools = make_case(root)
        runtime = make_runtime(tools, root,
                               workspace_store=WorkspaceStore(root / "workspace"))
        with formation_store(root / "formation") as formation:
            adapter = TaskRuntimeFormationAdapter(
                runtime, formation, branch_id="e3", purpose="test",
                writer_enabled=enabled)
            result = adapter.run(
                ScriptedProposer([write_action(), run_test_action()]), 2)
            later = formation.pin(
                branch_id="e3", purpose="test", mode="assistant_for_source",
                as_of=time.time_ns() + 3600 * 10 ** 9, task_id=tools.case_id,
                task_revision=0,
                include_verification_states=("verified", "unverified"))
            conditions[enabled] = {
                "decisions": result["decisions"],
                "status": result["formation"]["status"],
                "records": len(formation.visible(later)),
                "notes": len(formation.read_provisional(branch_id="e3",
                                                        purpose="test")),
                "success": result["success"],
            }
    assert conditions[True]["status"] == "committed"
    assert conditions[True]["records"] > 0
    assert conditions[True]["notes"] == conditions[True]["decisions"]
    assert conditions[False]["status"] is None
    assert conditions[False]["records"] == 0
    assert conditions[False]["notes"] == 0


def test_interventions_reset_task_and_identity_without_authority_change(tmp_path):
    tools = make_case(tmp_path)
    runtime = make_runtime(tools, tmp_path,
                           workspace_store=WorkspaceStore(tmp_path / "workspace"))
    with formation_store(tmp_path / "formation") as formation:
        adapter = TaskRuntimeFormationAdapter(runtime, formation, branch_id="e3",
                                              purpose="test")
        core = CoreState(torch.zeros(1, 2, 4), torch.ones(1, 2, dtype=torch.bool),
                         torch.zeros(1, 24), "producer", runtime.state.digest,
                         runtime.state.revision, 2)
        task_clearing = {"task_reset", "task_then_identity", "identity_then_task",
                         "both_reset"}
        identity_clearing = {"identity_null", "task_then_identity",
                             "identity_then_task", "both_reset"}
        for name in TaskRuntimeFormationAdapter.intervention_conditions():
            runtime.identity = None
            runtime.workspace_store.save(core)
            outcome = adapter.apply_intervention(name)
            assert outcome["task_workspace_cleared"] is (name in task_clearing)
            loaded = runtime.workspace_store.load(
                expected_task_digest=runtime.state.digest)
            if name in task_clearing:
                assert loaded is None
            else:
                assert loaded is not None
            assert outcome["identity_null"] is (name in identity_clearing)
            if name in identity_clearing:
                assert isinstance(runtime.identity, IdentityContext)
                assert runtime.identity.is_null
            else:
                assert runtime.identity is None


def test_identity_and_task_reset_orders_on_fresh_worlds(tmp_path):
    final = []
    for order in ("identity_then_task", "task_then_identity"):
        root = tmp_path / order
        root.mkdir()
        tools = make_case(root)
        runtime = make_runtime(tools, root,
                               workspace_store=WorkspaceStore(root / "workspace"))
        with formation_store(root / "formation") as formation:
            adapter = TaskRuntimeFormationAdapter(runtime, formation, branch_id="e3",
                                                  purpose="test")
            core = CoreState(torch.ones(1, 2, 4), torch.ones(1, 2, dtype=torch.bool),
                             torch.zeros(1, 24), "producer", runtime.state.digest,
                             runtime.state.revision, 2)
            runtime.workspace_store.save(core)
            adapter.apply_intervention(order)
            final.append({
                "identity_null": runtime.identity is not None
                and runtime.identity.is_null,
                "workspace": runtime.workspace_store.load(
                    expected_task_digest=runtime.state.digest),
                "goal": runtime.state.goal,
            })
    assert final[0]["identity_null"] and final[1]["identity_null"]
    assert final[0]["workspace"] is None and final[1]["workspace"] is None
    assert final[0]["goal"] == final[1]["goal"]


def test_identity_adapter_fictional_source_and_revocation(tmp_path):
    store = IdentityStore("fictional-subject")
    store.add_episode(SourceEpisode(
        episode_id="ep-1", subject_id="fictional-subject", source_id="interview",
        source_hash="a" * 64, observed_at=1, event_at=1,
        text="I prefer reliability first when cost is bounded.",
        kind="preference", authenticity="source_report"))
    store.add_principle(PrincipleHypothesis(
        principle_id="pr-1", subject_id="fictional-subject",
        statement="prefer reliability", condition="when cost is bounded",
        supporting_episode_ids=("ep-1",), status="approved"))
    store.build_snapshot("snap-1", version=1, created_at=1,
                         episode_ids=("ep-1",), principle_ids=("pr-1",))
    adapter = IdentityAdapter(store)
    context = adapter.resolve("snap-1", mode="assistant_for_source",
                              branch_id="branch-a", purpose="test", cutoff=10)
    assert context.evidence_ids == ("ep-1",)
    assert adapter.render().strip()
    adapter.reset()
    assert adapter.is_null and adapter.render() == ""
    store.revoke_episode("ep-1")
    with pytest.raises(PermissionError):
        adapter.resolve("snap-1", mode="assistant_for_source",
                        branch_id="branch-a", purpose="test", cutoff=10)


def test_equal_information_conditions_share_evidence(tmp_path):
    conditions = CoreWorkspaceAdapter.equal_information_conditions()
    names = [item["name"] for item in conditions]
    assert len(names) == len(set(names))
    assert set(names) >= {"full", "identity_null", "workspace_zero"}
    for item in conditions:
        assert set(item) >= {"name", "enabled_inputs"}
    sense, sense_valid, memory, memory_valid, task = tensors()
    adapter = CoreWorkspaceAdapter(SPEC)
    full = adapter.build_condition(
        "full", sense, sense_valid, memory, memory_valid, task,
        task_digest=DIGEST, revision=1)
    zero = adapter.build_condition(
        "workspace_zero", sense, sense_valid, memory, memory_valid, task,
        task_digest=DIGEST, revision=1)
    assert not bool(zero.workspace.any())
    assert torch.equal(full.task_features, zero.task_features)
    memory_only = adapter.build_condition(
        "memory_only", sense, sense_valid, memory, memory_valid, task,
        task_digest=DIGEST, revision=1)
    assert memory_only.task_digest == full.task_digest


def test_formation_authority_controls_tier_selection(tmp_path):
    limits = TierLimits(hot_bytes=2048, warm_bytes=4096, cold_bytes=8192,
                        identity_reservation_bytes=128,
                        replay_reservation_bytes=128)
    tiers = TieredBlobStore(tmp_path / "tiers", limits=limits,
                            hot_capacity_bytes=512)
    with formation_store(tmp_path / "formation") as formation:
        pin = formation.pin(branch_id="branch-a", purpose="test",
                            mode="assistant_for_source",
                            as_of=time.time_ns() + 3600 * 10 ** 9,
                            task_id="task-1", task_revision=0,
                            include_verification_states=("unverified",))
        observation = Observation(
            observation_id="o1", text="the answer is 7", source_id="host",
            source_epoch=0, observed_at=time.time_ns(), receipt_id="receipt-1",
            verification_state="unverified", authenticity="host_observed")
        proposal = InteractionProposal(
            interaction_id="i1", subject_id=pin.subject_id, branch_id=pin.branch_id,
            purpose=pin.purpose, mode=pin.mode, task_id=pin.task_id,
            task_revision=pin.task_revision, decision_revision=1, source_epoch=0,
            producer_versions=pin.producer_versions,
            representation_version=pin.representation_version,
            codec_id=pin.codec_id, permitted_uses=pin.permitted_uses,
            observations=(observation,))
        commit = formation.commit_interaction(proposal, pin=pin)
        record_id = commit.receipt.record_ids[0]
        tiers.publish(record_id, observation.text.encode("utf-8"), generation=commit.receipt.generation,
                      kind="evidence", producer="formation",
                      representation=pin.representation_version,
                      scope_digest=pin.digest)
        stale_authority = formation_authority(formation, pin)
        assert tiers.read(record_id, authority=stale_authority).status == "ineligible"
        later = formation.pin(branch_id="branch-a", purpose="test",
                              mode="assistant_for_source",
                              as_of=time.time_ns() + 3600 * 10 ** 9,
                              task_id="task-1", task_revision=0,
                              include_verification_states=("unverified",))
        authority = formation_authority(formation, later)
        assert tiers.read(record_id, authority=authority).status == "ready"
        formation.revoke(record_id, reason="source revoked")
        assert tiers.read(record_id, authority=authority).status == "ineligible"

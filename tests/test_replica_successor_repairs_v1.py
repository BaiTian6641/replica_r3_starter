"""Owned regressions for the v12 independent review findings (CPU only)."""
from __future__ import annotations

import json
import pathlib
import sqlite3
import time
import types

import pytest
import torch

from test_successor_formation_v12 import (commit, make_store, observation, pin,
                                          proposal)
from test_successor_chunks_v12 import (ChunkStore, make_session, tensors)
from replica_lab.successors.formation import (FormationStore, InteractionProposal,
                                              MemoryPin, Observation,
                                              OutcomeReceipt, ScopeViolation,
                                              IneligibleRecord)
from replica_lab.successors.chunks import (ChunkError, ChunkSession, MAX_TOKENS,
                                           functional_inner_update,
                                           tensor_digest)
from replica_lab.successors.study import answer_query
from replica_lab.successors.tiers import (DirtyState, ReadContract, TierError,
                                          TieredBlobStore, TierLimits)
from replica_lab.successors.adapters import (TaskRuntimeFormationAdapter,
                                             formation_authority)
from e3_runtime_helpers import make_case, make_runtime
from replica_lab.runtime.workspace_store import WorkspaceStore
from replica_lab.architecture.contracts import CoreState


def _obs(name, text, **changes):
    return observation(name, text, **changes)


def _pin(store, branch="branch-a", as_of=None, revision=0):
    return store.pin(branch_id=branch, purpose="formation-test",
                     mode="parallel_replica",
                     as_of=time.time_ns() + 3600 * 10 ** 9 if as_of is None else as_of,
                     task_id="task-1", task_revision=revision,
                     include_verification_states=("verified", "unverified"))


def _contract(producer="producer-v1", representation="rep-v1", scope="a" * 64,
              cutoff=10 ** 9):
    return ReadContract(producer, representation, scope, cutoff=cutoff)


def _tier(tmp_path, **changes):
    values = dict(limits=TierLimits(hot_bytes=4096, warm_bytes=8192,
                                    cold_bytes=16384, identity_reservation_bytes=64,
                                    replay_reservation_bytes=64),
                  hot_capacity_bytes=512)
    values.update(changes)
    return TieredBlobStore(tmp_path, **values)


def test_cross_branch_supersession_and_lineage_are_refused(tmp_path):
    with make_store(tmp_path) as store:
        other = commit(store, _pin(store, "branch-b"),
                       [_obs("other", "branch b fact")])
        foreign_id = other.receipt.record_ids[0]
        with pytest.raises(ScopeViolation):
            commit(store, _pin(store, "branch-a"),
                   [_obs("child", "cross branch supersede", supersedes=foreign_id)])
        with pytest.raises(ScopeViolation):
            commit(store, _pin(store, "branch-a"),
                   [_obs("lineage", "cross branch lineage",
                         lineage=(foreign_id,))])


def test_revocation_scope_and_host_wide_authority(tmp_path):
    with make_store(tmp_path) as store:
        a = commit(store, _pin(store, "branch-a"), [_obs("a", "branch a fact")])
        b = commit(store, _pin(store, "branch-b"), [_obs("b", "branch b fact")])
        b_id = b.receipt.record_ids[0]
        a_id = a.receipt.record_ids[0]
        store.revoke(b_id, reason="branch-local")
        assert a_id in {item.record_id for item in store.visible(_pin(store))}
        with pytest.raises(PermissionError):
            store.revoke(a_id, reason="host wide", scope="host_wide")
        store.revoke(a_id, reason="host wide", scope="host_wide", authorized_by="host")
        assert a_id not in {item.record_id for item in store.visible(_pin(store))}


def test_correction_is_immutable_but_privacy_revocation_is_immediate(tmp_path):
    with make_store(tmp_path) as store:
        base = commit(store, _pin(store), [_obs("base", "dated fact")],
                      interaction="interaction-base")
        record_id = base.receipt.record_ids[0]
        pinned = _pin(store)
        store.correct(record_id, text="corrected later", reason="late evidence",
                      pin=_pin(store))
        assert record_id in {item.record_id for item in store.visible(pinned)}
        other = commit(store, _pin(store), [_obs("other", "revocable fact")],
                       interaction="interaction-other")
        other_id = other.receipt.record_ids[0]
        revocation_pin = _pin(store)
        store.revoke(other_id, reason="privacy revocation")
        assert other_id not in {
            item.record_id for item in store.visible(revocation_pin)}
        assert other_id not in {item.record_id for item in store.visible(_pin(store))}


def test_outcomes_respect_cutoff_and_order(tmp_path):
    with make_store(tmp_path) as store:
        current = _pin(store)
        first_at = time.time_ns()
        time.sleep(0.005)
        second_at = time.time_ns()
        cutoff = first_at + (second_at - first_at) // 2
        outcomes = (
            OutcomeReceipt("u1", "a1", "completed", "r1", first_at, "one",
                           source_epoch=0),
            OutcomeReceipt("u2", "a2", "completed", "r2", second_at, "two",
                           source_epoch=0),
        )
        commit(store, current, outcomes=outcomes)
        later = _pin(store)
        assert [item["outcome_id"] for item in store.outcomes(later)] == ["u1", "u2"]
        assert [item["ordinal"] for item in store.outcomes(later)] == [0, 1]
        historical = _pin(store, as_of=cutoff)
        assert [item["outcome_id"] for item in store.outcomes(historical)] == ["u1"]


def test_record_order_follows_proposal_ordinal(tmp_path):
    with make_store(tmp_path) as store:
        current = _pin(store)
        report = commit(store, current, [
            _obs("zeta", "vault code alpha"),
            _obs("alpha", "vault code beta"),
        ])
        visible = store.visible(_pin(store))
        assert [item.record_id for item in visible] == list(report.receipt.record_ids)
        answer = answer_query("vault code", visible)
        assert answer["record_id"] == report.receipt.record_ids[1]


def test_derived_tables_are_authenticated(tmp_path):
    with make_store(tmp_path) as store:
        base = commit(store, _pin(store), [_obs("a", "authentic")])
        record_id = base.receipt.record_ids[0]
    connection = sqlite3.connect(tmp_path / "authority.sqlite3")
    connection.execute(
        "INSERT INTO tombstones(record_id, kind, reason, event_key, at_ns, "
        "observed_at, generation) VALUES (?, 'revoked', 'injected', 'absent', 0, 0, 1)",
        (record_id,))
    connection.commit()
    connection.close()
    with make_store(tmp_path) as store:
        report = store.recover()
        assert report.status == "degraded"
        assert report.derived_rebuilt is True
        assert record_id in {item.record_id for item in store.visible(_pin(store))}
        assert not store._connection.execute(
            "SELECT 1 FROM tombstones WHERE event_key='absent'").fetchone()


def test_recovery_restores_deleted_notes_and_lineage(tmp_path):
    from replica_lab.successors.formation import ProvisionalNote
    with make_store(tmp_path) as store:
        parent = commit(store, _pin(store), [_obs("parent", "parent fact")])
        parent_id = parent.receipt.record_ids[0]
        child = commit(store, _pin(store),
                       [_obs("child", "child fact", lineage=(parent_id,))],
                       notes=(ProvisionalNote("note-1", "provisional",
                                              interaction_id="interaction-2"),),
                       interaction="interaction-2")
        child_id = child.receipt.record_ids[0]
    connection = sqlite3.connect(tmp_path / "authority.sqlite3")
    connection.execute("DELETE FROM notes")
    connection.execute("DELETE FROM record_lineage")
    connection.commit()
    connection.close()
    with make_store(tmp_path) as store:
        report = store.recover()
        assert report.status == "degraded" and report.derived_rebuilt is True
        assert store._connection.execute(
            "SELECT 1 FROM record_lineage WHERE record_id=? AND ancestor_id=?",
            (child_id, parent_id)).fetchone() is not None
        assert len(store.read_provisional(branch_id="branch-a",
                                          purpose="formation-test")) == 1


def test_event_scope_tamper_fails_closed(tmp_path):
    with make_store(tmp_path) as store:
        commit(store, _pin(store), [_obs("a", "scoped")])
    connection = sqlite3.connect(tmp_path / "authority.sqlite3")
    connection.execute("UPDATE events SET branch_id='other-branch'")
    connection.commit()
    connection.close()
    with pytest.raises(Exception):
        make_store(tmp_path)


def test_cold_resume_and_fallback_revalidate_external_authority(tmp_path):
    store = _tier(tmp_path, io_delay_seconds=0.01)
    store.publish("cold-record", b"stale", generation=1, kind="evidence",
                  producer="producer-v1", representation="rep-v1",
                  scope_digest="a" * 64, tier="cold")
    store._hot.clear()
    authority = {"eligible": True, "generation": 1, "eligibility_revision": 0}
    pending = store.read("cold-record", contract=_contract(),
                         authority=lambda record_id: dict(authority),
                         expected_generation=1, budget_seconds=0)
    assert pending.status == "pending"
    authority["eligible"] = False
    assert store.resume(pending.ticket_id).status != "ready"

    fallback = _tier(tmp_path / "fallback")
    fallback.publish("canonical", b"canonical", generation=0, kind="canonical",
                     producer="host", representation="text", scope_digest="d" * 64)
    minimal = fallback.read(
        "absent", contract=_contract(), fallback_id="canonical",
        fallback_contract=_contract("host", "text", "d" * 64),
        authority=lambda record_id: {"eligible": True})
    assert minimal.status == "contract_required"

    calls = {"canonical": 0}

    def changing(record_id):
        identity = {"producer": "host", "representation": "text",
                    "scope_digest": "d" * 64, "cutoff": 10 ** 9}
        if record_id == "canonical":
            calls["canonical"] += 1
            return {"eligible": calls["canonical"] == 1, "generation": 0,
                    "eligibility_revision": 0, **identity}
        return {"eligible": True, "generation": None,
                "eligibility_revision": 0, **identity}

    result = fallback.read("absent", contract=_contract(),
                           fallback_id="canonical",
                           fallback_contract=_contract("host", "text", "d" * 64),
                           authority=changing, fallback_tokens=9)
    assert result.status == "rejected"
    assert result.fallback_tokens == 0


def test_p_clock_latest_selection_and_pointer(tmp_path):
    store = _tier(tmp_path)
    store.publish_p("expert", weights=b"w1", moments=b"m1", clock=1,
                    encoder="encoder-v1", lineage=["r1"])
    store.publish_p("expert", weights=b"w2", moments=b"m2", clock=2,
                    encoder="encoder-v2", lineage=["r1", "r2"], tier="cold")
    state = store.read_p("expert")
    assert state["clock"] == 2 and state["tier"] == "cold"
    pointer = json.loads((tmp_path / "p_current" / "expert.json").read_text())
    assert pointer["clock"] == 2 and pointer["record_id"] == "p_expert_c2"
    with pytest.raises(Exception):
        store.publish_p("expert", weights=b"old", moments=b"old", clock=1,
                        encoder="encoder-v1", lineage=["r1"])


def test_p_revocation_requires_host_authority(tmp_path):
    store = _tier(tmp_path)
    store.publish_p("expert", weights=b"w", moments=b"m", clock=1,
                    encoder="encoder-v1", lineage=["r1"])
    with pytest.raises(PermissionError):
        store.revoke_p("expert", reason="bad", authorized_by="model")
    store.revoke_p("expert", reason="host withdrawal", authorized_by="host")
    with pytest.raises(PermissionError):
        store.read_p("expert")
    with pytest.raises(PermissionError):
        store.publish_p("expert", weights=b"w2", moments=b"m2", clock=2,
                        encoder="encoder-v1", lineage=["r1"])


def test_chunk_recompute_prechunk_invalidate_and_pending_guard(tmp_path):
    with ChunkStore(tmp_path, subject_id="subject-1") as store:
        session = make_session(store)
        original = session.state.detach().clone()
        session.begin_chunk([20])
        session.propose(*tensors())
        with pytest.raises(ChunkError):
            session.begin_chunk([21])
        session.invalidate()
        assert torch.equal(session.state, original)
        session.begin_chunk([20, 21])
        session.propose(*tensors())
        receipt = session.commit().receipt
        assert session.recompute(*tensors())[1] == receipt.proposal_digest


def test_initial_prefix_bound_and_token_types(tmp_path):
    with ChunkStore(tmp_path, subject_id="subject-1") as store:
        with pytest.raises(ValueError):
            store.open_sequence("spark", "branch-a", "seq-1",
                                initial_prefix=range(MAX_TOKENS + 1))
        with pytest.raises(ValueError):
            store.open_sequence("spark", "branch-a", "seq-2",
                                initial_prefix=[1, "2"])


def test_tier_contract_mismatch_and_expiry(tmp_path):
    store = _tier(tmp_path)
    store.publish("rec", b"payload", generation=1, kind="evidence",
                  producer="producer-v1", representation="rep-v1",
                  scope_digest="a" * 64, valid_from=100, valid_to=200)
    assert store.read("rec", contract=_contract("other")).status == "mismatch"
    assert store.read("rec", contract=_contract(representation="other")).status == "mismatch"
    assert store.read("rec", contract=_contract(scope="b" * 64)).status == "mismatch"
    assert store.read("rec", contract=_contract(cutoff=50)).status == "not_yet_valid"
    assert store.read("rec", contract=_contract(cutoff=200)).status == "expired"
    assert store.read("rec", contract=_contract(cutoff=150)).status == "ready"


def test_tier_movement_is_accounted(tmp_path):
    store = _tier(tmp_path)
    store.publish("rec", b"movement", generation=1, kind="evidence",
                  producer="producer-v1", representation="rep-v1",
                  scope_digest="a" * 64)
    before = store.metrics_document()
    store.demote("rec")
    after_demote = store.metrics_document()
    assert after_demote["movement_bytes"] > 0
    assert after_demote["bytes_written"] > before["bytes_written"]
    store.promote("rec")
    assert store.metrics_document()["movement_bytes"] > after_demote["movement_bytes"]


def test_dirty_hot_does_not_block_clean_eviction(tmp_path):
    limits = TierLimits(hot_bytes=1024, warm_bytes=4096, cold_bytes=8192,
                        identity_reservation_bytes=16, replay_reservation_bytes=16)
    store = TieredBlobStore(tmp_path, limits=limits, hot_capacity_bytes=100)
    store.publish("p_dirty", b"a" * 30, generation=1, kind="p",
                  producer="p:expert", representation="r", scope_digest="s",
                  dirty=True)
    store.publish("clean", b"b" * 30, generation=1, kind="evidence",
                  producer="p", representation="r", scope_digest="s")
    store.publish("third", b"c" * 50, generation=1, kind="evidence",
                  producer="p", representation="r", scope_digest="s")
    assert "p_dirty" in store._hot
    assert "clean" not in store._hot
    assert "third" in store._hot


def test_malformed_manifest_is_quarantined(tmp_path):
    store = _tier(tmp_path)
    store.publish("bad", b"bytes", generation=1, kind="evidence",
                  producer="p", representation="r", scope_digest="s")
    path = tmp_path / "warm" / "bad" / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["generation"] = "not-an-integer"
    path.write_text(json.dumps(manifest))
    report = store.recover()
    assert report["status"] == "degraded"
    assert "warm/bad" in report["damaged_records"]
    assert not (tmp_path / "warm" / "bad").exists()


def test_reset_task_clears_previous_workspace_without_store(tmp_path):
    tools = make_case(tmp_path)
    runtime = make_runtime(tools, tmp_path)
    runtime.previous_workspace = CoreState(
        torch.zeros(1, 2, 4), torch.ones(1, 2, dtype=torch.bool),
        torch.zeros(1, 24), "producer", runtime.state.digest,
        runtime.state.revision, 2)
    with FormationStore(tmp_path / "formation", subject_id="subject-1",
                        source_epochs=(0,),
                        producer_versions={"runtime": "v1"},
                        representation_version="rep-v1",
                        codec_id="codec-v1") as formation:
        adapter = TaskRuntimeFormationAdapter(runtime, formation, branch_id="e3",
                                              purpose="test")
        adapter.reset_task_workspace()
        assert runtime.previous_workspace is None


def test_adapter_derives_admitted_epoch_and_validates_items(tmp_path):
    with FormationStore(tmp_path / "formation", subject_id="subject-1",
                        source_epochs=(0, 1),
                        producer_versions={"runtime": "v1"},
                        representation_version="rep-v1",
                        codec_id="codec-v1") as formation:
        current = formation.pin(branch_id="e3", purpose="test",
                                mode="assistant_for_source",
                                as_of=time.time_ns() + 10 ** 9,
                                task_id="task-1", task_revision=0,
                                include_verification_states=("unverified",))
        good = InteractionProposal(
            "epoch-1", "subject-1", "e3", "test", "assistant_for_source",
            "task-1", 0, 1, 1, current.producer_versions,
            current.representation_version, current.codec_id,
            current.permitted_uses,
            observations=(Observation("o1", "epoch one", "host", 1,
                                      time.time_ns(), "r1",
                                      verification_state="unverified"),),
            outcomes=(OutcomeReceipt("u1", "a1", "completed", "r1",
                                     time.time_ns(), "one", source_epoch=1),))
        assert formation.commit_interaction(good, pin=current).status == "committed"
        bad = InteractionProposal(
            "epoch-2", "subject-1", "e3", "test", "assistant_for_source",
            "task-1", 0, 2, 1, current.producer_versions,
            current.representation_version, current.codec_id,
            current.permitted_uses,
            observations=(Observation("o2", "mixed epoch", "host", 0,
                                      time.time_ns(), "r2",
                                      verification_state="unverified"),))
        with pytest.raises(ScopeViolation):
            formation.commit_interaction(bad, pin=current)
        runtime = types.SimpleNamespace(
            state=types.SimpleNamespace(task_id="task-1", revision=0, outcomes=(),
                                        source_epoch=2),
            journal=None, workspace_store=None)
        adapter = TaskRuntimeFormationAdapter(runtime, formation, branch_id="e3",
                                              purpose="test")
        with pytest.raises(ScopeViolation):
            adapter.begin_interaction()
        assert adapter._pin is None


def test_intervention_order_and_pin_cleanup(tmp_path):
    tools = make_case(tmp_path)
    runtime = make_runtime(tools, tmp_path,
                           workspace_store=WorkspaceStore(tmp_path / "workspace"))
    with FormationStore(tmp_path / "formation", subject_id="subject-1",
                        source_epochs=(0,),
                        producer_versions={"runtime": "v1"},
                        representation_version="rep-v1",
                        codec_id="codec-v1") as formation:
        adapter = TaskRuntimeFormationAdapter(runtime, formation, branch_id="e3",
                                              purpose="test")
        outcome = adapter.apply_intervention("identity_then_task")
        assert outcome["order"] == ["identity", "task"]

        class Boom:
            def __call__(self, request):
                raise RuntimeError("proposal failure")

        with pytest.raises(RuntimeError):
            adapter.run(Boom(), 1)
        assert adapter._pin is None
        adapter.begin_interaction()
        assert adapter._pin is not None


def test_unverified_successor_cannot_erase_verified_parent(tmp_path):
    with make_store(tmp_path) as store:
        current = _pin(store)
        verified = Observation(
            observation_id="verified", text="verified fact", source_id="host",
            source_epoch=0, observed_at=time.time_ns(), receipt_id="host-receipt",
            verification_state="verified", verifier_receipt_id="verifier-1")
        base = commit(store, current, [verified])
        record_id = base.receipt.record_ids[0]
        with pytest.raises(ScopeViolation):
            store.correct(record_id, text="unverified rewrite", reason="review",
                          pin=_pin(store))
        corrected = store.correct(record_id, text="verified rewrite",
                                  reason="verified review", pin=_pin(store),
                                  verifier_receipt_id="verifier-2")
        assert corrected.status == "committed"


def test_revoked_p_is_not_ready_through_generic_read(tmp_path):
    store = _tier(tmp_path)
    store.publish_p("expert", weights=b"w", moments=b"m", clock=1,
                    encoder="encoder-v1", lineage=["r1"])
    state = store.read_p("expert")
    manifest = state["manifest"]
    contract = ReadContract(manifest["producer"], manifest["representation"],
                            manifest["scope_digest"], cutoff=10 ** 9)
    assert store.read(state["record_id"], contract=contract).status == "ready"
    cache_key = store.projections.key(
        state["record_id"], generation=1, producer="p:expert",
        representation="encoder-v1", reader_release="r1", site=3,
        eligibility_revision=0)
    store.projections.put(cache_key, object())
    store.replay.register(state["record_id"], generation=1, producer="p:expert")
    second = _tier(tmp_path)
    assert second.read(state["record_id"], contract=contract).status == "ready"
    second_key = second.projections.key(
        state["record_id"], generation=1, producer="p:expert",
        representation="encoder-v1", reader_release="r1", site=3,
        eligibility_revision=0)
    second.projections.put(second_key, object())
    second.replay.register(state["record_id"], generation=1, producer="p:expert")
    store.revoke_p("expert", reason="revoked", authorized_by="host")
    assert second.read(state["record_id"], contract=contract).status == "ineligible"
    assert store.projections.get(cache_key) is None
    assert not store.replay.eligible(state["record_id"], generation=1,
                                     producer="p:expert")
    assert second.projections.get(second_key) is None
    assert not second.replay.eligible(state["record_id"], generation=1,
                                      producer="p:expert")
    with pytest.raises(PermissionError):
        store.demote(state["record_id"])
    with pytest.raises(PermissionError):
        store.promote(state["record_id"])
    with pytest.raises(PermissionError):
        store.evict_p("expert")
    recovery = store.recover()
    assert any(item.endswith(state["record_id"])
               for item in recovery["revoked_p_records"])
    store.restore_p("expert", authorized_by="host")
    assert store.read_p("expert")["clock"] == 1
    assert second.read_p("expert")["clock"] == 1


def test_formation_authority_binds_manifest_identity_and_cutoff(tmp_path):
    with make_store(tmp_path) as store:
        current = _pin(store)
        observation = Observation(
            observation_id="o1", text="tier fact", source_id="host",
            source_epoch=0, observed_at=time.time_ns(), receipt_id="r1",
            verification_state="unverified")
        report = commit(store, current, observation and [observation])
        record_id = report.receipt.record_ids[0]
        current = _pin(store)
        tiers = _tier(tmp_path / "tiers")
        manifest = tiers.publish(
            record_id, b"fact", generation=report.receipt.generation,
            kind="evidence", producer="formation",
            representation=current.representation_version,
            scope_digest=current.digest, valid_from=100)
        good = formation_authority(store, current)
        assert tiers.read(record_id, authority=good).status == "ready"
        wrong = formation_authority(store, current, representation="other")
        assert tiers.read(record_id, authority=wrong).status == "mismatch"
        expired = formation_authority(store, current,
                                      cutoff=manifest["valid_from"] - 1)
        assert tiers.read(record_id, authority=expired).status == "not_yet_valid"


def test_chunk_recompute_before_first_proposal_uses_chunk_start(tmp_path):
    with ChunkStore(tmp_path, subject_id="subject-1") as store:
        session = make_session(store)
        keys, values = tensors()
        session.begin_chunk([20])
        session.propose(keys, values)
        committed = session.commit().receipt
        committed_state = session.state.detach().clone()
        session.begin_chunk([21])
        expected = tensor_digest(functional_inner_update(
            keys, values, committed_state)[0].detach())
        digest = session.recompute(keys, values)[1]
        assert digest == expected
        assert digest != committed.proposal_digest


def test_initial_prefix_at_global_bound_is_sealed(tmp_path):
    with ChunkStore(tmp_path, subject_id="subject-1") as store:
        tokens = tuple(range(MAX_TOKENS))
        snapshot = store.open_sequence("spark", "branch-a", "sealed",
                                       initial_prefix=tokens)
        assert snapshot.committed_position == MAX_TOKENS
        assert snapshot.status == "sealed"
        session = ChunkSession(store, snapshot, torch.zeros(1, 4, 4),
                               committed_tokens=tokens)
        with pytest.raises(ChunkError):
            session.begin_chunk([1])


def test_p_pointer_is_validated_and_failure_atomic(tmp_path):
    store = _tier(tmp_path)
    store.publish_p("expert", weights=b"w1", moments=b"m1", clock=1,
                    encoder="encoder-v1", lineage=["r1"])
    store.publish_p("expert", weights=b"w2", moments=b"m2", clock=2,
                    encoder="encoder-v2", lineage=["r1"], tier="cold")
    state = store.read_p("expert")
    assert state["clock"] == 2 and state["pointer_state"] == "valid"
    pointer_path = tmp_path / "p_current" / "expert.json"
    pointer = json.loads(pointer_path.read_text())
    pointer["payload_sha256"] = "0" * 64
    pointer_path.write_text(json.dumps(pointer))
    with pytest.raises(Exception):
        store.read_p("expert")

    failing = _tier(tmp_path / "failure")
    real = failing._write_p_pointer
    def interrupted(*args, **kwargs):
        raise RuntimeError("pointer publication interrupted")
    failing._write_p_pointer = interrupted
    with pytest.raises(RuntimeError):
        failing.publish_p("expert", weights=b"w", moments=b"m", clock=1,
                          encoder="encoder-v1", lineage=["r1"])
    with pytest.raises(Exception):
        failing.read_p("expert")
    failing._write_p_pointer = real


def test_contract_cutoff_required_and_rejected_bytes_are_charged(tmp_path):
    store = _tier(tmp_path)
    payload = b"payload-bytes"
    store.publish("record", payload, generation=1, kind="e",
                  producer="expected", representation="r1", scope_digest="scope",
                  valid_from=0, valid_to=10)
    before = store.metrics_document()
    no_cutoff = ReadContract("expected", "r1", "scope")
    result = store.read("record", contract=no_cutoff)
    assert result.status == "contract_required"
    after = store.metrics_document()
    assert after["attempted_bytes"] >= len(payload)
    assert after["bytes_read"] == before["bytes_read"]
    wrong = store.read("record", contract=ReadContract("other", "r1", "scope",
                                                      cutoff=5))
    assert wrong.status == "mismatch"
    corrupt_store = _tier(tmp_path / "corrupt")
    corrupt_store.publish("bad", payload, generation=1, kind="e",
                          producer="expected", representation="r1",
                          scope_digest="scope")
    corrupt_store.demote("bad")
    (tmp_path / "corrupt" / "cold" / "bad" / "payload.bin").write_bytes(b"x")
    corrupt = _tier(tmp_path / "corrupt").read(
        "bad", contract=ReadContract("expected", "r1", "scope", cutoff=5))
    assert corrupt.status == "corrupt"
    assert corrupt.bytes_read >= 1


def test_combined_corruption_and_deleted_tombstone_recovery(tmp_path):
    with make_store(tmp_path) as store:
        healthy = commit(store, _pin(store), [_obs("healthy", "healthy fact")],
                         interaction="interaction-healthy")
        revoked = commit(store, _pin(store), [_obs("revoked", "revoked fact")],
                         interaction="interaction-revoked")
        corrupt = commit(store, _pin(store), [_obs("corrupt", "corrupt fact")],
                         interaction="interaction-corrupt")
        revoked_id = revoked.receipt.record_ids[0]
        healthy_id = healthy.receipt.record_ids[0]
        corrupt_payload = corrupt.receipt.payload_digest
        store.revoke(revoked_id, reason="privacy")
    (tmp_path / "payloads" / f"{corrupt_payload}.json").write_bytes(b"{broken")
    connection = sqlite3.connect(tmp_path / "authority.sqlite3")
    connection.execute("DELETE FROM tombstones WHERE record_id=?", (revoked_id,))
    connection.commit()
    connection.close()
    with make_store(tmp_path) as store:
        report = store.recover()
        assert report.status == "degraded"
        assert report.derived_rebuilt is True
        assert store.read_record(healthy_id, _pin(store)).text == "healthy fact"
        with pytest.raises(IneligibleRecord):
            store.read_record(revoked_id, _pin(store))
        assert store._connection.execute(
            "SELECT 1 FROM tombstones WHERE record_id=?", (revoked_id,)).fetchone()


def test_p_marker_fsync_failure_rolls_back_without_stickiness(tmp_path, monkeypatch):
    import replica_lab.successors.tiers as tiers
    first = _tier(tmp_path)
    first.publish_p("expert", weights=b"w", moments=b"m", clock=1,
                    encoder="encoder-v1", lineage=["r1"])
    second = _tier(tmp_path)
    real_fsync = tiers.TieredBlobStore._fsync_directory
    def failing(directory):
        raise RuntimeError("directory fsync failed")
    monkeypatch.setattr(tiers.TieredBlobStore, "_fsync_directory", staticmethod(failing))
    with pytest.raises(RuntimeError):
        first.revoke_p("expert", reason="crash", authorized_by="host")
    # Failed revoke rolled the marker back: no sticky revocation.
    assert second.read_p("expert")["clock"] == 1

    monkeypatch.setattr(tiers.TieredBlobStore, "_fsync_directory", staticmethod(real_fsync))
    first.revoke_p("expert", reason="revoked", authorized_by="host")
    with pytest.raises(PermissionError):
        second.read_p("expert")

    monkeypatch.setattr(tiers.TieredBlobStore, "_fsync_directory", staticmethod(failing))
    with pytest.raises(RuntimeError):
        first.restore_p("expert", authorized_by="host")
    # Failed restore restored the durable marker: still revoked.
    with pytest.raises(PermissionError):
        second.read_p("expert")

    monkeypatch.setattr(tiers.TieredBlobStore, "_fsync_directory", staticmethod(real_fsync))
    first.restore_p("expert", authorized_by="host")
    assert second.read_p("expert")["clock"] == 1


def test_p_pointer_stale_is_repaired_to_latest_valid(tmp_path):
    store = _tier(tmp_path)
    store.publish_p("expert", weights=b"w1", moments=b"m1", clock=1,
                    encoder="encoder-v1", lineage=["r1"])
    pointer_path = tmp_path / "p_current" / "expert.json"
    first_pointer = json.loads(pointer_path.read_text())
    store.publish_p("expert", weights=b"w2", moments=b"m2", clock=2,
                    encoder="encoder-v2", lineage=["r1"])
    pointer_path.write_text(json.dumps(first_pointer))
    state = store.read_p("expert")
    assert state["clock"] == 2 and state["pointer_state"] == "repaired"
    assert json.loads(pointer_path.read_text())["clock"] == 2


def test_p_movement_invalidates_clocked_owner_caches(tmp_path):
    store = _tier(tmp_path)
    store.publish_p("expert", weights=b"w", moments=b"m", clock=1,
                    encoder="encoder-v1", lineage=["r1"])
    state = store.read_p("expert")
    key = store.projections.key(state["record_id"], generation=1,
                                producer="p:expert", representation="encoder-v1",
                                reader_release="r1", site=3,
                                eligibility_revision=0)
    store.projections.put(key, object())
    store.replay.register(state["record_id"], generation=1, producer="p:expert")
    store.demote(state["record_id"])
    assert store.projections.get(key) is None
    assert not store.replay.eligible(state["record_id"], generation=1,
                                     producer="p:expert")


def test_outcomes_excluded_after_revocation(tmp_path):
    with make_store(tmp_path) as store:
        current = _pin(store)
        outcome = OutcomeReceipt("u1", "a1", "completed", "r1",
                                 time.time_ns(), "one")
        report = commit(store, current, [_obs("rev", "revocable")],
                        outcomes=(outcome,), interaction="interaction-rev")
        record_id = report.receipt.record_ids[0]
        old_pin = _pin(store)
        store.revoke(record_id, reason="privacy")
        assert store.outcomes(old_pin) == ()
        assert store.outcomes(_pin(store)) == ()


def test_minimal_authority_cannot_bypass_manifest_contract(tmp_path):
    store = _tier(tmp_path)
    store.publish("record", b"payload", generation=1, kind="e",
                  producer="expected", representation="r1",
                  scope_digest="scope", valid_to=10)
    minimal = store.read("record", authority=lambda record_id: {"eligible": True})
    assert minimal.status == "contract_required"
    partial = store.read("record", authority=lambda record_id: {
        "eligible": True, "producer": "expected"})
    assert partial.status == "contract_required"
    complete = {"eligible": True, "producer": "expected", "representation": "r1",
                "scope_digest": "scope", "cutoff": 20, "generation": 1,
                "eligibility_revision": 0}
    expired = store.read("record", authority=lambda record_id: dict(complete))
    assert expired.status == "expired"


def test_quarantined_correction_deletion_fails_closed(tmp_path):
    from replica_lab.successors.formation import Observation
    with make_store(tmp_path) as store:
        current = _pin(store)
        verified = Observation(
            observation_id="verified-parent", text="verified parent",
            source_id="host", source_epoch=0, observed_at=time.time_ns(),
            receipt_id="host-receipt", verification_state="verified",
            verifier_receipt_id="verifier-1")
        base = commit(store, current, [verified],
                      interaction="interaction-parent")
        parent_id = base.receipt.record_ids[0]
        corrected = store.correct(parent_id, text="corrected",
                                  reason="verified", pin=_pin(store),
                                  verifier_receipt_id="verifier-2")
        correction_id = corrected.receipt.record_ids[0]
        correction_payload = corrected.receipt.payload_digest
        healthy = commit(store, _pin(store), [_obs("healthy", "healthy fact")],
                         interaction="interaction-healthy")
        healthy_id = healthy.receipt.record_ids[0]
    (tmp_path / "payloads" / f"{correction_payload}.json").write_bytes(b"{broken")
    connection = sqlite3.connect(tmp_path / "authority.sqlite3")
    connection.execute("DELETE FROM tombstones")
    connection.commit()
    connection.close()
    with make_store(tmp_path) as store:
        report = store.recover()
        assert report.status == "degraded" and report.derived_rebuilt is True
        assert store.read_record(healthy_id, _pin(store)).text == "healthy fact"
        visible = {item.record_id for item in store.visible(_pin(store))}
        assert healthy_id in visible
        assert parent_id not in visible and correction_id not in visible
        with pytest.raises(IneligibleRecord):
            store.read_record(parent_id, _pin(store))
        with pytest.raises(Exception):
            store.read_record(correction_id, _pin(store))


def test_quarantined_event_derived_rows_are_not_trusted(tmp_path):
    with make_store(tmp_path) as store:
        healthy = commit(store, _pin(store), [_obs("healthy", "healthy fact")],
                         interaction="interaction-healthy")
        victim = commit(store, _pin(store), [_obs("victim", "victim fact")],
                        interaction="interaction-victim")
        healthy_id = healthy.receipt.record_ids[0]
        victim_id = victim.receipt.record_ids[0]
        victim_event = victim.receipt.event_key
        victim_payload = victim.receipt.payload_digest
    (tmp_path / "payloads" / f"{victim_payload}.json").write_bytes(b"{broken")
    connection = sqlite3.connect(tmp_path / "authority.sqlite3")
    connection.execute(
        "INSERT INTO notes(note_id, event_key, branch_id, purpose, interaction_id, "
        "origin, text, text_digest, committed_at_ns) VALUES "
        "('tampered-note', ?, 'branch-a', 'formation-test', 'i', 'model_generated', "
        "'x', ?, 0)", (victim_event, "0" * 64))
    connection.execute(
        "INSERT INTO record_lineage(record_id, ancestor_id) VALUES (?, ?)",
        (healthy_id, victim_id))
    connection.execute(
        "INSERT INTO tombstones(record_id, kind, reason, event_key, at_ns, "
        "observed_at, generation) VALUES (?, 'ancestor_superseded', 'tamper', ?, "
        "0, 0, 1)", (healthy_id, victim_event))
    connection.commit()
    connection.close()
    with make_store(tmp_path) as store:
        report = store.recover()
        assert report.status == "degraded"
        assert store.read_record(healthy_id, _pin(store)).text == "healthy fact"
        assert store.read_provisional(branch_id="branch-a",
                                      purpose="formation-test") == ()
        assert store._connection.execute(
            "SELECT 1 FROM record_lineage WHERE record_id=?", (healthy_id,)
        ).fetchone() is None
        assert store._connection.execute(
            "SELECT 1 FROM tombstones WHERE record_id=?", (healthy_id,)
        ).fetchone() is None


def test_quarantined_revocation_deletion_fails_closed(tmp_path):
    with make_store(tmp_path) as store:
        healthy = commit(store, _pin(store), [_obs("healthy", "healthy fact")],
                         interaction="interaction-healthy")
        revoked = commit(store, _pin(store), [_obs("revoked", "revoked fact")],
                         interaction="interaction-revoked")
        healthy_id = healthy.receipt.record_ids[0]
        revoked_id = revoked.receipt.record_ids[0]
        store.revoke(revoked_id, reason="privacy")
        revocation_payload = store._connection.execute(
            "SELECT payload_digest FROM events WHERE kind='revocation'").fetchone()[0]
    (tmp_path / "payloads" / f"{revocation_payload}.json").write_bytes(b"{broken")
    connection = sqlite3.connect(tmp_path / "authority.sqlite3")
    connection.execute("DELETE FROM tombstones")
    connection.execute("DELETE FROM revocations")
    connection.commit()
    connection.close()
    with make_store(tmp_path) as store:
        report = store.recover()
        assert report.status == "degraded" and report.derived_rebuilt is True
        assert store.read_record(healthy_id, _pin(store)).text == "healthy fact"
        with pytest.raises(IneligibleRecord):
            store.read_record(revoked_id, _pin(store))


def test_recover_reconciles_stale_and_missing_p_pointers(tmp_path):
    store = _tier(tmp_path)
    store.publish_p("expert", weights=b"w1", moments=b"m1", clock=1,
                    encoder="encoder-v1", lineage=["r1"])
    store.publish_p("expert", weights=b"w2", moments=b"m2", clock=2,
                    encoder="encoder-v2", lineage=["r1"])
    pointer_path = tmp_path / "p_current" / "expert.json"
    pointer = json.loads(pointer_path.read_text())
    pointer["record_id"] = "p_expert_c99"
    pointer["clock"] = 99
    pointer_path.write_text(json.dumps(pointer))
    with pytest.raises(TierError):
        store.read_p("expert")
    report = store.recover()
    assert report["status"] == "degraded"
    assert any(item["pointer"] in ("repaired", "repaired_invalid")
               for item in report["reconciled_p_pointers"])
    assert store.read_p("expert")["clock"] == 2
    pointer_path.unlink()
    assert store.recover()["status"] == "degraded"
    assert json.loads(pointer_path.read_text())["clock"] == 2


def test_post_replace_pointer_failure_restores_prior_pointer(tmp_path, monkeypatch):
    import replica_lab.successors.tiers as tiers
    first = _tier(tmp_path)
    first.publish_p("expert", weights=b"w1", moments=b"m1", clock=1,
                    encoder="encoder-v1", lineage=["r1"])
    second = _tier(tmp_path)
    real_fsync = tiers.TieredBlobStore._fsync_directory
    def failing(directory):
        raise RuntimeError("pointer directory fsync failed")
    monkeypatch.setattr(tiers.TieredBlobStore, "_fsync_directory", staticmethod(failing))
    with pytest.raises(RuntimeError):
        first.publish_p("expert", weights=b"w2", moments=b"m2", clock=2,
                        encoder="encoder-v2", lineage=["r1"])
    monkeypatch.setattr(tiers.TieredBlobStore, "_fsync_directory", staticmethod(real_fsync))
    pointer = json.loads((tmp_path / "p_current" / "expert.json").read_text())
    assert pointer["clock"] == 1
    assert first.read_p("expert")["clock"] == 1
    assert second.read_p("expert")["clock"] == 1


def test_fallback_accounts_primary_attempt_and_changes(tmp_path):
    primary = _tier(tmp_path / "primary")
    primary.publish("canonical", b"canonical", generation=0, kind="canonical",
                    producer="host", representation="text", scope_digest="d" * 64)
    primary.publish("bad", b"primary-bytes", generation=1, kind="e",
                    producer="producer-v1", representation="rep-v1",
                    scope_digest="a" * 64)
    primary.demote("bad")
    (tmp_path / "primary" / "cold" / "bad" / "payload.bin").write_bytes(b"corrupt")
    reader = _tier(tmp_path / "primary")
    before = reader.metrics_document()
    result = reader.read(
        "bad", contract=_contract("producer-v1", "rep-v1", "a" * 64),
        fallback_id="canonical",
        fallback_contract=_contract("host", "text", "d" * 64),
        fallback_tokens=3)
    assert result.status == "fallback"
    after = reader.metrics_document()
    assert after["attempted_bytes"] > before["attempted_bytes"]
    assert after["failed_reads"] >= 1
    assert after["fallback_tokens"] == 3


def test_p_read_revalidates_revocation_after_io(tmp_path, monkeypatch):
    store = _tier(tmp_path)
    store.publish_p("expert", weights=b"w", moments=b"m", clock=1,
                    encoder="encoder-v1", lineage=["r1"])
    real = store._p_versions
    def racing(owner):
        versions = real(owner)
        store.revoke_p(owner, reason="racing", authorized_by="host")
        return versions
    monkeypatch.setattr(store, "_p_versions", racing)
    with pytest.raises(PermissionError):
        store.read_p("expert")


def test_revoke_temp_marker_cleanup_preserves_original_error(tmp_path, monkeypatch):
    import replica_lab.successors.tiers as tiers
    store = _tier(tmp_path)
    store.publish_p("expert", weights=b"w", moments=b"m", clock=1,
                    encoder="encoder-v1", lineage=["r1"])
    real_replace = tiers.os.replace
    def failing_replace(source, target):
        if str(target).endswith("expert.json") and "p_revoked" in str(target):
            raise RuntimeError("marker replace failed")
        return real_replace(source, target)
    monkeypatch.setattr(tiers.os, "replace", failing_replace)
    with pytest.raises(RuntimeError):
        store.revoke_p("expert", reason="cleanup", authorized_by="host")
    leftovers = [path for path in (tmp_path / "p_revoked").iterdir()
                 if path.name.endswith(".tmp")]
    assert leftovers == []


def test_outcomes_revalidate_revocation_after_io(tmp_path):
    with make_store(tmp_path) as store:
        current = _pin(store)
        outcome = OutcomeReceipt("u1", "a1", "completed", "r1",
                                 time.time_ns(), "one")
        report = commit(store, current, [_obs("race", "race")],
                        outcomes=(outcome,), interaction="interaction-race")
        record_id = report.receipt.record_ids[0]
        real = store._load_payload
        state = {"raced": False}
        def racing(payload_digest):
            data = real(payload_digest)
            if not state["raced"]:
                state["raced"] = True
                store.revoke(record_id, reason="racing")
            return data
        store._load_payload = racing
        assert store.outcomes(_pin(store)) == ()


def test_p_publication_rechecks_revocation_before_pointer(tmp_path):
    store = _tier(tmp_path)
    store.publish_p("expert", weights=b"w1", moments=b"m1", clock=1,
                    encoder="encoder-v1", lineage=["r1"])
    real = store._write_p_pointer
    def racing(owner, **kwargs):
        store.revoke_p(owner, reason="racing", authorized_by="host")
        return real(owner, **kwargs)
    store._write_p_pointer = racing
    with pytest.raises(PermissionError):
        store.publish_p("expert", weights=b"w2", moments=b"m2", clock=2,
                        encoder="encoder-v2", lineage=["r1"])
    assert "p_expert_c2" not in store._p_record_ids("expert")
    with pytest.raises(PermissionError):
        store.read_p("expert")


def test_pointer_cleanup_preserves_original_error(tmp_path, monkeypatch):
    store = _tier(tmp_path)
    store.publish_p("expert", weights=b"w1", moments=b"m1", clock=1,
                    encoder="encoder-v1", lineage=["r1"])
    def failing_pointer(*args, **kwargs):
        raise RuntimeError("original pointer failure")
    def failing_cleanup(*args, **kwargs):
        raise RuntimeError("cleanup failure must not mask")
    monkeypatch.setattr(store, "_write_p_pointer", failing_pointer)
    monkeypatch.setattr(store, "_remove_p_record", failing_cleanup)
    with pytest.raises(RuntimeError, match="original pointer failure"):
        store.publish_p("expert", weights=b"w2", moments=b"m2", clock=2,
                        encoder="encoder-v2", lineage=["r1"])


def test_recover_removes_orphan_pointer(tmp_path):
    store = _tier(tmp_path)
    pointer_root = tmp_path / "p_current"
    pointer_root.mkdir()
    (pointer_root / "ghost.json").write_text(json.dumps({
        "format": "replica.successors.p_pointer.v1", "owner": "ghost",
        "clock": 1, "record_id": "p_ghost_c1",
        "payload_sha256": "0" * 64, "manifest_sha256": "0" * 64,
        "committed_at_ns": 1}))
    report = store.recover()
    assert report["status"] == "degraded"
    assert not (pointer_root / "ghost.json").exists()


def test_measurement_writer_is_no_overwrite(tmp_path):
    import importlib.util
    script = pathlib.Path(__file__).resolve().parents[1] / "reports" / \
        "replica_successors_v12" / "run_cpu_successors_v12.py"
    spec = importlib.util.spec_from_file_location("successor_runner", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    target = tmp_path / "measurements.json"
    assert module._write_new_bytes(target, b"first") is True
    assert module._write_new_bytes(target, b"second") is False
    assert target.read_bytes() == b"first"


def test_study_attribution_uses_actual_ids(tmp_path):
    from replica_lab.successors import study
    report = study.run_later_answer_comparison(tmp_path / "study")
    measurement = report["measurement"]
    assert measurement["enabled_used_new_observation"] is True
    assert measurement["disabled_used_new_observation"] is False
    assert measurement["removing_write_removes_improvement"] is True
    assert measurement["improvement_attributable_to_write"] is True
    assert measurement["provisional_not_evidence"] is True
    assert report["conditions"]["enabled"]["after"]["record_id"] == \
        report["conditions"]["enabled"]["new_record_id"]

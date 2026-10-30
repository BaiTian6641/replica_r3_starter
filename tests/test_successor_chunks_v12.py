"""S2b chunk-start/chunk-barrier CPU contracts (isolated, no native work)."""
from __future__ import annotations

import pytest
import torch

from replica_lab.successors.chunks import (AttemptAudit, ChunkConflict, ChunkError,
                                           ChunkKey, ChunkSession, ChunkStore,
                                           FORMAT, PositionGap, PrefixChanged,
                                           PrefixOverlap, StaleChunk,
                                           functional_inner_update, prefix_digest,
                                           tensor_digest)

TOKENS = (10, 11)
WIDTH = 6
WRITER_VERSION = "chunk-writer-v1"


def make_store(tmp_path) -> ChunkStore:
    return ChunkStore(tmp_path, subject_id="subject-1")


def make_session(store: ChunkStore, *, tokens=TOKENS, max_unroll=2,
                 training=False, detach_after_commit=True) -> ChunkSession:
    snapshot = store.open_sequence("spark", "branch-a", "seq-1",
                                   initial_prefix=tokens)
    return ChunkSession(store, snapshot, torch.zeros(1, WIDTH, WIDTH),
                        committed_tokens=tokens, writer_version=WRITER_VERSION,
                        max_unroll=max_unroll, training=training,
                        detach_after_commit=detach_after_commit)


def tensors(seed=3):
    generator = torch.Generator().manual_seed(seed)
    return (torch.randn(1, 3, WIDTH, generator=generator),
            torch.randn(1, 3, WIDTH, generator=generator))


def key_for(store, *, chunk_index: int, start: int, end: int,
            prefix: str, revision: int, decision: int = 0) -> ChunkKey:
    return ChunkKey(schema=FORMAT, subject_id="subject-1", owner="spark",
                    branch_id="branch-a", sequence_id="seq-1",
                    decision_revision=decision, chunk_index=chunk_index,
                    token_start=start, token_end=end, token_prefix_digest=prefix,
                    input_state_revision=revision, writer_version=WRITER_VERSION,
                    effect_kind="associate")


def test_growing_prefix_commits_once_and_advances_watermark(tmp_path):
    with make_store(tmp_path) as store:
        session = make_session(store)
        first_key = session.begin_chunk([20, 21, 22])
        keys, values = tensors()
        session.propose(keys, values)
        first = session.commit(audit=AttemptAudit(attempt_id="attempt-1",
                                                  checkpoint_recomputation_id="r1"))
        assert first.status == "committed"
        assert first.receipt.token_start == 2 and first.receipt.token_end == 5
        assert session.snapshot.committed_position == 5
        second_key = session.begin_chunk([23])
        session.propose(keys, values)
        second = session.commit()
        assert second.receipt.token_start == 5 and second.receipt.token_end == 6
        assert session.snapshot.committed_position == 6
        assert len(store.history("spark", "branch-a", "seq-1")) == 2
        assert first_key.logical_key != second_key.logical_key
        assert second.receipt.revision_after == first.receipt.revision_after + 1


def test_retry_returns_prior_receipt_without_advancing(tmp_path):
    with make_store(tmp_path) as store:
        session = make_session(store)
        key = session.begin_chunk([20, 21])
        keys, values = tensors()
        session.propose(keys, values)
        first = session.commit(audit=AttemptAudit(attempt_id="attempt-1",
                                                  transport_retries=0))
        snapshot = store.snapshot("spark", "branch-a", "seq-1")
        retry = store.barrier(snapshot, key=key,
                              proposal_digest=first.receipt.proposal_digest,
                              next_prefix_digest=first.receipt.prefix_digest,
                              audit=AttemptAudit(attempt_id="attempt-2",
                                                 transport_retries=1,
                                                 checkpoint_recomputation_id="recompute-9"))
        assert retry.status == "duplicate"
        assert retry.receipt.receipt_digest == first.receipt.receipt_digest
        assert store.snapshot("spark", "branch-a", "seq-1").revision == 1
        assert len(store.history("spark", "branch-a", "seq-1")) == 1


def test_conflicting_retry_is_rejected_and_recorded(tmp_path):
    with make_store(tmp_path) as store:
        session = make_session(store)
        key = session.begin_chunk([20])
        keys, values = tensors()
        session.propose(keys, values)
        first = session.commit()
        snapshot = store.snapshot("spark", "branch-a", "seq-1")
        with pytest.raises(ChunkConflict):
            store.barrier(snapshot, key=key, proposal_digest="f" * 64,
                          next_prefix_digest=first.receipt.prefix_digest)
        assert store.metrics_document()["conflicts"] == 1


def test_overlap_changed_prefix_and_gap_are_refused(tmp_path):
    with make_store(tmp_path) as store:
        session = make_session(store)
        session.begin_chunk([20])
        keys, values = tensors()
        session.propose(keys, values)
        session.commit()
        snapshot = store.snapshot("spark", "branch-a", "seq-1")
        overlap = key_for(store, chunk_index=9, start=0, end=4,
                          prefix=prefix_digest(()),
                          revision=snapshot.revision)
        with pytest.raises(PrefixOverlap):
            store.barrier(snapshot, key=overlap, proposal_digest="a" * 64,
                          next_prefix_digest=prefix_digest((1,)))
        gap = key_for(store, chunk_index=10, start=snapshot.committed_position + 2,
                      end=snapshot.committed_position + 4,
                      prefix=snapshot.prefix_digest, revision=snapshot.revision)
        with pytest.raises(PositionGap):
            store.barrier(snapshot, key=gap, proposal_digest="b" * 64,
                          next_prefix_digest=prefix_digest((1,)))
        changed = key_for(store, chunk_index=11, start=snapshot.committed_position,
                          end=snapshot.committed_position + 1,
                          prefix=prefix_digest((99,)),
                          revision=snapshot.revision)
        with pytest.raises(PrefixChanged):
            store.barrier(snapshot, key=changed, proposal_digest="c" * 64,
                          next_prefix_digest=prefix_digest((1,)))


def test_recomputation_is_pure(tmp_path):
    with make_store(tmp_path) as store:
        session = make_session(store)
        session.begin_chunk([20, 21])
        keys, values = tensors()
        session.propose(keys, values)
        first = session.commit()
        history = len(store.history("spark", "branch-a", "seq-1"))
        position = store.snapshot("spark", "branch-a", "seq-1").committed_position
        digest_a = session.recompute(keys, values)[1]
        digest_b = session.recompute(keys, values)[1]
        assert digest_a == digest_b
        assert len(store.history("spark", "branch-a", "seq-1")) == history
        assert store.snapshot("spark", "branch-a", "seq-1").committed_position == position
        assert store.snapshot("spark", "branch-a", "seq-1").revision == 1


def test_correction_discards_outstanding_chunk(tmp_path):
    with make_store(tmp_path) as store:
        session = make_session(store)
        session.begin_chunk([20])
        keys, values = tensors()
        session.propose(keys, values)
        session.commit()
        session.begin_chunk([21, 22])
        session.propose(keys, values)
        outstanding = session.snapshot
        corrected = store.correct("spark", "branch-a", "seq-1",
                                  reason="late correction")
        assert corrected.status == "active"
        assert corrected.decision_revision == 1
        with pytest.raises(StaleChunk):
            session.commit()
        session.invalidate()
        new_snapshot = store.snapshot("spark", "branch-a", "seq-1")
        assert new_snapshot.revision == outstanding.revision + 1
        assert new_snapshot.committed_position == outstanding.committed_position
        key = key_for(store, chunk_index=2, start=new_snapshot.committed_position,
                      end=new_snapshot.committed_position + 1,
                      prefix=new_snapshot.prefix_digest,
                      revision=new_snapshot.revision,
                      decision=new_snapshot.decision_revision)
        result = store.barrier(new_snapshot, key=key,
                               proposal_digest="d" * 64,
                               next_prefix_digest=prefix_digest(TOKENS + (20, 21)))
        assert result.status == "committed"
        assert result.receipt.decision_revision == 1


def test_branch_isolation_and_causal_visibility(tmp_path):
    with make_store(tmp_path) as store:
        branch_a = store.open_sequence("spark", "branch-a", "seq-1",
                                       initial_prefix=TOKENS)
        branch_b = store.open_sequence("spark", "branch-b", "seq-1",
                                       initial_prefix=TOKENS)
        assert branch_a.prefix_digest == branch_b.prefix_digest
        session = ChunkSession(store, branch_a, torch.zeros(1, WIDTH, WIDTH),
                               committed_tokens=TOKENS)
        session.begin_chunk([20])
        keys, values = tensors()
        session.propose(keys, values)
        session.commit()
        assert store.snapshot("spark", "branch-b", "seq-1").revision == 0
        assert store.snapshot("spark", "branch-b", "seq-1").committed_position == 2
        next_chunk = session.begin_chunk([21])
        assert next_chunk.input_state_revision == 1
        assert next_chunk.token_prefix_digest == session.snapshot.prefix_digest


def test_live_unroll_and_detach_boundaries(tmp_path):
    with make_store(tmp_path) as store:
        session = make_session(store, max_unroll=2, training=True,
                               detach_after_commit=False)
        keys, values = tensors()
        keys.requires_grad_(True)
        session.begin_chunk([20])
        state_one = session.propose(keys, values)
        assert state_one.requires_grad
        session.commit()
        session.begin_chunk([21])
        state_two = session.propose(keys, values)
        session.propose(keys, values)
        assert state_two.requires_grad and state_two.grad_fn is not None
        loss = state_two.square().sum() + state_one.square().sum()
        loss.backward()
        assert keys.grad is not None and bool(keys.grad.abs().sum() > 0)
        with pytest.raises(ChunkError):
            session.propose(keys, values)
        # A durable commit receives a detached digest, never the live graph.
        result = session.commit()
        assert result.status == "committed"
        session.begin_chunk([22])
        session.propose(keys, values)
        assert session.recompute(keys, values)[0].requires_grad


def test_restart_preserves_receipts_and_rejects_stale_snapshot(tmp_path):
    with make_store(tmp_path) as store:
        session = make_session(store)
        session.begin_chunk([20])
        keys, values = tensors()
        session.propose(keys, values)
        first = session.commit()
        stale = store.snapshot("spark", "branch-a", "seq-1")
        session.begin_chunk([21])
        session.propose(keys, values)
        session.commit()
    with make_store(tmp_path) as store:
        assert store.recover()["status"] == "ok"
        assert len(store.history("spark", "branch-a", "seq-1")) == 2
        retry = store.barrier(stale, key=ChunkKey(
            schema=FORMAT, subject_id="subject-1", owner="spark",
            branch_id="branch-a", sequence_id="seq-1", decision_revision=0,
            chunk_index=0, token_start=2, token_end=3,
            token_prefix_digest=prefix_digest(TOKENS),
            input_state_revision=0, writer_version=WRITER_VERSION,
            effect_kind="associate"),
            proposal_digest=first.receipt.proposal_digest,
            next_prefix_digest=first.receipt.prefix_digest)
        assert retry.status == "duplicate"
        assert retry.receipt.receipt_digest == first.receipt.receipt_digest


def test_tensor_digest_is_deterministic_and_sensitive():
    first = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4)
    second = first.clone()
    assert tensor_digest(first) == tensor_digest(second)
    second[0, 0, 0] = 9.0
    assert tensor_digest(first) != tensor_digest(second)
    assert functional_inner_update(
        torch.ones(1, 2, 4), torch.ones(1, 2, 4), torch.zeros(1, 4, 4))[0].shape \
        == (1, 4, 4)

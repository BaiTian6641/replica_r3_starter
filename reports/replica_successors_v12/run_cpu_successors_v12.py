#!/usr/bin/env python3
"""Run the isolated CPU successor measurements and write summary artifacts.

This script executes only the new ``replica_lab.successors`` modules plus the
existing CPU core/task/identity interfaces.  It loads no model, downloads
nothing, starts no native job and never imports ``runtime_pilot``.

Usage:
    .venv/bin/python reports/replica_successors_v12/run_cpu_successors_v12.py \
        --output reports/replica_successors_v12
"""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from replica_lab.architecture.contracts import CoreSpec, fingerprint  # noqa: E402
from replica_lab.architecture.core import CognitiveCore  # noqa: E402
from replica_lab.replica.identity import (IdentityStore, PrincipleHypothesis,  # noqa: E402
                                          SourceEpisode)
from replica_lab.replica.reader import IdentityReader  # noqa: E402
from replica_lab.runtime.journal import TaskJournal  # noqa: E402
from replica_lab.runtime.loop import ProposalOutput, TaskRuntime  # noqa: E402
from replica_lab.runtime.task_graph import (Budget, ExactTaskState,  # noqa: E402
                                            SuccessCondition)
from replica_lab.runtime.tools import FixtureWorkspace  # noqa: E402
from replica_lab.runtime.workspace_store import WorkspaceStore  # noqa: E402
from replica_lab.successors import study  # noqa: E402
from replica_lab.successors.adapters import (CoreWorkspaceAdapter,  # noqa: E402
                                             IdentityAdapter,
                                             TaskRuntimeFormationAdapter)
from replica_lab.successors.chunks import (AttemptAudit, ChunkConflict,  # noqa: E402
                                           ChunkKey, ChunkSession, ChunkStore,
                                           FORMAT, PrefixChanged, PrefixOverlap,
                                           StaleChunk, prefix_digest)
from replica_lab.successors.formation import FormationStore  # noqa: E402
from replica_lab.successors.protocols import all_protocols  # noqa: E402
from replica_lab.successors.tiers import (TieredBlobStore, TierLimits)  # noqa: E402

FIXTURE_GOAL = "Make the evaluator pass by setting the required value."
REPAIR = '{"rate": 0.25}\n'
ORIGINAL = '{"rate": 1.0}\n'


def _write_case(root: Path) -> FixtureWorkspace:
    case_root = root / "cases" / "successor-case"
    (case_root / "workspace").mkdir(parents=True)
    (case_root / "evaluator").mkdir(parents=True)
    (case_root / "workspace" / "config.json").write_text(ORIGINAL)
    (case_root / "evaluator" / "test_rate.py").write_text(
        'import json, pathlib\n'
        'cfg = json.loads(pathlib.Path("workspace/config.json").read_text())\n'
        'assert cfg["rate"] == 0.25, "config fails the bound check"\n'
        'print("ok")\n')
    case = {
        "format": "replica.e3.fixture.v1", "case_id": "successor-case",
        "role": "train", "goal": FIXTURE_GOAL,
        "success_condition": {"condition_id": "sc-1", "description": "evaluator exits 0"},
        "constraints": ["Only workspace/ is writable"],
        "allowed_tools": ["read_file", "write_file", "run_test"],
        "workspace_files": ["workspace/config.json"],
        "evaluator": "evaluator/test_rate.py",
        "workspace_entry": "workspace/config.json",
        "reference_repair": {"path": "workspace/config.json", "content": REPAIR},
        "variant": {"expected_value": 0.25},
    }
    (case_root / "case.json").write_text(json.dumps(case, indent=2))
    return FixtureWorkspace(case_root)


def _task_state(tools: FixtureWorkspace) -> ExactTaskState:
    return ExactTaskState(
        task_id=tools.case_id, subject="subject-1", branch="e3", purpose="test",
        source_epoch=0, goal=tools.goal,
        success_conditions=(SuccessCondition("sc-1", "evaluator exits 0"),),
        constraints=(), subgoals=(), claims=(), evidence=(), obligations=(),
        actions=(), outcomes=(), notes=(), budget=Budget(8, 16, 4096, 512),
        binding={"runtime": "v1"}, allowed_tools=tuple(tools.allowed_tools), revision=0)


class Scripted:
    def __init__(self, raws):
        self.raws = list(raws)
        self.requests = []

    def __call__(self, request):
        self.requests.append(request)
        raw = self.raws.pop(0) if self.raws else '{"kind": "stop"}'
        return ProposalOutput(
            raw_text=raw, prompt_tokens=11, generated_token_ids=(7, 8),
            generated_tokens=2, frame_digest="frame-" + str(len(raw)),
            workspace=None, workspace_stamp=None, wall_seconds=0.0, notes=())


def _task_raws() -> list[str]:
    return [
        json.dumps({"kind": "invoke_tool", "tool": "write_file",
                    "arguments": {"path": "workspace/config.json", "content": REPAIR}}),
        json.dumps({"kind": "invoke_tool", "tool": "run_test", "arguments": {}}),
    ]


def measure_task_formation(root: Path) -> dict:
    """Run one exact TaskRuntime with formation enabled and one disabled."""
    results = {}
    for enabled in (True, False):
        base = root / f"task_{'enabled' if enabled else 'disabled'}"
        base.mkdir(parents=True)
        tools = _write_case(base)
        journal = TaskJournal.open(base / "task.sqlite", {
            "task_id": tools.case_id, "subject": "subject-1", "branch": "e3",
            "purpose": "test", "source_epoch": 0, "goal_digest": fingerprint(tools.goal),
            "component_versions": {"runtime": "v1"}})
        runtime = TaskRuntime(tools, journal,
                              workspace_store=WorkspaceStore(base / "workspace_store"))
        runtime.initialize(_task_state(tools))
        with FormationStore(
                base / "formation", subject_id="subject-1", source_epochs=(0,),
                producer_versions={"runtime": "v1", "core": "successor-measure"},
                representation_version="rep-measure-v1",
                codec_id="codec-measure-v1") as formation:
            adapter = TaskRuntimeFormationAdapter(
                runtime, formation, branch_id="e3", purpose="test",
                writer_enabled=enabled)
            outcome = adapter.run(Scripted(_task_raws()), 2)
            later = formation.pin(
                branch_id="e3", purpose="test", mode="assistant_for_source",
                as_of=time.time_ns() + 3600 * 10 ** 9, task_id=tools.case_id,
                task_revision=0, include_verification_states=("verified", "unverified"))
            visible = formation.visible(later)
            provisional = formation.read_provisional(branch_id="e3", purpose="test")
            results["enabled" if enabled else "disabled"] = {
                "task_journal_bytes": (base / "task.sqlite").stat().st_size,
                "workspace_store_bytes": sum(
                    path.stat().st_size for path in (base / "workspace_store").glob("*")
                    if path.is_file()),
                "decisions": outcome["decisions"],
                "success": outcome["success"],
                "formation_status": outcome["formation"]["status"],
                "visible_records": len(visible),
                "provisional_notes": len(provisional),
                "record_kinds": [record.kind for record in visible],
            }
    return {"conditions": results,
            "improvement_requires_write": (
                results["enabled"]["visible_records"] > 0
                and results["disabled"]["visible_records"] == 0)}


def measure_core_and_identity(root: Path) -> dict:
    spec = CoreSpec(width=32, slots=8, steps=2, heads=4, task_features=24,
                    ffn_hidden=64, control_hidden=40)
    adapter = CoreWorkspaceAdapter(spec, producer="successor-core-measure")
    generator = torch.Generator().manual_seed(7)
    sense = torch.randn(1, 3, spec.width, generator=generator)
    sense_valid = torch.ones(1, 3, dtype=torch.bool)
    memory = torch.randn(1, 4, spec.width, generator=generator)
    memory_valid = torch.ones(1, 4, dtype=torch.bool)
    task = torch.zeros(1, spec.task_features)
    task[:, 0] = 0.25
    digest = fingerprint({"task": "successor-measure"})
    training = adapter.build(sense, sense_valid, memory, memory_valid, task,
                             task_digest=digest, revision=1, training=True)
    loss = training.workspace.square().sum()
    loss.backward()
    gradient = (adapter.core.slot_queries.grad is not None
                and bool(adapter.core.slot_queries.grad.abs().sum() > 0))
    inference = adapter.build(sense, sense_valid, memory, memory_valid, task,
                              task_digest=digest, revision=2)
    store = WorkspaceStore(root / "workspace_store")
    adapter.save(store, inference)
    loaded = adapter.load(store, expected_task_digest=digest, expected_revision=2)
    reader = IdentityReader(width=spec.width, heads=spec.heads)
    identity_adapter = CoreWorkspaceAdapter(spec, identity_reader=reader)
    slots = torch.randn(1, 8, spec.width, generator=generator)
    valid = torch.ones(1, 8, dtype=torch.bool)
    identity_adapter.previous = None
    without = identity_adapter.build(sense, sense_valid, memory, memory_valid, task,
                                     task_digest=digest, revision=3)
    identity_adapter.previous = None
    disabled = identity_adapter.build(
        sense, sense_valid, memory, memory_valid, task, task_digest=digest,
        revision=3, identity_slots=slots, identity_valid=valid,
        identity_enabled=False)
    identity_adapter.previous = None
    enabled_untrained = identity_adapter.build(
        sense, sense_valid, memory, memory_valid, task, task_digest=digest,
        revision=3, identity_slots=slots, identity_valid=valid,
        identity_enabled=True)
    adapter.reset_task(store)
    workspace_bytes = sum(tensor.numel() * tensor.element_size()
                          for tensor in inference.tensors())
    return {
        "core_parameters": sum(p.numel() for p in adapter.core.parameters()),
        "core_parameter_bytes_fp32": sum(
            p.numel() * p.element_size() for p in adapter.core.parameters()),
        "workspace_tensor_bytes": workspace_bytes,
        "identity_slots_bytes": int(slots.numel() * slots.element_size()),
        "identity_reservation_proposed_bytes": TierLimits().identity_reservation_bytes,
        "training_gradient_present": gradient,
        "inference_detached": not inference.workspace.requires_grad,
        "workspace_roundtrip": loaded is not None and loaded.stamp() == inference.stamp(),
        "workspace_reset_clears": store.load(expected_task_digest=digest,
                                             expected_revision=2) is None,
        "identity_disabled_exact_parity": bool(torch.equal(
            without.workspace, disabled.workspace)),
        "identity_untrained_zero_init": bool(torch.equal(
            without.workspace, enabled_untrained.workspace)),
        "identity_digest_bound": disabled.identity_digest is None
        and enabled_untrained.identity_digest is not None,
    }


def measure_identity_store(root: Path) -> dict:
    store = IdentityStore("fictional-subject")
    episode = SourceEpisode(
        episode_id="ep-1", subject_id="fictional-subject", source_id="interview",
        source_hash="a" * 64, observed_at=1, event_at=1,
        text="I prefer reliability first when the cost is bounded.",
        kind="preference", authenticity="source_report",
        permitted_uses=("retrieval",))
    store.add_episode(episode)
    principle = PrincipleHypothesis(
        principle_id="pr-1", subject_id="fictional-subject",
        statement="prefer reliability", condition="when cost is bounded",
        exceptions=(), supporting_episode_ids=("ep-1",),
        contradicting_episode_ids=(), status="approved")
    store.add_principle(principle)
    store.build_snapshot("snap-1", version=1, created_at=1,
                         episode_ids=("ep-1",), principle_ids=("pr-1",),
                         unresolved_questions=(), permitted_uses=("retrieval",))
    adapter = IdentityAdapter(store)
    context = adapter.resolve("snap-1", mode="assistant_for_source",
                              branch_id="branch-a", purpose="test", cutoff=10)
    rendered = adapter.render()
    adapter.reset()
    return {"rendered": bool(rendered.strip()), "null_renders_empty": adapter.render() == "",
            "null_is_null": adapter.is_null,
            "context_summary_bounded": len(context.summary) < 4096,
            "identity_cannot_verify": True}


def measure_chunks(root: Path) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    result = {}
    with ChunkStore(root, subject_id="subject-1") as store:
        tokens = (10, 11)
        snapshot = store.open_sequence("spark", "branch-a", "seq-1",
                                       initial_prefix=tokens)
        session = ChunkSession(store, snapshot, torch.zeros(1, 8, 8),
                               committed_tokens=tokens, max_unroll=2,
                               training=False)
        generator = torch.Generator().manual_seed(11)
        key1 = session.begin_chunk([20, 21, 22])
        keys = torch.randn(1, 3, 8, generator=generator)
        values = torch.randn(1, 3, 8, generator=generator)
        session.propose(keys, values)
        first = session.commit(audit=AttemptAudit(attempt_id="attempt-1",
                                                  checkpoint_recomputation_id="recompute-1"))
        # Recompute purity: identical tensor digest, no new commit.
        recomputed_a = session.recompute(keys, values)[1]
        recomputed_b = session.recompute(keys, values)[1]
        history_after_first = len(store.history("spark", "branch-a", "seq-1"))
        # Growing prefix: next chunk starts at the new watermark.
        key2 = session.begin_chunk([23, 24])
        session.propose(keys, values)
        second = session.commit(audit=AttemptAudit(attempt_id="attempt-3"))
        # Retry of the first logical key returns the prior receipt.
        retry = store.barrier(snapshot, key=key1,
                              proposal_digest=first.receipt.proposal_digest,
                              next_prefix_digest=first.receipt.prefix_digest,
                              audit=AttemptAudit(attempt_id="attempt-2"))
        # Conflict on the same logical key with different bytes.
        conflict = None
        try:
            store.barrier(snapshot, key=key1,
                          proposal_digest="f" * 64,
                          next_prefix_digest=prefix_digest(tokens + (20, 21, 22)))
        except ChunkConflict as error:
            conflict = str(error)
        # Overlap and changed-prefix refusals.
        overlap = changed = None
        overlap_key = ChunkKey(
            schema=FORMAT, subject_id="subject-1", owner="spark",
            branch_id="branch-a", sequence_id="seq-1",
            decision_revision=session.snapshot.decision_revision,
            chunk_index=10, token_start=0,
            token_end=session.snapshot.committed_position + 1,
            token_prefix_digest=prefix_digest(()),
            input_state_revision=session.snapshot.revision,
            writer_version="chunk-writer-v1", effect_kind="associate")
        try:
            store.barrier(session.snapshot, key=overlap_key,
                          proposal_digest="e" * 64,
                          next_prefix_digest=first.receipt.prefix_digest)
        except PrefixOverlap as error:
            overlap = str(error)
        changed_key = ChunkKey(
            schema=FORMAT, subject_id="subject-1", owner="spark",
            branch_id="branch-a", sequence_id="seq-1", decision_revision=0,
            chunk_index=9, token_start=session.snapshot.committed_position,
            token_end=session.snapshot.committed_position + 1,
            token_prefix_digest=prefix_digest(tokens),
            input_state_revision=session.snapshot.revision,
            writer_version="chunk-writer-v1", effect_kind="associate")
        try:
            store.barrier(session.snapshot, key=changed_key,
                          proposal_digest="d" * 64,
                          next_prefix_digest=prefix_digest(tokens + (99,)))
        except PrefixChanged as error:
            changed = str(error)
        # Correction during an outstanding chunk.
        outstanding_snapshot = session.snapshot
        session.begin_chunk([25])
        session.propose(keys, values)
        store.correct("spark", "branch-a", "seq-1", reason="late correction")
        stale = None
        try:
            session.commit()
        except StaleChunk as error:
            stale = str(error)
        # Track 1 was proposed before the correction and is discarded.
        session.invalidate()
        # Branch isolation: another branch has its own revision and watermark.
        other = store.open_sequence("spark", "branch-b", "seq-1",
                                    initial_prefix=tokens)
        history = store.history("spark", "branch-a", "seq-1")
        result = {
            "first_commit": first.status,
            "second_commit": second.status,
            "retry_status": retry.status,
            "retry_same_receipt": retry.receipt.receipt_digest
            == first.receipt.receipt_digest,
            "history_after_first": history_after_first,
            "recompute_digest_equal": recomputed_a == recomputed_b,
            "recompute_created_commits": len(history) != 2,
            "committed_watermark": session.snapshot.committed_position,
            "committed_tokens_expected": len(tokens) + 5,
            "conflict_rejected": conflict is not None,
            "overlap_refused": overlap is not None,
            "changed_prefix_refused": changed is not None,
            "stale_after_correction": stale is not None,
            "branch_b_position": other.committed_position,
            "branch_b_revision": other.revision,
            "correction_revision": store.snapshot(
                "spark", "branch-a", "seq-1").revision,
            "metrics": store.metrics_document(),
            "recovery": store.recover(),
        }
    return result


def measure_tiers(root: Path) -> dict:
    limits = TierLimits(hot_bytes=1024, warm_bytes=4096, cold_bytes=8192,
                        identity_reservation_bytes=64,
                        replay_reservation_bytes=64)
    root.mkdir(parents=True, exist_ok=True)
    store = TieredBlobStore(root, limits=limits, hot_capacity_bytes=256,
                            io_delay_seconds=0.002)
    payload = b"evidence-payload-" * 4
    manifest = store.publish("rec-a", payload, generation=1, kind="evidence",
                             producer="producer-v1", representation="rep-v1",
                             scope_digest="b" * 64)
    hot = store.read("rec-a")
    store.demote("rec-a")
    cold_store = TieredBlobStore(root, limits=limits, hot_capacity_bytes=256,
                                 io_delay_seconds=0.002)
    cold = cold_store.read("rec-a", expected_generation=1)
    cold_store.promote("rec-a")
    warm_store = TieredBlobStore(root, limits=limits, hot_capacity_bytes=256,
                                 io_delay_seconds=0.002)
    warm = warm_store.read("rec-a", expected_generation=1)
    warm_store.demote("rec-a")
    cold_store2 = TieredBlobStore(root, limits=limits, hot_capacity_bytes=256,
                                  io_delay_seconds=0.002)
    # Bounded pending on a cold read, then correction during the ticket.
    pending = cold_store2.read("rec-a", expected_generation=1, budget_seconds=0.0001)
    cold_store2.invalidate("rec-a", reason="correction", revoked=True)
    resumed = cold_store2.resume(pending.ticket_id) if pending.ticket_id else None
    # Missing and corrupt cases on separate records.
    missing = cold_store.read("rec-missing")
    cold_store.publish("rec-corrupt", b"good", generation=2, kind="evidence",
                       producer="producer-v1", representation="rep-v1",
                       scope_digest="c" * 64)
    (root / "warm" / "rec-corrupt" / "payload.bin").write_bytes(b"corrupted")
    corrupt_store = TieredBlobStore(root, limits=limits, hot_capacity_bytes=256)
    corrupt = corrupt_store.read("rec-corrupt")
    # P state publication, eviction and recovery.
    p_store = TieredBlobStore(root / "p", limits=limits, hot_capacity_bytes=256)
    p_store.publish_p("p1", weights=b"w" * 32, moments=b"m" * 32, clock=3,
                      encoder="encoder-v1", lineage=("ep-1",))
    p_state = p_store.read_p("p1")
    evicted_tier = p_store.evict_p("p1")
    p_after = p_store.read_p("p1")
    (root / "p" / "staging" / "interrupted-move").mkdir(parents=True, exist_ok=True)
    recovery = p_store.recover()
    # Projection cache identity includes the eligibility revision.
    cache = p_store.projections
    key = cache.key("rec-a", generation=1, producer="producer-v1",
                    representation="rep-v1", reader_release="reader-v1", site=3,
                    eligibility_revision=0)
    cache.put(key, "projection")
    replay_registered = p_store.replay.register("rec-a", generation=1,
                                                producer="producer-v1") is None
    p_store.replay.invalidate("rec-a")
    replay_after = p_store.replay.eligible("rec-a", generation=1,
                                           producer="producer-v1")
    cleared = cache.invalidate("rec-a")
    fallback_store = TieredBlobStore(root / "fallback", limits=limits,
                                     hot_capacity_bytes=256)
    fallback_store.publish("canonical", b"canonical-text", generation=0,
                           kind="canonical", producer="host", representation="text",
                           scope_digest="d" * 64)
    fallback = fallback_store.read("absent", fallback_id="canonical",
                                   fallback_tokens=12)
    return {
        "hot_status": hot.status, "warm_status": warm.status,
        "cold_status": cold.status,
        "warm_tier": warm.tier, "cold_tier": cold.tier,
        "pending_status": pending.status,
        "resumed_status": None if resumed is None else resumed.status,
        "missing_status": missing.status, "corrupt_status": corrupt.status,
        "p_state_generation": p_state["clock"],
        "p_state_evicted_tier": evicted_tier,
        "p_state_readable_after_evict": p_after["clock"] == 3,
        "recovery": recovery,
        "projection_entries_cleared": cleared,
        "replay_registered": replay_registered,
        "replay_eligible_after_invalidation": replay_after,
        "fallback_status": fallback.status,
        "fallback_tokens": fallback.fallback_tokens,
        "metrics": {
            "hot_read": store.metrics_document(),
            "cold_read": cold_store.metrics_document(),
            "warm_read": warm_store.metrics_document(),
            "cold_resume_and_failure": cold_store2.metrics_document(),
        },
    }


def measure_formation_recovery(root: Path) -> dict:
    from replica_lab.successors.formation import (CorruptAuthority, InteractionProposal,
                                                  Observation)

    def store_at(path):
        return FormationStore(path, subject_id="subject-1", source_epochs=(0,),
                              producer_versions={"reader": "r1", "writer": "w1"},
                              representation_version="rep-v1", codec_id="codec-v1")

    def pin(instance):
        return instance.pin(branch_id="branch-a", purpose="recovery",
                            mode="assistant_for_source",
                            as_of=time.time_ns() + 3600 * 10 ** 9,
                            task_id="task-1", task_revision=0,
                            include_verification_states=("unverified",))

    def make_proposal(current, name, text):
        return InteractionProposal(
            interaction_id="interaction-" + name, subject_id=current.subject_id,
            branch_id=current.branch_id, purpose=current.purpose,
            mode=current.mode, task_id=current.task_id,
            task_revision=current.task_revision, decision_revision=1,
            source_epoch=0, producer_versions=current.producer_versions,
            representation_version=current.representation_version,
            codec_id=current.codec_id, permitted_uses=current.permitted_uses,
            observations=(Observation(
                observation_id=name, text=text, source_id="host", source_epoch=0,
                observed_at=time.time_ns(), receipt_id="receipt-" + name,
                verification_state="unverified", authenticity="host_observed"),))

    crash_root = root / "crash"
    with store_at(crash_root) as store:
        current = pin(store)
        prepared = store.prepare(make_proposal(current, "o1", "prepared only"),
                                 pin=current)
    with store_at(crash_root) as store:
        before = store.recover()
        assert before.events == 0 and before.orphan_payloads == 1
        assert store.visible(current) == ()
        first = store.commit(prepared)
    with store_at(crash_root) as store:
        after = store.recover()
        duplicate = store.commit(prepared)
        crash = {"before_events": before.events, "before_orphans": before.orphan_payloads,
                 "after_events": after.events, "after_orphans": after.orphan_payloads,
                 "duplicate_status": duplicate.status,
                 "duplicate_same_receipt": duplicate.receipt.receipt_digest
                 == first.receipt.receipt_digest}

    corrupt_root = root / "corrupt"
    with store_at(corrupt_root) as store:
        current = pin(store)
        healthy = store.commit_interaction(make_proposal(current, "h1", "healthy"), pin=current)
        corrupt_proposal = make_proposal(current, "c1", "will corrupt")
        broken = store.commit_interaction(corrupt_proposal, pin=current)
        later = pin(store)
    (corrupt_root / "payloads" / f"{broken.receipt.payload_digest}.json").write_bytes(b"{bad")
    with store_at(corrupt_root) as store:
        report = store.recover()
        healthy_read = store.read_record(healthy.receipt.record_ids[0], later)
        try:
            store.visible(later)
            visible_refused = False
        except CorruptAuthority:
            visible_refused = True
        corrupt = {"status": report.status,
                   "quarantined": len(report.quarantined),
                   "healthy_read": healthy_read.text,
                   "corrupt_visible_refused": visible_refused}
    return {"crash": crash, "corrupt": corrupt}


def measure_interventions(root: Path) -> dict:
    from replica_lab.architecture.contracts import CoreState

    results = {}
    for order in ("none", "task_then_identity", "identity_then_task", "both_reset"):
        base = root / order
        base.mkdir(parents=True)
        tools = _write_case(base)
        runtime = TaskRuntime(tools, TaskJournal.open(base / "task.sqlite", {
            "task_id": tools.case_id, "subject": "subject-1", "branch": "e3",
            "purpose": "test", "source_epoch": 0,
            "goal_digest": fingerprint(tools.goal),
            "component_versions": {"runtime": "v1"}}),
            workspace_store=WorkspaceStore(base / "workspace_store"))
        runtime.initialize(_task_state(tools))
        width = runtime.workspace_width
        core = CoreState(torch.zeros(1, 8, width),
                         torch.ones(1, 8, dtype=torch.bool),
                         torch.zeros(1, 24), "producer", runtime.state.digest,
                         runtime.state.revision, 2)
        runtime.workspace_store.save(core)
        with FormationStore(base / "formation", subject_id="subject-1",
                            source_epochs=(0,),
                            producer_versions={"runtime": "v1"},
                            representation_version="rep-v1",
                            codec_id="codec-v1") as formation:
            adapter = TaskRuntimeFormationAdapter(runtime, formation, branch_id="e3",
                                                  purpose="test")
            outcome = adapter.apply_intervention(order)
            results[order] = {
                **outcome,
                "workspace_absent": runtime.workspace_store.load(
                    expected_task_digest=runtime.state.digest) is None,
                "goal_unchanged": runtime.state.goal == tools.goal,
            }
    comparable = ("identity_null", "task_workspace_cleared", "workspace_absent",
                  "goal_unchanged")
    return {"conditions": results,
            "orders_equivalent": all(
                results["task_then_identity"][key]
                == results["identity_then_task"][key] for key in comparable)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--only", choices=("formation", "chunks", "tiers",
                                           "adapters", "all"), default="all")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="successors-v12-") as temporary:
        work = Path(temporary)
        measurements: dict = {
            "format": "replica.successors.measurements.v1",
            "status": "isolated CPU engineering measurements; no native job",
            "environment": {"python": platform.python_version(),
                            "torch": torch.__version__,
                            "platform": platform.platform()},
        }
        if args.only in ("formation", "all"):
            measurements["formation_study"] = study.run_later_answer_comparison(
                work / "study")
            measurements["formation_recovery"] = measure_formation_recovery(
                work / "recovery")
        if args.only in ("adapters", "all"):
            measurements["task_formation"] = measure_task_formation(work / "task")
            measurements["core_identity"] = measure_core_and_identity(work / "core")
            measurements["identity_store"] = measure_identity_store(work / "identity")
            measurements["interventions"] = measure_interventions(work / "interventions")
        if args.only in ("chunks", "all"):
            measurements["chunks"] = measure_chunks(work / "chunks")
        if args.only in ("tiers", "all"):
            measurements["tiers"] = measure_tiers(work / "tiers")
        measurements["elapsed_seconds"] = time.perf_counter() - started
    (output / "measurements.json").write_text(
        json.dumps(measurements, indent=2) + "\n", encoding="utf-8")
    protocols = all_protocols()
    (output / "protocols.json").write_text(
        json.dumps(protocols, indent=2) + "\n", encoding="utf-8")
    (output / "interface_map.json").write_text(
        json.dumps(protocols["interface_map"], indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "elapsed_seconds": measurements["elapsed_seconds"],
        "formation_used_new": measurements.get("formation_study", {})
        .get("measurement", {}).get("enabled_used_new_observation"),
        "task_improvement_requires_write": measurements.get("task_formation", {})
        .get("improvement_requires_write"),
        "chunk_retry_same": measurements.get("chunks", {}).get("retry_same_receipt"),
        "tier_cold": measurements.get("tiers", {}).get("cold_status"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

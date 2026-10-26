"""Self-contained contracts for delayed visual-memory episodes and CPU checks."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import shutil

import pytest

from replica_lab.visual_memory_episodes import (
    ADMISSION_FORMAT,
    CONTEXT_FREE_ARMS,
    ContextRemovedError,
    ContractError,
    Episode,
    EpisodeRuntime,
    ExactSourceProvider,
    LabelBoundaryError,
    NativeEvaluationGated,
    ReferenceContractAdapter,
    StaleVisualSourceError,
    StateError,
    UNKNOWN_ANSWER,
    VisualQuery,
    admit_visual_memory_prepared,
    assert_native_evaluation_allowed,
    prepare_visual_memory,
    run_visual_memory_cpu,
    score_visual_memory,
    verify_visual_memory_prepared,
)

# The import above deliberately uses only public names; this alias keeps the
# label-boundary check below readable without importing a private constant.
LABEL_KEYS = {
    "expected", "expected_answer", "expected_observation_id", "gold", "label",
    "labels", "query_kind", "phase", "split", "role", "target_role",
    "correct", "incorrect", "quality",
}

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/visual_memory_v1.json"


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _episodes(prepared: Path) -> list[Episode]:
    return [Episode.from_dict(row)
            for row in _jsonl(prepared / "episodes.jsonl")]


def _queries(prepared: Path) -> list[dict]:
    return _jsonl(prepared / "queries.jsonl")


def _labels(prepared: Path) -> list[dict]:
    return _jsonl(prepared / "labels.jsonl")


def _image(prepared: Path, observation) -> bytes:
    return (prepared / "images" / f"{observation.asset_id}.png").read_bytes()


def _vq(row: dict) -> VisualQuery:
    return VisualQuery.from_dict(row)


def _episode_and_queries(prepared: Path, episode_index: int = 0):
    episode = _episodes(prepared)[episode_index]
    queries = [row for row in _queries(prepared)
               if row["episode_id"] == episode.episode_id]
    labels = {row["query_id"]: row for row in _labels(prepared)
              if row["episode_id"] == episode.episode_id}
    return episode, queries, labels


def _walk_keys(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _walk_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_keys(item)


@pytest.fixture(scope="session")
def prepared(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("visual_memory_prepared") / "prepared"
    prepare_visual_memory(out, config_path=CONFIG)
    receipt = admit_visual_memory_prepared(out)
    assert receipt["format"] == ADMISSION_FORMAT
    assert receipt["status"] == "admitted", receipt
    return out


def test_preparation_is_deterministic_and_admitted(tmp_path, prepared):
    second = tmp_path / "prepared2"
    manifest = prepare_visual_memory(second, config_path=CONFIG)
    first_manifest = json.loads((prepared / "manifest.json").read_text())
    assert manifest["config_sha256"] == first_manifest["config_sha256"]
    assert manifest["protocol_sha256"] == first_manifest["protocol_sha256"]
    assert manifest["counts"] == first_manifest["counts"]
    assert manifest["files"] == first_manifest["files"]
    assert (prepared / "protocol.json").read_bytes() == (second / "protocol.json").read_bytes()
    assert (prepared / "episodes.jsonl").read_bytes() == (second / "episodes.jsonl").read_bytes()
    assert (prepared / "queries.jsonl").read_bytes() == (second / "queries.jsonl").read_bytes()
    assert (prepared / "labels.jsonl").read_bytes() == (second / "labels.jsonl").read_bytes()
    for relative in manifest["files"]:
        if relative.startswith("images/"):
            assert (prepared / relative).read_bytes() == (second / relative).read_bytes()
    pins = verify_visual_memory_prepared(second)
    assert pins["implementation_sha256"] == manifest["implementation_sha256"]
    receipt = admit_visual_memory_prepared(second)
    assert receipt["status"] == "admitted"
    assert len(manifest["worlds"]) == manifest["counts"]["episodes"]
    assert manifest["counts"]["images"] == 2 * manifest["counts"]["episodes"]


def test_labels_are_separate_from_sensing_and_query_inputs(prepared):
    episode_keys = set(_walk_keys([json.loads(line) for line in
                                   (prepared / "episodes.jsonl").read_text().splitlines()]))
    query_keys = set(_walk_keys(_queries(prepared)))
    annotation_keys = set(_walk_keys(_jsonl(prepared / "source_annotations.jsonl")))
    assert not (episode_keys & LABEL_KEYS)
    assert not (query_keys & LABEL_KEYS)
    assert not (annotation_keys & LABEL_KEYS)
    labels = _labels(prepared)
    assert labels
    assert {"expected_answer", "query_kind", "split", "phase"} <= set(labels[0])
    for query in _queries(prepared):
        assert "expected_answer" not in query
        assert "query_kind" not in query


def test_admission_rejects_tampered_image_and_preserves_rejection(tmp_path, prepared):
    copy = tmp_path / "tampered"
    shutil.copytree(prepared, copy)
    image = next((copy / "images").glob("*.png"))
    image.write_bytes(image.read_bytes() + b"tampered")
    with pytest.raises(ContractError):
        verify_visual_memory_prepared(copy)
    receipt = admit_visual_memory_prepared(copy)
    assert receipt["status"] == "rejected"
    assert receipt["format"] == ADMISSION_FORMAT


def test_delayed_access_boundary_and_no_context_after_removal(prepared):
    episode, queries, labels = _episode_and_queries(prepared)
    runtime = EpisodeRuntime.create(prepared.parent / "runtime-boundary", episode)
    runtime.observe_initial(_image(prepared, episode.initial))
    runtime.commit_initial()
    runtime.remove_context()
    assert runtime.context_removed is True
    with pytest.raises(ContextRemovedError):
        _ = runtime.context_image_bytes
    with pytest.raises(ContextRemovedError):
        _ = runtime.context_annotations
    encodes = runtime.adapter.encoded_calls
    for query in queries[:3]:
        answer = runtime.ask(_vq(query), "memory")
        assert answer.text == labels[query["query_id"]]["expected_answer"]
        assert answer.image_reads == 0
        assert answer.annotation_reads == 0
        assert answer.context_reads == 0
        assert answer.source_bytes_returned == 0
        assert answer.retrieval_performed is False
    assert runtime.adapter.encoded_calls == encodes


def test_restart_without_reencoding_old_observations(prepared):
    episode, queries, labels = _episode_and_queries(prepared)
    root = prepared.parent / "runtime-restart"
    runtime = EpisodeRuntime.create(root, episode)
    runtime.observe_initial(_image(prepared, episode.initial))
    runtime.commit_initial()
    runtime.remove_context()
    before = runtime.state.state_sha256
    first = runtime.ask(_vq(queries[0]), "memory").text
    restarted, report = EpisodeRuntime.restart(root)
    restarted.attach_episode(episode)
    assert report.encode_calls == 0
    assert report.reencoded_assets == 0
    assert report.context_removed is True
    assert report.state_sha256 == before
    assert restarted.ask(_vq(queries[0]), "memory").text == first
    assert restarted.adapter.encoded_calls == 0


def test_interrupted_restart_persists_journal_and_rejects_partial_state(prepared):
    episode, queries, labels = _episode_and_queries(prepared)
    root = prepared.parent / "runtime-interrupted"
    runtime = EpisodeRuntime.create(root, episode)
    runtime.observe_initial(_image(prepared, episode.initial))
    runtime.commit_initial()
    runtime.remove_context()
    runtime.apply_correction(_image(prepared, episode.correction))
    published = runtime.state.state_sha256
    intent = runtime.interrupt_publication()
    recovered, report = EpisodeRuntime.restart(root)
    recovered.attach_episode(episode)
    assert report.interrupted is True
    assert intent in report.pending_intents
    assert report.state_sha256 == published
    assert report.encode_calls == 0
    assert recovered.state.state_sha256 == published
    assert recovered.ask(_vq(queries[3]), "memory").text == labels[queries[3]["query_id"]]["expected_answer"]
    assert not list(root.glob(".state.json.tmp*"))


def test_truncated_journal_tail_is_reported_and_recovered(prepared):
    episode, _queries, _labels = _episode_and_queries(prepared)
    root = prepared.parent / "runtime-truncated"
    runtime = EpisodeRuntime.create(root, episode)
    runtime.observe_initial(_image(prepared, episode.initial))
    runtime.commit_initial()
    runtime.remove_context()
    published = runtime.state.state_sha256
    with (root / EpisodeRuntime.JOURNAL_FILE).open("ab") as handle:
        handle.write(b'{"seq":999,"event":"partial"')
    recovered, report = EpisodeRuntime.restart(root)
    recovered.attach_episode(episode)
    assert report.interrupted is True
    assert report.truncated_tail_bytes > 0
    assert report.state_sha256 == published
    assert report.encode_calls == 0
    second, report2 = EpisodeRuntime.restart(root)
    assert report2.truncated_tail_bytes == 0
    assert second.state.state_sha256 == published


def test_dated_correction_and_unaffected_retention(prepared):
    episode, queries, labels = _episode_and_queries(prepared)
    adapter = ReferenceContractAdapter()
    state = adapter.commit(
        adapter.encode_observation(episode.initial, _image(prepared, episode.initial)),
        None, subject_id=episode.subject_id, branch_id=episode.branch_id,
        purpose=episode.purpose)
    before = {(fact.object_id, fact.attribute): fact.value for fact in state.facts}
    state = adapter.commit(
        adapter.encode_observation(episode.correction,
                                   _image(prepared, episode.correction)),
        state, correction=True, subject_id=episode.subject_id,
        branch_id=episode.branch_id, purpose=episode.purpose)
    after = {(fact.object_id, fact.attribute): fact.value for fact in state.facts}
    changed = episode.changed_object()
    changed_facts = [fact for fact in state.facts
                     if fact.object_id == changed and fact.attribute == "color"]
    assert any(fact.supersedes is not None for fact in changed_facts)
    assert after[(changed, "color")] != before[(changed, "color")]
    for object_id in episode.initial.object_ids():
        if object_id != changed:
            for attribute in ("color", "shape"):
                assert after[(object_id, attribute)] == before[(object_id, attribute)]
    assert state.correction_count == 1
    for query in queries:
        answer = adapter.answer(type("Q", (), {
            "target_object_id": query["target_object_id"],
            "attribute": query["attribute"],
            "asked_at": query["asked_at"],
        })(), state)
        assert answer == labels[query["query_id"]]["expected_answer"]


def test_correction_strictly_later_and_identity_preserving(prepared):
    episode, _queries, _labels = _episode_and_queries(prepared)
    adapter = ReferenceContractAdapter()
    state = adapter.commit(
        adapter.encode_observation(episode.initial, _image(prepared, episode.initial)),
        None, subject_id=episode.subject_id, branch_id=episode.branch_id,
        purpose=episode.purpose)
    early = replace(episode.correction,
                    observed_at=episode.initial.observed_at,
                    observed_date=episode.initial.observed_date)
    with pytest.raises(ContractError):
        adapter.commit(
            adapter.encode_observation(early, _image(prepared, episode.correction)),
            state, correction=True, subject_id=episode.subject_id,
            branch_id=episode.branch_id, purpose=episode.purpose)
    moved_object = replace(episode.correction.objects[0],
                           region=episode.correction.objects[0].region.__class__(
                               (1, 1, 9, 9), (5, 5)))
    moved = replace(episode.correction,
                    objects=(moved_object, *episode.correction.objects[1:]))
    with pytest.raises((ContractError, StateError)):
        adapter.commit(
            adapter.encode_observation(moved, _image(prepared, episode.correction)),
            state, correction=True, subject_id=episode.subject_id,
            branch_id=episode.branch_id, purpose=episode.purpose)


def test_unknown_query_abstains(prepared):
    episode, _queries, labels = _episode_and_queries(prepared)
    runtime = EpisodeRuntime.create(prepared.parent / "runtime-unknown", episode)
    runtime.observe_initial(_image(prepared, episode.initial))
    runtime.commit_initial()
    runtime.remove_context()
    unknown_queries = [q for q in _queries if q["episode_id"] == episode.episode_id
                       and labels[q["query_id"]]["query_kind"] == "unknown"]
    assert len(unknown_queries) == 2
    for query in unknown_queries:
        answer = runtime.ask(_vq(query), "memory")
        assert answer.text == UNKNOWN_ANSWER
        assert answer.answered is False


def test_zero_shuffled_disabled_and_context_removed_controls(prepared):
    episode, queries, labels = _episode_and_queries(prepared)
    other = _episodes(prepared)[1]
    adapter = ReferenceContractAdapter()
    state = adapter.commit(
        adapter.encode_observation(episode.initial, _image(prepared, episode.initial)),
        None, subject_id=episode.subject_id, branch_id=episode.branch_id,
        purpose=episode.purpose)
    other_state = adapter.commit(
        adapter.encode_observation(other.initial, _image(prepared, other.initial)),
        None, subject_id=other.subject_id, branch_id=other.branch_id,
        purpose=other.purpose)
    runtime = EpisodeRuntime.create(prepared.parent / "runtime-controls", episode)
    runtime.observe_initial(_image(prepared, episode.initial))
    runtime.commit_initial()
    runtime.remove_context()
    query = queries[0]
    for arm in CONTEXT_FREE_ARMS:
        answer = runtime.ask(_vq(query), arm, shuffled_state=other_state).as_dict()
        assert answer["context_reads"] == 0
        assert answer["image_reads"] == 0
        assert answer["annotation_reads"] == 0
        assert answer["retrieval_performed"] is False
        assert answer["source_bytes_returned"] == 0
        assert not (set(answer) & LABEL_KEYS)
        if arm == "memory":
            assert answer["text"] == labels[query["query_id"]]["expected_answer"]
        elif arm in ("disabled", "context_removed"):
            assert answer["text"] == UNKNOWN_ANSWER
        elif arm == "zero_state":
            assert answer["text"] == UNKNOWN_ANSWER
            assert answer["state_sha256"] != runtime.state.state_sha256
        elif arm == "shuffled_state":
            assert answer["state_sha256"] != runtime.state.state_sha256
            assert answer["source_kind"] == "shuffled_state"


def test_exact_source_retrieval_is_explicit_and_stale_rejected(tmp_path, prepared):
    episode, queries, labels = _episode_and_queries(prepared)
    provider = ExactSourceProvider(
        tmp_path / "ledger.db", tmp_path / "blobs",
        subject_id=episode.subject_id, purpose=episode.purpose)
    try:
        provider.admit_episode(
            episode, initial_bytes=_image(prepared, episode.initial),
            correction_bytes=_image(prepared, episode.correction))
        initial_query = queries[0]
        result = provider.retrieve(episode, type("Q", (), {
            "query_id": initial_query["query_id"],
            "episode_id": initial_query["episode_id"],
            "asked_at": initial_query["asked_at"],
        })())
        assert result.observation_id == episode.initial.observation_id
        assert result.retrieval_performed is True
        assert result.source_bytes_returned == episode.initial.byte_length
        post_query = queries[3]
        post = provider.retrieve(episode, type("Q", (), {
            "query_id": post_query["query_id"],
            "episode_id": post_query["episode_id"],
            "asked_at": post_query["asked_at"],
        })())
        assert post.observation_id == episode.correction.observation_id
        with pytest.raises(StaleVisualSourceError):
            provider.retrieve(episode, type("Q", (), {
                "query_id": post_query["query_id"],
                "episode_id": post_query["episode_id"],
                "asked_at": post_query["asked_at"],
            })(), expected_observation_id=episode.initial.observation_id)
        blob = provider.store.blob_path(episode.initial.image_sha256)
        blob.write_bytes(b"tampered")
        with pytest.raises(StaleVisualSourceError):
            provider.retrieve(episode, type("Q", (), {
                "query_id": initial_query["query_id"],
                "episode_id": initial_query["episode_id"],
                "asked_at": initial_query["asked_at"],
            })())
    finally:
        provider.close()


def test_full_context_control_requires_retained_context(prepared):
    episode, queries, labels = _episode_and_queries(prepared)
    root = prepared.parent / "runtime-context"
    runtime = EpisodeRuntime.create(root, episode)
    runtime.observe_initial(_image(prepared, episode.initial))
    runtime.commit_initial()
    runtime.apply_correction(_image(prepared, episode.correction))
    for query in queries:
        answer = runtime.ask(_vq(query), "full_context")
        assert answer.context_reads == 1
        assert answer.annotation_reads == 1
        assert answer.text == labels[query["query_id"]]["expected_answer"]
    runtime.remove_context()
    with pytest.raises(ContextRemovedError):
        runtime.ask(_vq(queries[0]), "full_context")


def test_score_uses_separate_labels_and_rejects_leakage(prepared):
    episode, queries, labels = _episode_and_queries(prepared)
    runtime = EpisodeRuntime.create(prepared.parent / "runtime-score", episode)
    runtime.observe_initial(_image(prepared, episode.initial))
    runtime.commit_initial()
    runtime.remove_context()
    answers = [runtime.ask(_vq(query), "memory").as_dict() for query in queries[:3]]
    # score_visual_memory accepts raw evaluator label rows.
    score = score_visual_memory(answers, [labels[q["query_id"]] for q in queries[:3]])
    assert score["outcome"]["memory"]["correct"] == 3
    assert score["quality_claim"] is False
    leaked = [dict(row) for row in answers]
    leaked[0]["expected_answer"] = "leak"
    with pytest.raises(LabelBoundaryError):
        score_visual_memory(leaked, [labels[q["query_id"]] for q in queries[:3]])
    with pytest.raises(ContractError):
        score_visual_memory(answers[:-1], [labels[q["query_id"]] for q in queries[:3]])


def test_run_cpu_contract_report_separates_outcomes_and_persistence(tmp_path, prepared):
    out = tmp_path / "cpu"
    report = run_visual_memory_cpu(prepared, out)
    assert report["status"] == "cpu_contract_checks_complete"
    assert report["native_evaluated"] is False
    assert report["adapter"]["trained"] is False
    assert report["counts"]["episodes"] == 8
    assert report["counts"]["queries"] == 64
    assert report["mechanical_persistence"]["gates"]["all_passed"] is True
    assert report["actual_outcome"]["memory"]["correct"] == 64
    assert report["actual_outcome"]["zero_state"]["correct"] < 64
    assert report["actual_outcome"]["disabled"]["correct"] < 64
    assert report["actual_outcome"]["context_removed"]["correct"] < 64
    assert report["corrections"]["memory"]["correct"] == 8
    assert report["unknowns"]["memory"]["false_answered"] == 0
    assert report["provenance"]["exact_source_retrievals"] == 64
    assert report["provenance"]["exact_source_bytes_returned"] > 0
    assert report["storage"]["state_bytes"] > 0
    assert report["storage"]["journal_bytes"] > 0
    assert report["storage"]["source_image_bytes"] > 0
    assert report["storage"]["reported_separately_from_mechanical_persistence"] is True
    assert (out / "report.json").is_file()
    answers = _jsonl(out / "answers.jsonl")
    assert answers
    assert all("expected_answer" not in row for row in answers)
    assert all(row["answer_format"] for row in answers)
    retrievals = _jsonl(out / "retrievals.jsonl")
    assert len(retrievals) == 64
    assert all(row["retrieval_performed"] for row in retrievals)


def test_native_evaluation_is_gated(prepared):
    protocol = json.loads((prepared / "protocol.json").read_text())
    with pytest.raises(NativeEvaluationGated):
        assert_native_evaluation_allowed(protocol)
    assert protocol["native_gate"]["status"] == "gated"
    assert protocol["native_gate"]["native_evaluated"] is False


def test_state_bytes_detect_tampering(prepared):
    episode, _queries, _labels = _episode_and_queries(prepared)
    adapter = ReferenceContractAdapter()
    state = adapter.commit(
        adapter.encode_observation(episode.initial, _image(prepared, episode.initial)),
        None, subject_id=episode.subject_id, branch_id=episode.branch_id,
        purpose=episode.purpose)
    data = state.to_bytes()
    with pytest.raises(StateError):
        type(state).from_bytes(data.replace(b'"value":"', b'"value":"x', 1))

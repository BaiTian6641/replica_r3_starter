"""Source-byte binding and causal role exports for the tiny synthetic slice."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest

from replica_lab.episodes import export_sft, input_view, verify_episode

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("make_episode_slice", ROOT / "scripts/make_episode_slice.py")
SLICE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SLICE)


def load_lines(path):
    return [json.loads(line) for line in path.read_bytes().splitlines()]


@pytest.fixture
def release(tmp_path):
    path = tmp_path / "slice"
    SLICE.build_slice(path)
    return path


def source_for(path, episode):
    return next(s for s in json.loads((path / "source_manifest.json").read_bytes())
                if s["source_id"] == episode["sources"][0]["source_id"])


def receipt(path, episode):
    return verify_episode(episode, verifier_id=SLICE.FAMILY + ".source-solver",
        verifier_sha256=SLICE.implementation()["generator_sha256"],
        verifier=lambda v: SLICE.independent_check(v, path, source_for(path, v)))


def test_two_regenerations_have_identical_source_spans_manifests_and_exports(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    assert SLICE.build_slice(first) == SLICE.build_slice(second)
    for path in first.rglob("*"):
        if path.is_file():
            assert path.read_bytes() == (second / path.relative_to(first)).read_bytes()


def test_twenty_cases_in_ten_preassigned_worlds_and_only_train_dev_flat_exports(release):
    manifest = SLICE.validate_release(release)
    assert manifest["lineage_audit"]["episodes"] == 20
    assert manifest["lineage_audit"]["lineage_groups"] == 10
    assert manifest["lineage_audit"]["split_counts"] == {role: 4 for role in SLICE.ROLES}
    assert sorted(p.name for p in (release / "exports").iterdir()) == ["dev.jsonl", "train.jsonl"]
    sources = json.loads((release / "source_manifest.json").read_bytes())
    assert len({s["sha256"] for s in sources}) == 10
    for role in ["train", "dev"]:
        rows = load_lines(release / "exports" / f"{role}.jsonl")
        assert len(rows) == 4 and all(row["verified"] for row in rows)
        assert {row["split"] for row in rows} == {role}
    assert manifest["tokenizer_independent"] is True


def test_old_cutoff_never_reads_late_correction_but_later_query_changes_answer(release):
    before, after, *_ = load_lines(release / "canonical/train_episodes.jsonl")
    assert before["query"]["at"] == 5 and after["query"]["at"] == 9
    assert before["target"]["response"] != after["target"]["response"]
    assert "mask-corrected" not in json.dumps(input_view(before))
    assert "mask-corrected" in json.dumps(input_view(after))
    assert SLICE.independent_check(before, release, source_for(release, before))
    contaminated = copy.deepcopy(before)
    contaminated["target"]["response"] = after["target"]["response"]
    assert receipt(release, contaminated).passed is False
    with pytest.raises(ValueError, match="failed"):
        export_sft(contaminated, verification=receipt(release, contaminated))


def test_unknown_ask_and_tombstone_use_visible_absence_and_explicit_question_policy(release):
    *_, ask, unknown = load_lines(release / "canonical/train_episodes.jsonl")
    assert ask["target"]["action"] == "ask" and unknown["target"]["action"] == "unknown"
    assert "ask for it" in ask["query"]["prompt"]
    assert "respond UNKNOWN" in unknown["query"]["prompt"]
    for episode in [ask, unknown]:
        assert len(input_view(episode)["memory_texts"]) == 1  # INPUT survives; MASK is unavailable.
        assert receipt(release, episode).passed is True
        assert "mask-late" not in json.dumps(input_view(episode))
        assert "mask-deleted" not in json.dumps(input_view(episode))


def test_solver_ignores_oracle_answers_and_context_never_contains_annotations(release):
    episode = load_lines(release / "canonical/train_episodes.jsonl")[0]
    episode["evaluator"]["oracle"]["invented_gold"] = "ORACLE_ONLY_SECRET"
    assert receipt(release, episode).passed is True
    assert "ORACLE_ONLY_SECRET" not in json.dumps(input_view(episode))


def test_current_revocation_blocks_even_historical_answer_export(release):
    episode = load_lines(release / "canonical/train_episodes.jsonl")[0]
    episode["sources"][0]["permissions"] = []
    assert receipt(release, episode).passed is True  # Truth check is not use authorization.
    assert input_view(episode)["memory_texts"] == []
    with pytest.raises(ValueError, match="lacks permission"):
        export_sft(episode, verification=receipt(release, episode))


def test_external_teacher_permission_cannot_be_invented_by_episode(release):
    episode = load_lines(release / "canonical/train_episodes.jsonl")[0]
    episode["permissions"].append("external_teacher")
    episode["sources"][0]["permissions"].append("external_teacher")
    with pytest.raises(ValueError, match="escalates"):
        receipt(release, episode)


def test_audit_packets_require_explicit_evaluation_and_never_enter_train_export(release):
    episode = load_lines(release / "evaluator_only/audit_episodes.jsonl")[0]
    with pytest.raises(ValueError, match="audit model input"):
        input_view(episode)
    with pytest.raises(ValueError, match="audit model input"):
        export_sft(episode, verification=receipt(release, episode))
    assert input_view(episode, purpose="local_evaluation")["memory_texts"]
    audit_groups = {e["group_id"] for e in load_lines(release / "evaluator_only/audit_episodes.jsonl")}
    assert not audit_groups & {r["group_id"] for r in load_lines(release / "exports/train.jsonl")}


def test_source_tamper_is_rejected_by_raw_binding_and_whole_release_validation(release):
    episode = load_lines(release / "canonical/train_episodes.jsonl")[0]
    source = source_for(release, episode)
    path = release / source["path"]
    path.write_bytes(path.read_bytes().replace(b"Register INPUT", b"Register WRONG", 1))
    with pytest.raises(ValueError, match="raw source bytes/hash"):
        receipt(release, episode)
    with pytest.raises(ValueError, match="release bytes mismatch"):
        SLICE.validate_release(release)


def test_source_span_and_event_text_cannot_be_rebound_without_raw_source(release):
    episode = load_lines(release / "canonical/train_episodes.jsonl")[0]
    source = source_for(release, episode)
    source["event_spans"][0]["byte_start"] += 1
    with pytest.raises(ValueError, match="span mismatch"):
        SLICE.independent_check(episode, release, source)
    episode["events"][0]["text"] = "Register INPUT = 0."
    with pytest.raises(ValueError, match="events differ"):
        receipt(release, episode)


def test_rehashed_wrong_flattened_target_still_fails_recompilation(release):
    path = release / "exports/train.jsonl"
    rows = load_lines(path)
    rows[0]["response"] = "incorrect"
    path.write_bytes(b"".join(SLICE.json_bytes(r) for r in rows))
    manifest_path = release / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["files_sha256"]["exports/train.jsonl"] = SLICE.digest(path.read_bytes())
    manifest_path.write_bytes(SLICE.json_bytes(manifest))
    with pytest.raises(ValueError, match="SFT export differs"):
        SLICE.validate_release(release)


def test_seed_changes_raw_worlds_and_overwrite_is_refused(tmp_path):
    first, other = tmp_path / "first", tmp_path / "other"
    SLICE.build_slice(first, seed=23)
    SLICE.build_slice(other, seed=24)
    assert (first / "raw_sources/train-world-0.jsonl").read_bytes() != (other / "raw_sources/train-world-0.jsonl").read_bytes()
    with pytest.raises(FileExistsError):
        SLICE.build_slice(first)
    with pytest.raises(ValueError, match="seed"):
        SLICE.build_slice(tmp_path / "bad", seed=True)

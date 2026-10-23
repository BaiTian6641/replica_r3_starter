"""Deterministic source-bound 2,000-target infrastructure admission; no model/network."""
import copy
import importlib.util
import json
from pathlib import Path
import shutil

import pytest

from replica_lab.episodes import export_sft, input_view, verify_episode
from replica_lab.infrastructure_solver import independent_check

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("make_infrastructure_slice", ROOT / "scripts/make_infrastructure_slice.py")
FACTORY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FACTORY)


def lines(path):
    return [json.loads(line) for line in path.read_bytes().splitlines()]


def source_for(root, episode):
    return next(source for source in json.loads((root / "source_manifest.json").read_bytes())
                if source["source_id"] == episode["sources"][0]["source_id"])


def check(root, episode):
    return independent_check(episode, root, source_for(root, episode))


def receipt(root, episode):
    return verify_episode(episode, verifier_id=FACTORY.VERIFIER_ID,
        verifier_sha256=FACTORY.implementation()["solver_sha256"], verifier=lambda value: check(root, value))


def copied(release, tmp_path):
    destination = tmp_path / "release"
    shutil.copytree(release, destination)
    return destination


def reseal(root, relative, data):
    (root / relative).write_bytes(data)
    path = root / "manifest.json"
    manifest = json.loads(path.read_bytes())
    manifest["files_sha256"][relative] = FACTORY.digest(data)
    path.write_bytes(FACTORY.json_bytes(manifest))


@pytest.fixture(scope="module")
def release(tmp_path_factory):
    root = tmp_path_factory.mktemp("infra") / "release"
    FACTORY.build_release(root)
    return root


def test_exact_preregistered_counts_roles_actions_and_world_independence(release):
    manifest = FACTORY.validate_release(release)
    assert manifest["world_count"] == 1000
    assert manifest["canonical_target_count"] == 2000
    assert manifest["export_counts"] == {"train": 1200, "dev": 200}
    assert manifest["histograms"]["actions"] == {"answer": 1000, "ask": 500, "unknown": 500}
    assert set(manifest["histograms"]["families"].values()) == {500}
    sources = json.loads((release / "source_manifest.json").read_bytes())
    assert len({source["semantic_sha256"] for source in sources}) == 1000
    assert len({source["sha256"] for source in sources}) == 1000
    assert sorted(path.name for path in (release / "exports").iterdir()) == ["dev.jsonl", "train.jsonl"]
    assert manifest["token_admission_status"] == "pending_external_offline_accounting"
    assert "No unseen-mechanism" in manifest["generalization_scope"]
    for role, data in manifest["histograms"]["by_role"].items():
        factor = 6 if role == "train" else 1
        assert data["actions"] == {"answer": 100 * factor, "ask": 50 * factor, "unknown": 50 * factor}


def test_same_seed_regeneration_is_byte_identical(release, tmp_path):
    other = tmp_path / "regenerated"
    first = json.loads((release / "manifest.json").read_bytes())
    assert FACTORY.build_release(other) == first
    for relative in ["manifest.json", *first["files_sha256"]]:
        assert (release / relative).read_bytes() == (other / relative).read_bytes()


def test_assignment_precedes_world_sampling_and_scenario_cycle_is_fixed():
    design = FACTORY._design()
    assignments = FACTORY._assignments(design, 41)
    assert len(assignments) == 1000
    for role in design["roles_order"]:
        for family_index, family in enumerate(design["family_order"]):
            chosen = [a for a in assignments if a["split"] == role and a["family"] == family]
            for index, assignment in enumerate(chosen):
                assert assignment["scenario"] == design["scenarios"][(index + family_index) % 4]
    other = FACTORY._assignments(design, 42)
    assert not {a["world_id"] for a in assignments} & {a["world_id"] for a in other}


def test_every_family_correction_changes_answer_only_after_observation(release):
    episodes = lines(release / "canonical/train_episodes.jsonl")
    for family in FACTORY._design()["family_order"]:
        before = next(e for e in episodes if e["evaluator"]["oracle"]["family"] == family
                      and e["evaluator"]["oracle"]["scenario"] == "late_correction"
                      and e["id"].endswith(".q0"))
        after = next(e for e in episodes if e["parent_episode_id"] == before["id"])
        assert before["target"]["response"] != after["target"]["response"]
        assert check(release, before) and check(release, after)
        assert "90-correct" not in json.dumps(input_view(before))
        assert "90-correct" in json.dumps(input_view(after))
        mutated = copy.deepcopy(before)
        mutated["target"]["response"] = after["target"]["response"]
        assert check(release, mutated) is False


def test_unknown_ask_deletion_and_static_recovery_are_distinct(release):
    episodes = lines(release / "canonical/train_episodes.jsonl")
    for family in FACTORY._design()["family_order"]:
        pair = [e for e in episodes if e["evaluator"]["oracle"]["family"] == family
                and e["evaluator"]["oracle"]["scenario"] == "tombstone"][:2]
        assert [e["target"]["action"] for e in pair] == ["ask", "unknown"]
        assert "ask for it" in pair[0]["query"]["prompt"]
        assert pair[1]["target"]["response"] == "UNKNOWN"
        assert all(check(release, e) for e in pair)
        assert all("99-delete" not in json.dumps(input_view(e)) for e in pair)
    answers = {e["target"]["response"] for e in episodes
               if e["evaluator"]["oracle"]["family"] == "tool_recovery" and e["target"]["action"] == "answer"}
    assert answers == {"VERIFY receipt", "RETRY lookup", "STOP"}


def test_oracle_and_generator_label_path_cannot_supply_verifier_answers(release, monkeypatch):
    episode = copy.deepcopy(lines(release / "canonical/train_episodes.jsonl")[0])
    episode["evaluator"]["oracle"] = {"answer": "UNTRUSTED_GOLD_SENTINEL"}
    monkeypatch.setattr(FACTORY, "_generated_answer", lambda *_: "UNTRUSTED_GENERATOR_SENTINEL")
    assert check(release, episode)
    assert "UNTRUSTED_GOLD_SENTINEL" not in json.dumps(input_view(episode))


def test_audit_and_nontraining_roles_never_gain_training_eligibility(release):
    episode = lines(release / "evaluator_only/audit_episodes.jsonl")[0]
    with pytest.raises(ValueError, match="audit model input"):
        input_view(episode)
    assert input_view(episode, purpose="local_evaluation")["memory_texts"]
    with pytest.raises(ValueError):
        export_sft(episode, verification=receipt(release, episode))
    for role in ["calibration", "promotion"]:
        item = lines(release / f"evaluator_only/{role}_episodes.jsonl")[0]
        with pytest.raises(ValueError):
            export_sft(item, verification=receipt(release, item))


def test_rehashed_wrong_flattened_response_still_fails_independent_recompilation(release, tmp_path):
    root = copied(release, tmp_path)
    rows = lines(root / "exports/train.jsonl")
    rows[0]["response"] = "WRONG"
    reseal(root, "exports/train.jsonl", b"".join(FACTORY.json_bytes(row) for row in rows))
    with pytest.raises(ValueError, match="SFT export differs"):
        FACTORY.validate_release(root)


def test_rehashed_wrong_canonical_label_is_rejected_before_receipt_trust(release, tmp_path):
    root = copied(release, tmp_path)
    rows = lines(root / "canonical/train_episodes.jsonl")
    rows[0]["target"]["response"] = "WRONG"
    reseal(root, "canonical/train_episodes.jsonl", b"".join(FACTORY.json_bytes(row) for row in rows))
    with pytest.raises(ValueError, match="source solver rejected"):
        FACTORY.validate_release(root)


def test_semantic_signature_ignores_names_and_roles_but_retains_chronology(release):
    episode = lines(release / "canonical/train_episodes.jsonl")[0]
    changed = copy.deepcopy(episode["events"])
    mapping = {e["event_id"]: f"renamed-{index}" for index, e in enumerate(changed)}
    for event in changed:
        event["source_id"] = "other-source"
        event["event_id"] = mapping[event["event_id"]]
        if event["replaces_event_id"] is not None:
            event["replaces_event_id"] = mapping[event["replaces_event_id"]]
    assert FACTORY.semantic_signature(episode["events"]) == FACTORY.semantic_signature(changed)
    changed[-1]["observed_at"] += 1
    assert FACTORY.semantic_signature(episode["events"]) != FACTORY.semantic_signature(changed)


def test_rehashed_renamed_duplicate_world_is_rejected_across_roles(release, tmp_path):
    root = copied(release, tmp_path)
    sources = json.loads((root / "source_manifest.json").read_bytes())
    original = next(s for s in sources if s["split"] == "train" and s["family"] == "register_xor8"
                    and s["scenario"] == "late_correction")
    target = next(s for s in sources if s["split"] == "dev" and s["family"] == original["family"]
                  and s["scenario"] == original["scenario"])
    header = lines(root / target["path"])[0]
    events = copy.deepcopy(lines(root / original["path"])[1:])
    mapping = {event["event_id"]: event["event_id"].replace(original["world_id"], target["world_id"])
               for event in events}
    for event in events:
        event["event_id"] = mapping[event["event_id"]]
        event["source_id"] = target["source_id"]
        if event["replaces_event_id"] is not None:
            event["replaces_event_id"] = mapping[event["replaces_event_id"]]
    raw, record = FACTORY._archive_source(header, events)
    assert record["semantic_sha256"] == original["semantic_sha256"]
    sources[sources.index(target)] = record
    reseal(root, target["path"], raw)
    reseal(root, "source_manifest.json", FACTORY.json_bytes(sources))
    with pytest.raises(ValueError, match="semantic duplicate worlds"):
        FACTORY.validate_release(root)


def test_raw_source_tamper_and_extra_release_file_are_rejected(release, tmp_path):
    root = copied(release, tmp_path)
    source = json.loads((root / "source_manifest.json").read_bytes())[0]
    path = root / source["path"]
    path.write_bytes(path.read_bytes().replace(b"RULE ", b"FAKE ", 1))
    with pytest.raises(ValueError, match="release bytes mismatch"):
        FACTORY.validate_release(root)
    path.write_bytes((release / source["path"]).read_bytes())
    (root / "unexpected.txt").write_text("not inventoried")
    with pytest.raises(ValueError, match="inventory differs"):
        FACTORY.validate_release(root)


def test_seed_validation_overwrite_refusal_and_safe_paths(release, tmp_path):
    with pytest.raises(FileExistsError):
        FACTORY.build_release(release)
    for seed in [True, -1, "41"]:
        with pytest.raises(ValueError, match="seed"):
            FACTORY.build_release(tmp_path / "bad", seed=seed)
    for relative in ["../escape", "/tmp/escape"]:
        with pytest.raises(ValueError, match="inside release"):
            FACTORY._safe_file(tmp_path, relative)


@pytest.mark.parametrize("key,value", [("generalization_scope", "unseen mechanisms"),
                                       ("audit_exposure", "sealed benchmark"),
                                       ("token_admission_status", "admitted"),
                                       ("teacher_used", 0)])
def test_manifest_cannot_overstate_scope_or_admission(release, tmp_path, key, value):
    root = copied(release, tmp_path)
    path = root / "manifest.json"
    manifest = json.loads(path.read_bytes())
    manifest[key] = value
    path.write_bytes(FACTORY.json_bytes(manifest))
    with pytest.raises(ValueError, match="declarations differ"):
        FACTORY.validate_release(root)

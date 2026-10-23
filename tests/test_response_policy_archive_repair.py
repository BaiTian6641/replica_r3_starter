"""Narrow tests for validation-only historical archive path resolution."""
import copy
import importlib.util
import json
from pathlib import Path
import shutil

import pytest

REAL_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("response_policy_archive_repair_test",
    REAL_ROOT / "scripts/score_response_policy_archive_repair.py")
REPAIR = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(REPAIR)


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    root = tmp_path / "project"
    root.mkdir()
    monkeypatch.setattr(REPAIR, "ROOT", root)
    monkeypatch.setattr(REPAIR.BASE, "ROOT", root)
    spec = {"schema_version": "replica.response_policy_preflight.v2",
            "preserved_failed_files_sha256": {}}
    entries = []
    for index, (baseline_key, spec_key, sha_key) in enumerate(REPAIR.HISTORICAL):
        original = root / "runs" / f"prior_{index}"
        archive = root / "reports" / f"prior_{index}" / "native_B1_raw"
        original.mkdir(parents=True)
        archive.mkdir(parents=True)
        for name in ("generation_report.json", "generations.jsonl", "profile_summary.json", "profile_samples.jsonl"):
            (original / name).write_text(json.dumps({"artifact": name, "run": index}) + "\n")
        guard = Path(str(original) + "_guard.json")
        host = Path(str(original) + "_guard.host.jsonl")
        child = Path(str(original) + "_guard.child.log")
        host.write_text('{"host": 1}\n')
        child.write_text("completed fixture\n")
        dump(guard, {"output": str(original), "host_samples": str(host), "child_log": str(child)})
        records = []
        for original_file in [*sorted(original.iterdir()), guard, host, child]:
            archived_file = archive / original_file.name
            shutil.copyfile(original_file, archived_file)
            value = REPAIR.digest(original_file)
            records.append({"original_path": str(original_file.relative_to(root)),
                "archive_path": str(archived_file.relative_to(root)), "sha256": value,
                "bytes": original_file.stat().st_size})
            spec["preserved_failed_files_sha256"][str(archived_file.relative_to(root))] = value
        manifest = archive.parent / "native_archive_manifest.json"
        dump(manifest, {"files": records})
        spec["preserved_failed_files_sha256"][str(manifest.relative_to(root))] = REPAIR.digest(manifest)
        prior_spec_path = archive.parent / "evaluation_spec.json"
        prior_spec = {"schema_version": "fixture", "index": index}
        dump(prior_spec_path, prior_spec)
        spec[spec_key], spec[sha_key] = str(prior_spec_path), REPAIR.digest(prior_spec_path)
        spec[baseline_key] = {"generation_report_path": str(archive / "generation_report.json"),
            "generation_report_sha256": REPAIR.digest(archive / "generation_report.json"),
            "generations_path": str(archive / "generations.jsonl"),
            "generations_sha256": REPAIR.digest(archive / "generations.jsonl"),
            "guard_path": str(archive / guard.name), "guard_sha256": REPAIR.digest(guard)}
        entries.append({"original": original, "archive": archive, "guard": archive / guard.name,
                        "manifest": manifest, "prior_spec": prior_spec})
    frozen = root / "scripts/score_response_policy_generations.py"
    frozen.parent.mkdir()
    shutil.copyfile(REAL_ROOT / "scripts/score_response_policy_generations.py", frozen)
    spec["implementation_sha256"] = {"scripts/score_response_policy_generations.py": REPAIR.digest(frozen)}
    spec_path = root / "specification.json"
    dump(spec_path, spec)
    calls = []

    def native_guard(directory, guard_path, call_spec):
        calls.append((directory, guard_path, call_spec))
        return {"exact_replay": True, "path": str(guard_path), "directory": str(directory)}

    monkeypatch.setattr(REPAIR.BASE, "_guard", native_guard)
    return {"root": root, "spec": spec, "spec_path": spec_path,
            "entries": entries, "calls": calls, "guard": native_guard}


def test_two_exact_pinned_archives_replay_all_identical_original_artifacts(bundle):
    original = REPAIR.BASE._guard
    with REPAIR.historical_guard_resolution(bundle["spec"]) as resolutions:
        for entry in bundle["entries"]:
            replay = REPAIR.BASE._guard(entry["archive"], entry["guard"], entry["prior_spec"])
            assert replay["exact_replay"]
        assert all(len(item["files"]) == 7 and len(item["guard_replays"]) == 1 for item in resolutions)
    assert REPAIR.BASE._guard is original
    assert [call[0] for call in bundle["calls"]] == [entry["original"] for entry in bundle["entries"]]


def test_current_v2_guard_arguments_pass_through_unchanged(bundle):
    directory = bundle["root"] / "runs" / "current_v2"
    guard = bundle["root"] / "runs" / "current_v2_guard.json"
    current_spec = {"current": True}
    with REPAIR.historical_guard_resolution(bundle["spec"]):
        REPAIR.BASE._guard(directory, guard, current_spec)
    assert bundle["calls"] == [(directory, guard, current_spec)]
    assert bundle["calls"][0][0] is directory
    assert bundle["calls"][0][1] is guard
    assert bundle["calls"][0][2] is current_spec


@pytest.mark.parametrize("name", ["generation_report.json", "generations.jsonl", "profile_summary.json",
    "profile_samples.jsonl", "prior_0_guard.json", "prior_0_guard.host.jsonl", "prior_0_guard.child.log"])
@pytest.mark.parametrize("side", ["original", "archive"])
def test_any_differing_old_report_profile_guard_or_trace_is_rejected(bundle, name, side):
    entry = bundle["entries"][0]
    path = entry[side] / name
    if side == "original" and name.startswith("prior_0_guard"):
        path = entry["original"].parent / name
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="bytes differ|guard pin differs"):
        with REPAIR.historical_guard_resolution(bundle["spec"]):
            pass
    assert not bundle["calls"]


def test_manifest_itself_must_be_predeclared_and_unchanged(bundle):
    path = bundle["entries"][0]["manifest"]
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="manifest is not predeclared or its bytes changed"):
        with REPAIR.historical_guard_resolution(bundle["spec"]):
            pass


def test_historical_guard_pair_and_original_spec_are_exact(bundle):
    entry = bundle["entries"][0]
    with REPAIR.historical_guard_resolution(bundle["spec"]):
        with pytest.raises(ValueError, match="exact pinned pair"):
            REPAIR.BASE._guard(entry["archive"], entry["guard"].with_name("other_guard.json"), entry["prior_spec"])
        with pytest.raises(ValueError, match="different specification"):
            REPAIR.BASE._guard(entry["archive"], entry["guard"], {"different": True})
    assert not bundle["calls"]


def test_guard_function_restores_even_when_frozen_scorer_fails(bundle, monkeypatch):
    def fail(*args, **kwargs):
        raise ValueError("frozen token check failed")
    monkeypatch.setattr(REPAIR.FROZEN, "score_generation", fail)
    with pytest.raises(ValueError, match="frozen token check failed"):
        REPAIR.score_generation("release", bundle["spec_path"], "report", "generations")
    assert REPAIR.BASE._guard is bundle["guard"]


def test_wrapper_retains_frozen_result_and_adds_only_explicit_validation_provenance(bundle, monkeypatch):
    raw_result = {"schema_version": REPAIR.FROZEN.VERSION, "overall": {"task_correct": {"count": 3}},
                  "examples": [{"untouched": True}]}
    def fake_frozen(*args, **kwargs):
        for entry in bundle["entries"]:
            REPAIR.BASE._guard(entry["archive"], entry["guard"], entry["prior_spec"])
        return copy.deepcopy(raw_result)
    monkeypatch.setattr(REPAIR.FROZEN, "score_generation", fake_frozen)
    result = REPAIR.score_generation("release", bundle["spec_path"], "report", "generations")
    assert {key: result[key] for key in raw_result} == raw_result
    assert set(result) == set(raw_result) | {"validation_repair"}
    repair = result["validation_repair"]
    assert not repair["grading_projection_and_native_settings_changed"]
    assert repair["supplementary_script_sha256"] == REPAIR.digest(REAL_ROOT / "scripts/score_response_policy_archive_repair.py")
    assert len(repair["historical_archives"]) == 2
    assert all(item["guard_replays"][0]["exact_replay"] for item in repair["historical_archives"])


def test_no_historical_guard_replay_cannot_claim_repair_passed(bundle, monkeypatch):
    monkeypatch.setattr(REPAIR.FROZEN, "score_generation", lambda *args, **kwargs: {})
    with pytest.raises(ValueError, match="did not replay both historical"):
        REPAIR.score_generation("release", bundle["spec_path"], "report", "generations")

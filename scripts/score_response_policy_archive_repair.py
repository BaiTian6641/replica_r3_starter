#!/usr/bin/env python3
"""Resolve only pinned historical archive paths, then run the frozen v2 scorer.

The initial scoring invocation stopped before grades because archived guard
metadata names the original run directory. This supplement proves that every
archived native artifact is identical to its retained original and replays the
unchanged guard against that original directory. Current-run validation and
all generation, projection, source, parser and correctness checks stay intact.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "frozen_response_policy_archive_scorer", ROOT / "scripts/score_response_policy_generations.py")
FROZEN = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(FROZEN)
BASE = FROZEN.BASE
VERSION = "replica.response_policy_archive_resolution.v1"
HISTORICAL = (("preserved_baseline", "parent_spec_path", "parent_spec_sha256"),
              ("previous_v1_baseline", "previous_v1_spec_path", "previous_v1_spec_sha256"))
require, digest, read, resolved, safe_file = (
    BASE.require, BASE.digest, BASE.read, BASE.resolved, BASE.safe_file)


def _safe_project_file(path):
    path = Path(path)
    path = path if path.is_absolute() else ROOT / path
    require(path.is_relative_to(ROOT.resolve()), "Archive resolution path escapes project")
    return safe_file(ROOT, path.relative_to(ROOT.resolve()))


def validate_archive(spec, baseline_key, historical_spec_key, historical_spec_sha_key):
    """Require all seven old artifacts in both locations and the frozen manifest."""
    prior = spec[baseline_key]
    archive_report = _safe_project_file(prior["generation_report_path"])
    archive_directory = archive_report.parent
    archive_guard = _safe_project_file(prior["guard_path"])
    require(archive_report.name == "generation_report.json"
            and archive_directory.is_relative_to((ROOT / "reports").resolve())
            and _safe_project_file(prior["generations_path"]) == archive_directory / "generations.jsonl"
            and archive_guard.parent == archive_directory,
            "Historical archive layout differs from the pinned baseline")
    manifest_path = _safe_project_file(archive_directory.parent / "native_archive_manifest.json")
    pins = spec["preserved_failed_files_sha256"]
    manifest_name = str(manifest_path.relative_to(ROOT.resolve()))
    require(manifest_name in pins and digest(manifest_path) == pins[manifest_name],
            "Historical archive manifest is not predeclared or its bytes changed")
    guard_name = str(archive_guard.relative_to(ROOT.resolve()))
    require(pins.get(guard_name) == prior["guard_sha256"] == digest(archive_guard),
            "Historical archive guard pin differs")
    guard = read(archive_guard)
    original_directory = resolved(guard["output"])
    require(original_directory.is_relative_to((ROOT / "runs").resolve())
            and original_directory != archive_directory,
            "Historical guard does not name a retained original run")
    original_guard = Path(str(original_directory) + "_guard.json")
    original_host = resolved(guard["host_samples"])
    original_child = resolved(guard["child_log"])
    require(original_host == Path(str(original_directory) + "_guard.host.jsonl")
            and original_child == Path(str(original_directory) + "_guard.child.log")
            and archive_guard.name == original_guard.name,
            "Historical guard trace layout differs")
    expected_pairs = {
        **{original_directory / name: archive_directory / name for name in (
            "generation_report.json", "generations.jsonl", "profile_summary.json", "profile_samples.jsonl")},
        original_guard: archive_guard,
        original_host: archive_directory / original_host.name,
        original_child: archive_directory / original_child.name}
    records = read(manifest_path)["files"]
    require(isinstance(records, list) and len(records) == len(expected_pairs),
            "Historical archive manifest must enumerate exactly seven artifacts")
    proof, seen_original, seen_archive = [], set(), set()
    for record in records:
        require(isinstance(record, dict) and set(record) == {"original_path", "archive_path", "sha256", "bytes"},
                "Historical archive manifest record differs")
        original = _safe_project_file(record["original_path"])
        archive = _safe_project_file(record["archive_path"])
        require(original in expected_pairs and expected_pairs[original] == archive
                and original not in seen_original and archive not in seen_archive,
                "Historical archive manifest has missing, duplicate or unexpected paths")
        seen_original.add(original)
        seen_archive.add(archive)
        archive_name = str(archive.relative_to(ROOT.resolve()))
        expected = record["sha256"]
        require(pins.get(archive_name) == expected
                and type(record["bytes"]) is int and record["bytes"] > 0
                and original.stat().st_size == archive.stat().st_size == record["bytes"]
                and digest(original) == digest(archive) == expected,
                f"Historical original/archive bytes differ: {archive_name}")
        proof.append({"original_path": str(original), "archive_path": str(archive),
                      "sha256": expected, "bytes": record["bytes"], "byte_identical": True})
    require(digest(archive_report) == prior["generation_report_sha256"]
            and digest(archive_directory / "generations.jsonl") == prior["generations_sha256"],
            "Historical native report/JSONL differs from fixed baseline")
    historical_spec_path = _safe_project_file(spec[historical_spec_key])
    require(digest(historical_spec_path) == spec[historical_spec_sha_key],
            "Historical specification bytes changed")
    return {"baseline_key": baseline_key, "archive_directory": str(archive_directory),
            "original_directory": str(original_directory), "archive_guard_path": str(archive_guard),
            "manifest_path": str(manifest_path), "manifest_sha256": digest(manifest_path),
            "historical_specification_path": str(historical_spec_path),
            "historical_specification_sha256": digest(historical_spec_path),
            "files": proof, "guard_replays": []}


@contextmanager
def historical_guard_resolution(spec):
    """Temporarily redirect the two exact old directory/guard pairs only."""
    resolutions = [validate_archive(spec, *entry) for entry in HISTORICAL]
    whitelist = {resolved(item["archive_directory"]): item for item in resolutions}
    require(len(whitelist) == 2, "Historical archive directories must be distinct")
    original_guard = BASE._guard

    def archive_aware_guard(directory, guard_path, call_spec):
        item = whitelist.get(resolved(directory))
        if item is None:
            return original_guard(directory, guard_path, call_spec)
        require(resolved(guard_path) == resolved(item["archive_guard_path"]),
                "Historical archive guard path differs from its exact pinned pair")
        require(call_spec == read(item["historical_specification_path"]),
                "Historical guard received a different specification")
        replay = original_guard(resolved(item["original_directory"]), guard_path, call_spec)
        require(replay.get("exact_replay") is True, "Historical original guard replay did not pass")
        item["guard_replays"].append(replay)
        return replay

    BASE._guard = archive_aware_guard
    try:
        yield resolutions
    finally:
        BASE._guard = original_guard


def score_generation(release, spec_path, report_path, generations_path, *, guard_path=None, tokenizer=None):
    spec_path = resolved(spec_path)
    spec = read(spec_path)
    require(spec["schema_version"] == "replica.response_policy_preflight.v2",
            "Archive repair is limited to the frozen v2 response-policy specification")
    frozen_name = "scripts/score_response_policy_generations.py"
    require(digest(ROOT / frozen_name) == spec["implementation_sha256"][frozen_name],
            "Frozen v2 scorer bytes changed")
    with historical_guard_resolution(spec) as resolutions:
        result = FROZEN.score_generation(release, spec_path, report_path, generations_path,
                                        guard_path=guard_path, tokenizer=tokenizer)
    require(all(item["guard_replays"] for item in resolutions),
            "Frozen scorer did not replay both historical native guards")
    result["validation_repair"] = {
        "schema_version": VERSION, "status": "exact_historical_archive_resolution_passed",
        "original_validation_failure": "ValueError: Guard output belongs to another run",
        "failure_reason": "Archived guard bytes retain their original runs directory; the frozen scorer passed the archive directory to exact replay.",
        "repair_scope": "Historical path resolution only after all seven original/archive files match the predeclared hashes; current v2 guard validation remains unchanged.",
        "supplementary_script_path": str(Path(__file__).resolve()),
        "supplementary_script_sha256": digest(Path(__file__)),
        "frozen_scorer_sha256": digest(ROOT / frozen_name), "specification_sha256": digest(spec_path),
        "grading_projection_and_native_settings_changed": False,
        "historical_archives": resolutions}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("release", "spec", "generation-report", "generations", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--guard", type=Path)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        parser.error("Use a fresh output file to preserve all earlier evidence")
    result = score_generation(args.release, args.spec, args.generation_report, args.generations,
                              guard_path=args.guard)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "examples": result["overall"]["examples"],
        "task_correct": result["overall"]["task_correct"]["count"],
        "errors": result["error_histogram"], "archive_repair": result["validation_repair"]["status"]}))


if __name__ == "__main__":
    main()

"""Immutable visual preparation manifests reject drift, leakage and duplicates."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import shutil

import pytest

from replica_lab.visual_data import (
    TASKS,
    VisualDataError,
    audit_visual_dataset,
    decode_rgb_png,
    generate_visual_dataset,
    write_exact,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/vision_data_preparation_v1.json"


def small_config():
    config = copy.deepcopy(json.loads(CONFIG.read_text(encoding="utf-8")))
    config["campaign"]["asset_ceiling_bytes"] = 1 << 30
    config["v0"].update(root_families=5, crop_families=2, max_assets=8,
                        cases_per_asset=4,
                        split_family_counts={split: 1 for split in
                                             ("train", "dev", "calibration",
                                              "promotion", "audit")})
    config["v1"].update(worlds=4, dev_worlds=1, max_assets=16, cases_per_asset=2)
    config["temporal"].update(worlds=2, dev_worlds=1)
    return config


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True,
                                     ensure_ascii=False) + "\n", encoding="utf-8")


def read_jsonl(path):
    return [json.loads(line) for line in
            Path(path).read_text(encoding="utf-8").splitlines() if line]


def write_jsonl(path, rows):
    Path(path).write_text("".join(
        json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
        for row in rows), encoding="utf-8")


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    config = small_config()
    output = tmp_path_factory.mktemp("vision") / "data"
    result = generate_visual_dataset(config, root=ROOT, output=output)
    assert result["audit"]["status"] == "passed"
    return config, output


@pytest.fixture
def dataset(prepared, tmp_path):
    config, source = prepared
    target = tmp_path / "data"
    shutil.copytree(source, target)
    return config, target


def audit(config, output):
    return audit_visual_dataset(config, root=ROOT, output=output)


def test_declared_counts_checks_and_inspectable_pngs(prepared):
    config, output = prepared
    manifest_v0 = read_json(output / "v0/vision_manifest.json")
    manifest_v1 = read_json(output / "v1/vision_manifest.json")
    temporal = read_json(output / "temporal/temporal_manifest.json")
    assert manifest_v0["counts"] == {
        "assets": 7, "groups": 5, "model_rows": 28, "verifier_cases": 28,
        "verifier_scenes": 7, "asset_bytes": manifest_v0["counts"]["asset_bytes"],
        "task_histogram": manifest_v0["counts"]["task_histogram"],
        "split_assets": manifest_v0["counts"]["split_assets"],
        "split_groups": manifest_v0["counts"]["split_groups"],
        "split_model_rows": manifest_v0["counts"]["split_model_rows"],
        "root_assets": 5, "crop_assets": 2,
    }
    assert set(manifest_v0["counts"]["split_groups"].values()) == {1}
    assert set(manifest_v0["counts"]["task_histogram"]) == set(TASKS)
    assert manifest_v1["counts"]["assets"] == 16
    assert manifest_v1["counts"]["root_assets"] == 8
    assert manifest_v1["counts"]["crop_assets"] == 8
    assert set(manifest_v1["counts"]["task_histogram"]) == set(TASKS)
    assert temporal["counts"] == {"assets": 6, "groups": 2, "observations": 6,
                                  "queries": 8, "verifier_scenes": 6,
                                  "asset_bytes": temporal["counts"]["asset_bytes"]}
    assert manifest_v0["training_admission"]["admitted_rows"] == 0
    assert manifest_v1["training_admission"]["admitted_rows"] == 0
    for relative in (manifest_v0["assets"][0]["path"],
                     manifest_v1["assets"][0]["path"],
                     temporal["assets"][0]["path"]):
        width, height, rows = decode_rgb_png((output / relative).read_bytes())
        assert (width, height) == (128, 96)
        assert len(rows) == height
    assert audit(config, output)["status"] == "passed"


def test_model_rows_expose_inputs_only_and_verifier_labels_are_separate(prepared):
    _, output = prepared
    rows = read_jsonl(output / "v0/model_rows.jsonl")
    cases = read_jsonl(output / "v0/verifier_cases.jsonl")
    assert all(set(row) == {"case_id", "asset_id", "image", "image_sha256",
                            "question"} for row in rows)
    assert {case["case_id"] for case in cases} == {row["case_id"] for row in rows}
    for case in cases:
        assert case["label_visibility"] == "verifier_only"
        assert case["check"]["expected"] == case["answer"]
    public_temporal = read_json(output / "temporal/temporal_worlds.json")
    text = json.dumps(public_temporal)
    assert "expected_answer" not in text
    assert "expected_status" not in text
    assert "events" not in text
    verifier_temporal = read_json(output / "temporal/verifier_worlds.json")
    statuses = {query["expected_status"]
                for world in verifier_temporal["worlds"] for query in world["queries"]}
    assert statuses == {"changed", "corrected", "revoked", "unchanged"}


def test_generation_is_deterministic_and_reruns_are_idempotent(prepared, tmp_path):
    config, output = prepared
    second = tmp_path / "second"
    generate_visual_dataset(config, root=ROOT, output=second)
    first_files = {path.relative_to(output): path.read_bytes()
                   for path in output.rglob("*")
                   if path.is_file() and not path.name.startswith(".")}
    second_files = {path.relative_to(second): path.read_bytes()
                    for path in second.rglob("*")
                    if path.is_file() and not path.name.startswith(".")}
    assert first_files == second_files
    # Re-running against identical artifacts verifies rather than overwrites.
    generate_visual_dataset(config, root=ROOT, output=second)
    changed = copy.deepcopy(config)
    changed["seed"] += 1
    third = tmp_path / "third"
    generate_visual_dataset(changed, root=ROOT, output=third)
    assert (read_json(third / "v0/vision_manifest.json")["generation_fingerprint"]
            != read_json(second / "v0/vision_manifest.json")["generation_fingerprint"])


def test_changed_asset_bytes_are_rejected(dataset):
    config, output = dataset
    path = output / read_json(output / "v0/vision_manifest.json")["assets"][0]["path"]
    path.write_bytes(path.read_bytes() + b"\x00")
    with pytest.raises(VisualDataError, match="Changed asset"):
        audit(config, output)


def test_changed_verifier_label_is_rejected(dataset):
    config, output = dataset
    path = output / "v0/verifier_cases.jsonl"
    cases = read_jsonl(path)
    cases[0]["answer"] = "do-not-admit"
    write_jsonl(path, cases)
    with pytest.raises(VisualDataError, match="Verifier label differs"):
        audit(config, output)


def test_wrong_split_is_rejected(dataset):
    config, output = dataset
    path = output / "v0/verifier_cases.jsonl"
    cases = read_jsonl(path)
    original = cases[0]["split"]
    cases[0]["split"] = "audit" if original != "audit" else "train"
    write_jsonl(path, cases)
    with pytest.raises(VisualDataError, match="Case crosses family or split"):
        audit(config, output)


def test_sibling_group_leakage_is_rejected(dataset):
    config, output = dataset
    assets = read_json(output / "v0/vision_manifest.json")["assets"]
    groups = sorted({asset["group_id"] for asset in assets})
    path = output / "v0/verifier_cases.jsonl"
    cases = read_jsonl(path)
    sibling = next(group for group in groups if group != cases[0]["group_id"])
    cases[0]["group_id"] = sibling
    write_jsonl(path, cases)
    with pytest.raises(VisualDataError, match="Case crosses family or split"):
        audit(config, output)


def test_duplicate_model_input_is_rejected(dataset):
    config, output = dataset
    rows_path = output / "v0/model_rows.jsonl"
    cases_path = output / "v0/verifier_cases.jsonl"
    rows = read_jsonl(rows_path)
    cases = read_jsonl(cases_path)
    duplicate = dict(rows[0], case_id=rows[0]["case_id"] + "-duplicate")
    duplicate_case = dict(cases[0], case_id=duplicate["case_id"])
    write_jsonl(rows_path, rows + [duplicate])
    write_jsonl(cases_path, cases + [duplicate_case])
    with pytest.raises(VisualDataError, match="Duplicate model inputs"):
        audit(config, output)


def test_label_field_in_model_row_is_rejected(dataset):
    config, output = dataset
    path = output / "v0/model_rows.jsonl"
    rows = read_jsonl(path)
    rows[0]["answer"] = "leaked"
    write_jsonl(path, rows)
    with pytest.raises(VisualDataError, match="Model row fields differ"):
        audit(config, output)


def test_changed_verifier_scene_truth_is_rejected(dataset):
    config, output = dataset
    path = output / "v0/verifier_scenes.jsonl"
    scenes = read_jsonl(path)
    original = scenes[0]["scene"]["objects"][0]["color"]
    scenes[0]["scene"]["objects"][0]["color"] = (
        "purple" if original != "purple" else "red")
    write_jsonl(path, scenes)
    with pytest.raises(VisualDataError, match="Scene truth differs"):
        audit(config, output)


def test_duplicate_asset_record_is_rejected(dataset):
    config, output = dataset
    path = output / "v0/vision_manifest.json"
    manifest = read_json(path)
    manifest["assets"][1] = dict(manifest["assets"][0])
    write_json(path, manifest)
    with pytest.raises(VisualDataError, match="Duplicate input asset"):
        audit(config, output)


def test_existing_artifacts_are_never_silently_overwritten(dataset):
    _, output = dataset
    path = output / "v0/assets"
    image = next(path.glob("*.png"))
    original = image.read_bytes()
    image.write_bytes(b"not-the-same-bytes")
    with pytest.raises(VisualDataError, match="Existing immutable artifact differs"):
        write_exact(image, original)
    assert image.read_bytes() == b"not-the-same-bytes"


def test_external_allowlist_separates_every_stage_without_acquisition(prepared):
    _, output = prepared
    allowlist = read_json(output / "external_component_allowlist.json")
    assert allowlist["counts"] == {
        "catalog_external_sources": 34, "metadata_pinned": 22,
        "awaiting_immutable_pin": 12, "assets_acquired": 0,
        "content_inspected": 0, "normalized": 0, "admitted": 0,
        "eligible_for_this_campaign": 0,
    }
    assert allowlist["acquisition_policy"]["assets_acquired_here"] == 0
    assert allowlist["acquisition_policy"]["bytes_acquired_here"] == 0
    assert allowlist["acquisition_policy"]["internal_generated_bytes_are_distinct"]
    for entry in allowlist["entries"]:
        assert set(entry["stages"]) == {"located", "pinned", "acquired",
                                        "inspected", "normalized", "admitted"}
        assert entry["stages"]["admitted"]["status"] is False
        assert entry["eligible_for_this_campaign"] is False
        assert "component_assets_not_acquired" in entry["blockers"]
    catalog_hash = hashlib.sha256(
        (ROOT / "docs/datasets/20261024_Source_Catalog.json").read_bytes()).hexdigest()
    assert allowlist["catalog"]["sha256"] == catalog_hash


def test_dataset_audits_and_regenerates_after_checkout_relocation(dataset, tmp_path):
    config, output = dataset
    # Reproduce the historical format, with both locators rooted at the old clone.
    path = output / "external_component_allowlist.json"
    allowlist = read_json(path)
    for name in ("catalog", "runtime_registry"):
        allowlist[name]["path"] = str(ROOT / config["external"][name])
    write_json(path, allowlist)
    original = {p.relative_to(output): p.read_bytes() for p in output.rglob("*") if p.is_file()}
    clone = tmp_path / "other-clone"
    for name in ("catalog", "runtime_registry"):
        relative = config["external"][name]
        (clone / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, clone / relative)
    assert audit_visual_dataset(config, root=clone, output=output)["status"] == "passed"
    generate_visual_dataset(config, root=clone, output=output)
    assert original == {p.relative_to(output): p.read_bytes() for p in output.rglob("*") if p.is_file()}
    fresh = tmp_path / "fresh"
    generate_visual_dataset(config, root=clone, output=fresh)
    portable = read_json(fresh / "external_component_allowlist.json")
    assert portable["catalog"]["path"] == config["external"]["catalog"]
    assert portable["runtime_registry"]["path"] == config["external"]["runtime_registry"]


@pytest.mark.parametrize("mutation", ["wrong_path", "different_roots", "wrong_hash", "acquired"])
def test_relocation_does_not_allow_source_or_admission_changes(dataset, mutation):
    config, output = dataset
    path = output / "external_component_allowlist.json"
    allowlist = read_json(path)
    for name in ("catalog", "runtime_registry"):
        allowlist[name]["path"] = str(ROOT / config["external"][name])
    if mutation == "wrong_path":
        allowlist["catalog"]["path"] = str(ROOT / "docs/unrelated.json")
    elif mutation == "different_roots":
        allowlist["catalog"]["path"] = str(Path("/other-clone") / config["external"]["catalog"])
    elif mutation == "wrong_hash":
        allowlist["catalog"]["sha256"] = "0" * 64
    else:
        allowlist["counts"]["assets_acquired"] = 1
    write_json(path, allowlist)
    before = path.read_bytes()
    with pytest.raises(VisualDataError, match="[Aa]llowlist"):
        audit(config, output)
    with pytest.raises(VisualDataError, match="[Aa]llowlist"):
        generate_visual_dataset(config, root=ROOT, output=output)
    assert path.read_bytes() == before


def test_config_rejects_external_acquisition_and_tiny_v1_worlds():
    config = small_config()
    config["external"]["acquire_assets"] = True
    with pytest.raises(VisualDataError, match="never acquires"):
        audit(config, ROOT / "no-such-output")
    config = small_config()
    config["v1"]["worlds"] = 2
    config["v1"]["max_assets"] = 8
    with pytest.raises(VisualDataError, match="cover every V1 task"):
        audit(config, ROOT / "no-such-output")

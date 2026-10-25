"""External caption intake: pinned provenance, split grouping and tamper refusal."""
from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
import struct

import pytest
import zipfile

from replica_lab import external_intake as intake
from replica_lab.external_intake import (
    ExternalIntakeError,
    FileRangeSource,
    RemoteZipIndex,
    assert_row_label_separation,
    inspect_image_bytes,
    leakage_report,
    load_config,
    parse_json_array_prefix,
    prepare_tranche,
    verify_tranche,
)

JPEG_B64 = "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDABQODxIPDRQSEBIXFRQYHjIhHhwcHj0sLiQySUBMS0dARkVQWnNiUFVtVkVGZIhlbXd7gYKBTmCNl4x9lnN+gXz/2wBDARUXFx4aHjshITt8U0ZTfHx8fHx8fHx8fHx8fHx8fHx8fHx8fHx8fHx8fHx8fHx8fHx8fHx8fHx8fHx8fHx8fHz/wAARCAAQABADASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwCnRRRXAe2f/9k="
JPEG = base64.b64decode(JPEG_B64)
IMAGE_CONSTRAINTS = {"formats": ["JPEG", "PNG"], "min_dimension": 4,
                     "max_dimension": 256, "max_bytes": 65536,
                     "require_extension_match": False}


def jpeg_variant(tag: int) -> bytes:
    """Distinct, valid JPEG bytes: a comment segment differs between assets."""
    comment = f"fixture-{tag}".encode()
    return JPEG[:-2] + b"\xff\xfe" + struct.pack(">H", len(comment) + 2) + comment + JPEG[-2:]


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _seed_for_rank(identifiers, target, limit, *, preferred=7):
    if identifiers.index(target) < limit:
        return preferred
    for seed in range(1, 5000):
        ranked = sorted(identifiers, key=lambda value: hashlib.sha256(
            f"{seed}\x00{value}".encode()).hexdigest())
        if ranked.index(target) < limit:
            return seed
    raise AssertionError("could not find a deterministic fixture seed")


def _conversation(prompt: str, answer: str) -> list[dict[str, str]]:
    return [{"from": "human", "value": prompt}, {"from": "gpt", "value": answer}]


def _raw_row(identifier: str, image: str, prompt: str, answer: str) -> dict:
    return {"id": identifier, "image": image,
            "conversations": _conversation(prompt, answer)}


def build_fixture(tmp_path: Path, *, good: int = 9) -> dict:
    root = tmp_path / "fixture"
    root.mkdir(parents=True)
    prompts = ["Describe the image.\n<image>", "<image>\nWhat does this show?",
               "Write a short caption for this picture.\n<image>"]
    annotations: list[dict] = []
    metadata: list[dict] = []
    members: list[tuple[str, bytes]] = []
    for index in range(good):
        identifier = f"{index + 1:09d}"
        image = f"{index + 1:05d}/{identifier}.jpg"
        caption = f"fixture caption {index + 1} on a plain canvas"
        annotations.append(_raw_row(identifier, image, prompts[index % len(prompts)],
                                    caption))
        metadata.append({"id": identifier, "image": image,
                         "blip_caption": caption,
                         "url": f"https://images.example.test/{index + 1}.jpg"})
        members.append((image, jpeg_variant(index)))

    extras: list[dict] = []

    def extra(identifier, image, prompt, answer, **meta):
        annotations.append(_raw_row(identifier, image, prompt, answer))
        if meta.get("write_meta", True):
            metadata.append({"id": identifier, "image": meta.get("meta_image", image),
                             "blip_caption": meta.get("blip", answer),
                             "url": meta.get("url", f"https://images.example.test/{identifier}.jpg")})
        if meta.get("write_member", True):
            members.append((image, meta.get("bytes", jpeg_variant(int(identifier[-3:]) + 100))))
        extras.append(identifier)

    extra("900000001", "90000/900000001.jpg", "Describe the image.\n<image>",
          "duplicate url fixture caption",
          url="https://images.example.test/1.jpg")
    extra("900000002", "90000/900000002.jpg", "Describe the image.\n<image>",
          "fixture caption 2 on a plain canvas")
    extra("900000003", "90000/900000003.jpg", "Describe the image.\n<image>",
          "metadata missing fixture caption", write_meta=False)
    extra("900000004", "90000/900000004.jpg", "Describe the image.\n<image>",
          "zip member missing fixture caption", write_member=False)
    extra("900000005", "90000/900000005.jpg", "Describe the image.\n<image>",
          "invalid image fixture caption", bytes=b"not an image at all")
    extra("900000006", "90000/900000006.jpg", "Describe the image.\n<image>",
          "forbidden family fixture caption",
          url="https://images.cocodataset.org/train2017/000000000001.jpg")
    extra("900000007", "90000/900000007.jpg", "Describe the image.\n<image>", "   ")
    extra("900000008", "90000/900000008.jpg",
          "A red bicycle is shown.\n<image>", "A red bicycle is shown.")
    extra("900000009", "90000/900000009.jpg", "Describe the image.\n<image>",
          "metadata image mismatch caption",
          meta_image="90000/900000099.jpg")
    annotations.append({"id": "900000010", "image": "90000/900000010.jpg",
                        "conversations": [{"from": "human", "value": "Describe."}]})
    metadata.append({"id": "900000010", "image": "90000/900000010.jpg",
                     "blip_caption": "bad conversation caption",
                     "url": "https://images.example.test/bad.jpg"})
    members.append(("90000/900000010.jpg", jpeg_variant(910)))

    all_ids = [row["id"] for row in annotations]
    seed = _seed_for_rank(all_ids, "900000005", 7)
    card = b"fixture card\nlicense: research-only image bundle\n"
    zip_path = root / "images.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("00001/", b"")
        for name, data in members:
            archive.writestr(name, data)
    annotation_path = root / "annotations.json"
    metadata_path = root / "metadata.json"
    card_path = root / "README.md"
    annotation_bytes = json.dumps(annotations, indent=2).encode()
    metadata_bytes = json.dumps(metadata, indent=2).encode()
    annotation_path.write_bytes(annotation_bytes)
    metadata_path.write_bytes(metadata_bytes)
    card_path.write_bytes(card)

    config = {
        "format": intake.CONFIG_FORMAT,
        "intake_id": "fixture_external_caption_tranche",
        "created_on": "2026-10-25",
        "dataset": {
            "id": "fixture/ExternalCaptions",
            "family": "fixture_root_image",
            "source_url": "https://example.test/dataset",
            "revision": "a" * 40,
            "split": "train",
            "shard": "annotations.json#all",
            "card": {"path": "README.md", "size": len(card), "sha256": _sha(card)},
            "files": {
                "annotation": {"path": "annotations.json",
                               "size": len(annotation_bytes),
                               "sha256": _sha(annotation_bytes)},
                "metadata": {"path": "metadata.json",
                             "size": len(metadata_bytes),
                             "sha256": _sha(metadata_bytes)},
                "images": {"path": "images.zip", "size": zip_path.stat().st_size,
                           "sha256": _sha(zip_path.read_bytes())},
            },
        },
        "terms": {
            "license_expression": "fixture-research-only",
            "permitted_use": "internal_research",
            "research_use_only": True,
            "redistribution": "prohibited",
            "commercial_use": False,
            "reviewed_on": "2026-10-25",
            "reviewed_by": "fixture",
            "evidence": [{"kind": "card", "path": "README.md", "sha256": _sha(card)}],
            "components": [
                {"name": "fixture images", "terms": "research-only fixture",
                 "used": "image"},
                {"name": "fixture captions", "terms": "fixture captions",
                 "used": "caption"},
            ],
            "acknowledgements": ["fixture"],
            "reserved_family_avoidance": {
                "reserved": ["coco_root_image"],
                "selected_family": "fixture_root_image",
                "note": "fixture",
            },
        },
        "sampling": {
            "seed": seed,
            "order": "ascending sha256('<seed>:<upstream_id>')",
            "window_rows": 40,
            "window_bytes": 65536,
            "max_window_bytes": 131072,
            "target_roots": 7,
            "split_quotas": {"train": 3, "development": 2,
                             "final_eval_untouched": 2},
        },
        "preprocessing": {
            "version": intake.PREPROCESSING_VERSION,
            "prompt_normalization": "collapse_whitespace_remove_image_token",
            "caption_normalization": "collapse_whitespace_strip_image_token",
            "min_caption_chars": 3,
            "max_caption_chars": 120,
            "image": dict(IMAGE_CONSTRAINTS),
        },
        "admission": {
            "forbidden_url_patterns": ["(?<![a-z])coco(?![a-z])", "cocodataset",
                                       "train2017"],
            "max_central_directory_bytes": 1024 * 1024,
            "max_acquired_bytes": 5 * 1024 * 1024,
            "campaign_disk_ceiling_bytes": 25 * 1024**3,
        },
        "outputs": {
            "data_root": "data/fixture_tranche",
            "report": "reports/fixture_tranche/report.json",
            "raw_dir": "raw",
            "corpus": "corpus.jsonl",
            "rejections": "rejections.jsonl",
            "manifest": "manifest.json",
            "manifests_dir": "manifests",
        },
    }
    config_path = root / "config.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    sources = {
        "card": FileRangeSource(card_path),
        "annotation": FileRangeSource(annotation_path),
        "metadata": FileRangeSource(metadata_path),
        "images_zip": FileRangeSource(zip_path),
    }
    return {"root": root, "config": config, "config_path": config_path,
            "sources": sources, "zip_path": zip_path, "seed": seed,
            "annotations": annotations, "metadata": metadata}


def _prepare(fixture, tmp_path, *, data_root=None, report=None, config=None):
    config_path = fixture["config_path"]
    selected_config, config_sha = load_config(config_path)
    return prepare_tranche(
        config or selected_config, project_root=tmp_path,
        config_sha256=config_sha, config_path=config_path,
        sources=fixture["sources"],
        data_root=data_root or tmp_path / "data" / "tranche",
        report_path=report or tmp_path / "reports" / "intake_report.json")


def _corpus(data_root: Path) -> list[dict]:
    return [json.loads(line) for line in
            (data_root / "corpus.jsonl").read_text(encoding="utf-8").splitlines()
            if line]


def test_provenance_pins_assets_and_derivatives(tmp_path):
    fixture = build_fixture(tmp_path)
    manifest = _prepare(fixture, tmp_path)
    data_root = tmp_path / "data" / "tranche"
    rows = _corpus(data_root)
    assert manifest["status"] == "admitted"
    assert manifest["dataset"]["id"] == "fixture/ExternalCaptions"
    assert manifest["dataset"]["revision"] == "a" * 40
    assert manifest["dataset"]["card"]["sha256"] == _sha(
        (fixture["root"] / "README.md").read_bytes())
    assert manifest["sampling"]["seed"] == fixture["seed"]
    assert manifest["preprocessing"]["version"] == intake.PREPROCESSING_VERSION
    assert manifest["acquisition"]["within_budget"] is True
    assert manifest["acquisition"]["sources"]["images_zip"]["members_verified"] == len(rows)
    for row in rows:
        assert row["format"] == intake.ROW_FORMAT
        assert row["source"]["id"] == "fixture/ExternalCaptions"
        assert row["source"]["revision"] == "a" * 40
        assert row["root_asset"]["upstream_image_path"] == row["root_asset"]["zip_member"]
        kinds = {item["kind"] for item in row["root_asset"]["derivatives"]}
        assert kinds == {"upstream_rehost", "annotation_row", "blip_caption_metadata"}
        image_path = data_root / "raw" / row["observation"]["image_ref"]
        assert _sha(image_path.read_bytes()) == row["root_asset"]["image_sha256"]
        assert row["root_asset"]["image_format"] == "JPEG"
        assert (row["root_asset"]["width"], row["root_asset"]["height"]) == (16, 16)
        assert row["provenance"]["image_inspection"]["structural_verified"] is True
        assert row["terms"]["research_use_only"] is True
    assert manifest["counts"]["admitted_rows"] == 7
    assert manifest["counts"]["admitted_root_assets"] == 7


def test_split_groups_are_complete_and_evaluation_is_untouched(tmp_path):
    fixture = build_fixture(tmp_path)
    _prepare(fixture, tmp_path)
    rows = _corpus(tmp_path / "data" / "tranche")
    assert [row["split"]["role"] for row in rows].count("train") == 3
    assert [row["split"]["role"] for row in rows].count("development") == 2
    assert [row["split"]["role"] for row in rows].count("final_eval_untouched") == 2
    group_roles = {}
    for row in rows:
        group = row["split"]["group_key"]
        assert row["root_asset"]["root_image_id"] == group
        assert group_roles.setdefault(group, row["split"]["role"]) == row["split"]["role"]
    assert len(group_roles) == 7
    eval_rows = [row for row in rows if row["split"]["role"] == "final_eval_untouched"]
    non_eval = [row for row in rows if row["split"]["role"] != "final_eval_untouched"]
    assert not ({row["root_asset"]["image_sha256"] for row in eval_rows}
                & {row["root_asset"]["image_sha256"] for row in non_eval})
    assert not ({row["target"]["caption"] for row in eval_rows}
                & {row["target"]["caption"] for row in non_eval})
    manifests = {}
    for role in intake.ROLES:
        manifests[role] = json.loads(
            (tmp_path / "data" / "tranche" / "manifests" / f"{role}.json")
            .read_text(encoding="utf-8"))
    assert manifests["final_eval_untouched"]["sealed"] is True
    assert manifests["final_eval_untouched"]["untouched"] is True
    assert manifests["train"]["sealed"] is False
    eval_ids = {row["row_id"] for row in manifests["final_eval_untouched"]["rows"]}
    train_ids = {row["row_id"] for row in manifests["train"]["rows"]}
    dev_ids = {row["row_id"] for row in manifests["development"]["rows"]}
    assert not (eval_ids & train_ids) and not (eval_ids & dev_ids)
    assert manifests["final_eval_untouched"]["seal_sha256"] == \
        intake._role_manifest_seal(manifests["final_eval_untouched"])


def test_deterministic_normalization_across_runs(tmp_path):
    fixture = build_fixture(tmp_path)
    first = _prepare(fixture, tmp_path, data_root=tmp_path / "run_a",
                     report=tmp_path / "run_a_report.json")
    second = _prepare(fixture, tmp_path, data_root=tmp_path / "run_b",
                      report=tmp_path / "run_b_report.json")
    for name in ("corpus.jsonl", "rejections.jsonl", "manifest.json"):
        assert (tmp_path / "run_a" / name).read_bytes() == \
            (tmp_path / "run_b" / name).read_bytes()
    for role in intake.ROLES:
        assert (tmp_path / "run_a" / "manifests" / f"{role}.json").read_bytes() == \
            (tmp_path / "run_b" / "manifests" / f"{role}.json").read_bytes()
    assert first["artifacts"]["corpus"]["sha256"] == second["artifacts"]["corpus"]["sha256"]
    # Re-running into the same directory is idempotent and rewrites nothing.
    _prepare(fixture, tmp_path, data_root=tmp_path / "run_a",
             report=tmp_path / "run_a_report.json")
    assert (tmp_path / "run_a" / "corpus.jsonl").read_bytes() == \
        (tmp_path / "run_b" / "corpus.jsonl").read_bytes()


def test_duplicates_and_exclusions_are_recorded_with_reasons(tmp_path):
    fixture = build_fixture(tmp_path)
    _prepare(fixture, tmp_path)
    data_root = tmp_path / "data" / "tranche"
    rejections = [json.loads(line) for line in
                  (data_root / "rejections.jsonl").read_text(encoding="utf-8").splitlines()
                  if line]
    reasons = {item["reason"] for item in rejections}
    assert {"duplicate_url", "duplicate_caption", "metadata_missing",
            "zip_member_missing", "upstream_url_forbidden", "caption_too_short",
            "label_in_prompt", "metadata_image_mismatch",
            "annotation_bad_conversation", "image_unsupported_format"} <= reasons
    for item in rejections:
        assert item["format"] == intake.REJECTION_FORMAT
        assert item["stage"] in {"metadata", "content"}
        assert item["detail"]
        assert item["root_image_id"].startswith("fixture/ExternalCaptions:")
    duplicates = [item for item in rejections if item["reason"] == "duplicate_url"]
    assert len(duplicates) == 1 and "already used by" in duplicates[0]["detail"]


def test_label_and_input_are_separated(tmp_path):
    fixture = build_fixture(tmp_path)
    _prepare(fixture, tmp_path)
    rows = _corpus(tmp_path / "data" / "tranche")
    for row in rows:
        caption = row["target"]["caption"]
        observation = intake.canonical_json(row["observation"])
        assert caption not in observation
        for forbidden in ("caption", "target", "answer", "gold", "label"):
            assert forbidden not in row["observation"]
    tampered = json.loads(json.dumps(rows[0]))
    tampered["observation"]["hint"] = tampered["target"]["caption"]
    with pytest.raises(ExternalIntakeError):
        assert_row_label_separation(tampered)


def test_tampered_corpus_manifest_and_image_are_rejected(tmp_path):
    fixture = build_fixture(tmp_path)
    _prepare(fixture, tmp_path)
    data_root = tmp_path / "data" / "tranche"
    config, config_sha = load_config(fixture["config_path"])
    assert verify_tranche(data_root, config=config,
                          config_sha256=config_sha)["status"] == "verified"

    row = _corpus(data_root)[0]
    image_path = data_root / "raw" / row["observation"]["image_ref"]
    original_image = image_path.read_bytes()
    image_path.write_bytes(original_image + b"tamper")
    with pytest.raises(ExternalIntakeError, match="image digest"):
        verify_tranche(data_root)
    image_path.write_bytes(original_image)

    corpus_path = data_root / "corpus.jsonl"
    original_corpus = corpus_path.read_bytes()
    corpus_path.write_bytes(original_corpus.replace(b"plain canvas", b"plain canvas!"))
    with pytest.raises(ExternalIntakeError, match="corpus digest"):
        verify_tranche(data_root)
    corpus_path.write_bytes(original_corpus)

    manifest_path = data_root / "manifest.json"
    original_manifest = manifest_path.read_bytes()
    manifest = json.loads(original_manifest)
    manifest["artifacts"]["manifests"]["final_eval_untouched"]["seal_sha256"] = "b" * 64
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                             encoding="utf-8")
    with pytest.raises(ExternalIntakeError, match="manifest digest|seal"):
        verify_tranche(data_root)
    manifest_path.write_bytes(original_manifest)
    assert verify_tranche(data_root)["status"] == "verified"


def test_zip_member_crc_tamper_is_rejected(tmp_path):
    zip_path = tmp_path / "stored.zip"
    member = "00001/000000001.jpg"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr(member, jpeg_variant(1))
    source = FileRangeSource(zip_path)
    index = RemoteZipIndex.from_source(source, size=zip_path.stat().st_size,
                                       max_cd_bytes=1024 * 1024)
    entry = index.entries[member]
    header = zip_path.read_bytes()[entry["offset"]:entry["offset"] + 30]
    fields = struct.unpack("<4s5H3L2H", header)
    data_start = entry["offset"] + 30 + fields[9] + fields[10]
    tampered = bytearray(zip_path.read_bytes())
    tampered[data_start] ^= 0xFF
    zip_path.write_bytes(bytes(tampered))
    with pytest.raises(ExternalIntakeError, match="CRC32 mismatch"):
        index.read_member(FileRangeSource(zip_path), member, max_bytes=65536)


def test_card_pin_and_budget_fail_closed(tmp_path):
    fixture = build_fixture(tmp_path)
    config = json.loads(json.dumps(fixture["config"]))
    config["dataset"]["card"]["sha256"] = "c" * 64
    config_path = fixture["root"] / "bad_card_config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ExternalIntakeError, match="card digest"):
        prepare_tranche(config, project_root=tmp_path,
                        config_sha256=_sha(config_path.read_bytes()),
                        config_path=config_path, sources=fixture["sources"],
                        data_root=tmp_path / "bad", report_path=tmp_path / "bad.json")

    fixture = build_fixture(tmp_path / "budget")
    config = json.loads(json.dumps(fixture["config"]))
    config["admission"]["max_acquired_bytes"] = 100
    config_path = fixture["root"] / "budget_config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ExternalIntakeError, match="budget"):
        prepare_tranche(config, project_root=tmp_path,
                        config_sha256=_sha(config_path.read_bytes()),
                        config_path=config_path, sources=fixture["sources"],
                        data_root=tmp_path / "budget_data",
                        report_path=tmp_path / "budget_report.json")


def test_config_gates_fail_closed(tmp_path):
    fixture = build_fixture(tmp_path)
    base = fixture["config"]
    cases = []
    bad = json.loads(json.dumps(base))
    bad["terms"]["permitted_use"] = "commercial"
    cases.append(bad)
    bad = json.loads(json.dumps(base))
    bad["terms"]["research_use_only"] = False
    cases.append(bad)
    bad = json.loads(json.dumps(base))
    bad["dataset"]["family"] = "coco_root_image"
    cases.append(bad)
    bad = json.loads(json.dumps(base))
    bad["dataset"]["revision"] = "main"
    cases.append(bad)
    bad = json.loads(json.dumps(base))
    bad["sampling"]["split_quotas"]["train"] = 4
    cases.append(bad)
    bad = json.loads(json.dumps(base))
    bad["admission"]["forbidden_url_patterns"] = ["("]
    cases.append(bad)
    bad = json.loads(json.dumps(base))
    bad["admission"]["max_acquired_bytes"] = 10**12
    cases.append(bad)
    for config in cases:
        with pytest.raises(ExternalIntakeError):
            intake.validate_config(config)


def test_json_array_prefix_discards_truncated_tail():
    rows = parse_json_array_prefix(b'[{"id":"1"},{"id":"2"},{"id":', 10)
    assert [row["id"] for row in rows] == ["1", "2"]
    assert parse_json_array_prefix(b'[{"id":"1"},{"id":"2"}]', 1) == [{"id": "1"}]
    with pytest.raises(ExternalIntakeError):
        parse_json_array_prefix(b'{"id":"1"}', 10)


def test_image_inspection_rejects_broken_bytes():
    with pytest.raises(intake.AssetRejected, match="unsupported"):
        inspect_image_bytes(b"not an image", IMAGE_CONSTRAINTS)
    with pytest.raises(intake.AssetRejected, match="end-of-image"):
        inspect_image_bytes(jpeg_variant(1)[:-2], IMAGE_CONSTRAINTS)
    info = inspect_image_bytes(jpeg_variant(1), IMAGE_CONSTRAINTS)
    assert info["format"] == "JPEG" and info["structural_verified"] is True
    assert info["sha256"] == _sha(jpeg_variant(1))
    assert info["decode_verified"] in (True, False)


def test_leakage_report_rejects_cross_split_groups():
    row = {
        "row_id": "fixture:000000001#raw_caption",
        "split": {"role": "final_eval_untouched", "group_key": "fixture:000000001"},
        "root_asset": {"root_image_id": "fixture:000000001",
                       "image_sha256": "a" * 64},
        "observation": {"task": "image_captioning", "prompt": "Describe.",
                        "image_ref": "images/1.jpg"},
        "target": {"caption": "a unique fixture caption"},
        "provenance": {},
        "source": {},
    }
    duplicated = json.loads(json.dumps(row))
    duplicated["row_id"] = "fixture:000000002#raw_caption"
    duplicated["split"]["role"] = "train"
    with pytest.raises(ExternalIntakeError, match="crosses splits"):
        leakage_report([row, duplicated])



def test_https_source_identity_is_the_canonical_url():
    url = "https://example.test/dataset/file.json"
    source = intake.HttpRangeSource(url, size=1)
    assert source.identity == url
    assert "https:https" not in source.identity

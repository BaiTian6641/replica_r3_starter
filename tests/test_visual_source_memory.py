"""Exact visual-source admission, restart, correction and revocation contracts."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path
import sqlite3

import pytest

from replica_lab.contracts import canonical_json
from replica_lab.memory_system import EligibilityScope, MemoryRecord
from replica_lab.runtime import visual_source_memory as visual
from replica_lab.visual_data import decode_rgb_png, encode_rgb_png


ROOT = Path(__file__).resolve().parents[1]
SUBJECT = "visual-source-subject"
PURPOSE = "exact_visual_source"
AS_OF = 20


def make_png(seed: int, width: int = 16, height: int = 12) -> bytes:
    rows = [bytes((seed, seed + 1, seed + 2)) * width for _ in range(height)]
    return encode_rgb_png(width, height, rows)


def make_scope(*, subject: str = SUBJECT, branch: str = visual.SOURCE_BRANCH,
               purpose: str = PURPOSE, as_of: int = AS_OF,
               mode: str = "source_current",
               snapshot_at: int | None = None) -> EligibilityScope:
    return EligibilityScope(subject, branch, purpose, as_of, mode, snapshot_at)


def make_identity(data: bytes, *, asset_id: str = "asset-a",
                  family_id: str = "family-a",
                  independent_world_id: str = "world-a",
                  owner_id: str = "owner-a", event_id: str = "event-a",
                  crop_box=None, layout=None) -> visual.VisualSourceIdentity:
    width, height, _rows = decode_rgb_png(data)
    return visual.VisualSourceIdentity(
        source_id=f"source:{family_id}", family_id=family_id,
        independent_world_id=independent_world_id, asset_id=asset_id,
        owner_id=owner_id, event_id=event_id,
        representation=visual.PNG_REPRESENTATION,
        content_sha256=hashlib.sha256(data).hexdigest(), width=width,
        height=height, byte_length=len(data), crop_box=crop_box,
        layout=layout if layout is not None else visual.SourceLayout("original"),
    )


def observe(store: visual.VisualSourceMemory, data: bytes, *,
            observation_id: str = "obs-a", asset_id: str = "asset-a",
            family_id: str = "family-a", independent_world_id: str = "world-a",
            owner_id: str = "owner-a", event_id: str = "event-a",
            observed_at: int = 1, crop_box=None, layout=None):
    identity = make_identity(
        data, asset_id=asset_id, family_id=family_id,
        independent_world_id=independent_world_id, owner_id=owner_id,
        event_id=event_id, crop_box=crop_box, layout=layout)
    receipt = store.observe(
        data, observation_id=observation_id, subject_id=SUBJECT,
        branch_id=visual.SOURCE_BRANCH, purpose=PURPOSE,
        source_id=identity.source_id, family_id=family_id,
        independent_world_id=independent_world_id, asset_id=asset_id,
        owner_id=owner_id, event_id=event_id, observed_at=observed_at,
        event_at=observed_at, producer_id="producer-owned",
        crop_box=crop_box, layout=layout,
        authorized_subject=SUBJECT, authorized_branch=visual.SOURCE_BRANCH,
        source_authorized=True)
    return receipt, identity


def make_request(identity: visual.VisualSourceIdentity, observation_id: str, *,
                 query_id: str = "query-a") -> visual.VisualSourceRequest:
    return visual.VisualSourceRequest(query_id, observation_id, identity)


def test_exact_source_retrieval_is_separate_from_provenance_only(tmp_path):
    with visual.VisualSourceMemory(tmp_path / "ledger.sqlite",
                                   tmp_path / "blobs") as store:
        data = make_png(10)
        layout = visual.SourceLayout("row", (0, 1, 2, 3))
        receipt, identity = observe(store, data, crop_box=(2, 2, 14, 10),
                                    layout=layout)
        request = make_request(identity, "obs-a")
        evidence = store.retrieve(request, make_scope())

        assert evidence.source_bytes == data
        assert evidence.observation.identity == identity
        assert evidence.record_id == "obs-a"
        assert evidence.record_hash == receipt.record_hash
        assert evidence.disposition == visual.SOURCE_DISPOSITION
        assert evidence.source_evidence is True
        assert evidence.factual_recall is False
        assert len(evidence.evidence_digest) == 64
        assert store.verify_evidence(evidence, scope=make_scope(), request=request)
        forged_scopes = (
            replace(make_scope(), subject_id="forged-subject"),
            replace(make_scope(), as_of=AS_OF - 1),
            EligibilityScope(SUBJECT, visual.SOURCE_BRANCH, PURPOSE, AS_OF,
                             "source_snapshot", AS_OF),
        )
        for forged_scope in forged_scopes:
            with pytest.raises(visual.ForeignScopeError):
                store.verify_evidence(replace(evidence, scope=forged_scope),
                                      scope=make_scope(), request=request)

        provenance = store.provenance_only(
            make_scope(), query_id="query-provenance", observation_id="obs-a")
        assert provenance.disposition == visual.PROVENANCE_ONLY_DISPOSITION
        assert provenance.source_evidence is False
        assert provenance.metadata_only is True
        assert provenance.source_bytes_returned == 0
        assert provenance.learned_factual_memory_recall == "not_evaluated"
        assert "not evaluated" in provenance.reason
        assert not hasattr(provenance, "facts")
        assert not hasattr(provenance, "answer")
        assert not hasattr(store, "memory_only_recall")

        tampered = replace(evidence, query_id="foreign-query")
        with pytest.raises(visual.QueryIdentityError):
            store.verify_evidence(tampered, scope=make_scope(), request=request)

        with pytest.raises(visual.SourceUnavailableError):
            store.retrieve(request, make_scope(subject="another-subject"))
        with pytest.raises(visual.ForeignScopeError):
            store.retrieve(request, make_scope(branch="replica"))
        with pytest.raises(visual.ForeignScopeError):
            store.retrieve(request, EligibilityScope(
                SUBJECT, visual.SOURCE_BRANCH, PURPOSE, AS_OF,
                "assistant_for_source", None))
        with pytest.raises(visual.StaleEpochError):
            store.retrieve(request, make_scope(), expected_epoch=evidence.epoch + 1)

        unsupported = make_request(
            replace(identity, representation="raw_bytes.original.v1"), "obs-a")
        with pytest.raises(visual.UnsupportedRepresentationError):
            store.retrieve(unsupported, make_scope())


def test_restart_reuses_original_bytes_without_reencoding(tmp_path, monkeypatch):
    ledger = tmp_path / "ledger.sqlite"
    blobs = tmp_path / "blobs"
    data = make_png(20)
    store = visual.VisualSourceMemory(ledger, blobs)
    receipt, identity = observe(store, data)
    ledger_id, epoch = store.ledger_id, store.epoch
    request = make_request(identity, "obs-a")
    assert store.blob_path(identity.content_sha256).read_bytes() == data
    store.close()

    reopened = visual.VisualSourceMemory(ledger, blobs)
    try:
        def forbidden_decode(*args, **kwargs):
            raise AssertionError("restart retrieval must not decode/re-encode source bytes")

        monkeypatch.setattr(visual, "decode_rgb_png", forbidden_decode)
        evidence = reopened.retrieve(request, make_scope(), expected_epoch=epoch)
        assert evidence.source_bytes == data
        assert evidence.record_hash == receipt.record_hash
        assert evidence.ledger_id == ledger_id
        assert reopened.ledger_id == ledger_id
        assert reopened.epoch == epoch
        assert reopened.blob_statistics()["blob_count"] == 1
        assert reopened.blob_statistics()["blob_bytes"] == len(data)
        assert reopened.verify_evidence(
            evidence, scope=make_scope(), request=request, expected_epoch=epoch)
    finally:
        reopened.close()


def test_correction_changes_only_target_and_unaffected_reference_survives(tmp_path):
    with visual.VisualSourceMemory(tmp_path / "ledger.sqlite",
                                   tmp_path / "blobs") as store:
        first_data, second_data = make_png(30), make_png(40)
        layout = visual.SourceLayout("row", (0, 1))
        first_receipt, first_identity = observe(
            store, first_data, observation_id="obs-a", asset_id="asset-a",
            family_id="family-a", observed_at=1, layout=layout)
        _second_receipt, second_identity = observe(
            store, second_data, observation_id="obs-b", asset_id="asset-b",
            family_id="family-b", observed_at=1, layout=layout)
        first_request = make_request(first_identity, "obs-a")
        second_request = make_request(second_identity, "obs-b")

        epoch_before = store.epoch
        first_evidence = store.retrieve(first_request, make_scope())
        second_evidence = store.retrieve(second_request, make_scope())
        first_reference = first_evidence.reference()
        second_reference = second_evidence.reference()

        corrected_crop = (2, 2, 14, 10)
        correction = store.correct(
            "obs-a", correction_id="obs-a-correction", scope=make_scope(),
            owner_id="owner-corrected", crop_box=corrected_crop,
            layout=visual.SourceLayout("row", (0, 1, 2, 3)),
            event_id="event-a-corrected", observed_at=2, event_at=2,
            authorized_subject=SUBJECT, authorized_branch=visual.SOURCE_BRANCH,
            source_authorized=True)

        assert correction.admitted is True
        assert correction.record_hash != first_receipt.record_hash
        assert correction.content_sha256 == first_identity.content_sha256
        assert correction.epoch == epoch_before + 1
        assert store.epoch == epoch_before + 1

        corrected_identity = replace(
            first_identity, owner_id="owner-corrected",
            crop_box=corrected_crop, event_id="event-a-corrected",
            layout=visual.SourceLayout("row", (0, 1, 2, 3)))
        corrected_request = make_request(corrected_identity, "obs-a-correction")
        corrected = store.retrieve(corrected_request, make_scope())
        assert corrected.source_bytes == first_data
        assert corrected.identity.owner_id == "owner-corrected"
        assert corrected.observation.supersedes == "obs-a"

        old_owner = make_request(
            replace(first_identity, crop_box=corrected_crop,
                    event_id="event-a-corrected",
                    layout=visual.SourceLayout("row", (0, 1, 2, 3))),
            "obs-a-correction")
        with pytest.raises(visual.StaleOwnerError):
            store.retrieve(old_owner, make_scope())

        old_crop = make_request(
            replace(first_identity, owner_id="owner-corrected",
                    event_id="event-a-corrected",
                    layout=visual.SourceLayout("row", (0, 1, 2, 3))),
            "obs-a-correction")
        with pytest.raises(visual.StaleCropError):
            store.retrieve(old_crop, make_scope())

        with pytest.raises(visual.SourceUnavailableError):
            store.retrieve(first_request, make_scope())
        with pytest.raises(visual.StaleEpochError):
            store.retrieve(corrected_request, make_scope(),
                           expected_epoch=epoch_before)
        with pytest.raises(visual.SourceUnavailableError):
            store.verify_reference(first_reference, scope=make_scope())

        unaffected = store.verify_reference(second_reference, scope=make_scope())
        assert unaffected.valid is True
        assert unaffected.epochs_advanced == 1
        assert unaffected.source_unchanged is True
        assert store.retrieve(second_request, make_scope()).source_bytes == second_data

        with pytest.raises(visual.SourceIntegrityError):
            store.correct(
                "obs-b", correction_id="obs-b-correction", scope=make_scope(),
                owner_id="owner-corrected", crop_box=None, layout=layout,
                event_id="event-b-corrected", observed_at=3, event_at=3,
                image_bytes=make_png(99), authorized_subject=SUBJECT,
                authorized_branch=visual.SOURCE_BRANCH, source_authorized=True)
        assert store.epoch == epoch_before + 1


def test_stale_source_crop_owner_layout_and_changed_bytes_are_rejected(tmp_path):
    with visual.VisualSourceMemory(tmp_path / "ledger.sqlite",
                                   tmp_path / "blobs") as store:
        data = make_png(50)
        layout = visual.SourceLayout("grid", (0, 1, 2, 3))
        _receipt, identity = observe(
            store, data, crop_box=(2, 2, 14, 10), layout=layout,
            family_id="family-a", independent_world_id="world-a")
        request = make_request(identity, "obs-a")
        evidence = store.retrieve(request, make_scope())
        reference = evidence.reference()

        stale_cases = [
            (replace(identity, content_sha256="0" * 64), visual.StaleSourceError),
            (replace(identity, width=identity.width + 1), visual.StaleSourceError),
            (replace(identity, height=identity.height + 1), visual.StaleSourceError),
            (replace(identity, asset_id="asset-other"), visual.StaleSourceError),
            (replace(identity, event_id="event-other"), visual.StaleSourceError),
            (replace(identity, family_id="family-other"), visual.StaleSourceError),
            (replace(identity, independent_world_id="world-other"),
             visual.StaleSourceError),
            (replace(identity, owner_id="owner-other"), visual.StaleOwnerError),
            (replace(identity, crop_box=(1, 1, 15, 11)), visual.StaleCropError),
            (replace(identity, layout=visual.SourceLayout("other")),
             visual.LayoutMismatchError),
        ]
        for stale_identity, error in stale_cases:
            with pytest.raises(error):
                store.retrieve(make_request(stale_identity, "obs-a"), make_scope())

        with pytest.raises(visual.ForeignScopeError):
            store.verify_reference(reference, scope=make_scope(purpose="other"))

        store.blob_path(identity.content_sha256).write_bytes(b"mutated source")
        with pytest.raises(visual.SourceIntegrityError):
            store.retrieve(request, make_scope())
        with pytest.raises(visual.SourceIntegrityError):
            store.verify_reference(reference, scope=make_scope())
        assert store.provenance_only(
            make_scope(), query_id="query-a", observation_id="obs-a").record_hash


def test_revocation_invalidates_source_and_correction_descendants(tmp_path):
    with visual.VisualSourceMemory(tmp_path / "ledger.sqlite",
                                   tmp_path / "blobs") as store:
        data = make_png(60)
        layout = visual.SourceLayout("row")
        _receipt, identity = observe(store, data, observed_at=1, layout=layout)
        _other_receipt, other_identity = observe(
            store, make_png(61), observation_id="obs-b", asset_id="asset-b",
            family_id="family-b", observed_at=1, layout=layout)
        store.correct(
            "obs-a", correction_id="obs-a-correction", scope=make_scope(),
            owner_id="owner-corrected", crop_box=(2, 2, 14, 10), layout=layout,
            event_id="event-a-corrected", observed_at=2, event_at=2,
            authorized_subject=SUBJECT, authorized_branch=visual.SOURCE_BRANCH,
            source_authorized=True)
        corrected_identity = replace(
            identity, owner_id="owner-corrected", crop_box=(2, 2, 14, 10),
            event_id="event-a-corrected")
        parent_request = make_request(identity, "obs-a")
        corrected_request = make_request(corrected_identity, "obs-a-correction")
        other_request = make_request(other_identity, "obs-b")
        corrected_reference = store.retrieve(
            corrected_request, make_scope()).reference()
        other_reference = store.retrieve(other_request, make_scope()).reference()
        epoch_before = store.epoch

        revocation = store.revoke(
            "obs-a", scope=make_scope(), authorized_subject=SUBJECT,
            authorized_branch=visual.SOURCE_BRANCH, source_authorized=True)
        assert set(revocation.revoked_observation_ids) == {
            "obs-a", "obs-a-correction"}
        assert revocation.epoch_before == epoch_before
        assert revocation.epoch_after == epoch_before + 1
        assert revocation.retained_blob_sha256 == (identity.content_sha256,)
        assert store.blob_path(identity.content_sha256).exists()

        for request in (parent_request, corrected_request):
            with pytest.raises(visual.SourceUnavailableError):
                store.retrieve(request, make_scope())
            with pytest.raises(visual.SourceUnavailableError):
                store.provenance_only(
                    make_scope(), query_id="query-a",
                    observation_id=request.observation_id)
        with pytest.raises(visual.SourceUnavailableError):
            store.verify_reference(corrected_reference, scope=make_scope())

        unaffected = store.verify_reference(other_reference, scope=make_scope())
        assert unaffected.valid is True
        assert unaffected.epochs_advanced == 1
        assert store.retrieve(other_request, make_scope()).source_bytes is not None
        remaining = store.inventory(make_scope()).observation_ids
        assert "obs-a" not in remaining
        assert "obs-a-correction" not in remaining
        assert "obs-b" in remaining

        with pytest.raises(ValueError):
            observe(store, data, observation_id="obs-a")
        with pytest.raises(visual.SourceUnavailableError):
            store.revoke("obs-a", scope=make_scope(), authorized_subject=SUBJECT,
                         authorized_branch=visual.SOURCE_BRANCH,
                         source_authorized=True)


def test_payload_whitelist_excludes_captions_labels_and_predictions(tmp_path):
    with visual.VisualSourceMemory(tmp_path / "ledger.sqlite",
                                   tmp_path / "blobs") as store:
        data = make_png(70)
        receipt, identity = observe(store, data)
        with sqlite3.connect(tmp_path / "ledger.sqlite") as database:
            rows = database.execute("SELECT payload FROM memory_records").fetchall()
        payload_text = json.loads(rows[0][0])["text"]
        payload = json.loads(payload_text)
        assert receipt.record_id == payload["observation_id"]
        assert set(payload) == visual.PAYLOAD_FIELDS
        assert not {"caption", "expected_answer", "question", "scene", "ocr",
                    "prediction", "label"} & set(payload)
        assert "caption" not in payload_text.casefold()

        with pytest.raises(TypeError):
            store.observe(
                data, observation_id="obs-label", subject_id=SUBJECT,
                branch_id=visual.SOURCE_BRANCH, purpose=PURPOSE,
                source_id="source:family-a", family_id="family-a",
                independent_world_id="world-a", asset_id="asset-label",
                owner_id="owner-a", event_id="event-label", observed_at=1,
                producer_id="producer-owned", caption="the circle is purple",
                authorized_subject=SUBJECT,
                authorized_branch=visual.SOURCE_BRANCH, source_authorized=True)
        with pytest.raises(visual.UnsupportedRepresentationError):
            store.observe(
                data, observation_id="obs-repr", subject_id=SUBJECT,
                branch_id=visual.SOURCE_BRANCH, purpose=PURPOSE,
                source_id="source:family-a", family_id="family-a",
                independent_world_id="world-a", asset_id="asset-repr",
                owner_id="owner-a", event_id="event-repr", observed_at=1,
                producer_id="producer-owned", representation="jpeg.v1",
                authorized_subject=SUBJECT,
                authorized_branch=visual.SOURCE_BRANCH, source_authorized=True)
        with pytest.raises(visual.SourceIntegrityError):
            store.observe(
                b"not a png", observation_id="obs-bad", subject_id=SUBJECT,
                branch_id=visual.SOURCE_BRANCH, purpose=PURPOSE,
                source_id="source:family-a", family_id="family-a",
                independent_world_id="world-a", asset_id="asset-bad",
                owner_id="owner-a", event_id="event-bad", observed_at=1,
                producer_id="producer-owned", authorized_subject=SUBJECT,
                authorized_branch=visual.SOURCE_BRANCH, source_authorized=True)
        with pytest.raises(ValueError):
            store.observe(
                data, observation_id="obs-crop", subject_id=SUBJECT,
                branch_id=visual.SOURCE_BRANCH, purpose=PURPOSE,
                source_id="source:family-a", family_id="family-a",
                independent_world_id="world-a", asset_id="asset-crop",
                owner_id="owner-a", event_id="event-crop", observed_at=1,
                producer_id="producer-owned", crop_box=(0, 0, 200, 200),
                authorized_subject=SUBJECT,
                authorized_branch=visual.SOURCE_BRANCH, source_authorized=True)

        observation = visual.VisualSourceObservation(
            identity=identity, observation_id="obs-evil", subject_id=SUBJECT,
            branch_id=visual.SOURCE_BRANCH, purpose=PURPOSE,
            producer_id="producer-owned", observed_at=10, event_at=10)
        poisoned = observation.as_dict()
        poisoned["caption"] = "the circle is purple"
        store.ledger.admit(
            MemoryRecord(
                record_id="obs-evil", subject_id=SUBJECT,
                branch_id=visual.SOURCE_BRANCH, purpose=PURPOSE,
                text=canonical_json(poisoned), observed_at=10, event_at=10,
                kind=visual.SOURCE_KIND, source_id=identity.source_id,
                authenticity=visual.SOURCE_AUTHENTICITY),
            authorized_subject=SUBJECT,
            authorized_branch=visual.SOURCE_BRANCH, source_authorized=True)
        with pytest.raises(visual.SourceIntegrityError):
            store.retrieve(make_request(identity, "obs-evil"), make_scope())


def test_inventory_counts_source_families_and_independent_worlds_separately(tmp_path):
    with visual.VisualSourceMemory(tmp_path / "ledger.sqlite",
                                   tmp_path / "blobs") as store:
        observe(store, make_png(80), observation_id="obs-1", asset_id="asset-1",
                family_id="family-a", independent_world_id="world-1",
                event_id="event-1")
        observe(store, make_png(81), observation_id="obs-2", asset_id="asset-2",
                family_id="family-b", independent_world_id="world-1",
                event_id="event-2")
        observe(store, make_png(82), observation_id="obs-3", asset_id="asset-3",
                family_id="family-c", independent_world_id="world-2",
                event_id="event-3")

        inventory = store.inventory(make_scope())
        assert inventory.observation_ids == ("obs-1", "obs-2", "obs-3")
        assert inventory.source_families == ("family-a", "family-b", "family-c")
        assert inventory.independent_worlds == ("world-1", "world-2")
        assert inventory.source_family_count == 3
        assert inventory.independent_world_count == 2

        summary = inventory.summary()
        assert summary["source_family_count"] == 3
        assert summary["independent_world_count"] == 2
        assert summary["source_family_observation_counts"] == {
            "family-a": 1, "family-b": 1, "family-c": 1}
        assert summary["independent_world_observation_counts"] == {
            "world-1": 2, "world-2": 1}

        store.correct(
            "obs-1", correction_id="obs-1-correction", scope=make_scope(),
            owner_id="owner-corrected", crop_box=None,
            layout=visual.SourceLayout("original"), event_id="event-1-corrected",
            observed_at=2, event_at=2, authorized_subject=SUBJECT,
            authorized_branch=visual.SOURCE_BRANCH, source_authorized=True)
        corrected_inventory = store.inventory(make_scope())
        assert "obs-1" not in corrected_inventory.observation_ids
        assert "obs-1-correction" in corrected_inventory.observation_ids
        assert corrected_inventory.observation_ids == (
            "obs-2", "obs-3", "obs-1-correction")


def test_corpus_asset_resolution_rejects_absolute_traversal_and_symlink_escapes(tmp_path):
    corpus = tmp_path / "corpus"
    assets = corpus / "temporal/assets"
    assets.mkdir(parents=True)
    inside = assets / "inside.png"
    inside.write_bytes(make_png(93))
    assert visual.resolve_corpus_asset(
        corpus, "temporal/assets/inside.png") == inside.resolve()

    outside = tmp_path / "outside.png"
    outside.write_bytes(make_png(94))
    (assets / "inside.txt").write_text("not an image", encoding="utf-8")
    (assets / "escape.png").symlink_to(outside)
    for unsafe in (str(outside), str(inside), "../outside.png",
                   "temporal/assets/inside.txt", "temporal/assets/escape.png"):
        with pytest.raises(visual.SourceIntegrityError):
            visual.resolve_corpus_asset(corpus, unsafe)


def _load_cli():
    spec = importlib.util.spec_from_file_location(
        "exact_visual_memory_cli_test",
        ROOT / "scripts/run_exact_visual_memory.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_cli_manifest(corpus: Path, *, relative_path: str, data: bytes,
                        source_family: str = "owned-renderer-test") -> None:
    manifest = {
        "format": "replica.temporal_vision_manifest.v1",
        "assets": [{
            "asset_id": "world-1-t1", "group_id": "world-1", "family": "world-1",
            "path": relative_path, "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(), "width": 16, "height": 12,
            "lineage": {"variant": "t1", "crop_box": None,
                        "layout_template": "row_4"},
        }],
        "config_fingerprint": "0" * 64,
        "generation_fingerprint": "1" * 64,
        "provenance": {"source_id": source_family},
        "stage": "temporal",
    }
    (corpus / "temporal").mkdir(parents=True, exist_ok=True)
    (corpus / "temporal/temporal_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8")


def test_cli_loader_counts_one_owned_family_separately_from_world_groups(tmp_path):
    corpus = tmp_path / "corpus"
    (corpus / "temporal/assets").mkdir(parents=True)
    data = make_png(95)
    (corpus / "temporal/assets/world-1-t1.png").write_bytes(data)
    _write_cli_manifest(corpus, relative_path="temporal/assets/world-1-t1.png",
                        data=data)
    corpus_info, assets, counts = _load_cli()._load_corpus(corpus, 1)
    assert len(assets) == 1
    assert corpus_info["owned_source_family"] == "owned-renderer-test"
    assert corpus_info["external_assets"] == 0
    assert counts["owned_source_families"] == ["owned-renderer-test"]
    assert counts["owned_source_family_count"] == 1
    assert counts["world_derivation_groups"] == ["world-1"]
    assert counts["world_derivation_group_count"] == 1
    assert counts["manifest_world_family_ids"] == ["world-1"]
    assert counts["manifest_world_family_count"] == 1
    assert counts["independent_world_count"] == 1


def test_cli_loader_rejects_external_manifest_png_before_any_read(tmp_path):
    corpus = tmp_path / "corpus"
    outside = tmp_path / "external.png"
    outside.write_bytes(make_png(96))
    _write_cli_manifest(corpus, relative_path=str(outside),
                        data=outside.read_bytes())
    with pytest.raises(visual.SourceIntegrityError):
        _load_cli()._load_corpus(corpus, 1)

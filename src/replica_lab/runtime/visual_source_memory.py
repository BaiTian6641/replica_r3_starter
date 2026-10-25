"""Exact authenticated visual-source memory; an engineering baseline, not perception.

The durable authority is the existing :class:`RecordLedger`: admissions,
corrections, revocations, eligibility and the conservative epoch all live there.
Original image bytes are retained in a content-addressed blob directory keyed by
their SHA-256. That directory is evidence payload storage, not a second
authoritative index: every retrieval must first resolve a live, eligible ledger
record and then authenticate the blob against the record's immutable hash.

Only observed source identity and lineage are admitted: source/family/world/
asset identifiers, exact byte hash, representation, crop, layout, producer,
owner, event and observation times, plus correction and revocation bindings.
Captions, evaluator labels, scene answers, color answers and OCR predictions are
not fields of this module and are rejected by the payload whitelist. This code
does not load a model, train, encode images, recognize scenes or replace neural
or factual memory.

Consumers must supply their own :class:`EligibilityScope`,
:class:`VisualSourceRequest` and query identifier. Retrieval returns original
source bytes and permits exact source introspection; ``provenance_only`` is a
separate metadata-only disposition. Learned factual memory recall is not
implemented or evaluated by this baseline.

Corrections here are source-lineage metadata corrections only: they may update
owner, crop and layout, but they never assert a corrected visual fact.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import uuid

from ..contracts import canonical_json, fingerprint
from ..memory_system import EligibilityScope, MemoryRecord, RecordLedger
from ..vision_qualification import QualificationError, resolve_relative
from ..visual_data import VisualDataError, decode_rgb_png


PNG_REPRESENTATION = "png_rgb8.original.v1"
SUPPORTED_REPRESENTATIONS = frozenset({PNG_REPRESENTATION})
PAYLOAD_SCHEMA = "replica.visual_source_memory.record.v1"
PAYLOAD_FIELDS = frozenset({
    "schema", "observation_id", "subject_id", "branch_id", "purpose",
    "source_id", "family_id", "independent_world_id", "asset_id", "owner_id",
    "producer_id", "event_id", "representation", "content_sha256", "width",
    "height", "byte_length", "crop_box", "layout", "observed_at", "event_at",
    "supersedes", "kind", "authenticity",
})
SOURCE_KIND = "visual_source"
CORRECTION_KIND = "visual_source_correction"
PAYLOAD_KINDS = frozenset({SOURCE_KIND, CORRECTION_KIND})
SOURCE_BRANCH = "source"
SOURCE_MODES = frozenset({"source_current", "source_snapshot"})
SOURCE_AUTHENTICITY = "observed_tool"
SOURCE_DISPOSITION = "source_evidence"
PROVENANCE_ONLY_DISPOSITION = "provenance_only"
PROVENANCE_ONLY_REASON = (
    "metadata-only provenance; learned factual memory recall is not evaluated "
    "or qualified and no source bytes are returned"
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


def _identifier(name: str, value: object) -> None:
    if (not isinstance(value, str) or not value.strip() or "\x00" in value):
        raise ValueError(f"{name} must be a nonempty string without NUL")


def _timestamp(name: str, value: object) -> None:
    if type(value) is not int or value < 0 or value > 2**63 - 1:
        raise ValueError(f"{name} must be a nonnegative SQLite integer timestamp")


def _positive_int(name: str, value: object) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _sha256_field(name: str, value: object) -> None:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")


def _bytes_value(name: str, value: object) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, (bytearray, memoryview)):
        return bytes(value)
    raise TypeError(f"{name} must be bytes-like")


def _normalized_crop(value: object, width: int, height: int
                     ) -> tuple[int, int, int, int] | None:
    if value is None:
        return None
    if not isinstance(value, (tuple, list)) or len(value) != 4:
        raise ValueError("crop_box must be null or four pixel coordinates")
    if any(type(item) is not int for item in value):
        raise ValueError("crop_box coordinates must be integers")
    left, top, right, bottom = value
    if not 0 <= left < right <= width or not 0 <= top < bottom <= height:
        raise ValueError("crop_box lies outside the source image")
    return (left, top, right, bottom)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def resolve_corpus_asset(corpus_root: str | Path, relative: str | Path) -> Path:
    """Resolve an owned-corpus asset under the shared safe-relative contract.

    Absolute paths, ``..`` traversal and symlinks that resolve outside the
    corpus root are rejected before any read. The caller must still verify the
    asset's declared hash and may only admit PNG source evidence.
    """
    try:
        path = resolve_relative(corpus_root, relative)
    except QualificationError as error:
        raise SourceIntegrityError(
            f"manifest asset path escapes the corpus root: {relative}") from error
    if path.suffix.casefold() != ".png":
        raise SourceIntegrityError("manifest assets must be exact .png files")
    return path


class VisualSourceError(Exception):
    """Base marker for exact visual-source contract failures."""


class ContractError(ValueError, VisualSourceError):
    """The supplied source or consumer contract is inconsistent."""


class ForeignScopeError(PermissionError, VisualSourceError):
    """The caller or record belongs to a different authenticated scope."""


class LedgerMismatchError(ForeignScopeError):
    """The reference was minted by a different ledger identity."""


class StaleEpochError(ContractError):
    """The consumer expected an epoch that is no longer current."""


class StaleSourceError(ContractError):
    """Source identity, bytes, representation or event lineage has changed."""


class StaleOwnerError(ContractError):
    """The source owner differs from the independently derived expectation."""


class StaleCropError(ContractError):
    """The source crop differs from the independently derived expectation."""


class LayoutMismatchError(ContractError):
    """The source layout differs from the independently derived expectation."""


class UnsupportedRepresentationError(ContractError):
    """The representation is outside this exact-source baseline."""


class SourceUnavailableError(PermissionError, VisualSourceError):
    """The source is absent, ineligible, revoked or superseded."""


class SourceIntegrityError(ContractError):
    """The payload, image bytes or immutable source binding no longer matches."""


class QueryIdentityError(ContractError):
    """The evidence was produced for a different consumer query identity."""


@dataclass(frozen=True, slots=True)
class SourceLayout:
    """Exact layout metadata observed with a source; no scene interpretation."""

    template: str
    cell_indices: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        _identifier("layout template", self.template)
        if not isinstance(self.cell_indices, tuple):
            raise ValueError("layout cell_indices must be a tuple")
        for index in self.cell_indices:
            if type(index) is not int or index < 0:
                raise ValueError("layout cell indices must be nonnegative integers")
        if len(set(self.cell_indices)) != len(self.cell_indices):
            raise ValueError("layout cell indices must be unique")

    def as_dict(self) -> dict[str, object]:
        return {"template": self.template, "cell_indices": list(self.cell_indices)}

    @property
    def layout_hash(self) -> str:
        return fingerprint(self.as_dict())

    @classmethod
    def from_dict(cls, value: object) -> SourceLayout:
        if not isinstance(value, dict) or set(value) != {"template", "cell_indices"}:
            raise SourceIntegrityError("layout payload has an unsupported shape")
        indices = value["cell_indices"]
        if not isinstance(indices, list):
            raise SourceIntegrityError("layout cell_indices must be a list")
        return cls(value["template"], tuple(indices))


@dataclass(frozen=True, slots=True)
class VisualSourceIdentity:
    """Independently observable identity of one source observation."""

    source_id: str
    family_id: str
    independent_world_id: str
    asset_id: str
    owner_id: str
    event_id: str
    representation: str
    content_sha256: str
    width: int
    height: int
    byte_length: int
    crop_box: tuple[int, int, int, int] | None
    layout: SourceLayout

    def __post_init__(self) -> None:
        for name in ("source_id", "family_id", "independent_world_id", "asset_id",
                     "owner_id", "event_id", "representation"):
            _identifier(name, getattr(self, name))
        _sha256_field("content_sha256", self.content_sha256)
        _positive_int("width", self.width)
        _positive_int("height", self.height)
        _positive_int("byte_length", self.byte_length)
        if not isinstance(self.layout, SourceLayout):
            raise ValueError("layout must be a SourceLayout")
        object.__setattr__(
            self, "crop_box", _normalized_crop(self.crop_box, self.width, self.height))

    def as_dict(self) -> dict[str, object]:
        return {
            "source_id": self.source_id,
            "family_id": self.family_id,
            "independent_world_id": self.independent_world_id,
            "asset_id": self.asset_id,
            "owner_id": self.owner_id,
            "event_id": self.event_id,
            "representation": self.representation,
            "content_sha256": self.content_sha256,
            "width": self.width,
            "height": self.height,
            "byte_length": self.byte_length,
            "crop_box": list(self.crop_box) if self.crop_box is not None else None,
            "layout": self.layout.as_dict(),
        }

    @classmethod
    def from_dict(cls, value: object) -> VisualSourceIdentity:
        if not isinstance(value, dict) or set(value) != set(cls.__dataclass_fields__):
            raise SourceIntegrityError("source identity payload has unsupported fields")
        crop = value["crop_box"]
        if crop is not None and not isinstance(crop, list):
            raise SourceIntegrityError("source crop must be a list or null")
        return cls(
            source_id=value["source_id"],
            family_id=value["family_id"],
            independent_world_id=value["independent_world_id"],
            asset_id=value["asset_id"],
            owner_id=value["owner_id"],
            event_id=value["event_id"],
            representation=value["representation"],
            content_sha256=value["content_sha256"],
            width=value["width"],
            height=value["height"],
            byte_length=value["byte_length"],
            crop_box=tuple(crop) if crop is not None else None,
            layout=SourceLayout.from_dict(value["layout"]),
        )


@dataclass(frozen=True, slots=True)
class VisualSourceObservation:
    """One immutable observation admitted to the shared ledger."""

    identity: VisualSourceIdentity
    observation_id: str
    subject_id: str
    branch_id: str
    purpose: str
    producer_id: str
    observed_at: int
    event_at: int
    supersedes: str | None = None
    kind: str = SOURCE_KIND
    authenticity: str = SOURCE_AUTHENTICITY

    def __post_init__(self) -> None:
        if not isinstance(self.identity, VisualSourceIdentity):
            raise ValueError("observation identity must be a VisualSourceIdentity")
        for name in ("observation_id", "subject_id", "branch_id", "purpose", "producer_id"):
            _identifier(name, getattr(self, name))
        if self.branch_id != SOURCE_BRANCH:
            raise ValueError(f"visual source records must live in the {SOURCE_BRANCH} branch")
        for name in ("observed_at", "event_at"):
            _timestamp(name, getattr(self, name))
        if self.supersedes is not None:
            _identifier("supersedes", self.supersedes)
            if self.supersedes == self.observation_id:
                raise ValueError("a source observation cannot supersede itself")
        if self.kind not in PAYLOAD_KINDS:
            raise ValueError("unsupported visual source record kind")
        if self.authenticity != SOURCE_AUTHENTICITY:
            raise ValueError("visual source observations require observed_tool authenticity")
        if self.kind == CORRECTION_KIND and self.supersedes is None:
            raise ValueError("a correction must name the observation it supersedes")
        if self.kind == SOURCE_KIND and self.supersedes is not None:
            raise ValueError("use the correction path to supersede a source")

    def as_dict(self) -> dict[str, object]:
        payload = {
            "schema": PAYLOAD_SCHEMA,
            "observation_id": self.observation_id,
            "subject_id": self.subject_id,
            "branch_id": self.branch_id,
            "purpose": self.purpose,
            "producer_id": self.producer_id,
            "observed_at": self.observed_at,
            "event_at": self.event_at,
            "supersedes": self.supersedes,
            "kind": self.kind,
            "authenticity": self.authenticity,
        }
        payload.update(self.identity.as_dict())
        return payload


@dataclass(frozen=True, slots=True)
class VisualSourceRequest:
    """A consumer's independently derived exact-source request."""

    query_id: str
    observation_id: str
    identity: VisualSourceIdentity

    def __post_init__(self) -> None:
        _identifier("query_id", self.query_id)
        _identifier("observation_id", self.observation_id)
        if not isinstance(self.identity, VisualSourceIdentity):
            raise ValueError("request identity must be a VisualSourceIdentity")


@dataclass(frozen=True, slots=True)
class VisualSourceEvidence:
    """Original source bytes plus authenticated exact-source lineage."""

    query_id: str
    scope: EligibilityScope
    ledger_id: str
    epoch: int
    record_id: str
    record_hash: str
    observation: VisualSourceObservation
    source_bytes: bytes
    evidence_digest: str
    disposition: str = SOURCE_DISPOSITION
    source_evidence: bool = True
    factual_recall: bool = False

    @property
    def observation_id(self) -> str:
        return self.observation.observation_id

    @property
    def identity(self) -> VisualSourceIdentity:
        return self.observation.identity

    def summary(self) -> dict[str, object]:
        return {
            "disposition": self.disposition,
            "source_evidence": self.source_evidence,
            "factual_recall": self.factual_recall,
            "query_id": self.query_id,
            "ledger_id": self.ledger_id,
            "epoch": self.epoch,
            "observation_id": self.observation_id,
            "record_hash": self.record_hash,
            "evidence_digest": self.evidence_digest,
            "source_sha256": _sha256_bytes(self.source_bytes),
            "source_bytes_returned": len(self.source_bytes),
            "identity": self.identity.as_dict(),
            "observed_at": self.observation.observed_at,
            "event_at": self.observation.event_at,
            "supersedes": self.observation.supersedes,
            "kind": self.observation.kind,
        }

    def reference(self) -> SourceReference:
        return SourceReference(
            ledger_id=self.ledger_id,
            epoch=self.epoch,
            scope=self.scope,
            query_id=self.query_id,
            observation_id=self.observation_id,
            record_hash=self.record_hash,
            content_sha256=self.identity.content_sha256,
            representation=self.identity.representation,
            crop_box=self.identity.crop_box,
            layout=self.identity.layout,
            owner_id=self.identity.owner_id,
            evidence_digest=self.evidence_digest,
        )


@dataclass(frozen=True, slots=True)
class SourceProvenance:
    """Metadata-only source provenance; never source evidence or factual recall."""

    query_id: str
    ledger_id: str
    epoch: int
    observation_id: str
    record_hash: str
    identity: VisualSourceIdentity
    disposition: str = PROVENANCE_ONLY_DISPOSITION
    source_evidence: bool = False
    metadata_only: bool = True
    source_bytes_returned: int = 0
    learned_factual_memory_recall: str = "not_evaluated"
    reason: str = PROVENANCE_ONLY_REASON

    def summary(self) -> dict[str, object]:
        return {
            "disposition": self.disposition,
            "source_evidence": self.source_evidence,
            "metadata_only": self.metadata_only,
            "source_bytes_returned": self.source_bytes_returned,
            "learned_factual_memory_recall": self.learned_factual_memory_recall,
            "query_id": self.query_id,
            "ledger_id": self.ledger_id,
            "epoch": self.epoch,
            "observation_id": self.observation_id,
            "record_hash": self.record_hash,
            "identity": self.identity.as_dict(),
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class SourceReference:
    """A checked reference to one live exact source; not a cached payload."""

    ledger_id: str
    epoch: int
    scope: EligibilityScope
    query_id: str
    observation_id: str
    record_hash: str
    content_sha256: str
    representation: str
    crop_box: tuple[int, int, int, int] | None
    layout: SourceLayout
    owner_id: str
    evidence_digest: str

    def summary(self) -> dict[str, object]:
        return {
            "ledger_id": self.ledger_id,
            "epoch": self.epoch,
            "query_id": self.query_id,
            "observation_id": self.observation_id,
            "record_hash": self.record_hash,
            "content_sha256": self.content_sha256,
            "representation": self.representation,
            "crop_box": list(self.crop_box) if self.crop_box is not None else None,
            "layout": self.layout.as_dict(),
            "owner_id": self.owner_id,
            "evidence_digest": self.evidence_digest,
        }


@dataclass(frozen=True, slots=True)
class ReferenceVerification:
    """Result of authenticating a reference against the current live ledger."""

    valid: bool
    observation_id: str
    ledger_id: str
    recorded_epoch: int
    current_epoch: int
    epochs_advanced: int
    source_unchanged: bool
    reason: str

    def summary(self) -> dict[str, object]:
        return {
            "valid": self.valid,
            "observation_id": self.observation_id,
            "ledger_id": self.ledger_id,
            "recorded_epoch": self.recorded_epoch,
            "current_epoch": self.current_epoch,
            "epochs_advanced": self.epochs_advanced,
            "source_unchanged": self.source_unchanged,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class ObservationReceipt:
    """Admission result for one observed or corrected source."""

    observation_id: str
    record_id: str
    record_hash: str
    content_sha256: str
    epoch: int
    kind: str
    supersedes: str | None
    admitted: bool
    blob_created: bool

    def summary(self) -> dict[str, object]:
        return {
            "observation_id": self.observation_id,
            "record_id": self.record_id,
            "record_hash": self.record_hash,
            "content_sha256": self.content_sha256,
            "epoch": self.epoch,
            "kind": self.kind,
            "supersedes": self.supersedes,
            "admitted": self.admitted,
            "blob_created": self.blob_created,
        }


@dataclass(frozen=True, slots=True)
class RevocationReceipt:
    """Atomic revocation of one source and its correction descendants."""

    observation_id: str
    revoked_observation_ids: tuple[str, ...]
    epoch_before: int
    epoch_after: int
    retained_blob_sha256: tuple[str, ...]

    def summary(self) -> dict[str, object]:
        return {
            "observation_id": self.observation_id,
            "revoked_observation_ids": list(self.revoked_observation_ids),
            "revoked_count": len(self.revoked_observation_ids),
            "epoch_before": self.epoch_before,
            "epoch_after": self.epoch_after,
            "retained_blob_sha256": list(self.retained_blob_sha256),
        }


@dataclass(frozen=True, slots=True)
class VisualSourceInventory:
    """Current live exact-source inventory, derived only from the ledger."""

    ledger_id: str
    epoch: int
    scope: EligibilityScope
    observations: tuple[VisualSourceObservation, ...]

    @property
    def observation_ids(self) -> tuple[str, ...]:
        return tuple(observation.observation_id for observation in self.observations)

    @property
    def source_families(self) -> tuple[str, ...]:
        return tuple(sorted({observation.identity.family_id
                             for observation in self.observations}))

    @property
    def independent_worlds(self) -> tuple[str, ...]:
        return tuple(sorted({observation.identity.independent_world_id
                             for observation in self.observations}))

    @property
    def source_family_count(self) -> int:
        return len(self.source_families)

    @property
    def independent_world_count(self) -> int:
        return len(self.independent_worlds)

    def summary(self) -> dict[str, object]:
        families: dict[str, int] = {}
        worlds: dict[str, int] = {}
        for observation in self.observations:
            family = observation.identity.family_id
            world = observation.identity.independent_world_id
            families[family] = families.get(family, 0) + 1
            worlds[world] = worlds.get(world, 0) + 1
        return {
            "ledger_id": self.ledger_id,
            "epoch": self.epoch,
            "observation_count": len(self.observations),
            "observation_ids": list(self.observation_ids),
            # Counted independently; a source family is not a world identity.
            "source_families": sorted(families),
            "source_family_count": self.source_family_count,
            "source_family_observation_counts": dict(sorted(families.items())),
            "independent_worlds": sorted(worlds),
            "independent_world_count": self.independent_world_count,
            "independent_world_observation_counts": dict(sorted(worlds.items())),
        }


class VisualSourceMemory:
    """Exact visual-source admission and retrieval over a shared RecordLedger."""

    def __init__(self, ledger_path: str | Path, blob_root: str | Path):
        self.ledger_path = Path(ledger_path)
        self.blob_root = Path(blob_root)
        self.blob_root.mkdir(parents=True, exist_ok=True)
        self.ledger = RecordLedger(self.ledger_path)

    def close(self) -> None:
        self.ledger.close()

    def __enter__(self) -> VisualSourceMemory:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    @property
    def epoch(self) -> int:
        return self.ledger.epoch

    @property
    def ledger_id(self) -> str:
        return self.ledger.ledger_id

    def blob_path(self, content_sha256: str) -> Path:
        _sha256_field("content_sha256", content_sha256)
        return self.blob_root / content_sha256[:2] / content_sha256

    def blob_statistics(self) -> dict[str, object]:
        hashes = []
        total = 0
        for path in self.blob_root.glob("*/*"):
            if not path.is_file() or _SHA256_RE.fullmatch(path.name) is None:
                continue
            hashes.append(path.name)
            total += path.stat().st_size
        return {"blob_count": len(hashes), "blob_bytes": total,
                "blob_sha256": sorted(hashes)}

    # ------------------------------------------------------------------ writes
    def observe(self, image_bytes: bytes, *, observation_id: str, subject_id: str,
                branch_id: str, purpose: str, source_id: str, family_id: str,
                independent_world_id: str, asset_id: str, owner_id: str,
                event_id: str, observed_at: int, producer_id: str,
                event_at: int | None = None, crop_box: object = None,
                layout: SourceLayout | None = None,
                representation: str = PNG_REPRESENTATION,
                authorized_subject: str, authorized_branch: str,
                source_authorized: bool = False) -> ObservationReceipt:
        """Observe exact image bytes and admit only source lineage.

        Width, height, byte length and content hash are derived from the supplied
        bytes. Captions, labels, answers and predictions are not accepted.
        """
        data = _bytes_value("image_bytes", image_bytes)
        if representation not in SUPPORTED_REPRESENTATIONS:
            raise UnsupportedRepresentationError(
                f"unsupported source representation: {representation}")
        self._authorize_write(subject_id, branch_id, authorized_subject,
                              authorized_branch, source_authorized)
        try:
            width, height, _rows = decode_rgb_png(data)
        except VisualDataError as error:
            raise SourceIntegrityError(
                "image bytes are not the declared exact PNG representation") from error
        if event_at is None:
            event_at = observed_at
        identity = VisualSourceIdentity(
            source_id=source_id, family_id=family_id,
            independent_world_id=independent_world_id, asset_id=asset_id,
            owner_id=owner_id, event_id=event_id, representation=representation,
            content_sha256=_sha256_bytes(data), width=width, height=height,
            byte_length=len(data),
            crop_box=_normalized_crop(crop_box, width, height),
            layout=layout if layout is not None else SourceLayout("original"),
        )
        observation = VisualSourceObservation(
            identity=identity, observation_id=observation_id,
            subject_id=subject_id, branch_id=branch_id, purpose=purpose,
            producer_id=producer_id, observed_at=observed_at, event_at=event_at,
        )
        return self._admit(observation, data, authorized_subject,
                           authorized_branch, source_authorized)

    def correct(self, parent_observation_id: str, *, correction_id: str,
                scope: EligibilityScope, owner_id: str, crop_box: object,
                layout: SourceLayout, event_id: str, observed_at: int,
                event_at: int, authorized_subject: str, authorized_branch: str,
                producer_id: str | None = None, image_bytes: bytes | None = None,
                source_authorized: bool = False) -> ObservationReceipt:
        """Append a metadata/crop/owner correction, not a corrected visual fact.

        Immutable source identity and bytes are preserved. If ``image_bytes`` is
        omitted the parent's retained original bytes are reused without
        re-encoding. Supplied bytes must match the parent hash.
        """
        self._require_source_scope(scope)
        self._authorize_write(scope.subject_id, scope.branch_id, authorized_subject,
                              authorized_branch, source_authorized)
        if not isinstance(layout, SourceLayout):
            raise TypeError("correction layout must be a SourceLayout")
        snapshot = self.ledger.snapshot(scope)
        parent_observation, _record = self._select(
            snapshot, parent_observation_id, require_current_scope=True)
        if observed_at <= parent_observation.observed_at:
            raise ContractError("a correction must be observed after its parent")
        if correction_id == parent_observation_id:
            raise ContractError("a correction needs a distinct observation identity")
        if event_id == parent_observation.identity.event_id:
            raise ContractError("a correction needs a distinct event identity")
        if image_bytes is None:
            data = self._read_blob(parent_observation.identity.content_sha256,
                                  parent_observation.identity.byte_length)
        else:
            data = _bytes_value("image_bytes", image_bytes)
            if _sha256_bytes(data) != parent_observation.identity.content_sha256:
                raise SourceIntegrityError(
                    "a correction cannot substitute different source bytes")
            if len(data) != parent_observation.identity.byte_length:
                raise SourceIntegrityError(
                    "a correction cannot change the original byte length")
        identity = replace(
            parent_observation.identity, owner_id=owner_id, event_id=event_id,
            crop_box=_normalized_crop(crop_box, parent_observation.identity.width,
                                      parent_observation.identity.height),
            layout=layout,
        )
        observation = VisualSourceObservation(
            identity=identity, observation_id=correction_id,
            subject_id=scope.subject_id, branch_id=scope.branch_id,
            purpose=scope.purpose,
            producer_id=(producer_id if producer_id is not None
                         else parent_observation.producer_id),
            observed_at=observed_at, event_at=event_at,
            supersedes=parent_observation_id, kind=CORRECTION_KIND,
        )
        return self._admit(observation, data, authorized_subject,
                           authorized_branch, source_authorized)

    def revoke(self, observation_id: str, *, scope: EligibilityScope,
               authorized_subject: str, authorized_branch: str,
               expected_epoch: int | None = None,
               source_authorized: bool = False) -> RevocationReceipt:
        """Revoke a live source (including a superseded parent) and descendants."""
        self._require_source_scope(scope)
        self._authorize_write(scope.subject_id, scope.branch_id, authorized_subject,
                              authorized_branch, source_authorized)
        snapshot = self.ledger.snapshot(scope)
        self._check_epoch(snapshot.epoch, expected_epoch)
        observation = self._revocation_observation(observation_id, scope, snapshot)
        before = snapshot.epoch
        removed = self.ledger.revoke(
            observation_id, authorized_subject=authorized_subject,
            authorized_branch=authorized_branch, source_authorized=True)
        if not removed:
            raise SourceUnavailableError("the source was not live at revocation time")
        return RevocationReceipt(
            observation_id=observation_id,
            revoked_observation_ids=removed,
            epoch_before=before,
            epoch_after=self.ledger.epoch,
            retained_blob_sha256=(observation.identity.content_sha256,),
        )

    # --------------------------------------------------------------- retrieval
    def retrieve(self, request: VisualSourceRequest, scope: EligibilityScope, *,
                 expected_epoch: int | None = None) -> VisualSourceEvidence:
        """Return original source evidence after authenticating every binding."""
        if not isinstance(request, VisualSourceRequest):
            raise TypeError("retrieve requires an independently derived VisualSourceRequest")
        if request.identity.representation not in SUPPORTED_REPRESENTATIONS:
            raise UnsupportedRepresentationError(
                f"unsupported source representation: {request.identity.representation}")
        self._require_source_scope(scope)
        snapshot = self.ledger.snapshot(scope)
        self._check_epoch(snapshot.epoch, expected_epoch)
        observation, record = self._select(
            snapshot, request.observation_id, require_current_scope=True)
        if observation.identity.representation not in SUPPORTED_REPRESENTATIONS:
            raise UnsupportedRepresentationError(
                "the admitted source uses an unsupported representation")
        self._check_request(request, observation)
        data = self._read_blob(observation.identity.content_sha256,
                              observation.identity.byte_length)
        digest = self._evidence_digest(request.query_id, scope, record, observation)
        return VisualSourceEvidence(
            query_id=request.query_id, scope=scope, ledger_id=snapshot.ledger_id,
            epoch=snapshot.epoch, record_id=record.record_id,
            record_hash=record.record_hash, observation=observation,
            source_bytes=data, evidence_digest=digest,
        )

    def verify_evidence(self, evidence: VisualSourceEvidence, *,
                        scope: EligibilityScope, request: VisualSourceRequest,
                        expected_epoch: int | None = None) -> bool:
        """Authenticate a returned evidence object against the consumer contract.

        When ``expected_epoch`` is omitted the evidence's own epoch is required,
        so a candidate cannot silently ride a later epoch after a correction.
        """
        if not isinstance(evidence, VisualSourceEvidence):
            raise TypeError("verify_evidence requires VisualSourceEvidence")
        if not isinstance(request, VisualSourceRequest):
            raise TypeError("verify_evidence requires an independent VisualSourceRequest")
        if evidence.scope != scope:
            raise ForeignScopeError(
                "evidence scope differs from the independently supplied canonical scope")
        if evidence.query_id != request.query_id:
            raise QueryIdentityError("evidence query identity differs from the consumer request")
        if expected_epoch is None:
            expected_epoch = evidence.epoch
        canonical = self.retrieve(request, scope, expected_epoch=expected_epoch)
        if evidence.ledger_id != canonical.ledger_id:
            raise LedgerMismatchError("evidence ledger identity differs")
        if evidence.epoch != canonical.epoch:
            raise StaleEpochError("evidence epoch differs from the verified epoch")
        if evidence.record_id != canonical.record_id:
            raise SourceIntegrityError("evidence record identity differs")
        if evidence.record_hash != canonical.record_hash:
            raise SourceIntegrityError("evidence record hash differs")
        if evidence.observation != canonical.observation:
            raise SourceIntegrityError("evidence source observation differs")
        if evidence.source_bytes != canonical.source_bytes:
            raise SourceIntegrityError("evidence source bytes differ")
        if evidence.evidence_digest != canonical.evidence_digest:
            raise SourceIntegrityError("evidence digest differs")
        if (evidence.disposition != canonical.disposition
                or evidence.source_evidence is not True
                or evidence.factual_recall is not False):
            raise SourceIntegrityError("evidence disposition is not exact source evidence")
        return True

    def provenance_only(self, scope: EligibilityScope, *, query_id: str,
                        observation_id: str,
                        expected_epoch: int | None = None) -> SourceProvenance:
        """Return metadata-only provenance without bytes or factual recall.

        This is intentionally a different disposition from exact source
        retrieval. It reports that a live source record exists for this scope
        and query identity; learned factual memory recall is not evaluated.
        """
        _identifier("query_id", query_id)
        _identifier("observation_id", observation_id)
        self._require_source_scope(scope)
        snapshot = self.ledger.snapshot(scope)
        self._check_epoch(snapshot.epoch, expected_epoch)
        observation, record = self._select(
            snapshot, observation_id, require_current_scope=True)
        return SourceProvenance(
            query_id=query_id, ledger_id=snapshot.ledger_id, epoch=snapshot.epoch,
            observation_id=observation_id, record_hash=record.record_hash,
            identity=observation.identity,
        )

    def inventory(self, scope: EligibilityScope, *,
                  expected_epoch: int | None = None) -> VisualSourceInventory:
        """Return the live exact visual-source inventory for this scope."""
        self._require_source_scope(scope)
        snapshot = self.ledger.snapshot(scope)
        self._check_epoch(snapshot.epoch, expected_epoch)
        observations = []
        for record in snapshot.records:
            if record.kind not in PAYLOAD_KINDS:
                raise SourceIntegrityError(
                    "an unrelated record kind shares the exact visual-source scope")
            observation = self._parse_record(record)
            self._require_scope(observation, scope)
            observations.append(observation)
        return VisualSourceInventory(snapshot.ledger_id, snapshot.epoch, scope,
                                     tuple(observations))

    def verify_reference(self, reference: SourceReference, *,
                         scope: EligibilityScope,
                         expected_epoch: int | None = None) -> ReferenceVerification:
        """Verify that an unaffected source reference is still live and unchanged.

        Epoch advancement from unrelated corrections is permitted; a superseded
        or revoked source, changed bytes, changed layout/crop/owner or a foreign
        ledger/scope is not.
        """
        if not isinstance(reference, SourceReference):
            raise TypeError("verify_reference requires a SourceReference")
        self._require_source_scope(scope)
        if reference.ledger_id != self.ledger.ledger_id:
            raise LedgerMismatchError("reference ledger identity differs")
        self._require_reference_scope(reference, scope)
        snapshot = self.ledger.snapshot(scope)
        self._check_epoch(snapshot.epoch, expected_epoch)
        if snapshot.epoch < reference.epoch:
            raise StaleEpochError("reference epoch is newer than the live ledger")
        observation, record = self._select(
            snapshot, reference.observation_id, require_current_scope=True)
        if record.record_hash != reference.record_hash:
            raise SourceIntegrityError("referenced source record hash changed")
        if observation.identity.content_sha256 != reference.content_sha256:
            raise SourceIntegrityError("referenced source content hash changed")
        if observation.identity.representation != reference.representation:
            raise StaleSourceError("referenced source representation changed")
        if observation.identity.crop_box != reference.crop_box:
            raise StaleCropError("referenced source crop changed")
        if observation.identity.layout != reference.layout:
            raise LayoutMismatchError("referenced source layout changed")
        if observation.identity.owner_id != reference.owner_id:
            raise StaleOwnerError("referenced source owner changed")
        digest = self._evidence_digest(reference.query_id, reference.scope,
                                       record, observation)
        if digest != reference.evidence_digest:
            raise SourceIntegrityError("reference evidence digest changed")
        self._read_blob(observation.identity.content_sha256,
                        observation.identity.byte_length)
        return ReferenceVerification(
            valid=True, observation_id=reference.observation_id,
            ledger_id=snapshot.ledger_id, recorded_epoch=reference.epoch,
            current_epoch=snapshot.epoch,
            epochs_advanced=snapshot.epoch - reference.epoch,
            source_unchanged=True,
            reason="live source record and original bytes match the reference",
        )

    # ----------------------------------------------------------------- helpers
    def _authorize_write(self, subject_id: str, branch_id: str,
                         authorized_subject: str, authorized_branch: str,
                         source_authorized: bool) -> None:
        if source_authorized is not True:
            raise ForeignScopeError("source admission requires explicit source authority")
        if (authorized_subject != subject_id or authorized_branch != branch_id):
            raise ForeignScopeError("write authority does not match the record scope")
        if branch_id != SOURCE_BRANCH:
            raise ForeignScopeError(f"visual sources must use the {SOURCE_BRANCH} branch")

    @staticmethod
    def _require_source_scope(scope: EligibilityScope) -> None:
        if not isinstance(scope, EligibilityScope):
            raise TypeError("an independently derived EligibilityScope is required")
        if scope.mode not in SOURCE_MODES:
            raise ForeignScopeError("exact source retrieval requires a source scope mode")
        if scope.branch_id != SOURCE_BRANCH:
            raise ForeignScopeError(f"exact source retrieval requires branch {SOURCE_BRANCH!r}")

    @staticmethod
    def _require_scope(observation: VisualSourceObservation,
                       scope: EligibilityScope) -> None:
        if (observation.subject_id, observation.branch_id, observation.purpose) != (
                scope.subject_id, scope.branch_id, scope.purpose):
            raise ForeignScopeError("record subject, branch or purpose is foreign to the scope")

    @staticmethod
    def _require_reference_scope(reference: SourceReference,
                                 scope: EligibilityScope) -> None:
        recorded = reference.scope
        if ((recorded.subject_id, recorded.branch_id, recorded.purpose, recorded.mode)
                != (scope.subject_id, scope.branch_id, scope.purpose, scope.mode)):
            raise ForeignScopeError("reference scope identity differs from the consumer")
        if (recorded.mode == "source_snapshot"
                and recorded.snapshot_at != scope.snapshot_at):
            raise ForeignScopeError("reference snapshot boundary differs from the consumer")

    @staticmethod
    def _check_epoch(current: int, expected: int | None) -> None:
        if expected is None:
            return
        if type(expected) is not int or expected < 0:
            raise ValueError("expected_epoch must be a nonnegative integer")
        if expected != current:
            raise StaleEpochError(
                f"expected epoch {expected} is stale; current epoch is {current}")

    def _admit(self, observation: VisualSourceObservation, data: bytes,
               authorized_subject: str, authorized_branch: str,
               source_authorized: bool) -> ObservationReceipt:
        self._authorize_write(observation.subject_id, observation.branch_id,
                              authorized_subject, authorized_branch, source_authorized)
        if _sha256_bytes(data) != observation.identity.content_sha256:
            raise SourceIntegrityError("source bytes do not match the observed content hash")
        if len(data) != observation.identity.byte_length:
            raise SourceIntegrityError("source bytes do not match the observed byte length")
        blob_path, blob_created = self._write_blob(data)
        record = MemoryRecord(
            record_id=observation.observation_id,
            subject_id=observation.subject_id,
            branch_id=observation.branch_id,
            purpose=observation.purpose,
            text=canonical_json(observation.as_dict()),
            observed_at=observation.observed_at,
            event_at=observation.event_at,
            kind=observation.kind,
            supersedes=observation.supersedes,
            source_id=observation.identity.source_id,
            authenticity=SOURCE_AUTHENTICITY,
        )
        admitted = self.ledger.admit(
            record, authorized_subject=authorized_subject,
            authorized_branch=authorized_branch, source_authorized=True)
        return ObservationReceipt(
            observation_id=observation.observation_id, record_id=record.record_id,
            record_hash=record.record_hash,
            content_sha256=observation.identity.content_sha256,
            epoch=self.ledger.epoch, kind=observation.kind,
            supersedes=observation.supersedes, admitted=admitted,
            blob_created=blob_created,
        )

    def _write_blob(self, data: bytes) -> tuple[Path, bool]:
        digest = _sha256_bytes(data)
        path = self.blob_path(digest)
        if path.exists():
            if _sha256_bytes(path.read_bytes()) != digest:
                raise SourceIntegrityError("retained source bytes differ from their hash")
            return path, False
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".tmp-{os.getpid()}-{uuid.uuid4().hex}")
        try:
            with temporary.open("xb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            if path.exists():
                # A concurrent writer won the content-addressed race.
                if _sha256_bytes(path.read_bytes()) != digest:
                    raise SourceIntegrityError("concurrent source bytes differ from their hash")
                temporary.unlink(missing_ok=True)
                return path, False
            os.replace(temporary, path)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        return path, True

    def _read_blob(self, content_sha256: str, byte_length: int) -> bytes:
        path = self.blob_path(content_sha256)
        try:
            data = path.read_bytes()
        except FileNotFoundError as error:
            raise SourceIntegrityError("the original source bytes are missing") from error
        if len(data) != byte_length:
            raise SourceIntegrityError("the original source byte length changed")
        if _sha256_bytes(data) != content_sha256:
            raise SourceIntegrityError("the original source bytes no longer match their hash")
        return data

    def _select(self, snapshot, observation_id: str, *,
                require_current_scope: bool) -> tuple[VisualSourceObservation, MemoryRecord]:
        record = next((item for item in snapshot.records
                       if item.record_id == observation_id), None)
        if record is None:
            raise SourceUnavailableError(
                "the source is absent, ineligible, revoked or superseded")
        if record.kind not in PAYLOAD_KINDS:
            raise UnsupportedRepresentationError("the record is not an exact visual source")
        observation = self._parse_record(record)
        if require_current_scope:
            self._require_scope(observation, snapshot.scope)
        return observation, record

    def _revocation_observation(self, observation_id: str,
                                scope: EligibilityScope, snapshot) -> VisualSourceObservation:
        """Locate a live source for revocation, including a superseded parent.

        The current eligible snapshot omits a record once a correction supersedes
        it, but revoking the original source must still invalidate descendants.
        This reads the existing ledger row (never a parallel store) only to
        resolve that target; authorization and revocation remain ledger-owned.
        """
        record = next((item for item in snapshot.records
                       if item.record_id == observation_id), None)
        if record is not None:
            observation = self._parse_record(record)
            self._require_scope(observation, snapshot.scope)
            return observation
        uri = f"file:{self.ledger_path}?mode=ro"
        try:
            with sqlite3.connect(uri, uri=True, timeout=10.0) as database:
                row = database.execute(
                    "SELECT subject_id,branch_id,purpose,revoked,payload,"
                    "record_hash,source_hash FROM memory_records WHERE record_id=?",
                    (observation_id,)).fetchone()
        except sqlite3.Error as error:
            raise SourceIntegrityError("cannot resolve the revocation target") from error
        if row is None or row[3]:
            raise SourceUnavailableError("the source is absent or already revoked")
        if (row[0], row[1], row[2]) != (scope.subject_id, scope.branch_id,
                                        scope.purpose):
            raise ForeignScopeError("revocation target is foreign to the scope")
        if row[4] is None:
            raise SourceUnavailableError("the source payload is no longer live")
        try:
            stored = MemoryRecord(**json.loads(row[4]))
        except (TypeError, ValueError) as error:
            raise SourceIntegrityError("the revocation target payload is invalid") from error
        if stored.record_hash != row[5] or stored.source_hash != row[6]:
            raise SourceIntegrityError("the revocation target no longer matches its ledger hashes")
        observation = self._parse_record(stored)
        self._require_scope(observation, scope)
        return observation

    @staticmethod
    def _parse_record(record: MemoryRecord) -> VisualSourceObservation:
        try:
            value = json.loads(record.text)
        except (TypeError, ValueError) as error:
            raise SourceIntegrityError("source payload is not valid JSON") from error
        if not isinstance(value, dict) or set(value) != PAYLOAD_FIELDS:
            raise SourceIntegrityError(
                "source payload contains unsupported fields or omits lineage")
        if canonical_json(value) != record.text:
            raise SourceIntegrityError("source payload is not canonically encoded")
        if value["schema"] != PAYLOAD_SCHEMA:
            raise SourceIntegrityError("source payload schema is unsupported")
        if value["kind"] not in PAYLOAD_KINDS:
            raise SourceIntegrityError("source payload kind is unsupported")
        if value["authenticity"] != SOURCE_AUTHENTICITY:
            raise SourceIntegrityError("source payload authenticity is unsupported")
        try:
            identity = VisualSourceIdentity.from_dict({
                "source_id": value["source_id"],
                "family_id": value["family_id"],
                "independent_world_id": value["independent_world_id"],
                "asset_id": value["asset_id"],
                "owner_id": value["owner_id"],
                "event_id": value["event_id"],
                "representation": value["representation"],
                "content_sha256": value["content_sha256"],
                "width": value["width"],
                "height": value["height"],
                "byte_length": value["byte_length"],
                "crop_box": value["crop_box"],
                "layout": value["layout"],
            })
            observation = VisualSourceObservation(
                identity=identity,
                observation_id=value["observation_id"],
                subject_id=value["subject_id"],
                branch_id=value["branch_id"],
                purpose=value["purpose"],
                producer_id=value["producer_id"],
                observed_at=value["observed_at"],
                event_at=value["event_at"],
                supersedes=value["supersedes"],
                kind=value["kind"],
                authenticity=value["authenticity"],
            )
        except SourceIntegrityError:
            raise
        except (TypeError, ValueError) as error:
            raise SourceIntegrityError(
                "source payload violates the exact-source schema") from error
        if (record.record_id, record.subject_id, record.branch_id,
                record.purpose, record.observed_at, record.event_at,
                record.supersedes, record.kind, record.source_id,
                record.authenticity) != (
                observation.observation_id, observation.subject_id,
                observation.branch_id, observation.purpose,
                observation.observed_at, observation.event_at,
                observation.supersedes, observation.kind,
                observation.identity.source_id, observation.authenticity):
            raise SourceIntegrityError(
                "ledger row metadata differs from the immutable source payload")
        if record.source_hash != _sha256_bytes(record.text.encode("utf-8")):
            raise SourceIntegrityError("ledger source hash differs from the exact payload")
        return observation

    @staticmethod
    def _check_request(request: VisualSourceRequest,
                       observation: VisualSourceObservation) -> None:
        expected = request.identity
        actual = observation.identity
        if expected.representation != actual.representation:
            raise StaleSourceError("source representation differs from the request")
        for name in ("source_id", "family_id", "independent_world_id", "asset_id",
                     "event_id", "content_sha256", "width", "height", "byte_length"):
            if getattr(expected, name) != getattr(actual, name):
                raise StaleSourceError(f"source {name} differs from the request")
        if expected.owner_id != actual.owner_id:
            raise StaleOwnerError("source owner differs from the request")
        if expected.crop_box != actual.crop_box:
            raise StaleCropError("source crop differs from the request")
        if expected.layout != actual.layout:
            raise LayoutMismatchError("source layout differs from the request")

    @staticmethod
    def _evidence_digest(query_id: str, scope: EligibilityScope,
                         record: MemoryRecord,
                         observation: VisualSourceObservation) -> str:
        return fingerprint({
            "schema": PAYLOAD_SCHEMA,
            "query_id": query_id,
            "scope": {
                "subject_id": scope.subject_id,
                "branch_id": scope.branch_id,
                "purpose": scope.purpose,
                "as_of": scope.as_of,
                "mode": scope.mode,
                "snapshot_at": scope.snapshot_at,
            },
            "record_id": record.record_id,
            "record_hash": record.record_hash,
            "observation": observation.as_dict(),
        })

"""Delayed visual-memory episodes, verified contracts and CPU-only checks.

This module owns a new procedural visual-memory corpus and the *contract* for
the V3 episode described in ``docs/architecture/18``::

    observe image/regions -> commit observation-only state -> remove original
    visual context -> answer several later questions -> save/reload without
    re-encoding old observations -> append a dated correction -> answer
    corrected and unaffected questions -> fresh-process continuation.

The rendered PNGs, checked source annotations, object/region identities,
coordinates, timestamps and provenance are deterministic. Gold answers live
only in ``labels.jsonl``; episodes, queries and source annotations are checked
to contain no evaluator expectation field. The CPU adapter in this module is a
deterministic reference contract adapter, not a trained or learned memory. It
exists to exercise access boundaries, correction and persistence; it makes no
perception, learned-memory, native or quality claim.

Native evaluation is deliberately gated. ``assert_native_evaluation_allowed``
raises unless a separate package supplies both qualified perception and
memory-use evidence. This module never loads model weights, executes a native
kernel or launches a native job. The exact-source control lazily reuses the
existing SQLite ledger runtime, which is imported only when that arm is used.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from datetime import date, datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import time
from typing import Any, Iterable, Mapping, Sequence

from .contracts import canonical_json, fingerprint
from .visual_data import COLOR_NAMES, SHAPES, decode_rgb_png, render_scene_png


VERSION = "replica.visual_memory_episodes.v1"
CONFIG_FORMAT = "replica.visual_memory_episodes.config.v1"
PREPARED_FORMAT = "replica.visual_memory_episodes.prepared.v1"
MANIFEST_FORMAT = "replica.visual_memory_episodes.manifest.v1"
PROTOCOL_FORMAT = "replica.visual_memory_episodes.protocol.v1"
ADMISSION_FORMAT = "replica.visual_memory_episodes.admission.v1"
STATE_FORMAT = "replica.visual_memory_episodes.state.v1"
JOURNAL_FORMAT = "replica.visual_memory_episodes.journal.v1"
RUN_FORMAT = "replica.visual_memory_episodes.run.v1"
ANSWER_FORMAT = "replica.visual_memory_episodes.answer.v1"
LABEL_FORMAT = "replica.visual_memory_episodes.labels.v1"
RETRIEVAL_FORMAT = "replica.visual_memory_episodes.retrieval.v1"

SPLITS = ("train", "dev", "audit")
EVALUATION_SPLITS = ("dev", "audit")
QUERY_KINDS = ("initial", "corrected", "unaffected", "unknown")
INITIAL_QUERY_COUNT = 3
POST_CORRECTION_QUERY_COUNT = 5
ARMS = (
    "memory",
    "exact_source",
    "full_context",
    "zero_state",
    "shuffled_state",
    "disabled",
    "context_removed",
)
STATE_ARMS = ("memory", "zero_state", "shuffled_state")
CONTEXT_FREE_ARMS = ("memory", "zero_state", "shuffled_state", "disabled",
                     "context_removed")
RETRIEVAL_ARMS = ("exact_source",)
UNKNOWN_ANSWER = "unknown"
ATTRIBUTES = ("color", "shape")
UNKNOWN_OBJECT_ID = "o9"

EPISODE_FIELDS = frozenset({
    "episode_id", "subject_id", "branch_id", "purpose", "world_id",
    "removal_at", "removal_date", "initial", "correction",
})
OBSERVATION_FIELDS = frozenset({
    "observation_id", "episode_id", "world_id", "asset_id", "image_sha256",
    "width", "height", "byte_length", "observed_at", "event_at",
    "observed_date", "event_date", "producer_id", "source_id", "family_id",
    "independent_world_id", "owner_id", "crop_box", "layout_template",
    "cell_indices", "objects", "supersedes_observation_id",
})
OBJECT_FIELDS = frozenset({"object_id", "region_id", "bbox", "center",
                           "color", "shape"})
QUERY_FIELDS = frozenset({
    "query_id", "episode_id", "world_id", "text", "asked_at", "asked_date",
    "target_object_id", "attribute",
})
LABEL_FIELDS = frozenset({
    "query_id", "episode_id", "expected_answer", "query_kind", "split",
    "phase", "target_object_id", "attribute", "expected_observation_id",
    "expected_observation_sha256",
})
SOURCE_ANNOTATION_FIELDS = frozenset({
    "observation_id", "episode_id", "world_id", "image_sha256", "observed_at",
    "annotation_kind", "objects", "provenance",
})
ANSWER_FIELDS = frozenset({
    "answer_format", "query_id", "episode_id", "arm", "text", "answered",
    "source_kind", "retrieval_performed", "source_bytes_returned",
    "image_reads", "annotation_reads", "context_reads", "state_reads",
    "state_sha256", "source_observation_id", "source_epoch",
    "stale_rejected", "adapter_kind",
})
FORBIDDEN_INPUT_FIELDS = frozenset({
    "expected", "expected_answer", "expected_observation_id", "gold", "label",
    "labels", "query_kind", "phase", "split", "role", "target_role",
    "correct", "incorrect", "quality",
})
FORBIDDEN_INPUT_MARKERS = ("expected_answer", "gold_answer",
                           "evaluator_label", "answer_key")
IMPLEMENTATION_PATHS = (
    "src/replica_lab/visual_memory_episodes.py",
    "scripts/prepare_visual_memory.py",
    "scripts/run_visual_memory.py",
    "tests/test_visual_memory_episodes.py",
    "configs/visual_memory_v1.json",
)
DEFAULT_AS_OF = 2**63 - 1
DEFAULT_MAX_WALL_SECONDS = 180
DEFAULT_LIMITS = {
    "max_episodes": 16,
    "max_queries": 256,
    "max_answers": 2048,
    "max_images": 32,
    "max_image_bytes": 2 * 1024 * 1024,
    "max_state_bytes": 256 * 1024,
    "max_journal_bytes": 2 * 1024 * 1024,
    "native_jobs": 0,
    "optimizer_updates": 0,
    "gradient_examples": 0,
    "max_wall_seconds": DEFAULT_MAX_WALL_SECONDS,
}
DEFAULT_MECHANICAL_GATES = {
    "require_exact_delayed_queries": True,
    "require_no_context_reads_after_removal": True,
    "require_restart_without_reencode": True,
    "require_interrupted_restart_recovery": True,
    "require_dated_correction": True,
    "require_unaffected_retention": True,
    "require_unknown_abstention": True,
    "require_stale_source_rejection": True,
    "require_label_boundary": True,
    "require_mechanical_separation": True,
}
NATIVE_GATE = {
    "status": "gated",
    "native_evaluated": False,
    "required_evidence": ["qualified_perception_producer",
                          "memory_use_evidence"],
    "perception": "not_established_by_this_package",
    "memory_use": "not_evaluated",
    "reason": (
        "This package supplies deterministic episodes, typed interfaces and "
        "CPU contract checks only. A native evaluation requires a qualified "
        "visual producer and measured useful memory above zero/shuffled/"
        "disabled/context-removed controls, with corrections and unaffected "
        "retention across a fresh process."
    ),
}


class VisualMemoryError(ValueError):
    """Base error for this module's contracts."""


class ContractError(VisualMemoryError):
    """A supplied object violates the declared visual-memory contract."""


class LabelBoundaryError(ContractError):
    """An evaluator expectation crossed into a sensing, writer or query input."""


class ContextRemovedError(ContractError):
    """Original visual context was removed and cannot be read by this arm."""


class StaleVisualSourceError(ContractError):
    """An exact-source request or reference is stale, superseded or unavailable."""


class JournalError(ContractError):
    """The persisted journal is malformed, truncated in the middle or inconsistent."""


class StateError(ContractError):
    """Persisted or supplied state is inconsistent with its declared hash."""


class NativeEvaluationGated(ContractError):
    """A native run was requested before both prerequisite evidence gates passed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: str | Path) -> str:
    return _sha256_bytes(Path(path).read_bytes())


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _json_bytes(value: Any) -> bytes:
    return (canonical_json(value) + "\n").encode("utf-8")


def _jsonl_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join(_json_bytes(row) for row in rows)


def _write_exact(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != data:
        raise ContractError(f"refusing to overwrite different bytes: {path}")
    path.write_bytes(data)


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ContractError(f"JSONL row is not an object: {path}")
        rows.append(value)
    return rows


def _identifier(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise ContractError(f"{name} must be a nonempty string without NUL")
    return value


def _positive_int(name: str, value: Any) -> int:
    if type(value) is not int or value < 1:
        raise ContractError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(name: str, value: Any) -> int:
    if type(value) is not int or value < 0:
        raise ContractError(f"{name} must be a nonnegative integer")
    return value


def _day(value: str) -> int:
    try:
        parsed = date.fromisoformat(value)
    except (TypeError, ValueError) as error:
        raise ContractError(f"invalid ISO date: {value!r}") from error
    return int(parsed.strftime("%Y%m%d"))


def _iso_day(value: int) -> str:
    _nonnegative_int("day", value)
    return date(int(str(value)[:4]), int(str(value)[4:6]),
                int(str(value)[6:8])).isoformat()


def _timestamp(value: Any, name: str) -> int:
    if type(value) is not int or value < 0 or value > 2**63 - 1:
        raise ContractError(f"{name} must be a nonnegative integer timestamp")
    return value


def _exact_keys(value: Any, expected: Iterable[str], name: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ContractError(f"{name} must be an object")
    actual = set(value)
    missing = set(expected) - actual
    extra = actual - set(expected)
    if missing or extra:
        raise ContractError(f"{name} fields differ; missing={sorted(missing)} "
                            f"extra={sorted(extra)}")
    return value


def _check_label_boundary(row: Mapping[str, Any], name: str) -> None:
    forbidden = set(row) & FORBIDDEN_INPUT_FIELDS
    if forbidden:
        raise LabelBoundaryError(f"{name} carries evaluator-only fields: {sorted(forbidden)}")
    text = canonical_json(row)
    for marker in FORBIDDEN_INPUT_MARKERS:
        if marker in text:
            raise LabelBoundaryError(f"{name} carries forbidden marker {marker!r}")


def _implementation_sha256(root: Path | None = None) -> str:
    root = Path(root) if root is not None else Path(__file__).resolve().parents[2]
    digest = hashlib.sha256()
    for relative in IMPLEMENTATION_PATHS:
        path = root / relative
        if not path.is_file():
            raise ContractError(f"implementation pin is missing: {relative}")
        digest.update(relative.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(path.read_bytes())
        digest.update(b"\x00")
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Typed episode and evaluator records
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Region:
    """Stable object region identity, pixel box and centre."""

    bbox: tuple[int, int, int, int]
    center: tuple[int, int]

    def __post_init__(self) -> None:
        if len(self.bbox) != 4 or any(type(v) is not int for v in self.bbox):
            raise ContractError("bbox must be four integers")
        if len(self.center) != 2 or any(type(v) is not int for v in self.center):
            raise ContractError("center must be two integers")
        x0, y0, x1, y1 = self.bbox
        if not (0 <= x0 < x1 and 0 <= y0 < y1):
            raise ContractError("bbox must be a positive pixel rectangle")
        if not (x0 <= self.center[0] < x1 and y0 <= self.center[1] < y1):
            raise ContractError("center must lie inside bbox")

    def as_dict(self) -> dict[str, Any]:
        return {"bbox": list(self.bbox), "center": list(self.center)}

    @classmethod
    def from_dict(cls, value: object) -> "Region":
        _exact_keys(value, ("bbox", "center"), "region")
        return cls(tuple(value["bbox"]), tuple(value["center"]))

    @property
    def sha256(self) -> str:
        return fingerprint(self.as_dict())


@dataclass(frozen=True, slots=True)
class ObjectObservation:
    """One observed object/region; source annotation, never a gold label."""

    object_id: str
    region_id: str
    region: Region
    color: str
    shape: str

    def __post_init__(self) -> None:
        _identifier("object_id", self.object_id)
        _identifier("region_id", self.region_id)
        if not isinstance(self.region, Region):
            raise ContractError("object region must be a Region")
        if self.color not in COLOR_NAMES:
            raise ContractError(f"unknown observed color: {self.color!r}")
        if self.shape not in SHAPES:
            raise ContractError(f"unknown observed shape: {self.shape!r}")

    def as_dict(self) -> dict[str, Any]:
        return {
            "object_id": self.object_id,
            "region_id": self.region_id,
            "bbox": list(self.region.bbox),
            "center": list(self.region.center),
            "color": self.color,
            "shape": self.shape,
        }

    @classmethod
    def from_dict(cls, value: object) -> "ObjectObservation":
        _exact_keys(value, OBJECT_FIELDS, "object observation")
        return cls(
            object_id=value["object_id"], region_id=value["region_id"],
            region=Region(tuple(value["bbox"]), tuple(value["center"])),
            color=value["color"], shape=value["shape"],
        )


@dataclass(frozen=True, slots=True)
class VisualObservation:
    """One dated visual observation with source and region provenance."""

    observation_id: str
    episode_id: str
    world_id: str
    asset_id: str
    image_sha256: str
    width: int
    height: int
    byte_length: int
    observed_at: int
    event_at: int
    observed_date: str
    event_date: str
    producer_id: str
    source_id: str
    family_id: str
    independent_world_id: str
    owner_id: str
    crop_box: tuple[int, int, int, int] | None
    layout_template: str
    cell_indices: tuple[int, ...]
    objects: tuple[ObjectObservation, ...]
    supersedes_observation_id: str | None = None

    def __post_init__(self) -> None:
        for name in ("observation_id", "episode_id", "world_id", "asset_id",
                     "producer_id", "source_id", "family_id",
                     "independent_world_id", "owner_id", "layout_template"):
            _identifier(name, getattr(self, name))
        if not _is_sha256(self.image_sha256):
            raise ContractError("image_sha256 must be a lowercase SHA-256")
        _positive_int("width", self.width)
        _positive_int("height", self.height)
        _positive_int("byte_length", self.byte_length)
        for name in ("observed_at", "event_at"):
            _timestamp(getattr(self, name), name)
        if tuple(self.cell_indices) != tuple(sorted(set(self.cell_indices))):
            raise ContractError("cell_indices must be unique and ordered")
        if self.crop_box is not None:
            if len(self.crop_box) != 4:
                raise ContractError("crop_box must have four integers")
            x0, y0, x1, y1 = self.crop_box
            if not (0 <= x0 < x1 <= self.width and 0 <= y0 < y1 <= self.height):
                raise ContractError("crop_box must lie inside the image")
        ids = [obj.object_id for obj in self.objects]
        regions = [obj.region_id for obj in self.objects]
        if len(set(ids)) != len(ids) or len(set(regions)) != len(regions):
            raise ContractError("object and region identities must be unique")
        if self.supersedes_observation_id is not None:
            _identifier("supersedes_observation_id", self.supersedes_observation_id)
            if self.supersedes_observation_id == self.observation_id:
                raise ContractError("an observation cannot supersede itself")

    def as_dict(self) -> dict[str, Any]:
        return {
            "observation_id": self.observation_id,
            "episode_id": self.episode_id,
            "world_id": self.world_id,
            "asset_id": self.asset_id,
            "image_sha256": self.image_sha256,
            "width": self.width,
            "height": self.height,
            "byte_length": self.byte_length,
            "observed_at": self.observed_at,
            "event_at": self.event_at,
            "observed_date": self.observed_date,
            "event_date": self.event_date,
            "producer_id": self.producer_id,
            "source_id": self.source_id,
            "family_id": self.family_id,
            "independent_world_id": self.independent_world_id,
            "owner_id": self.owner_id,
            "crop_box": list(self.crop_box) if self.crop_box is not None else None,
            "layout_template": self.layout_template,
            "cell_indices": list(self.cell_indices),
            "objects": [obj.as_dict() for obj in self.objects],
            "supersedes_observation_id": self.supersedes_observation_id,
        }

    @classmethod
    def from_dict(cls, value: object) -> "VisualObservation":
        _exact_keys(value, OBSERVATION_FIELDS, "observation")
        crop = value["crop_box"]
        return cls(
            observation_id=value["observation_id"], episode_id=value["episode_id"],
            world_id=value["world_id"], asset_id=value["asset_id"],
            image_sha256=value["image_sha256"], width=value["width"],
            height=value["height"], byte_length=value["byte_length"],
            observed_at=value["observed_at"], event_at=value["event_at"],
            observed_date=value["observed_date"], event_date=value["event_date"],
            producer_id=value["producer_id"], source_id=value["source_id"],
            family_id=value["family_id"],
            independent_world_id=value["independent_world_id"],
            owner_id=value["owner_id"],
            crop_box=tuple(crop) if crop is not None else None,
            layout_template=value["layout_template"],
            cell_indices=tuple(value["cell_indices"]),
            objects=tuple(ObjectObservation.from_dict(obj) for obj in value["objects"]),
            supersedes_observation_id=value["supersedes_observation_id"],
        )

    def object_ids(self) -> tuple[str, ...]:
        return tuple(obj.object_id for obj in self.objects)


@dataclass(frozen=True, slots=True)
class VisualQuery:
    """A delayed question. Contains no expected answer or evaluator role."""

    query_id: str
    episode_id: str
    world_id: str
    text: str
    asked_at: int
    asked_date: str
    target_object_id: str
    attribute: str

    def __post_init__(self) -> None:
        for name in ("query_id", "episode_id", "world_id", "text",
                     "asked_date", "target_object_id", "attribute"):
            _identifier(name, getattr(self, name))
        _timestamp(self.asked_at, "asked_at")
        if self.attribute not in ATTRIBUTES:
            raise ContractError(f"unsupported query attribute: {self.attribute!r}")
        if not self.text.endswith("?"):
            raise ContractError("query text must be a question")

    def as_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "episode_id": self.episode_id,
            "world_id": self.world_id,
            "text": self.text,
            "asked_at": self.asked_at,
            "asked_date": self.asked_date,
            "target_object_id": self.target_object_id,
            "attribute": self.attribute,
        }

    @classmethod
    def from_dict(cls, value: object) -> "VisualQuery":
        _exact_keys(value, QUERY_FIELDS, "query")
        return cls(**value)


@dataclass(frozen=True, slots=True)
class EpisodeLabel:
    """Evaluator-only expectation. Never supplied to a sensing or writer input."""

    query_id: str
    episode_id: str
    expected_answer: str
    query_kind: str
    split: str
    phase: str
    target_object_id: str
    attribute: str
    expected_observation_id: str
    expected_observation_sha256: str

    def __post_init__(self) -> None:
        for name in ("query_id", "episode_id", "expected_answer", "query_kind",
                     "split", "phase", "target_object_id", "attribute",
                     "expected_observation_id"):
            _identifier(name, getattr(self, name))
        if self.query_kind not in QUERY_KINDS:
            raise ContractError(f"unknown query kind: {self.query_kind!r}")
        if self.split not in SPLITS:
            raise ContractError(f"unknown split: {self.split!r}")
        if self.phase not in ("initial", "after_correction"):
            raise ContractError(f"unknown phase: {self.phase!r}")
        if self.attribute not in ATTRIBUTES:
            raise ContractError(f"unsupported label attribute: {self.attribute!r}")
        if not _is_sha256(self.expected_observation_sha256):
            raise ContractError("expected observation hash must be a SHA-256")

    def as_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "episode_id": self.episode_id,
            "expected_answer": self.expected_answer,
            "query_kind": self.query_kind,
            "split": self.split,
            "phase": self.phase,
            "target_object_id": self.target_object_id,
            "attribute": self.attribute,
            "expected_observation_id": self.expected_observation_id,
            "expected_observation_sha256": self.expected_observation_sha256,
        }

    @classmethod
    def from_dict(cls, value: object) -> "EpisodeLabel":
        _exact_keys(value, LABEL_FIELDS, "label")
        return cls(**value)


@dataclass(frozen=True, slots=True)
class Episode:
    """One deterministic visual-memory episode (sensing fields only)."""

    episode_id: str
    subject_id: str
    branch_id: str
    purpose: str
    world_id: str
    removal_at: int
    removal_date: str
    initial: VisualObservation
    correction: VisualObservation

    def __post_init__(self) -> None:
        for name in ("episode_id", "subject_id", "branch_id", "purpose",
                     "world_id", "removal_date"):
            _identifier(name, getattr(self, name))
        _timestamp(self.removal_at, "removal_at")
        if not isinstance(self.initial, VisualObservation):
            raise ContractError("initial observation is required")
        if not isinstance(self.correction, VisualObservation):
            raise ContractError("correction observation is required")
        if self.initial.episode_id != self.episode_id or self.correction.episode_id != self.episode_id:
            raise ContractError("observation episode identity differs")
        if self.initial.world_id != self.world_id or self.correction.world_id != self.world_id:
            raise ContractError("observation world identity differs")
        if self.correction.supersedes_observation_id != self.initial.observation_id:
            raise ContractError("correction must supersede the initial observation")
        if self.correction.observed_at <= self.initial.observed_at:
            raise ContractError("correction must be observed strictly after the initial view")
        if self.initial.object_ids() != self.correction.object_ids():
            raise ContractError("correction must preserve object identity")
        initial = {obj.object_id: obj for obj in self.initial.objects}
        corrected = {obj.object_id: obj for obj in self.correction.objects}
        changed = 0
        for object_id, first in initial.items():
            second = corrected[object_id]
            if first.region_id != second.region_id or first.region != second.region:
                raise ContractError("correction must preserve region identity and coordinates")
            if first.color != second.color:
                changed += 1
            if first.shape != second.shape:
                raise ContractError("this corpus changes only one object colour")
        if changed != 1:
            raise ContractError("correction must change exactly one observed colour")

    def as_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "subject_id": self.subject_id,
            "branch_id": self.branch_id,
            "purpose": self.purpose,
            "world_id": self.world_id,
            "removal_at": self.removal_at,
            "removal_date": self.removal_date,
            "initial": self.initial.as_dict(),
            "correction": self.correction.as_dict(),
        }

    @classmethod
    def from_dict(cls, value: object) -> "Episode":
        _exact_keys(value, EPISODE_FIELDS, "episode")
        return cls(
            episode_id=value["episode_id"], subject_id=value["subject_id"],
            branch_id=value["branch_id"], purpose=value["purpose"],
            world_id=value["world_id"], removal_at=value["removal_at"],
            removal_date=value["removal_date"],
            initial=VisualObservation.from_dict(value["initial"]),
            correction=VisualObservation.from_dict(value["correction"]),
        )

    @property
    def observations(self) -> tuple[VisualObservation, VisualObservation]:
        return (self.initial, self.correction)

    def changed_object(self) -> str:
        first = {obj.object_id: obj.color for obj in self.initial.objects}
        for obj in self.correction.objects:
            if first[obj.object_id] != obj.color:
                return obj.object_id
        raise ContractError("episode has no corrected object")


# ---------------------------------------------------------------------------
# Committed state. It carries identity, coordinates, dates and provenance, but
# not image bytes, annotations, evaluator labels or model tensors.
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class SourceBinding:
    """Provenance of one observation that contributed to committed state."""

    observation_id: str
    image_sha256: str
    observed_at: int
    event_at: int
    source_id: str
    family_id: str
    independent_world_id: str
    asset_id: str
    producer_id: str
    owner_id: str
    crop_box: tuple[int, int, int, int] | None
    layout_template: str
    cell_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        for name in ("observation_id", "source_id", "family_id",
                     "independent_world_id", "asset_id", "producer_id", "owner_id",
                     "layout_template"):
            _identifier(name, getattr(self, name))
        if not _is_sha256(self.image_sha256):
            raise ContractError("binding image hash must be a SHA-256")
        _timestamp(self.observed_at, "observed_at")
        _timestamp(self.event_at, "event_at")
        if self.crop_box is not None:
            if len(self.crop_box) != 4 or any(type(v) is not int for v in self.crop_box):
                raise ContractError("binding crop_box must be four integers")
        if tuple(self.cell_indices) != tuple(sorted(set(self.cell_indices))):
            raise ContractError("binding cell_indices must be unique and ordered")

    def as_dict(self) -> dict[str, Any]:
        return {
            "observation_id": self.observation_id,
            "image_sha256": self.image_sha256,
            "observed_at": self.observed_at,
            "event_at": self.event_at,
            "source_id": self.source_id,
            "family_id": self.family_id,
            "independent_world_id": self.independent_world_id,
            "asset_id": self.asset_id,
            "producer_id": self.producer_id,
            "owner_id": self.owner_id,
            "crop_box": list(self.crop_box) if self.crop_box is not None else None,
            "layout_template": self.layout_template,
            "cell_indices": list(self.cell_indices),
        }

    @classmethod
    def from_dict(cls, value: object) -> "SourceBinding":
        expected = {
            "observation_id", "image_sha256", "observed_at", "event_at",
            "source_id", "family_id", "independent_world_id", "asset_id",
            "producer_id", "owner_id", "crop_box", "layout_template",
            "cell_indices",
        }
        _exact_keys(value, expected, "source binding")
        crop = value["crop_box"]
        return cls(
            observation_id=value["observation_id"],
            image_sha256=value["image_sha256"],
            observed_at=value["observed_at"], event_at=value["event_at"],
            source_id=value["source_id"], family_id=value["family_id"],
            independent_world_id=value["independent_world_id"],
            asset_id=value["asset_id"], producer_id=value["producer_id"],
            owner_id=value["owner_id"],
            crop_box=tuple(crop) if crop is not None else None,
            layout_template=value["layout_template"],
            cell_indices=tuple(value["cell_indices"]),
        )


@dataclass(frozen=True, slots=True)
class StateFact:
    """One observed attribute bound to a stable object/region and date."""

    fact_id: str
    object_id: str
    region_id: str
    region: Region
    attribute: str
    value: str
    valid_from: int
    event_at: int
    observation_id: str
    supersedes: str | None
    image_sha256: str
    source_id: str
    producer_id: str

    def __post_init__(self) -> None:
        for name in ("fact_id", "object_id", "region_id", "attribute", "value",
                     "observation_id", "source_id", "producer_id"):
            _identifier(name, getattr(self, name))
        if not isinstance(self.region, Region):
            raise ContractError("fact region must be a Region")
        _timestamp(self.valid_from, "valid_from")
        _timestamp(self.event_at, "event_at")
        if not _is_sha256(self.image_sha256):
            raise ContractError("fact source hash must be a SHA-256")
        if self.supersedes is not None:
            _identifier("supersedes", self.supersedes)

    def as_dict(self) -> dict[str, Any]:
        return {
            "fact_id": self.fact_id,
            "object_id": self.object_id,
            "region_id": self.region_id,
            "bbox": list(self.region.bbox),
            "center": list(self.region.center),
            "attribute": self.attribute,
            "value": self.value,
            "valid_from": self.valid_from,
            "event_at": self.event_at,
            "observation_id": self.observation_id,
            "supersedes": self.supersedes,
            "image_sha256": self.image_sha256,
            "source_id": self.source_id,
            "producer_id": self.producer_id,
        }

    @classmethod
    def from_dict(cls, value: object) -> "StateFact":
        expected = {
            "fact_id", "object_id", "region_id", "bbox", "center", "attribute",
            "value", "valid_from", "event_at", "observation_id", "supersedes",
            "image_sha256", "source_id", "producer_id",
        }
        _exact_keys(value, expected, "fact")
        return cls(
            fact_id=value["fact_id"], object_id=value["object_id"],
            region_id=value["region_id"],
            region=Region(tuple(value["bbox"]), tuple(value["center"])),
            attribute=value["attribute"], value=value["value"],
            valid_from=value["valid_from"], event_at=value["event_at"],
            observation_id=value["observation_id"],
            supersedes=value["supersedes"], image_sha256=value["image_sha256"],
            source_id=value["source_id"], producer_id=value["producer_id"],
        )


def _state_payload(state: "CommittedState") -> dict[str, Any]:
    return {
        "format": STATE_FORMAT,
        "episode_id": state.episode_id,
        "world_id": state.world_id,
        "subject_id": state.subject_id,
        "branch_id": state.branch_id,
        "purpose": state.purpose,
        "created_at": state.created_at,
        "facts": [fact.as_dict() for fact in state.facts],
        "source_bindings": [binding.as_dict() for binding in state.source_bindings],
        "correction_count": state.correction_count,
        "context_removed": state.context_removed,
        "removal_at": state.removal_at,
    }


@dataclass(frozen=True, slots=True)
class CommittedState:
    """Observation-only committed state; no pixels, annotations or labels."""

    episode_id: str
    world_id: str
    subject_id: str
    branch_id: str
    purpose: str
    created_at: int
    facts: tuple[StateFact, ...]
    source_bindings: tuple[SourceBinding, ...]
    correction_count: int = 0
    context_removed: bool = False
    removal_at: int | None = None
    state_sha256: str = ""

    def __post_init__(self) -> None:
        for name in ("episode_id", "world_id", "subject_id", "branch_id", "purpose"):
            _identifier(name, getattr(self, name))
        _timestamp(self.created_at, "created_at")
        _nonnegative_int("correction_count", self.correction_count)
        if self.removal_at is not None:
            _timestamp(self.removal_at, "removal_at")
        if self.context_removed and self.removal_at is None:
            raise ContractError("removed context requires a removal timestamp")
        for fact in self.facts:
            if not isinstance(fact, StateFact):
                raise ContractError("facts must be StateFact objects")
        for binding in self.source_bindings:
            if not isinstance(binding, SourceBinding):
                raise ContractError("bindings must be SourceBinding objects")
        if self.state_sha256:
            if not _is_sha256(self.state_sha256):
                raise ContractError("state hash must be a SHA-256")
            if self.state_sha256 != fingerprint(_state_payload(self)):
                raise StateError("state hash differs from its canonical payload")

    def sealed(self) -> "CommittedState":
        payload = _state_payload(self)
        return replace(self, state_sha256=fingerprint(payload))

    def as_dict(self) -> dict[str, Any]:
        payload = _state_payload(self)
        payload["state_sha256"] = self.state_sha256
        return payload

    def to_bytes(self) -> bytes:
        if not self.state_sha256:
            raise StateError("state must be sealed before serialization")
        return _json_bytes(self.as_dict())

    @classmethod
    def from_dict(cls, value: object) -> "CommittedState":
        expected = {
            "format", "episode_id", "world_id", "subject_id", "branch_id",
            "purpose", "created_at", "facts", "source_bindings",
            "correction_count", "context_removed", "removal_at", "state_sha256",
        }
        _exact_keys(value, expected, "state")
        if value["format"] != STATE_FORMAT:
            raise StateError("state format is unsupported")
        return cls(
            episode_id=value["episode_id"], world_id=value["world_id"],
            subject_id=value["subject_id"], branch_id=value["branch_id"],
            purpose=value["purpose"], created_at=value["created_at"],
            facts=tuple(StateFact.from_dict(item) for item in value["facts"]),
            source_bindings=tuple(SourceBinding.from_dict(item)
                                  for item in value["source_bindings"]),
            correction_count=value["correction_count"],
            context_removed=value["context_removed"],
            removal_at=value["removal_at"],
            state_sha256=value["state_sha256"],
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> "CommittedState":
        try:
            value = json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as error:
            raise StateError("state bytes are not canonical JSON") from error
        if (canonical_json(value) + "\n").encode("utf-8") != data:
            raise StateError("state bytes are not exactly canonical")
        return cls.from_dict(value)


@dataclass(frozen=True, slots=True)
class ObservedEncoding:
    """Result of encoding one new observation; never includes old image bytes."""

    observation: VisualObservation
    facts: tuple[StateFact, ...]


def _fact_id(observation: VisualObservation, obj: ObjectObservation,
             attribute: str, value: str) -> str:
    return fingerprint({
        "observation_id": observation.observation_id,
        "object_id": obj.object_id,
        "region_id": obj.region_id,
        "attribute": attribute,
        "value": value,
    })[:24]


def _binding(observation: VisualObservation) -> SourceBinding:
    return SourceBinding(
        observation_id=observation.observation_id,
        image_sha256=observation.image_sha256,
        observed_at=observation.observed_at,
        event_at=observation.event_at,
        source_id=observation.source_id,
        family_id=observation.family_id,
        independent_world_id=observation.independent_world_id,
        asset_id=observation.asset_id,
        producer_id=observation.producer_id,
        owner_id=observation.owner_id,
        crop_box=observation.crop_box,
        layout_template=observation.layout_template,
        cell_indices=observation.cell_indices,
    )


class ReferenceContractAdapter:
    """Deterministic CPU reference adapter for interface and boundary checks.

    ``encode_observation`` verifies the image bytes and converts the supplied
    checked source annotation into facts. ``commit`` preserves unaffected facts
    and appends supersessions for changed values. This is not a perception or
    learned-memory model; the annotation is produced by the owned renderer and
    is explicitly labelled as such.
    """

    adapter_kind = "deterministic_reference_contract_adapter"
    encoded_calls = 0

    def __init__(self) -> None:
        self.encoded_calls = 0
        self.disabled = False

    def encode_observation(self, observation: VisualObservation,
                           image_bytes: bytes) -> ObservedEncoding:
        if self.disabled:
            raise ContractError("the disabled adapter cannot encode observations")
        if not isinstance(image_bytes, bytes):
            raise ContractError("image bytes must be bytes")
        if _sha256_bytes(image_bytes) != observation.image_sha256:
            raise ContractError("image bytes differ from the observation hash")
        if len(image_bytes) != observation.byte_length:
            raise ContractError("image byte length differs from the observation")
        width, height, _rows = decode_rgb_png(image_bytes)
        if (width, height) != (observation.width, observation.height):
            raise ContractError("decoded image dimensions differ from the observation")
        self.encoded_calls += 1
        facts = []
        for obj in observation.objects:
            for attribute, value in (("color", obj.color), ("shape", obj.shape)):
                facts.append(StateFact(
                    fact_id=_fact_id(observation, obj, attribute, value),
                    object_id=obj.object_id, region_id=obj.region_id,
                    region=obj.region, attribute=attribute, value=value,
                    valid_from=observation.observed_at,
                    event_at=observation.event_at,
                    observation_id=observation.observation_id,
                    supersedes=None, image_sha256=observation.image_sha256,
                    source_id=observation.source_id,
                    producer_id=observation.producer_id,
                ))
        return ObservedEncoding(observation=observation, facts=tuple(facts))

    def commit(self, encoding: ObservedEncoding, state: CommittedState | None, *,
               correction: bool = False, subject_id: str | None = None,
               branch_id: str | None = None, purpose: str | None = None) -> CommittedState:
        observation = encoding.observation
        if state is None:
            if correction:
                raise ContractError("a correction requires an existing committed state")
            if subject_id is None or branch_id is None or purpose is None:
                raise ContractError("initial commit requires episode scope metadata")
            facts = list(encoding.facts)
            bindings = [_binding(observation)]
            correction_count = 0
            created_at = observation.observed_at
            episode_id, world_id = observation.episode_id, observation.world_id
            context_removed, removal_at = False, None
        else:
            if (state.episode_id, state.world_id) != (observation.episode_id,
                                                      observation.world_id):
                raise ContractError("state and observation episode/world differ")
            _require(not state.context_removed or correction,
                     "further observations require an explicit correction event")
            previous_bindings = list(state.source_bindings)
            if correction:
                if not previous_bindings:
                    raise ContractError("a correction requires a previous source binding")
                if observation.observed_at <= previous_bindings[-1].observed_at:
                    raise ContractError("a dated correction must be observed later")
                if observation.supersedes_observation_id is None:
                    raise ContractError("a correction must name its parent observation")
                if observation.supersedes_observation_id != previous_bindings[-1].observation_id:
                    raise ContractError(
                        "a correction must supersede the immediately preceding observation")
            facts = list(state.facts)
            current = _current_facts(facts, observation.observed_at)
            changed = 0
            for incoming in encoding.facts:
                key = (incoming.object_id, incoming.attribute)
                existing = current.get(key)
                if existing is not None:
                    if existing.region_id != incoming.region_id or existing.region != incoming.region:
                        raise ContractError(
                            "a committed observation cannot silently move a region")
                    if existing.valid_from >= incoming.valid_from:
                        raise ContractError("new observation is not later than committed facts")
                    if existing.value == incoming.value:
                        continue
                    incoming = replace(incoming, supersedes=existing.fact_id)
                    changed += 1
                facts = [fact for fact in facts
                         if not (fact.object_id == incoming.object_id
                                 and fact.attribute == incoming.attribute
                                 and fact.valid_from > incoming.valid_from)]
                facts.append(incoming)
            if correction:
                if changed < 1:
                    raise ContractError("a dated correction must change at least one value")
            bindings = previous_bindings + [_binding(observation)]
            correction_count = state.correction_count + (1 if correction else 0)
            created_at = state.created_at
            episode_id, world_id = state.episode_id, state.world_id
            subject_id, branch_id, purpose = (state.subject_id, state.branch_id,
                                               state.purpose)
            context_removed, removal_at = state.context_removed, state.removal_at
        result = CommittedState(
            episode_id=episode_id, world_id=world_id, subject_id=subject_id,
            branch_id=branch_id, purpose=purpose, created_at=created_at,
            facts=tuple(facts), source_bindings=tuple(bindings),
            correction_count=correction_count, context_removed=context_removed,
            removal_at=removal_at,
        )
        return result.sealed()

    def answer(self, query: VisualQuery, state: CommittedState | None) -> str:
        if self.disabled or state is None:
            return UNKNOWN_ANSWER
        facts = [fact for fact in state.facts
                 if fact.object_id == query.target_object_id
                 and fact.attribute == query.attribute
                 and fact.valid_from <= query.asked_at]
        if not facts:
            return UNKNOWN_ANSWER
        latest = max(facts, key=lambda fact: (fact.valid_from, fact.fact_id))
        return latest.value

    @staticmethod
    def zero_state(state: CommittedState) -> CommittedState:
        return replace(state, facts=(), source_bindings=(),
                       correction_count=0, state_sha256="").sealed()

    @staticmethod
    def shuffled_state(state: CommittedState, other: CommittedState) -> CommittedState:
        if state.episode_id == other.episode_id:
            raise ContractError("a shuffled state must come from another episode")
        return replace(
            state, facts=other.facts, source_bindings=other.source_bindings,
            correction_count=other.correction_count, state_sha256="",
        ).sealed()


def _current_facts(facts: Sequence[StateFact], as_of: int) -> dict[tuple[str, str], StateFact]:
    current: dict[tuple[str, str], StateFact] = {}
    for fact in facts:
        if fact.valid_from > as_of:
            continue
        key = (fact.object_id, fact.attribute)
        existing = current.get(key)
        if existing is None or (fact.valid_from, fact.fact_id) > (existing.valid_from,
                                                                  existing.fact_id):
            current[key] = fact
    return current


# ---------------------------------------------------------------------------
# Persisted journal and restart. State publication is atomic; the journal is a
# hash chain. A dangling commit intent or a partial trailing line is reported as
# an interrupted restart and the last fully published state is used.
# ---------------------------------------------------------------------------

ZERO_HASH = "0" * 64
JOURNAL_FIELDS = frozenset({"seq", "event", "payload", "prev", "hash"})


def _journal_record(seq: int, event: str, payload: Mapping[str, Any],
                    prev: str) -> dict[str, Any]:
    body = {"seq": seq, "event": event, "payload": dict(payload), "prev": prev}
    body["hash"] = fingerprint(body)
    return body


def _journal_bytes(record: Mapping[str, Any]) -> bytes:
    return (_json_bytes(record))


def _load_journal(path: Path) -> tuple[list[dict[str, Any]], str, int]:
    if not path.is_file():
        raise JournalError("journal is missing")
    data = path.read_bytes()
    truncated = 0
    if data and not data.endswith(b"\n"):
        cut = data.rfind(b"\n")
        truncated = len(data) if cut < 0 else len(data) - cut - 1
        data = data if cut < 0 else data[:cut + 1]
    entries: list[dict[str, Any]] = []
    prev = ZERO_HASH
    for index, line in enumerate(data.splitlines()):
        if not line.strip():
            raise JournalError("journal contains an empty line")
        try:
            record = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as error:
            raise JournalError("journal line is not valid JSON") from error
        _exact_keys(record, JOURNAL_FIELDS, "journal record")
        if (canonical_json(record) + "\n").encode("utf-8") != line + b"\n":
            raise JournalError("journal line is not canonically encoded")
        if record["seq"] != index:
            raise JournalError("journal sequence is not contiguous")
        if record["prev"] != prev:
            raise JournalError("journal previous hash differs")
        if record["hash"] != fingerprint({key: record[key] for key in
                                          ("seq", "event", "payload", "prev")}):
            raise JournalError("journal record hash differs")
        entries.append(record)
        prev = record["hash"]
    return entries, prev, truncated


def _append_journal(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as handle:
        handle.write(_journal_bytes(record))
        handle.flush()
        os.fsync(handle.fileno())


@dataclass(frozen=True, slots=True)
class RestartReport:
    """Mechanical restart evidence, deliberately separate from answer quality."""

    interrupted: bool
    pending_intents: tuple[str, ...]
    truncated_tail_bytes: int
    journal_entries: int
    journal_head: str
    state_sha256: str
    encode_calls: int
    reencoded_assets: int
    context_removed: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "interrupted": self.interrupted,
            "pending_intents": list(self.pending_intents),
            "truncated_tail_bytes": self.truncated_tail_bytes,
            "journal_entries": self.journal_entries,
            "journal_head": self.journal_head,
            "state_sha256": self.state_sha256,
            "encode_calls": self.encode_calls,
            "reencoded_assets": self.reencoded_assets,
            "context_removed": self.context_removed,
        }


@dataclass(frozen=True, slots=True)
class Answer:
    """One arm's answer. It never carries an expected or evaluator label."""

    query_id: str
    episode_id: str
    arm: str
    text: str
    answered: bool
    source_kind: str
    retrieval_performed: bool
    source_bytes_returned: int
    image_reads: int
    annotation_reads: int
    context_reads: int
    state_reads: int
    state_sha256: str
    source_observation_id: str
    source_epoch: int
    stale_rejected: bool
    adapter_kind: str

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "answer_format": ANSWER_FORMAT,
            "query_id": self.query_id,
            "episode_id": self.episode_id,
            "arm": self.arm,
            "text": self.text,
            "answered": self.answered,
            "source_kind": self.source_kind,
            "retrieval_performed": self.retrieval_performed,
            "source_bytes_returned": self.source_bytes_returned,
            "image_reads": self.image_reads,
            "annotation_reads": self.annotation_reads,
            "context_reads": self.context_reads,
            "state_reads": self.state_reads,
            "state_sha256": self.state_sha256,
            "source_observation_id": self.source_observation_id,
            "source_epoch": self.source_epoch,
            "stale_rejected": self.stale_rejected,
            "adapter_kind": self.adapter_kind,
        }
        _check_label_boundary(payload, "answer")
        return payload


class EpisodeRuntime:
    """One CPU episode with committed state, a journal and a removal boundary."""

    STATE_FILE = "state.json"
    META_FILE = "runtime.json"
    JOURNAL_FILE = "journal.jsonl"

    def __init__(self, root: str | Path, episode: Episode, *,
                 adapter: ReferenceContractAdapter | None = None,
                 state: CommittedState | None = None,
                 journal_entries: list[dict[str, Any]] | None = None,
                 journal_head: str = ZERO_HASH) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.episode = episode
        self.adapter = adapter if adapter is not None else ReferenceContractAdapter()
        self._state = state
        self._context_observations: dict[str, VisualObservation] = {}
        self._context_images: dict[str, bytes] = {}
        self._context_removed = bool(state.context_removed) if state else False
        self._pending_encoding: ObservedEncoding | None = None
        self._journal_entries = list(journal_entries or [])
        self._journal_head = journal_head
        self._restart_report: RestartReport | None = None

    # -------------------------------------------------------------- creation
    @classmethod
    def create(cls, root: str | Path, episode: Episode, *,
               adapter: ReferenceContractAdapter | None = None) -> "EpisodeRuntime":
        runtime = cls(root, episode, adapter=adapter)
        _atomic_write(runtime._meta_path, _json_bytes(runtime._meta()))
        (runtime.root / runtime.JOURNAL_FILE).write_bytes(b"")
        runtime._append("create", {
            "episode_id": episode.episode_id,
            "world_id": episode.world_id,
            "initial_observation_id": episode.initial.observation_id,
            "correction_observation_id": episode.correction.observation_id,
        })
        return runtime

    @property
    def _meta_path(self) -> Path:
        return self.root / self.META_FILE

    @property
    def _state_path(self) -> Path:
        return self.root / self.STATE_FILE

    @property
    def _journal_path(self) -> Path:
        return self.root / self.JOURNAL_FILE

    @property
    def state(self) -> CommittedState | None:
        return self._state

    @property
    def context_removed(self) -> bool:
        return self._context_removed

    @property
    def restart_report(self) -> RestartReport | None:
        return self._restart_report

    def _meta(self) -> dict[str, Any]:
        return {
            "format": "replica.visual_memory_episodes.runtime.v1",
            "episode_id": self.episode.episode_id,
            "world_id": self.episode.world_id,
            "subject_id": self.episode.subject_id,
            "branch_id": self.episode.branch_id,
            "purpose": self.episode.purpose,
            "context_removed": self._context_removed,
            "state_sha256": self._state.state_sha256 if self._state else "",
        }

    def _append(self, event: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        record = _journal_record(len(self._journal_entries), event, payload,
                                 self._journal_head)
        _append_journal(self._journal_path, record)
        self._journal_entries.append(record)
        self._journal_head = record["hash"]
        return record

    def _publish(self, state: CommittedState) -> None:
        data = state.to_bytes()
        _atomic_write(self._state_path, data)
        self._state = state
        _atomic_write(self._meta_path, _json_bytes(self._meta()))

    # ------------------------------------------------------------- observation
    def observe_initial(self, image_bytes: bytes) -> ObservedEncoding:
        if self._state is not None or self._pending_encoding is not None:
            raise ContractError("initial observation was already supplied")
        if self._context_removed:
            raise ContextRemovedError("cannot observe into removed context")
        encoding = self.adapter.encode_observation(self.episode.initial, image_bytes)
        self._pending_encoding = encoding
        self._context_observations[self.episode.initial.observation_id] = self.episode.initial
        self._context_images[self.episode.initial.observation_id] = bytes(image_bytes)
        self._append("observe", {
            "observation_id": self.episode.initial.observation_id,
            "image_sha256": self.episode.initial.image_sha256,
            "observed_at": self.episode.initial.observed_at,
        })
        return encoding

    def commit_initial(self) -> CommittedState:
        if self._pending_encoding is None:
            raise ContractError("there is no encoded initial observation to commit")
        state = self.adapter.commit(
            self._pending_encoding, None,
            subject_id=self.episode.subject_id,
            branch_id=self.episode.branch_id,
            purpose=self.episode.purpose,
        )
        self._pending_encoding = None
        self._publish(state)
        self._append("commit_state", {
            "state_sha256": state.state_sha256,
            "facts": len(state.facts),
            "intent_id": None,
        })
        return state

    def remove_context(self) -> None:
        if self._state is None:
            raise ContractError("state must be committed before context removal")
        if self._context_removed:
            raise ContextRemovedError("context is already removed")
        self._context_observations.clear()
        self._context_images.clear()
        self._pending_encoding = None
        self._context_removed = True
        state = replace(self._state, context_removed=True,
                        removal_at=self.episode.removal_at,
                        state_sha256="").sealed()
        self._publish(state)
        self._append("remove_context", {
            "removal_at": self.episode.removal_at,
            "state_sha256": state.state_sha256,
        })
        self._append("commit_state", {
            "state_sha256": state.state_sha256,
            "facts": len(state.facts),
            "intent_id": None,
            "removal": True,
        })

    def apply_correction(self, image_bytes: bytes) -> CommittedState:
        if self._state is None:
            raise ContractError("a correction requires committed state")
        if self._state.correction_count:
            raise ContractError("the episode correction was already applied")
        encoding = self.adapter.encode_observation(self.episode.correction, image_bytes)
        state = self.adapter.commit(
            encoding, self._state, correction=True,
            subject_id=self.episode.subject_id,
            branch_id=self.episode.branch_id,
            purpose=self.episode.purpose,
        )
        if not self._context_removed:
            self._context_observations[self.episode.correction.observation_id] = self.episode.correction
            self._context_images[self.episode.correction.observation_id] = bytes(image_bytes)
        self._publish(state)
        self._append("commit_state", {
            "state_sha256": state.state_sha256,
            "facts": len(state.facts),
            "intent_id": None,
            "correction": True,
            "observation_id": self.episode.correction.observation_id,
        })
        return state

    # ------------------------------------------------------------ context read
    @property
    def context_image_bytes(self) -> Mapping[str, bytes]:
        if self._context_removed or not self._context_images:
            raise ContextRemovedError("original visual context is not available")
        return dict(self._context_images)

    @property
    def context_annotations(self) -> Mapping[str, VisualObservation]:
        if self._context_removed or not self._context_observations:
            raise ContextRemovedError("original visual annotations are not available")
        return dict(self._context_observations)

    # ------------------------------------------------------------------ queries
    def ask(self, query: VisualQuery, arm: str = "memory", *,
            shuffled_state: CommittedState | None = None,
            exact_provider: "ExactSourceProvider | None" = None) -> Answer:
        if not isinstance(query, VisualQuery):
            raise ContractError("ask requires a VisualQuery")
        if query.episode_id != self.episode.episode_id:
            raise ContractError("query belongs to another episode")
        if arm not in ARMS:
            raise ContractError(f"unknown arm: {arm!r}")
        if arm in CONTEXT_FREE_ARMS:
            if self._context_removed is False and arm == "memory":
                # A memory-only query does not need context, but it may still be
                # asked before removal. It must never touch the context objects.
                pass
            state_reads = 0
            image_reads = 0
            annotation_reads = 0
            context_reads = 0
            state_sha = ""
            source_observation = ""
            source_epoch = 0
            retrieval = False
            source_bytes = 0
            source_kind = "none"
            if arm == "memory":
                if self._state is None:
                    raise ContractError("memory arm requires committed state")
                text = self.adapter.answer(query, self._state)
                state_sha = self._state.state_sha256
                state_reads = 1
                source_kind = "committed_state"
            elif arm == "zero_state":
                if self._state is None:
                    raise ContractError("zero-state arm requires committed state")
                zero = self.adapter.zero_state(self._state)
                text = self.adapter.answer(query, zero)
                state_sha = zero.state_sha256
                state_reads = 1
                source_kind = "zero_state"
            elif arm == "shuffled_state":
                if self._state is None or shuffled_state is None:
                    raise ContractError("shuffled-state arm requires both states")
                shuffled = self.adapter.shuffled_state(self._state, shuffled_state)
                text = self.adapter.answer(query, shuffled)
                state_sha = shuffled.state_sha256
                state_reads = 1
                source_kind = "shuffled_state"
            elif arm == "disabled":
                text = UNKNOWN_ANSWER
                source_kind = "disabled"
            else:  # context_removed: same information boundary, no state read
                text = UNKNOWN_ANSWER
                source_kind = "context_removed"
        elif arm == "full_context":
            observations = self.context_annotations
            text, chosen = _answer_from_observations(query, observations.values())
            source_kind = "retained_visual_context"
            image_reads = 1
            annotation_reads = 1
            context_reads = 1
            state_reads = 0
            state_sha = self._state.state_sha256 if self._state else ""
            source_observation = chosen
            source_epoch = 0
            retrieval = False
            source_bytes = 0
        else:  # exact_source
            if exact_provider is None:
                raise ContractError("exact-source arm requires an ExactSourceProvider")
            retrieval_result = exact_provider.retrieve(self.episode, query)
            text, chosen = _answer_from_observations(
                query, [retrieval_result.annotations])
            source_kind = "exact_source_retrieval"
            image_reads = 1
            annotation_reads = 1
            context_reads = 0
            state_reads = 0
            state_sha = ""
            source_observation = retrieval_result.observation_id
            source_epoch = retrieval_result.epoch
            retrieval = True
            source_bytes = retrieval_result.source_bytes_returned
        answer = Answer(
            query_id=query.query_id, episode_id=query.episode_id, arm=arm,
            text=text, answered=text != UNKNOWN_ANSWER, source_kind=source_kind,
            retrieval_performed=retrieval, source_bytes_returned=source_bytes,
            image_reads=image_reads, annotation_reads=annotation_reads,
            context_reads=context_reads, state_reads=state_reads,
            state_sha256=state_sha, source_observation_id=source_observation,
            source_epoch=source_epoch, stale_rejected=False,
            adapter_kind=self.adapter.adapter_kind,
        )
        self._append("query", {
            "query_id": query.query_id, "arm": arm, "text": text,
            "source_kind": source_kind, "image_reads": image_reads,
            "annotation_reads": annotation_reads, "context_reads": context_reads,
            "state_reads": state_reads, "retrieval_performed": retrieval,
        })
        return answer

    # --------------------------------------------------------------- restart
    def interrupt_publication(self) -> str:
        """Persist an uncommitted intent plus a partial temporary state file.

        This is a deterministic crash simulation used by the CPU checks. The
        published state and journal head are untouched, so ``restart`` must
        recover the last committed state and report the pending intent.
        """
        if self._state is None:
            raise ContractError("cannot interrupt without a committed state")
        intent_id = fingerprint({
            "episode_id": self.episode.episode_id,
            "state_sha256": self._state.state_sha256,
            "journal_head": self._journal_head,
        })[:24]
        temporary = self.root / f".{self.STATE_FILE}.tmp-interrupted"
        temporary.write_bytes(b'{"format":"replica.visual_memory_episodes.state.v1"')
        self._append("commit_intent", {
            "intent_id": intent_id,
            "proposed_state_sha256": self._state.state_sha256,
            "temporary_bytes": temporary.stat().st_size,
        })
        return intent_id

    @classmethod
    def restart(cls, root: str | Path) -> tuple["EpisodeRuntime", RestartReport]:
        root = Path(root)
        meta = _read_json(root / cls.META_FILE)
        expected_meta = {
            "format", "episode_id", "world_id", "subject_id", "branch_id",
            "purpose", "context_removed", "state_sha256",
        }
        _exact_keys(meta, expected_meta, "runtime metadata")
        if meta["format"] != "replica.visual_memory_episodes.runtime.v1":
            raise StateError("runtime metadata format is unsupported")
        state_data = (root / cls.STATE_FILE).read_bytes()
        state = CommittedState.from_bytes(state_data)
        if (state.episode_id, state.world_id) != (meta["episode_id"], meta["world_id"]):
            raise StateError("state and runtime metadata identity differ")
        if state.context_removed != meta["context_removed"]:
            raise StateError("state and runtime metadata removal flags differ")
        entries, head, truncated = _load_journal(root / cls.JOURNAL_FILE)
        if truncated:
            # Keep the valid prefix; the partial bytes are reported, not hidden.
            valid = b"".join(_journal_bytes(record) for record in entries)
            _atomic_write(root / cls.JOURNAL_FILE, valid)
            head, truncated = _load_journal(root / cls.JOURNAL_FILE)[1], truncated
        commits = {
            record["payload"].get("intent_id"): record
            for record in entries if record["event"] == "commit_state"
            and record["payload"].get("intent_id") is not None
        }
        intents = [record["payload"]["intent_id"] for record in entries
                   if record["event"] == "commit_intent"]
        pending = tuple(intent for intent in intents if intent not in commits)
        state_commits = [record for record in entries
                         if record["event"] == "commit_state"]
        if state_commits and state_commits[-1]["payload"]["state_sha256"] != state.state_sha256:
            raise JournalError("last committed journal state differs from the state file")
        for temporary in root.glob(f".{cls.STATE_FILE}.tmp*"):
            temporary.unlink(missing_ok=True)
        episode_placeholder = _EpisodeStub(
            episode_id=state.episode_id, world_id=state.world_id,
            subject_id=state.subject_id, branch_id=state.branch_id,
            purpose=state.purpose,
        )
        runtime = cls(root, episode_placeholder, state=state,
                      journal_entries=entries, journal_head=head)
        report = RestartReport(
            interrupted=bool(pending) or truncated > 0,
            pending_intents=pending, truncated_tail_bytes=truncated,
            journal_entries=len(entries), journal_head=head,
            state_sha256=state.state_sha256, encode_calls=runtime.adapter.encoded_calls,
            reencoded_assets=0, context_removed=state.context_removed,
        )
        runtime._restart_report = report
        return runtime, report

    def attach_episode(self, episode: Episode) -> None:
        """Bind the persisted runtime to its exact prepared episode for queries."""
        if (episode.episode_id, episode.world_id) != (
                self.episode.episode_id, self.episode.world_id):
            raise ContractError("restart episode identity differs")
        self.episode = episode


@dataclass(frozen=True, slots=True)
class _EpisodeStub:
    """Minimal metadata carrier for a restarted runtime before attach_episode."""

    episode_id: str
    world_id: str
    subject_id: str
    branch_id: str
    purpose: str


def _answer_from_observations(query: VisualQuery,
                              observations: Iterable[VisualObservation]) -> tuple[str, str]:
    candidates = [obs for obs in observations if obs.observed_at <= query.asked_at]
    if not candidates:
        return UNKNOWN_ANSWER, ""
    selected = max(candidates, key=lambda obs: (obs.observed_at, obs.observation_id))
    for obj in selected.objects:
        if obj.object_id != query.target_object_id:
            continue
        if query.attribute == "color":
            return obj.color, selected.observation_id
        if query.attribute == "shape":
            return obj.shape, selected.observation_id
    return UNKNOWN_ANSWER, selected.observation_id


# ---------------------------------------------------------------------------
# Explicit exact visual-source retrieval. This is an intentional source access,
# logged separately from compressed-state-only recall.
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class ExactRetrieval:
    query_id: str
    episode_id: str
    world_id: str
    observation_id: str
    image_sha256: str
    observed_at: int
    record_hash: str
    evidence_digest: str
    epoch: int
    source_bytes_returned: int
    annotations: VisualObservation
    retrieval_performed: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "format": RETRIEVAL_FORMAT,
            "query_id": self.query_id,
            "episode_id": self.episode_id,
            "world_id": self.world_id,
            "observation_id": self.observation_id,
            "image_sha256": self.image_sha256,
            "observed_at": self.observed_at,
            "record_hash": self.record_hash,
            "evidence_digest": self.evidence_digest,
            "epoch": self.epoch,
            "source_bytes_returned": self.source_bytes_returned,
            "retrieval_performed": self.retrieval_performed,
            "annotation_kind": "checked_procedural_renderer",
        }


class ExactSourceProvider:
    """Owned exact-source store over the existing VisualSourceMemory runtime."""

    def __init__(self, ledger_path: str | Path, blob_root: str | Path, *,
                 subject_id: str, purpose: str = "exact_visual_source",
                 branch_id: str = "source",
                 annotations: Mapping[str, VisualObservation] | None = None) -> None:
        self.subject_id = _identifier("subject_id", subject_id)
        self.purpose = _identifier("purpose", purpose)
        self.branch_id = _identifier("branch_id", branch_id)
        if self.branch_id != "source":
            raise ContractError("exact visual sources require the source branch")
        # Imported lazily: the ledger runtime's package imports torch even for
        # its sqlite path. Preparation and the contract-only tests must not pay
        # that import or suggest a native dependency.
        from .runtime import visual_source_memory as _visual_source_memory
        self._runtime = _visual_source_memory
        self.annotations: dict[str, VisualObservation] = dict(annotations or {})
        self.store = _visual_source_memory.VisualSourceMemory(ledger_path, blob_root)
        self.scope = _visual_source_memory.EligibilityScope(
            self.subject_id, self.branch_id, self.purpose, DEFAULT_AS_OF,
            "source_current",
        )
        self.receipts: dict[str, Any] = {}

    def close(self) -> None:
        self.store.close()

    def __enter__(self) -> "ExactSourceProvider":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def add_annotations(self, observations: Iterable[VisualObservation]) -> None:
        for observation in observations:
            self.annotations[observation.observation_id] = observation

    def admit_episode(self, episode: Episode, *,
                      initial_bytes: bytes, correction_bytes: bytes) -> dict[str, Any]:
        if episode.subject_id != self.subject_id or episode.purpose != self.purpose:
            raise ContractError("exact source scope differs from the episode")
        receipts = {}
        for observation, image_bytes in ((episode.initial, initial_bytes),
                                         (episode.correction, correction_bytes)):
            if _sha256_bytes(image_bytes) != observation.image_sha256:
                raise StaleVisualSourceError("admitted image bytes differ from the episode")
            receipt = self.store.observe(
                image_bytes,
                observation_id=observation.observation_id,
                subject_id=episode.subject_id,
                branch_id=self.branch_id,
                purpose=episode.purpose,
                source_id=observation.source_id,
                family_id=observation.family_id,
                independent_world_id=observation.independent_world_id,
                asset_id=observation.asset_id,
                owner_id=observation.owner_id,
                event_id=observation.observation_id,
                observed_at=observation.observed_at,
                event_at=observation.event_at,
                producer_id=observation.producer_id,
                crop_box=observation.crop_box,
                layout=self._runtime.SourceLayout(
                    observation.layout_template, observation.cell_indices),
                authorized_subject=episode.subject_id,
                authorized_branch=self.branch_id,
                source_authorized=True,
            )
            receipts[observation.observation_id] = receipt
            self.annotations[observation.observation_id] = observation
        self.receipts.update(receipts)
        return receipts

    def _latest_observation(self, episode: Episode, asked_at: int) -> Any:
        inventory = self.store.inventory(self.scope)
        candidates = [
            observation for observation in inventory.observations
            if observation.identity.independent_world_id == episode.world_id
            and observation.observed_at <= asked_at
        ]
        if not candidates:
            raise StaleVisualSourceError(
                "no live exact source exists at the requested date")
        return max(candidates,
                   key=lambda item: (item.observed_at, item.event_at,
                                     item.observation_id)), inventory.epoch

    def retrieve(self, episode: Episode, query: VisualQuery, *,
                 expected_observation_id: str | None = None,
                 expected_epoch: int | None = None,
                 epoch: int | None = None) -> ExactRetrieval:
        if query.episode_id != episode.episode_id:
            raise ContractError("query and episode identity differ")
        try:
            observation, current_epoch = self._latest_observation(episode, query.asked_at)
            if expected_epoch is not None and expected_epoch != current_epoch:
                raise StaleVisualSourceError("expected source epoch is stale")
            if (expected_observation_id is not None
                    and expected_observation_id != observation.observation_id):
                raise StaleVisualSourceError(
                    "source reference is stale; a later eligible view exists")
            request = self._runtime.VisualSourceRequest(
                query.query_id, observation.observation_id, observation.identity)
            evidence = self.store.retrieve(request, self.scope,
                                          expected_epoch=epoch if epoch is not None
                                          else current_epoch)
        except StaleVisualSourceError:
            raise
        except (self._runtime.VisualSourceError,
                self._runtime.ContractError) as error:
            raise StaleVisualSourceError(f"exact source is unavailable: {error}") from error
        if _sha256_bytes(evidence.source_bytes) != evidence.identity.content_sha256:
            raise StaleVisualSourceError("retrieved source bytes changed")
        annotation = self.annotations.get(observation.observation_id)
        if annotation is None:
            raise StaleVisualSourceError("no checked source annotation is recorded")
        if (annotation.image_sha256 != evidence.identity.content_sha256
                or annotation.observation_id != observation.observation_id):
            raise StaleVisualSourceError("checked annotation differs from retrieved source")
        return ExactRetrieval(
            query_id=query.query_id, episode_id=episode.episode_id,
            world_id=episode.world_id, observation_id=observation.observation_id,
            image_sha256=evidence.identity.content_sha256,
            observed_at=observation.observed_at,
            record_hash=evidence.record_hash,
            evidence_digest=evidence.evidence_digest, epoch=evidence.epoch,
            source_bytes_returned=len(evidence.source_bytes),
            annotations=annotation,
        )


# ---------------------------------------------------------------------------
# Deterministic preparation
# ---------------------------------------------------------------------------

CONFIG_FIELDS = frozenset({
    "format", "declared_at_utc", "purpose", "hypothesis", "episode_contract",
    "seed", "splits", "evaluation_splits", "worlds_per_split",
    "objects_per_world", "image", "timeline", "subject_id", "branch_id",
    "memory_purpose", "producer_id", "source_id_prefix", "arms", "controls",
    "information_boundary", "access_policy", "native_gate", "limits",
    "mechanical_gates", "report_contract", "limitations",
})
IMAGE_FIELDS = frozenset({"width", "height", "layout_template", "margin",
                          "object_size"})
TIMELINE_FIELDS = frozenset({"initial_day", "initial_query_day",
                             "correction_day", "after_correction_day"})
CONTROL_FIELDS = frozenset(ARMS)
REPORT_CONTRACT_FIELDS = frozenset({
    "separate_actual_outcome", "separate_corrections", "separate_unknowns",
    "separate_provenance", "separate_storage_costs",
    "separate_mechanical_persistence", "native_evaluation_gated",
})
ACCESS_POLICY_FIELDS = frozenset(ARMS)


def validate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(config, Mapping):
        raise ContractError("configuration must be an object")
    _exact_keys(config, CONFIG_FIELDS, "configuration")
    if config["format"] != CONFIG_FORMAT:
        raise ContractError(f"configuration format must be {CONFIG_FORMAT}")
    _identifier("declared_at_utc", config["declared_at_utc"])
    _identifier("purpose", config["purpose"])
    _identifier("hypothesis", config["hypothesis"])
    if not isinstance(config["episode_contract"], Mapping):
        raise ContractError("episode_contract must be an object")
    _positive_int("seed", config["seed"])
    splits = tuple(config["splits"])
    if splits != SPLITS:
        raise ContractError(f"splits must be exactly {SPLITS}")
    evaluation = tuple(config["evaluation_splits"])
    if not evaluation or any(split not in splits for split in evaluation):
        raise ContractError("evaluation_splits must be a nonempty subset of splits")
    if tuple(dict.fromkeys(evaluation)) != evaluation:
        raise ContractError("evaluation_splits must be unique")
    worlds = config["worlds_per_split"]
    if not isinstance(worlds, Mapping) or set(worlds) != set(splits):
        raise ContractError("worlds_per_split must cover every split exactly")
    for split in splits:
        _positive_int(f"worlds_per_split[{split}]", worlds[split])
    if config["objects_per_world"] != 3:
        raise ContractError("this corpus requires exactly three objects per world")
    image = config["image"]
    _exact_keys(image, IMAGE_FIELDS, "image")
    _positive_int("image.width", image["width"])
    _positive_int("image.height", image["height"])
    _identifier("image.layout_template", image["layout_template"])
    _nonnegative_int("image.margin", image["margin"])
    _positive_int("image.object_size", image["object_size"])
    if image["width"] < 3 * image["object_size"] + 2 * image["margin"]:
        raise ContractError("image is too narrow for three non-overlapping objects")
    if image["height"] < image["object_size"] + 2 * image["margin"]:
        raise ContractError("image is too short for the declared object size")
    timeline = config["timeline"]
    _exact_keys(timeline, TIMELINE_FIELDS, "timeline")
    days = {key: _day(timeline[key]) for key in TIMELINE_FIELDS}
    if not (days["initial_day"] < days["initial_query_day"]
            <= days["correction_day"] < days["after_correction_day"]):
        raise ContractError(
            "timeline must satisfy initial < initial_query <= correction < after_correction")
    for name in ("subject_id", "branch_id", "memory_purpose", "producer_id",
                 "source_id_prefix"):
        _identifier(name, config[name])
    if config["branch_id"] != "source":
        raise ContractError("visual-memory source episodes require branch 'source'")
    if tuple(config["arms"]) != ARMS:
        raise ContractError(f"arms must be exactly {ARMS}")
    _exact_keys(config["controls"], CONTROL_FIELDS, "controls")
    if not isinstance(config["information_boundary"], Mapping):
        raise ContractError("information_boundary must be an object")
    if not isinstance(config["access_policy"], Mapping):
        raise ContractError("access_policy must be an object")
    _exact_keys(config["access_policy"], ACCESS_POLICY_FIELDS, "access_policy")
    native = config["native_gate"]
    if not isinstance(native, Mapping) or native.get("status") != "gated":
        raise ContractError("native_gate status must be 'gated'")
    if native.get("native_evaluated") is not False:
        raise ContractError("native_gate.native_evaluated must be false")
    if not native.get("required_evidence"):
        raise ContractError("native_gate must declare required evidence")
    limits = config["limits"]
    if not isinstance(limits, Mapping):
        raise ContractError("limits must be an object")
    resolved_limits = dict(DEFAULT_LIMITS)
    resolved_limits.update(limits)
    for name in ("max_episodes", "max_queries", "max_answers", "max_images",
                 "max_image_bytes", "max_state_bytes", "max_journal_bytes",
                 "max_wall_seconds"):
        _positive_int(f"limits.{name}", resolved_limits[name])
    for name in ("native_jobs", "optimizer_updates", "gradient_examples"):
        if resolved_limits[name] != 0:
            raise ContractError(f"limits.{name} must be zero; no native or training arm")
    gates = config["mechanical_gates"]
    if not isinstance(gates, Mapping) or set(gates) != set(DEFAULT_MECHANICAL_GATES):
        raise ContractError("mechanical_gates must declare every mechanical gate")
    if any(type(value) is not bool for value in gates.values()):
        raise ContractError("mechanical gates must be booleans")
    _exact_keys(config["report_contract"], REPORT_CONTRACT_FIELDS, "report_contract")
    if not isinstance(config["limitations"], list) or not config["limitations"]:
        raise ContractError("limitations must be a nonempty list")
    result = dict(config)
    result["splits"] = list(splits)
    result["evaluation_splits"] = list(evaluation)
    result["worlds_per_split"] = {split: int(worlds[split]) for split in splits}
    result["limits"] = resolved_limits
    result["controls"] = dict(config["controls"])
    result["access_policy"] = dict(config["access_policy"])
    result["native_gate"] = dict(native)
    result["mechanical_gates"] = dict(gates)
    result["report_contract"] = dict(config["report_contract"])
    total = sum(result["worlds_per_split"].values())
    if total > resolved_limits["max_episodes"]:
        raise ContractError("world inventory exceeds limits.max_episodes")
    if total * 2 > resolved_limits["max_images"]:
        raise ContractError("image inventory exceeds limits.max_images")
    if total * 8 > resolved_limits["max_queries"]:
        raise ContractError("query inventory exceeds limits.max_queries")
    return result


def load_config(path: str | Path) -> dict[str, Any]:
    return validate_config(json.loads(Path(path).read_text(encoding="utf-8")))


def _seed_rng(seed: int, *parts: str) -> Any:
    import random
    material = "|".join([str(seed), *parts]).encode("utf-8")
    return random.Random(int(hashlib.sha256(material).hexdigest()[:16], 16))


def _scene(config: Mapping[str, Any], asset_id: str,
           objects: Sequence[ObjectObservation]) -> dict[str, Any]:
    image = config["image"]
    return {
        "scene_id": asset_id,
        "width": image["width"],
        "height": image["height"],
        "layout": {
            "template": image["layout_template"],
            "cell_indices": list(range(len(objects))),
        },
        "objects": [
            {
                "id": obj.object_id,
                "shape": obj.shape,
                "color": obj.color,
                "bbox": list(obj.region.bbox),
                "center": list(obj.region.center),
            }
            for obj in objects
        ],
    }


def _world_episode(config: Mapping[str, Any], split: str, index: int) -> tuple[
        Episode, list[VisualQuery], list[EpisodeLabel], dict[str, bytes], dict[str, Any]]:
    seed = config["seed"]
    rng = _seed_rng(seed, "world", split, str(index))
    image = config["image"]
    width, height = image["width"], image["height"]
    margin, size = image["margin"], image["object_size"]
    count = config["objects_per_world"]
    centers = []
    usable = width - 2 * margin
    for position in range(count):
        cx = margin + (usable * (2 * position + 1)) // (2 * count)
        centers.append((cx, height // 2))
    shapes = rng.sample(list(SHAPES), count)
    colors = rng.sample(list(COLOR_NAMES), count)
    objects = []
    for position in range(count):
        cx, cy = centers[position]
        x0, y0 = cx - size // 2, cy - size // 2
        region = Region((x0, y0, x0 + size, y0 + size), (cx, cy))
        objects.append(ObjectObservation(
            object_id=f"o{position}", region_id=f"r{position}", region=region,
            color=colors[position], shape=shapes[position],
        ))
    episode_id = f"vm-{split}-{index:02d}"
    world_id = f"world-{split}-{index:02d}"
    family_id = f"family-{split}-{index:02d}"
    initial_asset = f"{episode_id}-initial"
    correction_asset = f"{episode_id}-correction"
    initial_id = f"{episode_id}-obs-initial"
    correction_id = f"{episode_id}-obs-correction"
    days = {key: _day(config["timeline"][key]) for key in TIMELINE_FIELDS}
    changed_index = index % count
    changed_object = objects[changed_index]
    remaining_colors = [color for color in COLOR_NAMES if color not in colors]
    new_color = rng.choice(remaining_colors)
    corrected_objects = [
        replace(obj, color=new_color) if obj.object_id == changed_object.object_id
        else obj for obj in objects
    ]
    initial_scene = _scene(config, initial_asset, objects)
    correction_scene = _scene(config, correction_asset, corrected_objects)
    initial_bytes = render_scene_png(initial_scene)
    correction_bytes = render_scene_png(correction_scene)
    if initial_bytes == correction_bytes:
        raise ContractError("a correction must change rendered pixels")
    initial_sha = _sha256_bytes(initial_bytes)
    correction_sha = _sha256_bytes(correction_bytes)
    initial = VisualObservation(
        observation_id=initial_id, episode_id=episode_id, world_id=world_id,
        asset_id=initial_asset, image_sha256=initial_sha, width=width,
        height=height, byte_length=len(initial_bytes),
        observed_at=days["initial_day"], event_at=days["initial_day"],
        observed_date=_iso_day(days["initial_day"]),
        event_date=_iso_day(days["initial_day"]),
        producer_id=config["producer_id"],
        source_id=f"{config['source_id_prefix']}:{split}:{index}:initial",
        family_id=family_id, independent_world_id=world_id, owner_id=episode_id,
        crop_box=None, layout_template=image["layout_template"],
        cell_indices=tuple(range(count)), objects=tuple(objects),
    )
    correction = VisualObservation(
        observation_id=correction_id, episode_id=episode_id, world_id=world_id,
        asset_id=correction_asset, image_sha256=correction_sha, width=width,
        height=height, byte_length=len(correction_bytes),
        observed_at=days["correction_day"], event_at=days["correction_day"],
        observed_date=_iso_day(days["correction_day"]),
        event_date=_iso_day(days["correction_day"]),
        producer_id=config["producer_id"],
        source_id=f"{config['source_id_prefix']}:{split}:{index}:correction",
        family_id=family_id, independent_world_id=world_id, owner_id=episode_id,
        crop_box=None, layout_template=image["layout_template"],
        cell_indices=tuple(range(count)), objects=tuple(corrected_objects),
        supersedes_observation_id=initial_id,
    )
    episode = Episode(
        episode_id=episode_id, subject_id=config["subject_id"],
        branch_id=config["branch_id"], purpose=config["memory_purpose"],
        world_id=world_id, removal_at=days["initial_query_day"],
        removal_date=_iso_day(days["initial_query_day"]),
        initial=initial, correction=correction,
    )
    colors_by_object = {obj.object_id: obj.color for obj in objects}
    shapes_by_object = {obj.object_id: obj.shape for obj in objects}
    after_objects = {obj.object_id: obj for obj in corrected_objects}
    unaffected = [obj for obj in objects if obj.object_id != changed_object.object_id]
    first_unaffected = unaffected[0]
    second_unaffected = unaffected[1]
    query_rows: list[VisualQuery] = []
    labels: list[EpisodeLabel] = []

    def query(qid: str, text: str, day: int, target: str, attribute: str) -> VisualQuery:
        return VisualQuery(
            query_id=qid, episode_id=episode_id, world_id=world_id, text=text,
            asked_at=day, asked_date=_iso_day(day), target_object_id=target,
            attribute=attribute,
        )

    def label(qid: str, expected: str, kind: str, split: str, phase: str,
              target: str, attribute: str, observation: VisualObservation) -> EpisodeLabel:
        return EpisodeLabel(
            query_id=qid, episode_id=episode_id, expected_answer=expected,
            query_kind=kind, split=split, phase=phase,
            target_object_id=target, attribute=attribute,
            expected_observation_id=observation.observation_id,
            expected_observation_sha256=observation.image_sha256,
        )

    q1 = query(f"{episode_id}-q01", f"What color is object {changed_object.object_id}?",
               days["initial_query_day"], changed_object.object_id, "color")
    q2 = query(f"{episode_id}-q02", f"What shape is object {first_unaffected.object_id}?",
               days["initial_query_day"], first_unaffected.object_id, "shape")
    q3 = query(f"{episode_id}-q03", f"What color is object {second_unaffected.object_id}?",
               days["initial_query_day"], second_unaffected.object_id, "color")
    q4 = query(f"{episode_id}-q04", f"What color is object {changed_object.object_id}?",
               days["after_correction_day"], changed_object.object_id, "color")
    q5 = query(f"{episode_id}-q05", f"What shape is object {first_unaffected.object_id}?",
               days["after_correction_day"], first_unaffected.object_id, "shape")
    q6 = query(f"{episode_id}-q06", f"What color is object {second_unaffected.object_id}?",
               days["after_correction_day"], second_unaffected.object_id, "color")
    q7 = query(f"{episode_id}-q07", f"What color is object {UNKNOWN_OBJECT_ID}?",
               days["after_correction_day"], UNKNOWN_OBJECT_ID, "color")
    q8 = query(f"{episode_id}-q08", f"What shape is object {UNKNOWN_OBJECT_ID}?",
               days["after_correction_day"], UNKNOWN_OBJECT_ID, "shape")
    query_rows.extend((q1, q2, q3, q4, q5, q6, q7, q8))
    labels.extend((
        label(q1.query_id, colors_by_object[changed_object.object_id], "initial",
              split, "initial", changed_object.object_id, "color", initial),
        label(q2.query_id, shapes_by_object[first_unaffected.object_id], "initial",
              split, "initial", first_unaffected.object_id, "shape", initial),
        label(q3.query_id, colors_by_object[second_unaffected.object_id], "initial",
              split, "initial", second_unaffected.object_id, "color", initial),
        label(q4.query_id, after_objects[changed_object.object_id].color, "corrected",
              split, "after_correction", changed_object.object_id, "color", correction),
        label(q5.query_id, shapes_by_object[first_unaffected.object_id], "unaffected",
              split, "after_correction", first_unaffected.object_id, "shape", correction),
        label(q6.query_id, colors_by_object[second_unaffected.object_id], "unaffected",
              split, "after_correction", second_unaffected.object_id, "color", correction),
        label(q7.query_id, UNKNOWN_ANSWER, "unknown", split, "after_correction",
              UNKNOWN_OBJECT_ID, "color", correction),
        label(q8.query_id, UNKNOWN_ANSWER, "unknown", split, "after_correction",
              UNKNOWN_OBJECT_ID, "shape", correction),
    ))
    world_summary = {
        "episode_id": episode_id,
        "world_id": world_id,
        "split": split,
        "family_id": family_id,
        "changed_object_id": changed_object.object_id,
        "initial_observation_id": initial_id,
        "correction_observation_id": correction_id,
        "initial_image": f"images/{initial_asset}.png",
        "correction_image": f"images/{correction_asset}.png",
        "initial_image_sha256": initial_sha,
        "correction_image_sha256": correction_sha,
        "query_count": len(query_rows),
        "labels": {
            "initial": 3,
            "corrected": 1,
            "unaffected": 2,
            "unknown": 2,
        },
    }
    images = {
        f"images/{initial_asset}.png": initial_bytes,
        f"images/{correction_asset}.png": correction_bytes,
    }
    return episode, query_rows, labels, images, world_summary


def _source_annotation_rows(episodes: Sequence[Episode]) -> list[dict[str, Any]]:
    rows = []
    for episode in episodes:
        for observation in episode.observations:
            rows.append({
                "observation_id": observation.observation_id,
                "episode_id": observation.episode_id,
                "world_id": observation.world_id,
                "image_sha256": observation.image_sha256,
                "observed_at": observation.observed_at,
                "annotation_kind": "checked_procedural_renderer",
                "objects": [obj.as_dict() for obj in observation.objects],
                "provenance": {
                    "source_id": observation.source_id,
                    "family_id": observation.family_id,
                    "independent_world_id": observation.independent_world_id,
                    "asset_id": observation.asset_id,
                    "producer_id": observation.producer_id,
                    "owner_id": observation.owner_id,
                    "observed_at": observation.observed_at,
                    "event_at": observation.event_at,
                    "crop_box": list(observation.crop_box)
                    if observation.crop_box is not None else None,
                    "layout_template": observation.layout_template,
                    "cell_indices": list(observation.cell_indices),
                },
            })
    return rows


def generate_episode_corpus(config: Mapping[str, Any]) -> tuple[
        list[Episode], list[VisualQuery], list[EpisodeLabel],
        dict[str, bytes], list[dict[str, Any]], list[dict[str, Any]]]:
    config = validate_config(config)
    episodes: list[Episode] = []
    queries: list[VisualQuery] = []
    labels: list[EpisodeLabel] = []
    images: dict[str, bytes] = {}
    worlds: list[dict[str, Any]] = []
    for split in config["splits"]:
        for index in range(config["worlds_per_split"][split]):
            episode, episode_queries, episode_labels, episode_images, summary = (
                _world_episode(config, split, index))
            episodes.append(episode)
            queries.extend(episode_queries)
            labels.extend(episode_labels)
            images.update(episode_images)
            worlds.append(summary)
    return episodes, queries, labels, images, worlds, _source_annotation_rows(episodes)


def _interface_contract() -> dict[str, Any]:
    return {
        "episode": "Episode",
        "observation": "VisualObservation",
        "query": "VisualQuery",
        "label": "EpisodeLabel (evaluator-only; separate file)",
        "state": "CommittedState",
        "fact": "StateFact",
        "adapter": "ReferenceContractAdapter",
        "exact_source": "ExactSourceProvider over VisualSourceMemory",
        "runtime": "EpisodeRuntime",
        "answer": "Answer (no expected field)",
        "restart": "RestartReport",
        "state_fields": [
            "object_id", "region_id", "bbox", "center", "attribute", "value",
            "valid_from", "event_at", "observation_id", "supersedes",
            "image_sha256", "source_id", "producer_id",
        ],
        "answer_fields": sorted(ANSWER_FIELDS),
        "label_fields": sorted(LABEL_FIELDS),
        "query_fields": sorted(QUERY_FIELDS),
    }


def _protocol(config: Mapping[str, Any], *, config_sha256: str,
              implementation_sha256: str, data_pins: Mapping[str, Any],
              counts: Mapping[str, int]) -> dict[str, Any]:
    return {
        "format": PROTOCOL_FORMAT,
        "version": VERSION,
        "declared_at_utc": config["declared_at_utc"],
        "purpose": config["purpose"],
        "hypothesis": config["hypothesis"],
        "episode_contract": dict(config["episode_contract"]),
        "interfaces": _interface_contract(),
        "arms": list(ARMS),
        "controls": dict(config["controls"]),
        "access_policy": dict(config["access_policy"]),
        "information_boundary": dict(config["information_boundary"]),
        "native_gate": dict(config["native_gate"]),
        "mechanical_gates": dict(config["mechanical_gates"]),
        "report_contract": dict(config["report_contract"]),
        "limits": dict(config["limits"]),
        "counts": dict(counts),
        "data_pins": dict(data_pins),
        "config_sha256": config_sha256,
        "implementation_sha256": implementation_sha256,
        "implementation_paths": list(IMPLEMENTATION_PATHS),
        "limitations": list(config["limitations"]),
    }


def _fresh_directory(path: Path) -> None:
    if path.exists():
        if not path.is_dir() or any(path.iterdir()):
            raise ContractError(f"preparation output must be fresh: {path}")


def prepare_visual_memory(output: str | Path, *, config_path: str | Path,
                          config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    config_path = Path(config_path)
    raw_config = config_path.read_bytes()
    if config is None:
        config = json.loads(raw_config.decode("utf-8"))
    config = validate_config(config)
    output = Path(output)
    _fresh_directory(output)
    output.mkdir(parents=True, exist_ok=True)
    episodes, queries, labels, images, worlds, source_rows = generate_episode_corpus(config)
    counts = {
        "episodes": len(episodes),
        "queries": len(queries),
        "labels": len(labels),
        "source_annotations": len(source_rows),
        "worlds": len(worlds),
        "images": len(images),
        "image_bytes": sum(len(value) for value in images.values()),
    }
    if counts["episodes"] > config["limits"]["max_episodes"]:
        raise ContractError("episode count exceeds the declared limit")
    if counts["queries"] > config["limits"]["max_queries"]:
        raise ContractError("query count exceeds the declared limit")
    if counts["image_bytes"] > config["limits"]["max_image_bytes"]:
        raise ContractError("image bytes exceed the declared limit")
    _write_exact(output / "config.json", raw_config)
    episodes_bytes = _jsonl_bytes(episode.as_dict() for episode in episodes)
    queries_bytes = _jsonl_bytes(query.as_dict() for query in queries)
    labels_bytes = _jsonl_bytes(label.as_dict() for label in labels)
    annotations_bytes = _jsonl_bytes(source_rows)
    _write_exact(output / "episodes.jsonl", episodes_bytes)
    _write_exact(output / "queries.jsonl", queries_bytes)
    _write_exact(output / "labels.jsonl", labels_bytes)
    _write_exact(output / "source_annotations.jsonl", annotations_bytes)
    for relative, data in images.items():
        _write_exact(output / relative, data)
    config_sha = _sha256_bytes(raw_config)
    data_pins = {
        "episodes_sha256": _sha256_bytes(episodes_bytes),
        "queries_sha256": _sha256_bytes(queries_bytes),
        "labels_sha256": _sha256_bytes(labels_bytes),
        "source_annotations_sha256": _sha256_bytes(annotations_bytes),
        "images": {
            relative: _sha256_bytes(data) for relative, data in sorted(images.items())
        },
    }
    implementation_sha = _implementation_sha256()
    protocol = _protocol(
        config, config_sha256=config_sha,
        implementation_sha256=implementation_sha, data_pins=data_pins,
        counts=counts)
    protocol_bytes = _json_bytes(protocol)
    _write_exact(output / "protocol.json", protocol_bytes)
    files = {
        "config.json": config_sha,
        "protocol.json": _sha256_bytes(protocol_bytes),
        "episodes.jsonl": data_pins["episodes_sha256"],
        "queries.jsonl": data_pins["queries_sha256"],
        "labels.jsonl": data_pins["labels_sha256"],
        "source_annotations.jsonl": data_pins["source_annotations_sha256"],
        **data_pins["images"],
    }
    manifest = {
        "format": MANIFEST_FORMAT,
        "version": VERSION,
        "declared_at_utc": config["declared_at_utc"],
        "config_sha256": config_sha,
        "protocol_sha256": files["protocol.json"],
        "implementation_sha256": implementation_sha,
        "seed": config["seed"],
        "splits": list(config["splits"]),
        "evaluation_splits": list(config["evaluation_splits"]),
        "counts": counts,
        "files": files,
        "worlds": worlds,
        "image_inventory": [
            {
                "path": relative,
                "sha256": _sha256_bytes(data),
                "byte_length": len(data),
            }
            for relative, data in sorted(images.items())
        ],
        "limitations": list(config["limitations"]),
    }
    _write_exact(output / "manifest.json", _json_bytes(manifest))
    return manifest


def _load_corpus(prepared: Path) -> tuple[
        dict[str, Any], dict[str, Any], list[Episode], list[VisualQuery],
        list[EpisodeLabel], list[dict[str, Any]], dict[str, VisualObservation]]:
    manifest = _read_json(prepared / "manifest.json")
    protocol = _read_json(prepared / "protocol.json")
    episodes = [Episode.from_dict(row) for row in _read_jsonl(prepared / "episodes.jsonl")]
    queries = [VisualQuery.from_dict(row) for row in _read_jsonl(prepared / "queries.jsonl")]
    labels = [EpisodeLabel.from_dict(row) for row in _read_jsonl(prepared / "labels.jsonl")]
    annotations = _read_jsonl(prepared / "source_annotations.jsonl")
    for row in annotations:
        _exact_keys(row, SOURCE_ANNOTATION_FIELDS, "source annotation")
    return (manifest, protocol, episodes, queries, labels, annotations,
            _episode_observation_map(episodes))


def _episode_observation_map(episodes: Sequence[Episode]) -> dict[str, VisualObservation]:
    return {observation.observation_id: observation
            for episode in episodes for observation in episode.observations}


def verify_visual_memory_prepared(prepared: str | Path) -> dict[str, Any]:
    prepared = Path(prepared)
    manifest, protocol, episodes, queries, labels, annotations, annotation_map = (
        _load_corpus(prepared))
    if manifest.get("format") != MANIFEST_FORMAT:
        raise ContractError("prepared manifest format is unsupported")
    if protocol.get("format") != PROTOCOL_FORMAT:
        raise ContractError("protocol format is unsupported")
    if protocol.get("native_gate") != NATIVE_GATE:
        raise ContractError("protocol native gate differs from the declared gated contract")
    for relative, digest in manifest["files"].items():
        if not _is_sha256(digest):
            raise ContractError(f"manifest hash is invalid for {relative}")
        path = prepared / relative
        if not path.is_file() or _sha256_file(path) != digest:
            raise ContractError(f"prepared file hash differs: {relative}")
    if _sha256_file(prepared / "config.json") != manifest["config_sha256"]:
        raise ContractError("config hash differs from the manifest")
    if manifest["implementation_sha256"] != _implementation_sha256():
        raise ContractError("implementation hash differs from the manifest")
    if protocol["implementation_sha256"] != manifest["implementation_sha256"]:
        raise ContractError("protocol implementation hash differs")
    if protocol["config_sha256"] != manifest["config_sha256"]:
        raise ContractError("protocol config hash differs")
    if protocol["counts"] != manifest["counts"]:
        raise ContractError("protocol counts differ from the manifest")
    data_pins = protocol["data_pins"]
    for name in ("episodes_sha256", "queries_sha256", "labels_sha256",
                 "source_annotations_sha256"):
        relative = name.replace("_sha256", ".jsonl")
        if _sha256_file(prepared / relative) != data_pins[name]:
            raise ContractError(f"protocol data pin differs: {name}")
    if data_pins["images"] != {
            item["path"]: item["sha256"] for item in manifest["image_inventory"]}:
        raise ContractError("protocol image pins differ from the manifest")
    counts = {
        "episodes": len(episodes),
        "queries": len(queries),
        "labels": len(labels),
        "source_annotations": len(annotations),
        "worlds": len(manifest["worlds"]),
        "images": len(manifest["image_inventory"]),
        "image_bytes": sum(item["byte_length"]
                           for item in manifest["image_inventory"]),
    }
    for name, value in counts.items():
        if manifest["counts"].get(name) != value:
            raise ContractError(f"prepared count differs: {name}")
    if len(queries) != len(labels):
        raise ContractError("each query requires exactly one evaluator label")
    label_by_query = {label.query_id: label for label in labels}
    if len(label_by_query) != len(labels):
        raise ContractError("duplicate evaluator label")
    query_by_id = {query.query_id: query for query in queries}
    if len(query_by_id) != len(queries):
        raise ContractError("duplicate query identity")
    if set(label_by_query) != set(query_by_id):
        raise ContractError("label and query inventories differ")
    for episode_id in {label.episode_id for label in labels}:
        kinds = Counter(label.query_kind for label in labels
                        if label.episode_id == episode_id)
        if (kinds["initial"] != INITIAL_QUERY_COUNT
                or kinds["corrected"] != 1
                or kinds["unaffected"] != 2
                or kinds["unknown"] != 2
                or sum(kinds.values()) != INITIAL_QUERY_COUNT + POST_CORRECTION_QUERY_COUNT):
            raise ContractError("episode query-kind inventory differs from the contract")
    episode_by_id = {episode.episode_id: episode for episode in episodes}
    if len(episode_by_id) != len(episodes):
        raise ContractError("duplicate episode identity")
    world_ids = [episode.world_id for episode in episodes]
    asset_ids = [observation.asset_id for episode in episodes
                 for observation in episode.observations]
    observation_ids = [observation.observation_id for episode in episodes
                       for observation in episode.observations]
    if len(set(world_ids)) != len(world_ids):
        raise ContractError("world identities are not unique")
    if len(set(asset_ids)) != len(asset_ids):
        raise ContractError("asset identities are not unique")
    if len(set(observation_ids)) != len(observation_ids):
        raise ContractError("observation identities are not unique")
    splits_by_world: dict[str, str] = {}
    for episode in episodes:
        if episode.subject_id != _read_json(prepared / "config.json")["subject_id"]:
            raise ContractError("episode subject differs from the configuration")
        if episode.purpose != _read_json(prepared / "config.json")["memory_purpose"]:
            raise ContractError("episode purpose differs from the configuration")
        world_row = next(
            world for world in manifest["worlds"]
            if world["episode_id"] == episode.episode_id)
        split = world_row["split"]
        splits_by_world[episode.world_id] = split
        if world_row["world_id"] != episode.world_id:
            raise ContractError("manifest world identity differs")
        for observation in episode.observations:
            image_path = prepared / "images" / f"{observation.asset_id}.png"
            if not image_path.is_file():
                raise ContractError("prepared image is missing")
            data = image_path.read_bytes()
            if _sha256_bytes(data) != observation.image_sha256:
                raise ContractError("prepared image hash differs")
            if len(data) != observation.byte_length:
                raise ContractError("prepared image byte length differs")
            width, height, _rows = decode_rgb_png(data)
            if (width, height) != (observation.width, observation.height):
                raise ContractError("prepared image dimensions differ")
        for query in queries:
            if query.episode_id != episode.episode_id:
                continue
            label = label_by_query[query.query_id]
            if label.episode_id != episode.episode_id:
                raise ContractError("label episode identity differs")
            if label.split != split:
                raise ContractError("label split differs from world split")
            if label.phase == "initial":
                expected_obs = episode.initial
            elif label.phase == "after_correction":
                expected_obs = episode.correction
            else:
                raise ContractError("unknown label phase")
            if (label.expected_observation_id != expected_obs.observation_id
                    or label.expected_observation_sha256 != expected_obs.image_sha256):
                raise ContractError("label expected source binding differs")
            if label.query_kind == "unknown":
                if query.target_object_id in episode.initial.object_ids():
                    raise ContractError("unknown query targets an observed object")
                if label.expected_answer != UNKNOWN_ANSWER:
                    raise ContractError("unknown query answer must be 'unknown'")
            else:
                objects = {obj.object_id: obj for obj in expected_obs.objects}
                target = objects.get(query.target_object_id)
                if target is None:
                    raise ContractError("non-unknown query target is absent")
                expected_value = target.color if query.attribute == "color" else target.shape
                if label.expected_answer != expected_value:
                    raise ContractError("label answer differs from the dated observation")
        if episode.initial.observed_at >= episode.correction.observed_at:
            raise ContractError("correction is not later than the initial view")
    for split in SPLITS:
        split_worlds = {world for world, value in splits_by_world.items()
                        if value == split}
        other_worlds = {world for world, value in splits_by_world.items()
                        if value != split}
        if split_worlds & other_worlds:
            raise ContractError("world lineage leaks across splits")
    for row in annotations:
        if row["observation_id"] not in _episode_observation_map(episodes):
            raise ContractError("source annotation references an unknown observation")
        annotation = annotation_map[row["observation_id"]]
        original = _episode_observation_map(episodes)[row["observation_id"]]
        if annotation.objects != original.objects:
            raise ContractError("source annotation objects differ from the episode")
        if annotation.image_sha256 != original.image_sha256:
            raise ContractError("source annotation image differs from the episode")
        provenance = row["provenance"]
        expected_provenance = {
            "source_id": original.source_id,
            "family_id": original.family_id,
            "independent_world_id": original.independent_world_id,
            "asset_id": original.asset_id,
            "producer_id": original.producer_id,
            "owner_id": original.owner_id,
            "observed_at": original.observed_at,
            "event_at": original.event_at,
            "crop_box": list(original.crop_box)
            if original.crop_box is not None else None,
            "layout_template": original.layout_template,
            "cell_indices": list(original.cell_indices),
        }
        if provenance != expected_provenance:
            raise ContractError("source annotation provenance differs from the episode")
        if row["episode_id"] != original.episode_id or row["world_id"] != original.world_id:
            raise ContractError("source annotation episode/world differs")
        if row["observed_at"] != original.observed_at:
            raise ContractError("source annotation date differs")
    for episode in episodes:
        for observation in episode.observations:
            _check_label_boundary(episode.as_dict(), "episode")
            _check_label_boundary(observation.as_dict(), "observation")
    for query in queries:
        _check_label_boundary(query.as_dict(), "query")
    for row in annotations:
        _check_label_boundary(row, "source annotation")
    if manifest["splits"] != list(SPLITS):
        raise ContractError("manifest splits differ")
    return {
        "prepared_manifest_sha256": _sha256_file(prepared / "manifest.json"),
        "protocol_sha256": _sha256_file(prepared / "protocol.json"),
        "config_sha256": _sha256_file(prepared / "config.json"),
        "episodes_sha256": _sha256_file(prepared / "episodes.jsonl"),
        "queries_sha256": _sha256_file(prepared / "queries.jsonl"),
        "labels_sha256": _sha256_file(prepared / "labels.jsonl"),
        "source_annotations_sha256": _sha256_file(prepared / "source_annotations.jsonl"),
        "implementation_sha256": manifest["implementation_sha256"],
        "counts": counts,
    }


def admit_visual_memory_prepared(prepared: str | Path, *,
                                 write_receipt: bool = True) -> dict[str, Any]:
    prepared = Path(prepared)
    try:
        current = verify_visual_memory_prepared(prepared)
        receipt = {
            "format": ADMISSION_FORMAT,
            "status": "admitted",
            "verified_at_utc": datetime.now(timezone.utc).isoformat(),
            **current,
            "checks": [
                "file hashes match the manifest",
                "implementation and configuration pins match",
                "evaluator labels are separated from episode/query/source inputs",
                "object, region, coordinate and timestamp lineage is consistent",
                "the dated correction is later and changes exactly one colour",
                "unaffected and unknown expectations are separate",
                "world and asset identities are split-separated",
            ],
        }
    except Exception as error:
        receipt = {
            "format": ADMISSION_FORMAT,
            "status": "rejected",
            "verified_at_utc": datetime.now(timezone.utc).isoformat(),
            "reason": f"{type(error).__name__}: {error}",
        }
    if write_receipt:
        _atomic_write(prepared / "admission.json", _json_bytes(receipt))
    return receipt


# ---------------------------------------------------------------------------
# CPU evaluation and scoring. The report separates reference-adapter outcomes
# from mechanical persistence; it does not convert either into a quality claim.
# ---------------------------------------------------------------------------

def _normalize_answer(value: str) -> str:
    return " ".join(value.strip().lower().split())


def _dir_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _load_image(prepared: Path, observation: VisualObservation) -> bytes:
    data = (prepared / "images" / f"{observation.asset_id}.png").read_bytes()
    if _sha256_bytes(data) != observation.image_sha256:
        raise StaleVisualSourceError("prepared image differs from its observation")
    return data


def _build_reference_state(episode: Episode, prepared: Path, phase: str,
                           adapter: ReferenceContractAdapter) -> CommittedState:
    initial_bytes = _load_image(prepared, episode.initial)
    state = adapter.commit(
        adapter.encode_observation(episode.initial, initial_bytes), None,
        subject_id=episode.subject_id, branch_id=episode.branch_id,
        purpose=episode.purpose)
    if phase == "after_correction":
        correction_bytes = _load_image(prepared, episode.correction)
        state = adapter.commit(
            adapter.encode_observation(episode.correction, correction_bytes),
            state, correction=True, subject_id=episode.subject_id,
            branch_id=episode.branch_id, purpose=episode.purpose)
    return state


def _validate_answer_row(row: Mapping[str, Any]) -> None:
    _check_label_boundary(row, "answer")
    _exact_keys(row, ANSWER_FIELDS, "answer")
    if row["answer_format"] != ANSWER_FORMAT:
        raise ContractError("answer format is unsupported")
    if row["arm"] not in ARMS:
        raise ContractError("answer carries an unknown arm")


def score_visual_memory(answers: Iterable[Mapping[str, Any]],
                        labels: Iterable[EpisodeLabel | Mapping[str, Any]]) -> dict[str, Any]:
    """Score answers against separately supplied evaluator labels.

    Answer rows are accepted only under the declared answer schema. A missing,
    duplicate or substituted answer/label is an inventory error, not a score.
    """
    answer_rows = [dict(row) for row in answers]
    if not answer_rows:
        raise ContractError("scoring requires at least one answer")
    for row in answer_rows:
        _validate_answer_row(row)
    label_rows: list[EpisodeLabel] = []
    for label in labels:
        if isinstance(label, EpisodeLabel):
            label_rows.append(label)
        else:
            label_rows.append(EpisodeLabel.from_dict(label))
    if not label_rows:
        raise ContractError("scoring requires at least one evaluator label")
    by_query = {label.query_id: label for label in label_rows}
    if len(by_query) != len(label_rows):
        raise ContractError("duplicate evaluator label")
    seen: set[tuple[str, str]] = set()
    for row in answer_rows:
        key = (row["arm"], row["query_id"])
        if key in seen:
            raise ContractError("duplicate arm/query answer")
        seen.add(key)
        if row["query_id"] not in by_query:
            raise ContractError("answer has no evaluator expectation")
    expected_keys = {(arm, query_id)
                     for arm in {row["arm"] for row in answer_rows}
                     for query_id in by_query}
    if seen != expected_keys:
        raise ContractError("answer inventory is incomplete for an arm")
    outcome: dict[str, Any] = {}
    corrections: dict[str, Any] = {}
    unknowns: dict[str, Any] = {}
    for arm in sorted({row["arm"] for row in answer_rows}):
        arm_rows = [row for row in answer_rows if row["arm"] == arm]
        correct = incorrect = answered = abstained = 0
        by_kind: dict[str, Counter[str]] = defaultdict(Counter)
        corrected_expected = corrected_correct = 0
        unknown_expected = unknown_abstained = unknown_false = 0
        for row in arm_rows:
            label = by_query[row["query_id"]]
            is_correct = _normalize_answer(row["text"]) == _normalize_answer(label.expected_answer)
            kind = label.query_kind
            by_kind[kind]["total"] += 1
            if is_correct:
                correct += 1
                by_kind[kind]["correct"] += 1
            else:
                incorrect += 1
                by_kind[kind]["incorrect"] += 1
            if row["answered"]:
                answered += 1
                by_kind[kind]["answered"] += 1
            else:
                abstained += 1
                by_kind[kind]["abstained"] += 1
            if kind == "corrected":
                corrected_expected += 1
                corrected_correct += int(is_correct)
            if kind == "unknown":
                unknown_expected += 1
                unknown_abstained += int(not row["answered"])
                unknown_false += int(bool(row["answered"]))
        total = len(arm_rows)
        outcome[arm] = {
            "total": total,
            "correct": correct,
            "incorrect": incorrect,
            "answered": answered,
            "abstained": abstained,
            "accuracy": correct / total if total else 0.0,
            "by_kind": {kind: dict(counter) for kind, counter in sorted(by_kind.items())},
        }
        corrections[arm] = {
            "expected": corrected_expected,
            "correct": corrected_correct,
            "accuracy": corrected_correct / corrected_expected
            if corrected_expected else 0.0,
        }
        unknowns[arm] = {
            "expected": unknown_expected,
            "abstained": unknown_abstained,
            "false_answered": unknown_false,
            "abstention_rate": unknown_abstained / unknown_expected
            if unknown_expected else 0.0,
        }
    exact_rows = [row for row in answer_rows if row["arm"] == "exact_source"]
    retrieval_count = sum(int(bool(row["retrieval_performed"])) for row in exact_rows)
    source_bytes = sum(int(row["source_bytes_returned"]) for row in exact_rows)
    state_rows = [row for row in answer_rows if row["state_reads"]]
    return {
        "outcome": outcome,
        "corrections": corrections,
        "unknowns": unknowns,
        "provenance": {
            "exact_source_answers": len(exact_rows),
            "exact_source_retrievals": retrieval_count,
            "exact_source_bytes_returned": source_bytes,
            "state_read_answers": len(state_rows),
            "distinct_state_hashes": len({row["state_sha256"] for row in state_rows
                                          if row["state_sha256"]}),
            "source_observations_used": len({row["source_observation_id"]
                                              for row in exact_rows
                                              if row["source_observation_id"]}),
        },
        "answer_count": len(answer_rows),
        "label_count": len(label_rows),
        "label_boundary": "answers carry no expected, gold, split or query-kind field",
        "quality_claim": False,
    }


def _mechanical_gates(records: Sequence[Mapping[str, Any]], *,
                      config: Mapping[str, Any]) -> dict[str, Any]:
    def all_true(name: str) -> bool:
        return bool(records) and all(bool(record[name]) for record in records)

    gates = {
        "require_exact_delayed_queries": all_true("delayed_queries_present"),
        "require_no_context_reads_after_removal": (
            all_true("context_free_reads_zero")
            and all_true("no_context_access_after_removal")),
        "require_restart_without_reencode": all_true("restart_no_reencode"),
        "require_interrupted_restart_recovery": all_true("interrupted_restart_recovered"),
        "require_dated_correction": all_true("dated_correction"),
        "require_unaffected_retention": all_true("unaffected_retention"),
        "require_unknown_abstention": all_true("unknown_queries_abstained"),
        "require_stale_source_rejection": all_true("stale_source_rejected"),
        "require_label_boundary": all_true("label_boundary_checked"),
        "require_mechanical_separation": True,
    }
    declared = config["mechanical_gates"]
    return {
        "declared": declared,
        "observed": gates,
        "all_passed": all(gates[name] for name in declared if declared[name]),
    }


def run_visual_memory_cpu(prepared: str | Path, output: str | Path, *,
                          require_admission: bool = True,
                          max_wall_seconds: int | None = None) -> dict[str, Any]:
    """Run the CPU contract checks and write answers, retrievals and a report.

    No native job, optimizer, gradient, model weight or training arm is used.
    The reference adapter's outcomes are mechanical contract evidence only.
    """
    started = time.monotonic()
    prepared = Path(prepared)
    output = Path(output)
    if require_admission:
        admission_path = prepared / "admission.json"
        if not admission_path.is_file():
            raise ContractError("CPU evaluation requires an admitted preparation")
        admission = _read_json(admission_path)
        if admission.get("format") != ADMISSION_FORMAT or admission.get("status") != "admitted":
            raise ContractError("preparation admission is not an admitted receipt")
        current = verify_visual_memory_prepared(prepared)
        for name in ("prepared_manifest_sha256", "protocol_sha256",
                     "episodes_sha256", "queries_sha256", "labels_sha256",
                     "implementation_sha256"):
            if admission.get(name) != current.get(name):
                raise ContractError(f"admission {name} differs from current bytes")
    else:
        verify_visual_memory_prepared(prepared)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ContractError("CPU evaluation output must be fresh")
    output.mkdir(parents=True, exist_ok=True)
    manifest, protocol, episodes, queries, labels, _annotations, annotation_map = (
        _load_corpus(prepared))
    config = _read_json(prepared / "config.json")
    if max_wall_seconds is None:
        max_wall_seconds = int(config["limits"]["max_wall_seconds"])
    evaluated_splits = list(manifest["evaluation_splits"])
    episode_by_id = {episode.episode_id: episode for episode in episodes}
    label_by_query = {label.query_id: label for label in labels}
    queries_by_episode: dict[str, list[VisualQuery]] = defaultdict(list)
    for query in queries:
        queries_by_episode[query.episode_id].append(query)
    evaluated = [episode for episode in episodes
                 if next(world["split"] for world in manifest["worlds"]
                         if world["episode_id"] == episode.episode_id)
                 in evaluated_splits]
    if not evaluated:
        raise ContractError("no episode belongs to a declared evaluation split")
    other_by_episode: dict[str, Episode] = {}
    for episode in evaluated:
        split = next(world["split"] for world in manifest["worlds"]
                     if world["episode_id"] == episode.episode_id)
        pool = [item for item in evaluated
                if next(world["split"] for world in manifest["worlds"]
                        if world["episode_id"] == item.episode_id) == split]
        index = pool.index(episode)
        other_by_episode[episode.episode_id] = pool[(index + 1) % len(pool)]
    provider = ExactSourceProvider(
        output / "exact_source" / "ledger.db",
        output / "exact_source" / "blobs",
        subject_id=config["subject_id"], purpose=config["memory_purpose"],
        annotations=annotation_map)
    answers: list[dict[str, Any]] = []
    retrievals: list[dict[str, Any]] = []
    mechanical: list[dict[str, Any]] = []
    storage: dict[str, Any] = {
        "source_image_bytes": 0,
        "state_bytes": 0,
        "journal_bytes": 0,
        "ledger_bytes": 0,
        "source_blob_bytes": 0,
    }
    try:
        for episode in evaluated:
            if time.monotonic() - started > max_wall_seconds:
                raise VisualMemoryError("CPU evaluation exceeded its wall-clock bound")
            split = next(world["split"] for world in manifest["worlds"]
                         if world["episode_id"] == episode.episode_id)
            initial_bytes = _load_image(prepared, episode.initial)
            correction_bytes = _load_image(prepared, episode.correction)
            storage["source_image_bytes"] += len(initial_bytes) + len(correction_bytes)
            receipts = provider.admit_episode(
                episode, initial_bytes=initial_bytes,
                correction_bytes=correction_bytes)
            if len(receipts) != 2:
                raise ContractError("exact source admission did not admit both views")
            episode_queries = queries_by_episode[episode.episode_id]
            initial_queries = [q for q in episode_queries
                               if label_by_query[q.query_id].phase == "initial"]
            post_queries = [q for q in episode_queries
                            if label_by_query[q.query_id].phase == "after_correction"]
            if not initial_queries or not post_queries:
                raise ContractError("episode must have initial and delayed post queries")
            primary_root = output / "artifacts" / "memory" / episode.episode_id
            runtime = EpisodeRuntime.create(primary_root, episode)
            runtime.observe_initial(initial_bytes)
            runtime.commit_initial()
            runtime.remove_context()
            state_before = runtime.state
            if state_before is None:
                raise ContractError("commit did not produce state")
            other = other_by_episode[episode.episode_id]
            other_initial_state = _build_reference_state(
                other, prepared, "initial", ReferenceContractAdapter())
            for query in initial_queries:
                for arm in CONTEXT_FREE_ARMS:
                    answers.append(runtime.ask(
                        query, arm,
                        shuffled_state=other_initial_state).as_dict())
            restarted, restart_before = EpisodeRuntime.restart(primary_root)
            restarted.attach_episode(episode)
            if restart_before.encode_calls != 0 or restart_before.reencoded_assets != 0:
                raise ContractError("restart re-encoded an old observation")
            restarted.apply_correction(correction_bytes)
            state_after = restarted.state
            if state_after is None:
                raise ContractError("correction did not produce state")
            intent = restarted.interrupt_publication()
            recovered, interrupted_report = EpisodeRuntime.restart(primary_root)
            recovered.attach_episode(episode)
            if intent not in interrupted_report.pending_intents:
                raise ContractError("interrupted restart did not report the pending intent")
            if recovered.state is None or recovered.state.state_sha256 != state_after.state_sha256:
                raise ContractError("interrupted restart did not recover the published state")
            if interrupted_report.encode_calls != 0:
                raise ContractError("interrupted restart re-encoded an observation")
            other_final_state = _build_reference_state(
                other, prepared, "after_correction", ReferenceContractAdapter())
            for query in post_queries:
                for arm in CONTEXT_FREE_ARMS:
                    answers.append(recovered.ask(
                        query, arm,
                        shuffled_state=other_final_state).as_dict())
            context_free_answers = [row for row in answers
                                    if row["episode_id"] == episode.episode_id
                                    and row["arm"] in CONTEXT_FREE_ARMS]
            context_free_reads_zero = all(
                row["context_reads"] == 0 and row["image_reads"] == 0
                and row["annotation_reads"] == 0
                and row["source_bytes_returned"] == 0
                for row in context_free_answers)
            context_removed = recovered.context_removed
            try:
                _ = recovered.context_image_bytes
                no_context_access = False
            except ContextRemovedError:
                no_context_access = True
            # Full visual context is a separate control: it deliberately keeps
            # the same information boundary but does not remove context.
            context_root = output / "artifacts" / "full_context" / episode.episode_id
            context_runtime = EpisodeRuntime.create(context_root, episode)
            context_runtime.observe_initial(initial_bytes)
            context_runtime.commit_initial()
            context_runtime.apply_correction(correction_bytes)
            for query in episode_queries:
                answers.append(context_runtime.ask(query, "full_context").as_dict())
            # Exact-source retrieval is explicit and never counted as state recall.
            for query in episode_queries:
                retrieved = provider.retrieve(episode, query)
                retrievals.append(retrieved.as_dict())
                answers.append(recovered.ask(
                    query, "exact_source", exact_provider=provider).as_dict())
            stale_query = post_queries[0]
            try:
                provider.retrieve(
                    episode, stale_query,
                    expected_observation_id=episode.initial.observation_id)
                stale_rejected = False
            except StaleVisualSourceError:
                stale_rejected = True
            # Dated correction and unaffected retention are checked from state,
            # not from a self-reported model answer.
            unaffected_ids = tuple(obj.object_id for obj in episode.initial.objects
                                   if obj.object_id != episode.changed_object())
            initial_values = {
                (fact.object_id, fact.attribute): fact.value
                for fact in state_before.facts
                if fact.object_id in unaffected_ids
            }
            after_values = {
                (fact.object_id, fact.attribute): fact.value
                for fact in state_after.facts
                if fact.object_id in unaffected_ids
            }
            unaffected_retention = initial_values == after_values
            changed_facts = [fact for fact in state_after.facts
                             if fact.object_id == episode.changed_object()
                             and fact.attribute == "color"]
            dated_correction = bool(
                state_after.correction_count == 1
                and any(fact.supersedes is not None for fact in changed_facts)
                and episode.correction.observed_at > episode.initial.observed_at)
            unknown_labels = [label_by_query[q.query_id] for q in episode_queries
                              if label_by_query[q.query_id].query_kind == "unknown"]
            memory_unknown_answers = [
                row for row in answers
                if row["episode_id"] == episode.episode_id and row["arm"] == "memory"
                and row["query_id"] in {label.query_id for label in unknown_labels}
            ]
            unknown_queries_abstained = bool(unknown_labels) and all(
                row["text"] == UNKNOWN_ANSWER and not row["answered"]
                for row in memory_unknown_answers)
            label_boundary_checked = True
            for row in answers:
                if row["episode_id"] == episode.episode_id:
                    _validate_answer_row(row)
            for runtime_root in (primary_root, context_root):
                storage["state_bytes"] += (
                    runtime_root / EpisodeRuntime.STATE_FILE).stat().st_size
                storage["journal_bytes"] += (
                    runtime_root / EpisodeRuntime.JOURNAL_FILE).stat().st_size
            mechanical.append({
                "episode_id": episode.episode_id,
                "world_id": episode.world_id,
                "split": split,
                "delayed_queries_present": (
                    len(initial_queries) == INITIAL_QUERY_COUNT
                    and len(post_queries) == POST_CORRECTION_QUERY_COUNT),
                "context_removed": context_removed,
                "context_free_reads_zero": context_free_reads_zero,
                "no_context_access_after_removal": no_context_access,
                "restart_no_reencode": (
                    restart_before.encode_calls == 0
                    and restart_before.reencoded_assets == 0),
                "interrupted_restart_recovered": bool(
                    interrupted_report.interrupted
                    and interrupted_report.pending_intents
                    and interrupted_report.state_sha256 == state_after.state_sha256
                    and interrupted_report.encode_calls == 0),
                "restart_before_correction": restart_before.as_dict(),
                "interrupted_restart": interrupted_report.as_dict(),
                "dated_correction": dated_correction,
                "unaffected_retention": unaffected_retention,
                "unknown_queries_abstained": unknown_queries_abstained,
                "stale_source_rejected": stale_rejected,
                "label_boundary_checked": label_boundary_checked,
                "state_sha256": state_after.state_sha256,
                "journal_head": interrupted_report.journal_head,
                "changed_object_id": episode.changed_object(),
                "initial_observation_id": episode.initial.observation_id,
                "correction_observation_id": episode.correction.observation_id,
                "initial_observed_at": episode.initial.observed_at,
                "correction_observed_at": episode.correction.observed_at,
            })
        ledger = output / "exact_source" / "ledger.db"
        storage["ledger_bytes"] = ledger.stat().st_size if ledger.is_file() else 0
        storage["source_blob_bytes"] = _dir_bytes(output / "exact_source" / "blobs")
    finally:
        provider.close()
    evaluated_labels = [label for label in labels
                        if label.episode_id in {episode.episode_id for episode in evaluated}]
    if len(evaluated_labels) > config["limits"]["max_queries"]:
        raise VisualMemoryError("evaluated query count exceeds the declared limit")
    if len(answers) > config["limits"]["max_answers"]:
        raise VisualMemoryError("answer count exceeds the declared limit")
    scoring = score_visual_memory(answers, evaluated_labels)
    answer_bytes = _jsonl_bytes(answers)
    retrieval_bytes = _jsonl_bytes(retrievals)
    _write_exact(output / "answers.jsonl", answer_bytes)
    _write_exact(output / "retrievals.jsonl", retrieval_bytes)
    storage["answer_bytes"] = len(answer_bytes)
    storage["retrieval_bytes"] = len(retrieval_bytes)
    storage["total_bytes"] = sum(storage.values())
    gates = _mechanical_gates(mechanical, config=config)
    report = {
        "format": RUN_FORMAT,
        "status": "cpu_contract_checks_complete",
        "native_evaluated": False,
        "adapter": {
            "kind": "deterministic_reference_contract_adapter",
            "trained": False,
            "optimizer_updates": 0,
            "gradient_examples": 0,
            "quality_claim": False,
        },
        "prepared": {
            "manifest_sha256": _sha256_file(prepared / "manifest.json"),
            "protocol_sha256": _sha256_file(prepared / "protocol.json"),
            "episodes_sha256": _sha256_file(prepared / "episodes.jsonl"),
            "queries_sha256": _sha256_file(prepared / "queries.jsonl"),
            "labels_sha256": _sha256_file(prepared / "labels.jsonl"),
            "evaluation_splits": evaluated_splits,
        },
        "counts": {
            "episodes": len(evaluated),
            "queries": len(evaluated_labels),
            "answers": len(answers),
            "retrievals": len(retrievals),
        },
        "actual_outcome_kind": (
            "deterministic_reference_contract_outcome_not_quality"),
        "actual_outcome": scoring["outcome"],
        "corrections": {
            **scoring["corrections"],
            "dated_corrections": len(mechanical),
            "dated_order_valid": all(record["dated_correction"]
                                     for record in mechanical),
            "correction_records": [
                {
                    "episode_id": record["episode_id"],
                    "changed_object_id": record["changed_object_id"],
                    "initial_observed_at": record["initial_observed_at"],
                    "correction_observed_at": record["correction_observed_at"],
                    "supersession_applied": record["dated_correction"],
                }
                for record in mechanical
            ],
        },
        "unknowns": scoring["unknowns"],
        "provenance": {
            **scoring["provenance"],
            "label_boundary": scoring["label_boundary"],
            "source_images": [
                {
                    "episode_id": episode.episode_id,
                    "initial_sha256": episode.initial.image_sha256,
                    "correction_sha256": episode.correction.image_sha256,
                }
                for episode in evaluated
            ],
        },
        "storage": {
            **storage,
            "reported_separately_from_mechanical_persistence": True,
            "state_and_journal_bytes_are_mechanical": True,
        },
        "mechanical_persistence": {
            "episodes": mechanical,
            "gates": gates,
            "note": (
                "Restart parity, journal recovery and stale-source rejection are "
                "mechanical persistence evidence, not answer quality or learned memory."
            ),
        },
        "native_gate": dict(protocol["native_gate"]),
        "limitations": [
            *config["limitations"],
            "The CPU reference adapter is not a learned memory and its outcome "
            "is contract-check evidence, not model quality.",
            "Native evaluation remains gated on independent perception and "
            "memory-use evidence.",
        ],
    }
    _write_exact(output / "report.json", _json_bytes(report))
    return report


def assert_native_evaluation_allowed(protocol: Mapping[str, Any]) -> None:
    """Refuse native evaluation until both prerequisite evidence gates pass."""
    gate = protocol.get("native_gate") if isinstance(protocol, Mapping) else None
    raise NativeEvaluationGated(
        "native evaluation is gated: a qualified perception producer and "
        "independent memory-use evidence have not been supplied by this package; "
        f"declared gate={gate or NATIVE_GATE}")

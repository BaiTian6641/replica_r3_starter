"""Atomic, idempotent publication of final run artifacts.

This is a deliberately small helper shared by the CPU value-readout runner and
future native runners.  It provides three things and nothing else:

``write_bytes``/``write_json``/``write_jsonl``/``write_torch``
    Every artifact is written to a temporary file in its destination directory,
    fsynced and moved into place with ``os.replace``.  A crash therefore never
    leaves a truncated artifact.  When the caller already supplies the exact
    artifact, an existing file is verified by content identity instead of being
    rewritten, so large checkpoint payloads are not duplicated or rewritten
    during a resumed final stage.

``content_identity``
    A canonical, JSON-serializable description of a payload in which every
    tensor is replaced by dtype, shape and content SHA-256.  Two payloads with
    the same identity are the same artifact content; this is what a resumed
    process uses to accept a checkpoint that is already durable.

``LiveByteMeter``
    Records the peak number of bytes physically present under a root, including
    temporary files and any retained recovery artifacts.  Measurement never
    deletes anything: overflow evidence is written next to the preserved
    artifacts by :func:`write_overflow_evidence`.

``Journal``
    An optional durable record of published stages.  ``stage_done`` re-reads
    every recorded artifact and verifies it by SHA-256, so a resumed process
    publishes only stages that are missing or no longer match.  A journal is
    never rewritten with fewer stage entries than it already holds.

Nothing in this module decides protocol policy: the caller owns the cap, the
stage order and the meaning of each artifact.
"""
from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

JOURNAL_FORMAT = "replica.artifact_publication.journal.v1"
OVERFLOW_FORMAT = "replica.artifact_publication.overflow.v1"


class PublicationError(RuntimeError):
    """A publication precondition, artifact or identity check failed."""


class PublicationOverflow(PublicationError):
    """The measured artifact tree exceeds the caller's declared ceiling."""

    def __init__(self, measured_bytes: int, cap_bytes: int,
                 evidence_path: Path):
        super().__init__(
            f"publication byte budget exceeded: {measured_bytes} > "
            f"{cap_bytes}; overflow evidence preserved at {evidence_path}")
        self.measured_bytes = int(measured_bytes)
        self.cap_bytes = int(cap_bytes)
        self.evidence_path = Path(evidence_path)


class Artifact(tuple):
    """Immutable ``(path, sha256, bytes)`` row for one published artifact."""

    __slots__ = ()

    def __new__(cls, path: str, sha256: str, size: int) -> "Artifact":
        return tuple.__new__(cls, (str(path), str(sha256), int(size)))

    @property
    def path(self) -> str:
        return self[0]

    @property
    def sha256(self) -> str:
        return self[1]

    @property
    def bytes(self) -> int:
        return self[2]

    def as_dict(self) -> dict:
        return {"path": self.path, "sha256": self.sha256, "bytes": self.bytes}

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> "Artifact":
        return cls(row["path"], row["sha256"], row["bytes"])

    @classmethod
    def of(cls, root: Path, relative: str) -> "Artifact":
        target = _target(root, relative)
        if not target.is_file():
            raise PublicationError(f"artifact is missing: {target}")
        return cls(relative, sha256_path(target), target.stat().st_size)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _target(root: Path, relative: str | Path) -> Path:
    root = Path(root)
    candidate = Path(relative)
    if candidate.is_absolute():
        raise PublicationError(f"artifact path must be relative: {relative}")
    if any(part == ".." for part in candidate.parts):
        raise PublicationError(f"artifact path escapes the root: {relative}")
    return root / candidate

def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _torch():
    import torch

    return torch


def tensor_digest(tensor: Any) -> dict:
    """Dtype, shape and content SHA-256 of one tensor."""

    value = tensor.detach().to("cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(str(tuple(value.shape)).encode("ascii"))
    if value.numel():
        digest.update(value.reshape(-1).view(_torch().uint8).numpy().tobytes())
    return {"dtype": str(value.dtype), "shape": list(value.shape),
            "sha256": digest.hexdigest()}


def content_identity(value: Any) -> Any:
    """Canonical JSON-serializable identity of a payload or artifact content."""

    torch = _torch()
    if isinstance(value, torch.Tensor):
        return {"tensor": tensor_digest(value)}
    if isinstance(value, Mapping):
        return {"mapping": [[str(key), content_identity(value[key])]
                            for key in sorted(value, key=str)]}
    if isinstance(value, (list, tuple)):
        return {"sequence": [content_identity(item) for item in value]}
    if isinstance(value, bytes):
        return {"bytes": sha256_bytes(value)}
    if isinstance(value, str):
        return {"str": value}
    if isinstance(value, bool):
        return {"bool": value}
    if isinstance(value, int):
        return {"int": value}
    if isinstance(value, float):
        return {"float": repr(value)}
    if value is None:
        return {"none": True}
    raise PublicationError(
        f"payload value of type {type(value).__name__} has no content identity")


def _encode_json(value: Any, *, sort_keys: bool) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=sort_keys,
                       allow_nan=False) + "\n").encode("utf-8")


def write_bytes(root: Path, relative: str, data: bytes, *,
                meter: "LiveByteMeter | None" = None) -> Artifact:
    """Write ``data`` atomically to ``root/relative``.

    The temporary file lives in the destination directory and therefore counts
    in live-byte measurement while it exists.
    """

    if not isinstance(data, (bytes, bytearray)):
        raise PublicationError("write_bytes requires bytes")
    target = _target(root, relative)
    if matches(root, relative, data):
        return Artifact(relative, sha256_path(target), target.stat().st_size)
    if meter is not None:
        meter.reserve(len(data))
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.parent / f".{target.name}.tmp"
    try:
        with temporary.open("wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if meter is not None:
            meter.observe()
        os.replace(temporary, target)
        _fsync_directory(target.parent)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise
    if meter is not None:
        meter.observe()
    return Artifact(relative, sha256_bytes(data), len(data))


def write_json(root: Path, relative: str, value: Mapping[str, Any], *,
               sort_keys: bool = True,
               meter: "LiveByteMeter | None" = None) -> Artifact:
    return write_bytes(root, relative, _encode_json(value, sort_keys=sort_keys),
                       meter=meter)


def write_jsonl(root: Path, relative: str, rows: Iterable[Mapping[str, Any]], *,
                meter: "LiveByteMeter | None" = None) -> Artifact:
    body = b"".join(
        (json.dumps(dict(row), sort_keys=True, allow_nan=False) + "\n")
        .encode("utf-8") for row in rows)
    return write_bytes(root, relative, body, meter=meter)


def load_torch(root: Path, relative: str) -> dict:
    return _torch().load(_target(root, relative), map_location="cpu",
                         weights_only=True)


def torch_identity_matches(root: Path, relative: str,
                           payload: Mapping[str, Any]) -> bool:
    """True when the durable artifact already holds exactly this payload.

    Content identity is dtype/shape/tensor-hash equality, so an unchanged
    checkpoint is accepted without rewriting the payload.  A missing or
    unreadable file is simply not a match.
    """

    target = _target(root, relative)
    if not target.is_file():
        return False
    try:
        existing = load_torch(root, relative)
    except BaseException:
        return False
    try:
        return content_identity(existing) == content_identity(payload)
    except PublicationError:
        return False


def write_torch(root: Path, relative: str, payload: Mapping[str, Any], *,
                meter: "LiveByteMeter | None" = None,
                reuse_identical: bool = True,
                on_write: Callable[[], None] | None = None) -> Artifact:
    """Publish a torch artifact, skipping an unchanged payload.

    The deterministic ``.<name>.tmp`` temporary name keeps the archive root
    stable, so identical payloads serialize to identical bytes.  With
    ``reuse_identical`` the durable artifact is loaded and kept as is when it
    already carries the same content identity; pass ``reuse_identical=False``
    for volatile payloads (for example per-step resume pointers) to avoid a
    needless read.
    """

    torch = _torch()
    target = _target(root, relative)
    if reuse_identical and torch_identity_matches(root, relative, payload):
        return Artifact(relative, sha256_path(target), target.stat().st_size)
    if meter is not None:
        meter.reserve(int(payload.get("_estimated_bytes", 0)))
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.parent / f".{target.name}.tmp"
    try:
        torch.save(dict(payload), temporary)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        if meter is not None:
            meter.observe()
        os.replace(temporary, target)
        _fsync_directory(target.parent)
        if on_write is not None:
            on_write()
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise
    if meter is not None:
        meter.observe()
    return Artifact(relative, sha256_path(target), target.stat().st_size)


def matches(root: Path, relative: str, data: bytes) -> bool:
    target = _target(root, relative)
    if not target.is_file():
        return False
    if target.stat().st_size != len(data):
        return False
    return sha256_path(target) == sha256_bytes(data)


def inspect(root: Path, artifact: Artifact) -> bool:
    """Re-read one recorded artifact and verify it by content."""

    target = _target(root, artifact.path)
    if not target.is_file():
        return False
    return (target.stat().st_size == artifact.bytes
            and sha256_path(target) == artifact.sha256)


def verify_all(root: Path, artifacts: Iterable[Artifact]) -> bool:
    return all(inspect(root, artifact) for artifact in artifacts)


def inventory(root: Path, *, exclude: Iterable[Path] = ()) -> list[dict]:
    root = Path(root)
    excluded = {Path(path).resolve() for path in exclude}
    rows = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.resolve() in excluded:
            continue
        rows.append({"path": path.relative_to(root).as_posix(),
                     "bytes": path.stat().st_size})
    return rows


def total_bytes(root: Path, *, exclude: Iterable[Path] = ()) -> int:
    return sum(row["bytes"] for row in inventory(root, exclude=exclude))


def release(root: Path, relatives: Iterable[str]) -> list[dict]:
    """Delete artifacts whose content is already durable elsewhere.

    Returns one row per released artifact with its path, SHA-256 and size so the
    deletion itself stays auditable.  Missing files are ignored; nothing else is
    ever deleted.
    """

    root = Path(root)
    rows = []
    parents: set[Path] = set()
    for relative in relatives:
        target = _target(root, relative)
        if not target.is_file():
            continue
        rows.append({"path": Path(relative).as_posix(),
                     "sha256": sha256_path(target),
                     "bytes": target.stat().st_size})
        parents.add(target.parent)
        target.unlink()
    for directory in sorted(parents, key=lambda path: len(path.parts),
                            reverse=True):
        current = directory
        while current != root and root in current.parents:
            try:
                if any(current.iterdir()):
                    break
                current.rmdir()
            except OSError:
                break
            current = current.parent
    return rows


def write_overflow_evidence(root: Path, *, context: str, cap_bytes: int,
                            measured_bytes: int,
                            files: Sequence[Mapping[str, Any]],
                            extra: Mapping[str, Any] | None = None,
                            name: str = "artifact_overflow.json") -> Path:
    """Preserve overflow evidence atomically.  Never deletes any artifact."""

    payload = {
        "format": OVERFLOW_FORMAT,
        "context": context,
        "status": "overflow",
        "cap_bytes": int(cap_bytes),
        "measured_bytes_before_overflow_evidence": int(measured_bytes),
        "files": [dict(row) for row in files],
        "deleted_artifacts": [],
        "preserved": True,
    }
    if extra:
        payload.update(dict(extra))
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    write_json(root, name, payload, sort_keys=False)
    return root / name


class LiveByteMeter:
    """Peak bytes physically present under a root, with optional enforcement."""

    def __init__(self, root: Path, *, label: str = "publication",
                 cap_bytes: int | None = None):
        self.root = Path(root)
        self.label = str(label)
        self.cap_bytes = None if cap_bytes is None else int(cap_bytes)
        self.observations = 0
        self.peak_bytes = 0
        self.peak_files: list[dict] = []

    def reserve(self, additional_bytes: int) -> None:
        """Refuse a projected overlap before creating a new temporary file."""
        if self.cap_bytes is None:
            return
        additional = int(additional_bytes)
        if additional < 0:
            raise PublicationError("additional bytes must be nonnegative")
        files = inventory(self.root)
        projected = sum(row["bytes"] for row in files) + additional
        if projected <= self.cap_bytes:
            return
        self.observations += 1
        if projected > self.peak_bytes:
            self.peak_bytes = projected
            self.peak_files = files
        evidence_path = self.root / "artifact_overflow.json"
        write_overflow_evidence(
            self.root, context=self.label, cap_bytes=self.cap_bytes,
            measured_bytes=projected, files=files,
            extra={"projected_overlap": True,
                   "additional_bytes": additional})
        raise PublicationOverflow(projected, self.cap_bytes, evidence_path)

    def observe(self) -> int:
        self.root.mkdir(parents=True, exist_ok=True)
        files = inventory(self.root)
        live = sum(row["bytes"] for row in files)
        self.observations += 1
        if live > self.peak_bytes:
            self.peak_bytes = live
            self.peak_files = files
        if self.cap_bytes is not None and live > self.cap_bytes:
            evidence_path = self.root / "artifact_overflow.json"
            write_overflow_evidence(
                self.root, context=self.label, cap_bytes=self.cap_bytes,
                measured_bytes=live, files=files)
            raise PublicationOverflow(live, self.cap_bytes, evidence_path)
        return live

    def report(self) -> dict:
        return {
            "label": self.label,
            "peak_live_bytes": self.peak_bytes,
            "peak_live_files": [row["path"] for row in self.peak_files],
            "peak_live_file_count": len(self.peak_files),
            "observations": self.observations,
            "includes_temporary_files": True,
            "includes_recovery_artifacts": True,
        }


def peak_union(*reports: Mapping[str, Any] | None) -> dict | None:
    """Largest live peak of the supplied reports, or ``None``."""

    usable = [dict(report) for report in reports if report]
    if not usable:
        return None
    best = max(usable, key=lambda report: int(report.get("peak_live_bytes", 0)))
    return best


class Journal:
    """Durable record of published final stages.

    ``record`` validates every artifact by content, then rewrites the journal
    atomically.  ``stage_done`` is only true when every recorded artifact for
    that stage still exists with the recorded size and SHA-256, so a resumed run
    republishes exactly the missing or invalid stages.
    """

    def __init__(self, root: Path, *, identity: Mapping[str, Any],
                 name: str = "publication_journal.json",
                 cap_bytes: int | None = None,
                 meter: LiveByteMeter | None = None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.identity = dict(identity)
        self.name = str(name)
        self.cap_bytes = None if cap_bytes is None else int(cap_bytes)
        if meter is None:
            meter = LiveByteMeter(self.root, cap_bytes=self.cap_bytes)
        if (self.cap_bytes is not None and meter.cap_bytes is not None
                and int(meter.cap_bytes) != self.cap_bytes):
            raise PublicationError("meter and journal artifact caps differ")
        meter.cap_bytes = self.cap_bytes
        self.meter = meter
        self.stages: "OrderedDict[str, list[Artifact]]" = OrderedDict()
        self.recovery: list[Artifact] = []
        self.live_peak: dict | None = None
        self._load()

    # -- persistence ------------------------------------------------------
    def _payload(self) -> dict:
        return {
            "format": JOURNAL_FORMAT,
            "identity": dict(self.identity),
            "cap_bytes": self.cap_bytes,
            "stages": [{"stage": stage,
                        "artifacts": [row.as_dict() for row in rows]}
                       for stage, rows in self.stages.items()],
            "recovery_artifacts": [row.as_dict() for row in self.recovery],
            "live_peak": self.live_peak,
        }

    def _load(self) -> None:
        path = self.root / self.name
        if not path.is_file():
            return
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise PublicationError(
                f"publication journal is unreadable: {path}") from error
        if payload.get("format") != JOURNAL_FORMAT:
            raise PublicationError("publication journal format differs")
        if dict(payload.get("identity", {})) != self.identity:
            raise PublicationError(
                "publication journal identity differs from this run")
        recorded_cap = payload.get("cap_bytes")
        if (self.cap_bytes is not None and recorded_cap is not None
                and int(recorded_cap) != self.cap_bytes):
            raise PublicationError("publication journal cap differs")
        for row in payload.get("stages", []):
            self.stages[str(row["stage"])] = [
                Artifact.from_mapping(item) for item in row["artifacts"]]
        self.recovery = [Artifact.from_mapping(row)
                         for row in payload.get("recovery_artifacts", [])]
        self.live_peak = payload.get("live_peak")

    def flush(self) -> None:
        write_json(self.root, self.name, self._payload(),
                   sort_keys=False, meter=self.meter)

    # -- stages -----------------------------------------------------------
    def artifacts(self, stage: str) -> list[Artifact]:
        return list(self.stages.get(stage, ()))

    def stage_done(self, stage: str) -> bool:
        rows = self.stages.get(stage, [])
        return bool(rows) and verify_all(self.root, rows)

    def record(self, stage: str, artifacts: Sequence[Artifact]) -> None:
        rows = [artifact if isinstance(artifact, Artifact)
                else Artifact.from_mapping(artifact) for artifact in artifacts]
        if not rows:
            raise PublicationError(f"stage {stage!r} published no artifacts")
        for artifact in rows:
            if not inspect(self.root, artifact):
                raise PublicationError(
                    f"stage {stage!r} artifact is missing or changed: "
                    f"{artifact.path}")
        self.stages[str(stage)] = rows
        self.flush()

    def record_recovery(self, artifacts: Sequence[Artifact]) -> None:
        self.recovery = [artifact if isinstance(artifact, Artifact)
                         else Artifact.from_mapping(artifact)
                         for artifact in artifacts]
        self.flush()

    def recovery_done(self) -> bool:
        return bool(self.recovery) and verify_all(self.root, self.recovery)

    def record_peak(self, report: Mapping[str, Any] | None) -> None:
        self.live_peak = peak_union(self.live_peak, report)
        self.flush()

    def summary(self) -> dict:
        return {
            "format": JOURNAL_FORMAT,
            "identity": dict(self.identity),
            "cap_bytes": self.cap_bytes,
            "stages": list(self.stages),
            "recovery_artifacts": [row.as_dict() for row in self.recovery],
            "live_peak": self.live_peak,
        }


def published_stage_done(root: Path, stage: str, *,
                         identity: Mapping[str, Any]) -> bool:
    """Return true only for this attempt's verified durable stage artifacts."""
    root = Path(root)
    path = root / "publication_journal.json"
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise PublicationError(
            f"publication journal is unreadable: {path}") from error
    if payload.get("format") != JOURNAL_FORMAT:
        raise PublicationError("publication journal format differs")
    if dict(payload.get("identity", {})) != dict(identity):
        raise PublicationError("publication journal identity differs from this run")
    for row in payload.get("stages", []):
        if row.get("stage") != stage:
            continue
        artifacts = [Artifact.from_mapping(item)
                     for item in row.get("artifacts", [])]
        return bool(artifacts) and verify_all(root, artifacts)
    return False

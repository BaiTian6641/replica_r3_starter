"""Hot/warm/cold validity, recovery and measured accounting, version 1.

The proposed subsystem ceilings (2 GiB hot, 8 GiB warm, 64 GiB cold, 8 MiB
identity inside hot and 256 MiB replay inside warm) are budgets, not allocations
and not evidence of fit.  This module enforces optional byte capacities, runs the
same validity cases across tier transitions and restarts, and reports real
measured file I/O and latency.  On Apple unified memory a directory name proves
nothing about physical GPU residency.

Format: ``replica.successors.tiers.v1``.  Every payload is content-addressed by
a manifest; selection revalidates scope, generation and producer authority
before reading and again after cold I/O.  Missing, partial or corrupt reads are
explicit statuses; canonical fallback is only used when separately eligible and
is charged.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import time
from typing import Callable, Mapping, Sequence

FORMAT = "replica.successors.tiers.v1"
MANIFEST_FORMAT = "replica.successors.tiers.manifest.v1"

HOT_BUDGET_BYTES = 2 * 1024 ** 3
WARM_BUDGET_BYTES = 8 * 1024 ** 3
COLD_BUDGET_BYTES = 64 * 1024 ** 3
IDENTITY_RESERVATION_BYTES = 8 * 1024 ** 2
REPLAY_RESERVATION_BYTES = 256 * 1024 ** 2

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:+-]{0,127}$")


class TierError(RuntimeError):
    """Base class for tier validity failures."""


class CapacityExceeded(TierError):
    """A proposed tier budget would be exceeded."""


class DirtyState(TierError):
    """A dirty P hot copy cannot be evicted before coherent publication."""


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_json(value: object) -> str:
    return _sha256_bytes(_canonical(value).encode("utf-8"))


def _identifier(name: str, value: object) -> None:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{name} must be a bounded identifier")


def _now_ns() -> int:
    return time.time_ns()


@dataclass(frozen=True, slots=True)
class TierLimits:
    hot_bytes: int = HOT_BUDGET_BYTES
    warm_bytes: int = WARM_BUDGET_BYTES
    cold_bytes: int = COLD_BUDGET_BYTES
    identity_reservation_bytes: int = IDENTITY_RESERVATION_BYTES
    replay_reservation_bytes: int = REPLAY_RESERVATION_BYTES

    def __post_init__(self) -> None:
        for name in ("hot_bytes", "warm_bytes", "cold_bytes",
                     "identity_reservation_bytes", "replay_reservation_bytes"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.identity_reservation_bytes >= self.hot_bytes:
            raise ValueError("identity reservation must sit inside the hot budget")
        if self.replay_reservation_bytes >= self.warm_bytes:
            raise ValueError("replay reservation must sit inside the warm budget")

    @property
    def hot_nonidentity_bytes(self) -> int:
        return self.hot_bytes - self.identity_reservation_bytes

    @property
    def warm_nonreplay_bytes(self) -> int:
        return self.warm_bytes - self.replay_reservation_bytes

    def document(self) -> dict:
        return {name: getattr(self, name) for name in (
            "hot_bytes", "warm_bytes", "cold_bytes", "identity_reservation_bytes",
            "replay_reservation_bytes")} | {
            "hot_nonidentity_bytes": self.hot_nonidentity_bytes,
            "warm_nonreplay_bytes": self.warm_nonreplay_bytes,
            "status": "proposed subsystem budgets, not allocations",
        }


@dataclass(frozen=True, slots=True)
class ReadResult:
    status: str  # ready | fallback | pending | unavailable | corrupt | ineligible | rejected
    record_id: str
    tier: str | None
    payload: bytes | None
    manifest: dict | None
    latency_ms: float
    bytes_read: int
    reason: str = ""
    ticket_id: str | None = None
    fallback_tokens: int = 0

    @property
    def ok(self) -> bool:
        return self.status in ("ready", "fallback")

    def document(self) -> dict:
        return {"status": self.status, "record_id": self.record_id, "tier": self.tier,
                "payload_sha256": (self.manifest or {}).get("payload_sha256"),
                "latency_ms": self.latency_ms, "bytes_read": self.bytes_read,
                "reason": self.reason, "ticket_id": self.ticket_id,
                "fallback_tokens": self.fallback_tokens}


@dataclass(frozen=True, slots=True)
class ReadContract:
    """Expected read identity and validity, enforced before and after I/O."""
    producer: str
    representation: str
    scope_digest: str
    cutoff: int | None = None
    generation: int | None = None

    def __post_init__(self) -> None:
        for name in ("producer", "representation", "scope_digest"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"read contract {name} required")
        if self.cutoff is not None and (type(self.cutoff) is not int
                                        or self.cutoff < 0):
            raise ValueError("read contract cutoff must be a nonnegative integer")
        if self.generation is not None and (type(self.generation) is not int
                                            or self.generation < 0):
            raise ValueError("read contract generation must be a nonnegative integer")

    def document(self) -> dict:
        return {"producer": self.producer,
                "representation": self.representation,
                "scope_digest": self.scope_digest, "cutoff": self.cutoff,
                "generation": self.generation}


class ProjectionCache:
    """Learned projection/KV cache keyed by complete eligibility identity."""

    def __init__(self) -> None:
        self._entries: dict[tuple, object] = {}

    @staticmethod
    def key(record_id: str, *, generation: int, producer: str, representation: str,
            reader_release: str, site: int, eligibility_revision: int) -> tuple:
        return (record_id, generation, producer, representation, reader_release,
                int(site), int(eligibility_revision))

    def put(self, key: tuple, value: object) -> None:
        self._entries[key] = value

    def get(self, key: tuple):
        return self._entries.get(key)

    def invalidate(self, record_id: str) -> int:
        return self.invalidate_prefix(record_id, exact=True)

    def invalidate_prefix(self, prefix: str, *, exact: bool = False) -> int:
        stale = [key for key in self._entries
                 if (key[0] == prefix if exact else key[0].startswith(prefix))]
        for key in stale:
            del self._entries[key]
        return len(stale)

    def __len__(self) -> int:
        return len(self._entries)


class ReplayEligibility:
    """Replay eligibility by current generation/producer; tombstone-aware."""

    def __init__(self) -> None:
        self._entries: dict[str, tuple[int, str]] = {}

    def register(self, record_id: str, *, generation: int, producer: str) -> None:
        self._entries[record_id] = (int(generation), producer)

    def eligible(self, record_id: str, *, generation: int, producer: str) -> bool:
        entry = self._entries.get(record_id)
        return entry is not None and entry == (int(generation), producer)

    def invalidate(self, record_id: str) -> bool:
        return self.invalidate_prefix(record_id, exact=True) > 0

    def invalidate_prefix(self, prefix: str, *, exact: bool = False) -> int:
        stale = [record_id for record_id in self._entries
                 if (record_id == prefix if exact
                     else record_id.startswith(prefix))]
        for record_id in stale:
            self._entries.pop(record_id, None)
        return len(stale)


@dataclass(frozen=True, slots=True)
class _HotEntry:
    payload: bytes
    manifest: dict
    eligibility_revision: int
    dirty: bool = False


class TieredBlobStore:
    """Directory-backed warm/cold tiers plus a bounded in-process hot cache."""

    def __init__(self, root: Path | str, *, limits: TierLimits | None = None,
                 hot_capacity_bytes: int | None = None,
                 io_delay_seconds: float = 0.0) -> None:
        self.root = Path(root)
        self.limits = limits or TierLimits()
        self.root.mkdir(parents=True, exist_ok=True)
        for name in ("warm", "cold", "staging", "quarantine"):
            (self.root / name).mkdir(exist_ok=True)
        if hot_capacity_bytes is None:
            hot_capacity_bytes = min(4 * 1024 ** 2, self.limits.hot_nonidentity_bytes)
        if type(hot_capacity_bytes) is not int or hot_capacity_bytes <= 0:
            raise ValueError("hot capacity must be a positive integer")
        if hot_capacity_bytes > self.limits.hot_nonidentity_bytes:
            raise CapacityExceeded("hot capacity exceeds the nonidentity hot budget")
        if io_delay_seconds < 0:
            raise ValueError("io delay must be nonnegative")
        self.hot_capacity_bytes = hot_capacity_bytes
        self.io_delay_seconds = float(io_delay_seconds)
        self._hot: OrderedDict[str, _HotEntry] = OrderedDict()
        self._eligibility: dict[str, int] = {}
        self._tombstones: dict[str, str] = {}
        self._p_revoked: set[str] = {
            path.stem for path in (self.root / "p_revoked").glob("*.json")
        } if (self.root / "p_revoked").exists() else set()
        self._tickets: dict[str, dict] = {}
        self.projections = ProjectionCache()
        self.replay = ReplayEligibility()
        self._metrics = {
            "reads": 0, "writes": 0, "cache_hits": 0, "cache_misses": 0,
            "failed_reads": 0, "ineligible": 0, "fallback_tokens": 0,
            "bytes_read": 0, "bytes_written": 0, "overlap_bytes": 0,
            "movement_bytes": 0,
            "latency_ns": {"warm": [], "cold": [], "hot": []},
            "tier_reads": {"hot": 0, "warm": 0, "cold": 0},
        }

    # -- manifest helpers --------------------------------------------------

    @staticmethod
    def _record_dir(tier: str, record_id: str) -> Path:
        if tier not in ("warm", "cold"):
            raise ValueError("durable tier must be warm or cold")
        return Path(tier) / record_id

    def _directory(self, tier: str, record_id: str) -> Path:
        return self.root / self._record_dir(tier, record_id)

    def _write_record(self, tier: str, record_id: str, payload: bytes,
                      manifest: Mapping[str, object]) -> int:
        staging = self.root / "staging" / f"{record_id}.{os.getpid()}.{time.time_ns()}"
        staging.mkdir(parents=True)
        payload_path = staging / "payload.bin"
        with open(payload_path, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        manifest_path = staging / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as handle:
            handle.write(_canonical(manifest) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        target = self._directory(tier, record_id)
        if target.exists():
            existing = self._read_manifest(target)
            if existing["payload_sha256"] != manifest["payload_sha256"]:
                shutil.rmtree(staging)
                raise TierError("record already exists with different payload bytes")
            shutil.rmtree(staging)
            return 0
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging, target)
        try:
            directory = os.open(target.parent, os.O_RDONLY)
        except OSError:
            directory = None
        if directory is not None:
            try:
                os.fsync(directory)
            except OSError:
                pass
            finally:
                os.close(directory)
        return len(payload) + len(_canonical(manifest)) + 1

    def _read_manifest(self, directory: Path) -> dict:
        path = directory / "manifest.json"
        if not path.is_file():
            raise TierError("manifest is missing")
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise TierError("manifest is not canonical JSON") from error
        self._validate_manifest(manifest)
        return manifest

    @staticmethod
    def _validate_manifest(manifest: object) -> None:
        if not isinstance(manifest, dict):
            raise TierError("manifest must be an object")
        if manifest.get("format") != MANIFEST_FORMAT:
            raise TierError("manifest format is unknown")
        record_id = manifest.get("record_id")
        if not isinstance(record_id, str) or not record_id:
            raise TierError("manifest record_id is invalid")
        for name in ("kind", "producer", "representation", "scope_digest"):
            value = manifest.get(name)
            if not isinstance(value, str) or not value:
                raise TierError(f"manifest {name} is invalid")
        generation = manifest.get("generation")
        if type(generation) is not int or generation < 0:
            raise TierError("manifest generation is invalid")
        valid_from = manifest.get("valid_from")
        valid_to = manifest.get("valid_to")
        if type(valid_from) is not int or valid_from < 0:
            raise TierError("manifest valid_from is invalid")
        if valid_to is not None and (type(valid_to) is not int
                                     or valid_to <= valid_from):
            raise TierError("manifest valid_to is invalid")
        digest = manifest.get("payload_sha256")
        if not isinstance(digest, str) or len(digest) != 64 \
                or any(char not in "0123456789abcdef" for char in digest):
            raise TierError("manifest payload digest is invalid")
        size = manifest.get("bytes")
        if type(size) is not int or size < 0:
            raise TierError("manifest byte count is invalid")
        created = manifest.get("created_at_ns")
        if type(created) is not int or created < 0:
            raise TierError("manifest creation time is invalid")
        extra = manifest.get("extra")
        if not isinstance(extra, dict):
            raise TierError("manifest extra metadata is invalid")

    def _read_payload(self, tier: str, record_id: str) -> tuple[bytes, dict]:
        directory = self._directory(tier, record_id)
        manifest = self._read_manifest(directory)
        payload_path = directory / "payload.bin"
        if not payload_path.is_file():
            raise TierError("payload is missing")
        data = payload_path.read_bytes()
        if _sha256_bytes(data) != manifest.get("payload_sha256"):
            raise TierError("payload bytes differ from the manifest digest")
        if len(data) != manifest.get("bytes"):
            raise TierError("payload byte count differs from the manifest")
        if manifest.get("record_id") != record_id:
            raise TierError("manifest record identity mismatch")
        return data, manifest

    # -- publication and movement -----------------------------------------

    def publish(self, record_id: str, payload: bytes, *, generation: int, kind: str,
                producer: str, representation: str, scope_digest: str,
                valid_from: int = 0, valid_to: int | None = None,
                extra: Mapping[str, object] | None = None,
                tier: str = "warm", dirty: bool = False) -> dict:
        _identifier("record_id", record_id)
        _identifier("kind", kind)
        _identifier("producer", producer)
        _identifier("representation", representation)
        _identifier("scope_digest", scope_digest)
        if not isinstance(payload, bytes):
            raise TypeError("payload must be bytes")
        if type(generation) is not int or generation < 0:
            raise ValueError("generation must be a nonnegative integer")
        if valid_to is not None and valid_to <= valid_from:
            raise ValueError("valid_to must follow valid_from")
        if kind == "p":
            owner = self._p_owner(record_id)
            if owner is not None and self._is_p_revoked(owner):
                raise PermissionError("revoked P state cannot be published")
        manifest = {
            "format": MANIFEST_FORMAT, "record_id": record_id, "kind": kind,
            "generation": generation, "producer": producer,
            "representation": representation, "scope_digest": scope_digest,
            "valid_from": valid_from, "valid_to": valid_to,
            "payload_sha256": _sha256_bytes(payload), "bytes": len(payload),
            "created_at_ns": _now_ns(), "extra": dict(extra or {}),
        }
        written = self._write_record(tier, record_id, payload, manifest)
        if written == 0:
            manifest = dict(self._read_manifest(self._directory(tier, record_id)))
        self._metrics["writes"] += 1
        self._metrics["bytes_written"] += written
        self._eligibility.setdefault(record_id, 0)
        self._hot[record_id] = _HotEntry(payload, manifest,
                                        self._eligibility[record_id], dirty)
        self._hot.move_to_end(record_id)
        self._enforce_hot(protect=record_id)
        return dict(manifest)

    def demote(self, record_id: str, *, to: str = "cold") -> dict:
        if to != "cold":
            raise ValueError("demotion target is cold")
        owner = self._p_owner(record_id)
        if owner is not None and self._is_p_revoked(owner):
            raise PermissionError("revoked P state cannot be moved")
        payload, manifest = self._read_payload("warm", record_id)
        self._metrics["overlap_bytes"] += len(payload) + len(_canonical(manifest)) + 1
        written = self._write_record("cold", record_id, payload, manifest)
        self._metrics["writes"] += 1
        self._metrics["bytes_written"] += written
        self._metrics["movement_bytes"] += written
        shutil.rmtree(self._directory("warm", record_id))
        if owner is not None:
            self._invalidate_p_owner(owner)
        return dict(manifest)

    def promote(self, record_id: str, *, to: str = "warm") -> dict:
        if to != "warm":
            raise ValueError("promotion target is warm")
        owner = self._p_owner(record_id)
        if owner is not None and self._is_p_revoked(owner):
            raise PermissionError("revoked P state cannot be moved")
        payload, manifest = self._read_payload("cold", record_id)
        self._metrics["overlap_bytes"] += len(payload) + len(_canonical(manifest)) + 1
        written = self._write_record("warm", record_id, payload, manifest)
        self._metrics["writes"] += 1
        self._metrics["bytes_written"] += written
        self._metrics["movement_bytes"] += written
        shutil.rmtree(self._directory("cold", record_id))
        if owner is not None:
            self._invalidate_p_owner(owner)
        return dict(manifest)

    # -- authority ---------------------------------------------------------

    def invalidate(self, record_id: str, *, reason: str = "correction",
                   revoked: bool = False) -> int:
        """Invalidate cached projections and make future reads of this record fail."""
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("invalidation reason required")
        self._eligibility[record_id] = self._eligibility.get(record_id, 0) + 1
        self._hot.pop(record_id, None)
        cleared = self.projections.invalidate(record_id)
        self.replay.invalidate(record_id)
        if revoked:
            self._tombstones[record_id] = reason
        return cleared

    def _is_p_revoked(self, owner: str) -> bool:
        # Re-read durable state on every operation: another instance may have
        # revoked or host-restored the owner since the last check.
        if self._p_revoked_path(owner).exists():
            if owner not in self._p_revoked:
                self._p_revoked.add(owner)
                self._invalidate_p_owner(owner)
            return True
        if owner in self._p_revoked:
            self._p_revoked.discard(owner)
        return False

    def _invalidate_p_owner(self, owner: str) -> None:
        prefix = f"p_{owner}_"
        legacy = f"p_{owner}"
        for key in [item for item in self._hot
                    if item.startswith(prefix) or item == legacy]:
            self._hot.pop(key, None)
        self.projections.invalidate_prefix(prefix)
        self.projections.invalidate(legacy)
        self.replay.invalidate_prefix(prefix)
        self.replay.invalidate(legacy)

    @staticmethod
    def _p_owner(record_id: str) -> str | None:
        if not record_id.startswith("p_"):
            return None
        rest = record_id[2:]
        if "_c" in rest:
            owner, _, clock = rest.rpartition("_c")
            if owner and clock.isdigit():
                return owner
        return rest or None

    @staticmethod
    def _contract_state(manifest: Mapping[str, object],
                        contract: ReadContract | None) -> tuple[str, str]:
        if contract is None:
            return "ok", ""
        if manifest.get("producer") != contract.producer:
            return "mismatch", "producer mismatch"
        if manifest.get("representation") != contract.representation:
            return "mismatch", "representation mismatch"
        if manifest.get("scope_digest") != contract.scope_digest:
            return "mismatch", "scope mismatch"
        if contract.generation is not None and \
                manifest.get("generation") != contract.generation:
            return "mismatch", "generation mismatch"
        if contract.cutoff is not None:
            valid_from = manifest.get("valid_from")
            valid_to = manifest.get("valid_to")
            if type(valid_from) is not int or valid_from > contract.cutoff:
                return "not_yet_valid", "record is not valid at the cutoff"
            if valid_to is not None and (type(valid_to) is not int
                                         or contract.cutoff >= valid_to):
                return "expired", "record has expired at the cutoff"
        return "ok", ""

    @staticmethod
    def _authority_manifest_state(manifest: Mapping[str, object],
                                  detail: Mapping[str, object]) -> tuple[str, str]:
        for field in ("producer", "representation", "scope_digest"):
            expected = detail.get(field)
            if expected is not None and manifest.get(field) != expected:
                return "mismatch", f"authority {field} mismatch"
        cutoff = detail.get("cutoff")
        if cutoff is not None:
            valid_from = manifest.get("valid_from")
            valid_to = manifest.get("valid_to")
            if type(cutoff) is not int:
                return "mismatch", "authority cutoff is invalid"
            if type(valid_from) is not int or valid_from > cutoff:
                return "not_yet_valid", "authority cutoff precedes validity"
            if valid_to is not None and (type(valid_to) is not int
                                         or cutoff >= valid_to):
                return "expired", "authority cutoff is past validity"
        expected_generation = detail.get("generation")
        if expected_generation is not None \
                and manifest.get("generation") != expected_generation:
            return "mismatch", "authority generation mismatch"
        return "ok", ""

    def _authority_state(self, record_id: str,
                         authority: Callable[[str], Mapping[str, object] | None] | None,
                         expected_generation: int | None) -> tuple[str, int, dict]:
        owner = self._p_owner(record_id)
        if owner is not None and self._is_p_revoked(owner):
            return "ineligible", self._eligibility.get(record_id, 0), {
                "reason": f"p_revoked:{owner}"}
        if record_id in self._tombstones:
            return "ineligible", self._eligibility.get(record_id, 0), {
                "reason": f"revoked:{self._tombstones[record_id]}"}
        if authority is None:
            current = {"eligible": True, "generation": None}
        else:
            current = authority(record_id)
            if current is None:
                return "ineligible", self._eligibility.get(record_id, 0), {
                    "reason": "no current authority"}
        revision = int(current.get("eligibility_revision",
                                   self._eligibility.get(record_id, 0)))
        if not current.get("eligible", False):
            return "ineligible", revision, {"reason": str(current.get("reason", "authority"))}
        identity_fields = ("producer", "representation", "scope_digest", "cutoff")
        if authority is not None and not any(field in current
                                             for field in identity_fields):
            if expected_generation is None or "generation" not in current:
                return "contract_required", revision, {
                    "reason": "authority must bind manifest identity or an expected generation"}
        if expected_generation is not None and current.get("generation") is not None \
                and int(current["generation"]) != expected_generation:
            return "rejected", revision, {"reason": "generation mismatch"}
        if revision != self._eligibility.get(record_id, 0):
            return "rejected", revision, {"reason": "eligibility revision changed"}
        detail = dict(current)
        detail["generation"] = current.get("generation")
        return "ok", revision, detail

    # -- reads -------------------------------------------------------------

    def _finish(self, result: ReadResult, tier: str | None) -> ReadResult:
        self._metrics["reads"] += 1
        if result.ok:
            self._metrics["bytes_read"] += result.bytes_read
            if tier is not None:
                self._metrics["tier_reads"][tier] += 1
                self._metrics["latency_ns"][tier].append(max(
                    1, int(result.latency_ms * 1e6)))
        else:
            self._metrics["failed_reads"] += 1
            if result.status == "ineligible":
                self._metrics["ineligible"] += 1
            if tier is not None:
                self._metrics["latency_ns"][tier].append(max(
                    1, int(result.latency_ms * 1e6)))
        return result

    def read(self, record_id: str, *,
             authority: Callable[[str], Mapping[str, object] | None] | None = None,
             contract: ReadContract | None = None,
             expected_generation: int | None = None,
             budget_seconds: float | None = None,
             fallback_id: str | None = None,
             fallback_contract: ReadContract | None = None,
             fallback_tokens: int = 0) -> ReadResult:
        _identifier("record_id", record_id)
        if contract is not None and not isinstance(contract, ReadContract):
            raise TypeError("contract must be a ReadContract or None")
        if fallback_contract is not None and not isinstance(fallback_contract,
                                                            ReadContract):
            raise TypeError("fallback_contract must be a ReadContract or None")
        if contract is None and authority is None:
            raise ValueError("tier reads require a ReadContract or authority callback")
        if budget_seconds is not None and budget_seconds < 0:
            raise ValueError("budget must be nonnegative")
        start = time.perf_counter_ns()
        state, revision, detail = self._authority_state(
            record_id, authority, expected_generation)
        if state != "ok":
            return self._finish(ReadResult(
                status=state, record_id=record_id, tier=None, payload=None,
                manifest=None, latency_ms=(time.perf_counter_ns() - start) / 1e6,
                bytes_read=0, reason=detail.get("reason", state)), None)
        cached = self._hot.get(record_id)
        if cached is not None and cached.eligibility_revision == revision:
            if expected_generation is not None \
                    and int(cached.manifest.get("generation", -1)) != expected_generation:
                return self._finish(ReadResult(
                    "rejected", record_id, "hot", None, None,
                    (time.perf_counter_ns() - start) / 1e6, 0,
                    "generation mismatch"), "hot")
            contract_status, contract_reason = self._contract_state(
                cached.manifest, contract)
            if contract_status != "ok":
                return self._finish(ReadResult(
                    contract_status, record_id, "hot", None, None,
                    (time.perf_counter_ns() - start) / 1e6, 0,
                    contract_reason), "hot")
            if contract is None and authority is not None and not (
                    all(field in detail for field in
                        ("producer", "representation", "scope_digest", "cutoff"))
                    and detail.get("cutoff") is not None):
                return self._finish(ReadResult(
                    "contract_required", record_id, "hot", None, None,
                    (time.perf_counter_ns() - start) / 1e6, 0,
                    "authority must bind all manifest identity/validity fields"),
                    "hot")
            authority_status, authority_reason = self._authority_manifest_state(
                cached.manifest, detail)
            if authority_status != "ok":
                return self._finish(ReadResult(
                    authority_status, record_id, "hot", None, None,
                    (time.perf_counter_ns() - start) / 1e6, 0,
                    authority_reason), "hot")
            current_generation = detail.get("generation")
            if current_generation is not None \
                    and int(current_generation) != int(cached.manifest["generation"]):
                return self._finish(ReadResult(
                    "rejected", record_id, "hot", None, None,
                    (time.perf_counter_ns() - start) / 1e6, 0,
                    "authority generation mismatch"), "hot")
            self._hot.move_to_end(record_id)
            latency = (time.perf_counter_ns() - start) / 1e6
            self._metrics["cache_hits"] += 1
            return self._finish(ReadResult(
                "ready", record_id, "hot", cached.payload, dict(cached.manifest),
                latency, len(cached.payload)), "hot")
        self._metrics["cache_misses"] += 1
        tier = "warm" if self._directory("warm", record_id).exists() else (
            "cold" if self._directory("cold", record_id).exists() else None)
        if tier is None:
            if fallback_id is not None:
                return self._fallback(fallback_id, authority, fallback_contract,
                                      start, fallback_tokens, "primary unavailable")
            return self._finish(ReadResult(
                "unavailable", record_id, None, None, None,
                (time.perf_counter_ns() - start) / 1e6, 0,
                "payload or manifest is missing"), None)
        if tier == "cold" and budget_seconds is not None \
                and self.io_delay_seconds > budget_seconds:
            ticket = "tk_" + _sha256_json({
                "record_id": record_id, "tier": tier, "revision": revision,
                "at": _now_ns()})[:32]
            self._tickets[ticket] = {
                "record_id": record_id, "tier": tier, "revision": revision,
                "expected_generation": expected_generation,
                "authority": authority, "contract": contract,
                "fallback_id": fallback_id, "fallback_contract": fallback_contract,
                "fallback_tokens": fallback_tokens,
                "budget_seconds": budget_seconds,
            }
            return self._finish(ReadResult(
                "pending", record_id, tier, None, None,
                (time.perf_counter_ns() - start) / 1e6, 0,
                "bounded cold budget exceeded; resume the ticket", ticket_id=ticket), tier)
        return self._read_tier(tier, record_id, authority, contract,
                               expected_generation, revision, start, fallback_id,
                               fallback_contract, fallback_tokens,
                               fallback_reason="cold read failed")

    def resume(self, ticket_id: str) -> ReadResult:
        ticket = self._tickets.pop(ticket_id, None)
        if ticket is None:
            raise TierError("unknown or consumed cold ticket")
        if self.io_delay_seconds:
            time.sleep(self.io_delay_seconds)
        start = time.perf_counter_ns()
        return self._read_tier(ticket["tier"], ticket["record_id"],
                               ticket["authority"], ticket["contract"],
                               ticket["expected_generation"], ticket["revision"],
                               start, ticket["fallback_id"],
                               ticket["fallback_contract"], ticket["fallback_tokens"],
                               fallback_reason="resumed cold read failed")

    def _read_tier(self, tier: str, record_id: str, authority,
                   contract: ReadContract | None, expected_generation,
                   revision: int, start: int, fallback_id: str | None,
                   fallback_contract: ReadContract | None,
                   fallback_tokens: int, *, fallback_reason: str) -> ReadResult:
        if tier == "cold" and self.io_delay_seconds:
            time.sleep(self.io_delay_seconds)
        try:
            payload, manifest = self._read_payload(tier, record_id)
        except TierError as error:
            reason = str(error)
            corrupt = "digest" in reason or "not canonical" in reason
            if fallback_id is not None:
                return self._fallback(fallback_id, authority, fallback_contract,
                                      start, fallback_tokens, fallback_reason)
            return self._finish(ReadResult(
                "corrupt" if corrupt else "unavailable", record_id, tier, None, None,
                (time.perf_counter_ns() - start) / 1e6, 0, reason), tier)
        elapsed = (time.perf_counter_ns() - start) / 1e6
        if expected_generation is not None \
                and int(manifest.get("generation", -1)) != expected_generation:
            return self._finish(ReadResult(
                "rejected", record_id, tier, None, None, elapsed,
                len(payload), "generation mismatch"), tier)
        contract_status, contract_reason = self._contract_state(manifest, contract)
        if contract_status != "ok":
            return self._finish(ReadResult(
                contract_status, record_id, tier, None, None, elapsed,
                len(payload), contract_reason), tier)
        after, after_revision, detail = self._authority_state(
            record_id, authority, expected_generation)
        if after == "ok":
            if contract is None and authority is not None and not (
                    all(field in detail for field in
                        ("producer", "representation", "scope_digest", "cutoff"))
                    and detail.get("cutoff") is not None):
                return self._finish(ReadResult(
                    "contract_required", record_id, tier, None, None, elapsed,
                    len(payload), "authority must bind all manifest identity/validity fields"), tier)
            authority_status, authority_reason = self._authority_manifest_state(
                manifest, detail)
            if authority_status != "ok":
                return self._finish(ReadResult(
                    authority_status, record_id, tier, None, None, elapsed,
                    len(payload), authority_reason), tier)
        if after != "ok" or after_revision != revision:
            self._hot.pop(record_id, None)
            self.projections.invalidate(record_id)
            if fallback_id is not None:
                return self._fallback(fallback_id, authority, fallback_contract,
                                      start, fallback_tokens,
                                      "authority changed during I/O")
            return self._finish(ReadResult(
                "rejected", record_id, tier, None, None, elapsed,
                len(payload), "authority changed during I/O"), tier)
        self._hot[record_id] = _HotEntry(payload, dict(manifest), revision, False)
        self._hot.move_to_end(record_id)
        self._enforce_hot(protect=record_id)
        return self._finish(ReadResult(
            "ready", record_id, tier, payload, dict(manifest), elapsed,
            len(payload)), tier)

    def _fallback(self, fallback_id: str, authority,
                  contract: ReadContract | None, start: int,
                  tokens: int, reason: str) -> ReadResult:
        state, revision, detail = self._authority_state(fallback_id, authority, None)
        if state != "ok":
            return self._finish(ReadResult(
                "unavailable", fallback_id, None, None, None,
                (time.perf_counter_ns() - start) / 1e6, 0,
                f"fallback ineligible: {detail.get('reason')}"), None)
        tier = "warm" if self._directory("warm", fallback_id).exists() else (
            "cold" if self._directory("cold", fallback_id).exists() else None)
        if tier is None:
            return self._finish(ReadResult(
                "unavailable", fallback_id, None, None, None,
                (time.perf_counter_ns() - start) / 1e6, 0,
                "fallback payload missing"), None)
        try:
            payload, manifest = self._read_payload(tier, fallback_id)
        except TierError as error:
            return self._finish(ReadResult(
                "corrupt", fallback_id, tier, None, None,
                (time.perf_counter_ns() - start) / 1e6, 0, str(error)), tier)
        contract_status, contract_reason = self._contract_state(manifest, contract)
        if contract_status != "ok":
            return self._finish(ReadResult(
                contract_status, fallback_id, tier, None, None,
                (time.perf_counter_ns() - start) / 1e6, len(payload),
                contract_reason), tier)
        after, after_revision, detail = self._authority_state(
            fallback_id, authority, None)
        if after == "ok":
            if contract is None and authority is not None and not (
                    all(field in detail for field in
                        ("producer", "representation", "scope_digest", "cutoff"))
                    and detail.get("cutoff") is not None):
                return self._finish(ReadResult(
                    "contract_required", fallback_id, tier, None, None,
                    (time.perf_counter_ns() - start) / 1e6, len(payload),
                    "authority must bind all manifest identity/validity fields"), tier)
            authority_status, authority_reason = self._authority_manifest_state(
                manifest, detail)
            if authority_status != "ok":
                return self._finish(ReadResult(
                    authority_status, fallback_id, tier, None, None,
                    (time.perf_counter_ns() - start) / 1e6, len(payload),
                    authority_reason), tier)
        if after != "ok" or after_revision != revision:
            self._hot.pop(fallback_id, None)
            self.projections.invalidate(fallback_id)
            return self._finish(ReadResult(
                "rejected", fallback_id, tier, None, None,
                (time.perf_counter_ns() - start) / 1e6, len(payload),
                "fallback authority changed during I/O"), tier)
        self._metrics["fallback_tokens"] += int(tokens)
        return self._finish(ReadResult(
            "fallback", fallback_id, tier, payload, dict(manifest),
            (time.perf_counter_ns() - start) / 1e6, len(payload), reason,
            fallback_tokens=int(tokens)), tier)

    def _enforce_hot(self, *, protect: str | None = None) -> None:
        while sum(len(item.payload) for item in self._hot.values()) > self.hot_capacity_bytes:
            candidates = [key for key in self._hot if key != protect]
            clean = [key for key in candidates if not self._hot[key].dirty]
            if clean:
                del self._hot[clean[0]]
                continue
            if candidates:
                raise DirtyState("dirty P hot state must be published before eviction")
            break

    # -- P state -----------------------------------------------------------

    @staticmethod
    def pack_p_state(*, owner: str, clock: int, encoder: str,
                     lineage: Sequence[str], weights: bytes, moments: bytes) -> bytes:
        _identifier("owner", owner)
        _identifier("encoder", encoder)
        if type(clock) is not int or clock < 0:
            raise ValueError("clock must be a nonnegative integer")
        header = {
            "format": "replica.successors.p_state.v1", "owner": owner,
            "clock": clock, "encoder": encoder, "lineage": list(lineage),
            "weights_sha256": _sha256_bytes(weights),
            "moments_sha256": _sha256_bytes(moments),
            "weights_bytes": len(weights), "moments_bytes": len(moments),
        }
        body = _canonical(header).encode("utf-8")
        return (b"PS1\0" + len(body).to_bytes(8, "little") + body + weights + moments)

    @staticmethod
    def unpack_p_state(payload: bytes) -> dict:
        if not payload.startswith(b"PS1\0"):
            raise TierError("P state payload magic mismatch")
        size = int.from_bytes(payload[4:12], "little")
        header = json.loads(payload[12:12 + size].decode("utf-8"))
        rest = payload[12 + size:]
        weights = rest[:header["weights_bytes"]]
        moments = rest[header["weights_bytes"]:]
        if _sha256_bytes(weights) != header["weights_sha256"] or \
                _sha256_bytes(moments) != header["moments_sha256"]:
            raise TierError("P state weights/moments digest mismatch")
        header["weights"] = weights
        header["moments"] = moments
        return header

    def _p_pointer_path(self, owner: str) -> Path:
        return self.root / "p_current" / f"{owner}.json"

    def _p_revoked_path(self, owner: str) -> Path:
        return self.root / "p_revoked" / f"{owner}.json"

    def _p_record_ids(self, owner: str) -> list[str]:
        prefix = f"p_{owner}_c"
        legacy = f"p_{owner}"
        found = []
        for tier in ("warm", "cold"):
            for directory in (self.root / tier).iterdir():
                if not directory.is_dir():
                    continue
                if directory.name == legacy or directory.name.startswith(prefix):
                    found.append(directory.name)
        return sorted(set(found))

    def _p_versions(self, owner: str) -> list[dict]:
        versions = []
        for record_id in self._p_record_ids(owner):
            tier = "warm" if self._directory("warm", record_id).exists() else "cold"
            try:
                payload, manifest = self._read_payload(tier, record_id)
                state = self.unpack_p_state(payload)
            except TierError:
                continue
            clock = state.get("clock")
            extra = manifest.get("extra", {})
            if state.get("owner") != owner or type(clock) is not int or clock < 0:
                continue
            if manifest.get("generation") != clock or extra.get("clock") != clock:
                continue
            if extra.get("owner") not in (None, owner):
                continue
            state["tier"] = tier
            state["manifest"] = manifest
            state["record_id"] = record_id
            versions.append(state)
        return versions

    def _latest_p(self, owner: str) -> dict:
        versions = sorted(self._p_versions(owner),
                          key=lambda item: (int(item["clock"]),
                                            item["tier"] != "warm"))
        if not versions:
            raise TierError("P state is unavailable")
        return versions[-1]

    def _write_p_pointer(self, owner: str, *, clock: int, record_id: str,
                         manifest: Mapping[str, object]) -> None:
        directory = self.root / "p_current"
        directory.mkdir(exist_ok=True)
        pointer = {
            "format": "replica.successors.p_pointer.v1", "owner": owner,
            "clock": clock, "record_id": record_id,
            "payload_sha256": manifest["payload_sha256"],
            "manifest_sha256": _sha256_json(manifest),
            "committed_at_ns": _now_ns(),
        }
        target = self._p_pointer_path(owner)
        temporary = target.parent / f".{owner}.{time.time_ns()}.tmp"
        try:
            with open(temporary, "w", encoding="utf-8") as handle:
                handle.write(_canonical(pointer) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
        finally:
            if temporary.exists():
                temporary.unlink()
        try:
            directory_fd = os.open(target.parent, os.O_RDONLY)
        except OSError:
            directory_fd = None
        if directory_fd is not None:
            try:
                os.fsync(directory_fd)
            except OSError:
                pass
            finally:
                os.close(directory_fd)

    def _read_p_pointer(self, owner: str) -> dict | None:
        path = self._p_pointer_path(owner)
        if not path.is_file():
            return None
        try:
            pointer = json.loads(path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise TierError("P current pointer is not canonical JSON") from error
        required = {"format", "owner", "clock", "record_id", "payload_sha256",
                    "manifest_sha256", "committed_at_ns"}
        if set(pointer) != required or pointer.get("format") != \
                "replica.successors.p_pointer.v1" or pointer.get("owner") != owner:
            raise TierError("P current pointer schema/identity mismatch")
        if type(pointer.get("clock")) is not int or pointer["clock"] < 0:
            raise TierError("P current pointer clock is invalid")
        if not isinstance(pointer.get("record_id"), str) or not pointer["record_id"]:
            raise TierError("P current pointer record id is invalid")
        for field in ("payload_sha256", "manifest_sha256"):
            value = pointer.get(field)
            if not isinstance(value, str) or len(value) != 64 \
                    or any(char not in "0123456789abcdef" for char in value):
                raise TierError(f"P current pointer {field} is invalid")
        if type(pointer.get("committed_at_ns")) is not int:
            raise TierError("P current pointer timestamp is invalid")
        return pointer

    def publish_p(self, owner: str, *, weights: bytes, moments: bytes, clock: int,
                  encoder: str, lineage: Sequence[str], tier: str = "warm",
                  dirty: bool = False) -> dict:
        _identifier("owner", owner)
        if type(clock) is not int or clock < 0:
            raise ValueError("P clock must be a nonnegative integer")
        if self._is_p_revoked(owner):
            raise PermissionError("P state is revoked; explicit host restoration required")
        versions = self._p_versions(owner)
        if versions and max(int(item["clock"]) for item in versions) > clock:
            raise TierError("P clock must not move backwards")
        record_id = f"p_{owner}_c{clock}"
        existed = any(item["record_id"] == record_id for item in versions)
        payload = self.pack_p_state(owner=owner, clock=clock, encoder=encoder,
                                    lineage=lineage, weights=weights, moments=moments)
        manifest = self.publish(
            record_id, payload, generation=clock, kind="p", producer=f"p:{owner}",
            representation=encoder, scope_digest=_sha256_json({"owner": owner}),
            extra={"clock": clock, "encoder": encoder, "lineage": list(lineage),
                   "owner": owner},
            tier=tier, dirty=dirty)
        try:
            self._write_p_pointer(owner, clock=clock, record_id=record_id,
                                  manifest=manifest)
        except BaseException:
            if not existed:
                self._hot.pop(record_id, None)
                for durable in ("warm", "cold"):
                    target = self._directory(durable, record_id)
                    if target.exists():
                        shutil.rmtree(target)
            raise
        return manifest

    def evict_p(self, owner: str) -> str:
        _identifier("owner", owner)
        if self._is_p_revoked(owner):
            raise PermissionError("revoked P state cannot be evicted as healthy")
        state = self._latest_p(owner)
        key = state["record_id"]
        entry = self._hot.get(key)
        if entry is not None and entry.dirty:
            raise DirtyState("publish dirty P state before eviction")
        self._hot.pop(key, None)
        self._invalidate_p_owner(owner)
        return str(state["tier"])

    def read_p(self, owner: str) -> dict:
        _identifier("owner", owner)
        if self._is_p_revoked(owner):
            raise PermissionError("P state is revoked")
        versions = self._p_versions(owner)
        if not versions:
            raise TierError("P state is unavailable")
        latest = max(versions, key=lambda item: int(item["clock"]))
        pointer = self._read_p_pointer(owner)
        pointer_state = "missing"
        if pointer is not None:
            pointer_state = "stale"
            matches = [item for item in versions
                       if item["record_id"] == pointer["record_id"]]
            if matches:
                target = matches[0]
                if (target["manifest"]["payload_sha256"] != pointer["payload_sha256"]
                        or _sha256_json(target["manifest"]) != pointer["manifest_sha256"]):
                    raise TierError("P current pointer digest mismatch")
                if int(target["clock"]) < int(latest["clock"]):
                    target = latest
                else:
                    pointer_state = "valid"
            else:
                if int(pointer["clock"]) >= int(latest["clock"]):
                    raise TierError("P current pointer target is unavailable")
                target = latest
            if pointer_state == "stale":
                self._write_p_pointer(owner, clock=int(target["clock"]),
                                      record_id=target["record_id"],
                                      manifest=target["manifest"])
                pointer_state = "repaired"
            chosen = target
        else:
            chosen = latest
        state = dict(chosen)
        state["pointer"] = pointer
        state["pointer_state"] = pointer_state
        return state

    def revoke_p(self, owner: str, *, reason: str, authorized_by: str) -> dict:
        """Explicit P revocation; only the host may authorize it."""
        _identifier("owner", owner)
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("P revocation reason required")
        if authorized_by != "host":
            raise PermissionError("P revocation requires host authority")
        directory = self.root / "p_revoked"
        directory.mkdir(exist_ok=True)
        record = {"format": "replica.successors.p_revocation.v1", "owner": owner,
                  "reason": reason, "authorized_by": authorized_by,
                  "revoked_at_ns": _now_ns()}
        target = self._p_revoked_path(owner)
        temporary = target.parent / f".{owner}.{time.time_ns()}.tmp"
        with open(temporary, "w", encoding="utf-8") as handle:
            handle.write(_canonical(record) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        self._p_revoked.add(owner)
        self._invalidate_p_owner(owner)
        return record

    def restore_p(self, owner: str, *, authorized_by: str) -> None:
        """Explicit host restoration; removes the revocation marker only."""
        _identifier("owner", owner)
        if authorized_by != "host":
            raise PermissionError("P restoration requires host authority")
        path = self._p_revoked_path(owner)
        if path.exists():
            path.unlink()
        self._p_revoked.discard(owner)
        self._invalidate_p_owner(owner)

    # -- recovery and metrics ---------------------------------------------

    def recover(self) -> dict:
        quarantined: list[str] = []
        healthy = 0
        damaged: list[str] = []
        revoked_p: list[str] = []
        marker_root = self.root / "p_revoked"
        if marker_root.exists():
            for marker in marker_root.glob("*.json"):
                self._p_revoked.add(marker.stem)
                self._invalidate_p_owner(marker.stem)
        for directory in sorted((self.root / "staging").iterdir()):
            target = self.root / "quarantine" / directory.name
            os.replace(directory, target)
            quarantined.append(directory.name)
        for tier in ("warm", "cold"):
            tier_root = self.root / tier
            for directory in sorted(tier_root.iterdir()):
                if not directory.is_dir():
                    continue
                owner = self._p_owner(directory.name)
                if owner is not None:
                    self._invalidate_p_owner(owner)
                if owner is not None and self._is_p_revoked(owner):
                    revoked_p.append(f"{tier}/{directory.name}")
                    continue
                try:
                    self._read_payload(tier, directory.name)
                    healthy += 1
                except TierError:
                    damaged.append(f"{tier}/{directory.name}")
                    target = self.root / "quarantine" / f"{tier}-{directory.name}-{_now_ns()}"
                    os.replace(directory, target)
        status = "degraded" if quarantined or damaged else "ok"
        return {"status": status, "healthy_records": healthy,
                "quarantined_staging": tuple(quarantined),
                "damaged_records": tuple(damaged),
                "revoked_p_records": tuple(revoked_p)}

    def usage(self) -> dict:
        tiers = {}
        for tier in ("warm", "cold"):
            total = 0
            count = 0
            for directory in (self.root / tier).iterdir():
                if directory.is_dir():
                    count += 1
                    for file in directory.iterdir():
                        if file.is_file():
                            total += file.stat().st_size
            tiers[tier] = {"records": count, "bytes": total}
        hot = sum(len(item.payload) for item in self._hot.values())
        tiers["hot"] = {"records": len(self._hot), "bytes": hot,
                        "capacity_bytes": self.hot_capacity_bytes}
        tiers["limits"] = self.limits.document()
        if hot > self.limits.hot_nonidentity_bytes:
            raise CapacityExceeded("hot tier exceeds its nonidentity budget")
        if tiers["warm"]["bytes"] > self.limits.warm_nonreplay_bytes:
            raise CapacityExceeded("warm tier exceeds its nonreplay budget")
        if tiers["cold"]["bytes"] > self.limits.cold_bytes:
            raise CapacityExceeded("cold tier exceeds its budget")
        return tiers

    def metrics_document(self) -> dict:
        def percentile(values: Sequence[int], fraction: float) -> float | None:
            if not values:
                return None
            ordered = sorted(values)
            index = min(len(ordered) - 1,
                        max(0, int(math.ceil(fraction * len(ordered))) - 1))
            return ordered[index] / 1e6

        return {
            "format": FORMAT,
            "reads": self._metrics["reads"],
            "writes": self._metrics["writes"],
            "cache_hits": self._metrics["cache_hits"],
            "cache_misses": self._metrics["cache_misses"],
            "failed_reads": self._metrics["failed_reads"],
            "ineligible_reads": self._metrics["ineligible"],
            "fallback_tokens": self._metrics["fallback_tokens"],
            "bytes_read": self._metrics["bytes_read"],
            "bytes_written": self._metrics["bytes_written"],
            "temporary_overlap_bytes": self._metrics["overlap_bytes"],
            "movement_bytes": self._metrics["movement_bytes"],
            "reads_by_tier": dict(self._metrics["tier_reads"]),
            "latency_ms": {
                tier: {"p50": percentile(samples, 0.5),
                       "p95": percentile(samples, 0.95),
                       "samples": len(samples)}
                for tier, samples in self._metrics["latency_ns"].items()},
            "usage": self.usage(),
        }

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
        stale = [key for key in self._entries if key[0] == record_id]
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
        return self._entries.pop(record_id, None) is not None


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
        self._tickets: dict[str, dict] = {}
        self.projections = ProjectionCache()
        self.replay = ReplayEligibility()
        self._metrics = {
            "reads": 0, "writes": 0, "cache_hits": 0, "cache_misses": 0,
            "failed_reads": 0, "ineligible": 0, "fallback_tokens": 0,
            "bytes_read": 0, "bytes_written": 0, "overlap_bytes": 0,
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
        if manifest.get("format") != MANIFEST_FORMAT:
            raise TierError("manifest format is unknown")
        return manifest

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
        manifest = {
            "format": MANIFEST_FORMAT, "record_id": record_id, "kind": kind,
            "generation": generation, "producer": producer,
            "representation": representation, "scope_digest": scope_digest,
            "valid_from": valid_from, "valid_to": valid_to,
            "payload_sha256": _sha256_bytes(payload), "bytes": len(payload),
            "created_at_ns": _now_ns(), "extra": dict(extra or {}),
        }
        written = self._write_record(tier, record_id, payload, manifest)
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
        payload, manifest = self._read_payload("warm", record_id)
        self._metrics["overlap_bytes"] += len(payload) + len(_canonical(manifest)) + 1
        self._write_record("cold", record_id, payload, manifest)
        shutil.rmtree(self._directory("warm", record_id))
        self._metrics["tier_reads"]["cold"] += 0
        return dict(manifest)

    def promote(self, record_id: str, *, to: str = "warm") -> dict:
        if to != "warm":
            raise ValueError("promotion target is warm")
        payload, manifest = self._read_payload("cold", record_id)
        self._metrics["overlap_bytes"] += len(payload) + len(_canonical(manifest)) + 1
        self._write_record("warm", record_id, payload, manifest)
        shutil.rmtree(self._directory("cold", record_id))
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

    def _authority_state(self, record_id: str,
                         authority: Callable[[str], Mapping[str, object] | None] | None,
                         expected_generation: int | None) -> tuple[str, int, dict]:
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
        if expected_generation is not None and current.get("generation") is not None \
                and int(current["generation"]) != expected_generation:
            return "rejected", revision, {"reason": "generation mismatch"}
        if revision != self._eligibility.get(record_id, 0):
            return "rejected", revision, {"reason": "eligibility revision changed"}
        return "ok", revision, {"generation": current.get("generation")}

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
             expected_generation: int | None = None,
             budget_seconds: float | None = None,
             fallback_id: str | None = None,
             fallback_tokens: int = 0) -> ReadResult:
        _identifier("record_id", record_id)
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
                return self._fallback(fallback_id, authority, start,
                                      fallback_tokens, "primary unavailable")
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
                "fallback_id": fallback_id, "fallback_tokens": fallback_tokens,
                "budget_seconds": budget_seconds,
            }
            return self._finish(ReadResult(
                "pending", record_id, tier, None, None,
                (time.perf_counter_ns() - start) / 1e6, 0,
                "bounded cold budget exceeded; resume the ticket", ticket_id=ticket), tier)
        return self._read_tier(tier, record_id, authority, expected_generation,
                               revision, start, fallback_id, fallback_tokens,
                               fallback_reason="cold read failed")

    def resume(self, ticket_id: str) -> ReadResult:
        ticket = self._tickets.pop(ticket_id, None)
        if ticket is None:
            raise TierError("unknown or consumed cold ticket")
        if self.io_delay_seconds:
            time.sleep(self.io_delay_seconds)
        start = time.perf_counter_ns()
        return self._read_tier(ticket["tier"], ticket["record_id"], None,
                               ticket["expected_generation"], ticket["revision"],
                               start, ticket["fallback_id"], ticket["fallback_tokens"],
                               fallback_reason="resumed cold read failed")

    def _read_tier(self, tier: str, record_id: str, authority, expected_generation,
                   revision: int, start: int, fallback_id: str | None,
                   fallback_tokens: int, *, fallback_reason: str) -> ReadResult:
        if tier == "cold" and self.io_delay_seconds:
            time.sleep(self.io_delay_seconds)
        try:
            payload, manifest = self._read_payload(tier, record_id)
        except TierError as error:
            reason = str(error)
            corrupt = "digest" in reason or "not canonical" in reason
            if fallback_id is not None:
                return self._fallback(fallback_id, authority, start,
                                      fallback_tokens, fallback_reason)
            return self._finish(ReadResult(
                "corrupt" if corrupt else "unavailable", record_id, tier, None, None,
                (time.perf_counter_ns() - start) / 1e6, 0, reason), tier)
        elapsed = (time.perf_counter_ns() - start) / 1e6
        if expected_generation is not None \
                and int(manifest.get("generation", -1)) != expected_generation:
            return self._finish(ReadResult(
                "rejected", record_id, tier, None, None, elapsed,
                len(payload), "generation mismatch"), tier)
        after, after_revision, detail = self._authority_state(
            record_id, authority, expected_generation)
        if after != "ok" or after_revision != revision:
            self._hot.pop(record_id, None)
            self.projections.invalidate(record_id)
            if fallback_id is not None:
                return self._fallback(fallback_id, authority, start,
                                      fallback_tokens, "authority changed during I/O")
            return self._finish(ReadResult(
                "rejected", record_id, tier, None, None, elapsed,
                len(payload), "authority changed during I/O"), tier)
        self._hot[record_id] = _HotEntry(payload, dict(manifest), revision, False)
        self._hot.move_to_end(record_id)
        self._enforce_hot(protect=record_id)
        return self._finish(ReadResult(
            "ready", record_id, tier, payload, dict(manifest), elapsed,
            len(payload)), tier)

    def _fallback(self, fallback_id: str, authority, start: int,
                  tokens: int, reason: str) -> ReadResult:
        state, _, detail = self._authority_state(fallback_id, authority, None)
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
        self._metrics["fallback_tokens"] += int(tokens)
        return self._finish(ReadResult(
            "fallback", fallback_id, tier, payload, dict(manifest),
            (time.perf_counter_ns() - start) / 1e6, len(payload), reason,
            fallback_tokens=int(tokens)), tier)

    def _enforce_hot(self, *, protect: str | None = None) -> None:
        while sum(len(item.payload) for item in self._hot.values()) > self.hot_capacity_bytes:
            candidate = next((key for key, item in self._hot.items()
                              if key != protect), None)
            if candidate is None:
                break
            entry = self._hot[candidate]
            if entry.dirty:
                raise DirtyState("dirty P hot state must be published before eviction")
            del self._hot[candidate]

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

    def publish_p(self, owner: str, *, weights: bytes, moments: bytes, clock: int,
                  encoder: str, lineage: Sequence[str], tier: str = "warm",
                  dirty: bool = False) -> dict:
        payload = self.pack_p_state(owner=owner, clock=clock, encoder=encoder,
                                    lineage=lineage, weights=weights, moments=moments)
        return self.publish(
            f"p_{owner}", payload, generation=clock, kind="p", producer=f"p:{owner}",
            representation=encoder, scope_digest=_sha256_json({"owner": owner}),
            extra={"clock": clock, "encoder": encoder, "lineage": list(lineage)},
            tier=tier, dirty=dirty)

    def evict_p(self, owner: str) -> str:
        key = f"p_{owner}"
        entry = self._hot.get(key)
        if entry is not None and entry.dirty:
            raise DirtyState("publish dirty P state before eviction")
        self._hot.pop(key, None)
        tier = "warm" if self._directory("warm", key).exists() else (
            "cold" if self._directory("cold", key).exists() else None)
        if tier is None:
            raise TierError("no published P state to evict")
        state = self.unpack_p_state(self._read_payload(tier, key)[0])
        if state["owner"] != owner:
            raise TierError("P state owner mismatch")
        return tier

    def read_p(self, owner: str) -> dict:
        key = f"p_{owner}"
        tier = "warm" if self._directory("warm", key).exists() else (
            "cold" if self._directory("cold", key).exists() else None)
        if tier is None:
            raise TierError("P state is unavailable")
        payload, manifest = self._read_payload(tier, key)
        state = self.unpack_p_state(payload)
        if state["owner"] != owner:
            raise TierError("P state owner mismatch")
        state["tier"] = tier
        state["manifest"] = manifest
        return state

    # -- recovery and metrics ---------------------------------------------

    def recover(self) -> dict:
        quarantined: list[str] = []
        healthy = 0
        damaged: list[str] = []
        for directory in sorted((self.root / "staging").iterdir()):
            target = self.root / "quarantine" / directory.name
            os.replace(directory, target)
            quarantined.append(directory.name)
        for tier in ("warm", "cold"):
            tier_root = self.root / tier
            for directory in sorted(tier_root.iterdir()):
                if not directory.is_dir():
                    continue
                try:
                    self._read_payload(tier, directory.name)
                    self._read_manifest(directory)
                    healthy += 1
                except TierError:
                    damaged.append(f"{tier}/{directory.name}")
        status = "degraded" if quarantined or damaged else "ok"
        return {"status": status, "healthy_records": healthy,
                "quarantined_staging": tuple(quarantined),
                "damaged_records": tuple(damaged)}

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
            "reads_by_tier": dict(self._metrics["tier_reads"]),
            "latency_ms": {
                tier: {"p50": percentile(samples, 0.5),
                       "p95": percentile(samples, 0.95),
                       "samples": len(samples)}
                for tier, samples in self._metrics["latency_ns"].items()},
            "usage": self.usage(),
        }

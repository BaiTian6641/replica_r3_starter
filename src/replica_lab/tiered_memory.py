"""Durable logical tier runtime for the Spark persistent learner.

This module is a separately versioned *engineering* implementation of the
proposal in ``configs/spark_persistent_learner_tiered_design_v1.json`` and
``docs/architecture/28_Spark_Persistent_Learner_and_Tiered_Memory.md``.  It
performs real SQLite and filesystem I/O on small temporary stores.  It does not
load Spark, allocate an expanded expert bank, run a GPU, train a model or
establish native capability.

What is implemented
-------------------
* Capacity profiles ``P1/P4/P16/P32/P64`` and ``P128_stretch`` as arithmetic and
  allocation receipts.  One active expert is the default.  A second active
  expert exists only as a separately named ablation.  Inactive shards are cold
  metadata until an owner explicitly materializes one; registering a bank does
  not create files or GPU-addressable state.
* Seven distinct state classes: exact evidence ``E``, verified procedures
  ``S``, plastic parameters ``P``, offline machinery ``phi``, ephemeral
  workspace ``C``, approved identity ``I`` and measured capability ``K``.  They
  use separate tables, quotas and read paths.
* Stable cluster IDs and immutable generation manifests.  Scope filters run in
  SQL before any semantic key is read.  Each generation manifest is a canonical
  JSON document with a digest, retained for recovery and accounting.
* Retrieval keys (512-wide deterministic signed unigram/bigram hashes) separate
  from ordered readable values (four width-256 float32 slots per record).
  Canonical text and exact-code-point span fallback are explicit, charged
  budgets.  Evidence is capped at eight records and 32 value slots; identity,
  workspace and P-query budgets are separate.
* Logical hot/warm/cold ceilings of 2 GiB / 8 GiB / 64 GiB, an 8 MiB identity
  reservation inside hot and a 256 MiB replay sub-cap inside warm.  Accounting
  is lazy and charged from real byte counts, including database/WAL, indices,
  journals, manifests/publication overlap, checkpoints, replay, workspace,
  artifacts, independent selection/cache copies and observed page-cache
  pressure.  These are logical tiers sharing Apple unified memory; there is no
  fictitious additional VRAM pool.
* Cold recall tickets with bounded wait/resume, post-I/O revalidation of scope,
  generation, revision, revocation and representation version, and refusal to
  mix a pre-correction payload with a post-correction decision boundary.
* Dirty expert state publication (weights, optimizer moments, clock, lineage
  and encoder identity) before eviction; ordered values are never silently
  reconstructed.  Recovery verifies manifests and rebuilds the SQLite index and
  accounting from real files.
* Bounded ignored-artifact storage that detects type from magic bytes, not the
  filename suffix.  Extensionless archives receive size/type/SHA receipts.  The
  existing historical artifacts are neither read from the repository nor
  deleted here.

Deliberate limits
-----------------
* ``native_execution_qualification`` reports ``unsupported``: the current
  FP8/MPS training path is not admitted.  Profile numbers are predictions from
  the design arithmetic, not measurements.
* A correction tombstone or an expert quarantine does not prove exact
  unlearning.  ``CorrectionResult.exact_unlearning_claimed`` is always false.
* Cold latency, page-cache residency, actual GPU residency and host free memory
  are not measured here.  Callers must supply host/driver observations to
  :func:`check_native_guards`; the 50 GiB driver, 512 MiB sustained swap growth
  and 15% free thresholds are unchanged.
* One ``TieredMemoryStore`` owns one SQLite connection.  Use one store per
  thread/process.  Active selections and recall tickets are process-local; a
  fresh process must select again after recovery.
* Workspace and selection copies use in-process buffers.  They are accounted
  as logical copies but this module does not pin physical memory or claim that
  mmap/page cache is free.
"""

from __future__ import annotations

from array import array
from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, replace
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sqlite3
import sys
import tempfile
import time
import uuid

__all__ = [
    "FORMAT",
    "SCHEMA_VERSION",
    "CLUSTER_MANIFEST_FORMAT",
    "EXPERT_MANIFEST_FORMAT",
    "CHECKPOINT_MANIFEST_FORMAT",
    "HOT_CEILING_BYTES",
    "WARM_CEILING_BYTES",
    "COLD_CEILING_BYTES",
    "IDENTITY_RESERVED_BYTES",
    "REPLAY_RESERVED_BYTES",
    "MINIMUM_VOLUME_FREE_BYTES",
    "EVIDENCE_RECORDS_MAX",
    "EVIDENCE_VALUE_SLOTS_MAX",
    "KEY_WIDTH",
    "VALUE_WIDTH",
    "VALUE_SLOTS_PER_RECORD",
    "STATE_CLASSES",
    "CAPACITY_LADDER",
    "STRETCH_PROFILE",
    "NATIVE_GUARDS",
    "NativeGuards",
    "TierCeilings",
    "StateQuotas",
    "RetrievalBudget",
    "CapacityProfile",
    "PROFILES",
    "ExpertAllocationPlan",
    "MemoryScope",
    "EvidenceRecordDraft",
    "SelectedEvidence",
    "ExactSpanHit",
    "RetrievalResult",
    "RecallTicket",
    "RecallResult",
    "RecalledRecord",
    "Outcome",
    "ClusterAdmission",
    "CorrectionResult",
    "RevocationResult",
    "RepresentationInvalidation",
    "IdentityAdmission",
    "IdentityView",
    "WorkspaceReceipt",
    "ArtifactReceipt",
    "ArtifactScan",
    "ExpertLoad",
    "ExpertRecord",
    "ExpertState",
    "ClusterGenerationDocument",
    "RecoveryReport",
    "TieredMemoryStore",
    "TieredMemoryError",
    "SchemaError",
    "CapacityExceeded",
    "BudgetError",
    "ScopeViolation",
    "CorruptState",
    "UnsupportedOperation",
    "InsufficientVolumeFree",
    "build_retrieval_key",
    "key_similarity",
    "pack_values",
    "unpack_values",
    "get_profile",
    "plan_profile",
    "design_contract",
    "native_execution_qualification",
    "exact_unlearning_supported",
    "check_native_guards",
    "detect_file_type",
    "stable_cluster_id",
    "engineering_limits",
]

FORMAT = "replica.tiered_memory.v1"
SCHEMA_VERSION = "1"
CLUSTER_MANIFEST_FORMAT = "replica.tiered_memory.cluster_generation.v1"
EXPERT_MANIFEST_FORMAT = "replica.tiered_memory.expert_publication.v1"
CHECKPOINT_MANIFEST_FORMAT = "replica.tiered_memory.checkpoint_generation.v1"
EXPERIMENT_LABEL = "engineering_tiered_memory_unsupported_native"

HOT_CEILING_BYTES = 2 * 1024**3          # 2 GiB
WARM_CEILING_BYTES = 8 * 1024**3         # 8 GiB
COLD_CEILING_BYTES = 64 * 1024**3        # 64 GiB
IDENTITY_RESERVED_BYTES = 8 * 1024**2    # 8 MiB inside hot
REPLAY_RESERVED_BYTES = 256 * 1024**2    # 256 MiB inside warm
MINIMUM_VOLUME_FREE_BYTES = 25 * 1024**3  # 25 GiB volume headroom
WORKSPACE_QUOTA_BYTES = 256 * 1024**2
ARTIFACT_BUDGET_BYTES = 1 * 1024**3      # local bound, well under the cold cap
FREQUENT_MIN_ACCESSES = 3
FREQUENT_RESERVE_FRACTION = 0.25

EVIDENCE_RECORDS_MAX = 8
EVIDENCE_VALUE_SLOTS_MAX = 32
VALUE_SLOTS_PER_RECORD = 4
VALUE_WIDTH = 256
KEY_WIDTH = 512
P_QUERY_SLOTS = 32
EXACT_TEXT_BUDGET_BYTES = 8192
EXACT_SPAN_BUDGET_BYTES = 2048
MAX_EXACT_SPANS = 8
DEFAULT_COLD_IO_BUDGET_BYTES = 2 * 1024**2

ACTIVE_EXPERTS_INITIAL = 1
ACTIVE_EXPERTS_MAX = 2
TWO_ACTIVE_ABLATION_ID = "two_active_ablation_v1"
NATIVE_OPTIMIZER = "NativeFusedAdamW"
MUON_STATUS = "not_active"

STATE_CLASSES = (
    "E_exact_evidence",
    "S_verified_procedures",
    "P_plastic_parameters",
    "phi_offline_memory_machinery",
    "C_ephemeral_workspace",
    "I_approved_identity",
    "K_measured_capability",
)

AUTHENTICITY_STATES = frozenset({
    "unverified", "source_report", "verified_source", "observed_tool",
    "agent_report", "user_approved_correction",
})
VERIFICATION_STATES = frozenset({
    "unverified", "verified_source", "observed_tool", "verified_human",
    "failed_check", "quarantined",
})
SCOPE_MODES = frozenset({
    "assistant_for_source", "parallel_replica", "source_current", "source_snapshot",
})
PLACEMENTS = frozenset({"warm", "cold"})
PROCEDURE_OUTCOMES = frozenset({"verified", "failed"})
OUTCOME_STATUSES = frozenset({
    "ok", "degraded", "incomplete", "unsupported", "stale", "invalidated",
    "denied", "failed", "in_progress", "truncated",
})
RECALL_STATUSES = frozenset({
    "pending", "in_progress", "ready", "stale", "invalidated", "denied",
    "incomplete", "failed", "unsupported",
})

_CLUSTER_ID_RE = re.compile(r"^cl_[0-9a-f]{32}$")
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


# ---------------------------------------------------------------------------
# Errors and small helpers
# ---------------------------------------------------------------------------

class TieredMemoryError(RuntimeError):
    """Base error for invalid tier operations or incomplete durable state."""


class SchemaError(TieredMemoryError):
    """The on-disk schema, format or declared policy does not match."""


class CapacityExceeded(TieredMemoryError):
    """A logical tier or a sub-quota would be exceeded."""

    def __init__(self, tier: str, category: str, used: int, limit: int, requested: int) -> None:
        self.tier, self.category = tier, category
        self.used, self.limit, self.requested = used, limit, requested
        super().__init__(
            f"{tier}/{category} would exceed its limit: used={used} "
            f"requested={requested} limit={limit}")


class BudgetError(TieredMemoryError):
    """A caller requested an invalid or wider-than-contract budget."""


class ScopeViolation(TieredMemoryError):
    """An operation tried to cross a Replica subject/branch/purpose scope."""


class CorruptState(TieredMemoryError):
    """A durable file or row failed verification; it is preserved, not hidden."""


class UnsupportedOperation(TieredMemoryError):
    """The requested action is deliberately outside this engineering module."""


class InsufficientVolumeFree(TieredMemoryError):
    """The proposed volume headroom is not available."""


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _sha256_json(value: object) -> str:
    return _sha256_bytes(_canonical(value).encode("utf-8"))


def _identifier(name: str, value: object) -> None:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise ValueError(f"{name} must be a nonempty string without NUL")
    if len(value) > 1024:
        raise ValueError(f"{name} is unreasonably long")


def _timestamp(name: str, value: object) -> None:
    if type(value) is not int or value < 0 or value > 2**63 - 1:
        raise ValueError(f"{name} must be a nonnegative integer timestamp")


def _positive_int(name: str, value: object) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _nonnegative_int(name: str, value: object) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


def _now_ns() -> int:
    return time.time_ns()


def _no_symlink(path: Path) -> None:
    if path.is_symlink() or any(parent.is_symlink() for parent in path.parents):
        raise CorruptState(f"tier storage refuses symlinked paths: {path}")


def _atomic_write(path: Path, data: bytes, *, overwrite: bool = False) -> None:
    """Write and fsync a file, then atomically rename it into place.

    A failed write never replaces an existing evidence file.  Interrupted
    ``.tmp-*`` siblings are left for :meth:`TieredMemoryStore.recover` to report
    and remove; they are not mistaken for manifests.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _no_symlink(path.parent)
    if path.exists() and not overwrite:
        raise FileExistsError(f"immutable file already exists: {path}")
    descriptor, temporary = tempfile.mkstemp(prefix=".tmp-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if path.exists() and not overwrite:
            raise FileExistsError(f"immutable file appeared concurrently: {path}")
        os.replace(temporary, path)
        temporary = ""
        try:
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        except OSError:
            pass
    finally:
        if temporary:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


def _copy_atomic(source: Path, destination: Path, expected_sha: str,
                 expected_size: int, *, overwrite: bool = False) -> None:
    """Stream-copy a file with digest verification before atomic publication."""
    source = Path(source)
    destination = Path(destination)
    _no_symlink(source)
    if not source.is_file():
        raise CorruptState(f"artifact source is not a regular file: {source}")
    if source.stat().st_size != expected_size:
        raise CorruptState("artifact source size changed before publication")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"artifact already exists: {destination}")
    descriptor, temporary = tempfile.mkstemp(prefix=".tmp-", dir=destination.parent)
    digest = hashlib.sha256()
    try:
        with os.fdopen(descriptor, "wb") as target, source.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
                target.write(chunk)
            target.flush()
            os.fsync(target.fileno())
        if digest.hexdigest() != expected_sha:
            raise CorruptState("artifact digest changed during copy")
        os.replace(temporary, destination)
        temporary = ""
    finally:
        if temporary:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


def _safe_file_bytes(path: Path) -> int:
    try:
        value = path.lstat()
    except FileNotFoundError:
        return 0
    if not os.path.isfile(path) and not os.path.islink(path):
        return 0
    return value.st_size


def _read_verified(relative_path: str, root: Path, expected_sha: str,
                   expected_bytes: int | None = None) -> bytes:
    path = root / relative_path
    _no_symlink(path)
    try:
        stat = path.stat()
    except FileNotFoundError as error:
        raise CorruptState(f"missing durable payload: {relative_path}") from error
    if not path.is_file():
        raise CorruptState(f"durable payload is not a regular file: {relative_path}")
    if expected_bytes is not None and stat.st_size != expected_bytes:
        raise CorruptState(
            f"durable payload size mismatch: {relative_path} "
            f"expected={expected_bytes} actual={stat.st_size}")
    payload = path.read_bytes()
    if _sha256_bytes(payload) != expected_sha:
        raise CorruptState(f"durable payload digest mismatch: {relative_path}")
    return payload


def _parse_int(value: object, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


# ---------------------------------------------------------------------------
# Logical tier policy and capacity profiles
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class NativeGuards:
    """Unchanged repository guards; caller-supplied observations are required."""

    sampled_driver_bytes: int = 50 * 1024**3
    sustained_swap_growth_bytes: int = 512 * 1024**2
    minimum_host_free_fraction: float = 0.15
    exclusive_mps_worker: bool = True

    def __post_init__(self) -> None:
        _positive_int("sampled_driver_bytes", self.sampled_driver_bytes)
        _positive_int("sustained_swap_growth_bytes", self.sustained_swap_growth_bytes)
        if not math.isfinite(self.minimum_host_free_fraction) or not (
                0 <= self.minimum_host_free_fraction <= 1):
            raise ValueError("minimum_host_free_fraction must be in [0, 1]")
        if not isinstance(self.exclusive_mps_worker, bool):
            raise ValueError("exclusive_mps_worker must be boolean")

    def document(self) -> dict:
        return asdict(self)


NATIVE_GUARDS = NativeGuards()


@dataclass(frozen=True, slots=True)
class TierCeilings:
    """Logical subsystem ceilings; these do not authorize 74 GiB residency."""

    hot_bytes: int = HOT_CEILING_BYTES
    warm_bytes: int = WARM_CEILING_BYTES
    cold_bytes: int = COLD_CEILING_BYTES
    identity_reserved_bytes: int = IDENTITY_RESERVED_BYTES
    replay_reserved_bytes: int = REPLAY_RESERVED_BYTES
    minimum_volume_free_bytes: int = MINIMUM_VOLUME_FREE_BYTES
    workspace_quota_bytes: int = WORKSPACE_QUOTA_BYTES
    artifact_budget_bytes: int = ARTIFACT_BUDGET_BYTES
    frequent_reserve_fraction: float = FREQUENT_RESERVE_FRACTION
    frequent_min_accesses: int = FREQUENT_MIN_ACCESSES

    def __post_init__(self) -> None:
        for name in ("hot_bytes", "warm_bytes", "cold_bytes",
                     "identity_reserved_bytes", "replay_reserved_bytes",
                     "minimum_volume_free_bytes", "workspace_quota_bytes",
                     "artifact_budget_bytes"):
            _nonnegative_int(name, getattr(self, name))
        for name in ("hot_bytes", "warm_bytes", "cold_bytes", "workspace_quota_bytes"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.identity_reserved_bytes > self.hot_bytes:
            raise ValueError("identity reservation cannot exceed the hot ceiling")
        if self.replay_reserved_bytes > self.warm_bytes:
            raise ValueError("replay reservation cannot exceed the warm ceiling")
        if self.artifact_budget_bytes > self.cold_bytes:
            raise ValueError("artifact budget cannot exceed the cold ceiling")
        if not math.isfinite(self.frequent_reserve_fraction) or not (
                0 <= self.frequent_reserve_fraction <= 1):
            raise ValueError("frequent_reserve_fraction must be in [0, 1]")
        if type(self.frequent_min_accesses) is not int or self.frequent_min_accesses < 1:
            raise ValueError("frequent_min_accesses must be a positive integer")

    @property
    def hot_nonidentity_bytes(self) -> int:
        return self.hot_bytes - self.identity_reserved_bytes

    @property
    def warm_nonreplay_bytes(self) -> int:
        return self.warm_bytes - self.replay_reserved_bytes

    def document(self) -> dict:
        return asdict(self)


DEFAULT_CEILINGS = TierCeilings()


@dataclass(frozen=True, slots=True)
class StateQuotas:
    """Explicit, non-borrowing read budgets for different state classes."""

    evidence_records: int = EVIDENCE_RECORDS_MAX
    evidence_value_slots: int = EVIDENCE_VALUE_SLOTS_MAX
    p_query_slots: int = P_QUERY_SLOTS
    identity_bytes: int = IDENTITY_RESERVED_BYTES
    workspace_bytes: int = WORKSPACE_QUOTA_BYTES
    exact_text_bytes: int = EXACT_TEXT_BUDGET_BYTES
    exact_span_bytes: int = EXACT_SPAN_BUDGET_BYTES

    def document(self) -> dict:
        return asdict(self)


STATE_QUOTAS = StateQuotas()


@dataclass(frozen=True, slots=True)
class RetrievalBudget:
    """Per-call budgets; none of these may silently borrow another class."""

    max_records: int = EVIDENCE_RECORDS_MAX
    max_value_slots: int = EVIDENCE_VALUE_SLOTS_MAX
    max_exact_text_bytes: int = EXACT_TEXT_BUDGET_BYTES
    max_span_bytes: int = EXACT_SPAN_BUDGET_BYTES
    max_spans: int = MAX_EXACT_SPANS
    max_cold_bytes: int = DEFAULT_COLD_IO_BUDGET_BYTES
    cold_wait_seconds: float = 2.0
    p_query_slots: int = P_QUERY_SLOTS
    workspace_bytes: int = WORKSPACE_QUOTA_BYTES
    identity_bytes: int = IDENTITY_RESERVED_BYTES

    def __post_init__(self) -> None:
        for name in ("max_records", "max_value_slots", "max_exact_text_bytes",
                     "max_span_bytes", "max_spans", "max_cold_bytes",
                     "p_query_slots", "workspace_bytes", "identity_bytes"):
            _nonnegative_int(name, getattr(self, name))
        if self.max_records > EVIDENCE_RECORDS_MAX:
            raise BudgetError(
                f"evidence selection is capped at {EVIDENCE_RECORDS_MAX} records")
        if self.max_value_slots > EVIDENCE_VALUE_SLOTS_MAX:
            raise BudgetError(
                f"evidence selection is capped at {EVIDENCE_VALUE_SLOTS_MAX} value slots")
        if self.p_query_slots > P_QUERY_SLOTS:
            raise BudgetError("P-query slots are separate and capped at 32")
        if self.identity_bytes > IDENTITY_RESERVED_BYTES:
            raise BudgetError("identity budget cannot exceed the 8 MiB reservation")
        if self.workspace_bytes > WORKSPACE_QUOTA_BYTES:
            raise BudgetError("workspace budget cannot exceed its separate quota")
        if not math.isfinite(self.cold_wait_seconds) or self.cold_wait_seconds < 0:
            raise ValueError("cold_wait_seconds must be finite and nonnegative")

    def quota_plan(self) -> dict:
        return {
            "evidence_records": EVIDENCE_RECORDS_MAX,
            "evidence_value_slots": EVIDENCE_VALUE_SLOTS_MAX,
            "p_query_slots": self.p_query_slots,
            "identity_bytes": self.identity_bytes,
            "workspace_bytes": self.workspace_bytes,
            "exact_text_bytes": self.max_exact_text_bytes,
            "exact_span_bytes": self.max_span_bytes,
            "cold_io_bytes": self.max_cold_bytes,
            "cross_class_borrowing": False,
            "record_values_hold_identity_workspace_or_p": False,
        }


@dataclass(frozen=True, slots=True)
class CapacityProfile:
    profile_id: str
    experts: int
    width: int
    hidden_width: int
    full_parameters: int
    fp32_weight_bytes: int
    all_expert_full_training_bytes: int
    status: str

    def __post_init__(self) -> None:
        _identifier("profile_id", self.profile_id)
        for name in ("experts", "width", "hidden_width", "full_parameters",
                     "fp32_weight_bytes", "all_expert_full_training_bytes"):
            _positive_int(name, getattr(self, name))
        if self.width != VALUE_WIDTH:
            raise ValueError("all declared profiles keep readout width 256")

    @property
    def one_active_full_training_bytes(self) -> int:
        return 3 * self.width * self.hidden_width * 16

    @property
    def d_only_parameters(self) -> int:
        return self.full_parameters // 3

    @property
    def d_only_all_bank_training_bytes(self) -> int:
        return self.full_parameters * 8

    @property
    def one_active_d_only_bytes(self) -> int:
        return self.width * self.hidden_width * 24

    def arithmetic_receipt(self) -> dict:
        return {
            "profile_id": self.profile_id,
            "experts": self.experts,
            "width": self.width,
            "hidden_width": self.hidden_width,
            "formula": "3*N*d*h bias-free gated expert",
            "full_parameters": self.full_parameters,
            "fp32_weight_bytes": self.fp32_weight_bytes,
            "all_expert_fp32_weight_gradient_adam_bytes":
                self.all_expert_full_training_bytes,
            "d_only_parameters": self.d_only_parameters,
            "d_only_all_bank_training_bytes": self.d_only_all_bank_training_bytes,
            "one_active_full_training_bytes": self.one_active_full_training_bytes,
            "one_active_d_only_bytes": self.one_active_d_only_bytes,
            "optimizer": NATIVE_OPTIMIZER,
            "muon": MUON_STATUS,
            "status": self.status,
            "engineering_arithmetic_only": True,
            "allocation": False,
            "native_capability": False,
        }


def _profile(profile_id: str, experts: int, hidden: int, status: str) -> CapacityProfile:
    parameters = 3 * experts * VALUE_WIDTH * hidden
    return CapacityProfile(
        profile_id=profile_id, experts=experts, width=VALUE_WIDTH,
        hidden_width=hidden, full_parameters=parameters,
        fp32_weight_bytes=parameters * 4,
        all_expert_full_training_bytes=parameters * 16,
        status=status)


PROFILES: dict[str, CapacityProfile] = {
    "v1_control": _profile("v1_control", 2, 64, "sealed_historical_shape"),
    "P1": _profile("P1", 1, 512, "proposed_unexecuted"),
    "P4": _profile("P4", 4, 512, "proposed_unexecuted"),
    "P16": _profile("P16", 16, 1024, "proposed_unexecuted"),
    "P32": _profile("P32", 32, 1024, "proposed_unexecuted"),
    "P64": _profile("P64", 64, 1024, "proposed_unexecuted"),
    "P128_stretch": _profile("P128_stretch", 128, 1024, "proposed_unexecuted_stretch"),
}
CAPACITY_LADDER = ("P1", "P4", "P16", "P32", "P64")
STRETCH_PROFILE = "P128_stretch"


def get_profile(profile_id: str) -> CapacityProfile:
    try:
        return PROFILES[profile_id]
    except KeyError as error:
        raise ValueError(f"unknown capacity profile: {profile_id!r}") from error


@dataclass(frozen=True, slots=True)
class ExpertAllocationPlan:
    profile_id: str
    active_experts: int
    stored_experts: int
    inactive_experts: int
    active_full_training_bytes: int
    inactive_full_training_bytes: int
    gpu_active_bytes: int
    gpu_inactive_bytes: int
    inactive_location: str
    all_banks_allocated: bool
    ablation: str | None
    status: str

    def document(self) -> dict:
        value = asdict(self)
        value.update({
            "optimizer": NATIVE_OPTIMIZER,
            "muon": MUON_STATUS,
            "engineering_allocation_receipt": True,
            "native_capability": False,
            "note": ("inactive shards are metadata only; materializing an owner "
                     "is a separate explicit action"),
        })
        return value


def plan_profile(profile_id: str, *, active_experts: int = ACTIVE_EXPERTS_INITIAL,
                 two_active_ablation: bool = False,
                 online_parameters: str = "full") -> ExpertAllocationPlan:
    """Return allocation arithmetic without allocating a bank.

    ``active_experts`` may be 1, or 2 only with ``two_active_ablation=True``.
    A stored bank never materializes its inactive experts.
    """
    profile = get_profile(profile_id)
    if type(active_experts) is not int or not 1 <= active_experts <= ACTIVE_EXPERTS_MAX:
        raise UnsupportedOperation(
            f"active experts must be 1..{ACTIVE_EXPERTS_MAX}; larger work is deferred")
    if active_experts > profile.experts:
        raise ValueError("active experts cannot exceed stored experts")
    if online_parameters not in ("full", "d_only"):
        raise ValueError("online_parameters must be 'full' or 'd_only'")
    if active_experts == 2:
        if not two_active_ablation:
            raise UnsupportedOperation(
                "two active experts require a separate declared ablation")
        if profile_id == "v1_control":
            raise UnsupportedOperation(
                "the sealed v1 control shape is not reopened by this runtime")
        ablation = TWO_ACTIVE_ABLATION_ID
        status = "proposed_unexecuted_ablation"
    else:
        ablation = None
        status = profile.status
    per_expert = (profile.one_active_full_training_bytes if online_parameters == "full"
                  else profile.one_active_d_only_bytes)
    active_bytes = active_experts * per_expert
    inactive_bytes = (profile.experts - active_experts) * per_expert
    return ExpertAllocationPlan(
        profile_id=profile.profile_id,
        active_experts=active_experts,
        stored_experts=profile.experts,
        inactive_experts=profile.experts - active_experts,
        active_full_training_bytes=active_bytes,
        inactive_full_training_bytes=inactive_bytes,
        gpu_active_bytes=active_bytes,
        gpu_inactive_bytes=0,
        inactive_location="cold",
        all_banks_allocated=False,
        ablation=ablation,
        status=status,
    )


# ---------------------------------------------------------------------------
# Scopes, records, retrieval result types
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class MemoryScope:
    subject_id: str
    branch_id: str
    purpose: str
    as_of: int
    mode: str = "assistant_for_source"
    snapshot_at: int | None = None

    def __post_init__(self) -> None:
        for name in ("subject_id", "branch_id", "purpose"):
            _identifier(name, getattr(self, name))
        _timestamp("as_of", self.as_of)
        if self.mode not in SCOPE_MODES:
            raise ValueError("unknown identity mode")
        if self.snapshot_at is not None:
            _timestamp("snapshot_at", self.snapshot_at)
        if self.mode == "source_snapshot" and self.snapshot_at is None:
            raise ValueError("source_snapshot requires snapshot_at")
        if self.mode != "source_snapshot" and self.snapshot_at is not None:
            raise ValueError("snapshot_at is only valid for source_snapshot")

    @property
    def cutoff(self) -> int:
        return (min(self.as_of, self.snapshot_at)
                if self.snapshot_at is not None else self.as_of)

    @property
    def identity_key(self) -> str:
        return _sha256_json({
            "subject_id": self.subject_id,
            "branch_id": self.branch_id,
            "purpose": self.purpose,
            "mode": self.mode,
            "snapshot_at": self.snapshot_at,
        })

    @property
    def scope_key(self) -> str:
        return "sc_" + self.identity_key[:32]

    def document(self) -> dict:
        value = asdict(self)
        value.update({
            "scope_key": self.scope_key,
            "cutoff": self.cutoff,
            "identity_key": self.identity_key,
        })
        return value


def stable_cluster_id(scope: MemoryScope, name: str,
                      cluster_kind: str = "evidence") -> str:
    _identifier("name", name)
    _identifier("cluster_kind", cluster_kind)
    return "cl_" + _sha256_json({
        "scope_identity": scope.identity_key,
        "name": name,
        "cluster_kind": cluster_kind,
        "format": FORMAT,
    })[:32]


def _validate_values(values: Sequence[Sequence[float]]) -> tuple[tuple[float, ...], ...]:
    if len(values) != VALUE_SLOTS_PER_RECORD:
        raise ValueError(f"readable values require exactly {VALUE_SLOTS_PER_RECORD} slots")
    result: list[tuple[float, ...]] = []
    for index, slot in enumerate(values):
        item = tuple(float(value) for value in slot)
        if len(item) != VALUE_WIDTH:
            raise ValueError(f"value slot {index} must have width {VALUE_WIDTH}")
        if not all(math.isfinite(value) for value in item):
            raise ValueError(f"value slot {index} contains nonfinite values")
        result.append(item)
    return tuple(result)


def pack_values(values: Sequence[Sequence[float]]) -> tuple[bytes, ...]:
    """Pack ordered float32 slots exactly as stored; width and order are checked."""
    checked = _validate_values(values)
    output = []
    for slot in checked:
        item = array("f", slot)
        if sys.byteorder != "little":
            item.byteswap()
        output.append(item.tobytes())
    return tuple(output)


def unpack_values(payloads: Sequence[bytes]) -> tuple[tuple[float, ...], ...]:
    if len(payloads) != VALUE_SLOTS_PER_RECORD:
        raise CorruptState("stored record does not have exactly four value slots")
    output = []
    for index, payload in enumerate(payloads):
        if len(payload) != VALUE_WIDTH * 4:
            raise CorruptState(f"value slot {index} has the wrong byte width")
        item = array("f")
        item.frombytes(payload)
        if sys.byteorder != "little":
            item.byteswap()
        values = tuple(float(value) for value in item)
        if not all(math.isfinite(value) for value in values):
            raise CorruptState(f"value slot {index} contains nonfinite values")
        output.append(values)
    return tuple(output)


def build_retrieval_key(text: str, dims: int = KEY_WIDTH) -> bytes:
    """Deterministic signed unigram/bigram key; never derived from stored values.

    This is a retrieval/control key, not a claim that content is encoded into
    usable neural memory.  It is deliberately separate from the readable value
    slots and from canonical text.
    """
    if not isinstance(text, str):
        raise ValueError("retrieval key input must be text")
    if dims != KEY_WIDTH:
        raise ValueError(f"retrieval keys are declared at width {KEY_WIDTH}")
    vector = [0] * dims
    tokens = _TOKEN_RE.findall(text.casefold())
    features: list[tuple[str, float]] = [(token, 1.0) for token in tokens]
    features.extend((f"{left}::{right}", 0.5) for left, right in zip(tokens, tokens[1:]))
    for feature, weight in features:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        raw = int.from_bytes(digest, "little")
        index = raw % dims
        sign = -1 if (raw >> 63) & 1 else 1
        vector[index] += sign * (1 if weight == 1.0 else 1)
    signed = array("b", [max(-127, min(127, value)) for value in vector])
    return signed.tobytes()


def key_similarity(left: bytes, right: bytes) -> float:
    if len(left) != KEY_WIDTH or len(right) != KEY_WIDTH:
        raise ValueError("retrieval keys must both be 512 bytes")
    left_values = array("b")
    left_values.frombytes(left)
    right_values = array("b")
    right_values.frombytes(right)
    dot = math.fsum(a * b for a, b in zip(left_values, right_values))
    left_norm = math.sqrt(math.fsum(a * a for a in left_values))
    right_norm = math.sqrt(math.fsum(b * b for b in right_values))
    if not left_norm or not right_norm:
        return 0.0
    return max(-1.0, min(1.0, dot / (left_norm * right_norm)))


@dataclass(frozen=True, slots=True)
class EvidenceRecordDraft:
    record_id: str
    text: str
    source_id: str
    source_epoch: int
    observed_at: int
    event_at: int | None = None
    valid_from: int | None = None
    valid_to: int | None = None
    authenticity: str = "unverified"
    verification_state: str = "unverified"
    importance: int = 0
    critical: bool = False
    kind: str = "source"
    supersedes: str | None = None
    values: tuple[tuple[float, ...], ...] | None = None
    key: bytes | None = None
    key_producer_id: str = "deterministic_unigram_bigram"
    key_producer_version: str = "v1"
    value_producer_id: str | None = None
    value_producer_version: str | None = None
    source_lineage: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _identifier("record_id", self.record_id)
        _identifier("source_id", self.source_id)
        _identifier("key_producer_id", self.key_producer_id)
        _identifier("key_producer_version", self.key_producer_version)
        _nonnegative_int("source_epoch", self.source_epoch)
        _timestamp("observed_at", self.observed_at)
        if self.event_at is None:
            object.__setattr__(self, "event_at", self.observed_at)
        else:
            _timestamp("event_at", self.event_at)
        if self.valid_from is None:
            object.__setattr__(self, "valid_from", self.observed_at)
        else:
            _timestamp("valid_from", self.valid_from)
        if self.valid_to is not None:
            _timestamp("valid_to", self.valid_to)
            if self.valid_to <= (self.valid_from or 0):
                raise ValueError("valid_to must be after valid_from")
        if not isinstance(self.text, str):
            raise ValueError("text must be an exact string")
        self.text.encode("utf-8")
        if self.authenticity not in AUTHENTICITY_STATES:
            raise ValueError("unknown authenticity state")
        if self.verification_state not in VERIFICATION_STATES:
            raise ValueError("unknown verification state")
        if self.kind not in ("source", "correction"):
            raise ValueError("record kind must be source or correction")
        if type(self.importance) is not int or not 0 <= self.importance <= 100:
            raise ValueError("importance must be an integer in [0, 100]")
        if not isinstance(self.critical, bool):
            raise ValueError("critical must be boolean")
        if self.supersedes is not None:
            _identifier("supersedes", self.supersedes)
            if self.supersedes == self.record_id:
                raise ValueError("a record cannot supersede itself")
        if self.kind == "correction" and self.supersedes is None:
            raise ValueError("a correction must name the record it supersedes")
        if self.kind == "source" and self.supersedes is not None:
            raise ValueError("a source record cannot supersede another record")
        if self.values is not None:
            object.__setattr__(self, "values", _validate_values(self.values))
            if not self.value_producer_id or not self.value_producer_version:
                raise ValueError("readable values require an explicit producer identity")
            _identifier("value_producer_id", self.value_producer_id)
            _identifier("value_producer_version", self.value_producer_version)
        elif self.value_producer_id is not None or self.value_producer_version is not None:
            raise ValueError("value producer identity requires actual values")
        if self.key is not None:
            if not isinstance(self.key, bytes) or len(self.key) != KEY_WIDTH:
                raise ValueError("an explicit retrieval key must be 512 bytes")
        for item in self.source_lineage:
            _identifier("source_lineage item", item)

    @property
    def text_bytes(self) -> int:
        return len(self.text.encode("utf-8"))

    @property
    def text_sha256(self) -> str:
        return _sha256_text(self.text)

    @property
    def key_bytes(self) -> bytes:
        return self.key if self.key is not None else build_retrieval_key(self.text)

    @property
    def representation_version(self) -> str:
        return "rep_" + _sha256_json({
            "key_producer_id": self.key_producer_id,
            "key_producer_version": self.key_producer_version,
            "value_producer_id": self.value_producer_id,
            "value_producer_version": self.value_producer_version,
        })[:24]

    def record_hash(self, scope_key: str) -> str:
        return _sha256_json({
            "format": FORMAT,
            "scope_key": scope_key,
            "record_id": self.record_id,
            "kind": self.kind,
            "supersedes": self.supersedes,
            "text_sha256": self.text_sha256,
            "source_id": self.source_id,
            "source_epoch": self.source_epoch,
            "source_lineage": tuple(self.source_lineage),
            "authenticity": self.authenticity,
            "verification_state": self.verification_state,
            "observed_at": self.observed_at,
            "event_at": self.event_at,
            "valid_from": self.valid_from,
            "valid_to": self.valid_to,
            "importance": self.importance,
            "critical": self.critical,
            "representation_version": self.representation_version,
        })


@dataclass(frozen=True, slots=True)
class SelectedEvidence:
    record_id: str
    cluster_id: str
    cluster_generation: int
    score: float
    text: str | None
    text_sha256: str
    text_bytes: int
    text_included: bool
    placement: str
    values: tuple[tuple[float, ...], ...]
    values_available: bool
    value_slots: int
    key_sha256: str
    representation_version: str
    revision: int
    source_id: str
    source_epoch: int
    verification_state: str
    observed_at: int
    valid_to: int | None
    critical: bool
    retrieval_count: int

    @property
    def value_slot_count(self) -> int:
        return self.value_slots


@dataclass(frozen=True, slots=True)
class ExactSpanHit:
    record_id: str
    phrase: str
    start: int
    end: int
    text: str
    start_sha256: str
    source_hash: str
    record_hash: str
    placement: str

    def document(self) -> dict:
        value = asdict(self)
        value["exclusive_end"] = True
        value["text_bytes"] = len(self.text.encode("utf-8"))
        return value


@dataclass(frozen=True, slots=True)
class Outcome:
    status: str
    reason: str
    detail: Mapping[str, object] = field(default_factory=dict)
    retryable: bool = False

    def __post_init__(self) -> None:
        if self.status not in OUTCOME_STATUSES:
            raise ValueError(f"unknown outcome status: {self.status}")
        _identifier("reason", self.reason)
        if not isinstance(self.detail, Mapping):
            raise ValueError("outcome detail must be a mapping")
        if not isinstance(self.retryable, bool):
            raise ValueError("retryable must be boolean")

    @property
    def ok(self) -> bool:
        return self.status in ("ok", "degraded")

    def document(self) -> dict:
        return {
            "status": self.status,
            "reason": self.reason,
            "retryable": self.retryable,
            "detail": dict(self.detail),
        }


def _outcome(status: str, reason: str, **detail: object) -> Outcome:
    return Outcome(status=status, reason=reason, detail=dict(detail))


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    selection_id: str
    scope: MemoryScope
    query: str
    outcome: Outcome
    records: tuple[SelectedEvidence, ...]
    exact_spans: tuple[ExactSpanHit, ...]
    generation_pins: tuple[tuple[str, int], ...]
    epoch: int
    candidate_count: int
    truncated: bool
    fallback_mode: str
    used_text_bytes: int
    used_value_slots: int
    used_span_bytes: int
    requires_cold_recall: bool
    cold_record_ids: tuple[str, ...]
    representation_stale: bool
    budget_plan: Mapping[str, object]

    @property
    def record_ids(self) -> tuple[str, ...]:
        return tuple(record.record_id for record in self.records)

    @property
    def scores(self) -> tuple[float, ...]:
        return tuple(record.score for record in self.records)

    def document(self) -> dict:
        return {
            "selection_id": self.selection_id,
            "scope": self.scope.document(),
            "query": self.query,
            "outcome": self.outcome.document(),
            "record_ids": self.record_ids,
            "scores": self.scores,
            "exact_spans": tuple(hit.document() for hit in self.exact_spans),
            "generation_pins": self.generation_pins,
            "epoch": self.epoch,
            "candidate_count": self.candidate_count,
            "truncated": self.truncated,
            "fallback_mode": self.fallback_mode,
            "used_text_bytes": self.used_text_bytes,
            "used_value_slots": self.used_value_slots,
            "used_span_bytes": self.used_span_bytes,
            "requires_cold_recall": self.requires_cold_recall,
            "cold_record_ids": self.cold_record_ids,
            "representation_stale": self.representation_stale,
            "budget": dict(self.budget_plan),
        }


@dataclass(frozen=True, slots=True)
class ClusterAdmission:
    outcome: Outcome
    cluster_id: str
    generation: int
    manifest_sha256: str
    record_ids: tuple[str, ...]
    placement: str

    def document(self) -> dict:
        return {
            "outcome": self.outcome.document(),
            "cluster_id": self.cluster_id,
            "generation": self.generation,
            "manifest_sha256": self.manifest_sha256,
            "record_ids": self.record_ids,
            "placement": self.placement,
        }


@dataclass(frozen=True, slots=True)
class CorrectionResult:
    outcome: Outcome
    old_record_id: str
    new_record_id: str
    cluster_id: str
    generation: int
    manifest_sha256: str
    quarantined_experts: tuple[str, ...]
    invalidated_procedures: tuple[tuple[str, int], ...]
    stale_representations: int
    epoch: int
    exact_unlearning_claimed: bool = False

    def document(self) -> dict:
        return {
            "outcome": self.outcome.document(),
            "old_record_id": self.old_record_id,
            "new_record_id": self.new_record_id,
            "cluster_id": self.cluster_id,
            "generation": self.generation,
            "manifest_sha256": self.manifest_sha256,
            "quarantined_experts": self.quarantined_experts,
            "invalidated_procedures": self.invalidated_procedures,
            "stale_representations": self.stale_representations,
            "epoch": self.epoch,
            "exact_unlearning_claimed": self.exact_unlearning_claimed,
            "quarantine_semantics": (
                "quarantine then authenticated reset or verified replay; "
                "a tombstone does not erase trained weights"),
        }


@dataclass(frozen=True, slots=True)
class RevocationResult:
    outcome: Outcome
    record_id: str
    cluster_id: str
    generation: int
    quarantined_experts: tuple[str, ...]
    invalidated_procedures: tuple[tuple[str, int], ...]
    stale_representations: int
    epoch: int
    exact_unlearning_claimed: bool = False

    def document(self) -> dict:
        value = asdict(self)
        value["outcome"] = self.outcome.document()
        return value


@dataclass(frozen=True, slots=True)
class RepresentationInvalidation:
    outcome: Outcome
    producer_version: str
    records: tuple[str, ...]
    clusters: tuple[tuple[str, int], ...]
    stale_keys: int
    stale_values: int
    evicted_cache_entries: tuple[str, ...]
    reencode_required: bool

    def document(self) -> dict:
        return {
            "outcome": self.outcome.document(),
            "producer_version": self.producer_version,
            "records": self.records,
            "clusters": self.clusters,
            "stale_keys": self.stale_keys,
            "stale_values": self.stale_values,
            "evicted_cache_entries": self.evicted_cache_entries,
            "reencode_required": self.reencode_required,
        }


@dataclass(frozen=True, slots=True)
class RecallTicket:
    ticket_id: str
    selection_id: str
    scope: MemoryScope
    record_ids: tuple[str, ...]
    pinned: tuple[Mapping[str, object], ...]
    required_bytes: int
    max_bytes: int
    status: str
    outcome: Outcome
    requested_at_ns: int
    io_bytes: int = 0
    decision_boundary: int = 0

    def __post_init__(self) -> None:
        if self.status not in RECALL_STATUSES:
            raise ValueError("unknown recall ticket status")

    @property
    def pending(self) -> bool:
        return self.status in ("pending", "in_progress")

    def document(self) -> dict:
        return {
            "ticket_id": self.ticket_id,
            "selection_id": self.selection_id,
            "scope_key": self.scope.scope_key,
            "record_ids": self.record_ids,
            "required_bytes": self.required_bytes,
            "max_bytes": self.max_bytes,
            "status": self.status,
            "outcome": self.outcome.document(),
            "requested_at_ns": self.requested_at_ns,
            "io_bytes": self.io_bytes,
            "decision_boundary": self.decision_boundary,
            "pins": tuple(dict(pin) for pin in self.pinned),
            "wait_resume": True,
            "artificial_delay": False,
        }


@dataclass(frozen=True, slots=True)
class RecalledRecord:
    record_id: str
    text: str
    text_sha256: str
    values: tuple[tuple[float, ...], ...]
    value_slots: int
    value_sha256: tuple[str, ...]
    placement: str

    def document(self) -> dict:
        return {
            "record_id": self.record_id,
            "text": self.text,
            "text_sha256": self.text_sha256,
            "value_slots": self.value_slots,
            "value_sha256": self.value_sha256,
            "placement": self.placement,
        }


@dataclass(frozen=True, slots=True)
class RecallResult:
    ticket_id: str
    selection_id: str
    status: str
    outcome: Outcome
    records: tuple[RecalledRecord, ...]
    io_bytes: int
    checks: tuple[str, ...]
    new_decision_boundary: bool
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.status not in RECALL_STATUSES:
            raise ValueError("unknown recall result status")

    @property
    def ready(self) -> bool:
        return self.status == "ready"

    def document(self) -> dict:
        return {
            "ticket_id": self.ticket_id,
            "selection_id": self.selection_id,
            "status": self.status,
            "outcome": self.outcome.document(),
            "records": tuple(record.document() for record in self.records),
            "io_bytes": self.io_bytes,
            "checks": self.checks,
            "new_decision_boundary": self.new_decision_boundary,
            "warnings": self.warnings,
        }


@dataclass(frozen=True, slots=True)
class IdentityAdmission:
    outcome: Outcome
    identity_id: str
    version: int
    sha256: str = ""
    bytes: int = 0

    def document(self) -> dict:
        return {
            "outcome": self.outcome.document(),
            "identity_id": self.identity_id,
            "version": self.version,
            "sha256": self.sha256,
            "bytes": self.bytes,
        }


@dataclass(frozen=True, slots=True)
class IdentityView:
    outcome: Outcome
    identity_id: str | None
    version: int | None
    text: str | None
    sha256: str | None
    bytes: int
    authority: str | None

    def document(self) -> dict:
        return {
            "outcome": self.outcome.document(),
            "identity_id": self.identity_id,
            "version": self.version,
            "text": self.text,
            "sha256": self.sha256,
            "bytes": self.bytes,
            "authority": self.authority,
        }


@dataclass(frozen=True, slots=True)
class WorkspaceReceipt:
    outcome: Outcome
    task_id: str
    path: str
    sha256: str
    bytes: int
    resumable: bool
    tier: str

    def document(self) -> dict:
        return asdict(self) | {"outcome": self.outcome.document()}


@dataclass(frozen=True, slots=True)
class ArtifactReceipt:
    status: str
    artifact_id: str
    path: str
    bytes: int
    sha256: str
    detected_type: str
    magic_hex: str
    pinned: bool
    label: str | None
    outcome: Outcome

    def document(self) -> dict:
        return {
            "status": self.status,
            "artifact_id": self.artifact_id,
            "path": self.path,
            "bytes": self.bytes,
            "sha256": self.sha256,
            "detected_type": self.detected_type,
            "magic_hex": self.magic_hex,
            "pinned": self.pinned,
            "label": self.label,
            "outcome": self.outcome.document(),
            "extension_used_for_detection": False,
        }


@dataclass(frozen=True, slots=True)
class ArtifactScan:
    outcome: Outcome
    receipts: tuple[ArtifactReceipt, ...]
    total_bytes: int
    over_budget: bool

    def document(self) -> dict:
        return {
            "outcome": self.outcome.document(),
            "receipts": tuple(receipt.document() for receipt in self.receipts),
            "total_bytes": self.total_bytes,
            "over_budget": self.over_budget,
        }


@dataclass(frozen=True, slots=True)
class ExpertRecord:
    expert_id: str
    profile_id: str
    parameter_count: int
    location: str
    dirty: bool
    quarantine_status: str
    publication_generation: int
    encoder_id: str
    optimizer: str
    ablation: str | None

    def document(self) -> dict:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ExpertState:
    expert_id: str
    weights: bytes
    moments: bytes
    clock: int
    lineage: tuple[str, ...]
    encoder_id: str
    generation: int
    source: str
    dirty: bool
    quarantine_status: str
    weights_sha256: str
    moments_sha256: str

    def document(self) -> dict:
        return {
            "expert_id": self.expert_id,
            "clock": self.clock,
            "lineage": self.lineage,
            "encoder_id": self.encoder_id,
            "generation": self.generation,
            "source": self.source,
            "dirty": self.dirty,
            "quarantine_status": self.quarantine_status,
            "weights_bytes": len(self.weights),
            "moments_bytes": len(self.moments),
            "weights_sha256": self.weights_sha256,
            "moments_sha256": self.moments_sha256,
            "optimizer": NATIVE_OPTIMIZER,
            "muon": MUON_STATUS,
        }


@dataclass(frozen=True, slots=True)
class ExpertLoad:
    outcome: Outcome
    state: ExpertState | None

    def document(self) -> dict:
        return {
            "outcome": self.outcome.document(),
            "state": None if self.state is None else self.state.document(),
        }


@dataclass(frozen=True, slots=True)
class ClusterGenerationDocument:
    cluster_id: str
    generation: int
    path: str
    sha256: str
    document: Mapping[str, object]

    def as_dict(self) -> dict:
        return dict(self.document)


@dataclass(frozen=True, slots=True)
class RecoveryReport:
    outcome: Outcome
    clusters_recovered: tuple[str, ...]
    generations_recovered: int
    records_recovered: int
    experts_recovered: tuple[str, ...]
    corrupt_manifests: tuple[str, ...]
    corrupt_artifacts: tuple[str, ...]
    ignored_temporaries: tuple[str, ...]
    accounting_rebuilt: bool
    notes: tuple[str, ...]

    def document(self) -> dict:
        return {
            "outcome": self.outcome.document(),
            "clusters_recovered": self.clusters_recovered,
            "generations_recovered": self.generations_recovered,
            "records_recovered": self.records_recovered,
            "experts_recovered": self.experts_recovered,
            "corrupt_manifests": self.corrupt_manifests,
            "corrupt_artifacts": self.corrupt_artifacts,
            "ignored_temporaries": self.ignored_temporaries,
            "accounting_rebuilt": self.accounting_rebuilt,
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# Design-contract and unsupported-capability receipts
# ---------------------------------------------------------------------------

def design_contract(config_path: Path | str | None = None) -> dict:
    """Bind module constants to the tracked design JSON without allocating."""
    path = Path(config_path) if config_path is not None else (
        Path(__file__).resolve().parents[2]
        / "configs/spark_persistent_learner_tiered_design_v1.json")
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("format") != "replica.spark_persistent_learner_tiered_design.v1":
        raise SchemaError("unexpected tiered design config format")
    if tuple(config.get("state_classes", ())) != STATE_CLASSES:
        raise SchemaError("state classes drifted from the design config")
    profiles = {item["id"]: item for item in config.get("capacity_profiles", ())}
    for profile_id, profile in PROFILES.items():
        declared = profiles.get(profile_id)
        if declared is None:
            raise SchemaError(f"profile missing from design config: {profile_id}")
        for key, value in (
            ("experts", profile.experts),
            ("width", profile.width),
            ("hidden_width", profile.hidden_width),
            ("full_expert_parameters", profile.full_parameters),
            ("fp32_weight_bytes", profile.fp32_weight_bytes),
            ("all_expert_fp32_weight_gradient_adam_bytes",
             profile.all_expert_full_training_bytes),
        ):
            if declared.get(key) != value:
                raise SchemaError(f"profile arithmetic drift for {profile_id}:{key}")
    guards = config.get("native_guards", {})
    if (guards.get("sampled_driver_bytes") != NATIVE_GUARDS.sampled_driver_bytes
            or guards.get("sustained_swap_growth_bytes")
            != NATIVE_GUARDS.sustained_swap_growth_bytes
            or guards.get("minimum_host_free_fraction")
            != NATIVE_GUARDS.minimum_host_free_fraction
            or guards.get("exclusive_mps_worker") is not True):
        raise SchemaError("native guard drift")
    tiers = config.get("tiers", {})
    if (tiers.get("hot", {}).get("max_memory_subsystem_bytes") != HOT_CEILING_BYTES
            or tiers.get("warm", {}).get("max_memory_subsystem_bytes") != WARM_CEILING_BYTES
            or tiers.get("cold", {}).get("max_memory_subsystem_bytes") != COLD_CEILING_BYTES
            or tiers.get("hot", {}).get("identity_reserved_bytes_included")
            != IDENTITY_RESERVED_BYTES
            or tiers.get("warm", {}).get("replay_bytes_included")
            != REPLAY_RESERVED_BYTES
            or tiers.get("cold", {}).get("minimum_volume_free_bytes")
            != MINIMUM_VOLUME_FREE_BYTES):
        raise SchemaError("tier ceiling drift")
    readout = config.get("readout", {})
    if (readout.get("width") != VALUE_WIDTH
            or readout.get("ordered_slots_per_record") != VALUE_SLOTS_PER_RECORD
            or readout.get("record_value_slots_max") != EVIDENCE_VALUE_SLOTS_MAX
            or readout.get("selected_records_max") != EVIDENCE_RECORDS_MAX
            or readout.get("key_width") != KEY_WIDTH
            or readout.get("key_and_value_representations_separate") is not True):
        raise SchemaError("readout separation/budget drift")
    policy = config.get("capacity_policy", {})
    if (policy.get("active_experts_initial") != ACTIVE_EXPERTS_INITIAL
            or policy.get("active_experts_expansion_max") != ACTIVE_EXPERTS_MAX
            or policy.get("active_experts_increase_requires_separate_ablation") is not True
            or policy.get("allocate_inactive_bank_on_gpu") is not False
            or policy.get("first_learnability_profile") not in CAPACITY_LADDER
            or policy.get("main_design_target") not in CAPACITY_LADDER
            or policy.get("stretch_profile") != STRETCH_PROFILE):
        raise SchemaError("capacity policy drift")
    return {
        "format": "replica.tiered_memory.design_contract.v1",
        "config_format": config["format"],
        "config_sha256": _sha256_bytes(path.read_bytes()),
        "profiles_verified": tuple(PROFILES),
        "capacity_ladder": CAPACITY_LADDER,
        "stretch_profile": STRETCH_PROFILE,
        "state_classes": STATE_CLASSES,
        "ceilings": DEFAULT_CEILINGS.document(),
        "native_guards": NATIVE_GUARDS.document(),
        "allocation_or_training": False,
        "native_capability": False,
    }


def native_execution_qualification() -> Outcome:
    return _outcome(
        "unsupported",
        "native_fp8_mps_training_path_unqualified",
        source="XHToken/Spark-X2.6-4B-FP8",
        revision="0318d6f6ab070c7b5b43052db571489b2508006c",
        bf16_control_and_dequantized_derivative="separate_explicit_identities",
        training_or_inference_confidence=None,
        note="this module operates on synthetic engineering files only",
    )


def exact_unlearning_supported() -> Outcome:
    return _outcome(
        "unsupported",
        "metadata_tombstone_does_not_unlearn_weights",
        required=("quarantine", "authenticated reset", "verified replay",
                  "old-knowledge retention tests"),
        claimed_exact_unlearning=False,
    )


def check_native_guards(*, driver_bytes: int, swap_growth_bytes: int,
                        host_free_fraction: float) -> Outcome:
    """Evaluate caller-supplied readings against the unchanged native guards."""
    _nonnegative_int("driver_bytes", driver_bytes)
    _nonnegative_int("swap_growth_bytes", swap_growth_bytes)
    if not math.isfinite(host_free_fraction) or not 0 <= host_free_fraction <= 1:
        raise ValueError("host_free_fraction must be in [0, 1]")
    failures = []
    if driver_bytes > NATIVE_GUARDS.sampled_driver_bytes:
        failures.append("sampled_driver_limit_exceeded")
    if swap_growth_bytes > NATIVE_GUARDS.sustained_swap_growth_bytes:
        failures.append("sustained_swap_growth_exceeded")
    if host_free_fraction < NATIVE_GUARDS.minimum_host_free_fraction:
        failures.append("minimum_host_free_fraction_violated")
    if failures:
        return _outcome(
            "failed", "native_guard_stop", failures=tuple(failures),
            driver_bytes=driver_bytes, swap_growth_bytes=swap_growth_bytes,
            host_free_fraction=host_free_fraction,
            measurement="caller_supplied_not_sampled_here")
    return _outcome(
        "ok", "native_guards_satisfied",
        driver_bytes=driver_bytes, swap_growth_bytes=swap_growth_bytes,
        host_free_fraction=host_free_fraction,
        measurement="caller_supplied_not_sampled_here")


def engineering_limits() -> dict:
    return {
        "format": FORMAT,
        "schema_version": SCHEMA_VERSION,
        "engineering_evidence": True,
        "native_capability": False,
        "no_model_loaded": True,
        "no_expanded_bank_allocated": True,
        "tier_ceilings_are_logical_not_physical_vram": True,
        "exact_unlearning_claimed": False,
        "cold_latency_measured": False,
        "page_cache_residency_measured": False,
        "identity_data": "synthetic until user-approved Replica records exist",
        "base_model_bytes_in_cold_ceiling": False,
        "allocation": "lazy",
        "remaining_unsupported": (
            "native FP8/MPS training, actual GPU residency, native readout "
            "benefit, distributed writers, and exact unlearning"),
    }


# ---------------------------------------------------------------------------
# File type detection for ignored artifacts (size/type, not suffix)
# ---------------------------------------------------------------------------

_MAGIC_TYPES: tuple[tuple[bytes, str], ...] = (
    (b"PK\x03\x04", "zip_archive"),
    (b"PK\x05\x06", "zip_archive"),
    (b"PK\x07\x08", "zip_archive"),
    (b"\x1f\x8b", "gzip_stream"),
    (b"BZh", "bzip2_stream"),
    (b"\xfd7zXZ\x00", "xz_stream"),
    (b"\x28\xb5\x2f\xfd", "zstd_stream"),
    (b"7z\xbc\xaf\x27\x1c", "7z_archive"),
    (b"Rar!\x1a\x07", "rar_archive"),
    (b"SQLite format 3\x00", "sqlite_database"),
    (b"\x89PNG\r\n\x1a\n", "png_image"),
    (b"\xff\xd8\xff", "jpeg_image"),
    (b"RIFF", "riff_container"),
    (b"\x7fELF", "elf_binary"),
    (b"\xcf\xfa\xed\xfe", "mach_o_binary"),
)


def _sniff_bytes(head: bytes) -> str:
    for magic, name in _MAGIC_TYPES:
        if head.startswith(magic):
            return name
    if len(head) >= 262 and head[257:262] == b"ustar":
        return "tar_archive"
    stripped = head.lstrip()
    if stripped.startswith(b"{") or stripped.startswith(b"["):
        return "json_text_candidate"
    if not head:
        return "empty"
    try:
        head.decode("utf-8")
    except UnicodeDecodeError:
        return "unknown_binary"
    return "utf8_text_candidate"


def detect_file_type(path: Path | str) -> tuple[str, str, int]:
    """Return (detected type, magic hex prefix, size). Extension is ignored."""
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise CorruptState(f"artifact is not an independent regular file: {path}")
    size = path.stat().st_size
    with path.open("rb") as stream:
        head = stream.read(512)
    kind = _sniff_bytes(head)
    return kind, head[:32].hex(), size


# ---------------------------------------------------------------------------
# SQLite schema
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS clusters (
    cluster_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    cluster_kind TEXT NOT NULL,
    scope_key TEXT NOT NULL,
    scope_json TEXT NOT NULL,
    current_generation INTEGER NOT NULL,
    manifest_sha256 TEXT,
    record_count INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL,
    created_at_ns INTEGER NOT NULL,
    updated_at_ns INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS clusters_scope ON clusters(scope_key, status);

CREATE TABLE IF NOT EXISTS cluster_generations (
    cluster_id TEXT NOT NULL,
    generation INTEGER NOT NULL,
    scope_key TEXT NOT NULL,
    manifest_path TEXT NOT NULL,
    manifest_sha256 TEXT NOT NULL,
    parent_generation INTEGER,
    record_count INTEGER NOT NULL,
    created_at_ns INTEGER NOT NULL,
    PRIMARY KEY (cluster_id, generation)
);
CREATE TRIGGER IF NOT EXISTS cluster_generations_no_update
BEFORE UPDATE ON cluster_generations
BEGIN SELECT RAISE(ABORT, 'cluster generation manifests are immutable'); END;
CREATE TRIGGER IF NOT EXISTS cluster_generations_no_delete
BEFORE DELETE ON cluster_generations
BEGIN SELECT RAISE(ABORT, 'cluster generation manifests are retained'); END;

CREATE TABLE IF NOT EXISTS records (
    record_id TEXT PRIMARY KEY,
    cluster_id TEXT NOT NULL,
    record_index INTEGER NOT NULL,
    record_hash TEXT NOT NULL,
    scope_key TEXT NOT NULL,
    kind TEXT NOT NULL,
    status TEXT NOT NULL,
    superseded_by TEXT,
    revoked_reason TEXT,
    text_path TEXT NOT NULL,
    text_sha256 TEXT NOT NULL,
    text_bytes INTEGER NOT NULL,
    source_id TEXT NOT NULL,
    source_epoch INTEGER NOT NULL,
    source_lineage TEXT NOT NULL,
    authenticity TEXT NOT NULL,
    verification_state TEXT NOT NULL,
    observed_at INTEGER NOT NULL,
    event_at INTEGER NOT NULL,
    valid_from INTEGER,
    valid_to INTEGER,
    importance INTEGER NOT NULL,
    critical INTEGER NOT NULL,
    placement TEXT NOT NULL,
    revision INTEGER NOT NULL,
    representation_version TEXT NOT NULL,
    created_at_ns INTEGER NOT NULL,
    updated_at_ns INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS records_scope ON records(scope_key, status, observed_at);
CREATE INDEX IF NOT EXISTS records_cluster ON records(cluster_id, record_index);

CREATE TABLE IF NOT EXISTS record_keys (
    record_id TEXT PRIMARY KEY,
    key_path TEXT NOT NULL,
    key_bytes INTEGER NOT NULL,
    key_sha256 TEXT NOT NULL,
    key_width INTEGER NOT NULL,
    producer_id TEXT NOT NULL,
    producer_version TEXT NOT NULL,
    status TEXT NOT NULL,
    updated_at_ns INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS record_values (
    record_id TEXT NOT NULL,
    slot_index INTEGER NOT NULL,
    value_path TEXT NOT NULL,
    value_bytes INTEGER NOT NULL,
    value_sha256 TEXT NOT NULL,
    value_width INTEGER NOT NULL,
    dtype TEXT NOT NULL,
    producer_id TEXT NOT NULL,
    producer_version TEXT NOT NULL,
    status TEXT NOT NULL,
    updated_at_ns INTEGER NOT NULL,
    PRIMARY KEY (record_id, slot_index)
);

CREATE TABLE IF NOT EXISTS record_spans (
    span_id TEXT PRIMARY KEY,
    record_id TEXT NOT NULL,
    start_offset INTEGER NOT NULL,
    end_offset INTEGER NOT NULL,
    span_sha256 TEXT NOT NULL,
    text_sha256 TEXT NOT NULL,
    created_at_ns INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS record_spans_record ON record_spans(record_id);

CREATE TABLE IF NOT EXISTS cluster_links (
    cluster_id TEXT NOT NULL,
    linked_cluster_id TEXT NOT NULL,
    relation TEXT NOT NULL,
    PRIMARY KEY (cluster_id, linked_cluster_id, relation)
);

CREATE TABLE IF NOT EXISTS journal_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    at_ns INTEGER NOT NULL,
    kind TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    detail_json TEXT NOT NULL,
    sha256 TEXT NOT NULL
);
CREATE TRIGGER IF NOT EXISTS journal_events_no_update
BEFORE UPDATE ON journal_events
BEGIN SELECT RAISE(ABORT, 'journal events are append-only'); END;
CREATE TRIGGER IF NOT EXISTS journal_events_no_delete
BEFORE DELETE ON journal_events
BEGIN SELECT RAISE(ABORT, 'journal events are retained'); END;

CREATE TABLE IF NOT EXISTS tier_usage (
    component_id TEXT PRIMARY KEY,
    tier TEXT NOT NULL,
    category TEXT NOT NULL,
    bytes INTEGER NOT NULL,
    detail TEXT NOT NULL,
    updated_at_ns INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS tier_usage_tier ON tier_usage(tier, category);

CREATE TABLE IF NOT EXISTS procedures (
    skill_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    title TEXT NOT NULL,
    applicability_json TEXT NOT NULL,
    prerequisites_json TEXT NOT NULL,
    steps_json TEXT NOT NULL,
    outcome TEXT NOT NULL,
    error TEXT,
    evidence_json TEXT NOT NULL,
    verified_by TEXT,
    created_at_ns INTEGER NOT NULL,
    PRIMARY KEY (skill_id, version)
);
CREATE TRIGGER IF NOT EXISTS procedures_no_update
BEFORE UPDATE ON procedures
BEGIN SELECT RAISE(ABORT, 'procedure versions are immutable'); END;
CREATE TRIGGER IF NOT EXISTS procedures_no_delete
BEFORE DELETE ON procedures
BEGIN SELECT RAISE(ABORT, 'failed and verified procedure evidence is retained'); END;

CREATE TABLE IF NOT EXISTS procedure_invalidations (
    skill_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    record_id TEXT NOT NULL,
    reason TEXT NOT NULL,
    at_ns INTEGER NOT NULL,
    PRIMARY KEY (skill_id, version, record_id)
);

CREATE TABLE IF NOT EXISTS expert_shards (
    expert_id TEXT PRIMARY KEY,
    profile_id TEXT NOT NULL,
    parameter_count INTEGER NOT NULL,
    location TEXT NOT NULL,
    dirty INTEGER NOT NULL,
    health TEXT NOT NULL,
    quarantine_status TEXT NOT NULL,
    quarantine_reason TEXT,
    publication_generation INTEGER NOT NULL,
    hot_weights_path TEXT,
    hot_moments_path TEXT,
    hot_bytes INTEGER NOT NULL,
    published_weights_sha TEXT,
    published_moments_sha TEXT,
    published_clock INTEGER,
    encoder_id TEXT NOT NULL,
    lineage_json TEXT NOT NULL,
    lineage_digest TEXT NOT NULL,
    optimizer TEXT NOT NULL,
    ablation TEXT,
    updated_at_ns INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS expert_publications (
    expert_id TEXT NOT NULL,
    generation INTEGER NOT NULL,
    manifest_path TEXT NOT NULL,
    manifest_sha256 TEXT NOT NULL,
    weights_path TEXT NOT NULL,
    weights_sha TEXT NOT NULL,
    weights_bytes INTEGER NOT NULL,
    moments_path TEXT NOT NULL,
    moments_sha TEXT NOT NULL,
    moments_bytes INTEGER NOT NULL,
    clock INTEGER NOT NULL,
    created_at_ns INTEGER NOT NULL,
    PRIMARY KEY (expert_id, generation)
);
CREATE TRIGGER IF NOT EXISTS expert_publications_no_update
BEFORE UPDATE ON expert_publications
BEGIN SELECT RAISE(ABORT, 'expert publications are immutable'); END;
CREATE TRIGGER IF NOT EXISTS expert_publications_no_delete
BEFORE DELETE ON expert_publications
BEGIN SELECT RAISE(ABORT, 'expert publication generations are retained'); END;

CREATE TABLE IF NOT EXISTS identity_versions (
    identity_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    status TEXT NOT NULL,
    core_path TEXT NOT NULL,
    core_sha TEXT NOT NULL,
    core_bytes INTEGER NOT NULL,
    authority TEXT NOT NULL,
    approved INTEGER NOT NULL,
    supersedes INTEGER,
    created_at_ns INTEGER NOT NULL,
    PRIMARY KEY (identity_id, version)
);

CREATE TABLE IF NOT EXISTS phi_releases (
    codec_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    producer_version TEXT NOT NULL,
    artifact_sha TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at_ns INTEGER NOT NULL,
    PRIMARY KEY (codec_id, version)
);

CREATE TABLE IF NOT EXISTS capability_records (
    capability_id TEXT PRIMARY KEY,
    model_id TEXT NOT NULL,
    mode TEXT NOT NULL,
    task_family TEXT NOT NULL,
    budget_json TEXT NOT NULL,
    measured_value REAL,
    uncertainty_json TEXT NOT NULL,
    measured INTEGER NOT NULL,
    created_at_ns INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS workspace_snapshots (
    task_id TEXT PRIMARY KEY,
    path TEXT NOT NULL,
    sha TEXT NOT NULL,
    bytes INTEGER NOT NULL,
    resumable INTEGER NOT NULL,
    created_at_ns INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS artifacts (
    artifact_id TEXT PRIMARY KEY,
    path TEXT NOT NULL UNIQUE,
    bytes INTEGER NOT NULL,
    sha256 TEXT NOT NULL,
    detected_type TEXT NOT NULL,
    magic_hex TEXT NOT NULL,
    pinned INTEGER NOT NULL,
    label TEXT,
    source_path TEXT,
    status TEXT NOT NULL,
    created_at_ns INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS checkpoint_generations (
    checkpoint_id TEXT NOT NULL,
    generation INTEGER NOT NULL,
    payload_path TEXT NOT NULL,
    payload_sha TEXT NOT NULL,
    payload_bytes INTEGER NOT NULL,
    manifest_path TEXT NOT NULL,
    manifest_sha TEXT NOT NULL,
    created_at_ns INTEGER NOT NULL,
    PRIMARY KEY (checkpoint_id, generation)
);
CREATE TRIGGER IF NOT EXISTS checkpoint_generations_no_update
BEFORE UPDATE ON checkpoint_generations
BEGIN SELECT RAISE(ABORT, 'checkpoint generations are immutable'); END;
CREATE TRIGGER IF NOT EXISTS checkpoint_generations_no_delete
BEFORE DELETE ON checkpoint_generations
BEGIN SELECT RAISE(ABORT, 'checkpoint generations are retained'); END;

CREATE TABLE IF NOT EXISTS replay_entries (
    episode_id TEXT PRIMARY KEY,
    path TEXT NOT NULL,
    sha TEXT NOT NULL,
    bytes INTEGER NOT NULL,
    created_at_ns INTEGER NOT NULL
);
"""


# ---------------------------------------------------------------------------
# TieredMemoryStore
# ---------------------------------------------------------------------------

class TieredMemoryStore:
    """One durable logical hot/warm/cold store over SQLite plus real files.

    Directory layout beneath ``root``::

        memory.sqlite3                 index, journals index, accounting
        storage/hot/                   identity, active experts, workspace
        storage/warm/records/          keys (retrieval index) and warm payloads
        storage/cold/records/          cold canonical text and ordered values
        storage/cold/experts/          published shard generations
        storage/cold/identity/         approved identity versions
        storage/{hot,warm,cold}/...    replay, workspace, checkpoints
        manifests/clusters/            immutable generation manifests
        manifests/experts/             immutable publication manifests
        manifests/checkpoints/         immutable checkpoint manifests
        journal/events.jsonl           append-only local audit
        artifacts/sha256/              bounded ignored-artifact storage
        staging/                       temporary publication overlap
        workspace/                     explicit task snapshots

    No constructor path materializes a capacity profile bank.
    """

    def __init__(self, root: Path | str, *, profile_id: str = "P1",
                 ceilings: TierCeilings | None = None,
                 two_active_ablation: bool = False,
                 ablation_label: str | None = None,
                 rebuild_if_missing: bool = True,
                 enforce_volume_free: bool = True) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.sqlite_path = self.root / "memory.sqlite3"
        self.ceilings = ceilings or DEFAULT_CEILINGS
        self.profile = get_profile(profile_id)
        self.profile_id = profile_id
        self.two_active_ablation = bool(two_active_ablation)
        self.ablation_label = (TWO_ACTIVE_ABLATION_ID if self.two_active_ablation
                               else None)
        if self.ablation_label is not None and ablation_label is not None:
            self.ablation_label = ablation_label
        if not self.two_active_ablation and ablation_label is not None:
            raise ValueError("ablation_label is only valid with two_active_ablation")
        self.enforce_volume_free = bool(enforce_volume_free)
        self._selections: dict[str, dict] = {}
        self._tickets: dict[str, dict] = {}
        self._ticket_results: dict[str, RecallResult] = {}
        self._hot_cache: dict[str, dict] = {}
        self._txn_depth = 0
        self._closed = False
        self._recovery: RecoveryReport | None = None

        self._directories = {
            "hot": self.root / "storage/hot",
            "warm": self.root / "storage/warm",
            "cold": self.root / "storage/cold",
            "manifests_clusters": self.root / "manifests/clusters",
            "manifests_experts": self.root / "manifests/experts",
            "manifests_checkpoints": self.root / "manifests/checkpoints",
            "journal": self.root / "journal",
            "artifacts": self.root / "artifacts/sha256",
            "staging": self.root / "staging",
            "workspace": self.root / "workspace",
        }
        for directory in self._directories.values():
            directory.mkdir(parents=True, exist_ok=True)

        existed = self.sqlite_path.exists()
        had_meta = False
        if existed:
            probe = sqlite3.connect(self.sqlite_path)
            try:
                had_meta = bool(probe.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name='meta'"
                ).fetchone())
            finally:
                probe.close()
        self._connection = sqlite3.connect(
            self.sqlite_path, timeout=30.0, isolation_level=None)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=FULL")
        self._connection.execute("PRAGMA foreign_keys=ON")
        self._connection.execute("PRAGMA busy_timeout=30000")
        self._connection.executescript(_SCHEMA)
        with self._transaction():
            self._initialise_meta(existed and had_meta)
        if (not (existed and had_meta)) and rebuild_if_missing:
            manifests_exist = any(self._directories[name].rglob("*.json") for name in (
                "manifests_clusters", "manifests_experts", "manifests_checkpoints"))
            if manifests_exist or not existed:
                self._recovery = self._recover_locked(
                    rebuild_index=bool(manifests_exist),
                    remove_temporaries=True)

    # -- context / connection helpers -------------------------------------

    @property
    def connection(self) -> sqlite3.Connection:
        return self._connection

    @property
    def recovery_report(self) -> RecoveryReport | None:
        return self._recovery

    def __enter__(self) -> "TieredMemoryStore":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    @contextmanager
    def _transaction(self):
        if self._closed:
            raise TieredMemoryError("store is closed")
        if self._txn_depth:
            self._txn_depth += 1
            try:
                yield
            finally:
                self._txn_depth -= 1
            return
        self._connection.execute("BEGIN IMMEDIATE")
        self._txn_depth = 1
        try:
            yield
            self._connection.execute("COMMIT")
        except BaseException:
            self._connection.execute("ROLLBACK")
            raise
        finally:
            self._txn_depth = 0

    def _initialise_meta(self, existing: bool) -> None:
        now = _now_ns()
        expected = {
            "format": FORMAT,
            "schema_version": SCHEMA_VERSION,
            "profile_id": self.profile_id,
            "two_active_ablation": _canonical(self.two_active_ablation),
            "ablation_label": self.ablation_label or "",
            "ceilings": _canonical(self.ceilings.document()),
            "native_guards": _canonical(NATIVE_GUARDS.document()),
            "epoch": "0",
            "page_cache_pressure_bytes": "0",
            "page_cache_source": "",
            "native_qualification": "unsupported",
            "expert_payload_label": "engineering_fixture_not_native_weights",
        }
        if existing:
            row = self._connection.execute(
                "SELECT value FROM meta WHERE key='format'").fetchone()
            if row is None or row["value"] != FORMAT:
                raise SchemaError("existing store format mismatch")
            if self._connection.execute(
                    "SELECT value FROM meta WHERE key='schema_version'").fetchone()["value"] \
                    != SCHEMA_VERSION:
                raise SchemaError("existing store schema version mismatch")
            self._meta_expect("profile_id", self.profile_id)
            self._meta_expect("two_active_ablation", _canonical(self.two_active_ablation))
            self._meta_expect("ceilings", _canonical(self.ceilings.document()))
            current = self._connection.execute(
                "SELECT value FROM meta WHERE key='ablation_label'").fetchone()
            if current is not None and (current["value"] or None) != self.ablation_label:
                raise SchemaError("existing ablation label mismatch")
            return
        for key, value in expected.items():
            self._connection.execute(
                "INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)", (key, value))
        # The creation timestamp is informational; it is not a truth clock.
        self._connection.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES ('created_at_ns', ?)",
            (str(now),))

    def _meta_expect(self, key: str, value: str) -> None:
        row = self._connection.execute(
            "SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        if row is None or row["value"] != value:
            raise SchemaError(f"existing store policy differs for {key}")

    def _meta(self, key: str, default: str = "") -> str:
        row = self._connection.execute(
            "SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return default if row is None else row["value"]

    def _set_meta(self, key: str, value: str) -> None:
        self._connection.execute(
            "INSERT INTO meta(key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, value))

    @property
    def epoch(self) -> int:
        return int(self._meta("epoch", "0"))

    def _next_epoch(self) -> int:
        value = self.epoch + 1
        self._set_meta("epoch", str(value))
        return value

    def _relative(self, path: Path) -> str:
        try:
            return path.relative_to(self.root).as_posix()
        except ValueError as error:
            raise TieredMemoryError(f"path is outside the store: {path}") from error

    def _absolute(self, relative_path: str) -> Path:
        path = (self.root / relative_path).resolve()
        try:
            path.relative_to(self.root)
        except ValueError as error:
            raise CorruptState("durable path escapes the store root") from error
        return path

    # -- volume, database and tier accounting ------------------------------

    def _check_volume_free(self) -> None:
        if not self.enforce_volume_free:
            return
        free = shutil.disk_usage(self.root).free
        required = self.ceilings.minimum_volume_free_bytes
        if free < required:
            raise InsufficientVolumeFree(
                f"volume free space {free} is below the declared headroom {required}")

    def _usage_rows_locked(self, tier: str) -> list[sqlite3.Row]:
        return list(self._connection.execute(
            "SELECT category, SUM(bytes) AS total, COUNT(*) AS components "
            "FROM tier_usage WHERE tier=? GROUP BY category ORDER BY category",
            (tier,)))

    def _usage_total_locked(self, tier: str | None = None) -> int:
        if tier is None:
            row = self._connection.execute(
                "SELECT COALESCE(SUM(bytes), 0) AS total FROM tier_usage").fetchone()
        else:
            row = self._connection.execute(
                "SELECT COALESCE(SUM(bytes), 0) AS total FROM tier_usage WHERE tier=?",
                (tier,)).fetchone()
        return int(row["total"])

    def _category_total_locked(self, tier: str, category: str) -> int:
        row = self._connection.execute(
            "SELECT COALESCE(SUM(bytes), 0) AS total FROM tier_usage "
            "WHERE tier=? AND category=?", (tier, category)).fetchone()
        return int(row["total"])

    def _component_total_locked(self, component_id: str) -> int:
        row = self._connection.execute(
            "SELECT bytes FROM tier_usage WHERE component_id=?",
            (component_id,)).fetchone()
        return 0 if row is None else int(row["bytes"])

    def _sync_database_usage_locked(self) -> None:
        for suffix, component in ((self.sqlite_path, "database:sqlite"),
                                  (Path(str(self.sqlite_path) + "-wal"), "database:wal"),
                                  (Path(str(self.sqlite_path) + "-shm"), "database:shm")):
            size = _safe_file_bytes(suffix)
            self._set_component_locked(component, "cold", "database", size,
                                       "measured SQLite index/accounting file")

    def _set_component_locked(self, component_id: str, tier: str,
                              category: str, byte_count: int, detail: str) -> None:
        if byte_count < 0:
            raise ValueError("component byte count cannot be negative")
        if byte_count == 0:
            self._connection.execute(
                "DELETE FROM tier_usage WHERE component_id=?", (component_id,))
            return
        existing = self._connection.execute(
            "SELECT tier, category FROM tier_usage WHERE component_id=?",
            (component_id,)).fetchone()
        if existing is not None and (existing["tier"] != tier
                                     or existing["category"] != category):
            raise SchemaError(
                f"accounting component changed identity: {component_id}")
        self._connection.execute(
            "INSERT INTO tier_usage(component_id, tier, category, bytes, detail, updated_at_ns) "
            "VALUES (?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(component_id) DO UPDATE SET "
            "bytes=excluded.bytes, detail=excluded.detail, updated_at_ns=excluded.updated_at_ns",
            (component_id, tier, category, byte_count, detail, _now_ns()))

    def _ensure_capacity_locked(self, tier: str, category: str,
                                additional: int) -> None:
        if additional < 0:
            raise ValueError("capacity charge cannot be negative")
        if not additional:
            return
        total = self._usage_total_locked(tier)
        if tier == "hot":
            identity_used = self._category_total_locked("hot", "identity")
            if category == "identity":
                if identity_used + additional > self.ceilings.identity_reserved_bytes:
                    raise CapacityExceeded(
                        "hot", "identity", identity_used,
                        self.ceilings.identity_reserved_bytes, additional)
            else:
                non_identity = total - identity_used
                limit = self.ceilings.hot_nonidentity_bytes
                if non_identity + additional > limit:
                    raise CapacityExceeded("hot", category, non_identity, limit, additional)
            if total + additional > self.ceilings.hot_bytes:
                raise CapacityExceeded("hot", category, total,
                                       self.ceilings.hot_bytes, additional)
        elif tier == "warm":
            replay_used = self._category_total_locked("warm", "replay")
            if category == "replay":
                if replay_used + additional > self.ceilings.replay_reserved_bytes:
                    raise CapacityExceeded(
                        "warm", "replay", replay_used,
                        self.ceilings.replay_reserved_bytes, additional)
            else:
                non_replay = total - replay_used
                limit = self.ceilings.warm_nonreplay_bytes
                if non_replay + additional > limit:
                    raise CapacityExceeded("warm", category, non_replay, limit, additional)
            if total + additional > self.ceilings.warm_bytes:
                raise CapacityExceeded("warm", category, total,
                                       self.ceilings.warm_bytes, additional)
        elif tier == "cold":
            if total + additional > self.ceilings.cold_bytes:
                raise CapacityExceeded("cold", category, total,
                                       self.ceilings.cold_bytes, additional)
        else:
            raise ValueError(f"unknown logical tier: {tier}")
        workspace_used = int(self._connection.execute(
            "SELECT COALESCE(SUM(bytes), 0) AS total FROM tier_usage "
            "WHERE category='workspace'").fetchone()["total"])
        if category == "workspace" and workspace_used + additional > (
                self.ceilings.workspace_quota_bytes):
            raise CapacityExceeded("workspace", category, workspace_used,
                                   self.ceilings.workspace_quota_bytes, additional)
        if category == "artifacts":
            artifact_used = int(self._connection.execute(
                "SELECT COALESCE(SUM(bytes), 0) AS total FROM tier_usage "
                "WHERE category='artifacts'").fetchone()["total"])
            if artifact_used + additional > self.ceilings.artifact_budget_bytes:
                raise CapacityExceeded("artifacts", category, artifact_used,
                                       self.ceilings.artifact_budget_bytes, additional)

    def _charge(self, tier: str, category: str, byte_count: int,
                component_id: str, detail: str = "") -> None:
        if byte_count < 0:
            raise ValueError("charge cannot be negative")
        if not byte_count:
            return
        self._check_volume_free()
        with self._transaction():
            self._sync_database_usage_locked()
            self._ensure_capacity_locked(tier, category, byte_count)
            existing = self._connection.execute(
                "SELECT bytes FROM tier_usage WHERE component_id=?",
                (component_id,)).fetchone()
            if existing is None:
                self._connection.execute(
                    "INSERT INTO tier_usage(component_id, tier, category, bytes, detail, "
                    "updated_at_ns) VALUES (?, ?, ?, ?, ?, ?)",
                    (component_id, tier, category, byte_count, detail, _now_ns()))
            else:
                self._connection.execute(
                    "UPDATE tier_usage SET bytes=bytes+?, updated_at_ns=? WHERE component_id=?",
                    (byte_count, _now_ns(), component_id))

    def _release(self, component_id: str) -> int:
        with self._transaction():
            row = self._connection.execute(
                "SELECT bytes FROM tier_usage WHERE component_id=?",
                (component_id,)).fetchone()
            if row is None:
                return 0
            self._connection.execute(
                "DELETE FROM tier_usage WHERE component_id=?", (component_id,))
            return int(row["bytes"])

    def _release_bytes(self, tier: str, category: str, byte_count: int,
                       component_id: str) -> None:
        """Remove a known number of bytes from one component without over-release."""
        if byte_count <= 0:
            return
        with self._transaction():
            row = self._connection.execute(
                "SELECT bytes FROM tier_usage WHERE component_id=?",
                (component_id,)).fetchone()
            if row is None:
                return
            remaining = int(row["bytes"]) - byte_count
            if remaining <= 0:
                self._connection.execute(
                    "DELETE FROM tier_usage WHERE component_id=?", (component_id,))
            else:
                self._connection.execute(
                    "UPDATE tier_usage SET bytes=?, updated_at_ns=? WHERE component_id=?",
                    (remaining, _now_ns(), component_id))
        del tier, category

    def usage(self) -> dict:
        with self._transaction():
            self._sync_database_usage_locked()
        tiers: dict[str, dict] = {}
        for tier in ("hot", "warm", "cold"):
            rows = self._usage_rows_locked(tier)
            categories = {row["category"]: int(row["total"]) for row in rows}
            total = sum(categories.values())
            entry = {
                "limit_bytes": getattr(self.ceilings, f"{tier}_bytes"),
                "used_bytes": total,
                "by_category": categories,
                "within_limit": total <= getattr(self.ceilings, f"{tier}_bytes"),
            }
            if tier == "hot":
                entry.update({
                    "identity_limit_bytes": self.ceilings.identity_reserved_bytes,
                    "identity_used_bytes": categories.get("identity", 0),
                    "non_identity_limit_bytes": self.ceilings.hot_nonidentity_bytes,
                    "non_identity_used_bytes": total - categories.get("identity", 0),
                })
            if tier == "warm":
                entry.update({
                    "replay_limit_bytes": self.ceilings.replay_reserved_bytes,
                    "replay_used_bytes": categories.get("replay", 0),
                    "non_replay_limit_bytes": self.ceilings.warm_nonreplay_bytes,
                    "non_replay_used_bytes": total - categories.get("replay", 0),
                })
            tiers[tier] = entry
        return {
            "format": "replica.tiered_memory.accounting.v1",
            "logical_tiers": True,
            "physical_vram_pool_invented": False,
            "tiers": tiers,
            "workspace_quota_bytes": self.ceilings.workspace_quota_bytes,
            "artifact_budget_bytes": self.ceilings.artifact_budget_bytes,
            "identity_quota_bytes": self.ceilings.identity_reserved_bytes,
            "replay_quota_bytes": self.ceilings.replay_reserved_bytes,
            "page_cache_pressure_bytes": int(self._meta(
                "page_cache_pressure_bytes", "0")),
            "page_cache_source": self._meta("page_cache_source", "") or None,
            "page_cache_is_distinct_from_logical": False,
            "selection_and_cache_components_are_independent_copies": True,
            "native_capability": False,
        }

    def observe_page_cache_pressure(self, byte_count: int, *, source: str) -> dict:
        """Record an externally measured page-cache estimate; do not add a tier."""
        _nonnegative_int("byte_count", byte_count)
        _identifier("source", source)
        with self._transaction():
            self._set_meta("page_cache_pressure_bytes", str(byte_count))
            self._set_meta("page_cache_source", source)
        return self.usage()

    # -- stable cluster admission and manifests ----------------------------

    def _cluster_row(self, cluster_id: str) -> sqlite3.Row:
        row = self._connection.execute(
            "SELECT * FROM clusters WHERE cluster_id=?", (cluster_id,)).fetchone()
        if row is None:
            raise CorruptState(f"unknown cluster: {cluster_id}")
        return row

    def _record_rows(self, cluster_id: str) -> list[sqlite3.Row]:
        return list(self._connection.execute(
            "SELECT * FROM records WHERE cluster_id=? ORDER BY record_index, record_id",
            (cluster_id,)))

    def _key_row(self, record_id: str) -> sqlite3.Row | None:
        return self._connection.execute(
            "SELECT * FROM record_keys WHERE record_id=?", (record_id,)).fetchone()

    def _value_rows(self, record_id: str) -> list[sqlite3.Row]:
        return list(self._connection.execute(
            "SELECT * FROM record_values WHERE record_id=? ORDER BY slot_index",
            (record_id,)))

    def _span_rows(self, record_id: str) -> list[sqlite3.Row]:
        return list(self._connection.execute(
            "SELECT * FROM record_spans WHERE record_id=? "
            "ORDER BY start_offset, end_offset",
            (record_id,)))

    def _manifest_envelope(self, manifest: Mapping[str, object]) -> bytes:
        digest = _sha256_json(manifest)
        return (_canonical({
            "manifest": manifest,
            "manifest_sha256": digest,
        }) + "\n").encode("utf-8")

    def _read_manifest(self, path: Path, expected_format: str) -> tuple[dict, str]:
        try:
            payload = path.read_bytes()
            document = json.loads(payload.decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise CorruptState(f"unreadable manifest: {path}") from error
        if not isinstance(document, dict) or "manifest" not in document:
            raise CorruptState(f"manifest envelope missing: {path}")
        manifest = document["manifest"]
        digest = document.get("manifest_sha256")
        if not isinstance(manifest, dict) or not isinstance(digest, str):
            raise CorruptState(f"invalid manifest envelope: {path}")
        if manifest.get("format") != expected_format:
            raise CorruptState(f"unexpected manifest format: {path}")
        actual = _sha256_json(manifest)
        if actual != digest:
            raise CorruptState(f"manifest digest mismatch: {path}")
        return manifest, actual

    def _record_manifest_entry(self, row: sqlite3.Row) -> dict:
        key = self._key_row(row["record_id"])
        values = self._value_rows(row["record_id"])
        spans = self._span_rows(row["record_id"])
        return {
            "record_id": row["record_id"],
            "record_index": int(row["record_index"]),
            "record_hash": row["record_hash"],
            "kind": row["kind"],
            "status": row["status"],
            "superseded_by": row["superseded_by"],
            "revoked_reason": row["revoked_reason"],
            "text": {
                "path": row["text_path"],
                "sha256": row["text_sha256"],
                "bytes": int(row["text_bytes"]),
            },
            "source_id": row["source_id"],
            "source_epoch": int(row["source_epoch"]),
            "source_lineage": json.loads(row["source_lineage"]),
            "authenticity": row["authenticity"],
            "verification_state": row["verification_state"],
            "observed_at": int(row["observed_at"]),
            "event_at": int(row["event_at"]),
            "valid_from": row["valid_from"],
            "valid_to": row["valid_to"],
            "importance": int(row["importance"]),
            "critical": bool(row["critical"]),
            "placement": row["placement"],
            "revision": int(row["revision"]),
            "representation_version": row["representation_version"],
            "key": None if key is None else {
                "path": key["key_path"],
                "bytes": int(key["key_bytes"]),
                "sha256": key["key_sha256"],
                "width": int(key["key_width"]),
                "producer_id": key["producer_id"],
                "producer_version": key["producer_version"],
                "status": key["status"],
            },
            "values": [{
                "slot": int(value["slot_index"]),
                "path": value["value_path"],
                "bytes": int(value["value_bytes"]),
                "sha256": value["value_sha256"],
                "width": int(value["value_width"]),
                "dtype": value["dtype"],
                "producer_id": value["producer_id"],
                "producer_version": value["producer_version"],
                "status": value["status"],
            } for value in values],
            "spans": [{
                "span_id": span["span_id"],
                "start": int(span["start_offset"]),
                "end": int(span["end_offset"]),
                "sha256": span["span_sha256"],
                "text_sha256": span["text_sha256"],
            } for span in spans],
        }

    def _publish_cluster_generation(self, cluster_id: str, *,
                                    parent_generation: int,
                                    created_at_ns: int) -> tuple[int, str]:
        cluster = self._cluster_row(cluster_id)
        generation = parent_generation + 1
        record_rows = self._record_rows(cluster_id)
        entries = [self._record_manifest_entry(row) for row in record_rows]
        links = [dict(row) for row in self._connection.execute(
            "SELECT linked_cluster_id, relation FROM cluster_links "
            "WHERE cluster_id=? ORDER BY linked_cluster_id, relation", (cluster_id,))]
        scope = json.loads(cluster["scope_json"])
        manifest = {
            "format": CLUSTER_MANIFEST_FORMAT,
            "store_format": FORMAT,
            "cluster_id": cluster_id,
            "cluster_name": cluster["name"],
            "cluster_kind": cluster["cluster_kind"],
            "scope": scope,
            "scope_key": cluster["scope_key"],
            "generation": generation,
            "parent_generation": parent_generation,
            "created_at_ns": created_at_ns,
            "record_count": len(entries),
            "records": entries,
            "links": links,
            "verification_states": sorted({entry["verification_state"] for entry in entries}),
            "source_epochs": sorted({entry["source_epoch"] for entry in entries}),
            "representation_versions": sorted({
                entry["representation_version"] for entry in entries}),
            "byte_sizes": {
                "text": sum(entry["text"]["bytes"] for entry in entries),
                "keys": sum((entry["key"] or {}).get("bytes", 0) for entry in entries),
                "values": sum(
                    sum(item["bytes"] for item in entry["values"]) for entry in entries),
            },
            "tier_locations": {
                "text": sorted({entry["placement"] for entry in entries}),
                "keys": "warm",
                "values": sorted({entry["placement"] for entry in entries}),
                "inactive_experts": "cold_metadata_only",
            },
            "state_classes": list(STATE_CLASSES),
            "engineering_evidence": True,
            "native_capability": False,
            "exact_unlearning_claimed": False,
        }
        envelope = self._manifest_envelope(manifest)
        relative_path = (f"manifests/clusters/{cluster_id}/"
                         f"{generation:08d}.json")
        path = self.root / relative_path
        if path.exists():
            raise CorruptState(f"generation manifest already exists: {relative_path}")
        self._charge("cold", "generations", len(envelope),
                     f"manifest:cluster:{cluster_id}:{generation}",
                     f"immutable cluster generation {generation}")
        _atomic_write(path, envelope)
        digest = _sha256_json(manifest)
        self._connection.execute(
            "INSERT INTO cluster_generations(cluster_id, generation, scope_key, "
            "manifest_path, manifest_sha256, parent_generation, record_count, "
            "created_at_ns) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (cluster_id, generation, cluster["scope_key"], relative_path, digest,
             parent_generation, len(entries), created_at_ns))
        self._connection.execute(
            "UPDATE clusters SET current_generation=?, manifest_sha256=?, "
            "record_count=?, updated_at_ns=? WHERE cluster_id=?",
            (generation, digest, len(entries), created_at_ns, cluster_id))
        return generation, digest

    def _write_record_payloads(self, draft: EvidenceRecordDraft, *,
                               placement: str) -> list[tuple[str, int]]:
        """Write real text/key/value files.  Returns (relative path, bytes)."""
        if placement not in PLACEMENTS:
            raise ValueError("record placement must be warm or cold")
        written: list[tuple[str, int]] = []
        try:
            text_path = self.root / f"storage/{placement}/records/{draft.record_id}/text.bin"
            text_bytes = draft.text.encode("utf-8")
            if text_path.exists():
                raise CorruptState(f"record text already exists: {draft.record_id}")
            _atomic_write(text_path, text_bytes)
            written.append((self._relative(text_path), len(text_bytes)))
            key_path = self.root / f"storage/warm/keys/{draft.record_id}.key"
            if key_path.exists():
                raise CorruptState(f"record key already exists: {draft.record_id}")
            key_bytes = draft.key_bytes
            _atomic_write(key_path, key_bytes)
            written.append((self._relative(key_path), len(key_bytes)))
            if draft.values is not None:
                for slot, payload in enumerate(pack_values(draft.values)):
                    value_path = (self.root
                                  / f"storage/{placement}/records/{draft.record_id}/"
                                    f"slot-{slot}.f32")
                    if value_path.exists():
                        raise CorruptState(
                            f"record value slot already exists: {draft.record_id}:{slot}")
                    _atomic_write(value_path, payload)
                    written.append((self._relative(value_path), len(payload)))
        except BaseException:
            for relative, _ in written:
                try:
                    (self.root / relative).unlink()
                except FileNotFoundError:
                    pass
                except OSError:
                    pass
            raise
        return written

    def _insert_record(self, draft: EvidenceRecordDraft, *, cluster_id: str,
                       scope_key: str, record_index: int, placement: str,
                       now: int) -> tuple[str, int]:
        text_relative = f"storage/{placement}/records/{draft.record_id}/text.bin"
        text_bytes = draft.text_bytes
        key_relative = f"storage/warm/keys/{draft.record_id}.key"
        key_bytes = draft.key_bytes
        record_hash = draft.record_hash(scope_key)
        self._connection.execute(
            "INSERT INTO records(record_id, cluster_id, record_index, record_hash, "
            "scope_key, kind, status, superseded_by, revoked_reason, text_path, "
            "text_sha256, text_bytes, source_id, source_epoch, source_lineage, "
            "authenticity, verification_state, observed_at, event_at, valid_from, "
            "valid_to, importance, critical, placement, revision, "
            "representation_version, created_at_ns, updated_at_ns) "
            "VALUES (?, ?, ?, ?, ?, ?, 'current', NULL, NULL, ?, ?, ?, ?, ?, ?, ?, ?, "
            "?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?)",
            (draft.record_id, cluster_id, record_index, record_hash, scope_key,
             draft.kind, text_relative, draft.text_sha256, text_bytes,
             draft.source_id, draft.source_epoch, _canonical(list(draft.source_lineage)),
             draft.authenticity, draft.verification_state, draft.observed_at,
             draft.event_at, draft.valid_from, draft.valid_to, draft.importance,
             int(draft.critical), placement, draft.representation_version, now, now))
        self._connection.execute(
            "INSERT INTO record_keys(record_id, key_path, key_bytes, key_sha256, "
            "key_width, producer_id, producer_version, status, updated_at_ns) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, 'current', ?)",
            (draft.record_id, key_relative, len(key_bytes), _sha256_bytes(key_bytes),
             KEY_WIDTH, draft.key_producer_id, draft.key_producer_version, now))
        if draft.values is not None:
            for slot, payload in enumerate(pack_values(draft.values)):
                value_relative = (f"storage/{placement}/records/{draft.record_id}/"
                                  f"slot-{slot}.f32")
                self._connection.execute(
                    "INSERT INTO record_values(record_id, slot_index, value_path, "
                    "value_bytes, value_sha256, value_width, dtype, producer_id, "
                    "producer_version, status, updated_at_ns) "
                    "VALUES (?, ?, ?, ?, ?, ?, 'float32', ?, ?, 'current', ?)",
                    (draft.record_id, slot, value_relative, len(payload),
                     _sha256_bytes(payload), VALUE_WIDTH, draft.value_producer_id,
                     draft.value_producer_version, now))
        return record_hash, text_bytes

    def _charge_record_files(self, draft: EvidenceRecordDraft, *,
                             placement: str) -> None:
        self._check_volume_free()
        self._charge("warm", "keys", len(draft.key_bytes),
                     f"record:{draft.record_id}:key", "warm retrieval key")
        self._charge(placement, "canonical_text", draft.text_bytes,
                     f"record:{draft.record_id}:text", "canonical exact text")
        if draft.values is not None:
            for slot, payload in enumerate(pack_values(draft.values)):
                self._charge(placement, "values", len(payload),
                             f"record:{draft.record_id}:value:{slot}",
                             "ordered width-256 float32 slot")

    def _cleanup_paths(self, paths: Iterable[str]) -> None:
        for relative in paths:
            try:
                (self.root / relative).unlink()
            except FileNotFoundError:
                pass
            except OSError:
                pass

    def _record_event(self, kind: str, entity_id: str, detail: Mapping[str, object],
                      *, at_ns: int | None = None) -> dict:
        _identifier("kind", kind)
        _identifier("entity_id", entity_id)
        timestamp = _now_ns() if at_ns is None else at_ns
        payload = {
            "at_ns": timestamp,
            "kind": kind,
            "entity_id": entity_id,
            "detail": dict(detail),
            "format": "replica.tiered_memory.journal_event.v1",
            "engineering_evidence": True,
            "native_capability": False,
        }
        line = _canonical(payload) + "\n"
        encoded = line.encode("utf-8")
        digest = _sha256_bytes(encoded)
        self._charge("cold", "journal", len(encoded), "journal:file",
                     "append-only local audit journal")
        self._connection.execute(
            "INSERT INTO journal_events(at_ns, kind, entity_id, detail_json, sha256) "
            "VALUES (?, ?, ?, ?, ?)",
            (timestamp, kind, entity_id, _canonical(dict(detail)), digest))
        journal_path = self._directories["journal"] / "events.jsonl"
        descriptor = os.open(journal_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            written = os.write(descriptor, encoded)
            if written != len(encoded):
                raise OSError("short journal write")
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return {"sha256": digest, "bytes": len(encoded), "at_ns": timestamp}

    def admit_cluster(self, name: str, *, scope: MemoryScope,
                      records: Sequence[EvidenceRecordDraft],
                      links: Sequence[tuple[str, str]] = (),
                      placement: str = "warm",
                      cluster_kind: str = "evidence") -> ClusterAdmission:
        """Admit exact evidence and publish an immutable generation manifest."""
        _identifier("name", name)
        _identifier("cluster_kind", cluster_kind)
        if not isinstance(scope, MemoryScope):
            raise TypeError("scope must be a MemoryScope")
        if placement not in PLACEMENTS:
            raise ValueError("placement must be warm or cold")
        drafts = tuple(records)
        if not drafts:
            raise ValueError("a cluster requires at least one record")
        if any(not isinstance(draft, EvidenceRecordDraft) for draft in drafts):
            raise TypeError("records must be EvidenceRecordDraft instances")
        if len({draft.record_id for draft in drafts}) != len(drafts):
            raise ValueError("record IDs must be unique within one admission")
        for draft in drafts:
            if draft.kind == "correction":
                raise ValueError("use correct_record for correction records")
            if draft.supersedes is not None:
                raise ValueError("admission cannot supersede existing evidence")
        cluster_id = stable_cluster_id(scope, name, cluster_kind)
        now = _now_ns()
        existing = self._connection.execute(
            "SELECT * FROM clusters WHERE cluster_id=?", (cluster_id,)).fetchone()
        if existing is not None:
            if existing["scope_key"] != scope.scope_key:
                raise ScopeViolation("cluster scope differs from existing cluster")
            if existing["name"] != name or existing["cluster_kind"] != cluster_kind:
                raise SchemaError("stable cluster ID collision")
            if existing["status"] != "active":
                raise CorruptState("cluster is not active")
            parent = int(existing["current_generation"])
        else:
            parent = 0
        written: list[str] = []
        try:
            with self._transaction():
                for draft in drafts:
                    duplicate = self._connection.execute(
                        "SELECT 1 FROM records WHERE record_id=?",
                        (draft.record_id,)).fetchone()
                    if duplicate is not None:
                        raise CorruptState(f"record already exists: {draft.record_id}")
                    self._charge_record_files(draft, placement=placement)
                    written.extend(path for path, _ in self._write_record_payloads(
                        draft, placement=placement))
                    next_index = 1 + int(self._connection.execute(
                        "SELECT COALESCE(MAX(record_index), 0) AS value FROM records "
                        "WHERE cluster_id=?", (cluster_id,)).fetchone()["value"])
                    self._insert_record(
                        draft, cluster_id=cluster_id, scope_key=scope.scope_key,
                        record_index=next_index, placement=placement, now=now)
                if existing is None:
                    self._connection.execute(
                        "INSERT INTO clusters(cluster_id, name, cluster_kind, scope_key, "
                        "scope_json, current_generation, manifest_sha256, record_count, "
                        "status, created_at_ns, updated_at_ns) "
                        "VALUES (?, ?, ?, ?, ?, 0, NULL, 0, 'active', ?, ?)",
                        (cluster_id, name, cluster_kind, scope.scope_key,
                         _canonical(scope.document()), now, now))
                for linked, relation in links:
                    _identifier("linked_cluster_id", linked)
                    _identifier("relation", relation)
                    if self._connection.execute(
                            "SELECT 1 FROM clusters WHERE cluster_id=?",
                            (linked,)).fetchone() is None:
                        raise CorruptState(f"link target cluster does not exist: {linked}")
                    self._connection.execute(
                        "INSERT OR IGNORE INTO cluster_links(cluster_id, "
                        "linked_cluster_id, relation) VALUES (?, ?, ?)",
                        (cluster_id, linked, relation))
                generation, digest = self._publish_cluster_generation(
                    cluster_id, parent_generation=parent, created_at_ns=now)
                self._record_event("cluster_admitted", cluster_id, {
                    "generation": generation,
                    "record_ids": [draft.record_id for draft in drafts],
                    "placement": placement,
                    "scope_key": scope.scope_key,
                }, at_ns=now)
        except BaseException:
            self._cleanup_paths(written)
            raise
        return ClusterAdmission(
            outcome=_outcome("ok", "cluster_admitted",
                             generation=generation,
                             manifest_sha256=digest,
                             count=len(drafts)),
            cluster_id=cluster_id,
            generation=generation,
            manifest_sha256=digest,
            record_ids=tuple(draft.record_id for draft in drafts),
            placement=placement,
        )

    def generation_manifest(self, cluster_id: str,
                            generation: int | None = None
                            ) -> ClusterGenerationDocument:
        cluster = self._cluster_row(cluster_id)
        target = (int(cluster["current_generation"])
                  if generation is None else generation)
        row = self._connection.execute(
            "SELECT * FROM cluster_generations WHERE cluster_id=? AND generation=?",
            (cluster_id, target)).fetchone()
        if row is None:
            raise CorruptState(f"unknown generation {target} for {cluster_id}")
        path = self._absolute(row["manifest_path"])
        if not path.is_file():
            raise CorruptState(f"missing generation manifest: {row['manifest_path']}")
        manifest, digest = self._read_manifest(path, CLUSTER_MANIFEST_FORMAT)
        if digest != row["manifest_sha256"]:
            raise CorruptState("generation manifest digest differs from the index")
        if manifest.get("cluster_id") != cluster_id or manifest.get("generation") != target:
            raise CorruptState("generation manifest identity mismatch")
        return ClusterGenerationDocument(
            cluster_id=cluster_id, generation=target,
            path=row["manifest_path"], sha256=digest, document=manifest)

    def cluster(self, cluster_id: str) -> dict:
        row = self._cluster_row(cluster_id)
        return {
            "cluster_id": cluster_id,
            "name": row["name"],
            "cluster_kind": row["cluster_kind"],
            "scope_key": row["scope_key"],
            "scope": json.loads(row["scope_json"]),
            "current_generation": int(row["current_generation"]),
            "manifest_sha256": row["manifest_sha256"],
            "record_count": int(row["record_count"]),
            "status": row["status"],
            "created_at_ns": int(row["created_at_ns"]),
            "updated_at_ns": int(row["updated_at_ns"]),
        }

    # -- selection ---------------------------------------------------------

    def _eligible_record_rows(self, scope: MemoryScope) -> list[sqlite3.Row]:
        cutoff = scope.cutoff
        return list(self._connection.execute(
            "SELECT r.*, c.current_generation AS cluster_generation, c.name AS cluster_name "
            "FROM records r JOIN clusters c ON c.cluster_id=r.cluster_id "
            "WHERE r.scope_key=? AND c.scope_key=? AND r.status='current' "
            "AND c.status='active' AND r.observed_at<=? "
            "AND (r.valid_from IS NULL OR r.valid_from<=?) "
            "AND (r.valid_to IS NULL OR r.valid_to>?) "
            "ORDER BY r.record_index, r.record_id",
            (scope.scope_key, scope.scope_key, cutoff, cutoff, cutoff)))

    def _validate_scope_argument(self, scope: MemoryScope) -> None:
        if not isinstance(scope, MemoryScope):
            raise TypeError("scope must be a MemoryScope")

    def select(self, query: str, *, scope: MemoryScope,
               budget: RetrievalBudget | None = None,
               exact_phrases: Sequence[str] = (),
               include_text: bool = True) -> RetrievalResult:
        """Scope-filtered semantic key selection plus charged exact fallback."""
        if not isinstance(query, str):
            raise ValueError("query must be a string")
        self._validate_scope_argument(scope)
        budget = budget or RetrievalBudget()
        phrases = tuple(exact_phrases)
        for phrase in phrases:
            if not isinstance(phrase, str) or not phrase:
                raise ValueError("exact phrases must be nonempty strings")
        if not include_text:
            budget = replace(budget, max_exact_text_bytes=0)
        selection_id = "sel_" + uuid.uuid4().hex
        now = _now_ns()
        query_key = build_retrieval_key(query)
        candidates = self._eligible_record_rows(scope)
        scored: list[tuple[float, sqlite3.Row, bytes, sqlite3.Row]] = []
        representation_stale = False
        for row in candidates:
            key = self._key_row(row["record_id"])
            if key is None or key["status"] != "current":
                representation_stale = True
                continue
            try:
                key_bytes = _read_verified(
                    key["key_path"], self.root, key["key_sha256"], int(key["key_bytes"]))
            except CorruptState:
                representation_stale = True
                continue
            score = key_similarity(query_key, key_bytes)
            if score > 0:
                scored.append((score, row, key_bytes, key))
        scored.sort(key=lambda item: (-item[0], -int(item[1]["observed_at"]),
                                      item[1]["record_id"]))
        selected_entries: list[SelectedEvidence] = []
        cold_ids: list[str] = []
        used_text = 0
        used_slots = 0
        used_span_bytes = 0
        pins: list[dict] = []
        selected_ids: list[str] = []
        pin_generation: dict[str, int] = {}
        for score, row, key_bytes, key in scored:
            if len(selected_entries) >= budget.max_records:
                break
            values = self._value_rows(row["record_id"])
            value_available = (
                len(values) == VALUE_SLOTS_PER_RECORD
                and all(value["status"] == "current" for value in values))
            if not value_available and values:
                representation_stale = True
            slot_count = VALUE_SLOTS_PER_RECORD if value_available else 0
            if used_slots + slot_count > budget.max_value_slots:
                continue
            placement = row["placement"]
            text: str | None = None
            text_included = False
            text_payload: bytes | None = None
            if placement == "warm":
                try:
                    text_payload = _read_verified(
                        row["text_path"], self.root, row["text_sha256"],
                        int(row["text_bytes"]))
                except CorruptState:
                    text_payload = None
                    representation_stale = True
                if text_payload is not None and used_text + len(text_payload) <= (
                        budget.max_exact_text_bytes):
                    text = text_payload.decode("utf-8")
                    text_included = True
                    used_text += len(text_payload)
            if placement == "cold":
                cold_ids.append(row["record_id"])
            values_tuple: tuple[tuple[float, ...], ...] = ()
            if value_available and placement == "warm":
                try:
                    payloads = [_read_verified(
                        value["value_path"], self.root, value["value_sha256"],
                        int(value["value_bytes"])) for value in values]
                    values_tuple = unpack_values(payloads)
                except CorruptState:
                    value_available = False
                    values_tuple = ()
                    slot_count = 0
                    representation_stale = True
            elif value_available and placement == "cold":
                value_available = False
                slot_count = 0
                # Metadata remains; the payload needs an explicit cold recall.
            if used_slots + slot_count > budget.max_value_slots:
                continue
            used_slots += slot_count
            selected_entries.append(SelectedEvidence(
                record_id=row["record_id"],
                cluster_id=row["cluster_id"],
                cluster_generation=int(row["cluster_generation"]),
                score=score,
                text=text,
                text_sha256=row["text_sha256"],
                text_bytes=int(row["text_bytes"]),
                text_included=text_included,
                placement=placement,
                values=values_tuple,
                values_available=value_available,
                value_slots=slot_count,
                key_sha256=key["key_sha256"],
                representation_version=row["representation_version"],
                revision=int(row["revision"]),
                source_id=row["source_id"],
                source_epoch=int(row["source_epoch"]),
                verification_state=row["verification_state"],
                observed_at=int(row["observed_at"]),
                valid_to=row["valid_to"],
                critical=bool(row["critical"]),
                retrieval_count=self.retrieval_count(row["record_id"]),
            ))
            selected_ids.append(row["record_id"])
            pin_generation[row["cluster_id"]] = int(row["cluster_generation"])
            pins.append(self._build_pin(row, key, values))
        truncated = len(scored) > len(selected_entries)

        # Exact canonical-text / exact-span fallback is charged independently.
        span_hits: list[ExactSpanHit] = []
        span_budget_exhausted = False
        text_budget_exhausted = used_text >= budget.max_exact_text_bytes
        cold_fallback_pending: list[str] = []
        if phrases:
            for row in candidates:
                if len(span_hits) >= budget.max_spans:
                    span_budget_exhausted = True
                    break
                if row["record_id"] in selected_ids:
                    selected = next((item for item in selected_entries
                                     if item.record_id == row["record_id"]), None)
                    text = selected.text if selected is not None else None
                else:
                    text = None
                if text is None:
                    if row["placement"] == "cold":
                        cold_fallback_pending.append(row["record_id"])
                        continue
                    size = int(row["text_bytes"])
                    if used_text + size > budget.max_exact_text_bytes:
                        text_budget_exhausted = True
                        continue
                    try:
                        payload = _read_verified(
                            row["text_path"], self.root, row["text_sha256"], size)
                    except CorruptState:
                        representation_stale = True
                        continue
                    used_text += size
                    text = payload.decode("utf-8")
                for phrase in phrases:
                    start = text.find(phrase)
                    if start < 0:
                        continue
                    end = start + len(phrase)
                    span_text = text[start:end]
                    span_bytes = len(span_text.encode("utf-8"))
                    if used_span_bytes + span_bytes > budget.max_span_bytes:
                        span_budget_exhausted = True
                        break
                    used_span_bytes += span_bytes
                    span_hits.append(ExactSpanHit(
                        record_id=row["record_id"],
                        phrase=phrase,
                        start=start,
                        end=end,
                        text=span_text,
                        start_sha256=_sha256_text(span_text),
                        source_hash=row["text_sha256"],
                        record_hash=row["record_hash"],
                        placement=row["placement"],
                    ))
                    if len(span_hits) >= budget.max_spans:
                        span_budget_exhausted = True
                        break
                if len(span_hits) >= budget.max_spans:
                    break

        if span_hits and selected_entries:
            fallback_mode = "mixed"
        elif span_hits:
            fallback_mode = "exact_text"
        elif selected_entries:
            fallback_mode = "semantic"
        else:
            fallback_mode = "none"

        requires_cold = bool(cold_ids)
        if requires_cold:
            status, reason = "incomplete", "cold_payload_requires_recall"
        elif not selected_entries and span_hits:
            status, reason = "degraded", "semantic_keys_unavailable_exact_fallback"
        elif not selected_entries:
            status, reason = "incomplete", "no_semantic_candidate"
        elif representation_stale:
            status, reason = "degraded", "some_representations_stale"
        else:
            status, reason = "ok", "selection_ready"
        detail = {
            "candidate_count": len(candidates),
            "semantic_candidate_count": len(scored),
            "truncated": truncated,
            "cold_record_ids": tuple(cold_ids),
            "cold_fallback_pending": tuple(cold_fallback_pending),
            "text_budget_exhausted": text_budget_exhausted,
            "span_budget_exhausted": span_budget_exhausted,
            "representation_stale": representation_stale,
            "budget": budget.quota_plan(),
        }
        outcome = Outcome(status=status, reason=reason, detail=detail)

        # Charge the independent hot copy of selected warm payloads.
        copy_bytes = sum(
            len(record.text.encode("utf-8")) if record.text is not None else 0
            for record in selected_entries)
        copy_bytes += sum(
            len(payload) for entry in selected_entries if entry.values_available
            for payload in pack_values(entry.values))
        charged = 0
        selection_outcome = outcome
        if copy_bytes:
            try:
                self._charge("hot", "selection", copy_bytes,
                             f"selection:{selection_id}",
                             "independent selected text/value copy")
                charged = copy_bytes
            except CapacityExceeded as error:
                selection_outcome = _outcome(
                    "incomplete", "hot_capacity_for_selected_values",
                    used=error.used, limit=error.limit, requested=error.requested)
                # Do not silently return payloads that were not admitted hot.
                selected_entries = [replace(
                    record, values=(), values_available=False, value_slots=0,
                    text=None, text_included=False) for record in selected_entries]
                cold_ids = [record.record_id for record in selected_entries]
                requires_cold = True
                used_slots = 0
                used_text = 0
                span_hits = []
                used_span_bytes = 0
        selection_generations = tuple(sorted(pin_generation.items()))
        result = RetrievalResult(
            selection_id=selection_id,
            scope=scope,
            query=query,
            outcome=selection_outcome,
            records=tuple(selected_entries),
            exact_spans=tuple(span_hits),
            generation_pins=selection_generations,
            epoch=self.epoch,
            candidate_count=len(candidates),
            truncated=truncated,
            fallback_mode=fallback_mode,
            used_text_bytes=used_text,
            used_value_slots=used_slots,
            used_span_bytes=used_span_bytes,
            requires_cold_recall=requires_cold,
            cold_record_ids=tuple(cold_ids),
            representation_stale=representation_stale,
            budget_plan=budget.quota_plan(),
        )
        self._selections[selection_id] = {
            "result": result,
            "pins": tuple(pins),
            "charged": charged,
            "scope_key": scope.scope_key,
            "record_ids": tuple(selected_ids),
            "invalidated": False,
            "created_at_ns": now,
        }
        for record_id in selected_ids:
            self._note_access(record_id)
        return result

    def _build_pin(self, row: sqlite3.Row, key: sqlite3.Row,
                   values: Sequence[sqlite3.Row]) -> dict:
        return {
            "record_id": row["record_id"],
            "cluster_id": row["cluster_id"],
            "cluster_generation": int(self._cluster_row(row["cluster_id"])[
                "current_generation"]),
            "revision": int(row["revision"]),
            "scope_key": row["scope_key"],
            "status": row["status"],
            "text_path": row["text_path"],
            "text_sha256": row["text_sha256"],
            "text_bytes": int(row["text_bytes"]),
            "key_path": key["key_path"],
            "key_sha256": key["key_sha256"],
            "key_bytes": int(key["key_bytes"]),
            "key_status": key["status"],
            "key_producer_version": key["producer_version"],
            "values": tuple({
                "slot": int(value["slot_index"]),
                "path": value["value_path"],
                "sha256": value["value_sha256"],
                "bytes": int(value["value_bytes"]),
                "status": value["status"],
                "producer_version": value["producer_version"],
            } for value in values),
            "representation_version": row["representation_version"],
            "placement": row["placement"],
        }

    def _note_access(self, record_id: str) -> None:
        # Retrieval count is local frequency evidence, not truth or benefit.
        with self._transaction():
            self._set_meta(
                f"access:{record_id}",
                str(int(self._meta(f"access:{record_id}", "0")) + 1))
            self._set_meta(f"last_access:{record_id}", str(_now_ns()))

    def retrieval_count(self, record_id: str) -> int:
        return int(self._meta(f"access:{record_id}", "0"))

    def release_selection(self, selection_id: str) -> bool:
        entry = self._selections.pop(selection_id, None)
        if entry is None:
            return False
        if entry["charged"]:
            self._release(f"selection:{selection_id}")
        return True

    def selection(self, selection_id: str) -> RetrievalResult | None:
        entry = self._selections.get(selection_id)
        return None if entry is None else entry["result"]

    # -- cold recall -------------------------------------------------------

    def _records_with_status(self, record_ids: Sequence[str]) -> list[sqlite3.Row]:
        if not record_ids:
            return []
        placeholders = ",".join("?" for _ in record_ids)
        return list(self._connection.execute(
            f"SELECT * FROM records WHERE record_id IN ({placeholders})",
            tuple(record_ids)))

    def begin_recall(self, selection_id: str, *,
                     record_ids: Sequence[str] | None = None,
                     budget: RetrievalBudget | None = None) -> RecallTicket:
        """Pin a selection and return a waitable ticket; read no payload."""
        now = _now_ns()
        selection_entry = self._selections.get(selection_id)
        if selection_entry is None:
            return RecallTicket(
                ticket_id="tkt_" + uuid.uuid4().hex,
                selection_id=selection_id,
                scope=MemoryScope("missing", "missing", "missing", 0),
                record_ids=(), pinned=(), required_bytes=0, max_bytes=0,
                status="unsupported",
                outcome=_outcome("unsupported", "selection_unknown_after_restart",
                                 selection_id=selection_id),
                requested_at_ns=now)
        result: RetrievalResult = selection_entry["result"]
        requested = tuple(record_ids) if record_ids is not None else result.record_ids
        if not requested:
            requested = tuple(selection_entry["record_ids"])
        pins = [pin for pin in selection_entry["pins"] if pin["record_id"] in requested]
        pin_map = {pin["record_id"]: pin for pin in pins}
        cold_ids = tuple(pin["record_id"] for pin in pins
                         if pin["placement"] == "cold")
        budget = budget or RetrievalBudget(max_cold_bytes=(
            result.budget_plan.get("cold_io_bytes", DEFAULT_COLD_IO_BUDGET_BYTES)))
        ticket_id = "tkt_" + uuid.uuid4().hex
        if not cold_ids:
            ticket = RecallTicket(
                ticket_id=ticket_id, selection_id=selection_id, scope=result.scope,
                record_ids=requested, pinned=tuple(pins), required_bytes=0,
                max_bytes=budget.max_cold_bytes, status="unsupported",
                outcome=_outcome("unsupported", "no_cold_records_in_selection",
                                 record_ids=requested),
                requested_at_ns=now)
            self._tickets[ticket_id] = {"ticket": ticket, "selection": selection_entry}
            return ticket
        required = sum(int(pin["text_bytes"]) + sum(item["bytes"] for item in pin["values"])
                       for pin in pins if pin["placement"] == "cold")
        ticket = RecallTicket(
            ticket_id=ticket_id, selection_id=selection_id, scope=result.scope,
            record_ids=cold_ids, pinned=tuple(pin_map[rid] for rid in cold_ids),
            required_bytes=required, max_bytes=budget.max_cold_bytes,
            status="pending",
            outcome=_outcome("in_progress", "cold_recall_pending",
                             required_bytes=required, max_bytes=budget.max_cold_bytes),
            requested_at_ns=now)
        self._tickets[ticket_id] = {"ticket": ticket, "selection": selection_entry}
        return ticket

    def ticket(self, ticket_id: str) -> RecallTicket | None:
        entry = self._tickets.get(ticket_id)
        return None if entry is None else entry["ticket"]

    def _set_ticket(self, ticket_id: str, ticket: RecallTicket,
                    result: RecallResult | None = None) -> None:
        entry = self._tickets[ticket_id]
        entry["ticket"] = ticket
        if result is not None:
            self._ticket_results[ticket_id] = result

    def _validate_after_io(self, ticket: RecallTicket,
                           staged: Mapping[str, tuple[bytes, tuple[bytes, ...]]]
                           ) -> tuple[str, str, tuple[str, ...]]:
        checks: list[str] = []
        for pin in ticket.pinned:
            record_id = pin["record_id"]
            row = self._connection.execute(
                "SELECT * FROM records WHERE record_id=?", (record_id,)).fetchone()
            if row is None:
                return "denied", "record_missing_after_io", tuple(checks)
            if row["scope_key"] != ticket.scope.scope_key or row["scope_key"] != pin["scope_key"]:
                return "denied", "cross_scope_after_io", tuple(checks)
            checks.append(f"scope:{record_id}")
            if row["status"] != "current":
                return "invalidated", "record_revoked_or_superseded_during_recall", tuple(checks)
            checks.append(f"status:{record_id}")
            if int(row["revision"]) != pin["revision"]:
                return "invalidated", "record_revision_changed_during_recall", tuple(checks)
            checks.append(f"revision:{record_id}")
            cluster = self._connection.execute(
                "SELECT current_generation FROM clusters WHERE cluster_id=?",
                (row["cluster_id"],)).fetchone()
            if cluster is None or int(cluster["current_generation"]) != int(
                    pin["cluster_generation"]):
                return "stale", "cluster_generation_changed_during_recall", tuple(checks)
            checks.append(f"generation:{record_id}")
            if row["representation_version"] != pin["representation_version"]:
                return "stale", "representation_version_changed_during_recall", tuple(checks)
            checks.append(f"representation:{record_id}")
            key = self._key_row(record_id)
            if key is None or key["status"] != "current" or key[
                    "key_sha256"] != pin["key_sha256"]:
                return "stale", "retrieval_key_stale_during_recall", tuple(checks)
            checks.append(f"key:{record_id}")
            values = self._value_rows(record_id)
            pinned_values = tuple(pin["values"])
            if len(values) != len(pinned_values):
                return "stale", "value_representation_changed_during_recall", tuple(checks)
            for value, pinned in zip(values, pinned_values):
                if (value["status"] != "current"
                        or value["value_sha256"] != pinned["sha256"]
                        or int(value["slot_index"]) != int(pinned["slot"])):
                    return "stale", "value_representation_stale_during_recall", tuple(checks)
            checks.append(f"values:{record_id}")
            text_payload, value_payloads = staged[record_id]
            if (_sha256_bytes(text_payload) != pin["text_sha256"]
                    or len(value_payloads) != len(pinned_values)
                    or any(_sha256_bytes(payload) != pinned["sha256"]
                           for payload, pinned in zip(value_payloads, pinned_values))):
                return "failed", "payload_digest_mismatch", tuple(checks)
            checks.append(f"payload:{record_id}")
        return "ready", "post_io_revalidation_passed", tuple(checks)

    def resume_recall(self, ticket_id: str) -> RecallResult:
        entry = self._tickets.get(ticket_id)
        if entry is None:
            return RecallResult(
                ticket_id=ticket_id, selection_id="", status="unsupported",
                outcome=_outcome("unsupported", "recall_ticket_unknown"),
                records=(), io_bytes=0, checks=(), new_decision_boundary=False)
        ticket: RecallTicket = entry["ticket"]
        if ticket.status not in ("pending", "in_progress"):
            existing = self._ticket_results.get(ticket_id)
            if existing is not None:
                return existing
            return RecallResult(
                ticket_id=ticket_id, selection_id=ticket.selection_id,
                status=ticket.status, outcome=ticket.outcome, records=(),
                io_bytes=ticket.io_bytes, checks=(), new_decision_boundary=False)
        if ticket.required_bytes > ticket.max_bytes:
            outcome = _outcome(
                "incomplete", "cold_io_budget_exceeded",
                required_bytes=ticket.required_bytes, max_bytes=ticket.max_bytes)
            result = RecallResult(
                ticket_id=ticket_id, selection_id=ticket.selection_id,
                status="incomplete", outcome=outcome, records=(),
                io_bytes=0, checks=(), new_decision_boundary=False)
            failed = replace(ticket, status="incomplete", outcome=outcome)
            self._set_ticket(ticket_id, failed, result)
            return result
        staged: dict[str, tuple[bytes, tuple[bytes, ...]]] = {}
        io_bytes = 0
        warnings: list[str] = []
        try:
            for pin in ticket.pinned:
                text = _read_verified(
                    pin["text_path"], self.root, pin["text_sha256"],
                    int(pin["text_bytes"]))
                io_bytes += len(text)
                values = []
                for pinned in pin["values"]:
                    payload = _read_verified(
                        pinned["path"], self.root, pinned["sha256"],
                        int(pinned["bytes"]))
                    io_bytes += len(payload)
                    values.append(payload)
                staged[pin["record_id"]] = (text, tuple(values))
        except (CorruptState, OSError) as error:
            outcome = _outcome("failed", "cold_payload_verification_failed",
                               error=str(error))
            result = RecallResult(
                ticket_id=ticket_id, selection_id=ticket.selection_id,
                status="failed", outcome=outcome, records=(),
                io_bytes=io_bytes, checks=(), new_decision_boundary=False)
            self._set_ticket(ticket_id, replace(
                ticket, status="failed", outcome=outcome, io_bytes=io_bytes), result)
            return result
        status, reason, checks = self._validate_after_io(ticket, staged)
        if status != "ready":
            outcome = _outcome(
                status, reason, record_ids=ticket.record_ids,
                new_decision_boundary=True,
                note="discard all staged payloads; create a new selection")
            try:
                self._record_event("cold_recall_" + status, ticket.selection_id, {
                    "ticket_id": ticket_id, "reason": reason,
                    "record_ids": ticket.record_ids,
                })
            except CapacityExceeded as error:
                warnings.append(f"journal_capacity: {error}")
            result = RecallResult(
                ticket_id=ticket_id, selection_id=ticket.selection_id,
                status=status, outcome=outcome, records=(),
                io_bytes=io_bytes, checks=checks, new_decision_boundary=True,
                warnings=tuple(warnings))
            self._set_ticket(ticket_id, replace(
                ticket, status=status, outcome=outcome, io_bytes=io_bytes,
                decision_boundary=int(self._meta(
                    "epoch", "0")) + 1), result)
            return result
        recalled = []
        for pin in ticket.pinned:
            text_payload, value_payloads = staged[pin["record_id"]]
            values = (unpack_values(value_payloads) if value_payloads else ())
            recalled.append(RecalledRecord(
                record_id=pin["record_id"],
                text=text_payload.decode("utf-8"),
                text_sha256=_sha256_bytes(text_payload),
                values=values,
                value_slots=len(values),
                value_sha256=tuple(_sha256_bytes(payload) for payload in value_payloads),
                placement="cold",
            ))
        outcome = _outcome("ok", "cold_recall_ready", record_ids=ticket.record_ids)
        try:
            self._record_event("cold_recall_ready", ticket.selection_id, {
                "ticket_id": ticket_id, "io_bytes": io_bytes,
                "record_ids": ticket.record_ids,
            })
        except CapacityExceeded as error:
            warnings.append(f"journal_capacity: {error}")
        result = RecallResult(
            ticket_id=ticket_id, selection_id=ticket.selection_id,
            status="ready", outcome=outcome, records=tuple(recalled),
            io_bytes=io_bytes, checks=checks, new_decision_boundary=False,
            warnings=tuple(warnings))
        self._set_ticket(ticket_id, replace(
            ticket, status="ready", outcome=outcome, io_bytes=io_bytes), result)
        return result

    def wait_recall(self, ticket_id: str, timeout_seconds: float) -> RecallResult:
        """Bounded wait.  A zero/expired timeout performs no hidden I/O delay."""
        if not math.isfinite(timeout_seconds) or timeout_seconds < 0:
            raise ValueError("timeout_seconds must be finite and nonnegative")
        entry = self._tickets.get(ticket_id)
        if entry is None:
            return RecallResult(
                ticket_id=ticket_id, selection_id="", status="unsupported",
                outcome=_outcome("unsupported", "recall_ticket_unknown"),
                records=(), io_bytes=0, checks=(), new_decision_boundary=False)
        ticket: RecallTicket = entry["ticket"]
        if ticket.status not in ("pending", "in_progress"):
            return self.resume_recall(ticket_id)
        if timeout_seconds <= 0:
            result = RecallResult(
                ticket_id=ticket_id, selection_id=ticket.selection_id,
                status="in_progress",
                outcome=_outcome("in_progress", "recall_wait_expired_without_io",
                                 retry_with="resume_recall"),
                records=(), io_bytes=0, checks=(), new_decision_boundary=False)
            entry["ticket"] = replace(
                ticket, status="in_progress",
                outcome=_outcome("in_progress", "recall_waitable"))
            return result
        return self.resume_recall(ticket_id)

    def read_record(self, record_id: str, *, scope: MemoryScope,
                    budget: RetrievalBudget | None = None) -> RecallResult:
        """Direct scoped payload read.  Cross-scope access is denied before I/O."""
        self._validate_scope_argument(scope)
        row = self._connection.execute(
            "SELECT * FROM records WHERE record_id=?", (record_id,)).fetchone()
        if row is None:
            return RecallResult(
                ticket_id="", selection_id="", status="denied",
                outcome=_outcome("denied", "record_not_found"),
                records=(), io_bytes=0, checks=(), new_decision_boundary=False)
        if row["scope_key"] != scope.scope_key:
            return RecallResult(
                ticket_id="", selection_id="", status="denied",
                outcome=_outcome("denied", "cross_scope_access",
                                 record_scope=row["scope_key"],
                                 requested_scope=scope.scope_key),
                records=(), io_bytes=0, checks=(), new_decision_boundary=False)
        if row["status"] != "current":
            return RecallResult(
                ticket_id="", selection_id="", status="invalidated",
                outcome=_outcome("invalidated", "record_not_current",
                                 record_status=row["status"]),
                records=(), io_bytes=0, checks=(), new_decision_boundary=True)
        values = self._value_rows(record_id)
        key = self._key_row(record_id)
        pin = {
            "record_id": record_id,
            "cluster_id": row["cluster_id"],
            "cluster_generation": int(self._cluster_row(row["cluster_id"])[
                "current_generation"]),
            "revision": int(row["revision"]),
            "scope_key": row["scope_key"],
            "status": row["status"],
            "text_path": row["text_path"],
            "text_sha256": row["text_sha256"],
            "text_bytes": int(row["text_bytes"]),
            "key_path": "" if key is None else key["key_path"],
            "key_sha256": "" if key is None else key["key_sha256"],
            "key_bytes": 0 if key is None else int(key["key_bytes"]),
            "key_status": "missing" if key is None else key["status"],
            "key_producer_version": "" if key is None else key["producer_version"],
            "values": tuple({
                "slot": int(value["slot_index"]),
                "path": value["value_path"],
                "sha256": value["value_sha256"],
                "bytes": int(value["value_bytes"]),
                "status": value["status"],
                "producer_version": value["producer_version"],
            } for value in values),
            "representation_version": row["representation_version"],
            "placement": row["placement"],
        }
        budget = budget or RetrievalBudget()
        required = int(row["text_bytes"]) + sum(item["bytes"] for item in pin["values"])
        if required > budget.max_cold_bytes:
            return RecallResult(
                ticket_id="", selection_id="", status="incomplete",
                outcome=_outcome("incomplete", "cold_io_budget_exceeded",
                                 required_bytes=required,
                                 max_bytes=budget.max_cold_bytes),
                records=(), io_bytes=0, checks=(), new_decision_boundary=False)
        try:
            text = _read_verified(row["text_path"], self.root, row["text_sha256"],
                                  int(row["text_bytes"]))
            value_payloads = tuple(
                _read_verified(item["path"], self.root, item["sha256"], int(item["bytes"]))
                for item in pin["values"])
        except (CorruptState, OSError) as error:
            return RecallResult(
                ticket_id="", selection_id="", status="failed",
                outcome=_outcome("failed", "payload_verification_failed",
                                 error=str(error)),
                records=(), io_bytes=0, checks=(), new_decision_boundary=False)
        ticket = RecallTicket(
            ticket_id="direct_" + uuid.uuid4().hex, selection_id="",
            scope=scope, record_ids=(record_id,), pinned=(pin,),
            required_bytes=required, max_bytes=budget.max_cold_bytes,
            status="pending", outcome=_outcome("in_progress", "direct_read_pending"),
            requested_at_ns=_now_ns())
        staged = {record_id: (text, value_payloads)}
        status, reason, checks = self._validate_after_io(ticket, staged)
        if status != "ready":
            return RecallResult(
                ticket_id=ticket.ticket_id, selection_id="", status=status,
                outcome=_outcome(status, reason, new_decision_boundary=True),
                records=(), io_bytes=required, checks=checks,
                new_decision_boundary=True)
        recalled = RecalledRecord(
            record_id=record_id, text=text.decode("utf-8"),
            text_sha256=_sha256_bytes(text),
            values=unpack_values(value_payloads) if value_payloads else (),
            value_slots=len(value_payloads),
            value_sha256=tuple(_sha256_bytes(payload) for payload in value_payloads),
            placement=row["placement"])
        return RecallResult(
            ticket_id=ticket.ticket_id, selection_id="", status="ready",
            outcome=_outcome("ok", "direct_read_ready"), records=(recalled,),
            io_bytes=required, checks=checks, new_decision_boundary=False)

    # -- corrections, revocation and representation invalidation -----------

    def _quarantine_dependent_experts(self, record_id: str,
                                      record_hash: str) -> tuple[str, ...]:
        quarantined: list[str] = []
        for row in list(self._connection.execute("SELECT * FROM expert_shards")):
            lineage = tuple(json.loads(row["lineage_json"]))
            if record_id in lineage or record_hash in lineage:
                self._connection.execute(
                    "UPDATE expert_shards SET quarantine_status='quarantined', "
                    "quarantine_reason=?, updated_at_ns=? WHERE expert_id=?",
                    (f"dependent evidence {record_id} was corrected or revoked",
                     _now_ns(), row["expert_id"]))
                quarantined.append(row["expert_id"])
        return tuple(sorted(quarantined))

    def _invalidate_dependent_procedures(self, record_id: str,
                                         reason: str) -> tuple[tuple[str, int], ...]:
        invalidated: list[tuple[str, int]] = []
        for row in list(self._connection.execute(
                "SELECT skill_id, version, evidence_json FROM procedures")):
            evidence = json.loads(row["evidence_json"])
            if record_id in tuple(evidence.get("record_ids", ())):
                self._connection.execute(
                    "INSERT OR IGNORE INTO procedure_invalidations("
                    "skill_id, version, record_id, reason, at_ns) VALUES (?, ?, ?, ?, ?)",
                    (row["skill_id"], int(row["version"]), record_id, reason, _now_ns()))
                invalidated.append((row["skill_id"], int(row["version"])))
        return tuple(invalidated)

    def _mark_representation_stale(self, record_id: str) -> int:
        count = 0
        for table, column in (("record_keys", "status"), ("record_values", "status")):
            cursor = self._connection.execute(
                f"UPDATE {table} SET status='stale', updated_at_ns=? "
                f"WHERE record_id=? AND status='current'",
                (_now_ns(), record_id))
            count += max(0, cursor.rowcount)
        self._connection.execute(
            "UPDATE records SET representation_version=?, updated_at_ns=? "
            "WHERE record_id=?",
            ("rep_stale_" + uuid.uuid4().hex[:20], _now_ns(), record_id))
        self._invalidate_cache_entry(record_id)
        self._invalidate_selections_for(record_id)
        return count

    def _invalidate_cache_entry(self, record_id: str) -> None:
        if self._hot_cache.pop(record_id, None) is not None:
            self._release(f"cache:{record_id}")

    def _invalidate_selections_for(self, record_id: str) -> None:
        for selection_id, entry in self._selections.items():
            if record_id in entry["record_ids"] and not entry["invalidated"]:
                entry["invalidated"] = True
                if entry["charged"]:
                    self._release(f"selection:{selection_id}")
                    entry["charged"] = 0

    def correct_record(self, record_id: str, *, correction_id: str | None = None,
                       text: str, source_id: str, source_epoch: int,
                       observed_at: int, event_at: int | None = None,
                       valid_from: int | None = None, valid_to: int | None = None,
                       importance: int = 80, values: Sequence[Sequence[float]] | None = None,
                       value_producer_id: str | None = None,
                       value_producer_version: str | None = None,
                       scope: MemoryScope,
                       authenticity: str = "user_approved_correction",
                       verification_state: str = "verified_human") -> CorrectionResult:
        """Append a correction, invalidate dependent tiers and preserve old bytes."""
        self._validate_scope_argument(scope)
        old = self._connection.execute(
            "SELECT * FROM records WHERE record_id=?", (record_id,)).fetchone()
        if old is None:
            raise CorruptState(f"unknown record: {record_id}")
        if old["scope_key"] != scope.scope_key:
            raise ScopeViolation("correction scope differs from the target record")
        if old["status"] != "current":
            raise CorruptState("target record is not current")
        new_id = correction_id or f"{record_id}.corr.{uuid.uuid4().hex[:12]}"
        draft = EvidenceRecordDraft(
            record_id=new_id, text=text, source_id=source_id,
            source_epoch=source_epoch, observed_at=observed_at,
            event_at=event_at, valid_from=valid_from, valid_to=valid_to,
            authenticity=authenticity,
            verification_state=verification_state, importance=importance,
            critical=True, kind="correction", supersedes=record_id,
            values=None if values is None else tuple(tuple(item) for item in values),
            value_producer_id=value_producer_id,
            value_producer_version=value_producer_version,
            source_lineage=tuple(json.loads(old["source_lineage"])) + (record_id,),
        )
        written: list[str] = []
        old_file_paths = [old["text_path"]]
        now = _now_ns()
        try:
            with self._transaction():
                duplicate = self._connection.execute(
                    "SELECT 1 FROM records WHERE record_id=?", (new_id,)).fetchone()
                if duplicate is not None:
                    raise CorruptState(f"correction record already exists: {new_id}")
                placement = old["placement"]
                self._charge_record_files(draft, placement=placement)
                written.extend(path for path, _ in self._write_record_payloads(
                    draft, placement=placement))
                stale = self._mark_representation_stale(record_id)
                self._connection.execute(
                    "UPDATE records SET status='superseded', superseded_by=?, "
                    "revision=revision+1, updated_at_ns=? WHERE record_id=?",
                    (new_id, now, record_id))
                next_index = 1 + int(self._connection.execute(
                    "SELECT COALESCE(MAX(record_index), 0) AS value FROM records "
                    "WHERE cluster_id=?", (old["cluster_id"],)).fetchone()["value"])
                self._insert_record(
                    draft, cluster_id=old["cluster_id"],
                    scope_key=old["scope_key"], record_index=next_index,
                    placement=placement, now=now)
                quarantined = self._quarantine_dependent_experts(
                    record_id, old["record_hash"])
                invalidated = self._invalidate_dependent_procedures(
                    record_id, "exact evidence corrected")
                epoch = self._next_epoch()
                cluster = self._cluster_row(old["cluster_id"])
                generation, digest = self._publish_cluster_generation(
                    old["cluster_id"],
                    parent_generation=int(cluster["current_generation"]),
                    created_at_ns=now)
                self._record_event("record_corrected", record_id, {
                    "correction_id": new_id,
                    "cluster_id": old["cluster_id"],
                    "generation": generation,
                    "quarantined_experts": quarantined,
                    "invalidated_procedures": invalidated,
                    "stale_representations": stale,
                    "exact_unlearning_claimed": False,
                }, at_ns=now)
        except BaseException:
            self._cleanup_paths(written)
            raise
        del old_file_paths
        return CorrectionResult(
            outcome=_outcome("ok", "correction_appended",
                             generation=generation, epoch=epoch),
            old_record_id=record_id, new_record_id=new_id,
            cluster_id=old["cluster_id"], generation=generation,
            manifest_sha256=digest, quarantined_experts=quarantined,
            invalidated_procedures=invalidated, stale_representations=stale,
            epoch=epoch, exact_unlearning_claimed=False)

    def revoke_record(self, record_id: str, *, reason: str,
                      scope: MemoryScope) -> RevocationResult:
        self._validate_scope_argument(scope)
        _identifier("reason", reason)
        row = self._connection.execute(
            "SELECT * FROM records WHERE record_id=?", (record_id,)).fetchone()
        if row is None:
            raise CorruptState(f"unknown record: {record_id}")
        if row["scope_key"] != scope.scope_key:
            raise ScopeViolation("revocation scope differs from the record")
        if row["status"] != "current":
            raise CorruptState("target record is not current")
        now = _now_ns()
        with self._transaction():
            stale = self._mark_representation_stale(record_id)
            self._connection.execute(
                "UPDATE records SET status='revoked', revoked_reason=?, "
                "revision=revision+1, updated_at_ns=? WHERE record_id=?",
                (reason, now, record_id))
            quarantined = self._quarantine_dependent_experts(
                record_id, row["record_hash"])
            invalidated = self._invalidate_dependent_procedures(
                record_id, "exact evidence revoked")
            epoch = self._next_epoch()
            cluster = self._cluster_row(row["cluster_id"])
            generation, digest = self._publish_cluster_generation(
                row["cluster_id"], parent_generation=int(cluster["current_generation"]),
                created_at_ns=now)
            self._record_event("record_revoked", record_id, {
                "reason": reason, "cluster_id": row["cluster_id"],
                "generation": generation, "quarantined_experts": quarantined,
                "invalidated_procedures": invalidated,
                "exact_unlearning_claimed": False,
            }, at_ns=now)
        return RevocationResult(
            outcome=_outcome("ok", "record_revoked", generation=generation,
                             epoch=epoch),
            record_id=record_id, cluster_id=row["cluster_id"],
            generation=generation, quarantined_experts=quarantined,
            invalidated_procedures=invalidated, stale_representations=stale,
            epoch=epoch, exact_unlearning_claimed=False)

    def invalidate_representations(self, producer_version: str, *,
                                   reason: str) -> RepresentationInvalidation:
        _identifier("producer_version", producer_version)
        _identifier("reason", reason)
        now = _now_ns()
        affected: set[str] = set()
        stale_keys = stale_values = 0
        with self._transaction():
            for table, kind in (("record_keys", "key"), ("record_values", "value")):
                rows = list(self._connection.execute(
                    f"SELECT DISTINCT record_id FROM {table} WHERE producer_version=?",
                    (producer_version,)))
                for row in rows:
                    cursor = self._connection.execute(
                        f"UPDATE {table} SET status='stale', updated_at_ns=? "
                        f"WHERE producer_version=? AND record_id=? AND status='current'",
                        (now, producer_version, row["record_id"]))
                    if kind == "key":
                        stale_keys += max(0, cursor.rowcount)
                    else:
                        stale_values += max(0, cursor.rowcount)
                    affected.add(row["record_id"])
            for record_id, in self._connection.execute(
                    "SELECT record_id FROM record_keys WHERE producer_version=? "
                    "UNION SELECT record_id FROM record_values WHERE producer_version=?",
                    (producer_version, producer_version)):
                affected.add(record_id)
            cache_evicted: list[str] = []
            for record_id in affected:
                self._connection.execute(
                    "UPDATE records SET representation_version=?, updated_at_ns=? "
                    "WHERE record_id=?",
                    ("rep_stale_" + uuid.uuid4().hex[:20], now, record_id))
                if record_id in self._hot_cache:
                    cache_evicted.append(record_id)
                self._invalidate_cache_entry(record_id)
                self._invalidate_selections_for(record_id)
            clusters = sorted({row["cluster_id"] for row in self._connection.execute(
                "SELECT DISTINCT cluster_id FROM records WHERE record_id IN ("
                + ",".join("?" for _ in affected) + ")", tuple(affected))
            }) if affected else []
            published: list[tuple[str, int]] = []
            for cluster_id in clusters:
                cluster = self._cluster_row(cluster_id)
                generation, _ = self._publish_cluster_generation(
                    cluster_id, parent_generation=int(cluster["current_generation"]),
                    created_at_ns=now)
                published.append((cluster_id, generation))
            epoch = self._next_epoch()
            self._record_event("representations_invalidated", producer_version, {
                "reason": reason, "records": sorted(affected),
                "clusters": published, "reencode_required": True,
                "exact_unlearning_claimed": False,
            }, at_ns=now)
        return RepresentationInvalidation(
            outcome=_outcome("degraded", "representations_stale_reencode_required",
                             epoch=epoch, record_count=len(affected)),
            producer_version=producer_version, records=tuple(sorted(affected)),
            clusters=tuple(published), stale_keys=stale_keys,
            stale_values=stale_values, evicted_cache_entries=tuple(sorted(cache_evicted)),
            reencode_required=True)

    # -- hot cache with critical/frequent reservations ---------------------

    def _frequent_reserved_ids(self) -> set[str]:
        entries = [entry for entry in self._hot_cache.values()
                   if entry["retrieval_count"] >= self.ceilings.frequent_min_accesses]
        entries.sort(key=lambda entry: (-entry["retrieval_count"], entry["last_access_ns"]))
        budget = int(self.ceilings.hot_bytes * self.ceilings.frequent_reserve_fraction)
        selected: set[str] = set()
        used = 0
        for entry in entries:
            if used + entry["bytes"] > budget:
                break
            selected.add(entry["record_id"])
            used += entry["bytes"]
        return selected

    def _cache_reserved(self, record_id: str, reserved_frequent: set[str]) -> bool:
        entry = self._hot_cache[record_id]
        return bool(entry["critical"]) or record_id in reserved_frequent

    def _hot_nonidentity_used(self) -> int:
        total = self._usage_total_locked("hot")
        identity = self._category_total_locked("hot", "identity")
        return total - identity

    def _hot_overflow_for(self, additional: int) -> int:
        used = self._hot_nonidentity_used()
        limit = self.ceilings.hot_nonidentity_bytes
        return max(0, used + additional - limit)

    def _evict_cache_until(self, additional: int) -> tuple[bool, list[str]]:
        evicted: list[str] = []
        while self._hot_overflow_for(additional) > 0:
            reserved_frequent = self._frequent_reserved_ids()
            candidates = [entry for entry in self._hot_cache.values()
                          if not self._cache_reserved(
                              entry["record_id"], reserved_frequent)]
            if not candidates:
                return False, evicted
            victim = min(candidates, key=lambda entry: (
                entry["last_access_ns"], entry["retrieval_count"],
                entry["record_id"]))
            self._hot_cache.pop(victim["record_id"], None)
            self._release(f"cache:{victim['record_id']}")
            evicted.append(victim["record_id"])
        return True, evicted

    def cache_record(self, record_id: str) -> Outcome:
        """Read a warm payload into an accounted hot cache copy."""
        row = self._connection.execute(
            "SELECT * FROM records WHERE record_id=?", (record_id,)).fetchone()
        if row is None:
            return _outcome("denied", "record_not_found")
        if row["status"] != "current":
            return _outcome("invalidated", "record_not_current",
                            record_status=row["status"])
        if row["placement"] == "cold":
            return _outcome(
                "incomplete", "cold_payload_requires_wait_resume",
                record_id=record_id,
                note="use begin_recall/resume_recall; no implicit cold I/O")
        values = self._value_rows(record_id)
        try:
            text = _read_verified(row["text_path"], self.root, row["text_sha256"],
                                  int(row["text_bytes"]))
            value_payloads = tuple(
                _read_verified(value["value_path"], self.root,
                               value["value_sha256"], int(value["value_bytes"]))
                for value in values)
        except CorruptState as error:
            return _outcome("failed", "cache_payload_verification_failed",
                            error=str(error))
        total = len(text) + sum(len(payload) for payload in value_payloads)
        if record_id in self._hot_cache:
            self._release(f"cache:{record_id}")
            self._hot_cache.pop(record_id, None)
        with self._transaction():
            ok, evicted = self._evict_cache_until(total)
            if not ok:
                return _outcome(
                    "incomplete", "hot_cache_reserved_only",
                    required_bytes=total,
                    reserved_entries=tuple(sorted(self._hot_cache)))
            try:
                self._ensure_capacity_locked("hot", "cache", total)
            except CapacityExceeded as error:
                return _outcome(
                    "incomplete", "hot_cache_capacity", used=error.used,
                    limit=error.limit, requested=error.requested)
            self._connection.execute(
                "INSERT INTO tier_usage(component_id, tier, category, bytes, detail, "
                "updated_at_ns) VALUES (?, 'hot', 'cache', ?, ?, ?)",
                (f"cache:{record_id}", total, "independent hot cache copy", _now_ns()))
            self._hot_cache[record_id] = {
                "record_id": record_id,
                "bytes": total,
                "critical": bool(row["critical"]),
                "retrieval_count": self.retrieval_count(record_id),
                "last_access_ns": _now_ns(),
                "text": text,
                "values": value_payloads,
            }
        return _outcome("ok", "cached_in_hot", bytes=total,
                        evicted=tuple(evicted))

    def trim_hot_cache(self, target_bytes: int) -> Outcome:
        _nonnegative_int("target_bytes", target_bytes)
        evicted: list[str] = []
        with self._transaction():
            reserved_frequent = self._frequent_reserved_ids()
            current = sum(entry["bytes"] for entry in self._hot_cache.values())
            while current > target_bytes:
                candidates = [entry for entry in self._hot_cache.values()
                              if not self._cache_reserved(
                                  entry["record_id"], reserved_frequent)]
                if not candidates:
                    break
                victim = min(candidates, key=lambda entry: (
                    entry["last_access_ns"], entry["retrieval_count"],
                    entry["record_id"]))
                self._hot_cache.pop(victim["record_id"], None)
                self._release(f"cache:{victim['record_id']}")
                evicted.append(victim["record_id"])
                current -= victim["bytes"]
            remaining = sum(entry["bytes"] for entry in self._hot_cache.values())
        if remaining > target_bytes:
            return _outcome("incomplete", "hot_cache_reserved_only",
                            remaining_bytes=remaining, target_bytes=target_bytes,
                            evicted=tuple(evicted))
        return _outcome("ok", "hot_cache_trimmed", remaining_bytes=remaining,
                        evicted=tuple(evicted))

    def hot_cache_snapshot(self) -> tuple[dict, ...]:
        return tuple({
            "record_id": entry["record_id"],
            "bytes": entry["bytes"],
            "critical": entry["critical"],
            "retrieval_count": entry["retrieval_count"],
        } for entry in sorted(self._hot_cache.values(),
                              key=lambda item: item["record_id"]))

    # -- procedures (S) ----------------------------------------------------

    def record_procedure(self, skill_id: str, version: int, *, title: str,
                         applicability: Mapping[str, object],
                         prerequisites: Sequence[str],
                         steps: Sequence[str], outcome: str,
                         error: str | None = None,
                         evidence_record_ids: Sequence[str] = (),
                         verified_by: str | None = None) -> Outcome:
        _identifier("skill_id", skill_id)
        _nonnegative_int("version", version)
        _identifier("title", title)
        if outcome not in PROCEDURE_OUTCOMES:
            raise ValueError("procedure outcome must be verified or failed")
        if outcome == "failed" and error is None:
            raise ValueError("failed procedure evidence must name the failure")
        if not isinstance(applicability, Mapping):
            raise ValueError("applicability must be a mapping")
        for item in prerequisites:
            _identifier("prerequisite", item)
        for item in steps:
            _identifier("step", item)
        for record_id in evidence_record_ids:
            _identifier("evidence_record_id", record_id)
        now = _now_ns()
        with self._transaction():
            if self._connection.execute(
                    "SELECT 1 FROM procedures WHERE skill_id=? AND version=?",
                    (skill_id, version)).fetchone() is not None:
                raise CorruptState("procedure version already exists")
            self._connection.execute(
                "INSERT INTO procedures(skill_id, version, title, applicability_json, "
                "prerequisites_json, steps_json, outcome, error, evidence_json, "
                "verified_by, created_at_ns) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (skill_id, version, title, _canonical(dict(applicability)),
                 _canonical(list(prerequisites)), _canonical(list(steps)),
                 outcome, error, _canonical({
                     "record_ids": list(evidence_record_ids),
                     "verification": outcome,
                 }), verified_by, now))
            self._record_event("procedure_recorded", f"{skill_id}@{version}", {
                "outcome": outcome, "verified_by": verified_by,
            }, at_ns=now)
        return _outcome("ok", "procedure_recorded", outcome=outcome,
                        reusable=outcome == "verified")

    def procedures(self, *, include_failed: bool = True) -> tuple[dict, ...]:
        rows = list(self._connection.execute(
            "SELECT * FROM procedures ORDER BY skill_id, version"))
        invalidated = {
            (row["skill_id"], int(row["version"]))
            for row in self._connection.execute(
                "SELECT skill_id, version FROM procedure_invalidations")}
        output = []
        for row in rows:
            if not include_failed and row["outcome"] != "verified":
                continue
            key = (row["skill_id"], int(row["version"]))
            output.append({
                "skill_id": row["skill_id"],
                "version": int(row["version"]),
                "title": row["title"],
                "applicability": json.loads(row["applicability_json"]),
                "prerequisites": json.loads(row["prerequisites_json"]),
                "steps": json.loads(row["steps_json"]),
                "outcome": row["outcome"],
                "error": row["error"],
                "evidence": json.loads(row["evidence_json"]),
                "verified_by": row["verified_by"],
                "usable": row["outcome"] == "verified" and key not in invalidated,
                "invalidated_by_evidence": key in invalidated,
            })
        return tuple(output)

    # -- identity (I), phi, capability (K), workspace (C) ------------------

    def approve_identity(self, identity_id: str, version: int, *, core_text: str,
                         authority: str = "user_approved",
                         supersedes: int | None = None) -> IdentityAdmission:
        _identifier("identity_id", identity_id)
        _nonnegative_int("version", version)
        _identifier("authority", authority)
        if not isinstance(core_text, str) or not core_text:
            raise ValueError("approved identity core must be nonempty text")
        if authority != "user_approved":
            return IdentityAdmission(
                outcome=_outcome("denied", "identity_requires_user_approval",
                                 authority=authority),
                identity_id=identity_id, version=version)
        payload = core_text.encode("utf-8")
        if len(payload) > self.ceilings.identity_reserved_bytes:
            return IdentityAdmission(
                outcome=_outcome("incomplete", "identity_quota_exceeded",
                                 bytes=len(payload),
                                 limit=self.ceilings.identity_reserved_bytes),
                identity_id=identity_id, version=version)
        digest = _sha256_bytes(payload)
        now = _now_ns()
        relative = f"storage/hot/identity/{identity_id}/{version:08d}.txt"
        path = self.root / relative
        try:
            with self._transaction():
                if self._connection.execute(
                        "SELECT 1 FROM identity_versions WHERE identity_id=? AND version=?",
                        (identity_id, version)).fetchone() is not None:
                    raise CorruptState("identity version already exists")
                self._charge("hot", "identity", len(payload),
                             f"identity:{identity_id}:{version}",
                             "approved identity core")
                _atomic_write(path, payload)
                self._connection.execute(
                    "UPDATE identity_versions SET status='superseded' "
                    "WHERE identity_id=? AND status='current'", (identity_id,))
                self._connection.execute(
                    "INSERT INTO identity_versions(identity_id, version, status, "
                    "core_path, core_sha, core_bytes, authority, approved, supersedes, "
                    "created_at_ns) VALUES (?, ?, 'current', ?, ?, ?, ?, 1, ?, ?)",
                    (identity_id, version, relative, digest, len(payload), authority,
                     supersedes, now))
                self._record_event("identity_approved", identity_id, {
                    "version": version, "bytes": len(payload), "authority": authority,
                }, at_ns=now)
        except CapacityExceeded as error:
            return IdentityAdmission(
                outcome=_outcome("incomplete", "identity_quota_exceeded",
                                 used=error.used, limit=error.limit,
                                 requested=error.requested),
                identity_id=identity_id, version=version)
        return IdentityAdmission(
            outcome=_outcome("ok", "identity_approved", authority=authority),
            identity_id=identity_id, version=version, sha256=digest, bytes=len(payload))

    def identity_core(self, identity_id: str | None = None) -> IdentityView:
        if identity_id is None:
            row = self._connection.execute(
                "SELECT * FROM identity_versions WHERE status='current' "
                "ORDER BY created_at_ns DESC LIMIT 1").fetchone()
        else:
            row = self._connection.execute(
                "SELECT * FROM identity_versions WHERE identity_id=? AND status='current'",
                (identity_id,)).fetchone()
        if row is None:
            return IdentityView(
                outcome=_outcome("incomplete", "no_approved_identity_core"),
                identity_id=identity_id, version=None, text=None, sha256=None,
                bytes=0, authority=None)
        try:
            payload = _read_verified(row["core_path"], self.root, row["core_sha"],
                                     int(row["core_bytes"]))
        except CorruptState as error:
            return IdentityView(
                outcome=_outcome("failed", "identity_core_verification_failed",
                                 error=str(error)),
                identity_id=row["identity_id"], version=int(row["version"]),
                text=None, sha256=row["core_sha"], bytes=int(row["core_bytes"]),
                authority=row["authority"])
        return IdentityView(
            outcome=_outcome("ok", "identity_core_available"),
            identity_id=row["identity_id"], version=int(row["version"]),
            text=payload.decode("utf-8"), sha256=row["core_sha"],
            bytes=int(row["core_bytes"]), authority=row["authority"])

    def register_phi_release(self, codec_id: str, version: int, *,
                             producer_version: str, artifact_sha256: str,
                             status: str = "unqualified") -> Outcome:
        _identifier("codec_id", codec_id)
        _nonnegative_int("version", version)
        _identifier("producer_version", producer_version)
        if not _SHA_RE.fullmatch(artifact_sha256):
            raise ValueError("phi artifact receipt requires an exact SHA256")
        _identifier("status", status)
        now = _now_ns()
        with self._transaction():
            if self._connection.execute(
                    "SELECT 1 FROM phi_releases WHERE codec_id=? AND version=?",
                    (codec_id, version)).fetchone() is not None:
                raise CorruptState("phi release already exists")
            self._connection.execute(
                "INSERT INTO phi_releases(codec_id, version, producer_version, "
                "artifact_sha, status, created_at_ns) VALUES (?, ?, ?, ?, ?, ?)",
                (codec_id, version, producer_version, artifact_sha256, status, now))
            self._record_event("phi_release_registered", f"{codec_id}@{version}", {
                "producer_version": producer_version, "status": status,
            }, at_ns=now)
        return _outcome("ok", "phi_release_registered",
                        engineering_only=True)

    def phi_releases(self) -> tuple[dict, ...]:
        return tuple({
            "codec_id": row["codec_id"],
            "version": int(row["version"]),
            "producer_version": row["producer_version"],
            "artifact_sha256": row["artifact_sha"],
            "status": row["status"],
        } for row in self._connection.execute(
            "SELECT * FROM phi_releases ORDER BY codec_id, version"))

    def record_capability(self, capability_id: str, *, model_id: str, mode: str,
                          task_family: str, budget: Mapping[str, object],
                          measured_value: float | None,
                          uncertainty: Mapping[str, object],
                          measured: bool) -> Outcome:
        _identifier("capability_id", capability_id)
        _identifier("model_id", model_id)
        _identifier("mode", mode)
        _identifier("task_family", task_family)
        if not isinstance(budget, Mapping) or not isinstance(uncertainty, Mapping):
            raise ValueError("capability budget and uncertainty must be mappings")
        if not isinstance(measured, bool):
            raise ValueError("measured must be boolean")
        if measured and (measured_value is None or not math.isfinite(measured_value)):
            raise ValueError("measured capability requires a finite measured value")
        if not measured and measured_value is not None:
            raise ValueError("unmeasured capability cannot claim a measured value")
        now = _now_ns()
        with self._transaction():
            if self._connection.execute(
                    "SELECT 1 FROM capability_records WHERE capability_id=?",
                    (capability_id,)).fetchone() is not None:
                raise CorruptState("capability record already exists")
            self._connection.execute(
                "INSERT INTO capability_records(capability_id, model_id, mode, "
                "task_family, budget_json, measured_value, uncertainty_json, measured, "
                "created_at_ns) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (capability_id, model_id, mode, task_family, _canonical(dict(budget)),
                 measured_value, _canonical(dict(uncertainty)), int(measured), now))
            self._record_event("capability_recorded", capability_id, {
                "measured": measured, "task_family": task_family,
            }, at_ns=now)
        return _outcome("ok", "capability_recorded", measured=measured)

    def capabilities(self, *, task_family: str | None = None) -> tuple[dict, ...]:
        if task_family is None:
            rows = self._connection.execute(
                "SELECT * FROM capability_records ORDER BY created_at_ns")
        else:
            rows = self._connection.execute(
                "SELECT * FROM capability_records WHERE task_family=? "
                "ORDER BY created_at_ns", (task_family,))
        return tuple({
            "capability_id": row["capability_id"],
            "model_id": row["model_id"],
            "mode": row["mode"],
            "task_family": row["task_family"],
            "budget": json.loads(row["budget_json"]),
            "measured_value": row["measured_value"],
            "uncertainty": json.loads(row["uncertainty_json"]),
            "measured": bool(row["measured"]),
        } for row in rows)

    def save_workspace_snapshot(self, task_id: str, payload: Mapping[str, object] | bytes,
                                *, resumable: bool = False) -> WorkspaceReceipt:
        _identifier("task_id", task_id)
        if isinstance(payload, Mapping):
            data = _canonical(dict(payload)).encode("utf-8")
        elif isinstance(payload, bytes):
            data = payload
        else:
            raise ValueError("workspace payload must be a mapping or bytes")
        if len(data) > self.ceilings.workspace_quota_bytes:
            return WorkspaceReceipt(
                outcome=_outcome("incomplete", "workspace_quota_exceeded",
                                 bytes=len(data),
                                 limit=self.ceilings.workspace_quota_bytes),
                task_id=task_id, path="", sha256="", bytes=len(data),
                resumable=resumable, tier="")
        tier = "cold" if resumable else "hot"
        relative = f"storage/{tier}/workspace/{task_id}.json"
        path = self.root / relative
        digest = _sha256_bytes(data)
        component = f"workspace:{task_id}"
        existing = self._connection.execute(
            "SELECT * FROM workspace_snapshots WHERE task_id=?", (task_id,)).fetchone()
        old_bytes = 0 if existing is None else int(existing["bytes"])
        old_tier = None if existing is None else (
            "cold" if int(existing["resumable"]) else "hot")
        try:
            with self._transaction():
                if old_bytes:
                    self._release(component)
                try:
                    self._charge(tier, "workspace", len(data), component,
                                 f"task snapshot resumable={resumable}")
                except CapacityExceeded as error:
                    if old_bytes:
                        # Restore the old accounting component atomically.
                        self._set_component_locked(
                            component, old_tier, "workspace", old_bytes,
                            "restored task snapshot")
                    raise
                _atomic_write(path, data, overwrite=True)
                self._connection.execute(
                    "INSERT INTO workspace_snapshots(task_id, path, sha, bytes, "
                    "resumable, created_at_ns) VALUES (?, ?, ?, ?, ?, ?) "
                    "ON CONFLICT(task_id) DO UPDATE SET path=excluded.path, "
                    "sha=excluded.sha, bytes=excluded.bytes, "
                    "resumable=excluded.resumable, created_at_ns=excluded.created_at_ns",
                    (task_id, relative, digest, len(data), int(resumable), _now_ns()))
                self._record_event("workspace_snapshot", task_id, {
                    "bytes": len(data), "resumable": resumable, "tier": tier,
                })
        except CapacityExceeded as error:
            return WorkspaceReceipt(
                outcome=_outcome("incomplete", "workspace_capacity",
                                 used=error.used, limit=error.limit,
                                 requested=error.requested),
                task_id=task_id, path="", sha256="", bytes=len(data),
                resumable=resumable, tier=tier)
        return WorkspaceReceipt(
            outcome=_outcome("ok", "workspace_snapshot_saved"),
            task_id=task_id, path=relative, sha256=digest, bytes=len(data),
            resumable=resumable, tier=tier)

    def discard_workspace(self, task_id: str) -> bool:
        row = self._connection.execute(
            "SELECT * FROM workspace_snapshots WHERE task_id=?", (task_id,)).fetchone()
        if row is None:
            return False
        with self._transaction():
            try:
                (self.root / row["path"]).unlink()
            except FileNotFoundError:
                pass
            self._release(f"workspace:{task_id}")
            self._connection.execute(
                "DELETE FROM workspace_snapshots WHERE task_id=?", (task_id,))
            self._record_event("workspace_discarded", task_id, {})
        return True

    def workspace_snapshots(self) -> tuple[dict, ...]:
        return tuple({
            "task_id": row["task_id"],
            "path": row["path"],
            "sha256": row["sha"],
            "bytes": int(row["bytes"]),
            "resumable": bool(row["resumable"]),
        } for row in self._connection.execute(
            "SELECT * FROM workspace_snapshots ORDER BY task_id"))

    # -- P-query budget (separate from evidence values) --------------------

    def read_p_query(self, expert_id: str, *, slots: int = P_QUERY_SLOTS,
                     budget: RetrievalBudget | None = None) -> dict:
        """Read P-owner metadata under a separate slot budget.

        This does not allocate tensors, return evidence values or borrow the
        32 evidence value slots.
        """
        _nonnegative_int("slots", slots)
        budget = budget or RetrievalBudget()
        if slots > budget.p_query_slots:
            raise BudgetError("P-query slots exceed the declared separate budget")
        row = self._connection.execute(
            "SELECT * FROM expert_shards WHERE expert_id=?", (expert_id,)).fetchone()
        if row is None:
            return {"outcome": _outcome("denied", "expert_not_found").document()}
        return {
            "outcome": _outcome("ok", "p_owner_metadata_only").document(),
            "expert_id": expert_id,
            "profile_id": row["profile_id"],
            "slots": slots,
            "p_query_slots_limit": budget.p_query_slots,
            "evidence_value_slots_borrowed": 0,
            "materialized_parameters": 0,
            "parameter_values_returned": False,
            "quarantine_status": row["quarantine_status"],
            "optimizer": row["optimizer"],
            "muon": MUON_STATUS,
        }

    # -- expert shards (P) -------------------------------------------------

    def _active_expert_limit(self) -> int:
        return ACTIVE_EXPERTS_MAX if self.two_active_ablation else ACTIVE_EXPERTS_INITIAL

    def _active_expert_ids(self) -> tuple[str, ...]:
        return tuple(row["expert_id"] for row in self._connection.execute(
            "SELECT expert_id FROM expert_shards WHERE location='hot' "
            "ORDER BY expert_id"))

    def register_expert(self, expert_id: str, *, profile_id: str | None = None,
                        lineage: Sequence[str] = (),
                        encoder_id: str = "synthetic-encoder-v1") -> ExpertRecord:
        """Register cold metadata only; no bank, shard or tensor is allocated."""
        _identifier("expert_id", expert_id)
        _identifier("encoder_id", encoder_id)
        profile = get_profile(profile_id or self.profile_id)
        values = tuple(lineage)
        for item in values:
            _identifier("lineage item", item)
        now = _now_ns()
        with self._transaction():
            if self._connection.execute(
                    "SELECT 1 FROM expert_shards WHERE expert_id=?",
                    (expert_id,)).fetchone() is not None:
                raise CorruptState(f"expert already registered: {expert_id}")
            self._connection.execute(
                "INSERT INTO expert_shards(expert_id, profile_id, parameter_count, "
                "location, dirty, health, quarantine_status, quarantine_reason, "
                "publication_generation, hot_weights_path, hot_moments_path, hot_bytes, "
                "published_weights_sha, published_moments_sha, published_clock, "
                "encoder_id, lineage_json, lineage_digest, optimizer, ablation, "
                "updated_at_ns) VALUES (?, ?, ?, 'cold', 0, 'ok', 'none', NULL, 0, "
                "NULL, NULL, 0, NULL, NULL, NULL, ?, ?, ?, ?, ?, ?)",
                (expert_id, profile.profile_id, profile.full_parameters, encoder_id,
                 _canonical(list(values)), _sha256_json(list(values)),
                 NATIVE_OPTIMIZER, self.ablation_label, now))
            self._record_event("expert_registered", expert_id, {
                "profile_id": profile.profile_id,
                "parameter_count": profile.full_parameters,
                "location": "cold",
                "materialized_bytes": 0,
                "all_bank_allocated": False,
            }, at_ns=now)
        return ExpertRecord(
            expert_id=expert_id, profile_id=profile.profile_id,
            parameter_count=profile.full_parameters, location="cold", dirty=False,
            quarantine_status="none", publication_generation=0,
            encoder_id=encoder_id, optimizer=NATIVE_OPTIMIZER,
            ablation=self.ablation_label)

    def write_expert_state(self, expert_id: str, *, weights: bytes, moments: bytes,
                           clock: int, lineage: Sequence[str] | None = None,
                           encoder_id: str | None = None) -> Outcome:
        """Write one active expert's hot state.  Bytes are caller-supplied payloads."""
        if not isinstance(weights, bytes) or not isinstance(moments, bytes):
            raise ValueError("expert weights and moments must be bytes")
        _nonnegative_int("clock", clock)
        row = self._connection.execute(
            "SELECT * FROM expert_shards WHERE expert_id=?", (expert_id,)).fetchone()
        if row is None:
            return _outcome("denied", "expert_not_registered")
        profile = get_profile(row["profile_id"])
        if len(weights) != profile.full_parameters * 4:
            raise ValueError(
                "weights must be exactly parameter_count FP32 bytes; "
                "this module does not reinterpret shapes")
        if len(moments) != profile.full_parameters * 8:
            raise ValueError("moments must be exactly two FP32 AdamW state arrays")
        activated = row["location"] != "hot"
        if activated and len(self._active_expert_ids()) >= self._active_expert_limit():
            return _outcome(
                "unsupported", "active_expert_limit",
                limit=self._active_expert_limit(),
                two_active_ablation=self.two_active_ablation,
                note="increasing active work requires the separate ablation")
        lineage_value = tuple(json.loads(row["lineage_json"])) if lineage is None else tuple(lineage)
        for item in lineage_value:
            _identifier("lineage item", item)
        encoder = encoder_id or row["encoder_id"]
        _identifier("encoder_id", encoder)
        total = len(weights) + len(moments)
        hot_weight_path = self.root / f"storage/hot/experts/{expert_id}/weights.bin"
        hot_moment_path = self.root / f"storage/hot/experts/{expert_id}/moments.bin"
        old_bytes = int(row["hot_bytes"])
        component = f"expert:{expert_id}:hot"
        try:
            with self._transaction():
                if old_bytes:
                    self._release(component)
                try:
                    self._charge("hot", "expert_active", total, component,
                                 "one active expert weights and AdamW moments")
                except CapacityExceeded as error:
                    if old_bytes:
                        self._set_component_locked(
                            component, "hot", "expert_active", old_bytes,
                            "restored active expert accounting")
                    return _outcome("incomplete", "hot_expert_capacity",
                                    used=error.used, limit=error.limit,
                                    requested=error.requested)
                _atomic_write(hot_weight_path, weights, overwrite=True)
                _atomic_write(hot_moment_path, moments, overwrite=True)
                self._connection.execute(
                    "UPDATE expert_shards SET location='hot', dirty=1, "
                    "hot_weights_path=?, hot_moments_path=?, hot_bytes=?, "
                    "encoder_id=?, lineage_json=?, lineage_digest=?, updated_at_ns=? "
                    "WHERE expert_id=?",
                    (self._relative(hot_weight_path), self._relative(hot_moment_path),
                     total, encoder, _canonical(list(lineage_value)),
                     _sha256_json(list(lineage_value)), _now_ns(), expert_id))
                self._set_meta(f"expert_clock:{expert_id}", str(clock))
                self._record_event("expert_state_written", expert_id, {
                    "clock": clock, "bytes": total, "dirty": True,
                    "active_experts": len(self._active_expert_ids()),
                })
        except CapacityExceeded as error:
            return _outcome("incomplete", "hot_expert_capacity",
                            used=error.used, limit=error.limit,
                            requested=error.requested)
        return _outcome("ok", "expert_state_dirty", bytes=total, clock=clock,
                        active_experts=len(self._active_expert_ids()))

    def reset_expert(self, expert_id: str, *, weights: bytes, moments: bytes,
                     clock: int, lineage: Sequence[str] = (), encoder_id: str | None = None,
                     authorization: str, reason: str) -> Outcome:
        """Replace dependent state after quarantine; never claim exact unlearning."""
        _identifier("authorization", authorization)
        _identifier("reason", reason)
        if authorization != "authenticated_reset":
            return _outcome("denied", "authenticated_reset_required")
        row = self._connection.execute(
            "SELECT * FROM expert_shards WHERE expert_id=?", (expert_id,)).fetchone()
        if row is None:
            return _outcome("denied", "expert_not_registered")
        previous = row["quarantine_status"]
        written = self.write_expert_state(
            expert_id, weights=weights, moments=moments, clock=clock,
            lineage=lineage, encoder_id=encoder_id)
        if not written.ok:
            return written
        loaded = self.load_expert(expert_id)
        if loaded.state is None:
            return _outcome("incomplete", "reset_readback_failed", quarantine_retained=True)
        expected_weights = _sha256_bytes(weights)
        expected_moments = _sha256_bytes(moments)
        if (loaded.state.weights_sha256 != expected_weights
                or loaded.state.moments_sha256 != expected_moments):
            return _outcome("incomplete", "reset_readback_digest_mismatch",
                            quarantine_retained=True)
        with self._transaction():
            self._connection.execute(
                "UPDATE expert_shards SET quarantine_status='none', quarantine_reason=NULL, "
                "health='ok', updated_at_ns=? WHERE expert_id=?", (_now_ns(), expert_id))
            self._record_event("expert_authenticated_reset", expert_id, {
                "authorization": authorization, "reason": reason,
                "previous_quarantine": previous, "clock": int(clock),
                "lineage": list(lineage), "exact_unlearning_claimed": False})
        return _outcome("ok", "expert_authenticated_reset",
                        previous_quarantine=previous, clock=int(clock),
                        weights_sha256=expected_weights, moments_sha256=expected_moments,
                        exact_unlearning_claimed=False)

    def publish_expert(self, expert_id: str) -> Outcome:
        """Atomically publish weights, moments, clock and lineage to cold."""
        row = self._connection.execute(
            "SELECT * FROM expert_shards WHERE expert_id=?", (expert_id,)).fetchone()
        if row is None:
            return _outcome("denied", "expert_not_registered")
        if row["location"] != "hot" or not row["dirty"]:
            return _outcome("ok", "expert_already_durable",
                            generation=int(row["publication_generation"]),
                            dirty=bool(row["dirty"]))
        if not row["hot_weights_path"] or not row["hot_moments_path"]:
            return _outcome("failed", "hot_expert_paths_missing")
        try:
            weights = (self.root / row["hot_weights_path"]).read_bytes()
            moments = (self.root / row["hot_moments_path"]).read_bytes()
        except OSError as error:
            return _outcome("failed", "hot_expert_state_missing", error=str(error))
        profile = get_profile(row["profile_id"])
        if (len(weights) != profile.full_parameters * 4
                or len(moments) != profile.full_parameters * 8):
            return _outcome("failed", "hot_expert_shape_mismatch",
                            weights_bytes=len(weights), moments_bytes=len(moments))
        weights_sha = _sha256_bytes(weights)
        moments_sha = _sha256_bytes(moments)
        lineage = tuple(json.loads(row["lineage_json"]))
        generation = int(row["publication_generation"]) + 1
        now = _now_ns()
        weight_relative = (f"storage/cold/experts/{expert_id}/gen-{generation:08d}/"
                           "weights.bin")
        moment_relative = (f"storage/cold/experts/{expert_id}/gen-{generation:08d}/"
                           "moments.bin")
        weight_path = self.root / weight_relative
        moment_path = self.root / moment_relative
        manifest = {
            "format": EXPERT_MANIFEST_FORMAT,
            "store_format": FORMAT,
            "expert_id": expert_id,
            "profile_id": row["profile_id"],
            "parameter_count": int(row["parameter_count"]),
            "generation": generation,
            "clock": 0,
            "optimizer": NATIVE_OPTIMIZER,
            "muon": MUON_STATUS,
            "encoder_id": row["encoder_id"],
            "lineage": list(lineage),
            "lineage_digest": _sha256_json(list(lineage)),
            "weights": {"path": weight_relative, "sha256": weights_sha,
                        "bytes": len(weights), "dtype": "float32"},
            "moments": {"path": moment_relative, "sha256": moments_sha,
                        "bytes": len(moments), "dtype": "float32",
                        "optimizer": NATIVE_OPTIMIZER},
            "quarantine_status": row["quarantine_status"],
            "created_at_ns": now,
            "engineering_payload": True,
            "native_capability": False,
            "exact_unlearning_claimed": False,
        }
        # The authoritative teaching clock was supplied with the hot state and
        # is persisted in the expert row's publication receipt below.
        published_clock = int(self._pending_clock(expert_id, now))
        manifest["clock"] = published_clock
        envelope = self._manifest_envelope(manifest)
        digest = _sha256_json(manifest)
        manifest_relative = (f"manifests/experts/{expert_id}/"
                             f"{generation:08d}.json")
        component_files = (f"expert:{expert_id}:cold:{generation}")
        charged = 0
        try:
            with self._transaction():
                total = len(weights) + len(moments) + len(envelope)
                self._ensure_capacity_locked("cold", "expert_state", total)
                for relative, payload in ((weight_relative, weights),
                                          (moment_relative, moments),
                                          (manifest_relative, envelope)):
                    path = self.root / relative
                    if path.exists():
                        raise CorruptState(f"expert publication exists: {relative}")
                self._charge("cold", "expert_state", total, component_files,
                             "immutable expert weights/moments/manifest generation")
                charged = total
                _atomic_write(weight_path, weights)
                _atomic_write(moment_path, moments)
                _atomic_write(self.root / manifest_relative, envelope)
                self._connection.execute(
                    "INSERT INTO expert_publications(expert_id, generation, "
                    "manifest_path, manifest_sha256, weights_path, weights_sha, "
                    "weights_bytes, moments_path, moments_sha, moments_bytes, clock, "
                    "created_at_ns) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (expert_id, generation, manifest_relative, digest, weight_relative,
                     weights_sha, len(weights), moment_relative, moments_sha, len(moments),
                     published_clock, now))
                self._connection.execute(
                    "UPDATE expert_shards SET publication_generation=?, dirty=0, "
                    "published_weights_sha=?, published_moments_sha=?, "
                    "published_clock=?, health='ok', updated_at_ns=? "
                    "WHERE expert_id=?",
                    (generation, weights_sha, moments_sha, published_clock, now,
                     expert_id))
                self._record_event("expert_published", expert_id, {
                    "generation": generation, "clock": published_clock,
                    "weights_sha256": weights_sha, "moments_sha256": moments_sha,
                    "lineage": lineage, "optimizer": NATIVE_OPTIMIZER,
                }, at_ns=now)
        except CapacityExceeded as error:
            return _outcome("incomplete", "cold_expert_publication_capacity",
                            used=error.used, limit=error.limit,
                            requested=error.requested, dirty=True)
        except BaseException:
            if charged:
                self._release(component_files)
            raise
        return _outcome("ok", "expert_published", generation=generation,
                        digest=digest, bytes=charged, clock=published_clock,
                        muon=MUON_STATUS)

    def _pending_clock(self, expert_id: str, now: int | None = None) -> int:
        """Return the clock supplied by the most recent hot write for this expert.

        The clock is part of the dirty state, not the wall clock.  It is stored
        in a meta key so a process crash does not silently advance it.
        """
        del now
        return int(self._meta(f"expert_clock:{expert_id}", "0"))

    def evict_expert(self, expert_id: str) -> Outcome:
        """Publish dirty weights/moments/clock before discarding the hot copy."""
        row = self._connection.execute(
            "SELECT * FROM expert_shards WHERE expert_id=?", (expert_id,)).fetchone()
        if row is None:
            return _outcome("denied", "expert_not_registered")
        if row["location"] != "hot":
            return _outcome("ok", "expert_already_cold",
                            generation=int(row["publication_generation"]))
        if row["dirty"]:
            published = self.publish_expert(expert_id)
            if not published.ok:
                return _outcome(
                    "incomplete", "dirty_expert_not_published_refusing_eviction",
                    publication=published.document(),
                    dirty_retained=True, hot_retained=True)
        # Read back the durable generation before deleting the only hot copy.
        loaded = self.load_expert(expert_id)
        if (not loaded.outcome.ok or loaded.state is None
                or loaded.state.source != "cold_publication"):
            return _outcome(
                "incomplete", "cold_publication_unverified_refusing_eviction",
                load=loaded.document(), dirty_retained=bool(row["dirty"]),
                hot_retained=True)
        try:
            for relative in (row["hot_weights_path"], row["hot_moments_path"]):
                if relative:
                    (self.root / relative).unlink(missing_ok=True)
        except OSError as error:
            return _outcome("failed", "hot_expert_cleanup_failed", error=str(error))
        with self._transaction():
            self._release(f"expert:{expert_id}:hot")
            self._connection.execute(
                "UPDATE expert_shards SET location='cold', dirty=0, "
                "hot_weights_path=NULL, hot_moments_path=NULL, hot_bytes=0, "
                "updated_at_ns=? WHERE expert_id=?", (_now_ns(), expert_id))
            self._record_event("expert_evicted", expert_id, {
                "location": "cold",
                "publication_generation": int(self._connection.execute(
                    "SELECT publication_generation FROM expert_shards WHERE expert_id=?",
                    (expert_id,)).fetchone()["publication_generation"]),
                "dirty_published_before_eviction": True,
            })
        return _outcome("ok", "expert_evicted_after_publication",
                        generation=int(self._connection.execute(
                            "SELECT publication_generation FROM expert_shards "
                            "WHERE expert_id=?", (expert_id,)).fetchone()[
                                "publication_generation"]))

    def load_expert(self, expert_id: str) -> ExpertLoad:
        row = self._connection.execute(
            "SELECT * FROM expert_shards WHERE expert_id=?", (expert_id,)).fetchone()
        if row is None:
            return ExpertLoad(_outcome("denied", "expert_not_registered"), None)
        if row["health"] != "ok":
            return ExpertLoad(
                _outcome("failed", "expert_health_not_ok", health=row["health"]), None)
        if row["location"] == "hot" and row["dirty"]:
            try:
                weights = (self.root / row["hot_weights_path"]).read_bytes()
                moments = (self.root / row["hot_moments_path"]).read_bytes()
            except OSError:
                return ExpertLoad(_outcome("failed", "hot_expert_state_missing"), None)
            return ExpertLoad(
                _outcome("degraded" if row["quarantine_status"] != "none" else "ok",
                         "hot_dirty_expert_loaded",
                         quarantine_status=row["quarantine_status"]),
                ExpertState(
                    expert_id=expert_id, weights=weights, moments=moments,
                    clock=int(self._pending_clock(expert_id, _now_ns())),
                    lineage=tuple(json.loads(row["lineage_json"])),
                    encoder_id=row["encoder_id"],
                    generation=int(row["publication_generation"]),
                    source="hot_dirty", dirty=True,
                    quarantine_status=row["quarantine_status"],
                    weights_sha256=_sha256_bytes(weights),
                    moments_sha256=_sha256_bytes(moments)))
        generation = int(row["publication_generation"])
        if generation <= 0:
            return ExpertLoad(
                _outcome("incomplete", "expert_never_materialized",
                         location=row["location"]), None)
        publication = self._connection.execute(
            "SELECT * FROM expert_publications WHERE expert_id=? AND generation=?",
            (expert_id, generation)).fetchone()
        if publication is None:
            return ExpertLoad(
                _outcome("failed", "expert_publication_index_missing",
                         generation=generation), None)
        try:
            weights = _read_verified(
                publication["weights_path"], self.root, publication["weights_sha"],
                int(publication["weights_bytes"]))
            moments = _read_verified(
                publication["moments_path"], self.root, publication["moments_sha"],
                int(publication["moments_bytes"]))
            manifest, digest = self._read_manifest(
                self._absolute(publication["manifest_path"]), EXPERT_MANIFEST_FORMAT)
        except CorruptState as error:
            return ExpertLoad(
                _outcome("failed", "expert_publication_verification_failed",
                         error=str(error)), None)
        if digest != publication["manifest_sha256"]:
            return ExpertLoad(_outcome("failed", "expert_manifest_digest_mismatch"), None)
        return ExpertLoad(
            _outcome("degraded" if row["quarantine_status"] != "none" else "ok",
                     "cold_expert_loaded",
                     quarantine_status=row["quarantine_status"],
                     exact_unlearning_claimed=False),
            ExpertState(
                expert_id=expert_id, weights=weights, moments=moments,
                clock=int(manifest.get("clock", publication["clock"])),
                lineage=tuple(json.loads(row["lineage_json"])),
                encoder_id=row["encoder_id"], generation=generation,
                source="cold_publication", dirty=False,
                quarantine_status=row["quarantine_status"],
                weights_sha256=_sha256_bytes(weights),
                moments_sha256=_sha256_bytes(moments)))

    def expert_records(self) -> tuple[ExpertRecord, ...]:
        return tuple(ExpertRecord(
            expert_id=row["expert_id"], profile_id=row["profile_id"],
            parameter_count=int(row["parameter_count"]), location=row["location"],
            dirty=bool(row["dirty"]), quarantine_status=row["quarantine_status"],
            publication_generation=int(row["publication_generation"]),
            encoder_id=row["encoder_id"], optimizer=row["optimizer"],
            ablation=row["ablation"])
            for row in self._connection.execute(
                "SELECT * FROM expert_shards ORDER BY expert_id"))

    def allocation(self, *, active_experts: int = ACTIVE_EXPERTS_INITIAL,
                   online_parameters: str = "full") -> ExpertAllocationPlan:
        return plan_profile(
            self.profile_id, active_experts=active_experts,
            two_active_ablation=self.two_active_ablation,
            online_parameters=online_parameters)

    # -- replay sub-cap ----------------------------------------------------

    def write_replay(self, episode_id: str, payload: bytes) -> Outcome:
        _identifier("episode_id", episode_id)
        if not isinstance(payload, bytes) or not payload:
            raise ValueError("replay payload must be nonempty bytes")
        relative = f"storage/warm/replay/{episode_id}.bin"
        path = self.root / relative
        if path.exists():
            return _outcome("denied", "replay_episode_already_exists")
        component = f"replay:{episode_id}"
        try:
            with self._transaction():
                self._charge("warm", "replay", len(payload), component,
                             "bounded replay episode")
                _atomic_write(path, payload)
                self._connection.execute(
                    "INSERT INTO replay_entries(episode_id, path, sha, bytes, "
                    "created_at_ns) VALUES (?, ?, ?, ?, ?)",
                    (episode_id, relative, _sha256_bytes(payload), len(payload),
                     _now_ns()))
        except CapacityExceeded as error:
            return _outcome("incomplete", "replay_capacity",
                            used=error.used, limit=error.limit,
                            requested=error.requested)
        return _outcome("ok", "replay_written", bytes=len(payload),
                        tier="warm", subcap_bytes=self.ceilings.replay_reserved_bytes)

    # -- publication overlap staging --------------------------------------

    def stage_publication(self, reservation_id: str, payload: bytes) -> Outcome:
        """Write a real temporary publication copy and account the overlap."""
        _identifier("reservation_id", reservation_id)
        if not isinstance(payload, bytes) or not payload:
            raise ValueError("publication payload must be nonempty bytes")
        relative = f"staging/{reservation_id}.bin"
        path = self.root / relative
        if path.exists():
            return _outcome("denied", "publication_reservation_already_exists")
        component = f"publication:{reservation_id}"
        try:
            with self._transaction():
                self._charge("cold", "publication_overlap", len(payload), component,
                             "temporary publication overlap")
                _atomic_write(path, payload)
                self._record_event("publication_staged", reservation_id, {
                    "bytes": len(payload), "tier": "cold",
                    "counted_as_overlap": True,
                })
        except CapacityExceeded as error:
            return _outcome("incomplete", "publication_overlap_capacity",
                            used=error.used, limit=error.limit,
                            requested=error.requested)
        return _outcome("ok", "publication_staged", bytes=len(payload),
                        path=relative, tier="cold")

    def release_publication(self, reservation_id: str) -> Outcome:
        _identifier("reservation_id", reservation_id)
        relative = f"staging/{reservation_id}.bin"
        path = self.root / relative
        if not path.exists():
            return _outcome("incomplete", "publication_reservation_not_found")
        size = path.stat().st_size
        with self._transaction():
            path.unlink()
            self._release(f"publication:{reservation_id}")
            self._record_event("publication_released", reservation_id, {
                "bytes": size, "overlap_removed": True,
            })
        return _outcome("ok", "publication_released", bytes=size)

    # -- checkpoints -------------------------------------------------------

    def publish_checkpoint(self, checkpoint_id: str,
                           payload: Mapping[str, object] | bytes) -> Outcome:
        _identifier("checkpoint_id", checkpoint_id)
        data = (payload if isinstance(payload, bytes)
                else _canonical(dict(payload)).encode("utf-8"))
        if not data:
            raise ValueError("checkpoint payload must be nonempty")
        last = self._connection.execute(
            "SELECT COALESCE(MAX(generation), 0) AS value FROM checkpoint_generations "
            "WHERE checkpoint_id=?", (checkpoint_id,)).fetchone()["value"]
        generation = int(last) + 1
        now = _now_ns()
        digest = _sha256_bytes(data)
        payload_relative = (f"storage/cold/checkpoints/{checkpoint_id}/"
                            f"gen-{generation:08d}/payload.bin")
        manifest_relative = (f"manifests/checkpoints/{checkpoint_id}/"
                             f"{generation:08d}.json")
        manifest = {
            "format": CHECKPOINT_MANIFEST_FORMAT,
            "store_format": FORMAT,
            "checkpoint_id": checkpoint_id,
            "generation": generation,
            "payload": {"path": payload_relative, "sha256": digest,
                        "bytes": len(data)},
            "created_at_ns": now,
            "engineering_evidence": True,
            "native_capability": False,
        }
        envelope = self._manifest_envelope(manifest)
        manifest_digest = _sha256_json(manifest)
        total = len(data) + len(envelope)
        component = f"checkpoint:{checkpoint_id}:{generation}"
        try:
            with self._transaction():
                self._charge("cold", "checkpoint", total, component,
                             "retained checkpoint generation")
                _atomic_write(self.root / payload_relative, data)
                _atomic_write(self.root / manifest_relative, envelope)
                self._connection.execute(
                    "INSERT INTO checkpoint_generations(checkpoint_id, generation, "
                    "payload_path, payload_sha, payload_bytes, manifest_path, "
                    "manifest_sha, created_at_ns) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (checkpoint_id, generation, payload_relative, digest, len(data),
                     manifest_relative, manifest_digest, now))
                self._record_event("checkpoint_published", checkpoint_id, {
                    "generation": generation, "bytes": total,
                }, at_ns=now)
        except CapacityExceeded as error:
            return _outcome("incomplete", "checkpoint_capacity",
                            used=error.used, limit=error.limit,
                            requested=error.requested)
        return _outcome("ok", "checkpoint_published", generation=generation,
                        bytes=total, digest=manifest_digest)

    def checkpoints(self) -> tuple[dict, ...]:
        return tuple({
            "checkpoint_id": row["checkpoint_id"],
            "generation": int(row["generation"]),
            "payload_path": row["payload_path"],
            "payload_sha256": row["payload_sha"],
            "payload_bytes": int(row["payload_bytes"]),
            "manifest_sha256": row["manifest_sha"],
        } for row in self._connection.execute(
            "SELECT * FROM checkpoint_generations ORDER BY checkpoint_id, generation"))

    # -- bounded ignored-artifact storage ----------------------------------

    def _artifact_path(self, digest: str) -> Path:
        return self._directories["artifacts"] / digest[:2] / f"{digest}.bin"

    def store_artifact(self, source: Path | str, *, pinned: bool = False,
                       label: str | None = None) -> ArtifactReceipt:
        """Store a bounded artifact; type detection ignores the filename suffix."""
        source = Path(source).resolve()
        _no_symlink(source)
        if not source.is_file():
            raise CorruptState(f"artifact source is not a regular file: {source}")
        if label is not None:
            _identifier("label", label)
        detected, magic_hex, size = detect_file_type(source)
        digest = hashlib.sha256()
        with source.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        sha = digest.hexdigest()
        target = self._artifact_path(sha)
        relative = self._relative(target)
        existing = self._connection.execute(
            "SELECT * FROM artifacts WHERE sha256=?", (sha,)).fetchone()
        if existing is not None:
            return ArtifactReceipt(
                status="existing", artifact_id=existing["artifact_id"],
                path=existing["path"], bytes=int(existing["bytes"]),
                sha256=sha, detected_type=existing["detected_type"],
                magic_hex=existing["magic_hex"], pinned=bool(existing["pinned"]),
                label=existing["label"],
                outcome=_outcome("ok", "artifact_already_stored"))
        artifact_used = int(self._connection.execute(
            "SELECT COALESCE(SUM(bytes), 0) AS total FROM tier_usage "
            "WHERE category='artifacts'").fetchone()["total"])
        if artifact_used + size > self.ceilings.artifact_budget_bytes:
            return ArtifactReceipt(
                status="rejected", artifact_id=sha, path="", bytes=size,
                sha256=sha, detected_type=detected, magic_hex=magic_hex,
                pinned=pinned, label=label,
                outcome=_outcome("incomplete", "artifact_budget_exceeded",
                                 used=artifact_used,
                                 limit=self.ceilings.artifact_budget_bytes,
                                 requested=size,
                                 existing_preserved=True))
        try:
            with self._transaction():
                self._charge("cold", "artifacts", size, f"artifact:{sha}",
                             f"bounded artifact {detected}")
                _copy_atomic(source, target, sha, size)
                self._connection.execute(
                    "INSERT INTO artifacts(artifact_id, path, bytes, sha256, "
                    "detected_type, magic_hex, pinned, label, source_path, status, "
                    "created_at_ns) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'stored', ?)",
                    (sha, relative, size, sha, detected, magic_hex, int(pinned),
                     label, str(source), _now_ns()))
                self._record_event("artifact_stored", sha, {
                    "bytes": size, "type": detected, "pinned": pinned,
                    "extension_used_for_detection": False,
                })
        except CapacityExceeded as error:
            return ArtifactReceipt(
                status="rejected", artifact_id=sha, path="", bytes=size,
                sha256=sha, detected_type=detected, magic_hex=magic_hex,
                pinned=pinned, label=label,
                outcome=_outcome("incomplete", "artifact_capacity",
                                 used=error.used, limit=error.limit,
                                 requested=error.requested,
                                 existing_preserved=True))
        except BaseException:
            try:
                target.unlink(missing_ok=True)
            except OSError:
                pass
            self._release(f"artifact:{sha}")
            raise
        return ArtifactReceipt(
            status="stored", artifact_id=sha, path=relative, bytes=size,
            sha256=sha, detected_type=detected, magic_hex=magic_hex,
            pinned=pinned, label=label,
            outcome=_outcome("ok", "artifact_stored"))

    def scan_artifacts(self) -> ArtifactScan:
        """Issue receipts for files already under artifacts, suffix-independent."""
        receipts: list[ArtifactReceipt] = []
        total = 0
        over_budget = False
        known = {row["path"]: row for row in self._connection.execute(
            "SELECT * FROM artifacts")}
        for path in sorted(self._directories["artifacts"].rglob("*")):
            if not path.is_file():
                continue
            relative = self._relative(path)
            size = path.stat().st_size
            total += size
            if relative in known:
                row = known[relative]
                receipts.append(ArtifactReceipt(
                    status="existing", artifact_id=row["artifact_id"],
                    path=relative, bytes=int(row["bytes"]), sha256=row["sha256"],
                    detected_type=row["detected_type"], magic_hex=row["magic_hex"],
                    pinned=bool(row["pinned"]), label=row["label"],
                    outcome=_outcome("ok", "artifact_receipt_reused")))
                continue
            detected, magic_hex, measured = detect_file_type(path)
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            sha = digest.hexdigest()
            artifact_used = int(self._connection.execute(
                "SELECT COALESCE(SUM(bytes), 0) AS total FROM tier_usage "
                "WHERE category='artifacts'").fetchone()["total"])
            within = (artifact_used + measured
                      <= self.ceilings.artifact_budget_bytes)
            if not within:
                over_budget = True
            with self._transaction():
                if within:
                    self._set_component_locked(
                        f"artifact:{sha}", "cold", "artifacts", measured,
                        f"scanned artifact {detected}")
                self._connection.execute(
                    "INSERT OR REPLACE INTO artifacts(artifact_id, path, bytes, "
                    "sha256, detected_type, magic_hex, pinned, label, source_path, "
                    "status, created_at_ns) VALUES (?, ?, ?, ?, ?, ?, 0, NULL, ?, ?, ?)",
                    (sha, relative, measured, sha, detected, magic_hex, str(path),
                     "scanned" if within else "over_budget", _now_ns()))
                self._record_event("artifact_scanned", str(relative), {
                    "bytes": measured, "type": detected, "within_budget": within,
                })
            receipts.append(ArtifactReceipt(
                status="scanned" if within else "rejected", artifact_id=sha,
                path=relative, bytes=measured, sha256=sha, detected_type=detected,
                magic_hex=magic_hex, pinned=False, label=None,
                outcome=_outcome("ok" if within else "incomplete",
                                 "artifact_scanned" if within
                                 else "artifact_scan_over_budget",
                                 existing_file_preserved=True)))
        return ArtifactScan(
            outcome=_outcome("ok" if not over_budget else "incomplete",
                             "artifact_scan_complete" if not over_budget
                             else "artifact_scan_over_budget",
                             existing_preserved=True),
            receipts=tuple(receipts), total_bytes=total, over_budget=over_budget)

    def artifact_receipts(self) -> tuple[dict, ...]:
        return tuple({
            "artifact_id": row["artifact_id"],
            "path": row["path"],
            "bytes": int(row["bytes"]),
            "sha256": row["sha256"],
            "detected_type": row["detected_type"],
            "magic_hex": row["magic_hex"],
            "pinned": bool(row["pinned"]),
            "label": row["label"],
            "status": row["status"],
        } for row in self._connection.execute(
            "SELECT * FROM artifacts ORDER BY created_at_ns"))

    # -- migration between warm and cold -----------------------------------

    def migrate_records(self, record_ids: Sequence[str], *, to_tier: str) -> Outcome:
        """Physically move text and ordered values; keys stay in the warm index."""
        if to_tier not in PLACEMENTS:
            raise ValueError("to_tier must be warm or cold")
        moved = 0
        for record_id in record_ids:
            row = self._connection.execute(
                "SELECT * FROM records WHERE record_id=?", (record_id,)).fetchone()
            if row is None:
                return _outcome("denied", "record_not_found", record_id=record_id)
            if row["placement"] == to_tier:
                continue
            values = self._value_rows(record_id)
            moves = [(row["text_path"], f"storage/{to_tier}/records/{record_id}/text.bin",
                      f"record:{record_id}:text", int(row["text_bytes"]),
                      "canonical_text")]
            moves.extend(
                (value["value_path"],
                 f"storage/{to_tier}/records/{record_id}/slot-{value['slot_index']}.f32",
                 f"record:{record_id}:value:{value['slot_index']}",
                 int(value["value_bytes"]), "values")
                for value in values)
            total = sum(item[3] for item in moves)
            try:
                with self._transaction():
                    self._ensure_capacity_locked(to_tier, "migration", total)
                    destination_root = self.root / f"storage/{to_tier}/records/{record_id}"
                    destination_root.mkdir(parents=True, exist_ok=True)
                    for source_relative, target_relative, component, byte_count, category in moves:
                        source = self.root / source_relative
                        target = self.root / target_relative
                        if target.exists():
                            raise CorruptState(f"migration target exists: {target_relative}")
                        self._release(component)
                        try:
                            self._charge(to_tier, category, byte_count, component,
                                         f"migrated {category} to {to_tier}")
                        except CapacityExceeded:
                            self._set_component_locked(
                                component, row["placement"], category, byte_count,
                                "restored after failed migration")
                            raise
                        os.replace(source, target)
                    for source_relative, *_ in moves:
                        parent = (self.root / source_relative).parent
                        if parent.is_dir() and not any(parent.iterdir()):
                            parent.rmdir()
                    self._connection.execute(
                        "UPDATE records SET placement=?, updated_at_ns=? "
                        "WHERE record_id=?", (to_tier, _now_ns(), record_id))
                    self._connection.execute(
                        "UPDATE record_values SET value_path=replace(value_path, ?, ?) "
                        "WHERE record_id=?",
                        (f"storage/{row['placement']}/", f"storage/{to_tier}/",
                         record_id))
                    self._connection.execute(
                        "UPDATE records SET text_path=replace(text_path, ?, ?) "
                        "WHERE record_id=?",
                        (f"storage/{row['placement']}/", f"storage/{to_tier}/",
                         record_id))
                    self._invalidate_cache_entry(record_id)
                    cluster = self._cluster_row(row["cluster_id"])
                    self._publish_cluster_generation(
                        row["cluster_id"],
                        parent_generation=int(cluster["current_generation"]),
                        created_at_ns=_now_ns())
                    self._record_event("records_migrated", record_id, {
                        "from": row["placement"], "to": to_tier, "bytes": total,
                        "keys_remain": "warm",
                    })
                    moved += 1
            except CapacityExceeded as error:
                return _outcome("incomplete", "migration_capacity",
                                record_id=record_id, used=error.used,
                                limit=error.limit, requested=error.requested,
                                note="no file was moved for this record")
        return _outcome("ok", "records_migrated", count=moved, to_tier=to_tier)

    # -- replay/disk audit and recovery ------------------------------------

    def audit_disk(self) -> dict:
        areas = {
            "hot": self.root / "storage/hot",
            "warm": self.root / "storage/warm",
            "cold": self.root / "storage/cold",
            "manifests": self.root / "manifests",
            "journal": self.root / "journal",
            "artifacts": self.root / "artifacts",
            "workspace": self.root / "workspace",
            "staging": self.root / "staging",
            "database": self.sqlite_path,
        }
        bytes_by_area: dict[str, int] = {}
        files_by_area: dict[str, int] = {}
        interrupted: list[str] = []
        for name, path in areas.items():
            total = 0
            count = 0
            if path.is_file():
                total = path.stat().st_size
                count = 1
            elif path.is_dir():
                for candidate in path.rglob("*"):
                    if candidate.is_file():
                        total += candidate.stat().st_size
                        count += 1
                        if candidate.name.startswith(".tmp-"):
                            interrupted.append(self._relative(candidate))
            bytes_by_area[name] = total
            files_by_area[name] = count
        for suffix, name in (("-wal", "database_wal"), ("-shm", "database_shm")):
            path = Path(str(self.sqlite_path) + suffix)
            size = _safe_file_bytes(path)
            if size:
                bytes_by_area[name] = size
                files_by_area[name] = 1
        usage = self.usage()
        free = shutil.disk_usage(self.root).free
        journal_path = self._directories["journal"] / "events.jsonl"
        journal_lines = journal_errors = 0
        if journal_path.is_file():
            with journal_path.open("r", encoding="utf-8", errors="replace") as stream:
                for line in stream:
                    if not line.strip():
                        continue
                    try:
                        json.loads(line)
                    except json.JSONDecodeError:
                        journal_errors += 1
                    else:
                        journal_lines += 1
        return {
            "format": "replica.tiered_memory.disk_audit.v1",
            "root": str(self.root),
            "bytes_by_area": bytes_by_area,
            "files_by_area": files_by_area,
            "total_bytes": sum(bytes_by_area.values()),
            "volume_free_bytes": free,
            "minimum_volume_free_bytes": self.ceilings.minimum_volume_free_bytes,
            "volume_headroom_satisfied": free >= self.ceilings.minimum_volume_free_bytes,
            "interrupted_temporaries": tuple(sorted(interrupted)),
            "journal_lines": journal_lines,
            "journal_parse_errors": journal_errors,
            "accounting": usage,
            "cold_logical_used_bytes": usage["tiers"]["cold"]["used_bytes"],
            "cold_logical_limit_bytes": self.ceilings.cold_bytes,
            "base_model_weights_counted": False,
            "page_cache_pressure_is_not_added_to_logical_usage": True,
            "native_capability": False,
        }

    def journal_events(self, *, limit: int = 100) -> tuple[dict, ...]:
        _nonnegative_int("limit", limit)
        rows = self._connection.execute(
            "SELECT * FROM journal_events ORDER BY event_id DESC LIMIT ?",
            (limit,)).fetchall()
        return tuple({
            "event_id": int(row["event_id"]),
            "at_ns": int(row["at_ns"]),
            "kind": row["kind"],
            "entity_id": row["entity_id"],
            "detail": json.loads(row["detail_json"]),
            "sha256": row["sha256"],
        } for row in reversed(rows))

    def recover(self) -> RecoveryReport:
        report = self._recover_locked(rebuild_index=True, remove_temporaries=True)
        self._recovery = report
        return report

    def _recover_locked(self, *, rebuild_index: bool,
                        remove_temporaries: bool) -> RecoveryReport:
        notes: list[str] = []
        corrupt_manifests: list[str] = []
        corrupt_artifacts: list[str] = []
        ignored: list[str] = []
        clusters_recovered: set[str] = set()
        generations_recovered = 0
        records_recovered = 0
        experts_recovered: set[str] = set()
        cluster_manifests: dict[str, list[tuple[int, Path, dict, str]]] = {}
        expert_manifests: dict[str, list[tuple[int, Path, dict, str]]] = {}
        for path in sorted(self._directories["manifests_clusters"].rglob("*.json")):
            try:
                manifest, digest = self._read_manifest(path, CLUSTER_MANIFEST_FORMAT)
                cluster_id = manifest["cluster_id"]
                generation = int(manifest["generation"])
                cluster_manifests.setdefault(cluster_id, []).append(
                    (generation, path, manifest, digest))
            except (CorruptState, KeyError, TypeError, ValueError) as error:
                corrupt_manifests.append(self._relative(path))
                notes.append(f"manifest_preserved: {error}")
        for path in sorted(self._directories["manifests_experts"].rglob("*.json")):
            try:
                manifest, digest = self._read_manifest(path, EXPERT_MANIFEST_FORMAT)
                expert_id = manifest["expert_id"]
                generation = int(manifest["generation"])
                expert_manifests.setdefault(expert_id, []).append(
                    (generation, path, manifest, digest))
            except (CorruptState, KeyError, TypeError, ValueError) as error:
                corrupt_manifests.append(self._relative(path))
                notes.append(f"expert_manifest_preserved: {error}")
        for pattern in (".tmp-*", "*.tmp-*"):
            for path in self.root.rglob(pattern):
                if path.is_file():
                    ignored.append(self._relative(path))
                    if remove_temporaries:
                        try:
                            path.unlink()
                        except OSError:
                            notes.append(f"temporary_not_removed: {path}")
        if rebuild_index:
            with self._transaction():
                for cluster_id, items in sorted(cluster_manifests.items()):
                    items.sort(key=lambda item: item[0])
                    latest_generation = 0
                    latest_manifest: dict | None = None
                    latest_digest = ""
                    latest_path: Path | None = None
                    for generation, path, manifest, digest in items:
                        existing = self._connection.execute(
                            "SELECT manifest_sha256 FROM cluster_generations "
                            "WHERE cluster_id=? AND generation=?",
                            (cluster_id, generation)).fetchone()
                        if existing is None:
                            self._connection.execute(
                                "INSERT INTO cluster_generations(cluster_id, generation, "
                                "scope_key, manifest_path, manifest_sha256, "
                                "parent_generation, record_count, created_at_ns) "
                                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                                (cluster_id, generation, manifest["scope_key"],
                                 self._relative(path), digest,
                                 manifest.get("parent_generation"),
                                 int(manifest["record_count"]),
                                 int(manifest["created_at_ns"])))
                            generations_recovered += 1
                        scope_document = manifest["scope"]
                        scope_key = manifest["scope_key"]
                        existing_cluster = self._connection.execute(
                            "SELECT current_generation FROM clusters WHERE cluster_id=?",
                            (cluster_id,)).fetchone()
                        if existing_cluster is None:
                            self._connection.execute(
                                "INSERT INTO clusters(cluster_id, name, cluster_kind, "
                                "scope_key, scope_json, current_generation, "
                                "manifest_sha256, record_count, status, created_at_ns, "
                                "updated_at_ns) VALUES (?, ?, ?, ?, ?, 0, NULL, 0, "
                                "'active', ?, ?)",
                                (cluster_id, manifest["cluster_name"],
                                 manifest["cluster_kind"], scope_key,
                                 _canonical(scope_document),
                                 int(manifest["created_at_ns"]),
                                 int(manifest["created_at_ns"])))
                        if latest_manifest is None or generation >= latest_generation:
                            latest_generation = generation
                            latest_manifest = manifest
                            latest_digest = digest
                            latest_path = path
                        for link in manifest.get("links", ()):
                            self._connection.execute(
                                "INSERT OR IGNORE INTO cluster_links(cluster_id, "
                                "linked_cluster_id, relation) VALUES (?, ?, ?)",
                                (cluster_id, link["linked_cluster_id"],
                                 link["relation"]))
                        for entry in manifest.get("records", ()):
                            self._rebuild_record_row(cluster_id, entry)
                            records_recovered += 1
                    if latest_manifest is not None and latest_path is not None:
                        current = self._connection.execute(
                            "SELECT current_generation FROM clusters WHERE cluster_id=?",
                            (cluster_id,)).fetchone()
                        if current is None or int(current["current_generation"]) < (
                                latest_generation):
                            self._connection.execute(
                                "UPDATE clusters SET current_generation=?, "
                                "manifest_sha256=?, record_count=?, updated_at_ns=? "
                                "WHERE cluster_id=?",
                                (latest_generation, latest_digest,
                                 int(latest_manifest["record_count"]),
                                 int(latest_manifest["created_at_ns"]), cluster_id))
                        clusters_recovered.add(cluster_id)
                        if latest_manifest["scope_key"] != latest_manifest["scope"][
                                "scope_key"]:
                            notes.append(f"scope_key_mismatch: {cluster_id}")
                for expert_id, items in sorted(expert_manifests.items()):
                    items.sort(key=lambda item: item[0])
                    for generation, path, manifest, digest in items:
                        try:
                            weights = _read_verified(
                                manifest["weights"]["path"], self.root,
                                manifest["weights"]["sha256"],
                                int(manifest["weights"]["bytes"]))
                            moments = _read_verified(
                                manifest["moments"]["path"], self.root,
                                manifest["moments"]["sha256"],
                                int(manifest["moments"]["bytes"]))
                        except (CorruptState, KeyError, TypeError) as error:
                            corrupt_artifacts.append(self._relative(path))
                            notes.append(f"expert_payload_missing: {expert_id}: {error}")
                            continue
                        existing = self._connection.execute(
                            "SELECT * FROM expert_shards WHERE expert_id=?",
                            (expert_id,)).fetchone()
                        lineage = tuple(manifest.get("lineage", ()))
                        if existing is None:
                            self._connection.execute(
                                "INSERT INTO expert_shards(expert_id, profile_id, "
                                "parameter_count, location, dirty, health, "
                                "quarantine_status, quarantine_reason, "
                                "publication_generation, hot_weights_path, "
                                "hot_moments_path, hot_bytes, published_weights_sha, "
                                "published_moments_sha, published_clock, encoder_id, "
                                "lineage_json, lineage_digest, optimizer, ablation, "
                                "updated_at_ns) VALUES (?, ?, ?, 'cold', 0, 'ok', ?, "
                                "NULL, ?, NULL, NULL, 0, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                                (expert_id, manifest.get("profile_id", self.profile_id),
                                 int(manifest.get("parameter_count", len(weights) // 4)),
                                 manifest.get("quarantine_status", "none"),
                                 generation, manifest["weights"]["sha256"],
                                 manifest["moments"]["sha256"], int(manifest["clock"]),
                                 manifest.get("encoder_id", "unknown"),
                                 _canonical(list(lineage)), _sha256_json(list(lineage)),
                                 manifest.get("optimizer", NATIVE_OPTIMIZER),
                                 self.ablation_label, _now_ns()))
                            self._connection.execute(
                                "INSERT OR IGNORE INTO expert_publications(expert_id, "
                                "generation, manifest_path, manifest_sha256, "
                                "weights_path, weights_sha, weights_bytes, moments_path, "
                                "moments_sha, moments_bytes, clock, created_at_ns) "
                                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                                (expert_id, generation, self._relative(path), digest,
                                 manifest["weights"]["path"],
                                 manifest["weights"]["sha256"],
                                 int(manifest["weights"]["bytes"]),
                                 manifest["moments"]["path"],
                                 manifest["moments"]["sha256"],
                                 int(manifest["moments"]["bytes"]),
                                 int(manifest["clock"]),
                                 int(manifest["created_at_ns"])))
                            experts_recovered.add(expert_id)
                        elif int(existing["publication_generation"]) < generation:
                            self._connection.execute(
                                "UPDATE expert_shards SET publication_generation=?, "
                                "published_weights_sha=?, published_moments_sha=?, "
                                "published_clock=?, health='ok', updated_at_ns=? "
                                "WHERE expert_id=?",
                                (generation, manifest["weights"]["sha256"],
                                 manifest["moments"]["sha256"], int(manifest["clock"]),
                                 _now_ns(), expert_id))
                            self._connection.execute(
                                "INSERT OR IGNORE INTO expert_publications(expert_id, "
                                "generation, manifest_path, manifest_sha256, "
                                "weights_path, weights_sha, weights_bytes, moments_path, "
                                "moments_sha, moments_bytes, clock, created_at_ns) "
                                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                                (expert_id, generation, self._relative(path), digest,
                                 manifest["weights"]["path"],
                                 manifest["weights"]["sha256"],
                                 int(manifest["weights"]["bytes"]),
                                 manifest["moments"]["path"],
                                 manifest["moments"]["sha256"],
                                 int(manifest["moments"]["bytes"]),
                                 int(manifest["clock"]),
                                 int(manifest["created_at_ns"])))
                            experts_recovered.add(expert_id)
                if rebuild_index:
                    self._rebuild_accounting_locked()
        # A selection/ticket from the old process must not be reused.
        self._selections.clear()
        self._tickets.clear()
        self._ticket_results.clear()
        self._hot_cache.clear()
        status = "incomplete" if (corrupt_manifests or corrupt_artifacts) else "ok"
        reason = ("recovery_incomplete_preserved_corrupt_items" if status == "incomplete"
                  else "recovery_scan_complete")
        return RecoveryReport(
            outcome=_outcome(status, reason,
                             corrupt_preserved=True,
                             native_capability=False),
            clusters_recovered=tuple(sorted(clusters_recovered)),
            generations_recovered=generations_recovered,
            records_recovered=records_recovered,
            experts_recovered=tuple(sorted(experts_recovered)),
            corrupt_manifests=tuple(corrupt_manifests),
            corrupt_artifacts=tuple(corrupt_artifacts),
            ignored_temporaries=tuple(sorted(ignored)),
            accounting_rebuilt=rebuild_index,
            notes=tuple(notes),
        )

    def _rebuild_record_row(self, cluster_id: str, entry: Mapping) -> None:
        text = entry["text"]
        key = entry.get("key")
        values = tuple(entry.get("values", ()))
        existing = self._connection.execute(
            "SELECT status, revision FROM records WHERE record_id=?",
            (entry["record_id"],)).fetchone()
        if existing is None:
            self._connection.execute(
                "INSERT INTO records(record_id, cluster_id, record_index, record_hash, "
                "scope_key, kind, status, superseded_by, revoked_reason, text_path, "
                "text_sha256, text_bytes, source_id, source_epoch, source_lineage, "
                "authenticity, verification_state, observed_at, event_at, valid_from, "
                "valid_to, importance, critical, placement, revision, "
                "representation_version, created_at_ns, updated_at_ns) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, "
                "?, ?, ?, ?, ?, ?, ?, ?)",
                (entry["record_id"], cluster_id, int(entry["record_index"]),
                 entry["record_hash"], self._cluster_row(cluster_id)["scope_key"],
                 entry["kind"], entry["status"], entry.get("superseded_by"),
                 entry.get("revoked_reason"), text["path"], text["sha256"],
                 int(text["bytes"]), entry["source_id"], int(entry["source_epoch"]),
                 _canonical(entry.get("source_lineage", [])), entry["authenticity"],
                 entry["verification_state"], int(entry["observed_at"]),
                 int(entry["event_at"]), entry.get("valid_from"),
                 entry.get("valid_to"), int(entry["importance"]),
                 int(bool(entry.get("critical"))), entry["placement"],
                 int(entry["revision"]), entry["representation_version"],
                 int(entry.get("created_at_ns", 0)), _now_ns()))
        else:
            self._connection.execute(
                "UPDATE records SET cluster_id=?, record_index=?, record_hash=?, "
                "kind=?, status=?, superseded_by=?, revoked_reason=?, text_path=?, "
                "text_sha256=?, text_bytes=?, source_id=?, source_epoch=?, "
                "source_lineage=?, authenticity=?, verification_state=?, "
                "observed_at=?, event_at=?, valid_from=?, valid_to=?, importance=?, "
                "critical=?, placement=?, revision=?, representation_version=?, "
                "updated_at_ns=? WHERE record_id=?",
                (cluster_id, int(entry["record_index"]), entry["record_hash"],
                 entry["kind"], entry["status"], entry.get("superseded_by"),
                 entry.get("revoked_reason"), text["path"], text["sha256"],
                 int(text["bytes"]), entry["source_id"], int(entry["source_epoch"]),
                 _canonical(entry.get("source_lineage", [])), entry["authenticity"],
                 entry["verification_state"], int(entry["observed_at"]),
                 int(entry["event_at"]), entry.get("valid_from"),
                 entry.get("valid_to"), int(entry["importance"]),
                 int(bool(entry.get("critical"))), entry["placement"],
                 int(entry["revision"]), entry["representation_version"],
                 _now_ns(), entry["record_id"]))
        if key is not None:
            self._connection.execute(
                "INSERT INTO record_keys(record_id, key_path, key_bytes, key_sha256, "
                "key_width, producer_id, producer_version, status, updated_at_ns) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(record_id) DO UPDATE SET key_path=excluded.key_path, "
                "key_bytes=excluded.key_bytes, key_sha256=excluded.key_sha256, "
                "key_width=excluded.key_width, producer_id=excluded.producer_id, "
                "producer_version=excluded.producer_version, status=excluded.status, "
                "updated_at_ns=excluded.updated_at_ns",
                (entry["record_id"], key["path"], int(key["bytes"]), key["sha256"],
                 int(key["width"]), key["producer_id"], key["producer_version"],
                 key["status"], _now_ns()))
        for value in values:
            self._connection.execute(
                "INSERT INTO record_values(record_id, slot_index, value_path, "
                "value_bytes, value_sha256, value_width, dtype, producer_id, "
                "producer_version, status, updated_at_ns) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(record_id, slot_index) DO UPDATE SET "
                "value_path=excluded.value_path, value_bytes=excluded.value_bytes, "
                "value_sha256=excluded.value_sha256, value_width=excluded.value_width, "
                "dtype=excluded.dtype, producer_id=excluded.producer_id, "
                "producer_version=excluded.producer_version, status=excluded.status, "
                "updated_at_ns=excluded.updated_at_ns",
                (entry["record_id"], int(value["slot"]), value["path"],
                 int(value["bytes"]), value["sha256"], int(value["width"]),
                 value["dtype"], value["producer_id"], value["producer_version"],
                 value["status"], _now_ns()))
        for span in entry.get("spans", ()):
            self._connection.execute(
                "INSERT OR IGNORE INTO record_spans(span_id, record_id, "
                "start_offset, end_offset, span_sha256, text_sha256, created_at_ns) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (span["span_id"], entry["record_id"], int(span["start"]),
                 int(span["end"]), span["sha256"], span["text_sha256"], _now_ns()))

    def _rebuild_accounting_locked(self) -> None:
        self._connection.execute("DELETE FROM tier_usage")
        for path in sorted(self.root.rglob("*")):
            if not path.is_file():
                continue
            relative = self._relative(path)
            parts = relative.split("/")
            if path.name.startswith(".tmp-"):
                continue
            size = path.stat().st_size
            if parts[0] == "storage" and len(parts) >= 3:
                tier = parts[1]
                if tier not in ("hot", "warm", "cold"):
                    continue
                if parts[2] == "records" and len(parts) >= 5:
                    record_id = parts[3]
                    name = parts[4]
                    if name == "text.bin":
                        self._set_component_locked(
                            f"record:{record_id}:text", tier, "canonical_text", size,
                            "recovered canonical text")
                    elif name.startswith("slot-") and name.endswith(".f32"):
                        slot = name[5:-4]
                        self._set_component_locked(
                            f"record:{record_id}:value:{slot}", tier, "values", size,
                            "recovered ordered value")
                elif parts[2] == "keys":
                    record_id = path.stem
                    self._set_component_locked(
                        f"record:{record_id}:key", tier, "keys", size,
                        "recovered retrieval key")
                elif parts[2] == "experts":
                    expert_id = parts[3]
                    if tier == "hot":
                        self._connection.execute(
                            "INSERT INTO tier_usage(component_id, tier, category, bytes, "
                            "detail, updated_at_ns) VALUES (?, ?, 'expert_active', ?, ?, ?) "
                            "ON CONFLICT(component_id) DO UPDATE SET bytes=bytes+?",
                            (f"expert:{expert_id}:hot", tier, size,
                             "recovered active expert", _now_ns(), size))
                    else:
                        generation = parts[4] if len(parts) > 4 else "unknown"
                        self._connection.execute(
                            "INSERT INTO tier_usage(component_id, tier, category, bytes, "
                            "detail, updated_at_ns) VALUES (?, ?, 'expert_state', ?, ?, ?) "
                            "ON CONFLICT(component_id) DO UPDATE SET bytes=bytes+?",
                            (f"expert:{expert_id}:cold:{generation}", tier, size,
                             "recovered expert publication", _now_ns(), size))
                elif parts[2] == "identity":
                    self._connection.execute(
                        "INSERT INTO tier_usage(component_id, tier, category, bytes, "
                        "detail, updated_at_ns) VALUES (?, ?, 'identity', ?, ?, ?) "
                        "ON CONFLICT(component_id) DO UPDATE SET bytes=bytes+?",
                        (f"identity:{parts[3]}:{Path(parts[4]).stem}", tier, size,
                         "recovered identity core", _now_ns(), size))
                elif parts[2] == "workspace":
                    task = Path(parts[3]).stem
                    self._connection.execute(
                        "INSERT INTO tier_usage(component_id, tier, category, bytes, "
                        "detail, updated_at_ns) VALUES (?, ?, 'workspace', ?, ?, ?) "
                        "ON CONFLICT(component_id) DO UPDATE SET bytes=bytes+?",
                        (f"workspace:{task}", tier, size,
                         "recovered task snapshot", _now_ns(), size))
                elif parts[2] == "replay":
                    episode = Path(parts[3]).stem
                    self._connection.execute(
                        "INSERT INTO tier_usage(component_id, tier, category, bytes, "
                        "detail, updated_at_ns) VALUES (?, ?, 'replay', ?, ?, ?) "
                        "ON CONFLICT(component_id) DO UPDATE SET bytes=bytes+?",
                        (f"replay:{episode}", tier, size,
                         "recovered replay", _now_ns(), size))
                elif parts[2] == "checkpoints":
                    checkpoint_id = parts[3]
                    generation = parts[4] if len(parts) > 4 else "unknown"
                    self._connection.execute(
                        "INSERT INTO tier_usage(component_id, tier, category, bytes, "
                        "detail, updated_at_ns) VALUES (?, ?, 'checkpoint', ?, ?, ?) "
                        "ON CONFLICT(component_id) DO UPDATE SET bytes=bytes+?",
                        (f"checkpoint:{checkpoint_id}:{generation}", tier, size,
                         "recovered checkpoint", _now_ns(), size))
            elif parts[0] == "manifests" and len(parts) >= 4:
                if parts[1] == "clusters":
                    component = f"manifest:cluster:{parts[2]}:{Path(parts[3]).stem}"
                elif parts[1] == "experts":
                    component = f"manifest:expert:{parts[2]}:{Path(parts[3]).stem}"
                else:
                    component = f"manifest:checkpoint:{parts[2]}:{Path(parts[3]).stem}"
                self._set_component_locked(component, "cold", "generations", size,
                                           "recovered immutable manifest")
            elif parts[0] == "journal":
                self._connection.execute(
                    "INSERT INTO tier_usage(component_id, tier, category, bytes, detail, "
                    "updated_at_ns) VALUES ('journal:file', 'cold', 'journal', ?, ?, ?) "
                    "ON CONFLICT(component_id) DO UPDATE SET bytes=bytes+?",
                    (size, "recovered journal", _now_ns(), size))
            elif parts[0] == "artifacts":
                sha = path.stem if _SHA_RE.fullmatch(path.stem) else _sha256_bytes(
                    relative.encode("utf-8"))
                self._set_component_locked(
                    f"artifact:{sha}", "cold", "artifacts", size,
                    "recovered bounded artifact")
            elif parts[0] == "workspace":
                self._connection.execute(
                    "INSERT INTO tier_usage(component_id, tier, category, bytes, detail, "
                    "updated_at_ns) VALUES (?, 'hot', 'workspace', ?, ?, ?) "
                    "ON CONFLICT(component_id) DO UPDATE SET bytes=bytes+?",
                    (f"workspace:{Path(parts[1]).stem}", size,
                     "recovered task snapshot", _now_ns(), size))
            elif parts[0] == "staging":
                self._connection.execute(
                    "INSERT INTO tier_usage(component_id, tier, category, bytes, detail, "
                    "updated_at_ns) VALUES ('publication_overlap:staging', 'cold', "
                    "'publication_overlap', ?, ?, ?) "
                    "ON CONFLICT(component_id) DO UPDATE SET bytes=bytes+?",
                    (size, "recovered publication overlap", _now_ns(), size))
        for suffix, component in ((self.sqlite_path, "database:sqlite"),
                                  (Path(str(self.sqlite_path) + "-wal"), "database:wal"),
                                  (Path(str(self.sqlite_path) + "-shm"), "database:shm")):
            self._set_component_locked(component, "cold", "database",
                                       _safe_file_bytes(suffix),
                                       "measured SQLite file")

    def state_summary(self) -> dict:
        counts = {}
        for table, key in (
                ("clusters", "E_clusters"), ("records", "E_records"),
                ("record_keys", "E_retrieval_keys"), ("record_values", "E_value_slots"),
                ("procedures", "S_procedure_versions"), ("expert_shards", "P_experts"),
                ("expert_publications", "P_publications"), ("phi_releases", "phi_releases"),
                ("workspace_snapshots", "C_workspace_snapshots"),
                ("identity_versions", "I_identity_versions"),
                ("capability_records", "K_capability_records")):
            counts[key] = int(self._connection.execute(
                f"SELECT COUNT(*) AS value FROM {table}").fetchone()["value"])
        procedures = self.procedures()
        counts["S_verified"] = sum(1 for item in procedures if item["outcome"] == "verified")
        counts["S_failed_preserved"] = sum(
            1 for item in procedures if item["outcome"] == "failed")
        counts["P_hot_active"] = len(self._active_expert_ids())
        counts["P_quarantined"] = int(self._connection.execute(
            "SELECT COUNT(*) AS value FROM expert_shards "
            "WHERE quarantine_status='quarantined'").fetchone()["value"])
        counts["I_approved_current"] = int(self._connection.execute(
            "SELECT COUNT(*) AS value FROM identity_versions "
            "WHERE status='current' AND approved=1").fetchone()["value"])
        counts["K_measured"] = int(self._connection.execute(
            "SELECT COUNT(*) AS value FROM capability_records "
            "WHERE measured=1").fetchone()["value"])
        return {
            "format": "replica.tiered_memory.state_summary.v1",
            "state_classes": STATE_CLASSES,
            "counts": counts,
            "values_hold_identity_workspace_or_p": False,
            "quotas": STATE_QUOTAS.document(),
            "usage": self.usage(),
            "engineering_evidence": True,
            "native_capability": False,
        }

    # -- shutdown ----------------------------------------------------------

    def close(self) -> None:
        if self._closed:
            return
        for selection_id in list(self._selections):
            self.release_selection(selection_id)
        for record_id in list(self._hot_cache):
            self._hot_cache.pop(record_id, None)
            self._release(f"cache:{record_id}")
        # Ephemeral workspace snapshots are explicitly non-resumable; remove
        # them on a clean close.  Resumable snapshots remain.
        for row in list(self._connection.execute(
                "SELECT task_id, path FROM workspace_snapshots WHERE resumable=0")):
            try:
                (self.root / row["path"]).unlink()
            except FileNotFoundError:
                pass
            self._release(f"workspace:{row['task_id']}")
            self._connection.execute(
                "DELETE FROM workspace_snapshots WHERE task_id=?", (row["task_id"],))
        try:
            self._connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        except sqlite3.Error:
            pass
        self._connection.close()
        self._closed = True

"""Deterministic CPU integration runner for the value-readout protocol.

This module is deliberately small and local.  It does not load MiniCPM5, a
tokenizer, a native runtime or any downloaded asset.  It implements the declared
22-class selected-expert readout over fixed, non-trainable projections, one
event clock per independent arm, three separately trained arms with exactly 24
optimizer updates each, and the seven declared control views.

The synthetic feature encoder carries the exact-address selected record value
into a fixed feature block.  That makes the CPU integration meaningful for
transaction, accounting, checkpoint, interruption, resume and admission tests,
but it is *not* evidence of learned retrieval, open-ended generation or general
persistent learning.  Those claims remain excluded.
"""
from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from replica_lab import artifact_publication as publication
from replica_lab.nested_learning import DedupLedger, TeachingClock, TransactionOutcome
from replica_lab.optimizers import NativeFusedAdamW

READOUT_CHECKPOINT_FORMAT = "replica.value_readout_abstention.checkpoint.v1"
FIT_REPORT_FORMAT = "replica.value_readout_abstention.fit_report.v1"
PREDICTION_FORMAT = "replica.value_readout_abstention.prediction.v1"
EVALUATION_REPORT_FORMAT = "replica.value_readout_abstention.evaluation_report.v1"
CONTROL_REPORT_FORMAT = "replica.value_readout_abstention.control_report.v1"
RUNNER_MANIFEST_FORMAT = "replica.value_readout_abstention.run_manifest.v1"
RESUME_FORMAT = "replica.value_readout_abstention.resume.v2"
RESUME_ARM_FORMAT = "replica.value_readout_abstention.resume_arm.v2"
RESUME_EVIDENCE_FORMAT = "replica.value_readout_abstention.resume_evidence.v1"
ARTIFACT_BUDGET_FORMAT = "replica.value_readout_abstention.artifact_budget.v1"
ARTIFACT_OVERFLOW_FORMAT = "replica.value_readout_abstention.artifact_budget_overflow.v1"
RUNNER_ENTRY_POINT = "scripts/run_value_readout_abstention_v1.py"

TRAINABLE_NAMES = (
    "selected_expert.gate",
    "selected_expert.up",
    "selected_expert.down",
)
FIXED_NAMES = ("read_projection", "decoder")
ARM_IDS = (
    "learned_value_readout",
    "exact_context",
    "selected_record_literal_readout",
    "memory_disabled",
    "memory_reset",
    "shuffled_values",
    "wrong_world_values",
)
TRAINED_ARMS = ("learned_value_readout", "shuffled_values", "wrong_world_values")
TREATMENT_ARM = "learned_value_readout"
WIDTH = 256
HIDDEN_WIDTH = 1024
VOCAB_SIZE = 22
EXPERT_SCALE = 0.01


class ReadoutError(ValueError):
    """One runner, artifact or accounting invariant failed."""


class RunInterrupted(RuntimeError):
    """A deliberate interruption after a durably saved resume checkpoint."""

    def __init__(self, completed_updates: int, resume_path: Path):
        super().__init__(
            f"interrupted after {completed_updates} updates; resume at {resume_path}")
        self.completed_updates = completed_updates
        self.resume_path = Path(resume_path)


class ArtifactBudgetExceeded(ReadoutError):
    """The complete post-write artifact tree exceeded the declared ceiling."""

    def __init__(self, measured_bytes: int, cap_bytes: int,
                 overflow_path: Path):
        super().__init__(
            f"artifact budget exceeded: {measured_bytes} > {cap_bytes}; "
            f"overflow evidence preserved at {overflow_path}")
        self.measured_bytes = int(measured_bytes)
        self.cap_bytes = int(cap_bytes)
        self.overflow_path = Path(overflow_path)


def _require(condition: bool, detail: str) -> None:
    if not condition:
        raise ReadoutError(detail)


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_json_atomic(path: Path, value: Mapping[str, Any], *,
                       sort_keys: bool = True,
                       meter: publication.LiveByteMeter | None = None
                       ) -> publication.Artifact:
    path = Path(path)
    return publication.write_json(path.parent, path.name, value,
                                  sort_keys=sort_keys, meter=meter)


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]], *,
                 meter: publication.LiveByteMeter | None = None
                 ) -> publication.Artifact:
    path = Path(path)
    return publication.write_jsonl(path.parent, path.name, rows, meter=meter)


def _artifact_inventory(root: Path, *, exclude: Iterable[Path] = ()) -> list[dict]:
    return publication.inventory(Path(root), exclude=exclude)


def _artifact_total(root: Path) -> int:
    return publication.total_bytes(Path(root))


def _render_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _resume_artifact_rows(files: Sequence[Mapping[str, Any]]) -> list[dict]:
    return [row for row in files
            if row["path"] == "resume.pt"
            or row["path"].startswith("resume_arms/")
            or row["path"] == "resume_evidence.json"]


def _merge_release_rows(existing: Iterable[Mapping[str, Any]],
                        extra: Iterable[Mapping[str, Any]]) -> list[dict]:
    """Path-keyed merge that keeps the first recorded row for an artifact."""

    merged: "OrderedDict[str, dict]" = OrderedDict()
    for row in existing:
        merged.setdefault(str(row["path"]), dict(row))
    for row in extra:
        merged.setdefault(str(row["path"]), dict(row))
    return list(merged.values())


def enforce_artifact_budget(output: Path, config: Mapping[str, Any], *,
                            context: str = "run",
                            live_peak: Mapping[str, Any] | None = None,
                            release_after_measurement: Sequence[Path] = ()
                            ) -> dict:
    """Measure the complete artifact tree after all final writes.

    The complete tree is measured *while every retained recovery artifact is
    still present*, so retention is never made to look smaller by deleting
    evidence first.  The budget report is itself included in the measured total,
    and any temporary file still on disk at measurement time is counted.

    ``release_after_measurement`` lists artifacts that become redundant only
    after that measurement succeeds (``resume.pt`` once every published stage is
    durable).  They are released afterwards, their content is recorded, and the
    tree is measured again so ``measured_bytes`` still equals the real tree.

    Overflow never deletes evidence: it writes the overflow receipt, keeps the
    recovery artifacts and raises.
    """

    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    ceiling = config["budget"]["authorized_run_ceiling"]
    cap = int(ceiling["artifact_bytes_max"])
    declared = dict(ceiling.get("artifact_budget", {}))
    summary_path = root / "artifact_budget.json"
    overflow_path = root / "artifact_budget_overflow.json"
    previous: dict = {}
    if summary_path.is_file():
        try:
            previous = read_json(summary_path)
        except (OSError, ValueError):
            previous = {}
    peak = publication.peak_union(live_peak, previous.get("live_peak"))
    targets = [Path(path).resolve() for path in release_after_measurement]
    relative_targets: list[str] = []
    planned_release: list[dict] = []
    for path in targets:
        if not path.is_file():
            continue
        relative = path.relative_to(root.resolve()).as_posix()
        relative_targets.append(relative)
        target = root / relative
        planned_release.append({"path": relative,
                                "sha256": publication.sha256_path(target),
                                "bytes": target.stat().st_size})
    released = _merge_release_rows(
        [dict(row) for row in previous.get("released_recovery_artifacts", [])],
        planned_release)
    enforcement = "after_final_report_and_receipt_writes_all_artifacts_included"

    def _report(*, files: Sequence[Mapping[str, Any]], measured: int,
                retained_before_release: int | None) -> dict:
        resume_rows = _resume_artifact_rows(files)
        return {
            "format": ARTIFACT_BUDGET_FORMAT,
            "protocol_id": "value_readout_abstention_v1",
            "context": context,
            "status": "within_budget",
            "cap_bytes": cap,
            "old_cap_bytes": declared.get("old_cap_bytes"),
            "old_cap_underprovisioned": declared.get("old_cap_underprovisioned"),
            "resume_complete_checkpoints": declared.get(
                "resume_complete_checkpoints"),
            "declared_derivation": declared.get("derivation"),
            "enforcement": enforcement,
            "enforcement_detail": (
                "measured_with_recovery_and_temporary_artifacts_present"),
            "resume_artifacts_included": True,
            "resume_artifact_bytes": sum(row["bytes"] for row in resume_rows),
            "resume_artifact_paths": [row["path"] for row in resume_rows],
            "recovery_state_measured_before_release": bool(relative_targets),
            "retained_bytes_before_release": retained_before_release,
            "released_recovery_artifacts": released,
            "live_peak": (None if peak is None else {
                **dict(peak),
                "cap_bytes": cap,
                "peak_within_cap": int(peak.get("peak_live_bytes", 0)) <= cap,
                "note": ("peak counts temporary files and the overlap between "
                         "retained recovery artifacts and their durable "
                         "replacement; the declared ceiling applies to retained "
                         "artifact trees and to the retained recovery state"),
            }),
            "measured_bytes": measured,
            "files_total_bytes": sum(row["bytes"] for row in files),
            "file_count": len(files) + 1,
            "files": list(files),
        }

    files = _artifact_inventory(root, exclude=(summary_path, overflow_path))
    files_total = sum(row["bytes"] for row in files)
    measured = files_total
    for _ in range(8):
        report = _report(files=files, measured=measured,
                         retained_before_release=None)
        text = _render_json(report)
        new_total = files_total + len(text.encode("utf-8"))
        if new_total == measured:
            break
        measured = new_total
    retained_before_release = measured
    if measured > cap:
        report["status"] = "overflow"
        report["measured_bytes"] = measured
        _write_json_atomic(summary_path, report, sort_keys=False)
        overflow_files = _artifact_inventory(root, exclude=(overflow_path,))
        overflow = {
            "format": ARTIFACT_OVERFLOW_FORMAT,
            "protocol_id": "value_readout_abstention_v1",
            "context": context,
            "status": "overflow",
            "cap_bytes": cap,
            "measured_bytes_before_overflow_evidence": measured,
            "old_cap_bytes": declared.get("old_cap_bytes"),
            "old_cap_underprovisioned": declared.get("old_cap_underprovisioned"),
            "declared_derivation": declared.get("derivation"),
            "summary_path": summary_path.relative_to(root).as_posix(),
            "files": overflow_files,
            "resume_artifacts_within_measurement": [
                row["path"] for row in _resume_artifact_rows(overflow_files)],
            "live_peak": report["live_peak"],
            "preserved": True,
            "measurement_after_final_report_and_receipt_writes": True,
            "deleted_artifacts": [],
        }
        publication.write_overflow_evidence(
            root, context=context, cap_bytes=cap, measured_bytes=measured,
            files=overflow_files, extra=overflow,
            name=overflow_path.name)
        raise ArtifactBudgetExceeded(measured, cap, overflow_path)
    # The complete tree, including every retained recovery artifact, is within
    # budget.  Only now may verified-redundant recovery state be released.
    _write_json_atomic(summary_path, report, sort_keys=False)
    targets = [Path(path).resolve() for path in release_after_measurement]
    relative_targets = []
    for path in targets:
        if path.is_file():
            relative_targets.append(path.relative_to(root.resolve()).as_posix())
    released.extend(publication.release(root, relative_targets))
    files = _artifact_inventory(root, exclude=(summary_path, overflow_path))
    files_total = sum(row["bytes"] for row in files)
    measured = files_total
    for _ in range(8):
        report = _report(files=files, measured=measured,
                         retained_before_release=retained_before_release)
        text = _render_json(report)
        new_total = files_total + len(text.encode("utf-8"))
        if new_total == measured:
            break
        measured = new_total
    _write_json_atomic(summary_path, report, sort_keys=False)
    actual = _artifact_total(root)
    for _ in range(8):
        if actual == measured:
            break
        measured = actual
        report["measured_bytes"] = measured
        _write_json_atomic(summary_path, report, sort_keys=False)
        actual = _artifact_total(root)
    if actual > cap:
        report["status"] = "overflow"
        report["measured_bytes"] = actual
        _write_json_atomic(summary_path, report, sort_keys=False)
        overflow_files = _artifact_inventory(root, exclude=(overflow_path,))
        publication.write_overflow_evidence(
            root, context=context, cap_bytes=cap, measured_bytes=actual,
            files=overflow_files,
            extra={
                "format": ARTIFACT_OVERFLOW_FORMAT,
                "protocol_id": "value_readout_abstention_v1",
                "summary_path": summary_path.relative_to(root).as_posix(),
                "resume_artifacts_within_measurement": [
                    row["path"] for row in _resume_artifact_rows(overflow_files)],
                "live_peak": report["live_peak"],
                "measurement_after_final_report_and_receipt_writes": True,
            },
            name=overflow_path.name)
        raise ArtifactBudgetExceeded(actual, cap, overflow_path)
    report["status"] = "within_budget"
    report["measured_bytes"] = actual
    _write_json_atomic(summary_path, report, sort_keys=False)
    return report


def _canonical_payload(value: Any) -> Any:
    """Rebuild the object graph deterministically before serialization.

    Torch's pickle memoization depends on object identity.  Fresh/restored runs
    otherwise share different strings, which changes checkpoint bytes even when
    the state is identical.  Rebuilding every metadata object (and cloning every
    tensor) gives the same pickle graph for an identical state.
    """
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, Mapping):
        return {str(key).encode("utf-8").decode("utf-8"):
                _canonical_payload(item)
                for key, item in sorted(value.items(),
                                        key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple)):
        return [_canonical_payload(item) for item in value]
    if isinstance(value, str):
        return value.encode("utf-8").decode("utf-8")
    if isinstance(value, (bool, int, float)) or value is None:
        return value
    return value


def _save_torch_atomic(path: Path, payload: Mapping[str, Any], *,
                       meter: publication.LiveByteMeter | None = None
                       ) -> publication.Artifact:
    """Publish a torch artifact atomically, keeping an identical payload.

    ``publication.write_torch`` uses the same deterministic ``.<name>.tmp``
    temporary name, so an unchanged checkpoint is verified by content identity
    and not rewritten.
    """
    path = Path(path)
    return publication.write_torch(path.parent, path.name,
                                   _canonical_payload(dict(payload)),
                                   meter=meter, reuse_identical=False)


def tensor_sha256(name: str, tensor: Tensor) -> str:
    value = tensor.detach().to("cpu", torch.float32).contiguous()
    digest = hashlib.sha256()
    digest.update(name.encode("utf-8"))
    digest.update(str(tuple(value.shape)).encode("ascii"))
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def state_sha256(state: Mapping[str, Tensor]) -> str:
    rows = {name: tensor_sha256(name, tensor)
            for name, tensor in state.items()}
    return sha256_json(rows)


def state_tensor_digests(state: Mapping[str, Tensor]) -> dict[str, str]:
    return {name: tensor_sha256(name, tensor) for name, tensor in state.items()}


def _state_clone(state: Mapping[str, Tensor]) -> "OrderedDict[str, Tensor]":
    return OrderedDict((name, tensor.detach().clone())
                       for name, tensor in state.items())


# Physical work counters.  A fresh process always starts at zero; resume
# evidence accumulates one record per process so a restart cannot hide repeated
# training behind a final logical update count.
WORK_COUNTERS = (
    "optimizer_steps",
    "backward_calls",
    "training_forward_calls",
    "evaluation_forward_calls",
    "retention_diagnostic_forwards",
    "training_write_events",
    "evaluation_write_events",
    "checkpoint_writes",
    "resume_artifact_writes",
)


def _empty_work() -> dict[str, int]:
    return {name: 0 for name in WORK_COUNTERS}


def _work_copy(work: Mapping[str, Any]) -> dict[str, int]:
    return {name: int(work.get(name, 0)) for name in WORK_COUNTERS}


def _work_add(into: dict[str, int], other: Mapping[str, Any]) -> None:
    for name in WORK_COUNTERS:
        into[name] = int(into.get(name, 0)) + int(other.get(name, 0))


def _sum_work(records: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    total = _empty_work()
    for record in records:
        _work_add(total, record.get("work", {}))
    return total


def _resume_arm_files(output: Path) -> list[Path]:
    arms_dir = Path(output) / "resume_arms"
    if not arms_dir.is_dir():
        return []
    return sorted(path for path in arms_dir.rglob("*")
                  if path.is_file() and not path.name.startswith("."))


def _resume_artifact_files(output: Path) -> list[Path]:
    root = Path(output)
    paths = []
    resume_path = root / "resume.pt"
    if resume_path.is_file():
        paths.append(resume_path)
    paths.extend(_resume_arm_files(root))
    return paths


def _measure_resume_state(output: Path) -> dict:
    """Measure the retained resume state without enforcing the ceiling.

    Used by the publication transition, where verified final checkpoints and
    their not-yet-released duplicates legitimately coexist for a moment; that
    overlap is reported by the live meter rather than by a retained-tree cap.
    """

    root = Path(output)
    files = _resume_artifact_files(root)
    return {"resume_artifact_bytes": sum(path.stat().st_size for path in files),
            "resume_artifact_paths": [path.relative_to(root).as_posix()
                                      for path in files]}


def enforce_resume_artifact_budget(output: Path, config: Mapping[str, Any], *,
                                   context: str = "resume") -> dict:
    """Include live resume state in the unchanged artifact ceiling.

    Completed-arm snapshots are large (full model and optimizer state), so the
    in-flight resume tree is measured after every durable save.  Overflow keeps
    every resume artifact and writes separate evidence instead of deleting or
    silently raising the declared 32 MiB ceiling.
    """
    root = Path(output)
    cap = int(config["budget"]["authorized_run_ceiling"]["artifact_bytes_max"])
    files = _artifact_inventory(root)
    measured = sum(row["bytes"] for row in files)
    resume_files = _resume_artifact_files(root)
    if measured > cap:
        overflow_path = root / "artifact_budget_overflow.json"
        overflow = {
            "format": ARTIFACT_OVERFLOW_FORMAT,
            "protocol_id": "value_readout_abstention_v1",
            "context": context,
            "status": "overflow",
            "cap_bytes": cap,
            "measured_bytes_before_overflow_evidence": measured,
            "resume_artifacts_within_measurement": [
                path.relative_to(root).as_posix() for path in resume_files],
            "summary_path": None,
            "files": files,
            "preserved": True,
            "deleted_artifacts": [],
        }
        _write_json_atomic(overflow_path, overflow, sort_keys=False)
        raise ArtifactBudgetExceeded(measured, cap, overflow_path)
    return {"cap_bytes": cap, "measured_bytes": measured,
            "resume_artifact_bytes": sum(path.stat().st_size
                                         for path in resume_files),
            "resume_artifact_paths": [path.relative_to(root).as_posix()
                                      for path in resume_files]}


# ---------------------------------------------------------------------------
# Fixed addressing and the declared selected-expert readout
# ---------------------------------------------------------------------------


class SelectedExpertReadout(nn.Module):
    """Selected expert only: gate/up/down are trainable, the rest is fixed."""

    def __init__(self, *, seed: int, width: int = WIDTH,
                 hidden_width: int = HIDDEN_WIDTH,
                 vocabulary_size: int = VOCAB_SIZE,
                 expert_scale: float = EXPERT_SCALE):
        super().__init__()
        self.width = width
        self.hidden_width = hidden_width
        self.vocabulary_size = vocabulary_size
        self.expert_scale = float(expert_scale)
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        self.gate = nn.Linear(width, hidden_width, bias=False)
        self.up = nn.Linear(width, hidden_width, bias=False)
        self.down = nn.Linear(hidden_width, width, bias=False)
        for matrix in (self.gate.weight, self.up.weight, self.down.weight):
            with torch.no_grad():
                matrix.copy_(torch.randn(
                    matrix.shape, generator=generator, dtype=torch.float32) * 0.02)
        projection = torch.zeros(vocabulary_size, width, dtype=torch.float32)
        for index in range(min(vocabulary_size, width)):
            projection[index, index] = 1.0
        self.register_buffer("read_projection", projection, persistent=True)
        with torch.no_grad():
            decoder = torch.randn(2048, vocabulary_size,
                                  generator=generator, dtype=torch.float32) * 0.02
        self.register_buffer("decoder", decoder, persistent=True)
        for parameter in (self.read_projection, self.decoder):
            parameter.requires_grad_(False)

    def forward(self, features: Tensor) -> Tensor:
        features = features.to(dtype=torch.float32)
        delta = self.down(F.silu(self.gate(features)) * self.up(features))
        slot = features + self.expert_scale * delta
        return F.linear(slot, self.read_projection)

    def model_state(self) -> "OrderedDict[str, Tensor]":
        return OrderedDict((
            ("selected_expert.gate", self.gate.weight),
            ("selected_expert.up", self.up.weight),
            ("selected_expert.down", self.down.weight),
        ))

    def fixed_state(self) -> "OrderedDict[str, Tensor]":
        return OrderedDict((
            ("read_projection", self.read_projection),
            ("decoder", self.decoder),
        ))

    def trainable_parameters(self) -> list[tuple[str, nn.Parameter]]:
        return [
            ("selected_expert.gate", self.gate.weight),
            ("selected_expert.up", self.up.weight),
            ("selected_expert.down", self.down.weight),
        ]

    def state_digest(self) -> str:
        return state_sha256({**self.model_state(), **self.fixed_state()})

    def model_state_digest(self) -> str:
        return state_sha256(self.model_state())

    def fixed_state_digest(self) -> str:
        return state_sha256(self.fixed_state())


def build_parent_state_dict(config: Mapping[str, Any], seed: int) -> OrderedDict:
    """Rebuild the deterministic parent state for independent verification."""
    width = int(config["architecture"]["expert_bank"]["declared_width"])
    hidden = int(config["architecture"]["expert_bank"]["declared_hidden_width"])
    vocabulary = int(config["architecture"]["answer_vocabulary_size"])
    module = SelectedExpertReadout(seed=seed, width=width,
                                   hidden_width=hidden,
                                   vocabulary_size=vocabulary)
    parent = OrderedDict()
    parent.update(module.model_state())
    parent.update(module.fixed_state())
    return parent


def parent_state_sha256(config: Mapping[str, Any], seed: int) -> str:
    return state_sha256(build_parent_state_dict(config, seed))


# ---------------------------------------------------------------------------
# Fixed record addressing, immutable query snapshots and arm views
# ---------------------------------------------------------------------------

_PERMUTATION = (7, 12, 18, 2, 20, 9, 15, 4, 21, 11, 1, 16, 6, 19, 3, 13,
                8, 17, 5, 14, 10, 0)


def shuffled_index(index: int) -> int:
    """A fixed derangement over the closed 22-token vocabulary."""
    return _PERMUTATION[int(index) % VOCAB_SIZE]


def sibling_world(world_id: str) -> str | None:
    suffix = world_id.removeprefix("world_")
    stem = suffix.rsplit("_", 1)[0]
    leaf = suffix.rsplit("_", 1)[-1]
    if leaf == "alpha":
        return f"world_{stem}_beta"
    if leaf == "beta":
        return f"world_{stem}_alpha"
    return None


@dataclass(frozen=True)
class QueryView:
    address: Mapping[str, str]
    world_id: str
    epoch: int

    def key(self) -> tuple[str, str, str, int]:
        return (self.world_id, self.address["entity"],
                self.address["attribute"], int(self.epoch))


class FixedRecordMemory:
    """Append-only chronological record view with epoch-bounded resolution."""

    def __init__(self) -> None:
        self._records: list[dict] = []
        self._superseded: set[str] = set()

    def write(self, event: Mapping[str, Any]) -> None:
        record = dict(event["record"])
        record["available_epoch"] = int(event["epoch"])
        supersedes = record.get("supersedes")
        if supersedes:
            self._superseded.add(str(supersedes))
        self._records.append(record)

    def resolve(self, address: Mapping[str, str], world_id: str,
                epoch: int) -> dict | None:
        wanted = (str(world_id), str(address["entity"]),
                  str(address["attribute"]))
        selected = None
        for record in self._records:
            if record["record_id"] in self._superseded:
                continue
            if int(record["available_epoch"]) > int(epoch):
                continue
            identity = (record["world_id"], record["entity"], record["attribute"])
            if identity != wanted:
                continue
            if selected is None or int(record["available_epoch"]) >= int(
                    selected["available_epoch"]):
                selected = record
        return None if selected is None else dict(selected)

    def snapshot_records(self) -> list[dict]:
        return [dict(record) for record in self._records]

    def restore_records(self, records: Sequence[Mapping[str, Any]]) -> None:
        self._records = [dict(record) for record in records]
        superseded: set[str] = set()
        for record in self._records:
            if record.get("supersedes"):
                superseded.add(str(record["supersedes"]))
        self._superseded = superseded

    def digest(self) -> str:
        return sha256_json(sorted(self.visible_records(), key=lambda row: (
            row["world_id"], row["entity"], row["attribute"],
            int(row["available_epoch"]), row["record_id"])))

    def visible_records(self) -> list[dict]:
        return [dict(record) for record in self._records
                if record["record_id"] not in self._superseded]


def arm_value_index(arm_id: str, *, record: Mapping[str, Any] | None,
                    address: Mapping[str, str], world_id: str, epoch: int,
                    memory: FixedRecordMemory, vocabulary: Sequence[str],
                    wrong_world_id: str | None = None) -> int:
    """The arm-specific readout index; abstention is only arm-correct for treatment."""
    if record is None:
        base = 0
    else:
        base = list(vocabulary).index(record["value"])
    if arm_id == TREATMENT_ARM:
        return base
    if arm_id == "shuffled_values":
        return shuffled_index(base)
    if arm_id == "wrong_world_values":
        sibling = wrong_world_id or sibling_world(world_id)
        other = None if sibling is None else memory.resolve(address, sibling, epoch)
        if other is not None:
            return list(vocabulary).index(other["value"])
        if base == 0:
            return 1 + (int(sha256_json({"case": address, "epoch": epoch})[:2], 16)
                        % (VOCAB_SIZE - 1))
        return 1 + ((base + 3) % (VOCAB_SIZE - 1))
    raise ReadoutError(f"unknown arm view {arm_id!r}")


def encode_features(case_id: str, case: Mapping[str, Any], *,
                    vocabulary: Sequence[str], index: int) -> Tensor:
    features = torch.zeros(1, WIDTH, dtype=torch.float32)
    features[0, int(index)] = 1.0
    # Deterministic query/address block.  The fixed projection has no weights
    # here, so this block can only influence the trainable expert residual.
    seed = hashlib.sha256(
        f"{case_id}|{case['query_text']}|{case['address']['entity']}|"
        f"{case['address']['attribute']}".encode("utf-8")).digest()
    for offset in range(32):
        position = VOCAB_SIZE + offset
        if position >= WIDTH:
            break
        features[0, position] = (seed[offset] / 255.0) * 0.01
    return features


def _read_train_metadata(prepared: Path, config: Mapping[str, Any]) -> dict:
    files = config["prepared"]["files"]
    train = read_jsonl(prepared / files["cases_train"])
    replay = read_jsonl(prepared / files["cases_train_replay"])
    events = [row for row in read_jsonl(prepared / files["episodes"])
              if row["split"] == "train"]
    events.sort(key=lambda row: (row["world_id"], int(row["epoch"])))
    cases = {row["case_id"]: row for row in train + replay}
    teaching = [row for row in train + replay if int(row["optimizer_steps"]) == 1]
    teaching.sort(key=lambda row: (row["world_id"], int(row["epoch"])))
    return {"train": train, "replay": replay, "events": events,
            "cases": cases, "teaching": teaching}


def read_fit_metadata(prepared: Path, config: Mapping[str, Any]) -> dict:
    """Public access to the exact chronological fit input sequence."""
    return _read_train_metadata(Path(prepared), config)


def _read_split_inputs(prepared: Path, config: Mapping[str, Any],
                       split: str) -> dict:
    files = config["prepared"]["files"]
    case_name = files["cases_dev"] if split == "dev" else files["cases_heldout"]
    cases = read_jsonl(prepared / case_name)
    episodes = [row for row in read_jsonl(prepared / files["episodes"])
                if row["split"] == split]
    episodes.sort(key=lambda row: (row["world_id"], int(row["epoch"])))
    return {"cases": cases, "cases_by_id": {row["case_id"]: row for row in cases},
            "events": episodes}


# ---------------------------------------------------------------------------
# Arm factory and transaction accounting
# ---------------------------------------------------------------------------


def build_optimizer(module: SelectedExpertReadout,
                    config: Mapping[str, Any]) -> NativeFusedAdamW:
    hyper = config["transaction"]["optimizer_hyperparameters"]
    parameters = [parameter for _name, parameter in module.trainable_parameters()]
    group = {
        "group_name": "selected_expert",
        "scope": "selected_expert_gate_up_down_only",
        "algorithm": "adamw_fused",
        "fused": True,
        "param_names": list(TRAINABLE_NAMES),
        "params": parameters,
        "lr": float(hyper["lr"]),
        "betas": tuple(hyper["betas"]),
        "eps": float(hyper["eps"]),
        "weight_decay": float(hyper["weight_decay"]),
    }
    return NativeFusedAdamW([group], optimizer_name="adamw_fused")


def _optimizer_meta(module: SelectedExpertReadout,
                    optimizer: NativeFusedAdamW) -> dict:
    group = optimizer.param_groups[0]
    steps = {}
    for name, parameter in module.trainable_parameters():
        state = optimizer.state.get(parameter, {})
        step = state.get("step")
        if isinstance(step, Tensor):
            step = int(step.detach().cpu().item())
        steps[name] = int(step or 0)
    return {
        "name": "NativeFusedAdamW",
        "class": "torch.optim.AdamW",
        "wrapper": "replica_lab.optimizers.NativeFusedAdamW",
        "algorithm": "adamw_fused",
        "fused": True,
        "lr": float(group["lr"]),
        "betas": [float(group["betas"][0]), float(group["betas"][1])],
        "eps": float(group["eps"]),
        "weight_decay": float(group["weight_decay"]),
        "param_names": list(TRAINABLE_NAMES),
        "step_counts": steps,
    }


def _optimizer_state(module: SelectedExpertReadout,
                     optimizer: NativeFusedAdamW) -> dict:
    result = {}
    for name, parameter in module.trainable_parameters():
        state = optimizer.state.get(parameter, {})
        entry = {}
        for key, value in state.items():
            if isinstance(value, Tensor):
                entry[key] = value.detach().to("cpu", torch.float32).clone()
            else:
                entry[key] = value
        result[name] = entry
    return result


def _restore_optimizer(module: SelectedExpertReadout,
                       optimizer: NativeFusedAdamW,
                       state: Mapping[str, Mapping[str, Any]]) -> None:
    by_name = dict(module.trainable_parameters())
    for name, entry in state.items():
        parameter = by_name[name]
        restored = {}
        for key, value in entry.items():
            if isinstance(value, Tensor):
                restored[key] = value.detach().to(
                    device=parameter.device, dtype=torch.float32).clone()
            else:
                restored[key] = value
        optimizer.state[parameter] = restored


def _model_state_from_checkpoint(state: Mapping[str, Tensor]) -> "OrderedDict[str, Tensor]":
    return OrderedDict((name, state[name].detach().clone()) for name in TRAINABLE_NAMES)


class Arm:
    def __init__(self, arm_id: str, config: Mapping[str, Any], seed: int):
        _require(arm_id in TRAINED_ARMS, f"unknown trained arm {arm_id!r}")
        self.arm_id = arm_id
        self.seed = int(seed)
        self.module = SelectedExpertReadout(seed=self.seed)
        self.optimizer = build_optimizer(self.module, config)
        self.parent_state_sha256 = state_sha256(
            {**self.module.model_state(), **self.module.fixed_state()})
        self.parent_trainable_sha256 = state_sha256(self.module.model_state())
        self.clock = TeachingClock(0)
        self.ledger = DedupLedger()
        self.memory = FixedRecordMemory()
        self.update_records: list[dict] = []
        self.event_index = 0
        self.completed_updates = 0

    def started(self) -> bool:
        return self.event_index > 0 or self.completed_updates > 0

    def state_digest(self) -> str:
        return state_sha256({**self.module.model_state(),
                             **self.module.fixed_state()})

    def snapshot(self) -> dict:
        return {
            "arm_id": self.arm_id,
            "seed": self.seed,
            "parent_state_sha256": self.parent_state_sha256,
            "model_state": {name: tensor.detach().clone()
                            for name, tensor in self.module.model_state().items()},
            "optimizer_state": _optimizer_state(self.module, self.optimizer),
            "clock_events": self.clock.eligible_events,
            "event_index": self.event_index,
            "completed_updates": self.completed_updates,
            "update_records": deepcopy(self.update_records),
            "ledger": self.ledger.to_dict(),
            "memory_records": self.memory.snapshot_records(),
            "state_sha256": self.state_digest(),
            "model_state_sha256": self.module.model_state_digest(),
            "fixed_state_sha256": self.module.fixed_state_digest(),
        }

    def restore(self, payload: Mapping[str, Any]) -> None:
        _require(payload["arm_id"] == self.arm_id, "resume arm differs")
        _require(int(payload["seed"]) == self.seed, "resume seed differs")
        _require(payload.get("parent_state_sha256") == self.parent_state_sha256,
                 "resume parent state differs")
        with torch.no_grad():
            for name, tensor in payload["model_state"].items():
                self.module.model_state()[name].copy_(tensor)
        _restore_optimizer(self.module, self.optimizer, payload["optimizer_state"])
        self.clock = TeachingClock(int(payload["clock_events"]))
        self.event_index = int(payload["event_index"])
        self.completed_updates = int(payload["completed_updates"])
        self.update_records = deepcopy(list(payload["update_records"]))
        self.ledger = DedupLedger.from_dict(payload["ledger"])
        self.memory = FixedRecordMemory()
        self.memory.restore_records(payload.get("memory_records", []))
        _require(self.state_digest() == payload.get("state_sha256"),
                 "restored arm state digest differs")
        _require(len(self.update_records) == self.completed_updates,
                 "restored update record count differs")
        if self.update_records:
            _require(self.update_records[-1].get("state_after_sha256")
                     == payload.get("state_sha256"),
                     "restored arm final update hash differs from its state")
        _require(self.clock.eligible_events == self.event_index,
                 "restored arm clock differs from its event cursor")

    def checkpoint_payload(self, *, config: Mapping[str, Any],
                           config_sha256: str,
                           prepared_manifest_sha256: str,
                           events_processed: int,
                           predictions_sha256: str | None = None) -> dict:
        model_state = {name: tensor.detach().clone()
                       for name, tensor in self.module.model_state().items()}
        fixed_state = {name: tensor.detach().clone()
                       for name, tensor in self.module.fixed_state().items()}
        combined = {**model_state, **fixed_state}
        return {
            "format": READOUT_CHECKPOINT_FORMAT,
            "protocol_id": "value_readout_abstention_v1",
            "config_sha256": config_sha256,
            "prepared_manifest_sha256": prepared_manifest_sha256,
            "arm_id": self.arm_id,
            "phase": ("fit_treatment" if self.arm_id == TREATMENT_ARM
                      else "control_fit"),
            "seed": self.seed,
            "completed_steps": self.completed_updates,
            "event_clock_steps": self.clock.eligible_events,
            "events_processed": int(events_processed),
            "trainable_names": list(TRAINABLE_NAMES),
            "fixed_names": list(FIXED_NAMES),
            "model_state": model_state,
            "fixed_state": fixed_state,
            "trainable_state_sha256": state_tensor_digests(model_state),
            "fixed_state_sha256": state_tensor_digests(fixed_state),
            "model_state_sha256": state_sha256(model_state),
            "parent_state_sha256": self.parent_state_sha256,
            "state_sha256": state_sha256(combined),
            "optimizer_meta": _optimizer_meta(self.module, self.optimizer),
            "optimizer_state": _optimizer_state(self.module, self.optimizer),
            "update_records": deepcopy(self.update_records),
            "predictions_sha256": predictions_sha256,
            "classification": {
                "native_model": False,
                "tokenizer": False,
                "downloads": 0,
                "device": "cpu",
                "scope": "deterministic_software_integration",
            },
        }


# ---------------------------------------------------------------------------
# Fit, evaluation and controls
# ---------------------------------------------------------------------------


@dataclass
class RunnerContext:
    config: Mapping[str, Any]
    config_sha256: str
    manifest_sha256: str
    prepared: Path
    config_path: Path


def _context(config_path: Path, prepared: Path) -> RunnerContext:
    config = read_json(config_path)
    manifest = read_json(prepared / "manifest.json")
    return RunnerContext(config=config, config_sha256=sha256_file(config_path),
                         manifest_sha256=sha256_file(prepared / "manifest.json"),
                         prepared=Path(prepared), config_path=Path(config_path))


def _event_case(event: Mapping[str, Any], cases: Mapping[str, Mapping[str, Any]]) -> dict:
    case_id = event["case_id"]
    _require(case_id in cases, f"event {event['event_id']} has no case")
    return dict(cases[case_id])


def _features_for_event(arm_id: str, event: Mapping[str, Any],
                        case: Mapping[str, Any], memory: FixedRecordMemory,
                        vocabulary: Sequence[str]) -> tuple[Tensor, int, dict | None]:
    record = memory.resolve(event["address"], event["world_id"], int(event["epoch"]))
    index = arm_value_index(
        arm_id, record=record, address=event["address"],
        world_id=event["world_id"], epoch=int(event["epoch"]),
        memory=memory, vocabulary=vocabulary,
        wrong_world_id=event.get("wrong_world_id"))
    features = encode_features(case["case_id"], case, vocabulary=vocabulary,
                               index=index)
    return features, index, record


def _resume_identities(context: RunnerContext, metadata: Mapping[str, Any],
                       vocabulary: Sequence[str]) -> dict:
    teaching = [row["case_id"] for row in metadata["teaching"]]
    return {
        "protocol_id": "value_readout_abstention_v1",
        "config_sha256": context.config_sha256,
        "prepared_manifest_sha256": context.manifest_sha256,
        "vocabulary_sha256": sha256_json(list(vocabulary)),
        "seed": int(context.config["transaction"]["seed"]),
        "event_sequence_sha256": sha256_json(metadata["events"]),
        "teaching_sequence_sha256": sha256_json(teaching),
        "case_sequence_sha256": sha256_json(
            [metadata["cases"][case_id] for case_id in sorted(metadata["cases"])]),
    }


def _require_resume_identities(payload: Mapping[str, Any], identities: Mapping[str, Any],
                               where: str) -> None:
    recorded = payload.get("identities")
    _require(isinstance(recorded, Mapping), f"{where} identities are missing")
    _require(dict(recorded) == dict(identities),
             f"{where} config/input identities differ")


def _load_resume(path: Path) -> dict:
    return torch.load(Path(path), map_location="cpu", weights_only=True)


def _arm_is_complete(arm: Arm, metadata: Mapping[str, Any]) -> bool:
    return (arm.event_index >= len(metadata["events"])
            and arm.completed_updates == 24)


class FitResumeState:
    """Durable global arm cursor for repeated fresh-process restarts.

    Semantics:
    - ``arm_order`` is the fixed tuple of trained arms.
    - ``cursor.next_arm_index`` is the first non-complete arm.  Every earlier
      arm is a durable completed-arm snapshot and is never retrained.
    - ``cursor.active_arm_id`` is the one partially trained arm, if any.  Its
      model, NativeFusedAdamW state, event clock, update records, dedup ledger
      and fixed-memory records are persisted after every optimizer step.
    - ``cursor.completed_arm_ids`` is exactly ``arm_order[:next_arm_index]``.
    - ``processes`` retains one physical-work record per process.  Cumulative
      work is their sum, so retraining completed arms cannot be hidden by a
      final logical count.
    - Heavy training snapshots (``resume_arms/<arm_id>.pt``) are released only
      after the matching final checkpoint is durable and verified by content
      identity, after which the same recovery is served by the final
      checkpoints recorded in ``resume.pt``.  Nothing heavy is released before
      that replacement exists, and ``resume.pt`` itself is released only after
      every published stage and the complete live-byte measurement succeeded.
    """

    def __init__(self, *, output: Path, context: RunnerContext,
                 metadata: Mapping[str, Any], vocabulary: Sequence[str],
                 arms: Mapping[str, Arm], resume_path: Path,
                 interrupt_after: int | None):
        self.output = Path(output)
        self.context = context
        self.metadata = metadata
        self.vocabulary = list(vocabulary)
        self.arms = dict(arms)
        self.arm_order = tuple(TRAINED_ARMS)
        self.resume_path = Path(resume_path)
        self.interrupt_after = interrupt_after
        self.identities = _resume_identities(context, metadata, vocabulary)
        self.processes: list[dict] = []
        self.completed_artifacts: dict[str, dict] = {}
        self.peak_resume_bytes = 0
        self.restored_completed_arms: list[str] = []
        self.recovery_mode = "training_snapshots"
        self.publication: dict | None = None
        self.publication_live_peak: dict | None = None
        self.released_recovery: list[dict] = []

    # -- cursor -----------------------------------------------------------
    def completed_ids(self) -> list[str]:
        completed = []
        for arm_id in self.arm_order:
            if _arm_is_complete(self.arms[arm_id], self.metadata):
                completed.append(arm_id)
            else:
                break
        return completed

    def next_arm_index(self) -> int:
        return len(self.completed_ids())

    def total_updates(self) -> int:
        return sum(int(arm.completed_updates)
                   for arm in self.arms.values() if arm.started())

    def current_process(self) -> dict:
        _require(bool(self.processes), "resume process history is empty")
        return self.processes[-1]

    def process_work(self) -> dict:
        return self.current_process()["work"]

    def add_process(self, *, resumed_from: Mapping[str, Any]) -> None:
        index = len(self.processes)
        self.processes.append({
            "process_index": index,
            "restart_index": index,
            "started_from": dict(resumed_from),
            "status": "running",
            "work": _empty_work(),
            "optimizer_steps_at_start": self.total_updates(),
        })

    # -- persistence ------------------------------------------------------
    def _completed_arm_path(self, arm_id: str) -> Path:
        return self.output / "resume_arms" / f"{arm_id}.pt"

    def _write_completed_arm(self, arm: Arm) -> None:
        entry = self.completed_artifacts.get(arm.arm_id)
        if entry is not None:
            path = self.output / entry["path"]
            _require(path.is_file(), "completed-arm resume artifact is missing")
            _require(sha256_file(path) == entry["sha256"],
                     "completed-arm resume artifact changed")
            return
        path = self._completed_arm_path(arm.arm_id)
        payload = {
            "format": RESUME_ARM_FORMAT,
            "identities": dict(self.identities),
            "arm_id": arm.arm_id,
            "arm": arm.snapshot(),
        }
        _save_torch_atomic(path, payload)
        work = self.process_work()
        work["resume_artifact_writes"] = int(
            work.get("resume_artifact_writes", 0)) + 1
        snapshot = payload["arm"]
        self.completed_artifacts[arm.arm_id] = {
            "arm_id": arm.arm_id,
            "path": path.relative_to(self.output).as_posix(),
            "sha256": sha256_file(path),
            "state_sha256": snapshot["state_sha256"],
            "model_state_sha256": snapshot["model_state_sha256"],
            "fixed_state_sha256": snapshot["fixed_state_sha256"],
            "completed_updates": int(snapshot["completed_updates"]),
            "event_clock_steps": int(snapshot["clock_events"]),
            "events_processed": int(snapshot["event_index"]),
            "memory_record_count": len(snapshot["memory_records"]),
            "update_records_sha256": sha256_json(snapshot["update_records"]),
        }

    def _resume_payload(self, *, active_arm_id: str | None,
                        status: str) -> dict:
        completed = self.completed_ids()
        for arm_id in completed:
            entry = self.completed_artifacts.get(arm_id)
            if (entry is None
                    or entry.get("artifact_format", "resume_arm")
                    == "resume_arm"):
                self._write_completed_arm(self.arms[arm_id])
        active = None
        if active_arm_id is not None:
            arm = self.arms[active_arm_id]
            if arm.started() and not _arm_is_complete(arm, self.metadata):
                active = arm.snapshot()
        process = self.current_process()
        process["status"] = status
        process["optimizer_steps_at_exit"] = self.total_updates()
        return {
            "format": RESUME_FORMAT,
            "protocol_id": "value_readout_abstention_v1",
            "identities": dict(self.identities),
            "arm_order": list(self.arm_order),
            "cursor": {
                "next_arm_index": len(completed),
                "completed_arm_ids": list(completed),
                "active_arm_id": (active_arm_id
                                  if active is not None else None),
                "global_optimizer_steps": self.total_updates(),
            },
            "completed_arms": [self.completed_artifacts[arm_id]
                               for arm_id in completed],
            "active_arm": active,
            "recovery_mode": self.recovery_mode,
            "publication": deepcopy(self.publication),
            "publication_live_peak": deepcopy(self.publication_live_peak),
            "released_recovery_artifacts": deepcopy(self.released_recovery),
            "processes": deepcopy(self.processes),
            "process_count": len(self.processes),
            "restart_count": len(self.processes) - 1,
            "cumulative_work": _sum_work(self.processes),
            "resume_artifact_bytes_peak": self.peak_resume_bytes,
            "classification": {
                "native_model": False, "tokenizer": False, "downloads": 0,
                "device": "cpu", "scope": "deterministic_resume_state",
            },
        }

    def save(self, *, active_arm_id: str | None,
             status: str = "running", check_budget: bool = True) -> None:
        """Persist the resume pointer.

        ``check_budget=False`` is only used by the publication transition, where
        a verified final checkpoint and its not-yet-released training duplicate
        coexist for one write.  That overlap is measured by the publication live
        meter instead; the retained resume state itself is still measured here.
        """

        work = self.process_work()
        work["resume_artifact_writes"] = int(
            work.get("resume_artifact_writes", 0)) + 1
        payload = self._resume_payload(active_arm_id=active_arm_id,
                                       status=status)
        _save_torch_atomic(self.resume_path, payload)
        if check_budget:
            budget = enforce_resume_artifact_budget(
                self.output, self.context.config, context="resume_save")
        else:
            budget = _measure_resume_state(self.output)
        self.peak_resume_bytes = max(self.peak_resume_bytes,
                                     int(budget["resume_artifact_bytes"]))

    def restore(self, payload: Mapping[str, Any]) -> None:
        _require(payload.get("format") == RESUME_FORMAT,
                 "resume format differs")
        _require(payload.get("protocol_id") == "value_readout_abstention_v1",
                 "resume protocol differs")
        _require_resume_identities(payload, self.identities, "resume")
        _require(list(payload.get("arm_order", [])) == list(self.arm_order),
                 "resume arm order differs")
        cursor = payload.get("cursor")
        _require(isinstance(cursor, Mapping), "resume cursor is missing")
        completed = list(cursor.get("completed_arm_ids", []))
        next_index = int(cursor.get("next_arm_index", -1))
        _require(completed == list(self.arm_order[:len(completed)]),
                 "resume completed arms are not a prefix of the arm order")
        _require(next_index == len(completed),
                 "resume cursor and completed arm list disagree")
        _require(next_index <= len(self.arm_order),
                 "resume cursor exceeds the arm order")
        completed_rows = payload.get("completed_arms", [])
        _require(isinstance(completed_rows, list)
                 and len(completed_rows) == len(completed),
                 "resume completed-arm artifacts differ from the cursor")
        by_id = {row["arm_id"]: row for row in completed_rows}
        _require(set(by_id) == set(completed),
                 "resume completed-arm snapshot ids differ")
        for arm_id in completed:
            self._restore_completed_entry(arm_id, by_id[arm_id])
        self.restored_completed_arms = completed
        active_id = cursor.get("active_arm_id")
        active_snapshot = payload.get("active_arm")
        if active_id is None:
            _require(active_snapshot is None,
                     "resume active arm disagrees with the cursor")
        else:
            _require(active_id in self.arms, "resume active arm is unknown")
            _require(active_id not in completed,
                     "resume active arm is already complete")
            _require(isinstance(active_snapshot, Mapping),
                     "resume active arm snapshot is missing")
            _require(active_snapshot.get("arm_id") == active_id,
                     "resume active arm id differs")
            self.arms[active_id].restore(active_snapshot)
            _require(not _arm_is_complete(self.arms[active_id],
                                          self.metadata),
                     "resume active arm is complete")
            for index, arm_id in enumerate(self.arm_order):
                if index > next_index:
                    _require(not self.arms[arm_id].started(),
                             "resume has a started arm beyond the cursor")
        processes = payload.get("processes")
        self._restore_process_history(payload, processes)
        peak = payload.get("resume_artifact_bytes_peak")
        if peak is not None:
            _require(type(peak) is int and peak >= 0,
                     "resume artifact peak is invalid")
            self.peak_resume_bytes = max(self.peak_resume_bytes, peak)
        self.recovery_mode = str(payload.get("recovery_mode")
                                 or "training_snapshots")
        self.publication = deepcopy(payload.get("publication"))
        self.publication_live_peak = deepcopy(
            payload.get("publication_live_peak"))
        self.released_recovery = [dict(row) for row in
                                  payload.get("released_recovery_artifacts", [])]

    def _restore_process_history(self, payload: Mapping[str, Any],
                                 processes: Any) -> None:
        _require(isinstance(processes, list) and processes,
                 "resume process history is missing")
        _require(int(payload.get("process_count", -1)) == len(processes),
                 "resume process count differs")
        _require(int(payload.get("restart_count", -1)) == len(processes) - 1,
                 "resume restart count differs")
        for index, row in enumerate(processes):
            _require(int(row.get("process_index", -1)) == index,
                     "resume process indices differ")
            _require(int(row.get("restart_index", -1)) == index,
                     "resume restart indices differ")
            _require(set(row.get("work", {})) == set(WORK_COUNTERS),
                     "resume process work schema differs")
            for key, value in row["work"].items():
                _require(type(value) is int and value >= 0,
                         f"resume process work {key} is invalid")
        cumulative = payload.get("cumulative_work")
        _require(isinstance(cumulative, Mapping),
                 "resume cumulative work is missing")
        expected = _sum_work(processes)
        _require(dict(cumulative) == expected,
                 "resume cumulative work differs from process records")
        _require(expected["optimizer_steps"] == self.total_updates(),
                 "resume physical optimizer steps differ from arm cursors")
        self.processes = [dict(row) for row in processes]

    def _restore_completed_entry(self, arm_id: str,
                                 entry: Mapping[str, Any]) -> None:
        path = self.output / str(entry["path"])
        _require(path.is_file(), "completed-arm resume artifact is missing")
        _require(sha256_file(path) == entry["sha256"],
                 "completed-arm resume artifact hash differs")
        artifact_format = str(entry.get("artifact_format", "resume_arm"))
        if artifact_format == "final_checkpoint":
            self._restore_final_checkpoint(arm_id, entry, path)
        elif artifact_format == "resume_arm":
            artifact = torch.load(path, map_location="cpu", weights_only=True)
            _require(artifact.get("format") == RESUME_ARM_FORMAT,
                     "completed-arm resume format differs")
            _require(artifact.get("arm_id") == arm_id,
                     "completed-arm resume id differs")
            _require_resume_identities(artifact, self.identities,
                                       f"completed arm {arm_id}")
            self.arms[arm_id].restore(artifact["arm"])
            _require(_arm_is_complete(self.arms[arm_id], self.metadata),
                     "resume completed arm is not complete")
            _require(self.arms[arm_id].state_digest()
                     == entry["state_sha256"],
                     "completed-arm state differs from its receipt")
            _require(entry.get("update_records_sha256")
                     == sha256_json(self.arms[arm_id].update_records),
                     "completed-arm update record hash differs")
        else:
            _require(False, f"unknown completed-arm artifact format "
                            f"{artifact_format!r}")
        _require(int(entry.get("completed_updates", -1)) == 24
                 and int(entry.get("event_clock_steps", -1)) == 36
                 and int(entry.get("events_processed", -1)) == 36,
                 "completed-arm cursor evidence differs")
        self.completed_artifacts[arm_id] = dict(entry)

    def _restore_final_checkpoint(self, arm_id: str,
                                  entry: Mapping[str, Any],
                                  path: Path) -> None:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        _require(payload.get("format") == READOUT_CHECKPOINT_FORMAT,
                 "final checkpoint format differs")
        _require(payload.get("protocol_id") == "value_readout_abstention_v1",
                 "final checkpoint protocol differs")
        _require(payload.get("arm_id") == arm_id, "final checkpoint arm differs")
        _require(payload.get("config_sha256")
                 == self.identities["config_sha256"],
                 "final checkpoint config identity differs")
        _require(payload.get("prepared_manifest_sha256")
                 == self.identities["prepared_manifest_sha256"],
                 "final checkpoint manifest identity differs")
        _require(int(payload.get("completed_steps", -1)) == 24,
                 "final checkpoint is not a completed 24-step arm")
        _require(payload.get("state_sha256") == entry["state_sha256"],
                 "final checkpoint state differs from its receipt")
        _require(payload.get("parent_state_sha256")
                 == self.arms[arm_id].parent_state_sha256,
                 "final checkpoint parent state differs")
        _require(payload.get("model_state_sha256")
                 == entry.get("model_state_sha256"),
                 "final checkpoint model state differs from its receipt")
        _require(payload.get("fixed_state_sha256")
                 == entry.get("fixed_state_sha256"),
                 "final checkpoint fixed state differs from its receipt")
        _require(sha256_json(payload.get("update_records"))
                 == entry.get("update_records_sha256"),
                 "final checkpoint update records differ from its receipt")
        _require(int(payload.get("event_clock_steps", -1)) == 36
                 and int(payload.get("events_processed", -1)) == 36,
                 "final checkpoint clock/event count differs")
        arm = self.arms[arm_id]
        with torch.no_grad():
            for name, tensor in payload["model_state"].items():
                arm.module.model_state()[name].copy_(tensor)
        _require(_model_state_from_checkpoint(payload["model_state"]).keys()
                 == arm.module.model_state().keys(),
                 "final checkpoint model state names differ")
        _restore_optimizer(arm.module, arm.optimizer, payload["optimizer_state"])
        arm.completed_updates = int(payload["completed_steps"])
        arm.clock = TeachingClock(int(payload["event_clock_steps"]))
        arm.event_index = int(payload["events_processed"])
        arm.update_records = deepcopy(list(payload["update_records"]))
        _require(arm.state_digest() == entry["state_sha256"],
                 "restored final-checkpoint state differs")
        _require(len(arm.update_records) == arm.completed_updates,
                 "restored final-checkpoint update record count differs")
        _require(_arm_is_complete(arm, self.metadata),
                 "restored final checkpoint is not complete")

    # -- final publication -------------------------------------------------
    def publication_ready(self) -> bool:
        return self.publication is not None

    def publication_artifacts(self) -> dict[str, list[dict]]:
        _require(self.publication is not None,
                 "final publication is not recorded")
        return {str(stage): [dict(row) for row in rows]
                for stage, rows in self.publication.get("artifacts", {}).items()}

    def verify_publication(self) -> None:
        """Verify every published stage by content identity."""

        _require(self.publication is not None,
                 "final publication is not recorded")
        stages = self.publication_artifacts()
        _require(bool(stages), "final publication recorded no stages")
        for stage, rows in stages.items():
            _require(bool(rows), f"published stage {stage} is empty")
            for row in rows:
                artifact = publication.Artifact(
                    str(row["path"]), str(row["sha256"]), int(row["bytes"]))
                _require(publication.inspect(self.output, artifact),
                         f"published {stage} artifact changed: {row['path']}")

    def bind_final_recovery(self, *,
                            bindings: Mapping[str, Mapping[str, Any]]) -> None:
        """Serve completed-arm recovery from verified final checkpoints.

        Each binding is loaded and validated by content identity before it
        replaces the training snapshot, so recovery never depends on an
        unverified artifact.
        """

        completed = self.completed_ids()
        _require(completed == list(self.arm_order),
                 "final recovery requires every arm to be complete")
        for arm_id in self.arm_order:
            _require(arm_id in bindings,
                     f"final checkpoint binding is missing for {arm_id}")
            row = dict(bindings[arm_id])
            entry = {
                "arm_id": arm_id,
                "artifact_format": "final_checkpoint",
                "path": str(row["path"]),
                "sha256": str(row["sha256"]),
                "state_sha256": str(row["state_sha256"]),
                "model_state_sha256": str(row["model_state_sha256"]),
                "fixed_state_sha256": deepcopy(row["fixed_state_sha256"]),
                "completed_updates": int(row["completed_updates"]),
                "event_clock_steps": int(row["event_clock_steps"]),
                "events_processed": int(row["events_processed"]),
                "update_records_sha256": str(row["update_records_sha256"]),
                "predictions_sha256": row.get("predictions_sha256"),
            }
            self._restore_completed_entry(arm_id, entry)
            self.completed_artifacts[arm_id] = entry
        self.recovery_mode = "final_checkpoints"

    def release_resume_snapshots(self) -> list[dict]:
        """Release heavy training snapshots only behind verified replacements."""

        _require(self.recovery_mode == "final_checkpoints",
                 "training snapshots need a durable final-checkpoint replacement")
        for arm_id in self.arm_order:
            entry = self.completed_artifacts.get(arm_id) or {}
            _require(entry.get("artifact_format") == "final_checkpoint",
                     "final-checkpoint replacement is not durable")
            durable = publication.Artifact.of(self.output, entry["path"])
            _require(durable.sha256 == entry["sha256"],
                     "final-checkpoint recovery artifact changed")
        relatives = [path.relative_to(self.output).as_posix()
                     for path in _resume_arm_files(self.output)]
        released = publication.release(self.output, relatives)
        self.released_recovery = list(self.released_recovery) + [
            row for row in released if row["path"].startswith("resume_arms/")]
        return released

    def mark_publication(self, *,
                         artifacts: Mapping[str, Sequence[Mapping[str, Any]]],
                         live_peak: Mapping[str, Any] | None) -> None:
        """Record the counted publication stage durably."""

        _require(self.recovery_mode == "final_checkpoints",
                 "publication requires final-checkpoint recovery")
        staged: dict[str, list[dict]] = {}
        for stage, records in artifacts.items():
            _require(bool(records), f"published stage {stage} is empty")
            rows = []
            for record in records:
                artifact = publication.Artifact(
                    str(record["path"]), str(record["sha256"]),
                    int(record["bytes"]))
                _require(publication.inspect(self.output, artifact),
                         f"published {stage} artifact changed: {artifact.path}")
                row = dict(record)
                row.update(artifact.as_dict())
                rows.append(row)
            staged[str(stage)] = rows
        self.publication = {
            "status": "final_artifacts_published",
            "artifacts": staged,
            "recovery_mode": self.recovery_mode,
            "released_recovery_artifacts": [dict(row) for row in
                                            self.released_recovery],
        }
        self.publication_live_peak = publication.peak_union(
            self.publication_live_peak, live_peak)

    def recover_from_evidence(self, evidence: Mapping[str, Any]) -> None:
        """Rebuild a completed fit from retained evidence and its artifacts.

        Used when ``resume.pt`` is already gone (publication finished, only a
        later stage failed).  Completed arms are restored from the final
        checkpoints recorded in the evidence and verified by content identity,
        so no optimizer work is repeated.
        """

        _require(evidence.get("format") == RESUME_EVIDENCE_FORMAT,
                 "resume evidence format differs")
        _require(evidence.get("protocol_id") == "value_readout_abstention_v1",
                 "resume evidence protocol differs")
        _require(evidence.get("status") == "training_complete",
                 "resume evidence is not complete")
        _require_resume_identities(evidence, self.identities, "resume evidence")
        _require(list(evidence.get("arm_order", [])) == list(self.arm_order),
                 "resume evidence arm order differs")
        cursor = evidence.get("cursor")
        _require(isinstance(cursor, Mapping), "resume evidence cursor is missing")
        _require(list(cursor.get("completed_arm_ids", []))
                 == list(self.arm_order),
                 "resume evidence did not complete every arm")
        _require(int(cursor.get("next_arm_index", -1)) == len(self.arm_order)
                 and cursor.get("active_arm_id") is None,
                 "resume evidence cursor is not at the completed boundary")
        rows = evidence.get("completed_arms")
        _require(isinstance(rows, list)
                 and len(rows) == len(self.arm_order),
                 "resume evidence completed-arm rows differ")
        by_id = {str(row["arm_id"]): row for row in rows}
        _require(set(by_id) == set(self.arm_order),
                 "resume evidence completed-arm ids differ")
        for arm_id in self.arm_order:
            self._restore_completed_entry(arm_id, by_id[arm_id])
        self.restored_completed_arms = list(self.arm_order)
        self._restore_process_history(evidence, evidence.get("processes"))
        peak = evidence.get("resume_artifact_bytes_peak")
        if peak is not None:
            _require(type(peak) is int and peak >= 0,
                     "resume artifact peak is invalid")
            self.peak_resume_bytes = max(self.peak_resume_bytes, peak)
        self.recovery_mode = str(evidence.get("recovery_mode")
                                 or "final_checkpoints")
        self.publication = deepcopy(evidence.get("publication"))
        self.publication_live_peak = deepcopy(
            evidence.get("publication_live_peak"))
        self.released_recovery = [dict(row) for row in
                                  evidence.get("released_recovery_artifacts", [])]

    def write_evidence(self, *, final_state_by_arm: Mapping[str, Mapping[str, Any]],
                       final_checkpoint_sha256: Mapping[str, str],
                       meter: publication.LiveByteMeter | None = None) -> Path:
        completed = self.completed_ids()
        _require(completed == list(self.arm_order),
                 "resume evidence requires every arm to be complete")
        work = self.process_work()
        work["resume_artifact_writes"] = int(
            work.get("resume_artifact_writes", 0)) + 1
        self.current_process()["status"] = "completed"
        cumulative = _sum_work(self.processes)
        _require(cumulative["optimizer_steps"]
                 == self.total_updates() == 72,
                 "resume cumulative optimizer work differs from 72")
        _require(cumulative["backward_calls"] == 72
                 and cumulative["training_forward_calls"] == 72,
                 "resume cumulative training work differs from one pass per update")
        resume_rows = [
            {"arm_id": arm_id,
             "artifact_format": self.completed_artifacts[arm_id].get(
                 "artifact_format", "resume_arm"),
             "path": self.completed_artifacts[arm_id]["path"],
             "sha256": self.completed_artifacts[arm_id]["sha256"],
             "state_sha256": self.completed_artifacts[arm_id]["state_sha256"],
             "model_state_sha256": self.completed_artifacts[arm_id].get(
                 "model_state_sha256"),
             "fixed_state_sha256": self.completed_artifacts[arm_id].get(
                 "fixed_state_sha256"),
             "completed_updates": self.completed_artifacts[arm_id][
                 "completed_updates"],
             "event_clock_steps": self.completed_artifacts[arm_id][
                 "event_clock_steps"],
             "events_processed": self.completed_artifacts[arm_id][
                 "events_processed"],
             "update_records_sha256": self.completed_artifacts[arm_id][
                 "update_records_sha256"],
             "predictions_sha256": self.completed_artifacts[arm_id].get(
                 "predictions_sha256"),
             "final_state_sha256": final_state_by_arm[arm_id]["state_sha256"],
             "final_model_state_sha256": final_state_by_arm[arm_id][
                 "model_state_sha256"],
             "final_fixed_state_sha256": final_state_by_arm[arm_id][
                 "fixed_state_sha256"],
             "final_checkpoint_sha256": final_checkpoint_sha256[arm_id]}
            for arm_id in completed]
        evidence = {
            "format": RESUME_EVIDENCE_FORMAT,
            "protocol_id": "value_readout_abstention_v1",
            "status": "training_complete",
            "identities": dict(self.identities),
            "arm_order": list(self.arm_order),
            "cursor": {
                "next_arm_index": len(completed),
                "completed_arm_ids": list(completed),
                "active_arm_id": None,
            },
            "processes": deepcopy(self.processes),
            "process_count": len(self.processes),
            "restart_count": len(self.processes) - 1,
            "cumulative_work": cumulative,
            "optimizer_steps_by_process": [
                int(row["work"]["optimizer_steps"]) for row in self.processes],
            "completed_arms": resume_rows,
            "completed_arms_never_retrained": True,
            "restored_completed_arms": list(self.restored_completed_arms),
            "recovery_mode": self.recovery_mode,
            "publication": deepcopy(self.publication),
            "publication_live_peak": deepcopy(self.publication_live_peak),
            "released_recovery_artifacts": deepcopy(self.released_recovery),
            "resume_artifact_bytes_peak": self.peak_resume_bytes,
            "artifact_cap_bytes": int(
                self.context.config["budget"]["authorized_run_ceiling"][
                    "artifact_bytes_max"]),
            "heavy_resume_artifacts_removed_after_completion": True,
            "released_after_final_checkpoint_verification": [
                row["path"] for row in self.released_recovery],
            "retained_evidence": True,
            "resume_pointer_state": "retained_until_publication_measurement",
            "deleted_after_measurement": ["resume.pt"],
        }
        path = self.output / "resume_evidence.json"
        _write_json_atomic(path, evidence, meter=meter)
        return path

    def remove_artifacts(self) -> None:
        for path in _resume_artifact_files(self.output):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        arms_dir = self.output / "resume_arms"
        if arms_dir.is_dir() and not any(arms_dir.iterdir()):
            try:
                arms_dir.rmdir()
            except OSError:
                pass


def _train_arm_events(arm: Arm, *, metadata: Mapping[str, Any],
                      vocabulary: Sequence[str], config: Mapping[str, Any],
                      work: dict[str, int],
                      after_update) -> None:
    events = metadata["events"]
    cases = metadata["cases"]
    torch.set_num_threads(1)
    while arm.event_index < len(events):
        event = events[arm.event_index]
        arm.clock = TeachingClock(arm.clock.eligible_events + 1)
        if event["kind"] != "query":
            arm.memory.write(event)
            work["training_write_events"] = int(
                work.get("training_write_events", 0)) + 1
            arm.event_index += 1
            continue
        case = _event_case(event, cases)
        features, index, record = _features_for_event(
            arm.arm_id, event, case, arm.memory, vocabulary)
        if int(event["optimizer_steps"]) == 1:
            fingerprint = sha256_json({
                "event_id": event["event_id"],
                "case_id": event["case_id"],
                "target_index": index,
                "clock": arm.clock.eligible_events,
                "feature": features.detach().cpu().tolist(),
            })
            prior = arm.ledger.require(event["event_id"], fingerprint)
            if prior is None:
                target = torch.tensor([index], dtype=torch.long)
                arm.optimizer.zero_grad(set_to_none=True)
                logits = arm.module(features)
                work["training_forward_calls"] = int(
                    work.get("training_forward_calls", 0)) + 1
                loss = F.cross_entropy(logits, target)
                _require(bool(torch.isfinite(loss)), "training loss is nonfinite")
                loss.backward()
                work["backward_calls"] = int(work.get("backward_calls", 0)) + 1
                gradient_norm = torch.sqrt(sum(
                    (parameter.grad.detach() ** 2).sum()
                    for _name, parameter in arm.module.trainable_parameters()
                    if parameter.grad is not None))
                gradient_norm_value = float(gradient_norm.detach().cpu())
                _require(bool(torch.isfinite(gradient_norm)), "gradient is nonfinite")
                torch.nn.utils.clip_grad_norm_(
                    [parameter for _name, parameter in arm.module.trainable_parameters()],
                    float(config["transaction"]["gradient_norm_max"]))
                arm.optimizer.step()
                work["optimizer_steps"] = int(
                    work.get("optimizer_steps", 0)) + 1
                arm.completed_updates += 1
                arm.update_records.append({
                    "step": arm.completed_updates - 1,
                    "event_id": event["event_id"],
                    "case_id": event["case_id"],
                    "world_id": event["world_id"],
                    "epoch": int(event["epoch"]),
                    "clock_step": arm.clock.eligible_events,
                    "loss": float(loss.detach().cpu()),
                    "gradient_norm": gradient_norm_value,
                    "gradient_finite": True,
                    "state_after_sha256": arm.module.state_digest(),
                })
                outcome = TransactionOutcome(
                    status="committed", event_id=event["event_id"],
                    fingerprint=fingerprint, expert_id=0,
                    revision=arm.completed_updates,
                    eligible_events=arm.clock.eligible_events,
                    steps=1, loss=float(loss.detach().cpu()))
                arm.ledger.commit(event["event_id"], fingerprint, outcome)
                arm.event_index += 1
                if after_update is not None:
                    after_update(arm)
                continue
        arm.event_index += 1


def _train_arm(arm: Arm, *, state: FitResumeState) -> None:
    def after_update(current: Arm) -> None:
        state.save(active_arm_id=current.arm_id)
        total = state.total_updates()
        if (state.interrupt_after is not None
                and total >= int(state.interrupt_after)):
            state.save(active_arm_id=current.arm_id, status="interrupted")
            raise RunInterrupted(total, state.resume_path)

    _train_arm_events(arm, metadata=state.metadata,
                      vocabulary=state.vocabulary,
                      config=state.context.config,
                      work=state.process_work(),
                      after_update=after_update)


def replay_fit_evidence(metadata: Mapping[str, Any],
                        config: Mapping[str, Any]) -> dict:
    """Deterministically replay the three uninterrupted 24-update arms.

    Admission uses this to bind update hashes, losses, event clocks, world and
    epoch relationships and the final checkpoint tensors to executed work
    rather than trusting a plausible final hash.
    """
    vocabulary = list(config["architecture"]["answer_vocabulary"])
    seed = int(config["transaction"]["seed"])
    result = {}
    for arm_id in TRAINED_ARMS:
        arm = Arm(arm_id, config, seed)
        work = _empty_work()
        _train_arm_events(arm, metadata=metadata, vocabulary=vocabulary,
                          config=config, work=work, after_update=None)
        _require(_arm_is_complete(arm, metadata),
                 f"deterministic replay did not complete {arm_id}")
        with torch.no_grad():
            predictions = _predict_split(
                arm_id, arm.module, split="fit", events=metadata["events"],
                cases=list(metadata["cases"].values()), vocabulary=vocabulary,
                update_digest=arm.module.state_digest())
        result[arm_id] = {
            "arm_id": arm_id,
            "completed_updates": arm.completed_updates,
            "event_clock_steps": arm.clock.eligible_events,
            "events_processed": arm.event_index,
            "parent_state_sha256": arm.parent_state_sha256,
            "state_sha256": arm.state_digest(),
            "model_state_sha256": arm.module.model_state_digest(),
            "fixed_state_sha256": arm.module.fixed_state_digest(),
            "update_records": deepcopy(arm.update_records),
            "predictions": predictions,
            "predictions_sha256": sha256_json(predictions),
            "work": _work_copy(work),
        }
    return result


def _predict_split(arm_id: str, module: SelectedExpertReadout, *,
                   split: str, events: Sequence[Mapping[str, Any]],
                   cases: Sequence[Mapping[str, Any]],
                   vocabulary: Sequence[str], update_digest: str,
                   work: dict[str, int] | None = None,
                   reset_memory_before_queries: bool = False) -> list[dict]:
    memory = FixedRecordMemory()
    by_id = {case["case_id"]: case for case in cases}
    rows = []
    module.eval()
    with torch.no_grad():
        clock = 0
        for event in events:
            clock += 1
            if event["kind"] != "query":
                memory.write(event)
                if work is not None:
                    work["evaluation_write_events"] = int(
                        work.get("evaluation_write_events", 0)) + 1
                continue
            if reset_memory_before_queries:
                memory = FixedRecordMemory()
            case = dict(by_id[event["case_id"]])
            before_memory = memory.digest()
            before_state = module.state_digest()
            features, index, record = _features_for_event(
                arm_id, event, case, memory, vocabulary)
            logits = module(features)
            if work is not None:
                work["evaluation_forward_calls"] = int(
                    work.get("evaluation_forward_calls", 0)) + 1
                if case["case_type"] == "unaffected_retention":
                    work["retention_diagnostic_forwards"] = int(
                        work.get("retention_diagnostic_forwards", 0)) + 1
            predicted_index = int(logits.argmax(dim=-1).item())
            _require(memory.digest() == before_memory,
                     "query mutated the fixed memory snapshot")
            _require(module.state_digest() == before_state,
                     "query mutated the readout state")
            rows.append({
                "format": PREDICTION_FORMAT,
                "protocol_id": "value_readout_abstention_v1",
                "split": split,
                "arm_id": arm_id,
                "case_id": case["case_id"],
                "event_id": event["event_id"],
                "world_id": case["world_id"],
                "epoch": int(case["epoch"]),
                "case_type": case["case_type"],
                "predicted_index": predicted_index,
                "predicted_token": list(vocabulary)[predicted_index],
                "model_state_sha256": before_state,
                "memory_state_sha256": before_memory,
                "clock_step": clock,
            })
    rows.sort(key=lambda row: row["case_id"])
    return rows


def _metrics_from_predictions(predictions: Sequence[Mapping[str, Any]],
                              targets: Mapping[str, Mapping[str, Any]]) -> dict:
    counts = {case_type: {"correct": 0, "total": 0} for case_type in (
        "future_answer", "correction", "unaffected_retention",
        "unknown_abstention", "wrong_world_abstention")}
    stale_hits = 0
    stale_total = 0
    covered = set()
    for row in predictions:
        case_id = row["case_id"]
        _require(case_id not in covered, f"duplicate prediction case {case_id}")
        covered.add(case_id)
        _require(case_id in targets, f"prediction has no authoritative target: {case_id}")
        target = targets[case_id]
        counts[target["case_type"]]["total"] += 1
        if row["predicted_token"] == target["target"]:
            counts[target["case_type"]]["correct"] += 1
        if target["case_type"] == "correction":
            stale_total += 1
            if row["predicted_token"] == target.get("stale_value"):
                stale_hits += 1
    metrics = {}
    for case_type, entry in counts.items():
        metrics[f"{case_type}_accuracy"] = (
            entry["correct"] / entry["total"] if entry["total"] else 0.0)
    metrics["stale_value_rate"] = (stale_hits / stale_total if stale_total else 0.0)
    return {
        "per_case": {
            "count": len(predictions),
            "covered": len(covered),
            "unique_ids": len(covered) == len(predictions),
            "counts": counts,
        },
        "metrics": metrics,
    }


def _fit_targets(metadata: Mapping[str, Any]) -> dict:
    targets = {}
    for case in metadata["train"] + metadata["replay"]:
        targets[case["case_id"]] = {
            "case_type": case["case_type"],
            "target": case["target"]["token"],
            "stale_value": case.get("stale_value"),
        }
    return targets


def _gates_from_metrics(config: Mapping[str, Any], group: str,
                        metrics: Mapping[str, Any]) -> tuple[dict, bool]:
    thresholds = dict(config["evaluation"][group]["thresholds"])
    passed = True
    for key, limit in thresholds.items():
        name = key[:-4]
        value = float(metrics[name])
        if key.endswith("_min"):
            passed = passed and value >= float(limit)
        else:
            passed = passed and value <= float(limit)
    return thresholds, passed


def _fresh_output(path: Path, *, resume: bool = False) -> Path:
    path = Path(path)
    if resume:
        path.mkdir(parents=True, exist_ok=True)
        return path
    if path.exists() and any(path.iterdir()):
        raise ReadoutError(f"output path is not fresh: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _check_runtime_precheck(prepared: Path, config_path: Path) -> dict:
    # The runner never parses evaluation labels during fit.  Import the contract
    # lazily so this module remains usable without the scripts package.
    import sys
    scripts = Path(__file__).resolve().parents[2] / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    from value_readout_abstention_contract_v1 import runtime_precheck
    return runtime_precheck(prepared, config_path)


def _case_accuracy(metrics: Mapping[str, Any]) -> float:
    counts = metrics["per_case"]["counts"]
    total = sum(int(entry["total"]) for entry in counts.values())
    correct = sum(int(entry["correct"]) for entry in counts.values())
    return correct / total if total else 0.0


def _baseline_report(*, name: str, rows: list[dict],
                     module: SelectedExpertReadout,
                     targets: Mapping[str, Mapping[str, Any]],
                     config: Mapping[str, Any],
                     split: str, memory_semantics: str,
                     predictions_sha256: str) -> dict:
    metrics = _metrics_from_predictions(rows, targets)
    thresholds, passed = _gates_from_metrics(config, "fit",
                                             metrics["metrics"])
    return {
        "arm_id": name,
        "optimizer_steps": 0,
        "backward_calls": 0,
        "gradient": False,
        "split": split,
        "memory": memory_semantics,
        "state_sha256": module.state_digest(),
        "model_state_sha256": module.model_state_digest(),
        "fixed_state_sha256": module.fixed_state_digest(),
        "predictions_sha256": predictions_sha256,
        "case_accuracy": _case_accuracy(metrics),
        "per_case": metrics["per_case"],
        "metrics": metrics["metrics"],
        "thresholds": thresholds,
        "all_declared_gates_passed": bool(passed),
    }


def _zero_update_baseline(*, name: str, rows: list[dict],
                          module: SelectedExpertReadout,
                          targets: Mapping[str, Mapping[str, Any]],
                          config: Mapping[str, Any],
                          split: str, memory_semantics: str,
                          root: Path, relative: str,
                          meter: publication.LiveByteMeter | None = None) -> dict:
    record = publication.write_jsonl(root, relative, rows, meter=meter)
    return _baseline_report(
        name=name, rows=rows, module=module, targets=targets, config=config,
        split=split, memory_semantics=memory_semantics,
        predictions_sha256=record.sha256)


def _arm_report(*, arm_id: str, arm: Arm, rows: Sequence[Mapping[str, Any]],
                targets: Mapping[str, Mapping[str, Any]],
                config: Mapping[str, Any], events_processed: int) -> dict:
    metrics = _metrics_from_predictions(rows, targets)
    thresholds, passed = _gates_from_metrics(config, "fit", metrics["metrics"])
    return {
        "arm_id": arm_id,
        "phase": ("fit_treatment" if arm_id == TREATMENT_ARM
                  else "control_fit"),
        "fit_receipt_binding": arm_id == TREATMENT_ARM,
        "optimizer_steps": arm.completed_updates,
        "event_clock_steps": arm.clock.eligible_events,
        "events_processed": int(events_processed),
        "parent_state_sha256": arm.parent_state_sha256,
        "state_sha256": arm.module.state_digest(),
        "model_state_sha256": arm.module.model_state_digest(),
        "fixed_state_sha256": arm.module.fixed_state_digest(),
        "optimizer_meta": _optimizer_meta(arm.module, arm.optimizer),
        "update_records": deepcopy(arm.update_records),
        "per_case": metrics["per_case"],
        "metrics": metrics["metrics"],
        "thresholds": thresholds,
        "all_declared_gates_passed": bool(passed),
        "predictions_sha256": sha256_json(rows),
    }


def _reuse_fit_publication(*, output: Path, arms: Mapping[str, Arm],
                           metadata: Mapping[str, Any],
                           targets: Mapping[str, Mapping[str, Any]],
                           context: RunnerContext,
                           prediction_records: Sequence[Mapping[str, Any]],
                           checkpoint_records: Sequence[Mapping[str, Any]]
                           ) -> dict:
    """Rebuild the fit report inputs from durable published artifacts.

    No forward pass, backward pass or optimizer step happens here: prediction
    rows are read back from their published files (verified by content) and the
    metrics and baseline digests are recomputed from them, so the resumed report
    is byte-identical to the uninterrupted one.
    """

    by_role = {str(row["role"]): row for row in prediction_records}
    rows_by_role = {}
    for role in (*TRAINED_ARMS, "baseline_parent", "baseline_reset"):
        _require(role in by_role,
                 f"published fit predictions are missing {role}")
        record = by_role[role]
        path = output / str(record["path"])
        _require(path.is_file(),
                 f"published fit predictions are missing: {record['path']}")
        _require(publication.sha256_path(path) == str(record["sha256"]),
                 f"published fit predictions changed: {record['path']}")
        rows_by_role[role] = read_jsonl(path)
    arm_reports = {
        arm_id: _arm_report(
            arm_id=arm_id, arm=arms[arm_id], rows=rows_by_role[arm_id],
            targets=targets, config=context.config,
            events_processed=len(metadata["events"]))
        for arm_id in TRAINED_ARMS}
    parent = SelectedExpertReadout(
        seed=int(context.config["transaction"]["seed"]))
    parent_baseline = _baseline_report(
        name="zero_update_parent", rows=rows_by_role["baseline_parent"],
        module=parent, targets=targets, config=context.config, split="fit",
        memory_semantics="chronological_fixed_record_view",
        predictions_sha256=str(by_role["baseline_parent"]["sha256"]))
    reset_baseline = _baseline_report(
        name="zero_update_reset", rows=rows_by_role["baseline_reset"],
        module=parent, targets=targets, config=context.config, split="fit",
        memory_semantics="reset_before_every_query",
        predictions_sha256=str(by_role["baseline_reset"]["sha256"]))
    return {
        "arm_reports": arm_reports,
        "parent_baseline": parent_baseline,
        "reset_baseline": reset_baseline,
        "checkpoint_sha256": {str(row["role"]): str(row["sha256"])
                              for row in checkpoint_records},
    }


def run_fit(prepared: Path, config_path: Path, output: Path, *,
            interrupt_after: int | None = None,
            resume: bool = False) -> dict:
    """Train the three independent 24-update arms and write fit evidence."""
    prepared = Path(prepared)
    config_path = Path(config_path)
    runtime = _check_runtime_precheck(prepared, config_path)
    context = _context(config_path, prepared)
    metadata = _read_train_metadata(prepared, context.config)
    vocabulary = list(context.config["architecture"]["answer_vocabulary"])
    output = Path(output)
    resume_path = output / "resume.pt"
    evidence_path = output / "resume_evidence.json"
    output = _fresh_output(
        output, resume=resume or resume_path.exists() or evidence_path.exists())
    arms = {arm_id: Arm(arm_id, context.config,
                        int(context.config["transaction"]["seed"]))
            for arm_id in TRAINED_ARMS}
    state = FitResumeState(
        output=output, context=context, metadata=metadata,
        vocabulary=vocabulary, arms=arms, resume_path=resume_path,
        interrupt_after=interrupt_after)
    if resume_path.exists():
        state.restore(_load_resume(resume_path))
        state.add_process(resumed_from={
            "kind": "resume_artifact",
            "resume_format": RESUME_FORMAT,
            "completed_arm_ids": list(state.restored_completed_arms),
            "next_arm_index": state.next_arm_index(),
            "recovery_mode": state.recovery_mode,
        })
    elif evidence_path.exists():
        state.recover_from_evidence(read_json(evidence_path))
        state.add_process(resumed_from={
            "kind": "resume_evidence",
            "resume_format": RESUME_EVIDENCE_FORMAT,
            "completed_arm_ids": list(state.restored_completed_arms),
            "next_arm_index": state.next_arm_index(),
            "recovery_mode": state.recovery_mode,
        })
    else:
        _require(not resume,
                 "resume requested but no durable resume state exists")
        state.add_process(resumed_from={"kind": "fresh_process"})
    # Every non-final process was terminated; the evidence contract only
    # distinguishes interrupted and completed processes.
    for row in state.processes[:-1]:
        if row.get("status") != "interrupted":
            row["recovered_from_status"] = row.get("status")
            row["status"] = "interrupted"
    for index in range(state.next_arm_index(), len(TRAINED_ARMS)):
        arm = arms[TRAINED_ARMS[index]]
        _train_arm(arm, state=state)
        _require(_arm_is_complete(arm, metadata),
                 f"arm {arm.arm_id} did not complete")
        state.save(active_arm_id=None)
    _require(state.completed_ids() == list(TRAINED_ARMS),
             "fit did not complete every trained arm")
    treatment = arms[TREATMENT_ARM]
    fit_dir = output / "fit"
    fit_dir.mkdir(parents=True, exist_ok=True)
    targets = _fit_targets(metadata)
    meter = publication.LiveByteMeter(output, label="fit_publication")
    meter.observe()
    work = state.process_work()
    def count_checkpoint_write() -> None:
        work["checkpoint_writes"] = int(
            work.get("checkpoint_writes", 0)) + 1
    if state.publication_ready():
        # The counted publication stage is already durable.  Verify it by
        # content identity and republish only the uncounted stages below, so a
        # resume never repeats optimizer, backward or forward work.
        state.verify_publication()
        if _resume_arm_files(output):
            # A previous process was interrupted while releasing the verified
            # training duplicates; finish the release behind the same durable
            # verification before publishing the uncounted stages.
            state.release_resume_snapshots()
            state.save(active_arm_id=None, status="publication_complete",
                       check_budget=False)
        published = state.publication_artifacts()
        _require("predictions" in published and "checkpoints" in published,
                 "published fit stages are incomplete")
        reused = _reuse_fit_publication(
            output=output, arms=arms, metadata=metadata, targets=targets,
            context=context,
            prediction_records=published["predictions"],
            checkpoint_records=published["checkpoints"])
        arm_reports = reused["arm_reports"]
        parent_baseline = reused["parent_baseline"]
        reset_baseline = reused["reset_baseline"]
        final_checkpoint_sha256 = reused["checkpoint_sha256"]
    else:
        # Fit predictions are a fresh, no-update replay against the final
        # treatment checkpoint.  Each arm gets its own replay for control
        # evidence.  These writes are idempotent: identical bytes are kept.
        arm_reports = {}
        prediction_records = []
        checkpoint_records = []
        checkpoint_payloads: dict[str, dict] = {}
        fit_predictions_sha256 = ""
        for arm_id in TRAINED_ARMS:
            arm = arms[arm_id]
            rows = _predict_split(
                arm_id, arm.module, split="fit", events=metadata["events"],
                cases=list(metadata["cases"].values()), vocabulary=vocabulary,
                update_digest=arm.module.state_digest(), work=work)
            state.save(active_arm_id=None, status="publication_predictions",
                       check_budget=False)
            relative = ("fit/fit_predictions.jsonl" if arm_id == TREATMENT_ARM
                        else f"fit/control_{arm_id}_predictions.jsonl")
            record = publication.write_jsonl(output, relative, rows,
                                             meter=meter)
            prediction_records.append({"role": arm_id, "arm_id": arm_id,
                                       **record.as_dict()})
            if arm_id == TREATMENT_ARM:
                fit_predictions_sha256 = record.sha256
            arm_reports[arm_id] = _arm_report(
                arm_id=arm_id, arm=arm, rows=rows, targets=targets,
                config=context.config, events_processed=len(metadata["events"]))
            if arm_id == TREATMENT_ARM:
                continue
            payload = arm.checkpoint_payload(
                config=context.config, config_sha256=context.config_sha256,
                prepared_manifest_sha256=context.manifest_sha256,
                events_processed=len(metadata["events"]),
                predictions_sha256=record.sha256)
            checkpoint_payloads[arm_id] = payload
            saved = publication.write_torch(
                output, f"fit/control_{arm_id}.pt",
                _canonical_payload(payload), meter=meter,
                on_write=count_checkpoint_write)
            checkpoint_records.append({"role": arm_id, "arm_id": arm_id,
                                       **saved.as_dict()})
            state.save(active_arm_id=None, status="publication_checkpoint",
                       check_budget=False)
        treatment_payload = treatment.checkpoint_payload(
            config=context.config, config_sha256=context.config_sha256,
            prepared_manifest_sha256=context.manifest_sha256,
            events_processed=len(metadata["events"]),
            predictions_sha256=fit_predictions_sha256)
        checkpoint_payloads[TREATMENT_ARM] = treatment_payload
        saved = publication.write_torch(
            output, "fit/checkpoint.pt",
            _canonical_payload(treatment_payload), meter=meter,
            on_write=count_checkpoint_write)
        checkpoint_records.append({"role": TREATMENT_ARM, "arm_id": TREATMENT_ARM,
                                   **saved.as_dict()})
        state.save(active_arm_id=None, status="publication_checkpoint",
                   check_budget=False)
        # Explicit zero-update baselines: the untouched parent already solves the
        # synthetic 22-class exact-address fit cases, so fit success is
        # integration evidence and is not presented as learning improvement.
        parent = SelectedExpertReadout(
            seed=int(context.config["transaction"]["seed"]))
        parent_rows = _predict_split(
            TREATMENT_ARM, parent, split="fit", events=metadata["events"],
            cases=list(metadata["cases"].values()), vocabulary=vocabulary,
            update_digest=parent.state_digest(), work=work)
        state.save(active_arm_id=None, status="publication_baseline",
                   check_budget=False)
        reset_rows = _predict_split(
            TREATMENT_ARM, parent, split="fit", events=metadata["events"],
            cases=list(metadata["cases"].values()), vocabulary=vocabulary,
            update_digest=parent.state_digest(),
            reset_memory_before_queries=True, work=work)
        state.save(active_arm_id=None, status="publication_baseline",
                   check_budget=False)
        parent_baseline = _zero_update_baseline(
            name="zero_update_parent", rows=parent_rows, module=parent,
            targets=targets, config=context.config, split="fit",
            memory_semantics="chronological_fixed_record_view",
            root=output, relative="fit/baseline_parent_predictions.jsonl",
            meter=meter)
        reset_baseline = _zero_update_baseline(
            name="zero_update_reset", rows=reset_rows, module=parent,
            targets=targets, config=context.config, split="fit",
            memory_semantics="reset_before_every_query",
            root=output, relative="fit/baseline_reset_predictions.jsonl",
            meter=meter)
        for role, relative in (
                ("baseline_parent", "fit/baseline_parent_predictions.jsonl"),
                ("baseline_reset", "fit/baseline_reset_predictions.jsonl")):
            artifact = publication.Artifact.of(output, relative)
            prediction_records.append({"role": role, "arm_id": None,
                                       **artifact.as_dict()})
        # Replace the heavy training snapshots with the final checkpoints they
        # were built from, but only after each final checkpoint is durable and
        # verified by content identity.  The pointer stays until measurement.
        bindings = {}
        for arm_id in TRAINED_ARMS:
            payload = checkpoint_payloads[arm_id]
            relative = ("fit/checkpoint.pt" if arm_id == TREATMENT_ARM
                        else f"fit/control_{arm_id}.pt")
            bindings[arm_id] = {
                "path": relative,
                "sha256": publication.Artifact.of(output, relative).sha256,
                "state_sha256": payload["state_sha256"],
                "model_state_sha256": payload["model_state_sha256"],
                "fixed_state_sha256": payload["fixed_state_sha256"],
                "completed_updates": int(payload["completed_steps"]),
                "event_clock_steps": int(payload["event_clock_steps"]),
                "events_processed": int(payload["events_processed"]),
                "update_records_sha256": sha256_json(
                    payload["update_records"]),
                "predictions_sha256": payload["predictions_sha256"],
            }
        state.bind_final_recovery(bindings=bindings)
        # The publication marker is recorded before the pointer is written, so a
        # crash after the durable save can never repeat the counted publication
        # stage; the verified duplicates are released immediately afterwards.
        state.mark_publication(
            artifacts={"predictions": prediction_records,
                       "checkpoints": checkpoint_records},
            live_peak=meter.report())
        state.save(active_arm_id=None, status="publication_ready",
                   check_budget=False)
        state.release_resume_snapshots()
        state.save(active_arm_id=None, status="publication_complete",
                   check_budget=False)
        final_checkpoint_sha256 = {
            arm_id: bindings[arm_id]["sha256"] for arm_id in TRAINED_ARMS}
    final_state_by_arm = {
        arm_id: {"state_sha256": arms[arm_id].module.state_digest(),
                 "model_state_sha256": arms[arm_id].module.model_state_digest(),
                 "fixed_state_sha256": arms[arm_id].module.fixed_state_digest()}
        for arm_id in TRAINED_ARMS}
    cumulative = _sum_work(state.processes)
    _require(cumulative["optimizer_steps"] == 72,
             "cumulative physical optimizer work differs from 72")
    _require(cumulative["backward_calls"] == 72
             and cumulative["training_forward_calls"] == 72,
             "cumulative physical training work differs from one pass per update")
    treatment_accuracy = _case_accuracy(
        arm_reports[TREATMENT_ARM])
    parent_accuracy = parent_baseline["case_accuracy"]
    fit_report = {
        "format": FIT_REPORT_FORMAT,
        "protocol_id": "value_readout_abstention_v1",
        "mode": "fit",
        "created_by": RUNNER_ENTRY_POINT,
        "status": "passed" if arm_reports[TREATMENT_ARM][
            "all_declared_gates_passed"] else "failed",
        "config_sha256": context.config_sha256,
        "prepared_manifest_sha256": context.manifest_sha256,
        "seed": int(context.config["transaction"]["seed"]),
        "arm_id": TREATMENT_ARM,
        "phase": "fit_treatment",
        "optimizer": "NativeFusedAdamW",
        "trainable_parameters": list(TRAINABLE_NAMES),
        "losses": ["future_answer_readout", "correction_readout",
                   "unknown_abstention_readout", "wrong_world_abstention_readout"],
        "train_case_count": 15,
        "replay_case_count": 12,
        "completed_steps": treatment.completed_updates,
        "event_clock_steps": treatment.clock.eligible_events,
        "events_processed": len(metadata["events"]),
        "parent_state_sha256": treatment.parent_state_sha256,
        "resulting_state_sha256": treatment.module.state_digest(),
        "model_state_sha256": treatment.module.model_state_digest(),
        "fixed_state_sha256": treatment.module.fixed_state_digest(),
        "optimizer_meta": _optimizer_meta(treatment.module, treatment.optimizer),
        "update_records": deepcopy(treatment.update_records),
        "per_case": arm_reports[TREATMENT_ARM]["per_case"],
        "metrics": arm_reports[TREATMENT_ARM]["metrics"],
        "thresholds": arm_reports[TREATMENT_ARM]["thresholds"],
        "all_declared_gates_passed": arm_reports[TREATMENT_ARM][
            "all_declared_gates_passed"],
        "predictions_sha256": sha256_file(fit_dir / "fit_predictions.jsonl"),
        "control_fits": [
            {"arm_id": row["arm_id"], "phase": row["phase"],
             "optimizer_steps": row["optimizer_steps"],
             "checkpoint_sha256": sha256_file(
                 fit_dir / f"control_{row['arm_id']}.pt"),
             "fit_receipt_binding": False,
             "predictions_sha256": row["predictions_sha256"]}
            for row in arm_reports.values() if row["arm_id"] != TREATMENT_ARM
        ],
        "zero_update_baselines": {
            "parent": parent_baseline,
            "reset": reset_baseline,
            "integration_vs_learning": {
                "integration_success": arm_reports[TREATMENT_ARM][
                    "all_declared_gates_passed"],
                "treatment_case_accuracy": treatment_accuracy,
                "parent_case_accuracy": parent_accuracy,
                "reset_case_accuracy": reset_baseline["case_accuracy"],
                "parent_already_passes_all_fit_gates": parent_baseline[
                    "all_declared_gates_passed"],
                "case_accuracy_improvement_over_parent": (
                    treatment_accuracy - parent_accuracy),
                "learning_improvement_claimed": False,
                "reason": (
                    "The untouched zero-update parent already achieves 27/27 "
                    "on the synthetic exact-address fit fixture; passing fit "
                    "gates establishes software integration only and is not "
                    "evidence of learning improvement."),
            },
        },
        "physical_work": {
            "scope": "fit_treatment_and_two_control_arms_plus_zero_update_baselines",
            "optimizer_steps": cumulative["optimizer_steps"],
            "backward_calls": cumulative["backward_calls"],
            "training_forward_calls": cumulative["training_forward_calls"],
            "evaluation_forward_calls": cumulative["evaluation_forward_calls"],
            "retention_diagnostic_forwards": cumulative[
                "retention_diagnostic_forwards"],
            "training_write_events": cumulative["training_write_events"],
            "evaluation_write_events": cumulative["evaluation_write_events"],
            "checkpoint_writes": cumulative["checkpoint_writes"],
            "matches_global_logical_updates": (
                cumulative["optimizer_steps"]
                == sum(row["optimizer_steps"]
                       for row in arm_reports.values())),
            "process_history": "resume_evidence.json",
        },
        "resume_semantics": {
            "format": RESUME_FORMAT,
            "arm_order": list(TRAINED_ARMS),
            "global_arm_cursor": (
                "cursor.next_arm_index is the first non-complete arm; "
                "arm_order[:next_arm_index] are durable completed-arm states"),
            "completed_arms_never_retrained": True,
            "active_state": (
                "active_arm persists model, NativeFusedAdamW state, event "
                "clock, update records, dedup ledger and fixed-memory records"),
            "clocks": (
                "each arm owns one independent clock equal to its processed "
                "event cursor; event clocks advance once per event"),
            "memory": "per-arm fixed record memory snapshot",
            "completed_arm_state": (
                "resume_arms/<arm_id>.pt with content hash in resume.pt"),
            "identities": dict(state.identities),
            "process_evidence": "resume_evidence.json",
            "heavy_resume_artifacts_removed_after_completion": True,
            "artifact_cap_bytes": int(
                context.config["budget"]["authorized_run_ceiling"][
                    "artifact_bytes_max"]),
        },
        "classification": {
            "native_model": False,
            "tokenizer": False,
            "downloads": 0,
            "device": "cpu",
            "scope": "deterministic_software_integration",
            "capability_claim": False,
        },
        "dataset_audit": runtime["dataset_audit"],
        "label_access": {
            "dev_parsed": False,
            "heldout_parsed": False,
            "opened_after_admission": False,
        },
        "model_evaluation": {
            "performed": True,
            "arm_ids": list(TRAINED_ARMS),
            "kind": "preserved_fit_predictions",
        },
        "limitations": [
            "This CPU runner is a deterministic software integration; its synthetic features carry the exact-address selected record and do not establish learned retrieval.",
            "No open-ended generation, write-through meta-gradient, native model or general persistent learning is claimed.",
            "Fit and control arms share one deterministic parent but own independent parameters, optimizers and clocks; only the treatment's 24 updates bind a fit receipt.",
            "Exact context is an ungated quality/cost reference and is not required to be beaten.",
        ],
        "artifact_budget": {
            "cap_bytes": context.config["budget"]["authorized_run_ceiling"][
                "artifact_bytes_max"],
            "old_cap_bytes": context.config["budget"]["authorized_run_ceiling"][
                "artifact_budget"]["old_cap_bytes"],
            "old_cap_underprovisioned": context.config["budget"][
                "authorized_run_ceiling"]["artifact_budget"][
                    "old_cap_underprovisioned"],
            "resume_complete_checkpoints": context.config["budget"][
                "authorized_run_ceiling"]["artifact_budget"][
                    "resume_complete_checkpoints"],
            "derivation": context.config["budget"]["authorized_run_ceiling"][
                "artifact_budget"]["derivation"],
            "enforcement": context.config["budget"]["authorized_run_ceiling"][
                "artifact_budget"]["enforcement"],
        },
        "runtime_precheck": runtime,
    }
    _write_json_atomic(fit_dir / "fit_report.json", fit_report, meter=meter)
    artifacts = {
        "checkpoint_sha256": sha256_file(fit_dir / "checkpoint.pt"),
        "fit_report_sha256": sha256_file(fit_dir / "fit_report.json"),
        "fit_predictions_sha256": sha256_file(fit_dir / "fit_predictions.jsonl"),
    }
    from value_readout_abstention_contract_v1 import seal_receipt
    receipt_body = {
        "format": "replica.value_readout_abstention.fit_receipt.v1",
        "protocol_id": "value_readout_abstention_v1",
        "status": "passed" if fit_report["all_declared_gates_passed"] else "failed",
        "mode": "fit",
        "created_by": RUNNER_ENTRY_POINT,
        "config_sha256": context.config_sha256,
        "prepared_manifest_sha256": context.manifest_sha256,
        "seed": int(context.config["transaction"]["seed"]),
        "optimizer": "NativeFusedAdamW",
        "train_case_count": 15,
        "replay_case_count": 12,
        "optimizer_steps": treatment.completed_updates,
        "losses": ["future_answer_readout", "correction_readout",
                   "unknown_abstention_readout", "wrong_world_abstention_readout"],
        "trainable_parameters": list(TRAINABLE_NAMES),
        "earlier_write_gradient": False,
        "heldout_opened": False,
        "dev_labels_read": False,
        "tuning_performed": False,
        "artifacts": artifacts,
        "fit_gates": {
            "all_declared_gates_passed": fit_report[
                "all_declared_gates_passed"],
            "local_associative_loss_gate": False,
            "metrics": fit_report["metrics"],
            "thresholds": fit_report["thresholds"],
        },
    }
    receipt = seal_receipt(receipt_body)
    _write_json_atomic(output / "fit_receipt.json", receipt, meter=meter)
    # Resume evidence is published before the budget measurement so the complete
    # tree, including retained recovery evidence, is measured with every
    # artifact present.  The pointer is released only after that measurement.
    resume_evidence_path = state.write_evidence(
        final_state_by_arm=final_state_by_arm,
        final_checkpoint_sha256=final_checkpoint_sha256, meter=meter)
    budget = enforce_artifact_budget(
        output, context.config, context="fit",
        live_peak=state.publication_live_peak,
        release_after_measurement=[resume_path])
    return {"status": fit_report["status"], "output": str(output),
            "fit_evidence": str(fit_dir), "receipt": receipt,
            "fit_report": fit_report, "runtime_precheck": runtime,
            "resume_evidence": str(resume_evidence_path),
            "physical_work": fit_report["physical_work"],
            "artifact_budget": budget}


def _load_arm_from_checkpoint(arm_id: str, path: Path,
                              config: Mapping[str, Any], seed: int) -> Arm:
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    _require(payload["format"] == READOUT_CHECKPOINT_FORMAT,
             "checkpoint format differs")
    _require(payload["arm_id"] == arm_id, "checkpoint arm differs")
    arm = Arm(arm_id, config, seed)
    _require(int(payload["completed_steps"]) == 24,
             "checkpoint is not a completed 24-step arm")
    with torch.no_grad():
        for name, tensor in payload["model_state"].items():
            arm.module.model_state()[name].copy_(tensor)
    _restore_optimizer(arm.module, arm.optimizer, payload["optimizer_state"])
    arm.completed_updates = int(payload["completed_steps"])
    arm.clock = TeachingClock(int(payload["event_clock_steps"]))
    arm.update_records = deepcopy(list(payload["update_records"]))
    _require(arm.module.state_digest() == payload["state_sha256"],
             "restored checkpoint state differs")
    _require(len(arm.update_records) == arm.completed_updates,
             "restored checkpoint update record count differs")
    _require(arm.update_records[-1].get("state_after_sha256")
             == payload["state_sha256"],
             "restored checkpoint final update hash differs from its state")
    _require(int(payload["events_processed"]) == 36
             and int(payload["event_clock_steps"]) == 36,
             "restored checkpoint clock/event count differs")
    return arm


def _admit_fit_or_raise(fit_dir: Path, prepared: Path,
                        config_path: Path) -> dict:
    import sys
    scripts = Path(__file__).resolve().parents[2] / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    from value_readout_abstention_contract_v1 import admit_fit_receipt, read_json
    receipt = read_json(Path(fit_dir) / "fit_receipt.json")
    return admit_fit_receipt(
        receipt, prepared, config_path,
        evidence_dir=Path(fit_dir) / "fit")


def _read_eval_targets(split: str, vault, admission: Mapping[str, Any]) -> list[dict]:
    # Labels are opened only through the vault after a validated admission.
    return vault.open_evaluation(split, admission=admission)


def _durable_stage_report(output: Path, *, stage: str,
                          report_relative: str, predictions_relative: str,
                          identity: Mapping[str, Any]) -> dict | None:
    """Return an already-durable stage report, or ``None`` for a fresh stage.

    Recovery from a durable report never reopens evaluation labels and never
    recomputes a prediction: the caller only re-seals the missing receipt and
    re-runs the artifact measurement.
    """

    root = Path(output)
    report_path = root / report_relative
    if not report_path.is_file():
        return None
    report = read_json(report_path)
    _require(isinstance(report, Mapping) and bool(report),
             f"{stage} report is unreadable")
    for key, value in identity.items():
        _require(report.get(key) == value,
                 f"{stage} report identity differs: {key}")
    predictions = root / predictions_relative
    _require(predictions.is_file(), f"{stage} predictions are missing")
    if "predictions_sha256" in report:
        _require(sha256_file(predictions) == report["predictions_sha256"],
                 f"{stage} predictions changed")
    return dict(report)


def _stage_output(output: Path, predictions_relative: str) -> Path:
    """Allow recovery only beside an existing partial prediction artifact."""

    output = Path(output)
    if (output / predictions_relative).is_file():
        output.mkdir(parents=True, exist_ok=True)
        return output
    return _fresh_output(output)


def _write_stage_predictions(output: Path, relative: str,
                             rows: Sequence[Mapping[str, Any]], *,
                             meter: publication.LiveByteMeter) -> None:
    path = Path(output) / relative
    expected = [dict(row) for row in rows]
    if path.is_file():
        _require(read_jsonl(path) == expected,
                 f"existing stage predictions differ: {relative}")
        return
    _write_jsonl(path, expected, meter=meter)


def run_development(prepared: Path, config_path: Path, fit_dir: Path,
                    output: Path) -> dict:
    prepared = Path(prepared)
    fit_dir = Path(fit_dir)
    context = _context(config_path, prepared)
    admission = _admit_fit_or_raise(fit_dir, prepared, config_path)
    from value_readout_abstention_contract_v1 import (
        LabelVault, read_json, read_jsonl, seal_receipt, sha256_file)
    output = Path(output)
    meter = publication.LiveByteMeter(output, label="development_publication")
    meter.observe()
    checkpoint_path = fit_dir / "fit" / "checkpoint.pt"
    fit_receipt = read_json(fit_dir / "fit_receipt.json")
    durable = _durable_stage_report(
        output, stage="development", report_relative="dev/dev_report.json",
        predictions_relative="dev/dev_predictions.jsonl",
        identity={
            "fit_receipt_sha256": fit_receipt["receipt_sha256"],
            "config_sha256": context.config_sha256,
            "prepared_manifest_sha256": context.manifest_sha256,
            "split": "dev",
            "mode": "dev-eval",
            "checkpoint_sha256": sha256_file(checkpoint_path),
        })
    dev_dir = output / "dev"
    if durable is not None:
        # Republish only the receipt from the durable report; evaluation labels
        # are not reopened and no prediction is recomputed.
        report = durable
        passed = bool(report.get("all_declared_gates_passed"))
        dev_dir.mkdir(parents=True, exist_ok=True)
    else:
        vault = LabelVault(prepared, context.config)
        labels = _read_eval_targets("dev", vault, admission)
        targets = {label["case_id"]: {
            "case_type": label["case_type"], "target": label["target"]["token"],
            "stale_value": label.get("stale_value")} for label in labels}
        split = _read_split_inputs(prepared, context.config, "dev")
        arm = _load_arm_from_checkpoint(
            TREATMENT_ARM, checkpoint_path, context.config,
            int(context.config["transaction"]["seed"]))
        rows = _predict_split(
            TREATMENT_ARM, arm.module, split="dev", events=split["events"],
            cases=split["cases"], vocabulary=list(
                context.config["architecture"]["answer_vocabulary"]),
            update_digest=arm.module.state_digest())
        _require({row["case_id"] for row in rows} == set(targets),
                 "development prediction coverage differs")
        metrics = _metrics_from_predictions(rows, targets)
        thresholds, passed = _gates_from_metrics(context.config, "dev",
                                                 metrics["metrics"])
        output = _stage_output(output, "dev/dev_predictions.jsonl")
        dev_dir = output / "dev"
        dev_dir.mkdir(parents=True, exist_ok=True)
        _write_stage_predictions(
            output, "dev/dev_predictions.jsonl", rows, meter=meter)
        report = {
            "format": EVALUATION_REPORT_FORMAT,
            "protocol_id": "value_readout_abstention_v1",
            "split": "dev",
            "mode": "dev-eval",
            "created_by": RUNNER_ENTRY_POINT,
            "status": "passed" if passed else "failed",
            "config_sha256": context.config_sha256,
            "prepared_manifest_sha256": context.manifest_sha256,
            "fit_receipt_sha256": fit_receipt["receipt_sha256"],
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "model_state_sha256": arm.module.state_digest(),
            "predictions_sha256": sha256_file(
                dev_dir / "dev_predictions.jsonl"),
            "cases": len(rows),
            "per_case": metrics["per_case"],
            "metrics": metrics["metrics"],
            "thresholds": thresholds,
            "all_declared_gates_passed": bool(passed),
            "optimizer_steps": 0,
            "gradient": False,
            "label_access": {"dev_parsed": True, "heldout_parsed": False,
                             "after_fit_admission": True},
            "dataset_audit": {"mode": "runtime_hash_precheck",
                              "performed": True,
                              "evaluation_labels_parsed": False},
            "model_evaluation": {"performed": True,
                                 "arm_ids": [TREATMENT_ARM],
                                 "kind": "preserved_development_predictions"},
            "classification": {"native_model": False, "tokenizer": False,
                               "downloads": 0, "device": "cpu"},
        }
        _write_json_atomic(dev_dir / "dev_report.json", report, meter=meter)
    body = {
        "format": "replica.value_readout_abstention.development_receipt.v1",
        "protocol_id": "value_readout_abstention_v1",
        "status": "passed" if passed else "failed",
        "mode": "dev-eval",
        "created_by": RUNNER_ENTRY_POINT,
        "attempt": 1,
        "fit_receipt_sha256": fit_receipt["receipt_sha256"],
        "config_sha256": context.config_sha256,
        "prepared_manifest_sha256": context.manifest_sha256,
        "heldout_opened": False,
        "tuning_performed": False,
        "all_declared_gates_passed": bool(passed),
        "dev_metrics": report["metrics"],
        "artifacts": {
            "predictions_sha256": sha256_file(dev_dir / "dev_predictions.jsonl"),
            "report_sha256": sha256_file(dev_dir / "dev_report.json"),
        },
    }
    receipt = seal_receipt(body)
    _write_json_atomic(output / "dev_receipt.json", receipt, meter=meter)
    budget = enforce_artifact_budget(output, context.config,
                                     context="development",
                                     live_peak=meter.report())
    return {"status": report["status"], "output": str(output),
            "dev_evidence": str(dev_dir), "receipt": receipt,
            "report": report, "fit_admission": admission,
            "artifact_budget": budget}


def run_heldout(prepared: Path, config_path: Path, fit_dir: Path,
                dev_dir: Path, output: Path) -> dict:
    prepared = Path(prepared)
    fit_dir = Path(fit_dir)
    dev_dir = Path(dev_dir)
    context = _context(config_path, prepared)
    from value_readout_abstention_contract_v1 import (
        LabelVault, admit_development, read_json, read_jsonl, seal_receipt,
        sha256_file)
    fit_receipt = read_json(fit_dir / "fit_receipt.json")
    dev_receipt = read_json(dev_dir / "dev_receipt.json")
    admission = admit_development(
        dev_receipt, fit_receipt=fit_receipt, prepared_dir=prepared,
        config_path=config_path, evidence_dir=dev_dir / "dev",
        fit_evidence_dir=fit_dir / "fit")
    output = Path(output)
    meter = publication.LiveByteMeter(output, label="heldout_publication")
    meter.observe()
    checkpoint_path = fit_dir / "fit" / "checkpoint.pt"
    durable = _durable_stage_report(
        output, stage="heldout",
        report_relative="heldout/heldout_report.json",
        predictions_relative="heldout/heldout_predictions.jsonl",
        identity={
            "fit_receipt_sha256": fit_receipt["receipt_sha256"],
            "dev_receipt_sha256": dev_receipt["receipt_sha256"],
            "config_sha256": context.config_sha256,
            "prepared_manifest_sha256": context.manifest_sha256,
            "split": "heldout",
            "mode": "heldout-eval",
            "checkpoint_sha256": sha256_file(checkpoint_path),
        })
    heldout_dir = output / "heldout"
    if durable is not None:
        # Republish only the receipt.  Held-out labels are not reopened: the
        # durable report and its verified predictions are preserved evidence.
        report = durable
        passed = bool(report.get("all_declared_gates_passed"))
        heldout_dir.mkdir(parents=True, exist_ok=True)
    else:
        vault = LabelVault(prepared, context.config)
        labels = vault.open_evaluation("heldout", admission=admission)
        targets = {label["case_id"]: {
            "case_type": label["case_type"], "target": label["target"]["token"],
            "stale_value": label.get("stale_value")} for label in labels}
        split = _read_split_inputs(prepared, context.config, "heldout")
        arm = _load_arm_from_checkpoint(
            TREATMENT_ARM, checkpoint_path, context.config,
            int(context.config["transaction"]["seed"]))
        rows = _predict_split(
            TREATMENT_ARM, arm.module, split="heldout", events=split["events"],
            cases=split["cases"], vocabulary=list(
                context.config["architecture"]["answer_vocabulary"]),
            update_digest=arm.module.state_digest())
        _require({row["case_id"] for row in rows} == set(targets),
                 "heldout prediction coverage differs")
        metrics = _metrics_from_predictions(rows, targets)
        thresholds, passed = _gates_from_metrics(context.config, "dev",
                                                 metrics["metrics"])
        output = _stage_output(output, "heldout/heldout_predictions.jsonl")
        heldout_dir = output / "heldout"
        heldout_dir.mkdir(parents=True, exist_ok=True)
        _write_stage_predictions(
            output, "heldout/heldout_predictions.jsonl", rows, meter=meter)
        report = {
            "format": EVALUATION_REPORT_FORMAT,
            "protocol_id": "value_readout_abstention_v1",
            "split": "heldout",
            "mode": "heldout-eval",
            "created_by": RUNNER_ENTRY_POINT,
            "status": "passed" if passed else "failed",
            "config_sha256": context.config_sha256,
            "prepared_manifest_sha256": context.manifest_sha256,
            "fit_receipt_sha256": fit_receipt["receipt_sha256"],
            "dev_receipt_sha256": dev_receipt["receipt_sha256"],
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "model_state_sha256": arm.module.state_digest(),
            "predictions_sha256": sha256_file(
                heldout_dir / "heldout_predictions.jsonl"),
            "cases": len(rows),
            "per_case": metrics["per_case"],
            "metrics": metrics["metrics"],
            "thresholds": thresholds,
            "all_declared_gates_passed": bool(passed),
            "optimizer_steps": 0,
            "gradient": False,
            "label_access": {"dev_parsed": False, "heldout_parsed": True,
                             "after_development_admission": True},
            "dataset_audit": {"mode": "runtime_hash_precheck",
                              "performed": True,
                              "evaluation_labels_parsed": False},
            "model_evaluation": {"performed": True,
                                 "arm_ids": [TREATMENT_ARM],
                                 "kind": "preserved_heldout_predictions"},
            "classification": {"native_model": False, "tokenizer": False,
                               "downloads": 0, "device": "cpu"},
        }
        _write_json_atomic(heldout_dir / "heldout_report.json", report,
                           meter=meter)
    body = {
        "format": "replica.value_readout_abstention.heldout_receipt.v1",
        "protocol_id": "value_readout_abstention_v1",
        "status": "passed" if passed else "failed",
        "mode": "heldout-eval",
        "created_by": RUNNER_ENTRY_POINT,
        "attempt": 1,
        "dev_receipt_sha256": dev_receipt["receipt_sha256"],
        "config_sha256": context.config_sha256,
        "prepared_manifest_sha256": context.manifest_sha256,
        "tuning_performed": False,
        "heldout_metrics": report["metrics"],
        "artifacts": {
            "predictions_sha256": sha256_file(
                heldout_dir / "heldout_predictions.jsonl"),
            "report_sha256": sha256_file(
                heldout_dir / "heldout_report.json"),
        },
    }
    receipt = seal_receipt(body)
    _write_json_atomic(output / "heldout_receipt.json", receipt, meter=meter)
    budget = enforce_artifact_budget(output, context.config, context="heldout",
                                     live_peak=meter.report())
    return {"status": report["status"], "output": str(output),
            "heldout_evidence": str(heldout_dir), "receipt": receipt,
            "report": report, "development_admission": admission,
            "artifact_budget": budget}


def run_controls(prepared: Path, config_path: Path, fit_dir: Path,
                 output: Path, *, split: str = "dev",
                 dev_dir: Path | None = None) -> dict:
    prepared = Path(prepared)
    fit_dir = Path(fit_dir)
    context = _context(config_path, prepared)
    from value_readout_abstention_contract_v1 import (
        LabelVault, admit_development, read_json, seal_receipt, sha256_file)
    if split == "heldout":
        _require(dev_dir is not None,
                 "held-out controls require a development evidence directory")
        fit_receipt = read_json(fit_dir / "fit_receipt.json")
        dev_receipt = read_json(Path(dev_dir) / "dev_receipt.json")
        admission = admit_development(
            dev_receipt, fit_receipt=fit_receipt, prepared_dir=prepared,
            config_path=config_path, evidence_dir=Path(dev_dir) / "dev",
            fit_evidence_dir=fit_dir / "fit")
    else:
        admission = _admit_fit_or_raise(fit_dir, prepared, config_path)
    output = Path(output)
    meter = publication.LiveByteMeter(
        output, label=f"controls_{split}_publication")
    meter.observe()
    controls_dir = output / "controls"
    report_relative = f"controls/{split}_control_report.json"
    predictions_relative = f"controls/{split}_control_predictions.jsonl"
    durable = None
    if (output / report_relative).is_file():
        candidate = read_json(output / report_relative)
        _require(candidate.get("format") == CONTROL_REPORT_FORMAT
                 and candidate.get("split") == split
                 and candidate.get("fit_admission_sha256")
                 == admission["fit_receipt_sha256"]
                 and set(candidate.get("arms", {})) == set(ARM_IDS),
                 "control report identity differs")
        rows = read_jsonl(output / predictions_relative)
        for arm_id in ARM_IDS:
            _require(sha256_json([row for row in rows
                                  if row["arm_id"] == arm_id])
                     == candidate["arms"][arm_id]["predictions_sha256"],
                     f"control predictions changed: {arm_id}")
        durable = candidate
    if durable is not None:
        # Republish nothing but the measurement from the durable report; the
        # evaluation split is not reopened.
        controls_dir.mkdir(parents=True, exist_ok=True)
        budget = enforce_artifact_budget(output, context.config,
                                         context=f"controls_{split}",
                                         live_peak=meter.report())
        return _controls_result(durable, output, admission, budget)
    vault = LabelVault(prepared, context.config)
    labels = vault.open_evaluation(split, admission=admission)
    targets = {label["case_id"]: {
        "case_type": label["case_type"], "target": label["target"]["token"],
        "stale_value": label.get("stale_value")} for label in labels}
    split_inputs = _read_split_inputs(prepared, context.config, split)
    vocabulary = list(context.config["architecture"]["answer_vocabulary"])
    checkpoint_path = fit_dir / "fit" / "checkpoint.pt"
    treatment = _load_arm_from_checkpoint(
        TREATMENT_ARM, checkpoint_path, context.config,
        int(context.config["transaction"]["seed"]))
    arms = {TREATMENT_ARM: treatment}
    for arm_id in ("shuffled_values", "wrong_world_values"):
        arms[arm_id] = _load_arm_from_checkpoint(
            arm_id, fit_dir / "fit" / f"control_{arm_id}.pt",
            context.config, int(context.config["transaction"]["seed"]))
    predictions = {}
    for arm_id, arm in arms.items():
        predictions[arm_id] = _predict_split(
            arm_id, arm.module, split=split, events=split_inputs["events"],
            cases=split_inputs["cases"], vocabulary=vocabulary,
            update_digest=arm.module.state_digest())
    # Non-neural deterministic views.
    memory = FixedRecordMemory()
    disabled = []
    reset = []
    literal = []
    exact = []
    for event in split_inputs["events"]:
        if event["kind"] != "query":
            memory.write(event)
            continue
        case = split_inputs["cases_by_id"][event["case_id"]]
        record = memory.resolve(event["address"], event["world_id"],
                                int(event["epoch"]))
        base_index = (0 if record is None else vocabulary.index(record["value"]))
        token = vocabulary[base_index]
        row_base = {
            "format": PREDICTION_FORMAT,
            "protocol_id": "value_readout_abstention_v1",
            "split": split,
            "case_id": case["case_id"],
            "event_id": event["event_id"],
            "world_id": case["world_id"],
            "epoch": int(case["epoch"]),
            "case_type": case["case_type"],
            "predicted_index": base_index,
            "predicted_token": token,
            "model_state_sha256": "0" * 64,
            "memory_state_sha256": memory.digest(),
            "clock_step": event["clock_step"],
            "context_tokens": (len(str(case["query_text"]).split())
                               + 2 * len(memory.visible_records())),
        }
        for arm_id, bucket in (("memory_disabled", disabled),
                               ("memory_reset", reset),
                               ("selected_record_literal_readout", literal),
                               ("exact_context", exact)):
            row = dict(row_base)
            row["arm_id"] = arm_id
            if arm_id in ("memory_disabled", "memory_reset"):
                row["predicted_index"] = 0
                row["predicted_token"] = "UNKNOWN"
            bucket.append(row)
    for arm_id, rows in (("memory_disabled", disabled),
                         ("memory_reset", reset),
                         ("selected_record_literal_readout", literal),
                         ("exact_context", exact)):
        predictions[arm_id] = sorted(rows, key=lambda row: row["case_id"])
    results = {}
    for arm_id in ARM_IDS:
        rows = predictions[arm_id]
        metrics = _metrics_from_predictions(rows, targets)
        non_abstain = sum(1 for row in rows
                          if row["predicted_token"] != "UNKNOWN") / len(rows)
        correct = sum(entry["correct"]
                      for entry in metrics["per_case"]["counts"].values())
        entry = {
            "arm_id": arm_id,
            "cases": len(rows),
            "per_case": metrics["per_case"],
            "metrics": metrics["metrics"],
            "case_accuracy": correct / len(rows),
            "non_abstain_rate": non_abstain,
            "predictions_sha256": sha256_json(rows),
        }
        if arm_id == "exact_context":
            entry["quality_metric"] = "case_accuracy"
            entry["quality"] = entry["case_accuracy"]
            entry["cost_metric"] = "context_tokens_per_query"
            entry["context_tokens_per_query"] = sum(
                row["context_tokens"] for row in rows) / len(rows)
        if arm_id == "selected_record_literal_readout":
            entry["diagnostic_only"] = True
        if arm_id in ("memory_disabled", "memory_reset"):
            entry["gate"] = "must_abstain_all"
            entry["gate_passed"] = non_abstain == 0.0
        if arm_id in ("shuffled_values", "wrong_world_values"):
            entry["gate"] = "must_not_pass_primary_gates"
            entry["max_any_case_accuracy"] = max(
                metrics["metrics"][key] for key in metrics["metrics"]
                if key.endswith("_accuracy"))
            entry["gate_passed"] = entry["max_any_case_accuracy"] <= 0.5
        results[arm_id] = entry
    output = _stage_output(
        output, f"controls/{split}_control_predictions.jsonl")
    controls_dir = output / "controls"
    controls_dir.mkdir(parents=True, exist_ok=True)
    _write_stage_predictions(
        output, f"controls/{split}_control_predictions.jsonl",
        [row for arm_id in ARM_IDS for row in predictions[arm_id]], meter=meter)
    report = {
        "format": CONTROL_REPORT_FORMAT,
        "protocol_id": "value_readout_abstention_v1",
        "split": split,
        "fit_receipt_sha256": read_json(fit_dir / "fit_receipt.json")[
            "receipt_sha256"],
        "fit_admission_sha256": admission["fit_receipt_sha256"],
        "arms": results,
        "distinct_arms": len(results) == 7,
        "treatment_required_to_beat_reference": False,
        "dataset_audit": {"mode": "runtime_hash_precheck",
                          "performed": True,
                          "evaluation_labels_parsed": False},
        "label_access": {"split": split, "opened": True,
                         "after_fit_admission": True},
        "model_evaluation": {"performed": True, "arm_ids": list(ARM_IDS),
                             "kind": "preserved_control_predictions"},
        "classification": {"native_model": False, "tokenizer": False,
                           "downloads": 0, "device": "cpu"},
    }
    _write_json_atomic(controls_dir / f"{split}_control_report.json", report,
                       sort_keys=False, meter=meter)
    budget = enforce_artifact_budget(output, context.config,
                                     context=f"controls_{split}",
                                     live_peak=meter.report())
    return _controls_result(report, output, admission, budget)


def _controls_result(report: Mapping[str, Any], output: Path,
                     admission: Mapping[str, Any],
                     budget: Mapping[str, Any]) -> dict:
    results = report["arms"]
    return {"status": "passed" if all(
        arm.get("gate_passed", True) for arm in results.values()) else "failed",
        "output": str(output), "report": report, "admission": admission,
        "artifact_budget": budget}

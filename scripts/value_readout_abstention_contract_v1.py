"""Deterministic fixture and fail-closed validation for value-readout abstention.

This module is preparation-only.  It builds chronological synthetic episodes,
sealed development/heldout labels, disjoint training replay cases, control-arm
rows and a manifest; it validates those artifacts with the Python standard
library only.  No model, tokenizer, optimizer, gradient, checkpoint or native
job is constructed or run.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "value_readout_abstention_protocol_v1.json"
PREPARED_DIR = ROOT / "data" / "value_readout_abstention_20261027" / "fixture01"

CONFIG_FORMAT = "replica.value_readout_abstention.protocol.v1"
MANIFEST_FORMAT = "replica.value_readout_abstention.prepared.v1"
EVENT_FORMAT = "replica.value_readout_abstention.event.v1"
CASE_FORMAT = "replica.value_readout_abstention.case.v1"
LABEL_FORMAT = "replica.value_readout_abstention.label.v1"
CONTROL_FORMAT = "replica.value_readout_abstention.control.v1"
VALIDATION_FORMAT = "replica.value_readout_abstention.validation.v1"
PREFLIGHT_FORMAT = "replica.value_readout_abstention.preflight_receipt.v1"
FIT_RECEIPT_FORMAT = "replica.value_readout_abstention.fit_receipt.v1"
DEVELOPMENT_RECEIPT_FORMAT = "replica.value_readout_abstention.development_receipt.v1"
HELDOUT_RECEIPT_FORMAT = "replica.value_readout_abstention.heldout_receipt.v1"
AUTHORIZATION_FORMAT = "replica.value_readout_abstention.authorization.v1"

PROTOCOL_ID = "value_readout_abstention_v1"
SEED = 20261027
GUARD_BYTES = {
    "peak_driver_bytes_max": 53687091200,
    "sustained_swap_growth_bytes_max": 536870912,
    "min_host_free_fraction": 0.15,
}

SPLIT_WORLDS = {
    "train": ("train_alpha", "train_beta", "train_gamma"),
    "dev": ("dev_alpha", "dev_beta"),
    "heldout": ("heldout_alpha", "heldout_beta"),
}
SPLIT_EXPECTED = {"train": 3, "dev": 2, "heldout": 2}
BASE_EPOCHS = 8
WRITES_PER_EPISODE = 3
TEACHING_CASE_TYPES = (
    "future_answer",
    "correction",
    "unknown_abstention",
    "wrong_world_abstention",
)
CASE_TYPES = (
    "future_answer",
    "correction",
    "unaffected_retention",
    "unknown_abstention",
    "wrong_world_abstention",
)
LOSS_BY_CASE = {
    "future_answer": "future_answer_readout",
    "correction": "correction_readout",
    "unknown_abstention": "unknown_abstention_readout",
    "wrong_world_abstention": "wrong_world_abstention_readout",
}
LOSS_NAMES = tuple(LOSS_BY_CASE.values())
CONTROL_ARM_IDS = (
    "learned_value_readout",
    "exact_context",
    "selected_record_literal_readout",
    "memory_disabled",
    "memory_reset",
    "shuffled_values",
    "wrong_world_values",
)
FORBIDDEN_EVAL_CASE_FIELDS = frozenset({
    "target", "expected_value", "expected_status", "label_source_epoch",
    "authority", "source_record_id", "stale_record_id", "stale_value",
    "rejected_world_id", "rejected_record_id", "rejected_record_source_epoch",
})
FIT_RECEIPT_FIELDS = (
    "artifacts",
    "config_sha256",
    "created_by",
    "dev_labels_read",
    "earlier_write_gradient",
    "fit_gates",
    "format",
    "heldout_opened",
    "losses",
    "mode",
    "optimizer",
    "optimizer_steps",
    "prepared_manifest_sha256",
    "protocol_id",
    "replay_case_count",
    "seed",
    "status",
    "train_case_count",
    "trainable_parameters",
    "tuning_performed",
)
DEVELOPMENT_RECEIPT_FIELDS = (
    "all_declared_gates_passed",
    "attempt",
    "config_sha256",
    "created_by",
    "dev_metrics",
    "fit_receipt_sha256",
    "format",
    "heldout_opened",
    "mode",
    "prepared_manifest_sha256",
    "protocol_id",
    "status",
    "tuning_performed",
)
HELDOUT_RECEIPT_FIELDS = (
    "attempt",
    "config_sha256",
    "created_by",
    "dev_receipt_sha256",
    "format",
    "heldout_metrics",
    "mode",
    "prepared_manifest_sha256",
    "protocol_id",
    "status",
    "tuning_performed",
)

FIT_METRIC_KEYS = (
    "future_answer_accuracy",
    "correction_accuracy",
    "unaffected_retention_accuracy",
    "unknown_abstention_accuracy",
    "wrong_world_abstention_accuracy",
    "stale_value_rate",
)


class ProtocolError(ValueError):
    """One fail-closed protocol, fixture or admission check failed."""

    def __init__(self, check: str, detail: str):
        super().__init__(f"{check}: {detail}")
        self.check = check
        self.detail = detail


# ---------------------------------------------------------------------------
# Deterministic helpers
# ---------------------------------------------------------------------------

def _require(condition: bool, check: str, detail: str) -> None:
    if not condition:
        raise ProtocolError(check, detail)


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False, ensure_ascii=True)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    path = Path(path)
    _require(path.is_file(), "read", f"missing JSON file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ProtocolError("read", f"unreadable JSON {path}: {error}") from error
    _require(isinstance(value, dict), "read", f"JSON root is not an object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict]:
    path = Path(path)
    _require(path.is_file(), "read", f"missing JSONL file: {path}")
    rows: list[dict] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError as error:
            raise ProtocolError("read", f"{path}:{number} is not JSON: {error}") from error
        _require(isinstance(row, dict), "read",
                 f"{path}:{number} is not a JSON object")
        rows.append(row)
    return rows


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(canonical(row) + "\n" for row in rows)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _is_hex64(value: Any) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def _finite_number(value: Any, check: str, name: str) -> float:
    _require(isinstance(value, (int, float)) and not isinstance(value, bool)
             and math.isfinite(float(value)), check, f"{name} must be finite")
    return float(value)


# ---------------------------------------------------------------------------
# Vocabulary and world naming
# ---------------------------------------------------------------------------

def _all_suffixes() -> list[str]:
    return [suffix for split in ("train", "dev", "heldout")
            for suffix in SPLIT_WORLDS[split]]


def build_answer_vocabulary() -> list[str]:
    values = [f"value_{suffix}_{color}" for suffix in _all_suffixes()
              for color in ("amber", "cobalt", "jade")]
    return ["UNKNOWN"] + sorted(values)


def _world_id(suffix: str) -> str:
    return f"world_{suffix}"


def _split_for_world(world_id: str) -> str:
    suffix = world_id.removeprefix("world_")
    for split, suffixes in SPLIT_WORLDS.items():
        if suffix in suffixes:
            return split
    raise ProtocolError("world", f"unknown world {world_id!r}")


def _sibling_world(world_id: str) -> str:
    split = _split_for_world(world_id)
    suffix = world_id.removeprefix("world_")
    siblings = SPLIT_WORLDS[split]
    index = siblings.index(suffix)
    return _world_id(siblings[(index + 1) % len(siblings)])


def _entity(world_id: str, role: str) -> str:
    suffix = world_id.removeprefix("world_")
    return f"entity_{suffix}_{role}"


def _attribute(world_id: str, role: str) -> str:
    suffix = world_id.removeprefix("world_")
    return f"attribute_{suffix}_{role}"


def _value(world_id: str, color: str) -> str:
    return f"value_{world_id.removeprefix('world_')}_{color}"


def _address(world_id: str, role: str) -> dict:
    return {"entity": _entity(world_id, role),
            "attribute": _attribute(world_id, role)}


def _primary_address(world_id: str) -> dict:
    return _address(world_id, "primary")


def _secondary_address(world_id: str) -> dict:
    return _address(world_id, "secondary")


def _absent_address(world_id: str) -> dict:
    return _address(world_id, "absent")


def _query_forms() -> dict[str, tuple[str, ...]]:
    return {
        "future_answer": (
            "What is the {attribute} of {entity}?",
            "Report the current {attribute} for {entity}.",
            "Which value is recorded for {attribute} of {entity}?",
        ),
        "correction": (
            "After any correction, what is the {attribute} of {entity}?",
            "What is the current {attribute} for {entity}?",
            "State the latest {attribute} of {entity}.",
        ),
        "unaffected_retention": (
            "What is the {attribute} of {entity}?",
            "Report {entity}'s {attribute}.",
            "Which value belongs to {attribute} of {entity}?",
        ),
        "unknown_abstention": (
            "What is the {attribute} of {entity}?",
            "Report the current {attribute} for {entity}.",
            "Which value is recorded for {attribute} of {entity}?",
        ),
        "wrong_world_abstention": (
            "What is the {attribute} of {entity}?",
            "Report the current {attribute} for {entity}.",
            "Which value is recorded for {attribute} of {entity}?",
        ),
    }


def _query_form(seed: int, world_id: str, case_type: str, epoch: int,
                replay: bool) -> int:
    payload = {"seed": seed, "world_id": world_id, "case_type": case_type,
               "epoch": epoch, "replay": replay}
    return int(sha256_json(payload)[:8], 16) % 3


def _query_text(case_type: str, form: int, address: Mapping[str, str]) -> str:
    template = _query_forms()[case_type][form]
    return template.format(entity=address["entity"],
                           attribute=address["attribute"])


def _feature_digest(seed: int, case_id: str, world_id: str, epoch: int,
                    address: Mapping[str, str], query_form: int,
                    source_record_id: str | None, target: Mapping[str, Any]) -> str:
    return sha256_json({
        "width": 256,
        "seed": seed,
        "case_id": case_id,
        "world_id": world_id,
        "epoch": epoch,
        "address": dict(address),
        "query_form": query_form,
        "source_record_id": source_record_id,
        "target_token": target["token"],
    })


def _semantic_digest(case: Mapping[str, Any],
                     target: Mapping[str, Any]) -> str:
    return sha256_json({
        "world_id": case["world_id"],
        "epoch": case["epoch"],
        "case_type": case["case_type"],
        "address": case["address"],
        "query_text": case["query_text"],
        "target": target,
    })


# ---------------------------------------------------------------------------
# Authoritative targets
# ---------------------------------------------------------------------------

def _current_records(records: Sequence[Mapping[str, Any]], world_id: str,
                     entity: str, attribute: str,
                     epoch: int) -> list[Mapping[str, Any]]:
    visible = [record for record in records
               if record["world_id"] == world_id
               and record["entity"] == entity
               and record["attribute"] == attribute
               and int(record["source_epoch"]) <= epoch]
    superseded = {record.get("supersedes") for record in visible
                  if record.get("supersedes")}
    return [record for record in visible
            if record["record_id"] not in superseded]


def _target_object(token: str, vocabulary: Sequence[str]) -> dict:
    _require(token in vocabulary, "target", f"{token!r} is outside the vocabulary")
    return {"kind": "status" if token == "UNKNOWN" else "value",
            "token": token, "index": list(vocabulary).index(token)}


def _derive_target(records: Sequence[Mapping[str, Any]],
                   query: Mapping[str, Any],
                   vocabulary: Sequence[str]) -> dict:
    address = query["address"]
    world_id = query["world_id"]
    epoch = int(query["epoch"])
    case_type = query["case_type"]
    current = _current_records(records, world_id, address["entity"],
                               address["attribute"], epoch)
    _require(len(current) <= 1, "label",
             f"ambiguous address for {query['case_id']}: {address}")
    if current:
        _require(case_type not in ("unknown_abstention", "wrong_world_abstention"),
                 "label", f"{query['case_id']} is an abstention case but a record exists")
        record = current[0]
        target = _target_object(record["value"], vocabulary)
        label = {
            "target": target,
            "label_source_epoch": int(record["source_epoch"]),
            "authority": ("latest_non_superseded_record_at_query_epoch"
                          if case_type == "correction"
                          else "current_record_at_query_epoch"),
            "source_record_id": record["record_id"],
            "stale_record_id": None,
            "stale_value": None,
            "rejected_world_id": None,
            "rejected_record_id": None,
            "rejected_record_source_epoch": None,
        }
        if case_type == "correction":
            stale_id = record.get("supersedes")
            _require(bool(stale_id), "label",
                     f"correction case {query['case_id']} lacks a superseded record")
            stale = [item for item in records
                     if item["world_id"] == world_id
                     and item["record_id"] == stale_id]
            _require(len(stale) == 1, "label",
                     f"superseded record missing for {query['case_id']}")
            stale = stale[0]
            _require(stale["value"] != record["value"], "label",
                     f"correction case {query['case_id']} did not change the value")
            _require(int(stale["source_epoch"]) < int(record["source_epoch"]),
                     "label", f"correction case {query['case_id']} is not later")
            label["stale_record_id"] = stale["record_id"]
            label["stale_value"] = stale["value"]
        return label

    _require(case_type in ("unknown_abstention", "wrong_world_abstention"),
             "label", f"answer case {query['case_id']} has no current record")
    other_worlds: dict[str, list[Mapping[str, Any]]] = {}
    for split_world in [_world_id(suffix) for suffix in _all_suffixes()]:
        if split_world == world_id:
            continue
        found = _current_records(records, split_world, address["entity"],
                                 address["attribute"], epoch)
        if found:
            other_worlds[split_world] = found
    if case_type == "unknown_abstention":
        _require(not other_worlds, "label",
                 f"unknown case {query['case_id']} collides with a visible record")
        return {
            "target": _target_object("UNKNOWN", vocabulary),
            "label_source_epoch": epoch,
            "authority": "absence_in_current_world_at_query_epoch",
            "source_record_id": None,
            "stale_record_id": None,
            "stale_value": None,
            "rejected_world_id": None,
            "rejected_record_id": None,
            "rejected_record_source_epoch": None,
        }

    rejected_world = query["wrong_world_id"]
    _require(rejected_world == _sibling_world(world_id), "label",
             f"wrong-world case {query['case_id']} must name its sibling world")
    _require(_split_for_world(rejected_world) == _split_for_world(world_id),
             "label", f"wrong-world case {query['case_id']} crosses a split")
    _require(rejected_world in other_worlds, "label",
             f"wrong-world case {query['case_id']} has no rejected record")
    rejected = other_worlds[rejected_world]
    _require(len(rejected) == 1, "label",
             f"wrong-world case {query['case_id']} is ambiguous")
    record = rejected[0]
    return {
        "target": _target_object("UNKNOWN", vocabulary),
        "label_source_epoch": epoch,
        "authority": "world_scope_rejection_at_query_epoch",
        "source_record_id": None,
        "stale_record_id": None,
        "stale_value": None,
        "rejected_world_id": rejected_world,
        "rejected_record_id": record["record_id"],
        "rejected_record_source_epoch": int(record["source_epoch"]),
    }


# ---------------------------------------------------------------------------
# Fixture construction
# ---------------------------------------------------------------------------

def _fresh_output(path: Path) -> Path:
    path = Path(path)
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ProtocolError("output", f"output path is not fresh: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _record(world_id: str, record_id: str, entity: str, attribute: str,
            value: str, epoch: int, supersedes: str | None = None) -> dict:
    return {
        "record_id": record_id,
        "world_id": world_id,
        "entity": entity,
        "attribute": attribute,
        "value": value,
        "source_epoch": epoch,
        "supersedes": supersedes,
    }


def _event_header(format_value: str, world_id: str, epoch: int,
                  kind: str) -> dict:
    suffix = world_id.removeprefix("world_")
    return {
        "format": format_value,
        "event_id": f"event_{suffix}_{epoch:02d}",
        "episode_id": f"episode_{world_id}",
        "world_id": world_id,
        "split": _split_for_world(world_id),
        "epoch": epoch,
        "clock_step": epoch,
        "kind": kind,
    }


def _query_row(world_id: str, epoch: int, case_type: str, case_id: str,
               query_id: str, address: Mapping[str, str], seed: int,
               vocabulary: Sequence[str], records: Sequence[Mapping[str, Any]],
               replay: bool, replay_of_case_id: str | None,
               teaching: bool, wrong_world_id: str | None = None) -> tuple[dict, dict]:
    form = _query_form(seed, world_id, case_type, epoch, replay)
    text = _query_text(case_type, form, address)
    query = {
        "case_id": case_id,
        "query_id": query_id,
        "case_type": case_type,
        "world_id": world_id,
        "epoch": epoch,
        "address": dict(address),
        "query_form": form,
        "query_text": text,
        "replay": replay,
        "replay_of_case_id": replay_of_case_id,
        "wrong_world_id": wrong_world_id,
    }
    label = _derive_target(records, query, vocabulary)
    event = _event_header(EVENT_FORMAT, world_id, epoch, "query")
    event.update({
        "optimizer_steps": 1 if teaching else 0,
        "gradient": bool(teaching),
        "loss_name": LOSS_BY_CASE.get(case_type) if teaching else None,
        "case_id": case_id,
        "query_id": query_id,
        "case_type": case_type,
        "address": dict(address),
        "query_form": form,
        "query_text": text,
        "replay": replay,
        "replay_of_case_id": replay_of_case_id,
        "wrong_world_id": wrong_world_id,
        "expert_id": 0,
        "feature_digest": _feature_digest(
            seed, case_id, world_id, epoch, address, form,
            label["source_record_id"], label["target"]),
    })
    return event, label


def _case_row(event: Mapping[str, Any], label: Mapping[str, Any],
              include_target: bool) -> dict:
    row = {
        "format": CASE_FORMAT,
        "case_id": event["case_id"],
        "case_type": event["case_type"],
        "split": event["split"],
        "replay": event["replay"],
        "replay_of_case_id": event["replay_of_case_id"],
        "episode_id": event["episode_id"],
        "world_id": event["world_id"],
        "event_id": event["event_id"],
        "epoch": event["epoch"],
        "query_id": event["query_id"],
        "query_text": event["query_text"],
        "address": event["address"],
        "query_form": event["query_form"],
        "expert_id": event["expert_id"],
        "optimizer_steps": event["optimizer_steps"],
        "loss_name": (LOSS_BY_CASE.get(event["case_type"])
                      if event["optimizer_steps"] else None),
        "feature_digest": event["feature_digest"],
    }
    if include_target:
        row.update(label)
    return row


def _label_row(case: Mapping[str, Any], label: Mapping[str, Any]) -> dict:
    return {
        "format": LABEL_FORMAT,
        "case_id": case["case_id"],
        "case_type": case["case_type"],
        "split": case["split"],
        "epoch": case["epoch"],
        "target": label["target"],
        "label_source_epoch": label["label_source_epoch"],
        "authority": label["authority"],
        "source_record_id": label["source_record_id"],
        "stale_record_id": label["stale_record_id"],
        "stale_value": label["stale_value"],
        "rejected_world_id": label["rejected_world_id"],
        "rejected_record_id": label["rejected_record_id"],
        "rejected_record_source_epoch": label["rejected_record_source_epoch"],
    }


def _world_records(world_id: str, suffix: str) -> list[dict]:
    initial = _record(
        world_id, f"record_{suffix}_primary_initial",
        _entity(world_id, "primary"), _attribute(world_id, "primary"),
        _value(world_id, "amber"), 0)
    secondary = _record(
        world_id, f"record_{suffix}_secondary",
        _entity(world_id, "secondary"), _attribute(world_id, "secondary"),
        _value(world_id, "jade"), 2)
    corrected = _record(
        world_id, f"record_{suffix}_primary_corrected",
        _entity(world_id, "primary"), _attribute(world_id, "primary"),
        _value(world_id, "cobalt"), 3, initial["record_id"])
    return [initial, secondary, corrected]


def build_fixture(config: Mapping[str, Any], config_path: Path,
                  output_dir: Path) -> dict:
    """Build the complete deterministic fixture and manifest."""
    validate_config(config, config_path)
    output = _fresh_output(output_dir)
    vocabulary = list(config["architecture"]["answer_vocabulary"])
    _require(vocabulary == build_answer_vocabulary(), "config",
             "answer vocabulary differs from the deterministic vocabulary")
    seed = int(config["transaction"]["seed"])
    file_names = config["prepared"]["files"]

    events: list[dict] = []
    train_cases: list[dict] = []
    replay_cases: list[dict] = []
    dev_cases: list[dict] = []
    heldout_cases: list[dict] = []
    dev_labels: list[dict] = []
    heldout_labels: list[dict] = []
    all_records: list[dict] = []
    for split in ("train", "dev", "heldout"):
        for suffix in SPLIT_WORLDS[split]:
            all_records.extend(_world_records(_world_id(suffix), suffix))

    for split in ("train", "dev", "heldout"):
        for suffix in SPLIT_WORLDS[split]:
            world_id = _world_id(suffix)
            sibling = _sibling_world(world_id)
            records = _world_records(world_id, suffix)
            initial, secondary, corrected = records

            write_events = [
                (0, "observation", "initial", initial),
                (2, "observation", "secondary", secondary),
                (3, "correction", "correction", corrected),
            ]
            for write_index, (epoch, kind, _role, record) in enumerate(write_events):
                event = _event_header(EVENT_FORMAT, world_id, epoch, kind)
                event.update({
                    "optimizer_steps": 0,
                    "gradient": False,
                    "write_index": write_index,
                    "record": dict(record),
                })
                events.append(event)

            base_queries = [
                (1, "future_answer", _primary_address(world_id), "future", None, True),
                (4, "correction", _primary_address(world_id), "correction", None, True),
                (5, "unaffected_retention", _secondary_address(world_id),
                 "retention", None, False),
                (6, "unknown_abstention", _absent_address(world_id),
                 "unknown", None, True),
                (7, "wrong_world_abstention", _primary_address(sibling),
                 "wrong_world", sibling, True),
            ]
            for epoch, case_type, address, tag, wrong_world, potential_teaching in base_queries:
                teaching = split == "train" and potential_teaching
                case_id = f"case_{suffix}_{tag}"
                query_id = f"query_{suffix}_{tag}"
                event, label = _query_row(
                    world_id, epoch, case_type, case_id, query_id, address,
                    seed, vocabulary, all_records, False, None, teaching, wrong_world)
                events.append(event)
                case = _case_row(event, label, include_target=split == "train")
                if split == "train":
                    train_cases.append(case)
                elif split == "dev":
                    dev_cases.append(case)
                    dev_labels.append(_label_row(case, label))
                else:
                    heldout_cases.append(case)
                    heldout_labels.append(_label_row(case, label))

            if split == "train":
                base_future = f"case_{suffix}_future"
                base_correction = f"case_{suffix}_correction"
                base_unknown = f"case_{suffix}_unknown"
                base_wrong = f"case_{suffix}_wrong_world"
                replays = [
                    (8, "future_answer", _primary_address(world_id),
                     "future_replay", base_future, None),
                    (9, "correction", _primary_address(world_id),
                     "correction_replay", base_correction, None),
                    (10, "unknown_abstention", _absent_address(world_id),
                     "unknown_replay", base_unknown, None),
                    (11, "wrong_world_abstention", _primary_address(sibling),
                     "wrong_world_replay", base_wrong, sibling),
                ]
                for epoch, case_type, address, tag, replay_of, wrong_world in replays:
                    case_id = f"case_{suffix}_{tag}"
                    query_id = f"query_{suffix}_{tag}"
                    event, label = _query_row(
                        world_id, epoch, case_type, case_id, query_id, address,
                        seed, vocabulary, all_records, True, replay_of, True,
                        wrong_world)
                    events.append(event)
                    replay_cases.append(_case_row(event, label, include_target=True))

    controls = []
    expected_outcome = {
        "learned_value_readout": "evaluate_primary_gates",
        "exact_context": "report_quality_and_cost_only",
        "selected_record_literal_readout": "diagnostic_reference_only",
        "memory_disabled": "must_return_unknown_for_every_case",
        "memory_reset": "must_return_unknown_for_every_case",
        "shuffled_values": "must_not_pass_primary_gates",
        "wrong_world_values": "must_not_pass_primary_gates",
    }
    for arm in config["controls"]["arms"]:
        row = dict(arm)
        row["format"] = CONTROL_FORMAT
        row["expected_outcome"] = expected_outcome[arm["arm_id"]]
        controls.append(row)

    events.sort(key=lambda row: (row["split"] != "train", row["world_id"], row["epoch"]))
    train_cases.sort(key=lambda row: (row["world_id"], row["epoch"]))
    replay_cases.sort(key=lambda row: (row["world_id"], row["epoch"]))
    dev_cases.sort(key=lambda row: (row["world_id"], row["epoch"]))
    heldout_cases.sort(key=lambda row: (row["world_id"], row["epoch"]))
    dev_labels.sort(key=lambda row: row["case_id"])
    heldout_labels.sort(key=lambda row: row["case_id"])

    payloads = {
        file_names["episodes"]: events,
        file_names["cases_train"]: train_cases,
        file_names["cases_train_replay"]: replay_cases,
        file_names["cases_dev"]: dev_cases,
        file_names["cases_heldout"]: heldout_cases,
        file_names["labels_dev"]: dev_labels,
        file_names["labels_heldout"]: heldout_labels,
        file_names["controls"]: controls,
    }
    for name, rows in payloads.items():
        write_jsonl(output / name, rows)

    counts = _count_rows(events, train_cases, replay_cases, dev_cases,
                         heldout_cases, dev_labels, heldout_labels)
    _require(counts == config["prepared"]["expected_counts"], "fixture",
             f"generated counts differ from config: {counts}")
    loss_counts = _loss_counts(train_cases + replay_cases)
    _require(loss_counts == config["prepared"]["loss_counts"], "fixture",
             "generated loss counts differ from config")
    disjoint = _disjointness_from_rows(
        events, train_cases, replay_cases, dev_cases, heldout_cases)
    planned = _planned_workload(config, counts)

    manifest = {
        "format": MANIFEST_FORMAT,
        "protocol_id": PROTOCOL_ID,
        "generated_by": "scripts/prepare_value_readout_abstention_v1.py",
        "generator_contract": "scripts/value_readout_abstention_contract_v1.py",
        "seed": seed,
        "deterministic": True,
        "native_computation": False,
        "model_loaded": False,
        "tokenizer_loaded": False,
        "downloads": 0,
        "heldout_opened": False,
        "config": {
            "path": _relative_config_path(config_path),
            "sha256": sha256_file(config_path),
        },
        "files": {name: sha256_file(output / name) for name in payloads},
        "counts": counts,
        "splits": _split_counts(events, train_cases, replay_cases,
                                dev_cases, heldout_cases),
        "disjointness": disjoint,
        "answer_vocabulary": vocabulary,
        "loss_counts": loss_counts,
        "label_seal": {
            "dev_labels_sha256": sha256_file(output / file_names["labels_dev"]),
            "heldout_labels_sha256": sha256_file(output / file_names["labels_heldout"]),
            "dev_label_cases": len(dev_labels),
            "heldout_label_cases": len(heldout_labels),
            "heldout_opened": False,
        },
        "controls": {"arm_ids": list(CONTROL_ARM_IDS),
                     "count": len(controls)},
        "fit_receipt_requirements": config["admission"]["fit_receipt"],
        "planned_workload": planned,
        "workload_derivation": config["budget"]["authorized_run_ceiling"][
            "workload_derivation"],
        "manifest_sha256": "",
    }
    manifest["manifest_sha256"] = sha256_json(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"})
    write_json(output / "manifest.json", manifest)
    return manifest


def _relative_config_path(path: Path) -> str:
    path = Path(path).resolve()
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _count_rows(events: Sequence[Mapping[str, Any]],
                train_cases: Sequence[Mapping[str, Any]],
                replay_cases: Sequence[Mapping[str, Any]],
                dev_cases: Sequence[Mapping[str, Any]],
                heldout_cases: Sequence[Mapping[str, Any]],
                dev_labels: Sequence[Mapping[str, Any]],
                heldout_labels: Sequence[Mapping[str, Any]]) -> dict:
    episodes = {event["episode_id"] for event in events}
    teaching = sum(1 for case in list(train_cases) + list(replay_cases)
                   if int(case["optimizer_steps"]) == 1)
    return {
        "episodes": len(episodes),
        "episodes_train": SPLIT_EXPECTED["train"],
        "episodes_dev": SPLIT_EXPECTED["dev"],
        "episodes_heldout": SPLIT_EXPECTED["heldout"],
        "events": len(events),
        "events_train": sum(1 for event in events if event["split"] == "train"),
        "events_dev": sum(1 for event in events if event["split"] == "dev"),
        "events_heldout": sum(1 for event in events if event["split"] == "heldout"),
        "queries": sum(1 for event in events if event["kind"] == "query"),
        "write_events": sum(1 for event in events if event["kind"] != "query"),
        "cases": len(train_cases) + len(replay_cases) + len(dev_cases) + len(heldout_cases),
        "cases_train": len(train_cases),
        "cases_train_replay": len(replay_cases),
        "cases_dev": len(dev_cases),
        "cases_heldout": len(heldout_cases),
        "teaching_cases": teaching,
        "teaching_cases_base": sum(1 for case in train_cases
                                   if int(case["optimizer_steps"]) == 1),
        "teaching_cases_replay": sum(1 for case in replay_cases
                                     if int(case["optimizer_steps"]) == 1),
        "retention_diagnostic_cases": sum(
            1 for case in train_cases
            if case["case_type"] == "unaffected_retention"),
        "optimizer_steps": sum(int(case["optimizer_steps"])
                               for case in list(train_cases) + list(replay_cases)
                               + list(dev_cases) + list(heldout_cases)),
        "labels_dev": len(dev_labels),
        "labels_heldout": len(heldout_labels),
    }


def _loss_counts(cases: Sequence[Mapping[str, Any]]) -> dict:
    counts = {name: 0 for name in LOSS_NAMES}
    for case in cases:
        if case["loss_name"]:
            _require(case["loss_name"] in counts, "loss",
                     f"unknown loss {case['loss_name']!r}")
            counts[case["loss_name"]] += 1
    return counts


def _disjointness_from_rows(events: Sequence[Mapping[str, Any]],
                            train_cases: Sequence[Mapping[str, Any]],
                            replay_cases: Sequence[Mapping[str, Any]],
                            dev_cases: Sequence[Mapping[str, Any]],
                            heldout_cases: Sequence[Mapping[str, Any]]) -> dict:
    worlds = {split: {event["world_id"] for event in events
                      if event["split"] == split}
              for split in ("train", "dev", "heldout")}
    ids = {
        "train": {case["case_id"] for case in train_cases},
        "replay": {case["case_id"] for case in replay_cases},
        "dev": {case["case_id"] for case in dev_cases},
        "heldout": {case["case_id"] for case in heldout_cases},
    }
    return {
        "train_worlds_disjoint_from_dev": not (worlds["train"] & worlds["dev"]),
        "train_worlds_disjoint_from_heldout": not (worlds["train"] & worlds["heldout"]),
        "dev_worlds_disjoint_from_heldout": not (worlds["dev"] & worlds["heldout"]),
        "train_cases_disjoint_from_dev": not (ids["train"] & ids["dev"]),
        "train_cases_disjoint_from_heldout": not (ids["train"] & ids["heldout"]),
        "replay_cases_disjoint_from_dev": not (ids["replay"] & ids["dev"]),
        "replay_cases_disjoint_from_heldout": not (ids["replay"] & ids["heldout"]),
        "all_case_ids_unique": (len(ids["train"]) + len(ids["replay"])
                                + len(ids["dev"]) + len(ids["heldout"])
                                == len(set().union(*ids.values()))),
        "training_semantics_disjoint_from_evaluation": True,
        "heldout_labels_sealed": True,
    }


def _split_counts(events: Sequence[Mapping[str, Any]],
                  train_cases: Sequence[Mapping[str, Any]],
                  replay_cases: Sequence[Mapping[str, Any]],
                  dev_cases: Sequence[Mapping[str, Any]],
                  heldout_cases: Sequence[Mapping[str, Any]]) -> dict:
    return {
        "train": {
            "episodes": len({event["episode_id"] for event in events
                             if event["split"] == "train"}),
            "events": sum(1 for event in events if event["split"] == "train"),
            "writes": sum(1 for event in events if event["split"] == "train"
                          and event["kind"] != "query"),
            "cases": len(train_cases) + len(replay_cases),
            "base_cases": len(train_cases),
            "replay_cases": len(replay_cases),
            "teaching_cases": sum(1 for case in list(train_cases) + list(replay_cases)
                                  if int(case["optimizer_steps"]) == 1),
        },
        "dev": {
            "episodes": len({event["episode_id"] for event in events
                             if event["split"] == "dev"}),
            "events": sum(1 for event in events if event["split"] == "dev"),
            "writes": sum(1 for event in events if event["split"] == "dev"
                          and event["kind"] != "query"),
            "cases": len(dev_cases),
            "teaching_cases": 0,
        },
        "heldout": {
            "episodes": len({event["episode_id"] for event in events
                             if event["split"] == "heldout"}),
            "events": sum(1 for event in events if event["split"] == "heldout"),
            "writes": sum(1 for event in events if event["split"] == "heldout"
                          and event["kind"] != "query"),
            "cases": len(heldout_cases),
            "teaching_cases": 0,
        },
    }


def _planned_workload(config: Mapping[str, Any], counts: Mapping[str, Any]) -> dict:
    arms = config["controls"]["arms"]
    writing_arms = [arm for arm in arms if arm["writes"] != "none"]
    trained_arms = [arm for arm in arms
                    if arm["arm_id"] in ("learned_value_readout",
                                         "shuffled_values",
                                         "wrong_world_values")]
    writes = counts["write_events"] * len(writing_arms)
    updates = counts["teaching_cases"] * len(trained_arms)
    evaluation_cases = counts["cases_dev"] + counts["cases_heldout"]
    readouts = (updates
                + evaluation_cases * len(arms)
                + counts["retention_diagnostic_cases"] * len(trained_arms))
    context_cases = evaluation_cases
    return {
        "write_events_total": writes,
        "optimizer_updates_total": updates,
        "backward_calls_total": updates,
        "readout_forward_calls_total": readouts,
        "context_reference_cases": context_cases,
        "context_tokens_total": context_cases * 24,
        "generated_tokens_total": 0,
        "checkpoint_writes": len(trained_arms),
    }


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------

def _expected_data_limits() -> dict:
    return {
        "episodes": 7,
        "events": 68,
        "queries": 47,
        "cases": 47,
        "teaching_cases": 24,
        "optimizer_steps": 24,
        "write_events": 21,
        "answer_vocabulary_size": 22,
        "prepared_bytes_max": 1048576,
        "max_events_per_episode": 12,
        "max_cases_per_split": 27,
    }


def _expected_planned_workload() -> dict:
    return {
        "write_events_total": 105,
        "optimizer_updates_total": 72,
        "backward_calls_total": 72,
        "readout_forward_calls_total": 221,
        "context_reference_cases": 20,
        "context_tokens_total": 480,
        "generated_tokens_total": 0,
        "checkpoint_writes": 3,
    }


def _expected_workload_derivation() -> dict:
    return {
        "basis": ("three independent trained arms, each with its own "
                  "one-step-per-teaching-event clock; no arm shares or replays "
                  "another arm's optimizer step"),
        "arms": [
            {
                "arm_id": "learned_value_readout",
                "phase": "fit_treatment",
                "teaching_cases": 24,
                "optimizer_steps": 24,
                "backward_calls": 24,
                "fit_receipt_binding": True,
            },
            {
                "arm_id": "shuffled_values",
                "phase": "control_fit",
                "teaching_cases": 24,
                "optimizer_steps": 24,
                "backward_calls": 24,
                "fit_receipt_binding": False,
            },
            {
                "arm_id": "wrong_world_values",
                "phase": "control_fit",
                "teaching_cases": 24,
                "optimizer_steps": 24,
                "backward_calls": 24,
                "fit_receipt_binding": False,
            },
        ],
        "fit_receipt_arm_id": "learned_value_readout",
        "fit_receipt_optimizer_steps": 24,
        "control_fit_optimizer_steps": 48,
        "total_optimizer_steps": 72,
        "readout_forward_calls_derivation": {
            "training_forwards": 72,
            "evaluation_forwards": 140,
            "retention_diagnostic_forwards": 9,
            "total": 221,
        },
        "write_events_derivation": {
            "writing_arms": 5,
            "writes_per_arm": 21,
            "total": 105,
        },
        "checkpoint_writes_derivation": {
            "trained_arms": 3,
            "checkpoints_per_trained_arm": 1,
            "total": 3,
        },
        "one_step_per_teaching_event": True,
        "independent_arm_clocks": 3,
        "shared_clock_across_arms": False,
        "extra_steps_per_event": 0,
    }


def _expected_fit_scope() -> dict:
    return {
        "arm_id": "learned_value_readout",
        "phase": "fit_treatment",
        "optimizer_steps": 24,
        "excluded_trained_controls": ["shuffled_values", "wrong_world_values"],
        "reason": ("the fit receipt admits the treatment arm only; the two "
                   "trained control arms run under the experiment ceiling and "
                   "are never part of development admission"),
    }


def validate_config(config: Mapping[str, Any],
                    config_path: Path | None = None) -> None:
    """Validate every declared protocol invariant without reading the fixture."""
    _require(isinstance(config, Mapping), "config", "config must be an object")
    _require(config.get("format") == CONFIG_FORMAT, "config",
             "config format differs")
    _require(config.get("protocol_id") == PROTOCOL_ID, "config",
             "protocol id differs")

    classification = config["classification"]
    _require(classification.get("preparation_only") is True, "config",
             "classification.preparation_only must be true")
    for key in ("native", "model_execution", "tokenizer_load", "generation",
                "persistent_learning", "calibrated", "learned_selector",
                "native_execution_allowed"):
        _require(classification.get(key) is False, "config",
                 f"classification.{key} must be false")
    _require(classification.get("heldout_opened") is False, "config",
             "heldout must be closed")
    _require(classification.get("oracle") == "diagnostic_only_not_executed",
             "config", "oracle must not be executed")

    architecture = config["architecture"]
    bank = architecture["expert_bank"]
    _require(bank["declared_count"] == 16 and bank["declared_width"] == 256
             and bank["declared_hidden_width"] == 1024, "config",
             "expert bank shape differs")
    _require(bank["bias"] is False, "config", "expert bias is not allowed")
    _require(bank["matrices"] == ["gate", "up", "down"], "config",
             "expert matrices differ")
    _require(bank["selected_expert_id"] == 0, "config",
             "selected expert must be 0")
    trainable = ["selected_expert.gate", "selected_expert.up",
                 "selected_expert.down"]
    _require(architecture["trainable_parameters"] == trainable, "config",
             "trainable parameters must be selected gate/up/down only")
    fixed = architecture["fixed"]
    for key in ("backbone", "read_projection", "decoder", "writer", "router",
                "personality", "fast_state", "input_encoder", "selector"):
        _require(isinstance(fixed.get(key), str) and fixed[key], "config",
                 f"fixed component {key} is not declared")
    _require(architecture["no_architecture_expansion"] is True, "config",
             "architecture expansion is forbidden")
    vocabulary = architecture["answer_vocabulary"]
    _require(vocabulary == build_answer_vocabulary(), "config",
             "answer vocabulary differs")
    _require(len(vocabulary) == 22 and len(set(vocabulary)) == 22, "config",
             "answer vocabulary must contain 22 unique tokens")
    _require(architecture["status_vocabulary"] == ["UNKNOWN"], "config",
             "UNKNOWN must be the sole status token")
    readout = architecture["readout"]
    _require(readout["objective"] == "closed_vocabulary_cross_entropy",
             "config", "readout objective differs")
    _require(readout["training_readout_positions_per_case"] == 1
             and readout["evaluation_readout_positions_per_case"] == 1,
             "config", "one readout position per case is required")

    transaction = config["transaction"]
    exact_transaction = {
        "steps_per_event": 1,
        "clock_steps_per_event": 1,
        "max_optimizer_steps_per_event": 1,
        "optimizer_steps_per_teaching_event": 1,
        "optimizer_steps_per_write_event": 0,
        "optimizer_steps_per_query_event": 0,
        "optimizer_steps_per_evaluation_query": 0,
        "batch_size": 1,
        "gradient_accumulation": "none",
        "epochs_per_case": 1,
        "retries": 0,
        "optimizer": "NativeFusedAdamW",
        "seed": SEED,
    }
    for key, value in exact_transaction.items():
        _require(transaction.get(key) == value, "config",
                 f"transaction.{key} must be {value!r}")
    _require(transaction.get("gradient_norm_max") == 1.0, "config",
             "gradient norm cap differs")
    hyper = transaction["optimizer_hyperparameters"]
    _require(hyper["lr"] == 0.001 and hyper["betas"] == [0.9, 0.999]
             and hyper["eps"] == 1e-08 and hyper["weight_decay"] == 0.0,
             "config", "optimizer hyperparameters differ")
    schedule = transaction["one_step_schedule"]
    _require(schedule["teaching_events"] == 24
             and schedule["write_events"] == 21
             and schedule["retention_diagnostic_events"] == 3
             and schedule["evaluation_query_events"] == 20
             and schedule["total_events"] == 68, "config",
             "one-step schedule counts differ")

    losses = config["losses"]
    terms = losses["terms"]
    _require([term["name"] for term in terms] == list(LOSS_NAMES), "config",
             "loss terms differ")
    for term in terms:
        _require(term["case_type"] in TEACHING_CASE_TYPES, "config",
                 f"loss {term['name']} has an unknown case type")
        _require(LOSS_BY_CASE[term["case_type"]] == term["name"], "config",
                 f"loss {term['name']} is bound to the wrong case type")
        _require(term["weight"] == 1.0, "config",
                 f"loss {term['name']} weight differs")
        _require(term["objective"] == "cross_entropy", "config",
                 f"loss {term['name']} objective differs")
        _require(term["affects"] == trainable, "config",
                 f"loss {term['name']} affects parameters outside the selected expert")
        _require(term["expected_teaching_cases"] == 6, "config",
                 f"loss {term['name']} teaching count differs")
    for key in ("retention_teaching_loss", "local_associative_loss",
                "writer_loss", "backbone_loss", "router_loss",
                "evaluation_loss"):
        _require(losses[key]["enabled"] is False, "config",
                 f"{key} must be disabled")
    gradient = losses["gradient_contract"]
    exact_gradient = {
        "gradient_scope": "selected_expert_gate_up_down_only",
        "backprop_through_earlier_writes": False,
        "backprop_through_stored_state": False,
        "backprop_through_retention_evaluation": False,
        "backprop_through_dev_or_heldout": False,
        "write_path_stop_gradient": True,
        "stored_state_detached": True,
        "cross_event_gradient": "none",
        "inactive_experts_receive_gradients": False,
        "optimizer_state_touched": "selected_expert_only",
    }
    for key, value in exact_gradient.items():
        _require(gradient.get(key) == value, "config",
                 f"gradient_contract.{key} must be {value!r}")

    controls = config["controls"]
    arms = controls["arms"]
    _require(tuple(arm["arm_id"] for arm in arms) == CONTROL_ARM_IDS, "config",
             "control arms differ")
    by_id = {arm["arm_id"]: arm for arm in arms}
    _require(by_id["learned_value_readout"]["gate"] == "primary", "config",
             "treatment gate differs")
    _require(by_id["exact_context"]["gate"] == "none", "config",
             "exact context must not be gated")
    _require(by_id["exact_context"]["treatment_required_to_beat"] is False,
             "config", "exact context must not be required to be beaten")
    _require(by_id["selected_record_literal_readout"]["gate"] == "none",
             "config", "literal readout must be diagnostic only")
    _require(by_id["memory_disabled"]["state"] != by_id["memory_reset"]["state"],
             "config", "disabled and reset memory states must differ")
    for arm_id in ("memory_disabled", "memory_reset"):
        _require(by_id[arm_id]["gate"] == "must_abstain_all", "config",
                 f"{arm_id} must abstain on every case")
    for arm_id in ("shuffled_values", "wrong_world_values"):
        _require(by_id[arm_id]["gate"] == "must_not_pass_primary_gates",
                 "config", f"{arm_id} must not pass primary gates")
        _require(by_id[arm_id]["max_any_case_accuracy"] <= 0.5, "config",
                 f"{arm_id} control ceiling differs")
    _require(by_id["shuffled_values"]["state"]
             != by_id["wrong_world_values"]["state"], "config",
             "shuffled and wrong-world states must differ")
    for arm in arms:
        _require(arm["combine_with_other_arms"] is False, "config",
                 f"arm {arm['arm_id']} must not be combined")
        _require(arm["treatment_required_to_beat"] is False, "config",
                 f"arm {arm['arm_id']} treatment ordering differs")
    for key, value in controls["distinctness"].items():
        _require(value is True, "config", f"distinctness.{key} must be true")

    evaluation = config["evaluation"]
    _require(evaluation["order"] == ["fit_receipt", "development", "heldout"],
             "config", "evaluation order differs")
    fit = evaluation["fit"]
    dev = evaluation["dev"]
    expected_thresholds = {
        "future_answer_accuracy_min": 0.8,
        "correction_accuracy_min": 0.7,
        "unaffected_retention_accuracy_min": 0.8,
        "unknown_abstention_accuracy_min": 0.9,
        "wrong_world_abstention_accuracy_min": 0.9,
        "stale_value_rate_max": 0.05,
    }
    _require(fit["thresholds"] == expected_thresholds, "config",
             "fit thresholds differ")
    _require(dev["thresholds"] == expected_thresholds, "config",
             "dev thresholds differ")
    _require(fit["local_associative_loss_gate"] is False, "config",
             "fit cannot gate a local associative loss")
    _require(dev["training_loss_gate"] is False, "config",
             "dev cannot gate a training loss")
    _require(dev["local_associative_loss_gate"] is False, "config",
             "dev cannot gate a local associative loss")
    _require(dev["requires_fit_receipt"] is True, "config",
             "dev must require a fit receipt")
    _require(dev["tuning"] == "forbidden", "config", "dev tuning is forbidden")
    _require(dev["gate_basis"] == list(CASE_TYPES), "config",
             "dev gate basis differs")
    heldout = evaluation["heldout"]
    _require(heldout["after_development_pass_only"] is True
             and heldout["attempts"] == 1, "config",
             "heldout closure differs")
    _require(heldout["labels_sealed_until_admission"] is True
             and heldout["tuning"] == "forbidden", "config",
             "heldout seal differs")
    reference = evaluation["exact_context_reference"]
    _require(reference["gate"] is False
             and reference["treatment_required_to_beat"] is False, "config",
             "exact context reference ordering differs")
    _require(reference["quality_metric"] == "case_accuracy"
             and reference["cost_metric"] == "context_tokens_per_query",
             "config", "exact context metrics differ")

    admission = config["admission"]
    fit_receipt = admission["fit_receipt"]
    _require(fit_receipt["format"] == FIT_RECEIPT_FORMAT, "config",
             "fit receipt format differs")
    _require(fit_receipt["required_before_development"] is True, "config",
             "fit receipt must be required")
    _require(tuple(sorted(fit_receipt["fields"])) == tuple(sorted(FIT_RECEIPT_FIELDS)),
             "config", "fit receipt fields differ")
    exact_values = fit_receipt["exact_values"]
    expected_exact = {
        "created_by": "future_authorized_runner",
        "dev_labels_read": False,
        "earlier_write_gradient": False,
        "heldout_opened": False,
        "mode": "fit",
        "optimizer": "NativeFusedAdamW",
        "optimizer_steps": 24,
        "protocol_id": PROTOCOL_ID,
        "replay_case_count": 12,
        "seed": SEED,
        "status": "passed",
        "train_case_count": 15,
        "tuning_performed": False,
    }
    _require(exact_values == expected_exact, "config",
             "fit receipt exact values differ")
    development = admission["development"]
    _require(development["format"] == DEVELOPMENT_RECEIPT_FORMAT, "config",
             "development receipt format differs")
    _require(development["requires_fit_receipt"] is True
             and development["one_shot"] is True
             and development["attempt"] == 1, "config",
             "development admission differs")
    _require(development["tuning_forbidden"] is True
             and development["heldout_opened"] is False, "config",
             "development closure differs")
    _require(admission["heldout"]["format"] == HELDOUT_RECEIPT_FORMAT,
             "config", "heldout receipt format differs")
    _require(admission["heldout"]["attempts"] == 1, "config",
             "heldout attempts differ")

    budget = config["budget"]
    preparation = budget["preparation"]
    for key in ("native_jobs", "model_loads", "tokenizer_loads",
                "optimizer_updates", "gradients", "checkpoints_written",
                "downloads"):
        _require(preparation[key] == 0, "config",
                 f"preparation.{key} must be zero")
    _require(preparation["device"] == "cpu" and preparation["network"] == "disabled",
             "config", "preparation must be offline CPU")
    ceiling = budget["authorized_run_ceiling"]
    expected_workload = _expected_planned_workload()
    for key, value in expected_workload.items():
        _require(ceiling[key] == value, "config",
                 f"authorized_run_ceiling.{key} must be {value!r}")
    _require(ceiling["model_loads"] == 1, "config", "model load ceiling differs")
    _require(ceiling["tokenizer_loads"] == 1, "config",
             "tokenizer load ceiling differs")
    _require(ceiling["peak_driver_bytes_max"] == GUARD_BYTES["peak_driver_bytes_max"],
             "config", "driver guard differs")
    _require(ceiling["sustained_swap_growth_bytes_max"]
             == GUARD_BYTES["sustained_swap_growth_bytes_max"], "config",
             "swap guard differs")
    _require(ceiling["min_host_free_fraction"]
             == GUARD_BYTES["min_host_free_fraction"], "config",
             "host-free guard differs")
    _require(ceiling["exclusive_worker"] is True
             and ceiling["cpu_fallback"] is False, "config",
             "worker exclusion or fallback differs")
    _require(ceiling["downloads"] == 0, "config", "downloads must be zero")
    derivation = _expected_workload_derivation()
    _require(ceiling["workload_derivation"] == derivation, "config",
             "workload derivation differs")
    _require(ceiling["optimizer_updates_total"]
             == derivation["total_optimizer_steps"], "config",
             "optimizer ceiling does not equal the arm derivation")
    _require(ceiling["backward_calls_total"]
             == sum(arm["backward_calls"] for arm in derivation["arms"]),
             "config", "backward ceiling does not equal the arm derivation")
    _require(ceiling["readout_forward_calls_total"]
             == derivation["readout_forward_calls_derivation"]["total"],
             "config", "forward ceiling does not equal its derivation")
    _require(ceiling["write_events_total"]
             == derivation["write_events_derivation"]["total"], "config",
             "write ceiling does not equal its derivation")
    _require(ceiling["checkpoint_writes"]
             == derivation["checkpoint_writes_derivation"]["total"], "config",
             "checkpoint ceiling does not equal its derivation")
    _require(derivation["fit_receipt_optimizer_steps"]
             + derivation["control_fit_optimizer_steps"]
             == derivation["total_optimizer_steps"], "config",
             "fit plus control steps do not equal the total")
    _require(derivation["independent_arm_clocks"]
             == len(derivation["arms"]), "config",
             "arm clocks do not match the trained arms")
    _require(derivation["shared_clock_across_arms"] is False
             and derivation["extra_steps_per_event"] == 0
             and derivation["one_step_per_teaching_event"] is True, "config",
             "the one-step clock is not preserved across arms")
    _require(fit_receipt["scope"] == _expected_fit_scope(), "config",
             "fit receipt scope is not the treatment arm")
    _require(fit_receipt["scope"]["optimizer_steps"]
             == fit_receipt["exact_values"]["optimizer_steps"], "config",
             "fit receipt scope and exact step count differ")
    _require(budget["data_limits"] == _expected_data_limits(), "config",
             "data limits differ")

    launch = config["launch"]
    _require(launch["review_only"] is True and launch["native_execution"] is False,
             "config", "launch must be review-only")
    _require(launch["requires_authorization_receipt"] is True, "config",
             "launch must require an authorization receipt")
    _require(launch["on_missing_receipt"] == "refuse_nonzero"
             and launch["on_receipt_without_separate_admission_runner"]
             == "refuse_nonzero", "config", "launch refusal policy differs")
    _require(launch["native_execution_status"] == "refused_preparation_only",
             "config", "native execution status differs")
    _require("preflight_value_readout_abstention_v1.py" in launch["command"]
             and "--review" in launch["command"], "config",
             "launch command is not the review-only preflight")
    _require(launch["receipt_format"] == PREFLIGHT_FORMAT, "config",
             "preflight receipt format differs")
    _require(launch["authorization_receipt_format"] == AUTHORIZATION_FORMAT,
             "config", "authorization receipt format differs")

    prepared = config["prepared"]
    _require(prepared["expected_counts"] == _expected_counts(), "config",
             "prepared counts differ")
    _require(prepared["loss_counts"] == {name: 6 for name in LOSS_NAMES},
             "config", "prepared loss counts differ")
    _require(set(prepared["files"]) == {
        "episodes", "cases_train", "cases_train_replay", "cases_dev",
        "cases_heldout", "labels_dev", "labels_heldout", "controls"}, "config",
        "prepared file keys differ")
    _require(config.get("limitations"), "config", "limitations are missing")


def _expected_counts() -> dict:
    return {
        "episodes": 7,
        "episodes_train": 3,
        "episodes_dev": 2,
        "episodes_heldout": 2,
        "events": 68,
        "events_train": 36,
        "events_dev": 16,
        "events_heldout": 16,
        "queries": 47,
        "write_events": 21,
        "cases": 47,
        "cases_train": 15,
        "cases_train_replay": 12,
        "cases_dev": 10,
        "cases_heldout": 10,
        "teaching_cases": 24,
        "teaching_cases_base": 12,
        "teaching_cases_replay": 12,
        "retention_diagnostic_cases": 3,
        "optimizer_steps": 24,
        "labels_dev": 10,
        "labels_heldout": 10,
    }


# ---------------------------------------------------------------------------
# Prepared fixture validation
# ---------------------------------------------------------------------------

def _check_manifest_digest(manifest: Mapping[str, Any]) -> None:
    digest = manifest.get("manifest_sha256")
    _require(_is_hex64(digest), "manifest", "manifest digest is missing")
    body = {key: value for key, value in manifest.items()
            if key != "manifest_sha256"}
    _require(sha256_json(body) == digest, "manifest",
             "manifest digest differs")


def _expected_epoch_specs(split: str) -> dict[int, dict]:
    specs = {
        0: {"kind": "observation", "role": "initial", "case_type": None},
        1: {"kind": "query", "role": None, "case_type": "future_answer"},
        2: {"kind": "observation", "role": "secondary", "case_type": None},
        3: {"kind": "correction", "role": "correction", "case_type": None},
        4: {"kind": "query", "role": None, "case_type": "correction"},
        5: {"kind": "query", "role": None, "case_type": "unaffected_retention"},
        6: {"kind": "query", "role": None, "case_type": "unknown_abstention"},
        7: {"kind": "query", "role": None, "case_type": "wrong_world_abstention"},
    }
    if split == "train":
        specs.update({
            8: {"kind": "query", "role": None, "case_type": "future_answer"},
            9: {"kind": "query", "role": None, "case_type": "correction"},
            10: {"kind": "query", "role": None, "case_type": "unknown_abstention"},
            11: {"kind": "query", "role": None, "case_type": "wrong_world_abstention"},
        })
    return specs


def _expected_teaching(split: str, case_type: str) -> bool:
    return split == "train" and case_type in TEACHING_CASE_TYPES


def _check_episode_events(events: Sequence[Mapping[str, Any]]) -> dict[str, list[dict]]:
    by_episode: dict[str, list[dict]] = {}
    event_ids: set[str] = set()
    for event in events:
        _require(event.get("format") == EVENT_FORMAT, "event", "event format differs")
        event_id = event["event_id"]
        _require(event_id not in event_ids, "event", f"duplicate event {event_id}")
        event_ids.add(event_id)
        by_episode.setdefault(event["episode_id"], []).append(dict(event))
    _require(len(by_episode) == 7, "event", "episode count differs")
    for episode_id, rows in by_episode.items():
        rows.sort(key=lambda row: int(row["epoch"]))
        world_id = rows[0]["world_id"]
        split = _split_for_world(world_id)
        _require(len({row["world_id"] for row in rows}) == 1, "event",
                 f"{episode_id} spans worlds")
        _require(len({row["split"] for row in rows}) == 1, "event",
                 f"{episode_id} spans splits")
        specs = _expected_epoch_specs(split)
        _require(len(rows) == len(specs), "event",
                 f"{episode_id} has {len(rows)} events, expected {len(specs)}")
        writes = 0
        for index, row in enumerate(rows):
            expected = specs.get(index)
            _require(expected is not None, "event",
                     f"{episode_id} has an unexpected epoch {index}")
            _require(int(row["epoch"]) == index, "event",
                     f"{episode_id} epoch {index} is out of order")
            _require(int(row["clock_step"]) == index, "event",
                     f"{episode_id} clock step differs at epoch {index}")
            _require(row["kind"] == expected["kind"], "event",
                     f"{episode_id} epoch {index} kind differs")
            if expected["case_type"]:
                _require(row["case_type"] == expected["case_type"], "event",
                         f"{episode_id} epoch {index} case type differs")
                _require(row["replay"] == (split == "train" and index >= BASE_EPOCHS),
                         "event", f"{episode_id} epoch {index} replay flag differs")
                teaching = _expected_teaching(split, expected["case_type"])
                _require(int(row["optimizer_steps"]) == (1 if teaching else 0),
                         "event", f"{episode_id} epoch {index} step count differs")
                _require(row["gradient"] is bool(teaching), "event",
                         f"{episode_id} epoch {index} gradient flag differs")
                _require(row["loss_name"] == (LOSS_BY_CASE[expected["case_type"]]
                                              if teaching else None), "event",
                         f"{episode_id} epoch {index} loss name differs")
                _require(row["expert_id"] == 0, "event",
                         f"{episode_id} epoch {index} expert differs")
                _require(_is_hex64(row["feature_digest"]), "event",
                         f"{episode_id} epoch {index} feature digest differs")
            else:
                writes += 1
                _require(int(row["optimizer_steps"]) == 0
                         and row["gradient"] is False, "event",
                         f"{episode_id} write epoch {index} has a gradient")
                record = row["record"]
                _require(record["world_id"] == world_id, "event",
                         f"{episode_id} record world differs")
                _require(int(record["source_epoch"]) == index, "event",
                         f"{episode_id} record source epoch differs")
                _require(record["value"] in build_answer_vocabulary(), "event",
                         f"{episode_id} record value is outside the vocabulary")
                if expected["role"] == "initial":
                    _require(record.get("supersedes") is None, "event",
                             f"{episode_id} initial record supersedes another")
                if expected["role"] == "secondary":
                    _require(record.get("supersedes") is None, "event",
                             f"{episode_id} secondary record supersedes another")
                if expected["role"] == "correction":
                    _require(record.get("supersedes") ==
                             f"record_{world_id.removeprefix('world_')}_primary_initial",
                             "event", f"{episode_id} correction supersedes the wrong record")
        _require(writes == WRITES_PER_EPISODE, "event",
                 f"{episode_id} write count differs")
    return by_episode


def _check_case_against_event(case: Mapping[str, Any], event: Mapping[str, Any]) -> None:
    _require(case.get("format") == CASE_FORMAT, "case", "case format differs")
    for key in ("case_id", "case_type", "split", "replay", "replay_of_case_id",
                "episode_id", "world_id", "event_id", "epoch", "query_id",
                "query_text", "address", "query_form", "expert_id",
                "optimizer_steps", "loss_name", "feature_digest"):
        _require(case.get(key) == event.get(key), "case",
                 f"{case['case_id']} field {key} differs from its event")


def _check_label(case: Mapping[str, Any], label: Mapping[str, Any],
                 derived: Mapping[str, Any], check: str) -> None:
    _require(label.get("format") == LABEL_FORMAT, check, "label format differs")
    _require(label["case_id"] == case["case_id"], check, "label case id differs")
    _require(label["case_type"] == case["case_type"], check,
             "label case type differs")
    _require(label["split"] == case["split"], check, "label split differs")
    _require(int(label["epoch"]) == int(case["epoch"]), check,
             "label epoch differs")
    for key in ("target", "label_source_epoch", "authority",
                "source_record_id", "stale_record_id", "stale_value",
                "rejected_world_id", "rejected_record_id",
                "rejected_record_source_epoch"):
        _require(label.get(key) == derived.get(key), check,
                 f"{case['case_id']} label {key} differs")


def _check_target_semantics(case: Mapping[str, Any], target: Mapping[str, Any],
                            case_type: str) -> None:
    if case_type in ("unknown_abstention", "wrong_world_abstention"):
        _require(target["kind"] == "status" and target["token"] == "UNKNOWN",
                 "label", f"{case['case_id']} does not abstain")
    else:
        _require(target["kind"] == "value" and target["token"] != "UNKNOWN",
                 "label", f"{case['case_id']} does not return a value")
    if case_type == "correction":
        _require(case.get("stale_value") is not None
                 and case["stale_value"] != case["target"]["token"], "label",
                 f"{case['case_id']} correction lacks a distinct stale value")
    if case_type == "wrong_world_abstention":
        _require(case.get("rejected_world_id") is not None, "label",
                 f"{case['case_id']} wrong-world rejection is missing")


def _validate_cases_and_labels(
        episodes: Mapping[str, Sequence[Mapping[str, Any]]],
        train_rows: Sequence[Mapping[str, Any]],
        replay_rows: Sequence[Mapping[str, Any]],
        dev_rows: Sequence[Mapping[str, Any]],
        heldout_rows: Sequence[Mapping[str, Any]],
        dev_label_rows: Sequence[Mapping[str, Any]],
        heldout_label_rows: Sequence[Mapping[str, Any]],
        config: Mapping[str, Any]) -> dict:
    events_by_case: dict[str, Mapping[str, Any]] = {}
    records: list[dict] = []
    for rows in episodes.values():
        for event in rows:
            if event["kind"] == "query":
                _require(event["case_id"] not in events_by_case, "case",
                         f"duplicate query case {event['case_id']}")
                events_by_case[event["case_id"]] = event
            else:
                records.append(dict(event["record"]))
    vocabulary = config["architecture"]["answer_vocabulary"]
    seen: set[str] = set()
    derived_targets: dict[str, dict] = {}
    counts = {split: {case_type: 0 for case_type in CASE_TYPES}
              for split in ("train", "dev", "heldout")}

    def check_split(rows: Sequence[Mapping[str, Any]], split: str,
                    replay_file: bool, include_target: bool) -> None:
        for case in rows:
            case_id = case["case_id"]
            _require(case_id not in seen, "case", f"duplicate case {case_id}")
            seen.add(case_id)
            _require(case["split"] == split, "case",
                     f"{case_id} split differs from its file")
            _require(case["replay"] is replay_file, "case",
                     f"{case_id} replay flag differs from its file")
            _require(case["case_type"] in CASE_TYPES, "case",
                     f"{case_id} case type is unknown")
            _require(_is_hex64(case["feature_digest"]), "case",
                     f"{case_id} feature digest differs")
            event = events_by_case.get(case_id)
            _require(event is not None, "case", f"{case_id} has no query event")
            _check_case_against_event(case, event)
            query = {
                "case_id": case_id,
                "case_type": case["case_type"],
                "world_id": case["world_id"],
                "epoch": int(case["epoch"]),
                "address": case["address"],
                "wrong_world_id": event.get("wrong_world_id"),
            }
            derived = _derive_target(records, query, vocabulary)
            derived_targets[case_id] = derived
            expected_steps = 1 if _expected_teaching(split, case["case_type"]) else 0
            _require(int(case["optimizer_steps"]) == expected_steps, "case",
                     f"{case_id} optimizer steps differ")
            _require(case["loss_name"] == (LOSS_BY_CASE[case["case_type"]]
                                           if expected_steps else None), "case",
                     f"{case_id} loss name differs")
            expected_feature = _feature_digest(
                int(config["transaction"]["seed"]), case_id, case["world_id"],
                int(case["epoch"]), case["address"], int(case["query_form"]),
                derived["source_record_id"], derived["target"])
            _require(case["feature_digest"] == expected_feature, "case",
                     f"{case_id} feature digest is not deterministic")
            counts[split][case["case_type"]] += 1
            if include_target:
                for key in ("target", "label_source_epoch", "authority",
                            "source_record_id", "stale_record_id", "stale_value",
                            "rejected_world_id", "rejected_record_id",
                            "rejected_record_source_epoch"):
                    _require(key in case, "case",
                             f"{case_id} lacks label field {key}")
                _check_label(case, {**case, "format": LABEL_FORMAT},
                             derived, "case")
                _check_target_semantics(case, derived["target"], case["case_type"])
            else:
                leaked = FORBIDDEN_EVAL_CASE_FIELDS & set(case)
                _require(not leaked, "case",
                         f"{case_id} evaluation case leaks label fields {sorted(leaked)}")

    check_split(train_rows, "train", False, True)
    check_split(replay_rows, "train", True, True)
    check_split(dev_rows, "dev", False, False)
    check_split(heldout_rows, "heldout", False, False)

    base_cases = {case["case_id"]: case for case in train_rows}
    for case in replay_rows:
        _require(case["replay_of_case_id"] in base_cases, "case",
                 f"{case['case_id']} does not reference a base training case")
        _require(base_cases[case["replay_of_case_id"]]["case_type"]
                 == case["case_type"], "case",
                 f"{case['case_id']} replay case type differs from its base")
        _require(int(case["epoch"]) > int(
            base_cases[case["replay_of_case_id"]]["epoch"]), "case",
                 f"{case['case_id']} replay does not follow its base")

    for label_rows, split, case_rows in (
            (dev_label_rows, "dev", dev_rows),
            (heldout_label_rows, "heldout", heldout_rows)):
        expected_ids = {case["case_id"] for case in case_rows}
        label_ids = {label["case_id"] for label in label_rows}
        _require(label_ids == expected_ids, "label",
                 f"{split} labels do not exactly cover its cases")
        for label in label_rows:
            case = next(item for item in case_rows
                        if item["case_id"] == label["case_id"])
            derived = derived_targets[case["case_id"]]
            _check_label(case, label, derived, "label")
            _check_target_semantics(label, derived["target"], case["case_type"])

    return {"counts": counts, "targets": derived_targets,
            "records": records, "events_by_case": events_by_case}


def _check_disjointness_from_validation(
        episodes: Mapping[str, Sequence[Mapping[str, Any]]],
        train_rows: Sequence[Mapping[str, Any]],
        replay_rows: Sequence[Mapping[str, Any]],
        dev_rows: Sequence[Mapping[str, Any]],
        heldout_rows: Sequence[Mapping[str, Any]],
        targets: Mapping[str, Mapping[str, Any]]) -> dict:
    world_sets = {split: {event["world_id"] for event in events}
                  for split, events in
                  (("train", [row for rows in episodes.values() for row in rows
                              if row["split"] == "train"]),
                   ("dev", [row for rows in episodes.values() for row in rows
                            if row["split"] == "dev"]),
                   ("heldout", [row for rows in episodes.values() for row in rows
                                if row["split"] == "heldout"]))}
    train_ids = {case["case_id"] for case in train_rows}
    replay_ids = {case["case_id"] for case in replay_rows}
    dev_ids = {case["case_id"] for case in dev_rows}
    heldout_ids = {case["case_id"] for case in heldout_rows}
    train_digests = {_semantic_digest(case, targets[case["case_id"]])
                     for case in list(train_rows) + list(replay_rows)}
    eval_digests = {_semantic_digest(case, targets[case["case_id"]])
                    for case in list(dev_rows) + list(heldout_rows)}
    result = {
        "train_worlds_disjoint_from_dev": not (world_sets["train"] & world_sets["dev"]),
        "train_worlds_disjoint_from_heldout": not (world_sets["train"] & world_sets["heldout"]),
        "dev_worlds_disjoint_from_heldout": not (world_sets["dev"] & world_sets["heldout"]),
        "train_cases_disjoint_from_dev": not (train_ids & dev_ids),
        "train_cases_disjoint_from_heldout": not (train_ids & heldout_ids),
        "replay_cases_disjoint_from_dev": not (replay_ids & dev_ids),
        "replay_cases_disjoint_from_heldout": not (replay_ids & heldout_ids),
        "all_case_ids_unique": (len(train_ids) + len(replay_ids)
                                + len(dev_ids) + len(heldout_ids)
                                == len(train_ids | replay_ids | dev_ids | heldout_ids)),
        "training_semantics_disjoint_from_evaluation": not (train_digests & eval_digests),
        "heldout_labels_sealed": True,
    }
    for key, value in result.items():
        _require(value is True, "disjointness", f"{key} is false")
    return result


def _check_controls(rows: Sequence[Mapping[str, Any]],
                    config: Mapping[str, Any]) -> dict:
    _require([row["arm_id"] for row in rows] == list(CONTROL_ARM_IDS), "control",
             "control rows differ")
    by_id = {arm["arm_id"]: arm for arm in config["controls"]["arms"]}
    for row in rows:
        _require(row.get("format") == CONTROL_FORMAT, "control",
                 "control format differs")
        arm = by_id[row["arm_id"]]
        for key, value in arm.items():
            _require(row.get(key) == value, "control",
                     f"control {row['arm_id']} field {key} differs")
        _require(row.get("expected_outcome"), "control",
                 f"control {row['arm_id']} lacks an expected outcome")
    _require(len({row["arm_id"] for row in rows}) == 7, "control",
             "control arm ids are not unique")
    return {"arm_ids": [row["arm_id"] for row in rows], "count": len(rows)}


def validate_prepared(prepared_dir: Path = PREPARED_DIR,
                      config_path: Path = CONFIG_PATH) -> dict:
    """Validate the complete fixture, labels, controls, budgets and closure."""
    prepared = Path(prepared_dir)
    config_file = Path(config_path)
    config = read_json(config_file)
    validate_config(config, config_file)
    manifest = read_json(prepared / "manifest.json")
    _require(manifest.get("format") == MANIFEST_FORMAT, "manifest",
             "manifest format differs")
    _require(manifest.get("protocol_id") == PROTOCOL_ID, "manifest",
             "protocol id differs")
    _check_manifest_digest(manifest)
    _require(manifest["config"]["sha256"] == sha256_file(config_file), "manifest",
             "manifest config hash differs")
    _require(manifest["config"]["path"] == _relative_config_path(config_file),
             "manifest", "manifest config path differs")
    file_names = config["prepared"]["files"]
    _require(set(manifest["files"]) == set(file_names.values()), "manifest",
             "manifest file set differs")
    for name in file_names.values():
        _require(sha256_file(prepared / name) == manifest["files"][name],
                 "manifest", f"file hash differs: {name}")

    rows = {key: read_jsonl(prepared / name) for key, name in file_names.items()}
    episodes = _check_episode_events(rows["episodes"])
    case_check = _validate_cases_and_labels(
        episodes, rows["cases_train"], rows["cases_train_replay"],
        rows["cases_dev"], rows["cases_heldout"], rows["labels_dev"],
        rows["labels_heldout"], config)
    counts = _count_rows(rows["episodes"], rows["cases_train"],
                         rows["cases_train_replay"], rows["cases_dev"],
                         rows["cases_heldout"], rows["labels_dev"],
                         rows["labels_heldout"])
    _require(counts == config["prepared"]["expected_counts"], "manifest",
             "validated counts differ from config")
    _require(counts == manifest["counts"], "manifest",
             "validated counts differ from the manifest")
    loss_counts = _loss_counts(rows["cases_train"] + rows["cases_train_replay"])
    _require(loss_counts == config["prepared"]["loss_counts"], "loss",
             "validated loss counts differ")
    _require(loss_counts == manifest["loss_counts"], "manifest",
             "manifest loss counts differ")

    disjoint = _check_disjointness_from_validation(
        episodes, rows["cases_train"], rows["cases_train_replay"],
        rows["cases_dev"], rows["cases_heldout"], case_check["targets"])
    _require(disjoint == manifest["disjointness"], "manifest",
             "manifest disjointness differs")
    _require(manifest["splits"] == _split_counts(
        rows["episodes"], rows["cases_train"], rows["cases_train_replay"],
        rows["cases_dev"], rows["cases_heldout"]), "manifest",
        "manifest split counts differ")
    _require(manifest["answer_vocabulary"]
             == config["architecture"]["answer_vocabulary"], "manifest",
             "manifest vocabulary differs")
    _require(manifest["label_seal"]["dev_labels_sha256"]
             == sha256_file(prepared / file_names["labels_dev"]), "manifest",
             "dev label seal differs")
    _require(manifest["label_seal"]["heldout_labels_sha256"]
             == sha256_file(prepared / file_names["labels_heldout"]), "manifest",
             "heldout label seal differs")
    _require(manifest["label_seal"]["dev_label_cases"] == 10
             and manifest["label_seal"]["heldout_label_cases"] == 10,
             "manifest", "label case counts differ")
    _require(manifest["label_seal"]["heldout_opened"] is False, "manifest",
             "heldout must remain closed")
    _require(manifest["heldout_opened"] is False, "manifest",
             "manifest opened heldout")
    _require(manifest["native_computation"] is False
             and manifest["model_loaded"] is False
             and manifest["tokenizer_loaded"] is False
             and manifest["downloads"] == 0, "manifest",
             "manifest claims native work")
    _require(manifest["deterministic"] is True, "manifest",
             "manifest is not deterministic")
    _require(manifest["fit_receipt_requirements"]
             == config["admission"]["fit_receipt"], "manifest",
             "fit receipt requirements differ")
    _require(manifest["controls"]["arm_ids"] == list(CONTROL_ARM_IDS)
             and manifest["controls"]["count"] == 7, "manifest",
             "manifest controls differ")

    controls = _check_controls(rows["controls"], config)
    planned = _planned_workload(config, counts)
    _require(planned == _expected_planned_workload(), "budget",
             "computed planned workload differs")
    _require(planned == manifest["planned_workload"], "manifest",
             "manifest planned workload differs")
    _require(manifest["workload_derivation"]
             == config["budget"]["authorized_run_ceiling"]["workload_derivation"],
             "manifest", "manifest workload derivation differs")
    _require(planned["optimizer_updates_total"]
             == manifest["workload_derivation"]["total_optimizer_steps"],
             "manifest", "planned updates differ from the arm derivation")
    ceiling = config["budget"]["authorized_run_ceiling"]
    for key, value in planned.items():
        _require(ceiling[key] == value, "budget",
                 f"authorized ceiling {key} differs from the fixture")
    _require(sum((prepared / name).stat().st_size for name in file_names.values())
             <= config["budget"]["data_limits"]["prepared_bytes_max"], "budget",
             "prepared fixture exceeds its byte cap")
    _require(counts["optimizer_steps"] == counts["teaching_cases"], "schedule",
             "optimizer steps do not equal teaching cases")
    _require(counts["events"] == 68 and counts["queries"] == 47, "schedule",
             "event counts differ")

    report_body = {
        "format": VALIDATION_FORMAT,
        "protocol_id": PROTOCOL_ID,
        "status": "valid",
        "config_sha256": sha256_file(config_file),
        "prepared_manifest_sha256": sha256_file(prepared / "manifest.json"),
        "files": dict(manifest["files"]),
        "counts": counts,
        "splits": manifest["splits"],
        "loss_counts": loss_counts,
        "controls": controls,
        "disjointness": disjoint,
        "planned_workload": planned,
        "workload_derivation": manifest["workload_derivation"],
        "native_computation": False,
        "model_loaded": False,
        "tokenizer_loaded": False,
        "optimizer_updates": 0,
        "gradients": 0,
        "checkpoints_written": 0,
        "downloads": 0,
        "heldout_opened": False,
        "authorization_required": True,
        "checks": [
            "config",
            "manifest",
            "file_hashes",
            "chronology",
            "one_step_schedule",
            "derived_labels",
            "evaluation_label_seal",
            "train_evaluation_disjointness",
            "control_distinctness",
            "budget",
            "heldout_closure",
            "fit_receipt_admission",
        ],
    }
    report_body["validation_sha256"] = sha256_json(report_body)
    return report_body


# ---------------------------------------------------------------------------
# Receipt admission
# ---------------------------------------------------------------------------

def seal_receipt(receipt: Mapping[str, Any]) -> dict:
    body = {key: value for key, value in receipt.items()
            if key != "receipt_sha256"}
    sealed = dict(body)
    sealed["receipt_sha256"] = sha256_json(body)
    return sealed


def _check_receipt_digest(receipt: Mapping[str, Any], check: str) -> None:
    _require(_is_hex64(receipt.get("receipt_sha256")), check,
             "receipt digest is missing")
    body = {key: value for key, value in receipt.items()
            if key != "receipt_sha256"}
    _require(sha256_json(body) == receipt["receipt_sha256"], check,
             "receipt digest differs")


def _metric_thresholds(config: Mapping[str, Any], group: str) -> dict:
    return dict(config["evaluation"][group]["thresholds"])


def _metric_names_from_thresholds(thresholds: Mapping[str, Any]) -> tuple[str, ...]:
    names = []
    for key in thresholds:
        names.append(key[:-4] if key.endswith("_min") else key[:-4])
    return tuple(sorted(names))


def _check_metrics(metrics: Mapping[str, Any], thresholds: Mapping[str, Any],
                   check: str) -> bool:
    _require(set(metrics) == set(_metric_names_from_thresholds(thresholds)),
             check, "metric keys differ")
    passed = True
    for key, limit in thresholds.items():
        name = key[:-4]
        value = _finite_number(metrics[name], check, name)
        if key.endswith("_min"):
            passed = passed and value >= float(limit)
        else:
            passed = passed and value <= float(limit)
    return passed


def admit_fit_receipt(receipt: Mapping[str, Any],
                      prepared_dir: Path = PREPARED_DIR,
                      config_path: Path = CONFIG_PATH) -> dict:
    """Fail closed unless a passing fit receipt binds this exact fixture."""
    report = validate_prepared(prepared_dir, config_path)
    config = read_json(config_path)
    expected = config["admission"]["fit_receipt"]
    check = "fit_receipt"
    _require(isinstance(receipt, Mapping), check, "receipt must be an object")
    _require(set(receipt) == set(FIT_RECEIPT_FIELDS) | {"receipt_sha256"}, check,
             "receipt fields differ from the declared schema")
    _check_receipt_digest(receipt, check)
    _require(receipt["format"] == FIT_RECEIPT_FORMAT, check,
             "receipt format differs")
    _require(receipt["protocol_id"] == PROTOCOL_ID, check,
             "receipt protocol differs")
    for key, value in expected["exact_values"].items():
        _require(receipt.get(key) == value, check,
                 f"receipt {key} must be {value!r}")
    _require(receipt["config_sha256"] == report["config_sha256"], check,
             "receipt config hash differs")
    _require(receipt["prepared_manifest_sha256"]
             == report["prepared_manifest_sha256"], check,
             "receipt manifest hash differs")
    _require(receipt["losses"] == list(LOSS_NAMES), check,
             "receipt losses differ")
    _require(receipt["trainable_parameters"]
             == config["architecture"]["trainable_parameters"], check,
             "receipt trainable parameters differ")
    artifacts = receipt["artifacts"]
    _require(isinstance(artifacts, Mapping)
             and set(artifacts) == {"checkpoint_sha256", "fit_report_sha256"},
             check, "receipt artifacts differ")
    for key, value in artifacts.items():
        _require(_is_hex64(value), check, f"artifact {key} is not a SHA-256")
    fit_gates = receipt["fit_gates"]
    _require(isinstance(fit_gates, Mapping)
             and set(fit_gates) == {"all_declared_gates_passed",
                                    "local_associative_loss_gate",
                                    "metrics", "thresholds"}, check,
             "fit gate report differs")
    thresholds = _metric_thresholds(config, "fit")
    _require(fit_gates["thresholds"] == thresholds, check,
             "fit gate thresholds differ")
    computed = _check_metrics(fit_gates["metrics"], thresholds, check)
    _require(fit_gates["all_declared_gates_passed"] is True
             and computed is True, check, "fit gates did not pass")
    _require(fit_gates["local_associative_loss_gate"] is False, check,
             "fit receipt gates a local associative loss")
    return {
        "admitted": True,
        "fit_receipt_sha256": receipt["receipt_sha256"],
        "protocol_id": PROTOCOL_ID,
        "config_sha256": report["config_sha256"],
        "prepared_manifest_sha256": report["prepared_manifest_sha256"],
        "optimizer_steps": receipt["optimizer_steps"],
        "heldout_opened": False,
        "dev_labels_read": False,
        "tuning_performed": False,
    }


def admit_development(receipt: Mapping[str, Any], *,
                      fit_receipt: Mapping[str, Any],
                      prepared_dir: Path = PREPARED_DIR,
                      config_path: Path = CONFIG_PATH) -> dict:
    """Admit one development evaluation only after a passing fit receipt."""
    fit = admit_fit_receipt(fit_receipt, prepared_dir, config_path)
    config = read_json(config_path)
    expected = config["admission"]["development"]
    check = "development_receipt"
    _require(isinstance(receipt, Mapping), check, "receipt must be an object")
    _require(set(receipt) == set(DEVELOPMENT_RECEIPT_FIELDS) | {"receipt_sha256"},
             check, "receipt fields differ from the declared schema")
    _check_receipt_digest(receipt, check)
    _require(receipt["format"] == expected["format"], check,
             "receipt format differs")
    _require(receipt["protocol_id"] == PROTOCOL_ID, check,
             "receipt protocol differs")
    _require(receipt["status"] == "passed", check, "development did not pass")
    _require(receipt["mode"] == "dev-eval", check, "receipt mode differs")
    _require(receipt["created_by"] == "future_authorized_runner", check,
             "receipt creator differs")
    _require(receipt["attempt"] == expected["attempt"], check,
             "development attempt differs")
    _require(receipt["fit_receipt_sha256"] == fit["fit_receipt_sha256"], check,
             "development does not bind the admitted fit receipt")
    _require(receipt["config_sha256"] == fit["config_sha256"], check,
             "development config hash differs")
    _require(receipt["prepared_manifest_sha256"]
             == fit["prepared_manifest_sha256"], check,
             "development manifest hash differs")
    _require(receipt["heldout_opened"] is False, check,
             "development opened heldout")
    _require(receipt["tuning_performed"] is False, check,
             "development performed tuning")
    thresholds = _metric_thresholds(config, "dev")
    computed = _check_metrics(receipt["dev_metrics"], thresholds, check)
    _require(receipt["all_declared_gates_passed"] is True and computed is True,
             check, "development gates did not pass")
    return {
        "admitted": True,
        "development_receipt_sha256": receipt["receipt_sha256"],
        "fit_receipt_sha256": fit["fit_receipt_sha256"],
        "heldout_opened": False,
        "attempt": receipt["attempt"],
    }


def admit_heldout(receipt: Mapping[str, Any], *,
                  development_receipt: Mapping[str, Any],
                  fit_receipt: Mapping[str, Any],
                  prepared_dir: Path = PREPARED_DIR,
                  config_path: Path = CONFIG_PATH) -> dict:
    """Admit the single post-development heldout evaluation."""
    development = admit_development(
        development_receipt, fit_receipt=fit_receipt,
        prepared_dir=prepared_dir, config_path=config_path)
    config = read_json(config_path)
    expected = config["admission"]["heldout"]
    check = "heldout_receipt"
    _require(isinstance(receipt, Mapping), check, "receipt must be an object")
    _require(set(receipt) == set(HELDOUT_RECEIPT_FIELDS) | {"receipt_sha256"},
             check, "receipt fields differ from the declared schema")
    _check_receipt_digest(receipt, check)
    _require(receipt["format"] == expected["format"], check,
             "receipt format differs")
    _require(receipt["protocol_id"] == PROTOCOL_ID, check,
             "receipt protocol differs")
    _require(receipt["status"] == "passed", check, "heldout did not pass")
    _require(receipt["mode"] == "heldout-eval", check, "receipt mode differs")
    _require(receipt["created_by"] == "future_authorized_runner", check,
             "receipt creator differs")
    _require(receipt["attempt"] == expected["attempts"], check,
             "heldout attempt differs")
    _require(receipt["dev_receipt_sha256"]
             == development["development_receipt_sha256"], check,
             "heldout does not bind the admitted development receipt")
    _require(receipt["config_sha256"] == sha256_file(config_path), check,
             "heldout config hash differs")
    _require(receipt["prepared_manifest_sha256"]
             == validate_prepared(prepared_dir, config_path)["prepared_manifest_sha256"],
             check, "heldout manifest hash differs")
    _require(receipt["tuning_performed"] is False, check,
             "heldout performed tuning")
    thresholds = _metric_thresholds(config, "dev")
    computed = _check_metrics(receipt["heldout_metrics"], thresholds, check)
    _require(computed is True, check, "heldout gates did not pass")
    return {
        "admitted": True,
        "heldout_receipt_sha256": receipt["receipt_sha256"],
        "development_receipt_sha256": development["development_receipt_sha256"],
        "attempt": receipt["attempt"],
        "heldout_opened": True,
    }


# ---------------------------------------------------------------------------
# Review-only preflight and native refusal
# ---------------------------------------------------------------------------

def _check_authorization(receipt: Mapping[str, Any], config: Mapping[str, Any],
                         report: Mapping[str, Any]) -> None:
    check = "authorization"
    fields = config["launch"]["authorization_fields"]
    _require(isinstance(receipt, Mapping), check, "authorization must be an object")
    _require(set(receipt) == set(fields), check,
             "authorization fields differ")
    _check_receipt_digest(receipt, check)
    _require(receipt["format"] == AUTHORIZATION_FORMAT, check,
             "authorization format differs")
    _require(receipt["protocol_id"] == PROTOCOL_ID, check,
             "authorization protocol differs")
    _require(receipt["config_sha256"] == report["config_sha256"], check,
             "authorization config hash differs")
    _require(receipt["prepared_manifest_sha256"]
             == report["prepared_manifest_sha256"], check,
             "authorization manifest hash differs")
    _require(isinstance(receipt["authorized_by"], str)
             and receipt["authorized_by"], check, "authorization lacks an author")
    _require(receipt["native_execution_allowed"] is True, check,
             "authorization does not allow native execution")
    _require(receipt["single_attempt"] is True, check,
             "authorization must be single-attempt")


def preflight_report(*, config_path: Path = CONFIG_PATH,
                     prepared_dir: Path = PREPARED_DIR,
                     receipt_path: Path,
                     authorization_path: Path | None = None,
                     launch: bool = False) -> dict:
    """Write a fresh review receipt; never execute native work."""
    report = validate_prepared(prepared_dir, config_path)
    config = read_json(config_path)
    target = Path(receipt_path)
    _require(not target.exists(), "preflight",
             f"receipt path is not fresh: {target}")
    authorization_present = authorization_path is not None
    authorization_sha256 = None
    if launch:
        if authorization_path is None:
            launch_status = "refused_missing_authorization_receipt"
            exit_code = 3
        else:
            authorization = read_json(Path(authorization_path))
            _check_authorization(authorization, config, report)
            authorization_sha256 = sha256_file(Path(authorization_path))
            launch_status = "refused_preparation_only"
            exit_code = 3
    else:
        launch_status = "not_requested"
        exit_code = 0
    body = {
        "format": PREFLIGHT_FORMAT,
        "protocol_id": PROTOCOL_ID,
        "status": "reviewed" if exit_code == 0 else "refused",
        "config_sha256": report["config_sha256"],
        "prepared_manifest_sha256": report["prepared_manifest_sha256"],
        "validation_sha256": report["validation_sha256"],
        "review_only": True,
        "native_execution": False,
        "launch_requested": bool(launch),
        "authorization_present": authorization_present,
        "authorization_sha256": authorization_sha256,
        "launch_status": launch_status,
        "native_execution_status": config["launch"]["native_execution_status"],
        "exit_code": exit_code,
        "receipt_path": str(target),
    }
    body["receipt_sha256"] = sha256_json(body)
    _write_json_atomic(target, body)
    return body

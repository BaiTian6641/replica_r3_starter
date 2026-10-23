"""Small host-owned action boundary; no model, learner or external side effects.

Host snapshots are trusted inputs, not authenticated by this module. The journal
is an integrity/replay artifact, not an authenticity or historical-erasure system.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re


VERSION = "replica.host_action_contract.v1"
TOOL = "register_xor8.v1"
ZERO = "0" * 64
MODES = {"source_snapshot", "source_current", "parallel_replica", "assistant_for_source"}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def _fields(value, names):
    if not isinstance(value, dict) or set(value) != set(names):
        raise ValueError(f"expected exactly these fields: {sorted(names)}")


def _text(value, maximum=512):
    if not isinstance(value, str) or not value.strip() or len(value.encode()) > maximum:
        raise ValueError("expected bounded nonempty text")


def _integer(value, maximum=1_000_000):
    if type(value) is not int or not 0 <= value <= maximum:
        raise ValueError("expected bounded nonnegative integer")


def _sha(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("expected lowercase SHA256")


def _ids(value, maximum=16):
    if not isinstance(value, (list, tuple)) or not 0 < len(value) <= maximum:
        raise ValueError("expected bounded nonempty ID list")
    for item in value:
        _text(item)
    if len(value) != len(set(value)):
        raise ValueError("duplicate ID")


@dataclass(frozen=True)
class HostBinding:
    model_manifest_sha256: str
    policy_sha256: str
    identity_mode: str
    subject_id: str
    branch_id: str
    agent_id: str
    goal: str
    purpose: str
    observation_cutoff: int
    allowed_source_ids: tuple[str, ...]
    max_decisions: int = 8
    max_tool_calls: int = 2

    def __post_init__(self):
        _sha(self.model_manifest_sha256)
        _sha(self.policy_sha256)
        if self.identity_mode not in MODES or self.purpose != "local_evaluation":
            raise ValueError("unsupported identity mode or pilot purpose")
        for name in ("subject_id", "branch_id", "agent_id", "goal"):
            _text(getattr(self, name))
        _integer(self.observation_cutoff)
        if not isinstance(self.allowed_source_ids, tuple):
            raise ValueError("source allowlist must be immutable")
        _ids(self.allowed_source_ids)
        for value in (self.max_decisions, self.max_tool_calls):
            _integer(value, 128)
        if not self.max_decisions:
            raise ValueError("a positive decision budget is required")

    @classmethod
    def from_dict(cls, value):
        _fields(value, cls.__dataclass_fields__)
        return cls(**{**value, "allowed_source_ids": tuple(value["allowed_source_ids"])})


@dataclass(frozen=True)
class HostObservation:
    source_id: str
    revision: int
    subject_id: str
    branch_id: str
    observed_at: int
    text: str
    sha256: str
    eligible: bool = True
    purpose: str = "local_evaluation"

    def __post_init__(self):
        for name in ("source_id", "subject_id", "branch_id", "purpose"):
            _text(getattr(self, name))
        _integer(self.revision)
        _integer(self.observed_at)
        _text(self.text, 4096)
        _sha(self.sha256)
        if type(self.eligible) is not bool:
            raise ValueError("eligibility must be boolean")
        if hashlib.sha256(self.text.encode()).hexdigest() != self.sha256:
            raise ValueError("observation bytes disagree with hash")

    @classmethod
    def make(cls, source_id, text, *, revision=1, subject_id="fictional-subject",
             branch_id="source", observed_at=1, eligible=True, purpose="local_evaluation"):
        return cls(source_id, revision, subject_id, branch_id, observed_at, text,
                   hashlib.sha256(text.encode()).hexdigest(), eligible, purpose)

    @classmethod
    def from_dict(cls, value):
        _fields(value, cls.__dataclass_fields__)
        return cls(**value)


@dataclass(frozen=True)
class HostSnapshot:
    epoch: int
    observations: tuple[HostObservation, ...]

    def __post_init__(self):
        _integer(self.epoch)
        if not isinstance(self.observations, tuple) or len(self.observations) > 16:
            raise ValueError("expected bounded immutable host observations")
        if any(not isinstance(x, HostObservation) for x in self.observations):
            raise ValueError("expected typed host observation")
        ids = [x.source_id for x in self.observations]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate host source ID")

    @classmethod
    def from_dict(cls, value):
        _fields(value, cls.__dataclass_fields__)
        return cls(value["epoch"], tuple(HostObservation.from_dict(x)
                                         for x in value["observations"]))


def validate_proposal(value):
    if (not isinstance(value, dict) or not isinstance(value.get("kind"), str)
            or value["kind"] not in {"verify", "answer", "stop"}):
        raise ValueError("unsupported proposal kind")
    extra = {"verify": {"source_ids"}, "answer": {"receipt_id", "value"},
             "stop": {"reason"}}[value["kind"]]
    _fields(value, {"action_id", "kind", "epoch"} | extra)
    _text(value["action_id"])
    _integer(value["epoch"])
    if value["kind"] == "verify":
        _ids(value["source_ids"])
    elif value["kind"] == "answer":
        _sha(value["receipt_id"])
        _integer(value["value"], 255)
    else:
        _text(value["reason"])
    return json.loads(canonical(value))


def _xor_result(observations):
    values = {}
    for observation in observations:
        for line in observation.text.splitlines():
            match = re.fullmatch(r"(INPUT|MASK)=([0-9]{1,3})", line)
            if not match or int(match[2]) > 255:
                return {"value": None, "error": "malformed_register"}
            if match[1] in values:
                return {"value": None, "error": "duplicate_register"}
            values[match[1]] = int(match[2])
    if set(values) != {"INPUT", "MASK"}:
        return {"value": None, "error": "missing_register"}
    return {"value": values["INPUT"] ^ values["MASK"], "error": None}


def run_xor_tool(observations):
    """Actual allowlisted pure tool; receives only eligible host-observed bytes."""
    return _xor_result(observations)


class HostActionSession:
    """Synchronous bounded contract fixture, not a production runtime.

    The caller owns snapshots and synchronizes revocation before submit. There is
    no concurrent cancellation, streaming, neural cache, database or shell runner.
    """

    def __init__(self, binding: HostBinding, snapshot: HostSnapshot):
        if not isinstance(binding, HostBinding) or not isinstance(snapshot, HostSnapshot):
            raise ValueError("typed host binding and snapshot required")
        self.binding, self.snapshot = binding, snapshot
        self.decisions_used = self.tool_calls = 0
        self.stopped = False
        self.receipts, self.actions, self.records = {}, {}, []
        self._record("created", {"binding": asdict(binding), "snapshot": asdict(snapshot)})

    @property
    def head(self):
        return self.records[-1]["sha256"] if self.records else ZERO

    def _record(self, kind, payload):
        if len(self.records) >= 2048:
            raise ValueError("journal record bound exhausted")
        body = {"version": VERSION, "sequence": len(self.records), "previous": self.head,
                "kind": kind, "payload": payload}
        self.records.append(json.loads(canonical({**body, "sha256": digest(body)})))

    def update_snapshot(self, snapshot: HostSnapshot):
        if not isinstance(snapshot, HostSnapshot):
            raise ValueError("typed snapshot required")
        if canonical(asdict(snapshot)) == canonical(asdict(self.snapshot)):
            return
        if snapshot.epoch <= self.snapshot.epoch:
            raise ValueError("a host observation change requires a higher epoch")
        self._record("snapshot", {"snapshot": asdict(snapshot)})
        self.snapshot = snapshot
        self.receipts.clear()

    def _eligible(self, observation):
        binding = self.binding
        branches = {"source"} if binding.identity_mode in {"source_snapshot", "source_current"} \
            else {"source", binding.branch_id}
        return (observation.eligible and observation.source_id in binding.allowed_source_ids
                and observation.subject_id == binding.subject_id
                and observation.branch_id in branches
                and observation.purpose == binding.purpose
                and observation.observed_at <= binding.observation_cutoff)

    def _execute(self, proposal, *, replay=False):
        stamp = {"subject_id": self.binding.subject_id, "branch_id": self.binding.branch_id,
                 "agent_id": self.binding.agent_id, "identity_mode": self.binding.identity_mode,
                 "model_manifest_sha256": self.binding.model_manifest_sha256,
                 "policy_sha256": self.binding.policy_sha256, "epoch": self.snapshot.epoch}

        def outcome(status, **detail):
            return {"status": status, "kind": proposal["kind"], "host_provenance": stamp,
                    "detail": detail}

        if self.decisions_used >= self.binding.max_decisions:
            return outcome("denied", reason="decision_budget")
        self.decisions_used += 1
        if proposal["epoch"] != self.snapshot.epoch:
            return outcome("denied", reason="stale_epoch")
        if self.stopped:
            return outcome("denied", reason="session_stopped")
        if proposal["kind"] == "stop":
            self.stopped = True
            return outcome("ok", reason=proposal["reason"])
        if proposal["kind"] == "answer":
            receipt = self.receipts.get(proposal["receipt_id"])
            if receipt is None or receipt["epoch"] != self.snapshot.epoch:
                return outcome("denied", reason="current_actual_receipt_required")
            if receipt["error"] is not None or receipt["value"] != proposal["value"]:
                return outcome("denied", reason="matching_successful_receipt_required")
            return outcome("ok", value=proposal["value"], receipt_id=proposal["receipt_id"])
        if self.tool_calls >= self.binding.max_tool_calls:
            return outcome("denied", reason="tool_budget")
        available = {x.source_id: x for x in self.snapshot.observations if self._eligible(x)}
        if any(x not in available for x in proposal["source_ids"]):
            return outcome("denied", reason="source_not_eligible")
        selected = tuple(available[x] for x in sorted(proposal["source_ids"]))
        self.tool_calls += 1
        # Restart checks the pure result against recorded input bytes. It does not
        # invoke the tool entry point or count a new actual execution.
        result = _xor_result(selected) if replay else run_xor_tool(selected)
        receipt = {"tool": TOOL, "action_id": proposal["action_id"],
                   "epoch": self.snapshot.epoch, "host_provenance": stamp,
                   "inputs": [asdict(x) for x in selected], **result}
        receipt_id = digest(receipt)
        self.receipts[receipt_id] = receipt
        return outcome("tool_error" if result["error"] else "ok",
                       receipt_id=receipt_id, receipt=receipt)

    def submit(self, proposal):
        proposal = validate_proposal(proposal)
        cached = self.actions.get(proposal["action_id"])
        if cached is not None:
            if canonical(cached["proposal"]) != canonical(proposal):
                raise ValueError("action ID already binds different proposal bytes")
            # Revocation precedes the idempotency return, including old answers.
            if proposal["epoch"] != self.snapshot.epoch:
                return {"status": "denied", "kind": proposal["kind"],
                        "detail": {"reason": "cached_action_invalidated"}}
            return json.loads(canonical(cached["outcome"]))
        if len(self.records) >= 2048:
            raise ValueError("journal record bound exhausted")
        result = self._execute(proposal)
        payload = {"proposal": proposal, "outcome": result}
        self._record("action", payload)
        self.actions[proposal["action_id"]] = payload
        return json.loads(canonical(result))

    def write_journal(self, path):
        """Write one immutable evidence snapshot; caller selects a fresh path."""
        with Path(path).open("xb") as handle:
            handle.write("".join(canonical(x) + "\n" for x in self.records).encode())

    @classmethod
    def read_journal(cls, path, *, current_snapshot, expected_binding, trusted_head=None):
        raw = Path(path).read_bytes()
        if not raw or len(raw) > 8_000_000 or not raw.endswith(b"\n"):
            raise ValueError("empty, incomplete or oversized journal")
        rows = [json.loads(line) for line in raw.splitlines()]
        if raw != "".join(canonical(x) + "\n" for x in rows).encode():
            raise ValueError("journal is not canonical")
        previous = ZERO
        for index, row in enumerate(rows):
            _fields(row, {"version", "sequence", "previous", "kind", "payload", "sha256"})
            body = {k: v for k, v in row.items() if k != "sha256"}
            if (row["version"] != VERSION or type(row["sequence"]) is not int
                    or row["sequence"] != index or row["previous"] != previous
                    or digest(body) != row["sha256"]):
                raise ValueError("journal hash/sequence mismatch")
            previous = row["sha256"]
        if trusted_head is not None and previous != trusted_head:
            raise ValueError("trusted head mismatch, including possible truncation")
        _fields(rows[0]["payload"], {"binding", "snapshot"})
        binding = HostBinding.from_dict(rows[0]["payload"]["binding"])
        if binding != expected_binding:
            raise ValueError("journal binding mismatch")
        session = cls(binding, HostSnapshot.from_dict(rows[0]["payload"]["snapshot"]))
        if rows[0] != session.records[0]:
            raise ValueError("invalid journal creation")
        for row in rows[1:]:
            payload = row["payload"]
            if row["kind"] == "snapshot":
                _fields(payload, {"snapshot"})
                session.update_snapshot(HostSnapshot.from_dict(payload["snapshot"]))
            elif row["kind"] == "action":
                _fields(payload, {"proposal", "outcome"})
                proposal = validate_proposal(payload["proposal"])
                if proposal["action_id"] in session.actions:
                    raise ValueError("journal duplicates an action ID")
                result = session._execute(proposal, replay=True)
                if result != payload["outcome"]:
                    raise ValueError("recorded outcome disagrees with host transition")
                session._record("action", payload)
                session.actions[proposal["action_id"]] = payload
            else:
                raise ValueError("unknown journal event")
            if session.records[-1] != row:
                raise ValueError("journal replay mismatch")
        session.update_snapshot(current_snapshot)
        return session

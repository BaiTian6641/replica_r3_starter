"""Strict local episode intake and causal export; no model or network dependencies.

The JSON schema checks shape. This module also checks ownership, chronology,
supersession, permissions and family lineage. Neither authenticates source
metadata or detects a gold answer disguised as an ordinary observation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import re
from typing import Callable, Iterable

from .contracts import KINDS, SPLITS, canonical_json, validate_example


VERSION = "replica.episode.v1"
USES = {"local_retrieval", "local_training", "local_evaluation", "external_teacher", "sharing"}
TARGET_EXPORT_USES = {"local_training", "local_evaluation", "external_teacher"}
SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def _fields(value, required, name):
    if not isinstance(value, dict) or set(value) != set(required):
        raise ValueError(f"{name} fields must be exactly {sorted(required)}")


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")


def _time(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative logical integer time")


def _hash(value, name):
    if not isinstance(value, str) or not SHA256.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA256")


def _choice(value, allowed, name):
    if not isinstance(value, str) or value not in allowed:
        raise ValueError(f"unknown {name}")


def _strings(value, name, allowed=None):
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    for item in value:
        _text(item, name)
        if allowed is not None and item not in allowed:
            raise ValueError(f"unknown {name}: {item}")
    if len(value) != len(set(value)):
        raise ValueError(f"duplicate {name}")


def _json_value(value):
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float and math.isfinite(value):
        return
    if isinstance(value, list):
        for item in value:
            _json_value(item)
        return
    if isinstance(value, dict) and all(isinstance(k, str) for k in value):
        for item in value.values():
            _json_value(item)
        return
    raise ValueError("episode must contain only finite JSON values")


def _owner(source):
    return tuple(source[key] for key in ("subject_id", "branch_id", "actor_id", "kind"))


def _active_events(payload, *, cutoff):
    """Apply visible observed edits before permission filtering, never oracle edits.

    An inaccessible correction must not resurrect its older eligible predecessor.
    Source permissions represent current intake eligibility, even for old queries.
    """
    active = {}
    for event in payload["events"]:
        if (event["channel"] != "observed" or event["observed_at"] > cutoff
                or event["event_time"] > cutoff):
            continue
        if event["replaces_event_id"] is not None:
            active.pop(event["replaces_event_id"], None)
        if event["operation"] != "tombstone":
            active[event["event_id"]] = event
    return list(active.values())


def _validate_payload(payload):
    _json_value(payload)
    _fields(payload, {"schema_version", "id", "group_id", "split", "subject_id", "branch_id",
                      "parent_episode_id", "privacy", "permissions", "query", "sources",
                      "events", "target", "evaluator"}, "episode")
    if payload["schema_version"] != VERSION:
        raise ValueError("unsupported episode schema_version")
    for key in ("id", "group_id", "subject_id", "branch_id"):
        _text(payload[key], key)
    _choice(payload["split"], SPLITS, "split")
    _choice(payload["privacy"], {"public", "private"}, "episode privacy")
    _strings(payload["permissions"], "permissions", USES)
    parent = payload["parent_episode_id"]
    if parent is not None:
        _text(parent, "parent_episode_id")
        if parent == payload["id"]:
            raise ValueError("episode cannot parent itself")
    _fields(payload["query"], {"at", "prompt"}, "query")
    _time(payload["query"]["at"], "query.at")
    _text(payload["query"]["prompt"], "query.prompt")
    if not isinstance(payload["sources"], list) or not isinstance(payload["events"], list):
        raise ValueError("sources and events must be lists")
    sources = {}
    for source in payload["sources"]:
        _fields(source, {"source_id", "group_id", "split", "sha256", "subject_id", "branch_id",
                         "actor_id", "kind", "privacy", "permissions"}, "source")
        for key in ("source_id", "group_id", "subject_id", "branch_id", "actor_id"):
            _text(source[key], f"source.{key}")
        _hash(source["sha256"], "source.sha256")
        if source["source_id"] in sources:
            raise ValueError("duplicate source_id")
        if source["group_id"] != payload["group_id"] or source["split"] != payload["split"]:
            raise ValueError("source must inherit episode group and split")
        if source["subject_id"] != payload["subject_id"]:
            raise ValueError("source subject binding differs")
        if source["branch_id"] not in {"source", payload["branch_id"]}:
            raise ValueError("source belongs to another agent branch")
        _choice(source["kind"], KINDS, "source kind")
        if source["kind"] == "human_source" and (
                source["branch_id"] != "source" or source["actor_id"] != source["subject_id"]):
            raise ValueError("human_source requires source branch and subject actor")
        if source["kind"] in {"agent_experience", "inference"} and (
                source["branch_id"] == "source" or source["actor_id"] == source["subject_id"]):
            raise ValueError("agent experience/inference cannot own source-person history")
        _choice(source["privacy"], {"public", "private"}, "source privacy")
        if source["privacy"] == "private" and payload["privacy"] != "private":
            raise ValueError("episode containing a private source must remain private")
        _strings(source["permissions"], "source permissions", USES)
        sources[source["source_id"]] = source
    events, active = {}, {"observed": set(), "evaluator_only": set()}
    previous_order = None
    for event in payload["events"]:
        _fields(event, {"event_id", "source_id", "event_time", "observed_at", "operation",
                        "replaces_event_id", "channel", "text"}, "event")
        _text(event["event_id"], "event_id")
        _text(event["source_id"], "event.source_id")
        if event["event_id"] in events:
            raise ValueError("duplicate event_id")
        if event["source_id"] not in sources:
            raise ValueError("event references unknown source")
        _time(event["event_time"], "event_time")
        _time(event["observed_at"], "observed_at")
        if event["event_time"] > event["observed_at"]:
            raise ValueError("this episode version supports occurred events, not future scheduled events")
        order = (event["observed_at"], event["event_id"])
        if previous_order is not None and order <= previous_order:
            raise ValueError("events must be chronological by (observed_at, event_id)")
        previous_order = order
        _choice(event["channel"], active, "event channel")
        _choice(event["operation"], {"observe", "correct", "tombstone"}, "event operation")
        replaced = event["replaces_event_id"]
        if event["operation"] == "observe":
            if replaced is not None:
                raise ValueError("observe cannot replace another event")
        else:
            _text(replaced, "replaces_event_id")
            if replaced not in events or replaced not in active[event["channel"]]:
                raise ValueError("correction/tombstone must replace an earlier active same-channel event")
            prior = events[replaced]
            if _owner(sources[prior["source_id"]]) != _owner(sources[event["source_id"]]):
                raise ValueError("correction/tombstone cannot cross source ownership")
            active[event["channel"]].remove(replaced)
        if event["operation"] == "tombstone":
            if event["text"] != "":
                raise ValueError("tombstone must have empty text")
        else:
            _text(event["text"], "event.text")
            active[event["channel"]].add(event["event_id"])
        events[event["event_id"]] = event
    target = payload["target"]
    _fields(target, {"action", "response", "support_event_ids"}, "target")
    _choice(target["action"], {"answer", "ask", "unknown"}, "target action")
    _text(target["response"], "target.response")
    _strings(target["support_event_ids"], "support_event_ids")
    if target["action"] == "answer" and not target["support_event_ids"]:
        raise ValueError("this evidence-grounded answer requires support_event_ids")
    visible = {e["event_id"] for e in _active_events(payload, cutoff=payload["query"]["at"])}
    if not set(target["support_event_ids"]).issubset(visible):
        raise ValueError("target support must be active observed evidence at query cutoff, never future/oracle")
    _fields(payload["evaluator"], {"oracle"}, "evaluator")
    if not isinstance(payload["evaluator"]["oracle"], dict):
        raise ValueError("evaluator.oracle must be an object")


@dataclass(frozen=True, init=False)
class Episode:
    """Validated immutable JSON snapshot; to_dict returns a detached copy."""

    _canonical: str

    def __init__(self, payload: dict):
        _validate_payload(payload)
        object.__setattr__(self, "_canonical", canonical_json(payload))

    @classmethod
    def from_dict(cls, payload: dict) -> "Episode":
        return cls(payload)

    def to_dict(self) -> dict:
        return json.loads(self._canonical)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self._canonical.encode("utf-8")).hexdigest()


def validate_episode(payload: dict | Episode) -> Episode:
    return payload if isinstance(payload, Episode) else Episode(payload)


def _allowed(payload, purpose):
    _choice(purpose, USES, "export purpose")
    if payload["split"] == "audit" and purpose != "local_evaluation":
        raise ValueError("audit model input requires explicit local_evaluation purpose")
    if purpose not in payload["permissions"]:
        raise ValueError(f"episode does not allow {purpose}")
    if purpose in {"local_training", "external_teacher"} and payload["split"] != "train":
        raise ValueError("training/teacher export requires split=train")


def _selected(payload, purpose):
    _allowed(payload, purpose)
    sources = {s["source_id"]: s for s in payload["sources"]}
    events = [e for e in _active_events(payload, cutoff=payload["query"]["at"])
              if purpose in sources[e["source_id"]]["permissions"]]
    return events, sources


def input_view(episode: dict | Episode, *, purpose="local_retrieval") -> dict:
    """Only prompt and eligible observations; target/oracle are never copied."""
    payload = validate_episode(episode).to_dict()
    events, sources = _selected(payload, purpose)
    texts, source_ids = [], []
    for event in events:
        source = sources[event["source_id"]]
        texts.append(
            f"{event['event_id']} | event_time={event['event_time']} | "
            f"observed_at={event['observed_at']} | kind={source['kind']} | "
            f"subject={source['subject_id']} | branch={source['branch_id']} | {event['text']}")
        if event["source_id"] not in source_ids:
            source_ids.append(event["source_id"])
    return {"prompt": payload["query"]["prompt"], "memory_texts": texts, "source_ids": source_ids}


@dataclass(frozen=True)
class VerificationReceipt:
    """Host-created attestation bound to exact bytes; not a signature or sandbox."""

    episode_sha256: str
    verifier_id: str
    verifier_sha256: str
    passed: bool

    def __post_init__(self):
        _hash(self.episode_sha256, "episode_sha256")
        _text(self.verifier_id, "verifier_id")
        _hash(self.verifier_sha256, "verifier_sha256")
        if type(self.passed) is not bool:
            raise ValueError("verification passed must be boolean")

    def to_dict(self):
        return asdict(self)


def verify_episode(episode: dict | Episode, *, verifier_id: str, verifier_sha256: str,
                   verifier: Callable[[dict], bool]) -> VerificationReceipt:
    """Invoke a trusted independent verifier; it may read evaluator-side oracle.

    The host supplies reviewed verifier code and its hash. This module neither
    executes arbitrary episode directives nor accepts an episode's own pass flag.
    """
    episode = validate_episode(episode)
    _text(verifier_id, "verifier_id")
    _hash(verifier_sha256, "verifier_sha256")
    if not callable(verifier):
        raise ValueError("verifier must be a trusted callable")
    passed = verifier(episode.to_dict())
    return VerificationReceipt(episode.sha256, verifier_id, verifier_sha256, passed)


def export_sft(episode: dict | Episode, *, purpose="local_training",
               verification: VerificationReceipt | None = None, require_verified=True) -> dict:
    """Deterministically emit exactly the existing eight trainer fields."""
    _choice(purpose, TARGET_EXPORT_USES, "target-bearing export purpose")
    episode = validate_episode(episode)
    payload = episode.to_dict()
    events, _ = _selected(payload, purpose)
    if not set(payload["target"]["support_event_ids"]).issubset(e["event_id"] for e in events):
        raise ValueError("target support lacks permission for requested export purpose")
    if verification is not None and not isinstance(verification, VerificationReceipt):
        raise ValueError("verification must be a host VerificationReceipt")
    if verification is not None and verification.episode_sha256 != episode.sha256:
        raise ValueError("verification receipt belongs to different episode bytes")
    verified = verification is not None and verification.passed
    if require_verified and not verified:
        raise ValueError("independent verification is absent or failed; episode remains quarantined")
    row = {"id": payload["id"], "group_id": payload["group_id"], "split": payload["split"],
           **input_view(episode, purpose=purpose), "response": payload["target"]["response"],
           "verified": verified}
    return validate_example(row, require_verified=require_verified)


def audit_episode_families(episodes: Iterable[dict | Episode]) -> dict:
    """Audit declared source/family/parent lineage, not semantic contamination."""
    payloads = [validate_episode(e).to_dict() for e in episodes]
    ids, groups, sources, source_hash_lineage, event_bytes = {}, {}, {}, {}, {}
    for payload in payloads:
        if payload["id"] in ids:
            raise ValueError("duplicate episode id")
        ids[payload["id"]] = payload
        if groups.setdefault(payload["group_id"], payload["split"]) != payload["split"]:
            raise ValueError("episode family crosses splits")
        for source in payload["sources"]:
            body = canonical_json(source)
            if sources.setdefault(source["source_id"], body) != body:
                raise ValueError("source identity/permissions/lineage differs across episodes")
            lineage = (source["group_id"], source["split"])
            if source_hash_lineage.setdefault(source["sha256"], lineage) != lineage:
                raise ValueError("declared source bytes must retain one group and split across episodes")
        for event in payload["events"]:
            key = (event["source_id"], event["event_id"])
            body = canonical_json(event)
            if event_bytes.setdefault(key, body) != body:
                raise ValueError("existing source/event bytes are immutable; use a new correction event")
    complete, visiting = set(), set()

    def visit(episode_id):
        if episode_id in visiting:
            raise ValueError("episode parent cycle")
        if episode_id in complete:
            return
        visiting.add(episode_id)
        payload = ids[episode_id]
        parent_id = payload["parent_episode_id"]
        if parent_id is not None:
            if parent_id not in ids:
                raise ValueError("episode parent missing from lineage audit")
            parent = ids[parent_id]
            if any(parent[k] != payload[k] for k in ("group_id", "split", "subject_id", "branch_id")):
                raise ValueError("descendant must inherit parent family/split/subject/branch")
            visit(parent_id)
        visiting.remove(episode_id)
        complete.add(episode_id)

    for episode_id in ids:
        visit(episode_id)
    return {"episodes": len(ids), "lineage_groups": len(groups), "sources": len(sources),
            "split_counts": {s: sum(p["split"] == s for p in payloads) for s in sorted(SPLITS)},
            "semantic_grounding_checked": False, "source_authenticity_checked": False}

"""Small, explicit record contracts. All fixture identities are fictional."""
from dataclasses import asdict, dataclass
import hashlib
import json

KINDS = {"human_source", "world_evidence", "agent_experience", "inference", "synthetic"}
MODES = {"source_snapshot", "source_current", "parallel_replica", "assistant_for_source"}
SPLITS = {"train", "dev", "calibration", "promotion", "audit"}


def canonical_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def fingerprint(value):
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Record:
    record_id: str
    subject_id: str
    branch_id: str
    actor_id: str
    kind: str
    observed_at: int
    content: str
    source_id: str
    confirmed: bool = False
    train_allowed: bool = False
    synthetic: bool = False
    supersedes: str | None = None
    event_time: str | None = None

    def validate(self):
        for key in ("record_id", "subject_id", "branch_id", "actor_id", "content", "source_id"):
            if not isinstance(getattr(self, key), str) or not getattr(self, key).strip():
                raise ValueError(f"{key} must be a nonempty string")
        if self.kind not in KINDS:
            raise ValueError("unknown record kind")
        for key in ("confirmed", "train_allowed", "synthetic"):
            if type(getattr(self, key)) is not bool:
                raise ValueError(f"{key} must be boolean")
        if isinstance(self.observed_at, bool) or not isinstance(self.observed_at, int) or self.observed_at < 0:
            raise ValueError("observed_at must be a nonnegative logical timestamp")
        if self.kind == "human_source":
            if self.synthetic or self.branch_id != "source" or self.actor_id != self.subject_id:
                raise ValueError("human source requires authentic subject attribution and source branch")
            if self.train_allowed and not self.confirmed:
                raise ValueError("unconfirmed human records cannot be training targets")
        if self.kind == "agent_experience" and self.branch_id == "source":
            raise ValueError("agent experience cannot own source history")
        if self.kind == "synthetic" and not self.synthetic:
            raise ValueError("synthetic record must retain synthetic flag")
        if self.supersedes == self.record_id:
            raise ValueError("record cannot supersede itself")
        return self

    def to_dict(self):
        return asdict(self)


def validate_example(row, require_verified=False):
    """Validate the deliberately small SFT export, not its factual correctness."""
    required = {"id", "group_id", "split", "prompt", "response", "memory_texts", "source_ids", "verified"}
    if not isinstance(row, dict):
        raise ValueError("SFT example must be an object")
    if set(row) != required:
        raise ValueError(f"SFT fields differ: missing={required-set(row)}, extra={set(row)-required}")
    for key in ("id", "group_id", "prompt", "response"):
        if not isinstance(row[key], str) or not row[key].strip():
            raise ValueError(f"invalid {key}")
    if row["split"] not in SPLITS:
        raise ValueError("unknown split")
    for key in ("memory_texts", "source_ids"):
        if not isinstance(row[key], list) or any(not isinstance(s, str) or not s for s in row[key]):
            raise ValueError(f"{key} must be a list of nonempty strings")
    if row["memory_texts"] and not row["source_ids"]:
        raise ValueError("memory evidence needs source identifiers")
    if type(row["verified"]) is not bool:
        raise ValueError("verified must be boolean")
    if require_verified and not row["verified"]:
        raise ValueError("unverified examples are quarantined")
    return row


def audit_splits(rows):
    """Catch exact duplication and lineage leakage; semantic audit is still required."""
    ids, groups, bodies = set(), {}, {}
    count = {s: 0 for s in sorted(SPLITS)}
    for row in rows:
        validate_example(row)
        if row["id"] in ids:
            raise ValueError(f"duplicate example id: {row['id']}")
        ids.add(row["id"])
        split, group = row["split"], row["group_id"]
        if group in groups and groups[group] != split:
            raise ValueError(f"lineage crosses splits: {group}")
        groups[group] = split
        body = fingerprint([row["prompt"].strip(), row["response"].strip(), row["memory_texts"]])
        if body in bodies and bodies[body] != split:
            raise ValueError("identical example crosses splits")
        bodies[body] = split
        count[split] += 1
    return {"examples": len(ids), "lineage_groups": len(groups), "split_counts": count}

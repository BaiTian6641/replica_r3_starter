"""Candidate evaluation and atomic manifest publication; no arbitrary code execution."""
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import random
import tempfile
from .contracts import canonical_json, fingerprint


@dataclass(frozen=True)
class GatePolicy:
    min_pairs: int = 200
    min_target_gain: float = 0.02
    max_retention_loss: float = 0.01
    bootstrap_draws: int = 2000
    confidence: float = 0.95


def paired_interval(previous, candidate, *, draws=2000, confidence=0.95, seed=7):
    if len(previous) != len(candidate) or not previous:
        raise ValueError("nonempty paired observations required")
    if draws < 100 or not 0 < confidence < 1:
        raise ValueError("invalid interval parameters")
    if any(not math.isfinite(x) or not 0 <= x <= 1 for x in previous + candidate):
        raise ValueError("scores must be finite values in [0,1]")
    delta = [b-a for a,b in zip(previous, candidate)]
    if max(delta) == min(delta):
        # An empirical bootstrap of a constant sample degenerates to [c,c].
        # Bound unseen departures with the zero-count binomial upper bound.
        # This assumes independent pair units; duplicate questions are not units.
        unseen = 1 - ((1-confidence)/2) ** (1/len(delta))
        c = delta[0]
        return {"mean": c, "low": c-(c+1)*unseen, "high": c+(1-c)*unseen,
                "n": len(delta), "method": "constant_sample_unseen_departure_guard"}
    rng = random.Random(seed)
    samples = sorted(sum(rng.choices(delta, k=len(delta))) / len(delta) for _ in range(draws))
    tail = (1-confidence)/2
    return {"mean": sum(delta)/len(delta),
            "low": samples[int(tail*draws)],
            "high": samples[min(draws-1, int((1-tail)*draws))], "n": len(delta),
            "method": "paired_percentile_bootstrap"}


def evaluate_candidate(target_before, target_after, retention_before, retention_after,
                       invariants, policy=GatePolicy()):
    if not invariants or any(v is not True for v in invariants.values()):
        return {"decision": "reject", "reason": "hard_invariant_failure", "invariants": invariants}
    if min(len(target_before), len(retention_before)) < policy.min_pairs:
        return {"decision": "inconclusive", "reason": "insufficient_pairs"}
    target = paired_interval(list(target_before), list(target_after),
                             draws=policy.bootstrap_draws, confidence=policy.confidence)
    retention = paired_interval(list(retention_before), list(retention_after),
                                draws=policy.bootstrap_draws, confidence=policy.confidence)
    if retention["high"] < -policy.max_retention_loss or target["high"] < policy.min_target_gain:
        decision = "reject"
    elif target["low"] >= policy.min_target_gain and retention["low"] >= -policy.max_retention_loss:
        decision = "accept"
    else:
        decision = "inconclusive"
    return {"decision": decision, "target_gain": target, "retention_delta": retention,
            "invariants": invariants, "interval_unit": "caller_supplied_independent_pair"}


class ManifestRegistry:
    """Single-writer reference. Production needs an interprocess lock + artifact hashing."""
    def __init__(self, directory):
        self.root = Path(directory)
        self.root.mkdir(parents=True, exist_ok=True)
        self.current = self.root / "active.json"

    def active(self):
        return json.loads(self.current.read_text()) if self.current.exists() else None

    def publish(self, payload, evaluation, *, expected_parent=None):
        if evaluation.get("decision") != "accept":
            raise PermissionError("only accepted candidates may be published")
        active = self.active()
        parent = active["manifest_id"] if active else None
        if parent != expected_parent:
            raise ValueError("active parent changed; reevaluate the candidate")
        if evaluation.get("candidate_digest") != fingerprint(payload):
            raise ValueError("evaluation is not bound to these candidate bytes")
        if evaluation.get("parent_manifest_id") != parent:
            raise ValueError("evaluation is not bound to the active parent")
        if not payload.get("base_revision") or not payload.get("identity_version"):
            raise ValueError("base and identity revisions required")
        body = {"payload": payload, "parent": parent, "evaluation": evaluation}
        manifest = {"manifest_id": fingerprint(body), **body}
        archive = self.root / (manifest["manifest_id"] + ".json")
        encoded = canonical_json(manifest)
        if archive.exists() and archive.read_text() != encoded:
            raise ValueError("manifest collision")
        self._atomic_write(archive, encoded)
        self._atomic_write(self.current, encoded)
        return manifest

    def rollback(self, manifest_id):
        active = self.active()
        if active is None:
            raise ValueError("no active revision")
        # A rollback target must be an ancestor, not an arbitrary file/path.
        cursor = active
        while cursor and cursor["manifest_id"] != manifest_id:
            parent = cursor.get("parent")
            cursor = json.loads((self.root / (parent + ".json")).read_text()) if parent else None
        if cursor is None:
            raise ValueError("rollback target is not an ancestor")
        self._atomic_write(self.current, canonical_json(cursor))
        return cursor

    def _atomic_write(self, target, text):
        fd, tmp = tempfile.mkstemp(prefix="pending-", dir=self.root)
        try:
            with os.fdopen(fd, "w") as handle:
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, target)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

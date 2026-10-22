"""SQLite evidence ledger. Logical scope checks are not OS/process isolation."""
import json
import sqlite3
from .contracts import Record, MODES, canonical_json, fingerprint


class EvidenceStore:
    def __init__(self, path):
        self.db = sqlite3.connect(path)
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS records (
            id TEXT PRIMARY KEY, subject TEXT NOT NULL, branch TEXT NOT NULL,
            observed INTEGER NOT NULL, kind TEXT NOT NULL, supersedes TEXT,
            digest TEXT NOT NULL, payload TEXT NOT NULL, deleted INTEGER NOT NULL DEFAULT 0,
            FOREIGN KEY(supersedes) REFERENCES records(id));
        CREATE TABLE IF NOT EXISTS dependencies (
            artifact TEXT NOT NULL, source TEXT NOT NULL, invalidated INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY(artifact,source), FOREIGN KEY(source) REFERENCES records(id));
        """)

    def close(self):
        self.db.close()

    def put(self, record: Record, *, authorized_subject: str, authorized_branch: str,
            source_write_authorized=False):
        record.validate()
        if record.subject_id != authorized_subject or record.branch_id != authorized_branch:
            raise PermissionError("write scope mismatch")
        if record.branch_id == "source" and not source_write_authorized:
            raise PermissionError("source archive requires source authority")
        value, digest = record.to_dict(), fingerprint(record.to_dict())
        old = self.db.execute("SELECT digest,deleted FROM records WHERE id=?", (record.record_id,)).fetchone()
        if old:
            if old[1]:
                raise ValueError("tombstoned record identifier cannot be reused")
            if old[0] != digest:
                raise ValueError("immutable record identifier collision")
            return False
        if record.supersedes:
            parent = self.db.execute(
                "SELECT subject,branch,observed,deleted FROM records WHERE id=?", (record.supersedes,)
            ).fetchone()
            if not parent or parent[3] or parent[:2] != (record.subject_id, record.branch_id):
                raise PermissionError("supersession must stay in the live subject and branch")
            if parent[2] >= record.observed_at:
                raise ValueError("correction must be observed after its parent")
            if self.db.execute("SELECT 1 FROM records WHERE supersedes=? AND deleted=0",
                               (record.supersedes,)).fetchone():
                raise ValueError("supersession already exists; correct its current descendant")
        with self.db:
            self.db.execute("INSERT INTO records VALUES(?,?,?,?,?,?,?,?,0)",
                (record.record_id, record.subject_id, record.branch_id, record.observed_at,
                 record.kind, record.supersedes, digest, canonical_json(value)))
            if record.supersedes:
                self.db.execute("UPDATE dependencies SET invalidated=1 WHERE source=?",
                                (record.supersedes,))
        return True

    def visible(self, *, subject_id, branch_id, as_of, mode="assistant_for_source",
                snapshot_at=None):
        if mode not in MODES:
            raise ValueError("unknown identity mode")
        for name, value in (("as_of", as_of), ("snapshot_at", snapshot_at)):
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError(f"{name} must be a nonnegative logical timestamp")
        if as_of is None:
            raise ValueError("as_of is required")
        if mode == "source_snapshot" and snapshot_at is None:
            raise ValueError("source_snapshot requires snapshot_at")
        # Scope is supplied by a trusted host, never inferred from memory text.
        branches = ["source"] if mode in {"source_snapshot", "source_current"} else ["source", branch_id]
        rows = self.db.execute(
            "SELECT payload,branch,observed,supersedes,deleted FROM records "
            "WHERE subject=? AND observed<=? ORDER BY observed,id", (subject_id, as_of)).fetchall()
        rows = [r for r in rows if r[1] in branches and
                (mode != "source_snapshot" or r[2] <= snapshot_at)]
        # Deleted correction metadata still supersedes the old fact. Redaction
        # must never revive obsolete evidence in the current view.
        superseded = {r[3] for r in rows if r[3]}
        records = [Record(**json.loads(r[0])) for r in rows if not r[4]]
        return [r for r in records if r.record_id not in superseded]

    def register_artifact(self, artifact_id, record_ids):
        if not record_ids:
            raise ValueError("derived artifact must have parents")
        with self.db:
            for rid in record_ids:
                row = self.db.execute("SELECT deleted FROM records WHERE id=?", (rid,)).fetchone()
                if not row or row[0]:
                    raise ValueError("missing or deleted source")
                self.db.execute("INSERT INTO dependencies VALUES(?,?,0)", (artifact_id, rid))

    def invalidated_artifacts(self):
        return [row[0] for row in self.db.execute(
            "SELECT DISTINCT artifact FROM dependencies WHERE invalidated=1 ORDER BY artifact")]

    def forget(self, record_id, *, authorized_subject, authorized_branch,
               source_delete_authorized=False):
        """Tombstone correction descendants; flag directly derived artifacts for rebuild.

        Does not erase learned weights, old exports, filesystem backups, or OS pages.
        A production deletion service must handle those separately.
        """
        parent = self.db.execute("SELECT subject,branch FROM records WHERE id=?", (record_id,)).fetchone()
        if not parent or parent != (authorized_subject, authorized_branch):
            raise PermissionError("unknown or unauthorized record")
        if authorized_branch == "source" and not source_delete_authorized:
            raise PermissionError("source deletion requires source authority")
        todo, removed = [record_id], set()
        while todo:
            rid = todo.pop()
            if rid in removed:
                continue
            removed.add(rid)
            todo.extend(r[0] for r in self.db.execute("SELECT id FROM records WHERE supersedes=?", (rid,)))
        with self.db:
            for rid in removed:
                self.db.execute("UPDATE records SET deleted=1,payload=? WHERE id=?",
                                (canonical_json({"redacted": True}), rid))
                self.db.execute("UPDATE dependencies SET invalidated=1 WHERE source=?", (rid,))
        affected = self.invalidated_artifacts()
        return {"removed_record_ids": sorted(removed), "invalidated_artifacts": affected,
                "neural_unlearning_performed": False}

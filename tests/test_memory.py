from pathlib import Path
import tempfile
import unittest
from replica_lab.contracts import Record
from replica_lab.memory import EvidenceStore


class MemoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "ledger.sqlite"
        self.store = EvidenceStore(self.path)

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def source(self, rid="e1", time=1, content="prefer blue", supersedes=None):
        return Record(rid, "fixture-person", "source", "fixture-person", "human_source",
                      time, content, rid, confirmed=True, supersedes=supersedes)

    def put(self, r):
        return self.store.put(r, authorized_subject="fixture-person",
                              authorized_branch="source", source_write_authorized=True)

    def read(self, at=10, **kwargs):
        return self.store.visible(subject_id="fixture-person", branch_id="replica-a", as_of=at, **kwargs)

    def test_future_correction_does_not_rewrite_past_view(self):
        self.put(self.source())
        self.put(self.source("e2", 5, "prefer green", "e1"))
        self.assertEqual([r.content for r in self.read(at=3)], ["prefer blue"])
        self.assertEqual([r.content for r in self.read(at=8)], ["prefer green"])

    def test_source_snapshot_ignores_later_revision(self):
        self.put(self.source())
        self.put(self.source("e2", 5, "prefer green", "e1"))
        self.assertEqual(self.read(mode="source_snapshot", snapshot_at=3)[0].record_id, "e1")

    def test_cross_subject_read_returns_nothing(self):
        self.put(self.source())
        self.assertEqual(self.store.visible(subject_id="other", branch_id="replica-a", as_of=10), [])

    def test_unapproved_source_write_is_rejected(self):
        with self.assertRaises(PermissionError):
            self.store.put(self.source(), authorized_subject="fixture-person", authorized_branch="source")

    def test_idempotence_and_collision(self):
        self.assertTrue(self.put(self.source()))
        self.assertFalse(self.put(self.source()))
        with self.assertRaises(ValueError):
            self.put(self.source(content="changed in place"))

    def test_restart_and_forgetting_invalidate_derivations(self):
        self.put(self.source())
        self.put(self.source("e2", 5, "prefer green", "e1"))
        self.store.register_artifact("adapter-a", ["e1", "e2"])
        self.store.close()
        self.store = EvidenceStore(self.path)
        self.assertEqual(self.read()[0].record_id, "e2")
        report = self.store.forget("e1", authorized_subject="fixture-person",
                                  authorized_branch="source", source_delete_authorized=True)
        self.assertEqual(report["invalidated_artifacts"], ["adapter-a"])
        self.assertFalse(report["neural_unlearning_performed"])
        self.assertEqual(self.read(), [])
        self.assertEqual(self.read(at=2), [])
        with self.assertRaises(ValueError):
            self.put(self.source())

    def test_branch_experience_not_visible_in_source_mode(self):
        r = Record("a1", "fixture-person", "replica-a", "agent", "agent_experience", 2,
                   "ran a simulator", "tool-run")
        self.store.put(r, authorized_subject="fixture-person", authorized_branch="replica-a")
        self.assertEqual(len(self.read()), 1)
        self.assertEqual(self.read(mode="source_current"), [])

    def test_invalid_timestamp_cannot_bypass_causal_cutoff(self):
        self.put(self.source())
        with self.assertRaises(ValueError):
            self.read(at="not-a-time")

    def test_deleting_correction_does_not_revive_superseded_fact(self):
        self.put(self.source())
        self.store.register_artifact("vector-a", ["e1"])
        self.put(self.source("e2", 5, "prefer green", "e1"))
        self.assertEqual(self.store.invalidated_artifacts(), ["vector-a"])
        self.store.forget("e2", authorized_subject="fixture-person",
                          authorized_branch="source", source_delete_authorized=True)
        self.assertEqual(self.read(), [])
        self.assertEqual(self.read(at=2)[0].record_id, "e1")

    def test_replica_cannot_delete_source(self):
        self.put(self.source())
        with self.assertRaises(PermissionError):
            self.store.forget("e1", authorized_subject="fixture-person", authorized_branch="source")

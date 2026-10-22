import tempfile
import unittest
from replica_lab.learning import GatePolicy, ManifestRegistry, evaluate_candidate
from replica_lab.contracts import fingerprint


def bound(payload, parent=None):
    return {"decision": "accept", "kind": "fixture", "candidate_digest": fingerprint(payload),
            "parent_manifest_id": parent}


class LearningTests(unittest.TestCase):
    def test_invariant_failure_overrides_large_gain(self):
        result = evaluate_candidate([0]*300, [1]*300, [1]*300, [1]*300,
                                    {"causal": False, "identity": True})
        self.assertEqual(result["decision"], "reject")

    def test_small_sample_is_inconclusive(self):
        result = evaluate_candidate([0]*4, [1]*4, [1]*4, [1]*4, {"causal": True})
        self.assertEqual(result["decision"], "inconclusive")

    def test_retention_damage_rejects_candidate(self):
        p = GatePolicy(bootstrap_draws=200)
        r = evaluate_candidate([0]*250, [1]*250, [1]*250, [0]*250, {"ok": True}, p)
        self.assertEqual(r["decision"], "reject")

    def test_manifest_publication_restart_and_rollback(self):
        with tempfile.TemporaryDirectory() as root:
            r = ManifestRegistry(root)
            p = {"base_revision": "fixture-base-immutable", "identity_version": "fixture-source-v1"}
            m0 = r.publish(p, bound(p))
            p1 = {**p, "skill": "new"}
            m1 = r.publish(p1, bound(p1, m0["manifest_id"]),
                           expected_parent=m0["manifest_id"])
            resumed = ManifestRegistry(root)
            self.assertEqual(resumed.active()["manifest_id"], m1["manifest_id"])
            self.assertEqual(resumed.rollback(m0["manifest_id"])["manifest_id"], m0["manifest_id"])

    def test_stale_parent_and_rejected_candidate_cannot_publish(self):
        with tempfile.TemporaryDirectory() as root:
            r = ManifestRegistry(root)
            p = {"base_revision": "fixture", "identity_version": "fixture"}
            r.publish(p, bound(p))
            with self.assertRaises(ValueError):
                r.publish(p, {"decision": "accept"}, expected_parent=None)
            with self.assertRaises(PermissionError):
                r.publish(p, {"decision": "reject"})

    def test_zero_observed_loss_does_not_certify_one_percent_budget_at_n200(self):
        r = evaluate_candidate([0]*200, [1]*200, [1]*200, [1]*200, {"ok": True})
        self.assertEqual(r["decision"], "inconclusive")
        self.assertLess(r["retention_delta"]["low"], -0.01)

    def test_evaluation_of_different_candidate_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            r = ManifestRegistry(root)
            p = {"base_revision": "fixture", "identity_version": "fixture"}
            with self.assertRaises(ValueError):
                r.publish(p, bound({**p, "wrong": True}))

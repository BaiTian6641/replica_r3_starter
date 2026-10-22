import unittest
from replica_lab.contracts import Record, audit_splits, validate_example


def example(i="a", group="g", split="train", prompt="question"):
    return {"id": i, "group_id": group, "split": split, "prompt": prompt, "response": "answer",
            "memory_texts": ["past observation"], "source_ids": ["event"], "verified": True}


class ContractTests(unittest.TestCase):
    def test_lineage_leak_is_rejected_even_if_words_change(self):
        with self.assertRaises(ValueError):
            audit_splits([example(), example("b", split="dev", prompt="paraphrase")])

    def test_exact_duplicate_with_new_group_is_rejected(self):
        with self.assertRaises(ValueError):
            audit_splits([example(), example("b", "other", "audit")])

    def test_hypothetical_cannot_be_human_history(self):
        with self.assertRaises(ValueError):
            Record("x", "person", "source", "person", "human_source", 1, "fiction", "s",
                   synthetic=True).validate()

    def test_agent_experience_cannot_own_source(self):
        with self.assertRaises(ValueError):
            Record("x", "person", "source", "agent", "agent_experience", 1, "I ran a task", "s").validate()

    def test_verification_is_not_truthy_string(self):
        row = example()
        row["verified"] = "true"
        with self.assertRaises(ValueError):
            validate_example(row, require_verified=True)

    def test_unverified_teacher_output_is_quarantined(self):
        row = example()
        row["verified"] = False
        with self.assertRaises(ValueError):
            validate_example(row, require_verified=True)

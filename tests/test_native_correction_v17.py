"""Independent synthetic correction regression; no evaluation files or labels."""
from test_intrinsic_acceptance_v15 import compound_row, COMPOUND_FAMILIES
from replica_lab.intrinsic_memory_acceptance_v15 import chronological_correction, parse_compound_chronology


def test_compound_correction_preserves_original_facts_and_revision_links():
    original = compound_row(COMPOUND_FAMILIES[0])
    first = chronological_correction(original, "blue")
    second = chronological_correction(first, "gold")
    assert first["facts"] == second["facts"] == original["facts"]
    assert [item["revision"] for item in second["revision_history"]] == [1, 2, 3]
    assert second["correction"]["supersedes_revision"] == 2
    assert parse_compound_chronology(second["text"], second["facts"],
                                    original["rule_family"]) == second["revision_history"]

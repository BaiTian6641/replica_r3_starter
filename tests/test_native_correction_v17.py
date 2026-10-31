"""Native development-row chronological correction regression."""
import json
from pathlib import Path

from replica_lab.intrinsic_memory_acceptance_v15 import (
    chronological_correction, parse_compound_chronology)


def test_native_compound_correction_preserves_original_facts_and_revision_links():
    rows = json.loads((Path(__file__).resolve().parents[1]
                       / "data/intrinsic_memory_coverage_v12_20261029/development.json").read_text())["rows"]
    original = rows[0]
    first = chronological_correction(original, rows[1]["facts"]["answer"])
    assert first["facts"] == original["facts"]
    assert first["revision_history"] == [
        {"revision": 1, "answer": "gold", "anchor_colour": "green"},
        {"revision": 2, "answer": "blue", "anchor_colour": "green"},
    ]
    assert "The recorded color of Ulmara is gold" in first["text"]
    assert "Revision 2: " in first["text"]
    second = chronological_correction(first, "gold")
    assert second["facts"] == original["facts"]
    assert [item["revision"] for item in second["revision_history"]] == [1, 2, 3]
    assert second["correction"]["supersedes_revision"] == 2
    assert parse_compound_chronology(
        second["text"], second["facts"], original["rule_family"]) == second["revision_history"]

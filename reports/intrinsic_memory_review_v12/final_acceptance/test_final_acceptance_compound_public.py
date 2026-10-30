"""Public compound oracle consistency check; no development data read."""
from replica_lab.intrinsic_memory_coverage_v12 import (
    COMPOUND_FAMILIES, _compound_answer, _render_compound)


def test_unrecorded_target_does_not_render_answer_as_recorded():
    case = {"anchor": "A", "target": "B", "answer": "red",
            "anchor_colour": "blue", "unrecorded_target": True}
    assert _compound_answer(case, COMPOUND_FAMILIES[1], "A") == "unknown"
    assert "The recorded color of B is red" not in _render_compound(
        case, COMPOUND_FAMILIES[1])

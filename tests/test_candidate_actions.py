"""Reusable candidate menu and selection contracts."""
import json
import pytest
from replica_lab.runtime.candidate_actions import (CandidateReranker,
    CandidateSelection, build_action_menu, canonical_action)
from tests.e3_runtime_helpers import make_case, make_runtime, read_action

def test_menu_is_bounded_and_includes_valid_actions(tmp_path):
    tools=make_case(tmp_path); runtime=make_runtime(tools,tmp_path)
    menu=build_action_menu(tools,runtime.state)
    raw=[canonical_action(x) for x in menu]
    assert len(menu)==len(set(raw))<=8
    assert any(x.get('kind')=='read' for x in menu)
    assert any(x.get('kind')=='invoke_tool' and x.get('tool')=='run_test' for x in menu)
    assert any(x.get('kind')=='stop' for x in menu)

def test_selection_requires_matching_unique_scores():
    with pytest.raises(ValueError):
        CandidateSelection(({'kind':'stop'},), (), 0)
    with pytest.raises(ValueError):
        CandidateSelection(({'kind':'stop'},), (1.0,), 1)
    selection=CandidateSelection(({'kind':'stop'},{'kind':'read'}), (2.0,1.0), 1)
    assert selection.chosen=={'kind':'read'}

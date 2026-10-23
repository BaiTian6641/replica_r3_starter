"""Reusable candidate menu and selection contracts."""
import json
import pytest
from replica_lab.runtime.candidate_actions import (CandidateReranker,
    CandidateSelection, build_action_menu, canonical_action)
from tests.e3_runtime_helpers import (ScriptedProposer, make_case, make_runtime,
                                      make_state, read_action)
from replica_lab.runtime.tools import FixtureWorkspace
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

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

def test_observed_invalid_value_is_removed_but_two_repairs_remain(tmp_path):
    tools=FixtureWorkspace(ROOT/'fixtures/e3_set/cases/set_train_retry_01')
    runtime=make_runtime(tools,tmp_path)
    test=json.dumps({'kind':'invoke_tool','tool':'run_test','arguments':{}})
    runtime.run(ScriptedProposer([test]),max_decisions=1)
    menu=build_action_menu(tools,runtime.state)
    repairs=[x for x in menu if x.get('kind')=='invoke_tool' and x.get('tool')=='set_constant']
    failed=[x for x in repairs if x['arguments']['name']=='RETRY_ATTEMPTS']
    assert failed
    assert all(x['arguments']['value']>=5 for x in failed)
    assert len({x['arguments']['value'] for x in failed})>=2


def test_numeric_string_constants_enter_the_menu():
    tools=FixtureWorkspace(ROOT/'fixtures/e3_set/cases/set_train_retry_01')
    menu=build_action_menu(tools,make_state(tools))
    repairs={(x['arguments']['name'],x['arguments']['value']) for x in menu
             if x.get('kind')=='invoke_tool' and x.get('tool')=='set_constant'}
    assert ('BACKOFF_SECONDS',0.4) in repairs
    assert ('REQUEST_TIMEOUT_MS',1200.0) in repairs
    assert ('RETRY_ATTEMPTS',2.0) in repairs

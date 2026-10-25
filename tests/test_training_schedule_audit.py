"""The seeded case sequence and each update are evidence, not aggregate counts."""
from copy import deepcopy

import pytest

from replica_lab.memory_reliability import build_episodes, training_schedule
from replica_lab.reliability_scoring import validate_training_routes
from test_memory_reliability_v5 import _route_report


def test_duplicate_update_three_cannot_replace_update_48():
    episodes = build_episodes(family_plan=[('train', range(8)), ('dev', range(100, 104)),
                                          ('heldout', range(200, 204))])
    episodes, _, report, _, protocol = _route_report(episodes)
    report['training']['updates'][-1] = deepcopy(report['training']['updates'][2])
    with pytest.raises(ValueError, match='unique and ordered'):
        validate_training_routes(report, episodes, protocol)


@pytest.mark.parametrize('mutation', ['missing_update', 'reordered_updates', 'boolean_update',
                                      'boolean_row_update', 'wrong_group', 'reordered_rows',
                                      'duplicate_case', 'substituted_case', 'case_ids', 'seed',
                                      'boolean_seed', 'boolean_budget', 'counter', 'boolean_counter',
                                      'route', 'state_use', 'gradient_scope', 'missing_probe',
                                      'gradient_nonzero_inventory', 'gradient_boolean_eligible'])
def test_schedule_and_accounting_mutations_are_rejected(mutation):
    episodes, rows, report, _, protocol = _route_report()
    training = report['training']
    updates = training['updates']
    if mutation == 'missing_update':
        updates.pop()
    elif mutation == 'reordered_updates':
        updates[2], updates[3] = updates[3], updates[2]
    elif mutation == 'boolean_update':
        updates[0]['update'] = True
    elif mutation == 'boolean_row_update':
        rows[0]['update'] = True
    elif mutation == 'wrong_group':
        rows[0]['update'] = 2
    elif mutation == 'reordered_rows':
        rows[0], rows[1] = rows[1], rows[0]
        training['case_ids'] = [row['case_id'] for row in rows]
    elif mutation == 'duplicate_case':
        rows[1] = deepcopy(rows[0])
        training['case_ids'] = [row['case_id'] for row in rows]
    elif mutation == 'substituted_case':
        rows[0]['case_id'] = 'not-a-scheduled-case'
        training['case_ids'][0] = rows[0]['case_id']
    elif mutation == 'case_ids':
        training['case_ids'].reverse()
    elif mutation == 'seed':
        protocol['training']['seed'] += 1
    elif mutation == 'boolean_seed':
        protocol['training']['seed'] = True
    elif mutation == 'boolean_budget':
        protocol['training']['examples_per_update'] = True
    elif mutation == 'counter':
        updates[0]['backward_calls'] -= 1
    elif mutation == 'boolean_counter':
        updates[0]['gate_off_examples'] = False
    elif mutation == 'route':
        rows[0]['route'] = 'disabled' if rows[0]['route'] == 'fast' else 'fast'
    elif mutation == 'state_use':
        rows[0]['state_use'] = 'none' if rows[0]['state_use'] == 'fast' else 'fast'
    elif mutation == 'gradient_scope':
        updates[0]['gradient_participation']['scope'] = 'per_example'
    elif mutation == 'missing_probe':
        updates[0]['gradients'] = None
        updates[0]['gradient_participation'] = None
    elif mutation == 'gradient_nonzero_inventory':
        updates[0]['gradients']['new']['nonzero'] = ['unobserved.tensor']
    else:
        updates[0]['gradients']['new']['eligible'] = False
    with pytest.raises(ValueError):
        validate_training_routes(report, episodes, protocol)


def test_identified_historical_v5_schedule_remains_auditable():
    episodes, _, report, _, protocol = _route_report()
    protocol['format'] = 'replica.memory_reliability.protocol.v5'
    for field in ('schedule_version', 'repeat_policy'):
        protocol['training'].pop(field)
        report['training'].pop(field)
    report['training'].pop('format')
    result = validate_training_routes(report, episodes, protocol)
    assert result['historical_protocol'] == protocol['format']
    assert result['updates_checked'] == protocol['training']['updates']
    protocol['format'] = 'unknown.historical.format'
    with pytest.raises(ValueError, match='undeclared'):
        validate_training_routes(report, episodes, protocol)


@pytest.mark.parametrize('repeat', ['twice', 2, True])
def test_unsupported_repeat_policy_is_not_silently_applied(repeat):
    episodes, _, _, _, protocol = _route_report()
    protocol['training']['repeat_policy'] = repeat
    with pytest.raises(ValueError, match='repeat policy'):
        training_schedule(episodes, protocol)

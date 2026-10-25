"""V5 scorer corrections for explicit route and backward accounting."""
import pytest

from replica_lab.memory_reliability import build_episodes, cases
from replica_lab.reliability_scoring import validate_training_routes


def _route_report():
    episodes = build_episodes()
    scheduled = [case for episode in episodes if episode['split'] == 'train'
                 for stage in ('initial', 'corrected') for case in cases(episode, stage)]
    rows = []
    for case in scheduled:
        route = 'fast' if case['kind'] in {'fact', 'changed', 'unaffected'} else 'disabled'
        rows.append({'case_id': case['case_id'], 'update': 1, 'route': route,
                     'state_use': 'fast' if route == 'fast' else 'none',
                     'memory_active': route == 'fast', 'backward_called': True})
    counts = {route: sum(row['route'] == route for row in rows) for route in ('fast', 'disabled')}
    update = {'update': 1, 'scheduled_examples': len(rows),
              'memory_active_examples': counts['fast'], 'backward_calls': len(rows),
              'gate_off_examples': counts['disabled'], 'gate_on_examples': counts['fast'],
              'route_counts': counts, 'gradients': None,
              'gradient_participation': {'scope': 'accumulated_update', 'audit': None}}
    return episodes, rows, {'training': {'cases': rows, 'updates': [update]}}, counts


def test_gate_off_backward_is_recorded_without_calling_it_fast_participation():
    episodes, rows, report, counts = _route_report()
    protocol = {'exact_reference': {'enabled': True}, 'evidence_gate': {'enabled': True},
                'training': {'updates': 1, 'expected_route_counts': counts}}
    result = validate_training_routes(report, episodes, protocol)
    assert result['memory_active_examples'] == counts['fast']
    assert result['backward_calls'] == len(rows)
    assert result['disabled_backward_calls'] == counts['disabled']


def test_fast_route_missing_backward_is_rejected():
    episodes, rows, report, counts = _route_report()
    next(row for row in rows if row['route'] == 'fast')['backward_called'] = False
    report['training']['updates'][0]['backward_calls'] -= 1
    protocol = {'exact_reference': {'enabled': True}, 'evidence_gate': {'enabled': True},
                'training': {'updates': 1, 'expected_route_counts': counts}}
    with pytest.raises(ValueError, match='Fast-route'):
        validate_training_routes(report, episodes, protocol)

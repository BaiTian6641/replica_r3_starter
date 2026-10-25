"""V5 scorer corrections for explicit route and backward accounting."""
import pytest

from replica_lab.memory_reliability import (TRAINING_REPORT_FORMAT, TRAINING_SCHEDULE_VERSION,
                                           build_episodes, training_schedule)
from replica_lab.reliability_scoring import validate_training_routes


def _route_report(episodes=None):
    episodes = build_episodes() if episodes is None else episodes
    update_count = sum(episode['split'] == 'train' for episode in episodes) * 3
    protocol = {'exact_reference': {'enabled': True}, 'evidence_gate': {'enabled': True},
                'training': {'updates': update_count, 'examples_per_update': 4, 'seed': 20261025,
                             'schedule_version': TRAINING_SCHEDULE_VERSION, 'repeat_policy': 'once',
                             'gradient_probe_updates': [1, 2]}}
    scheduled = training_schedule(episodes, protocol)
    rows, updates = [], []
    for position, (_, case) in enumerate(scheduled):
        route = 'fast' if case['kind'] in {'fact', 'changed', 'unaffected'} else 'disabled'
        rows.append({'case_id': case['case_id'], 'update': position // 4 + 1, 'route': route,
                     'state_use': 'fast' if route == 'fast' else 'none',
                     'memory_active': route == 'fast', 'backward_called': True})
    for index in range(update_count):
        subset = rows[index * 4:(index + 1) * 4]
        counts = {route: sum(row['route'] == route for row in subset) for route in ('fast', 'disabled')}
        gradient = ({'original': {'eligible': 0, 'present': [], 'nonzero': [], 'nonfinite': []},
                     'new': {'eligible': 0, 'present': [], 'nonzero': [], 'nonfinite': []}}
                    if index < 2 else None)
        updates.append({'update': index + 1, 'scheduled_examples': len(subset),
                        'memory_active_examples': counts['fast'], 'backward_calls': len(subset),
                        'gate_off_examples': counts['disabled'], 'gate_on_examples': counts['fast'],
                        'route_counts': {key: value for key, value in counts.items() if value},
                        'gradients': gradient, 'gradient_participation': (
                            {'scope': 'accumulated_update', 'audit': gradient} if gradient is not None else None)})
    counts = {route: sum(row['route'] == route for row in rows) for route in ('fast', 'disabled')}
    protocol['training']['expected_route_counts'] = counts
    report = {'training': {'format': TRAINING_REPORT_FORMAT, 'schedule_version': TRAINING_SCHEDULE_VERSION,
                           'repeat_policy': 'once', 'case_ids': [row['case_id'] for row in rows],
                           'cases': rows, 'updates': updates}}
    return episodes, rows, report, counts, protocol


def test_gate_off_backward_is_recorded_without_calling_it_fast_participation():
    episodes, rows, report, counts, protocol = _route_report()
    result = validate_training_routes(report, episodes, protocol)
    assert result['memory_active_examples'] == counts['fast']
    assert result['backward_calls'] == len(rows)
    assert result['disabled_backward_calls'] == counts['disabled']


def test_fast_route_missing_backward_is_rejected():
    episodes, rows, report, counts, protocol = _route_report()
    next(row for row in rows if row['route'] == 'fast')['backward_called'] = False
    report['training']['updates'][0]['backward_calls'] -= 1
    with pytest.raises(ValueError, match='Fast-route'):
        validate_training_routes(report, episodes, protocol)


def test_dataset_audit_comparison_uses_json_key_semantics():
    import json
    from scripts.score_memory_reliability import dataset_audits_match
    from replica_lab.memory_reliability import audit_dataset
    episodes = build_episodes(general_by_split='general_v2')
    independent = audit_dataset(episodes, general_oracle='general_v2')
    cached = json.loads(json.dumps(independent))
    assert dataset_audits_match(cached, independent)
    cached['query_forms']['0'] += 1
    assert not dataset_audits_match(cached, independent)

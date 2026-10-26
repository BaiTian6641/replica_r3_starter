"""CPU tiny-adapter contracts for the frozen update-dynamics discriminator.

The tests exercise the real SGDN-2 writer, reduction, read and scoring paths on
random local tensors.  They never load the native model, never run a native job
and never update a parameter.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from architecture_helpers import tiny_adapter
from conftest import require_local_artifacts
from replica_lab.memory_diagnosis import query_key
from replica_lab.memory_matched import (
    _record_views,
    derive_matched_labels,
    episode_states,
    generate_matched_episodes,
)
from replica_lab.memory_update_dynamics import (
    CODEBOOK_FORMAT,
    CONFIG_FORMAT,
    DEFAULT_LIMITS,
    INVENTORY_FORMAT,
    READOUT_FORMAT,
    READ_ARMS,
    _full_fact_parity,
    admit_update_dynamics_prepared,
    arm_applies,
    build_protocol,
    planned_inventory,
    planned_workload,
    prepare_update_dynamics,
    resolve_variants,
    run_update_dynamics,
    score_update_dynamics,
    select_first_failing_mechanism,
    variant_applies,
    variant_by_name,
    variant_history,
    variant_names,
    variant_plan,
)
from test_memory_diagnosis import SPEC, Tokenizer

ROOT = Path(__file__).resolve().parents[1]


def _sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False)
                          .encode()).hexdigest()


# One deterministic family per split exercises every query form; the tests use
# one episode per complexity so a tiny adapter stays fast.
_ALL_EPISODES = generate_matched_episodes(
    seed=7, families_per_split={'train': 1, 'dev': 1, 'heldout': 1},
    complexities=('single', 'pair', 'triple', 'correction', 'interference'),
    variants=2)
_SEEN, EPISODES = set(), []
for _episode in _ALL_EPISODES:
    if _episode['complexity'] not in _SEEN:
        _SEEN.add(_episode['complexity'])
        EPISODES.append(_episode)
LABELS = derive_matched_labels(EPISODES)
LAYERS = (0, 2)
FIXTURE_VARIANTS = variant_names()


def _protocol(**overrides):
    value = build_protocol(
        EPISODES, LABELS, layers=LAYERS,
        variants=resolve_variants(FIXTURE_VARIANTS),
        limits={'max_rows': 20000})
    value.update(overrides)
    return value


@pytest.fixture(scope='module')
def tiny_run(tmp_path_factory):
    model = tiny_adapter(fast_memory=SPEC).eval()
    output = tmp_path_factory.mktemp('update-dynamics-run')
    report = run_update_dynamics(
        model, Tokenizer(), _protocol(), output_dir=output, device='cpu',
        episodes=EPISODES, labels_by_episode=LABELS)
    return model, output, report


def test_variant_plan_is_deterministic_and_ordered():
    first = variant_plan()
    assert first == variant_plan()
    names = [item['name'] for item in first]
    assert names == variant_names()
    assert len(names) == len(set(names)) == 20
    assert set(item['reducer'] for item in first) == {'shared_mean', 'per_depth'}
    assert {'target_only', 'prefix_through_target', 'target_and_later'} <= {
        item['order'] for item in first}
    assert {'forced_full_erase', 'forced_full_decay',
            'forced_full_gate', 'forced_full_forget'} <= {
        item['gate_control'] for item in first}
    assert any(item['refresh_target'] for item in first)
    assert variant_by_name('shared_mean/full')['reducer'] == 'shared_mean'
    with pytest.raises(ValueError, match='Unknown requested'):
        resolve_variants(['not/a/variant'])
    with pytest.raises(ValueError, match='Duplicate requested'):
        resolve_variants(['shared_mean/full', 'shared_mean/full'])
    assert variant_applies(variant_by_name('shared_mean/full'), 'unknown')
    assert not variant_applies(variant_by_name('shared_mean/target_only'),
                               'unknown')
    assert arm_applies('exact_key_value', 'fact')
    assert not arm_applies('exact_key_value', 'unknown')
    assert all(arm_applies(arm, 'unknown')
               for arm in READ_ARMS if arm != 'exact_key_value')


def test_inventory_and_workload_are_exact_and_duplicate_free():
    variants = resolve_variants(['shared_mean/full', 'per_depth/full',
                                 'shared_mean/target_only'])
    inventory = planned_inventory(EPISODES, LABELS, variants=variants,
                                  layers=LAYERS)
    assert inventory == planned_inventory(EPISODES, LABELS, variants=variants,
                                          layers=LAYERS)
    assert len(inventory) == len(set(inventory))
    assert all(len(row) == 6 for row in inventory)
    assert all(row[5] in LAYERS for row in inventory)
    assert any(row[1] == 'exact_key_value' for row in inventory)
    assert all(row[1] != 'exact_key_value'
               or LABELS[row[2]][row[3]]['targets'][row[4]]['kind'] == 'fact'
               for row in inventory)
    workload = planned_workload(EPISODES, LABELS, variants=variants,
                                protocol={'layers': LAYERS,
                                          'limits': DEFAULT_LIMITS})
    assert workload['readout_rows'] == len(inventory)
    assert workload['variants'] == len(variants)
    assert workload['state_builds'] >= workload['states']
    assert workload['planned_write_encodes'] > 0
    with pytest.raises(ValueError, match='labels'):
        planned_inventory(EPISODES, {})


def test_write_position_histories_ablate_exactly():
    episode = next(item for item in EPISODES
                   if item['complexity'] == 'correction')
    states = episode_states(episode)
    corrected = states[-1]
    records, _, _ = _record_views(episode, corrected)
    labels = LABELS[episode['episode_id']]['corrected']
    correction_query = labels['corrected_query_ids'][0]
    target = labels['targets'][correction_query]['target_entry_id']
    full = variant_history(records, variant_by_name('shared_mean/full'))
    reversed_history = variant_history(
        records, variant_by_name('shared_mean/reversed'))
    assert full == list(records)
    assert sorted(item['record_id'] for item in reversed_history) == sorted(
        item['record_id'] for item in records)
    assert reversed_history != full
    assert reversed_history[-1]['kind'] == 'source_correction'
    assert {item['observed_at'] for item in reversed_history[:-1]} == {
        records[0]['observed_at']}
    only = variant_history(records, variant_by_name('shared_mean/target_only'),
                           target_id=target)
    prefix = variant_history(
        records, variant_by_name('shared_mean/prefix_through_target'),
        target_id=target)
    later = variant_history(
        records, variant_by_name('shared_mean/target_and_later'),
        target_id=target)
    refresh = variant_history(
        records, variant_by_name('shared_mean/target_refresh_last'),
        target_id=target)
    assert [item['record_id'] for item in only] == [target]
    index = [item['record_id'] for item in records].index(target)
    assert [item['record_id'] for item in prefix] == [
        item['record_id'] for item in records[:index + 1]]
    assert [item['record_id'] for item in later] == (
        [target] + [item['record_id'] for item in records[index + 1:]])
    assert [item['record_id'] for item in refresh] == (
        [item['record_id'] for item in records] + [target])
    with pytest.raises(ValueError, match='requires a target'):
        variant_history(records, variant_by_name('shared_mean/target_only'))


def test_tiny_run_holds_rows_candidates_and_parity(tiny_run):
    model, output, report = tiny_run
    assert report['status'] == 'completed'
    assert report['optimizer_updates'] == 0
    assert report['frozen']['parameter_versions_equal'] is True
    assert report['frozen']['requires_grad_all_false'] is True
    assert report['counts']['readout_rows'] == len(report['inventory'])
    assert report['counts']['readout_rows'] == len(
        (output / 'readouts.jsonl').read_text().splitlines())
    assert report['counts']['state_builds'] == len(
        (output / 'states.jsonl').read_text().splitlines())
    assert report['mechanical_gates']['production_parity']['status'] == 'pass'
    assert report['mechanical_gates']['full_fact_parity']['status'] == 'pass'
    assert report['mechanical_gates']['candidate_inventory_parity']['status']
    assert report['mechanical_gates']['row_inventory']['status'] == 'pass'
    assert report['mechanical_gates']['full_fact_parity']['metrics'][
        'ordered_identity'] is True
    assert report['mechanical_valid'] is True
    assert report['role_quality']['exact_key_value']['status'] == 'report_only'
    assert report['role_quality']['exact_key_value']['enforced'] is False
    assert report['role_quality']['query_key_value']['status'] == 'report_only'
    for name in ('readouts.jsonl', 'states.jsonl', 'codebooks.json',
                 'rankings.jsonl', 'failures.json', 'mechanical.json'):
        assert (output / name).is_file()
    rows = [json.loads(line) for line in
            (output / 'readouts.jsonl').read_text().splitlines()]
    assert rows == sorted(rows, key=lambda row: (
        row['variant'], row['episode_id'], row['state'], row['case_id'],
        row['arm'], row['layer']))
    for row in rows:
        assert not any(marker in row for marker in
                       ('expected', 'expected_answer', 'gold', 'label',
                        'audit_label', 'target_role', 'kind'))
        assert row['state_sha256'] and row['final_write_id']
    codebooks = json.loads((output / 'codebooks.json').read_text())['codebooks']
    for key, book in codebooks.items():
        assert book['format'] == CODEBOOK_FORMAT
        assert set(book['layers']) == {str(layer) for layer in LAYERS}
        for layer in LAYERS:
            assert sorted(book['layers'][str(layer)]) == sorted(
                book['live_entry_ids'] + book['negative_entry_ids'])
    rankings = [json.loads(line) for line in
                (output / 'rankings.jsonl').read_text().splitlines()]
    groups = {}
    for row in rankings:
        key = (row['episode_id'], row['state'], row['case_id'],
               row['layer'], row['arm'])
        groups.setdefault(key, set()).add(
            (row['candidate_count'], tuple(sorted(row['scores']))))
    assert groups and all(len(signatures) == 1
                          for signatures in groups.values())
    # Every one of the 20 variants is preserved, including per-depth and
    # failing reversed/position/gate/refresh combinations.
    reported = {row['variant'] for row in rows}
    assert reported == set(FIXTURE_VARIANTS)
    assert len(reported) == 20
    assert {'per_depth/reversed', 'per_depth/target_only',
            'per_depth/prefix_through_target', 'per_depth/target_and_later',
            'per_depth/forced_full_erase', 'per_depth/forced_full_decay',
            'per_depth/forced_full_gate', 'per_depth/forced_full_forget',
            'per_depth/target_refresh_last'} <= reported
    failures = json.loads((output / 'failures.json').read_text())
    assert failures['failed_variants']
    assert set(failures['failed_variants']) <= reported


def test_query_key_address_is_raw_and_variant_independent(tiny_run):
    model, output, report = tiny_run
    rows = [json.loads(line) for line in
            (output / 'readouts.jsonl').read_text().splitlines()]
    address = [row for row in rows if row['arm'] == 'query_key_address']
    assert address
    episodes = {episode['episode_id']: episode for episode in EPISODES}
    expected_by_key = {}
    for row in address:
        key = (row['episode_id'], row['case_id'], row['layer'])
        if key not in expected_by_key:
            query = next(item for item in episodes[row['episode_id']]['queries']
                         if item['query_id'] == row['case_id'])
            expected = query_key(model, Tokenizer(), query['text'],
                                 layer=row['layer'])[0, 0]
            expected_by_key[key] = expected.detach().float().cpu().tolist()
        assert row['readout'] == expected_by_key[key]
    groups = {}
    for row in address:
        key = (row['episode_id'], row['state'], row['case_id'], row['layer'])
        groups.setdefault(key, set()).add(row['readout_sha256'])
    assert groups and all(len(values) == 1 for values in groups.values())
    rankings = [json.loads(line) for line in
                (output / 'rankings.jsonl').read_text().splitlines()]
    ranking_groups = {}
    for row in rankings:
        if row['arm'] != 'query_key_address':
            continue
        key = (row['episode_id'], row['state'], row['case_id'], row['layer'])
        ranking_groups.setdefault(key, set()).add(
            (tuple(sorted(row['scores'])), row['target_rank'],
             row['top_entry_id']))
    assert ranking_groups and all(len(values) == 1
                                  for values in ranking_groups.values())


def test_ordered_fact_parity_rejects_reordered_history(tiny_run):
    _, output, _ = tiny_run
    states = [json.loads(line) for line in
              (output / 'states.jsonl').read_text().splitlines()]
    variants = resolve_variants(FIXTURE_VARIANTS)
    good = _full_fact_parity(states, LABELS, variants, EPISODES)
    assert good['all_equal'] is True
    target = next(item for item in states
                  if item['variant'] == 'shared_mean/full'
                  and len(item['written_record_ids']) > 1)
    changed = copy.deepcopy(states)
    match = next(item for item in changed
                 if item['variant'] == target['variant']
                 and item['episode_id'] == target['episode_id']
                 and item['state'] == target['state']
                 and item['case_id'] == target['case_id'])
    match['written_record_ids'][0], match['written_record_ids'][1] = (
        match['written_record_ids'][1], match['written_record_ids'][0])
    rejected = _full_fact_parity(changed, LABELS, variants, EPISODES)
    assert rejected['all_equal'] is False
    assert rejected['ordered_identity'] is True
    assert rejected['failures']


def _metric(top1, *, rank=1.0, margin=0.1, n=10, false_recall=0.0):
    return {
        'fact': {'n': n, 'top1_rate': top1, 'mean_rank': rank,
                 'mean_margin': margin},
        'false_recall_rate': false_recall,
    }


def _metrics(**variants):
    result = {
        'shared_mean/full': {
            'exact_key_value': _metric(0.15, rank=3.0, margin=-0.02),
            'query_key_value': _metric(0.15, rank=3.0, margin=-0.02),
            'query_key_address': _metric(0.95, rank=1.03, margin=0.08),
        },
        'per_depth/full': {
            'exact_key_value': _metric(0.2, rank=3.0, margin=-0.02),
        },
    }
    for key, value in variants.items():
        result.setdefault(key, {}).update(value)
    return result


def test_first_failing_mechanism_order_and_precedence():
    shared = select_first_failing_mechanism(_metrics(**{
        'per_depth/full': {'exact_key_value': _metric(
            0.95, rank=1.05, margin=0.08)}}))
    assert shared['decision'] == 'shared_reduction_interference'
    assert shared['first_failing_mechanism'] == 'shared_depth_convex_mean'
    assert shared['address_passes'] is True

    position = select_first_failing_mechanism(_metrics(**{
        'shared_mean/target_only': {
            'exact_key_value': _metric(0.95, rank=1.05, margin=0.08)}}))
    assert position['decision'] == 'write_position_or_recency_interference'
    assert position['first_failing_mechanism'] == 'write_position'

    gate = select_first_failing_mechanism(_metrics(**{
        'shared_mean/forced_full_decay': {
            'exact_key_value': _metric(0.95, rank=1.05, margin=0.08)}}))
    assert gate['decision'] == 'forced_full_gate_dynamics_interference'
    assert gate['first_failing_mechanism'] == 'forced_full_erase_decay_gate'

    refresh = select_first_failing_mechanism(_metrics(**{
        'shared_mean/target_refresh_last': {
            'exact_key_value': _metric(0.95, rank=1.05, margin=0.08)}}))
    assert refresh['decision'] == 'last_write_recency_interference'

    # Refresh is evaluated before forced-full gate controls.
    refresh_precedence = select_first_failing_mechanism(_metrics(**{
        'shared_mean/target_refresh_last': {
            'exact_key_value': _metric(0.95, rank=1.05, margin=0.08)},
        'shared_mean/forced_full_decay': {
            'exact_key_value': _metric(0.95, rank=1.05, margin=0.08)}}))
    assert refresh_precedence['decision'] == 'last_write_recency_interference'

    # Shared reduction takes precedence when a later mechanism also passes.
    precedence = select_first_failing_mechanism(_metrics(**{
        'per_depth/full': {'exact_key_value': _metric(
            0.95, rank=1.05, margin=0.08)},
        'shared_mean/target_only': {'exact_key_value': _metric(
            0.95, rank=1.05, margin=0.08)}}))
    assert precedence['decision'] == 'shared_reduction_interference'

    downstream = select_first_failing_mechanism(_metrics())
    assert downstream['decision'] == 'value_readout_downstream'
    value_passes = select_first_failing_mechanism(_metrics(**{
        'shared_mean/full': {
            'exact_key_value': _metric(0.15, rank=3.0, margin=-0.02),
            'query_key_value': _metric(0.95, rank=1.05, margin=0.08),
            'query_key_address': _metric(0.95, rank=1.03, margin=0.08)}}))
    assert value_passes['decision'] == 'exact_key_specific_failure'
    assert value_passes['first_failing_mechanism'] == 'exact_observed_key_read_path'

    unresolved_metrics = _metrics()
    unresolved_metrics['shared_mean/full']['query_key_address'] = _metric(
        0.2, rank=3.0, margin=-0.02)
    unresolved = select_first_failing_mechanism(unresolved_metrics)
    assert unresolved['decision'] == 'no_isolated_update_dynamics_cause'
    assert unresolved['first_failing_mechanism'] == 'unresolved'

    invalid = select_first_failing_mechanism(
        _metrics(), mechanical_valid=False)
    assert invalid['decision'] == 'invalid_mechanical_evidence'
    with pytest.raises(ValueError, match='baseline'):
        select_first_failing_mechanism({'per_depth/full': {}})
    # A small improvement below the declared delta is not isolation.
    weak = select_first_failing_mechanism(_metrics(**{
        'per_depth/full': {'exact_key_value': _metric(
            0.25, rank=2.7, margin=-0.01)}}))
    assert weak['decision'] == 'value_readout_downstream'


def _synthetic_case():
    query_texts = {
        'q0': 'What is attr for ent?',
        'q1': 'What is retained attr?',
        'q2': 'What is corrected attr?',
        'unknown': 'Tell me the absent attr.',
    }
    entries = {
        'rec-target': {'value_vector': [[1.0, 0.0, 0.0]],
                       'write_key': [[1.0, 0.0, 0.0]]},
        'rec-retained': {'value_vector': [[0.0, 1.0, 0.0]],
                         'write_key': [[0.0, 1.0, 0.0]]},
        'rec-corr': {'value_vector': [[0.0, 0.0, 1.0]],
                     'write_key': [[0.0, 0.0, 1.0]]},
        'rec-old': {'value_vector': [[-1.0, 0.0, 0.0]],
                    'write_key': [[-1.0, 0.0, 0.0]]},
    }
    codebook = {
        'format': CODEBOOK_FORMAT, 'episode_id': 'epi-x', 'world_id': 'world-x',
        'state': 'corrected', 'cutoff': 2,
        'observation_sha256': 'obs', 'layers': {'0': entries},
        'live_entry_ids': ['rec-corr', 'rec-retained', 'rec-target'],
        'negative_entry_ids': ['rec-old'],
    }
    labels = {
        'state': 'corrected', 'cutoff': 2, 'split': 'train',
        'complexity': 'correction', 'episode_id': 'epi-x',
        'world_id': 'world-x',
        'live_entries': [
            {'record_id': 'rec-target', 'role': 'source',
             'source_family': 'source'},
            {'record_id': 'rec-retained', 'role': 'retained',
             'source_family': 'source'},
            {'record_id': 'rec-corr', 'role': 'correction_target',
             'source_family': 'source'},
        ],
        'negative_entries': [
            {'record_id': 'rec-old', 'role': 'superseded_negative',
             'source_family': 'source'}],
        'targets': {
            'q0': {'kind': 'fact', 'target_entry_id': 'rec-target',
                   'target_role': 'source',
                   'target_source_family': 'source',
                   'query_text_sha256': _sha(query_texts['q0'])},
            'q1': {'kind': 'fact', 'target_entry_id': 'rec-retained',
                   'target_role': 'retained',
                   'target_source_family': 'source',
                   'query_text_sha256': _sha(query_texts['q1'])},
            'q2': {'kind': 'fact', 'target_entry_id': 'rec-corr',
                   'target_role': 'correction_target',
                   'target_source_family': 'source',
                   'query_text_sha256': _sha(query_texts['q2'])},
            'unknown': {'kind': 'unknown', 'target_entry_id': None,
                        'target_role': 'unknown',
                        'target_source_family': 'none',
                        'query_text_sha256': _sha(query_texts['unknown'])},
        },
    }
    readouts = {
        ('exact_key_value', 'q0'): [1.0, 0.0, 0.0],
        ('exact_key_value', 'q1'): [-1.0, 0.0, 0.0],
        ('exact_key_value', 'q2'): [0.0, 0.0, 1.0],
        ('query_key_value', 'q0'): [1.0, 0.0, 0.0],
        ('query_key_value', 'q1'): [-1.0, 0.0, 0.0],
        ('query_key_value', 'q2'): [0.0, 0.0, 1.0],
        ('query_key_value', 'unknown'): [1.0, 0.2, 0.2],
        ('query_key_address', 'q0'): [1.0, 0.0, 0.0],
        ('query_key_address', 'q1'): [0.0, 1.0, 0.0],
        ('query_key_address', 'q2'): [0.0, 0.0, 1.0],
        ('query_key_address', 'unknown'): [1.0, 0.2, 0.2],
        ('context_removed', 'q0'): [0.0, 0.0, 0.0],
        ('context_removed', 'q1'): [0.0, 0.0, 0.0],
        ('context_removed', 'q2'): [0.0, 0.0, 0.0],
        ('context_removed', 'unknown'): [0.0, 0.0, 0.0],
    }
    final_ids = {'q0': 'rec-target', 'q1': 'rec-retained',
                 'q2': 'rec-corr', 'unknown': 'rec-target'}
    rows = []
    for (arm, query_id), values in readouts.items():
        vector = [values]
        rows.append({
            'format': READOUT_FORMAT, 'variant': 'shared_mean/full',
            'reducer': 'shared_mean', 'arm': arm,
            'readout_space': ('key' if arm == 'query_key_address'
                              else 'value'),
            'episode_id': 'epi-x', 'world_id': 'world-x',
            'state': 'corrected', 'cutoff': 2,
            'case_id': query_id, 'query_id': query_id,
            'query_text_sha256': labels['targets'][query_id]['query_text_sha256'],
            'codebook_key': 'epi-x/corrected', 'layer': 0,
            'observation_sha256': 'obs', 'state_sha256': 'state',
            'final_write_id': final_ids[query_id], 'status': 'ok',
            'readout': vector, 'readout_sha256': _sha(vector),
            'readout_norm': sum(value * value for value in values) ** 0.5,
        })
    rows.sort(key=lambda row: (row['case_id'], row['arm']))
    inventory = [(row['variant'], row['arm'], row['episode_id'],
                  row['state'], row['case_id'], row['layer'])
                 for row in rows]
    mechanical = {
        'frozen': {'optimizer_updates': 0, 'parameters_updated': [],
                   'parameter_versions_equal': True,
                   'requires_grad_all_false': True},
        'production_parity': {'rows': [{'equal': True}],
                              'all_equal': True, 'max_abs_difference': 0.0},
        'full_fact_parity': {'rows': 1, 'all_equal': True},
        'limits': DEFAULT_LIMITS,
    }
    return rows, {'epi-x/corrected': codebook}, {'epi-x': {'corrected': labels}}, \
        inventory, mechanical


def test_scoring_unknown_correction_unaffected_and_false_recall_kinds():
    rows, codebooks, labels, inventory, mechanical = _synthetic_case()
    scored = score_update_dynamics(
        rows, codebooks, labels, mechanical=mechanical, inventory=inventory,
        variants=[variant_by_name('shared_mean/full')])
    metrics = scored['metrics']['shared_mean/full']
    assert metrics['exact_key_value']['correction']['top1_rate'] == 1.0
    assert metrics['exact_key_value']['unaffected']['top1_rate'] == 0.0
    assert metrics['query_key_value']['unknown']['false_recall_rate'] == 1.0
    assert metrics['context_removed']['false_recall_rate'] == 0.0
    assert metrics['exact_key_value']['final_write_top_rate'] is not None
    exact = [row for row in scored['rankings']
             if row['arm'] == 'exact_key_value']
    retained = next(row for row in exact if row['case_id'] == 'q1')
    assert retained['target_rank'] == 2
    assert retained['false_recall_kind'] == 'superseded'
    assert retained['superseded_selected'] is True
    unknown = next(row for row in scored['rankings']
                   if row['arm'] == 'query_key_value'
                   and row['case_id'] == 'unknown')
    assert unknown['false_recall_kind'] == 'unknown_address'
    assert unknown['target_entry_id'] is None
    assert scored['mechanical_valid'] is True


def test_scoring_rejects_duplicate_missing_and_foreign_rows():
    rows, codebooks, labels, inventory, mechanical = _synthetic_case()
    duplicated = copy.deepcopy(rows)
    duplicated.append(copy.deepcopy(duplicated[0]))
    with pytest.raises(ValueError, match='Duplicate'):
        score_update_dynamics(duplicated, codebooks, labels,
                              mechanical=mechanical, inventory=inventory,
                              variants=[variant_by_name('shared_mean/full')])
    missing = copy.deepcopy(rows)
    missing.pop()
    with pytest.raises(ValueError, match='Missing'):
        score_update_dynamics(missing, codebooks, labels,
                              mechanical=mechanical, inventory=inventory,
                              variants=[variant_by_name('shared_mean/full')])
    forged = copy.deepcopy(codebooks)
    forged['epi-x/corrected']['observation_sha256'] = 'changed'
    with pytest.raises(ValueError, match='observation differs'):
        score_update_dynamics(rows, forged, labels,
                              mechanical=mechanical, inventory=inventory,
                              variants=[variant_by_name('shared_mean/full')])
    extra = copy.deepcopy(rows)
    extra.append(dict(copy.deepcopy(rows[0]), case_id='q2', arm='exact_key_value'))
    with pytest.raises(ValueError, match='Duplicate'):
        score_update_dynamics(extra, codebooks, labels,
                              mechanical=mechanical, inventory=inventory,
                              variants=[variant_by_name('shared_mean/full')])
    # Mechanical failure invalidates rather than selecting a mechanism.
    broken = copy.deepcopy(mechanical)
    broken['frozen']['parameter_versions_equal'] = False
    invalid = score_update_dynamics(
        rows, codebooks, labels, mechanical=broken, inventory=inventory,
        variants=[variant_by_name('shared_mean/full')])
    assert invalid['mechanical_valid'] is False
    assert invalid['decision'] == 'invalid_mechanical_evidence'


@require_local_artifacts('checkpoints/minicpm5_source',
                         'configs/minicpm5_sealed_lock.local.json',
                         'runs/memory_reliability_v5_attempt01/candidate_extension')
def test_prepare_and_admit_are_deterministic(tmp_path):
    source = tmp_path / 'prepared02'
    source.mkdir()
    (source / 'episodes.json').write_text(json.dumps({
        'format': 'replica.memory_matched.episodes.v1',
        'episodes': EPISODES}, sort_keys=True, indent=2, allow_nan=False) + '\n')
    (source / 'labels.json').write_text(json.dumps({
        'format': 'replica.memory_matched.labels.v1',
        'evaluation_only': True, 'episodes': LABELS},
        sort_keys=True, indent=2, allow_nan=False) + '\n')
    config = json.loads(
        (ROOT / 'configs/memory_update_dynamics_v1.json').read_text())
    config['source_prepared'] = {'path': str(source)}
    config_path = tmp_path / 'config.json'
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + '\n')
    first, second = tmp_path / 'first', tmp_path / 'second'
    manifest = prepare_update_dynamics(
        first, config_path=config_path, config=config)
    prepare_update_dynamics(second, config_path=config_path, config=config)
    assert manifest['format'] == 'replica.memory_update_dynamics.prepared.v1'
    for name in ('episodes.json', 'labels.json', 'inventory.json',
                 'protocol.json'):
        assert (first / name).read_bytes() == (second / name).read_bytes()
    first_manifest = json.loads((first / 'manifest.json').read_text())
    second_manifest = json.loads((second / 'manifest.json').read_text())
    first_manifest.pop('created_at_utc')
    second_manifest.pop('created_at_utc')
    assert first_manifest == second_manifest
    inventory = json.loads((first / 'inventory.json').read_text())
    assert inventory['format'] == INVENTORY_FORMAT
    assert len(inventory['rows']) == manifest['planned_workload']['readout_rows']
    receipt = admit_update_dynamics_prepared(first)
    assert receipt['status'] == 'admitted'
    assert receipt['checks']['inventory_regenerated'] is True
    assert receipt['checks']['no_native_computation'] is True
    protocol = json.loads((first / 'protocol.json').read_text())
    assert protocol['format'] == 'replica.memory_update_dynamics.protocol.v1'
    assert protocol['config_format'] == CONFIG_FORMAT
    assert protocol['source_prepared']['path'] == str(source)
    assert protocol['variants'] == variant_plan()
    assert protocol['planned_workload'] == manifest['planned_workload']
    # A changed source byte is rejected, and the rejection is preserved.
    (source / 'episodes.json').write_text(
        (source / 'episodes.json').read_text() + '\n')
    with pytest.raises(ValueError, match='Source prepared02 bytes changed'):
        admit_update_dynamics_prepared(first)
    rejected = json.loads((first / 'admission.json').read_text())
    assert rejected['status'] == 'rejected'

"""Deterministic no-update store/address/readout and preparation-pin tests."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from architecture_helpers import tiny_adapter
from replica_lab.architecture.contracts import FastMemorySpec
from replica_lab.memory_diagnosis import (
    DEFAULT_LIMITS,
    PROTOCOL_FORMAT,
    admit_probe,
    generate_worlds,
    native_reader_query,
    planned_workload,
    prepare_probe,
    run_diagnostic,
    score_diagnostic,
    validate_worlds,
)
from replica_lab.runtime.memory_representation import FACT_REPRESENTATION

ROOT = Path(__file__).resolve().parents[1]
SPEC = FastMemorySpec(heads=2, key_size=8, value_size=8, update_rule='sgdn2',
                      depth_mode='shared', depth_writers=True)


class Tokenizer:
    eos_token_id = 63

    def __call__(self, text, *, add_special_tokens, return_offsets_mapping=False,
                 **kwargs):
        ids = [1 + byte % 61 for byte in hashlib.sha256(text.encode()).digest()[:8]]
        result = {'input_ids': ids}
        if return_offsets_mapping:
            result['offset_mapping'] = [(index, index + 1) for index in range(len(ids))]
        return result

    def decode(self, ids, **kwargs):
        return ' '.join(map(str, ids))

    def apply_chat_template(self, messages, **kwargs):
        return messages[0]['content']


def protocol(**overrides):
    value = {
        'format': PROTOCOL_FORMAT,
        'seed': 731,
        'world_count': 2,
        'facts_per_world': 2,
        'distractors_per_world': 1,
        'correction_index': 0,
        'revocation_index': 1,
        'layer': 0,
        'representation': FACT_REPRESENTATION,
        'native_reader': True,
        'thresholds': {},
    }
    value.update(overrides)
    return value


def synthetic_codebook():
    result = {}
    for stage in ('initial', 'corrected', 'current_view'):
        key = f'synthetic/world/{stage}/layer-0'
        result[key] = {
            'key': key, 'world_id': 'synthetic/world', 'stage': stage, 'layer': 0,
            'entries': {
                'a': {'value_vector': [[1.0, 0.0]],
                      'role': 'source' if stage == 'initial' else 'correction_target',
                      'source_family': 'source'},
                'b': {'value_vector': [[0.0, 1.0]], 'role': 'source',
                      'source_family': 'source'},
                'c': {'value_vector': [[-1.0, 0.0]], 'role': 'distractor',
                      'source_family': 'independent_world'},
            },
            'negative_entries': ({
                'old': {'value_vector': [[0.0, -1.0]], 'role': 'superseded_negative',
                        'negative': True, 'live': False, 'source_family': 'source'},
            } if stage != 'initial' else {}),
            'key_index': {},
        }
    return result


def synthetic_row(kind, readout, *, stage='initial', target='a', expected='target',
                  role='source', source_family='source', **extra):
    row = {
        'status': 'ok',
        'world_id': 'synthetic/world',
        'query_kind': kind,
        'stage': stage,
        'expected': expected,
        'role': role,
        'source_family': source_family,
        'target_entry_id': target,
        'codebook_key': f'synthetic/world/{stage}/layer-0',
        'readout': readout,
        'layer': 0,
    }
    row.update(extra)
    return row


def synthetic_mechanical():
    return {
        'frozen': {'optimizer_updates': 0, 'parameter_versions_equal': True,
                   'parameters_updated': [], 'requires_grad_all_false': True},
        'correction': {'superseded_entry_absent': True,
                       'superseded_negative_present': True},
        'revocation': {'revoked_entry_absent': True},
        'source_binding': {'all_bound': True, 'stale_rejected': True,
                           'require_source_binding': True},
        'restart': {'all_state_equal': True, 'all_readout_equal': True},
    }


def passing_rows():
    rows = [
        synthetic_row('observed_write_key', [[1.0, 0.0]], target='a'),
        synthetic_row('observed_write_key', [[0.0, 1.0]], target='b'),
        synthetic_row('observed_write_key', [[-1.0, 0.0]], target='c',
                      role='distractor', source_family='independent_world'),
        synthetic_row('canonical_address', [[1.0, 0.0]], target='a'),
        synthetic_row('canonical_paraphrase', [[0.0, 1.0]], target='b'),
        synthetic_row('canonical_address', [[-1.0, 0.0]], target='c',
                      role='distractor', source_family='independent_world'),
        synthetic_row('native_reader', [[1.0, 0.0]], target='a'),
        synthetic_row('native_reader', [[0.0, 1.0]], target='b'),
        synthetic_row('wrong_key', [[0.0, 1.0]], target='a',
                      expected='key_owner', role='control', source_family='control',
                      control_owner_entry_id='b'),
        synthetic_row('zero', [[0.0, 0.0]], target=None, expected='no_match',
                      role='control', source_family='control'),
        synthetic_row('unknown', [[0.3, 0.3]], target=None, expected='no_match',
                      role='control', source_family='control'),
        synthetic_row('cross_world', [[0.3, 0.3]], target=None, expected='no_match',
                      role='control', source_family='control'),
        synthetic_row('observed_write_key', [[1.0, 0.0]], stage='corrected',
                      target='a', role='correction_target', superseded_entry_id='old'),
        synthetic_row('canonical_address', [[1.0, 0.0]], stage='corrected',
                      target='a', role='correction_target', superseded_entry_id='old'),
        synthetic_row('observed_write_key', [[0.0, 1.0]], stage='corrected',
                      target='b', role='source'),
        synthetic_row('revoked_query', [[0.3, 0.3]], stage='current_view',
                      target=None, expected='no_match', role='revoked_query',
                      source_family='control', revoked_entry_id='revoked'),
    ]
    return rows


def test_world_generation_is_seeded_disjoint_and_label_free():
    first = generate_worlds(seed=55, world_count=2, facts_per_world=2,
                            distractors_per_world=1, correction_index=0,
                            revocation_index=1)
    second = generate_worlds(seed=55, world_count=2, facts_per_world=2,
                             distractors_per_world=1, correction_index=0,
                             revocation_index=1)
    assert first == second
    other = generate_worlds(seed=56, world_count=2, facts_per_world=2,
                            distractors_per_world=1, correction_index=0,
                            revocation_index=1)
    assert first != other
    values = [fact['value'] for world in first for fact in world['facts']]
    assert len(values) == len(set(values))
    assert all(collections_are_same(first[index], index) for index in (0, 1))
    for world in first:
        assert set(world) == {'world_id', 'family_id', 'facts', 'distractors',
                              'correction', 'revocation_fact_id'}
        for fact in world['facts']:
            assert 'expected_answer' not in fact and 'label' not in fact
        assert world['distractors'][0]['origin_world_id'] != world['world_id']
    bad = deepcopy(first)
    bad[0]['facts'][0]['expected_answer'] = 'forbidden'
    with pytest.raises(ValueError):
        validate_worlds(bad)
    duplicate = deepcopy(first)
    duplicate[0]['facts'][1]['value'] = duplicate[0]['facts'][0]['value']
    with pytest.raises(ValueError):
        validate_worlds(duplicate)


def collections_are_same(world, index):
    return world['family_id'] == f'family-{index:02d}'


def test_prepare_and_admit_bind_exact_source_data_and_protocol_pins(tmp_path):
    prepared = tmp_path / 'prepared'
    manifest = prepare_probe(
        prepared, seed=19, world_count=2, facts_per_world=2,
        distractors_per_world=1, correction_index=0, revocation_index=1,
        layer=13, model_path=ROOT / 'checkpoints/minicpm5_source',
        model_lock=ROOT / 'configs/minicpm5_sealed_lock.local.json')
    assert manifest['format'] == 'replica.memory_storage_probe.prepared.v1'
    assert manifest['native_computation'] is False
    for name in ('data.json', 'protocol.json', 'manifest.json'):
        assert (prepared / name).is_file()
    protocol_value = json.loads((prepared / 'protocol.json').read_text())
    assert protocol_value['source']['source_revision'] == (
        'f97400052a43d642bbc6e9975e2397e3ae6a6b52')
    assert protocol_value['data']['sha256'] == hashlib.sha256(
        (prepared / 'data.json').read_bytes()).hexdigest()
    assert protocol_value['implementation_sha256']
    receipt = admit_probe(prepared)
    assert receipt['status'] == 'admitted'
    assert receipt['checks']['seeded_world_regeneration'] is True
    assert (prepared / 'admission.json').is_file()

    original = (prepared / 'data.json').read_bytes()
    tampered = json.loads(original)
    tampered['worlds'][0]['facts'][0]['expected_answer'] = 'injected'
    (prepared / 'data.json').write_text(json.dumps(tampered))
    with pytest.raises(ValueError):
        admit_probe(prepared, write_receipt=False)
    (prepared / 'data.json').write_bytes(original)

    protocol_bytes = (prepared / 'protocol.json').read_bytes()
    tampered_protocol = json.loads(protocol_bytes)
    tampered_protocol['thresholds']['storage_top1_floor'] = 1.0
    (prepared / 'protocol.json').write_text(json.dumps(tampered_protocol))
    with pytest.raises(ValueError):
        admit_probe(prepared, write_receipt=False)
    (prepared / 'protocol.json').write_bytes(protocol_bytes)
    assert admit_probe(prepared, write_receipt=False)['status'] == 'admitted'


def test_native_reader_unavailable_is_reported_not_fabricated():
    query, reason = native_reader_query(object(), Tokenizer(), 'query', layer=0)
    assert query is None
    assert 'fast_ports' in reason


def test_planned_workload_counts_reloads_and_enforces_limit():
    workload = planned_workload(protocol())
    # 2 worlds x 2 facts x 1 distractor, native reader, three states, two
    # evaluations per state: targets 128, controls 52, total 180.
    assert workload['total_planned_rows'] == 180
    assert workload['planned_target_rows'] == 128
    assert workload['planned_control_rows'] == 52
    assert workload['planned_unsupported_rows'] == 0
    assert workload['planned_diagnostic_cases'] == 90
    assert workload['planned_target_cases'] == 64
    assert workload['planned_control_cases'] == 26
    assert workload['evaluations_per_state'] == 2
    assert workload['states'] == 6
    assert workload['per_stage']['initial']['live_entries'] == 3
    assert workload['per_stage']['current_view']['live_entries'] == 2

    with pytest.raises(ValueError, match='Scheduled workload'):
        planned_workload(protocol(world_count=64, facts_per_world=32,
                                  distractors_per_world=64))
    with pytest.raises(ValueError, match='Scheduled workload'):
        planned_workload(protocol(limits=dict(DEFAULT_LIMITS, max_queries=100)))
    assert planned_workload(
        protocol(limits=dict(DEFAULT_LIMITS, max_queries=180)))['total_planned_rows'] == 180

    zero = planned_workload(protocol(world_count=1, distractors_per_world=0))
    assert zero['distractors_per_world'] == 0
    assert zero['total_planned_rows'] == 66
    assert zero['planned_diagnostic_cases'] == 33
    worlds = generate_worlds(seed=9, world_count=1, facts_per_world=2,
                             distractors_per_world=0, correction_index=0,
                             revocation_index=1)
    assert worlds[0]['distractors'] == []
    validate_worlds(worlds)


def test_scoring_rejects_poor_storage_before_any_reader_comparator():
    rows = passing_rows()
    for row in rows:
        if row['query_kind'] == 'observed_write_key' and row['target_entry_id'] in {'a', 'b'}:
            row['readout'] = [[0.0, 1.0]] if row['target_entry_id'] == 'a' else [[1.0, 0.0]]
            row['matched'] = True  # Producer claims are ignored by the scorer.
            row['top_entry_id'] = row['target_entry_id']
    result = score_diagnostic(
        rows, synthetic_codebook(),
        thresholds={'storage_top1_floor': 0.8},
        mechanical=synthetic_mechanical())
    assert result['gates']['storage_recovery']['metrics']['storage']['top1'] == 0.2
    assert result['gates']['storage_recovery']['status'] == 'fail'
    assert result['decision'] == 'defer_useful_memory_storage_recovery_failed'
    assert result['pointer_copy_comparator'] == (
        'not_constructed_storage_recovery_gate_blocked')
    assert result['promote_v5_memory'] is False
    assert result['smallest_next_repair_discriminator']['gate'] == 'storage_recovery'


def test_scoring_recomputes_passes_from_raw_readouts_and_keeps_v5_rejected():
    rows = passing_rows()
    for row in rows:
        if row['query_kind'] == 'observed_write_key' and row['target_entry_id'] == 'a':
            row['matched'] = False
            row['top_entry_id'] = 'b'
    result = score_diagnostic(rows, synthetic_codebook(),
                              thresholds=None, mechanical=synthetic_mechanical())
    assert result['gates']['storage_recovery']['status'] == 'pass'
    assert result['gates']['storage_recovery']['metrics']['storage']['top1'] == 1.0
    assert result['gates']['addressing']['status'] == 'pass'
    assert result['gates']['native_readout']['status'] == 'pass'
    assert result['gates']['correction']['status'] == 'pass'
    assert result['gates']['revocation']['status'] == 'pass'
    assert result['decision'] == 'diagnostic_supports_targeted_readout_experiment_only'
    assert result['targeted_training_justified'] is True
    assert result['promote_v5_memory'] is False
    assert result['v5_status'] == 'rejected_by_prior_evidence'
    assert result['neural_promotion'] is False
    assert result['deployment'] == 'diagnostic_only_not_published'
    assert result['pointer_copy_comparator'] == 'not_constructed_no_learned_readout_gate'
    assert 'full_blind_training_arm' in result['unrun_scope']


def test_scoring_detects_readout_selecting_superseded_value():
    codebook = synthetic_codebook()
    old_vector = codebook['synthetic/world/corrected/layer-0'][
        'negative_entries']['old']['value_vector']
    rows = passing_rows()
    mutated = 0
    for row in rows:
        if (row['stage'] == 'corrected' and row['role'] == 'correction_target'
                and row['status'] == 'ok'):
            row['readout'] = old_vector
            mutated += 1
    assert mutated >= 2
    result = score_diagnostic(rows, codebook, thresholds=None,
                              mechanical=synthetic_mechanical())
    assert result['gates']['storage_recovery']['status'] == 'pass'
    assert result['gates']['addressing']['status'] == 'pass'
    assert result['gates']['correction']['status'] == 'fail'
    assert result['gates']['correction']['metrics']['superseded']['rate'] == 1.0
    assert result['decision'] == 'defer_useful_memory_correction_or_retention_failed'
    assert result['promote_v5_memory'] is False


def test_scoring_reports_missing_native_reader_without_promoting_memory():
    rows = [row for row in passing_rows() if row['query_kind'] != 'native_reader']
    rows.append(synthetic_row('native_reader', None, target=None,
                              expected='unsupported', role='native_reader',
                              source_family='none', status='unavailable',
                              unavailable_reason='native fast reader unavailable: no fast_ports'))
    result = score_diagnostic(rows, synthetic_codebook(),
                              thresholds=None, mechanical=synthetic_mechanical())
    assert result['gates']['native_readout']['status'] == 'not_available'
    assert result['decision'] == 'defer_native_readout_unavailable'
    assert result['smallest_next_repair_discriminator']['gate'] == 'native_readout'
    assert result['promote_v5_memory'] is False


def test_tiny_fixture_runs_frozen_diagnostic_with_controls_and_restart(tmp_path):
    model = tiny_adapter(fast_memory=SPEC).eval()
    result = run_diagnostic(
        model, Tokenizer(), protocol(), output_dir=tmp_path / 'diagnosis',
        ledger_dir=tmp_path / 'ledger', device='cpu')
    rows = result['readouts']
    assert result['status'] == 'completed'
    assert result['optimizer_updates'] == 0
    assert result['parameters_updated'] == []
    assert result['frozen']['parameter_versions_equal'] is True
    assert result['frozen']['requires_grad_all_false'] is True
    assert result['representation'] == FACT_REPRESENTATION
    assert result['representation_config']['name'] == 'fact_binding_key_value'
    assert result['producer'] and result['model_producer'] and result['dependencies']
    assert result['codebooks'] and result['readouts_sha256']
    readouts_file = tmp_path / 'diagnosis' / 'readouts.jsonl'
    assert readouts_file.is_file()
    assert (tmp_path / 'diagnosis' / 'codebooks.json').is_file()
    persisted_rows = [json.loads(line) for line in readouts_file.read_text().splitlines()]
    assert len(persisted_rows) == 180
    assert {row['phase'] for row in persisted_rows} == {'before_reload', 'after_reload'}
    assert sum(row['phase'] == 'before_reload' for row in persisted_rows) == 90
    assert sum(row['phase'] == 'after_reload' for row in persisted_rows) == 90
    assert result['readouts_sha256'] == hashlib.sha256(
        readouts_file.read_bytes()).hexdigest()

    kinds = {row['query_kind'] for row in rows}
    assert {'observed_write_key', 'canonical_address', 'canonical_paraphrase',
            'native_reader', 'wrong_key', 'zero', 'unknown', 'cross_world',
            'revoked_query'} <= kinds
    zero_rows = [row for row in rows if row['query_kind'] == 'zero']
    assert zero_rows and all(row['readout_norm'] == 0.0 for row in zero_rows)
    assert all(row['accepted'] is False for row in zero_rows)
    storage_rows = [row for row in rows
                    if row['query_kind'] == 'observed_write_key'
                    and row['target_entry_id'] is not None and row['status'] == 'ok']
    assert storage_rows
    assert max(row['target_cosine'] for row in storage_rows) > 0.0
    assert any(row['top_entry_id'] == row['target_entry_id'] for row in storage_rows)
    native_rows = [row for row in rows
                   if row['query_kind'] == 'native_reader' and row['status'] == 'ok']
    assert native_rows and all(row['readout_norm'] > 0.0 for row in native_rows)

    # Codebooks contain only observed source values and addresses, never labels.
    for codebook in result['codebooks'].values():
        for entry in codebook['entries'].values():
            assert 'expected_answer' not in entry
            assert 'value' not in entry
            assert entry['value_sha256']
    correction_entries = [entry
                          for codebook in result['codebooks'].values()
                          if codebook['stage'] == 'corrected'
                          for entry in codebook['entries'].values()
                          if entry['role'] == 'correction_target']
    assert correction_entries and all(entry['correction'] for entry in correction_entries)
    assert result['correction']['superseded_entry_absent'] is True
    assert result['correction']['superseded_negative_present'] is True
    assert result['revocation']['revoked_entry_absent'] is True
    assert [row for row in rows if row['query_kind'] == 'revoked_query'
            and row['status'] == 'ok']
    assert result['source_binding']['all_bound'] is True
    assert result['source_binding']['stale_rejected'] is True
    assert result['restart']['all_state_equal'] is True
    assert result['restart']['all_readout_equal'] is True
    assert result['promote_v5_memory'] is False
    assert result['neural_promotion'] is False
    assert result['decision'].startswith(('defer_', 'diagnostic_'))
    assert result['pointer_copy_comparator'] in {
        'not_constructed_storage_recovery_gate_blocked',
        'not_constructed_no_learned_readout_gate'}
    assert result['counts']['rows'] == len(rows) == 180
    assert result['counts']['evaluated_rows'] == 180
    assert result['counts']['diagnostic_cases'] == 90
    assert result['counts']['before_reload_rows'] == 90
    assert result['counts']['after_reload_rows'] == 90
    assert result['counts']['unique_diagnostic_cases'] == 90
    assert result['counts']['evaluated_target_queries'] == 128
    assert result['counts']['diagnostic_target_queries'] == 64
    assert result['counts']['evaluated_controls'] == 52
    assert result['counts']['diagnostic_controls'] == 26
    assert result['workload']['total_planned_rows'] == 180
    assert result['workload']['actual_rows'] == len(rows) == 180
    assert result['workload']['actual_diagnostic_cases'] == 90
    assert result['workload']['unique_after_reload_cases'] == 90
    assert result['restart']['evaluated_pairs'] == 90
    assert len(result['restart']['details']) == 6
    for detail in result['restart']['details']:
        assert detail['equal'] is True
        assert detail['before_rows'] == detail['after_rows']
        assert detail['mismatch_count'] == 0
    phases = {row['phase'] for row in rows}
    assert phases == {'before_reload', 'after_reload'}
    for summary in result['worlds']:
        assert summary['facts'] == 2
        assert summary['source_family_ids'] == [summary['family_id']]
        assert summary['source_family_count'] == 1
        assert summary['independent_world_family_count'] == 1

    corrected_codebooks = [codebook for codebook in result['codebooks'].values()
                           if codebook['stage'] == 'corrected']
    assert corrected_codebooks
    for codebook in corrected_codebooks:
        assert codebook['negative_entries']
        assert set(codebook['negative_entries']).isdisjoint(codebook['entries'])
        assert all(entry['negative'] is True and entry['live'] is False
                   for entry in codebook['negative_entries'].values())

    after_rows = [row for row in rows if row['phase'] == 'after_reload']
    assert len(after_rows) == 90
    mechanical = {name: result[name] for name in
                  ('frozen', 'correction', 'revocation', 'source_binding', 'restart')}
    consumer = score_diagnostic(after_rows, result['codebooks'], thresholds=None,
                                mechanical=mechanical)
    assert consumer['summary'] == result['summary']
    assert consumer['decision'] == result['decision']
    assert consumer['gates'] == result['gates']
    for field, value in (('role', 'unaffected'), ('expected', 'no_match'),
                         ('source_family', 'control'), ('world_id', 'foreign-world'),
                         ('stage', 'initial')):
        forged = deepcopy(after_rows)
        target = next(row for row in forged if row['stage'] == 'corrected'
                      and row['role'] == 'correction_target' and row['status'] == 'ok')
        target[field] = value
        with pytest.raises(ValueError):
            score_diagnostic(forged, result['codebooks'], thresholds=None, mechanical=mechanical)

    mutated = deepcopy(after_rows)
    changed = 0
    for row in mutated:
        if (row['stage'] == 'corrected' and row['role'] == 'correction_target'
                and row['query_kind'] in ('canonical_address', 'canonical_paraphrase')
                and row['status'] == 'ok'):
            codebook = result['codebooks'][row['codebook_key']]
            old_ids = sorted(codebook['negative_entries'])
            if old_ids:
                row['readout'] = codebook['negative_entries'][old_ids[0]]['value_vector']
                changed += 1
    assert changed >= 2
    rescored = score_diagnostic(mutated, result['codebooks'], thresholds=None,
                                mechanical=mechanical)
    assert rescored['gates']['correction']['metrics']['superseded']['rate'] > 0.0
    assert rescored['promote_v5_memory'] is False

    assert result['resource']['wall_seconds'] >= 0.0
    assert result['resource']['max_rss_bytes'] > 0

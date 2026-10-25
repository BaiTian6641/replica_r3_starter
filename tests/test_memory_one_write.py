"""Versioned one-fact/one-write storage discriminator contracts.

Synthetic tiny adapters exercise the real writer, reduction, read and
save/load paths on CPU. Nothing here loads the native model or promotes memory.
"""
import copy
import json
from pathlib import Path

import pytest
import torch

from architecture_helpers import tiny_adapter
from replica_lab.memory_diagnosis import DEFAULT_LIMITS
from replica_lab.memory_one_write import (
    ONE_WRITE_FORMAT, ONE_WRITE_THRESHOLDS, admit_one_write_probe,
    generate_one_write_worlds, one_write_workload, prepare_one_write,
    run_one_write_diagnostic, score_one_write, validate_one_write_worlds,
)
from test_memory_diagnosis import SPEC, Tokenizer


def protocol(**overrides):
    value = {
        'format': ONE_WRITE_FORMAT, 'mode': 'onewrite', 'seed': 5,
        'world_count': 2, 'layers': [0, 2], 'layer': None,
        'representation': 'fact_binding_key_value.v1', 'native_reader': False,
        'thresholds': dict(ONE_WRITE_THRESHOLDS), 'limits': dict(DEFAULT_LIMITS),
    }
    value.update(overrides)
    return value


def tiny_run(tmp_path, **overrides):
    model = tiny_adapter(fast_memory=SPEC).eval()
    report = run_one_write_diagnostic(
        model, Tokenizer(), protocol(**overrides), output_dir=tmp_path / 'probe',
        ledger_dir=tmp_path / 'ledger', device='cpu')
    rows = report.pop('readouts')
    return model, report, rows


def test_one_write_worlds_are_seeded_single_fact_and_label_free():
    first = generate_one_write_worlds(seed=11, world_count=3)
    assert first == generate_one_write_worlds(seed=11, world_count=3)
    assert first != generate_one_write_worlds(seed=12, world_count=3)
    assert validate_one_write_worlds(first) is True
    keys = [(world['facts'][0]['entity'], world['facts'][0]['attribute'])
            for world in first]
    values = [world['facts'][0]['value'] for world in first]
    assert len(set(keys)) == len(keys) == 3 and len(set(values)) == 3
    for world in first:
        assert world['distractors'] == [] and world['correction'] is None
        assert set(world['facts'][0]) == {'fact_id', 'entity', 'attribute',
                                          'value', 'source_world_id'}
        assert 'expected_answer' not in json.dumps(world)

    duplicated = copy.deepcopy(first)
    duplicated[1]['facts'][0]['value'] = duplicated[0]['facts'][0]['value']
    with pytest.raises(ValueError, match='disjoint'):
        validate_one_write_worlds(duplicated)
    extra = copy.deepcopy(first)
    extra[0]['facts'].append(dict(extra[0]['facts'][0]))
    with pytest.raises(ValueError, match='Duplicate one-write fact'):
        validate_one_write_worlds(extra)
    with pytest.raises(ValueError, match='nonempty'):
        validate_one_write_worlds([])


def test_workload_bounds_rows_before_any_model_allocation():
    workload = one_write_workload(world_count=4, layers=[0, 2])
    assert workload['rows_per_world'] == 16
    assert workload['planned_rows'] == 64 and workload['planned_cases'] == 32
    with pytest.raises(ValueError, match='distinct'):
        one_write_workload(world_count=4, layers=[0, 0])
    with pytest.raises(ValueError, match='distinct'):
        one_write_workload(world_count=4, layers=[0])


def test_tiny_one_write_probe_measures_every_path_without_updates(tmp_path):
    model, report, rows = tiny_run(tmp_path)
    layers = [0, 2]
    assert report['format'] == 'replica.memory_one_write_probe.run.v1'
    assert report['optimizer_updates'] == 0
    assert report['parameters_updated'] == []
    assert report['frozen']['parameter_versions_equal'] is True
    assert report['frozen']['requires_grad_all_false'] is True
    assert report['counts']['evaluated_rows'] == len(rows) == 2 * len(layers) * 4 * 2
    assert report['counts']['before_save_rows'] == report['counts']['after_save_rows']
    assert {row['against'] for row in rows} == {
        'per_depth', 'shared_mean', 'cross_depth', 'zero_state'}
    assert {row['phase'] for row in rows} == {'before_save', 'after_save'}
    zero_rows = [row for row in rows if row['against'] == 'zero_state']
    assert zero_rows and all(row['readout_norm'] == 0.0 for row in zero_rows)
    gates = report['gates']
    assert gates['zero_state']['status'] == 'pass'
    assert gates['frozen_weights']['status'] == 'pass'
    assert gates['save_reload_parity']['status'] == 'pass'
    assert report['promote_v5_memory'] is False
    assert report['neural_promotion'] is False
    assert report['decision'] in {
        'one_write_storage_ok', 'fail_writer_or_gate_normalization',
        'fail_depth_reduction_interference', 'fail_read_path_leakage'}
    assert (tmp_path / 'probe' / 'readouts.jsonl').is_file()
    assert (tmp_path / 'probe' / 'codebooks.json').is_file()
    assert report['readouts_sha256']
    for codebook_key, book in report['codebooks'].items():
        assert list(book['layers']) == [codebook_key.rsplit('layer-', 1)[1]]
        assert book['source_binding']['view_digest']
        assert book['key_cross_cosine'] is not None


def test_scoring_recomputes_cosines_from_raw_rows_and_source_keys(tmp_path):
    model, report, rows = tiny_run(tmp_path)
    codebooks = report['codebooks']
    mechanical = {'frozen': report['frozen']}
    baseline = score_one_write(rows, codebooks, mechanical=mechanical)
    assert baseline['summary'] == report['summary']

    # An exact ideal-vector readout scores 1.0 for that row.
    ideal_rows = copy.deepcopy(rows)
    target = next(row for row in ideal_rows if row['against'] == 'per_depth'
                  and row['phase'] == 'after_save')
    entry = codebooks[target['codebook_key']]['layers'][str(target['layer'])]
    target['readout'] = entry['value_vector']
    target['readout_norm'] = 1.0
    ideal = score_one_write(ideal_rows, codebooks, mechanical=mechanical)
    assert ideal['gates']['per_depth_recovery']['metrics']['max_abs_cosine'] == 1.0

    zeroed = copy.deepcopy(rows)
    for row in zeroed:
        if row['against'] == 'per_depth':
            row['readout'] = [[0.0] * len(entry['value_vector'][0])
                              for _ in entry['value_vector']]
            row['readout_norm'] = 0.0
    rejected = score_one_write(zeroed, codebooks, mechanical=mechanical)
    assert rejected['decision'] == 'fail_writer_or_gate_normalization'

    forged = copy.deepcopy(rows)
    forged[0]['key_sha256'] = '0' * 64
    with pytest.raises(ValueError, match='key differs'):
        score_one_write(forged, codebooks, mechanical=mechanical)
    foreign = copy.deepcopy(rows)
    foreign[0]['world_id'] = 'foreign-world'
    with pytest.raises(ValueError, match='foreign codebook'):
        score_one_write(foreign, codebooks, mechanical=mechanical)


def test_save_reload_parity_detects_changed_after_rows(tmp_path):
    model, report, rows = tiny_run(tmp_path)
    codebooks = report['codebooks']
    mechanical = {'frozen': report['frozen']}
    changed = copy.deepcopy(rows)
    target = next(row for row in changed if row['phase'] == 'after_save'
                  and row['against'] == 'shared_mean')
    target['readout'] = [[value + 1.0 for value in head] for head in target['readout']]
    scored = score_one_write(changed, codebooks, mechanical=mechanical)
    assert scored['gates']['save_reload_parity']['status'] == 'fail'
    assert scored['decision'] == 'fail_save_reload_parity'



def test_save_reload_parity_covers_earlier_facts_and_rejects_bad_rows(tmp_path):
    model, report, rows = tiny_run(tmp_path, facts_per_world=2)
    codebooks = report['codebooks']
    mechanical = {'frozen': report['frozen']}
    first_fact = min(codebooks, key=lambda key: (codebooks[key]['world_id'],
                                                codebooks[key]['fact_id']))
    fact_id = codebooks[first_fact]['fact_id']

    changed = copy.deepcopy(rows)
    target = next(row for row in changed
                  if row['codebook_key'] == first_fact
                  and row['against'] == 'shared_mean'
                  and row['phase'] == 'before_save')
    target['readout'] = [[0.0 for _ in head] for head in target['readout']]
    scored = score_one_write(changed, codebooks, mechanical=mechanical)
    assert scored['decision'] == 'fail_save_reload_parity', fact_id
    parity = report['gates']['save_reload_parity']['metrics']
    assert parity['shared_mean']['n'] == 8

    duplicated = copy.deepcopy(rows)
    duplicated.append(copy.deepcopy(duplicated[0]))
    with pytest.raises(ValueError, match='Duplicate'):
        score_one_write(duplicated, codebooks, mechanical=mechanical)

    missing = copy.deepcopy(rows)
    missing.remove(next(row for row in missing
                        if row['codebook_key'] == first_fact
                        and row['against'] == 'shared_mean'
                        and row['phase'] == 'before_save'))
    with pytest.raises(ValueError, match='Missing'):
        score_one_write(missing, codebooks, mechanical=mechanical)


def test_prepare_and_admit_bind_exact_one_write_pins(tmp_path):
    model_root = tmp_path / 'source'
    model_root.mkdir()
    (model_root / 'config.json').write_text('{}')
    lock = tmp_path / 'lock.json'
    lock.write_text(json.dumps({
        'revision': 'f97400052a43d642bbc6e9975e2397e3ae6a6b52',
        'snapshot_files_sha256': {'config.json': 'a' * 64},
    }))
    prepared = tmp_path / 'prepared'
    manifest = prepare_one_write(prepared, seed=7, world_count=2, layers=(0, 2),
                                 model_path=model_root, model_lock=lock)
    assert manifest['format'] == 'replica.memory_one_write_probe.prepared.v1'
    assert manifest['native_computation'] is False
    receipt = admit_one_write_probe(prepared)
    assert receipt['status'] == 'admitted'
    assert receipt['checks']['seeded_world_regeneration'] is True
    assert (prepared / 'admission.json').is_file()

    data_path = prepared / 'data.json'
    original = data_path.read_bytes()
    tampered = json.loads(original)
    tampered['worlds'][0]['facts'][0]['value'] = 'forged-value'
    data_path.write_text(json.dumps(tampered))
    with pytest.raises(ValueError):
        admit_one_write_probe(prepared, write_receipt=False)
    data_path.write_bytes(original)

    protocol_path = prepared / 'protocol.json'
    protocol_bytes = protocol_path.read_bytes()
    tampered_protocol = json.loads(protocol_bytes)
    tampered_protocol['layers'] = [0]
    protocol_path.write_text(json.dumps(tampered_protocol))
    with pytest.raises(ValueError):
        admit_one_write_probe(prepared, write_receipt=False)
    protocol_path.write_bytes(protocol_bytes)
    assert admit_one_write_probe(prepared, write_receipt=False)['status'] == 'admitted'


def test_interference_two_fact_worlds_read_every_fact_from_final_state(tmp_path):
    model = tiny_adapter(fast_memory=SPEC).eval()
    report = run_one_write_diagnostic(
        model, Tokenizer(), protocol(facts_per_world=2, mode='interference'),
        output_dir=tmp_path / 'interference', ledger_dir=tmp_path / 'ledger',
        device='cpu')
    rows = report.pop('readouts')
    assert report['mode'] == 'interference'
    assert report['counts']['facts'] == 4
    assert report['workload']['planned_rows'] == len(rows) == 64
    assert {row['against'] for row in rows} == {
        'per_depth', 'shared_mean', 'cross_depth', 'zero_state'}
    assert len(report['codebooks']) == 8
    assert report['gates']['save_reload_parity']['status'] == 'pass'
    assert report['gates']['frozen_weights']['status'] == 'pass'
    assert report['optimizer_updates'] == 0
    scored = score_one_write(rows, report['codebooks'],
                             mechanical={'frozen': report['frozen']})
    assert scored['summary'] == report['summary']
    assert report['decision'] in {
        'one_write_storage_ok', 'fail_writer_or_gate_normalization',
        'fail_depth_reduction_interference'}


def test_interference_three_fact_workload_and_preparation(tmp_path):
    model_root = tmp_path / 'source'
    model_root.mkdir()
    (model_root / 'config.json').write_text('{}')
    lock = tmp_path / 'lock.json'
    lock.write_text(json.dumps({
        'revision': 'f97400052a43d642bbc6e9975e2397e3ae6a6b52',
        'snapshot_files_sha256': {'config.json': 'a' * 64}}))
    prepared = tmp_path / 'interference_prep'
    manifest = prepare_one_write(prepared, seed=9, world_count=1, layers=(0, 2),
                                 facts_per_world=3, model_path=model_root,
                                 model_lock=lock)
    assert manifest['facts_per_world'] == 3
    assert manifest['planned_workload']['planned_rows'] == 48
    receipt = admit_one_write_probe(prepared)
    assert receipt['status'] == 'admitted'
    protocol_value = json.loads((prepared / 'protocol.json').read_text())
    assert protocol_value['mode'] == 'interference'


def test_one_write_workload_scales_with_fact_count():
    assert one_write_workload(world_count=2, layers=[0, 2])['planned_rows'] == 32
    assert one_write_workload(world_count=2, layers=[0, 2],
                              facts_per_world=3)['planned_rows'] == 96

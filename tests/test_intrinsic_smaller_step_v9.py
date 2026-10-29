import copy
import json
from pathlib import Path

import pytest
import torch

from replica_lab.intrinsic_memory_smaller_step_v9 import FORMAT, PACKET, TRAINING, validate_protocol
from replica_lab.intrinsic_memory_execution_v9 import validate_bindings, validate_predecessor

ROOT = Path(__file__).resolve().parents[1]


def test_only_learning_rate_and_execution_identity_change():
    old = json.loads((ROOT / 'configs/spark_intrinsic_supported_values_v8.json').read_text())
    new = json.loads((ROOT / 'configs/spark_intrinsic_smaller_step_v9.json').read_text())
    validate_protocol(new)
    validate_predecessor(ROOT, new)
    assert new['training'] == {**old['training'], 'lr': 0.0001}
    assert new['data'] == old['data']
    for arm in new['arms']:
        assert new['arms'][arm]['budget'] == old['arms'][arm]['budget']


@pytest.mark.parametrize('mutation', ['rate', 'updates', 'gate', 'budget'])
def test_smaller_step_does_not_relax_other_limits(mutation):
    p = json.loads((ROOT / 'configs/spark_intrinsic_smaller_step_v9.json').read_text())
    if mutation == 'rate': p['training']['lr'] = 0.001
    elif mutation == 'updates': p['training']['updates'] = 256
    elif mutation == 'gate': p['utility']['false_recall_max'] = 1
    else: p['arms']['two']['budget']['fit_032']['updates'] = 64
    with pytest.raises(ValueError): validate_protocol(p)


def synthetic(tmp_path):
    from test_intrinsic_exposure_v7 import synthetic_v7
    from test_intrinsic_execution_v5 import write
    packet_path = synthetic_v7(tmp_path)
    previous = json.loads((tmp_path / 'protocol.json').read_text())
    previous_pin = write(tmp_path, 'previous.json', previous)
    result = {'protocol_sha256': previous_pin['sha256'], 'source_kind': 'official_bf16',
              'arm': 'two', 'native_competence': True, 'useful_recall': False,
              'main': {'fit_memory_dependence_passed': False, 'fit_updates': 128,
                       'original_weights_unchanged': True},
              'restart': {'fresh_process_logits_exact': True, 'actual_continuation_updates': 1}}
    pin = write(tmp_path, previous['arms']['two']['output_root'] + '/result.json', result)
    p = copy.deepcopy(previous)
    p.update(format=FORMAT, training=copy.deepcopy(TRAINING), diagnostic_predecessor=pin,
             experiment_variant={'predecessor': previous_pin})
    for arm in p['arms']: p['arms'][arm]['output_root'] = 'output_v9/' + arm
    protocol_pin = write(tmp_path, 'protocol.json', p)
    write(tmp_path, 'packet.json', {'format': PACKET, 'protocol': protocol_pin, 'arm': 'two',
                                  'attempts': 1, 'retries': 0, 'output_root': 'output_v9/two'})
    return packet_path, p, previous


@pytest.mark.parametrize('failure', ['passed_fit', 'missing_receipt', 'opened_development', 'changed_data'])
def test_predecessor_and_unopened_labels_are_required(tmp_path, failure):
    from test_intrinsic_execution_v5 import write
    packet, p, previous = synthetic(tmp_path)
    validate_bindings(tmp_path, packet)
    if failure == 'passed_fit':
        f = p['diagnostic_predecessor']['path']
        result = json.loads((tmp_path / f).read_text())
        result['main']['fit_memory_dependence_passed'] = True
        p['diagnostic_predecessor'] = write(tmp_path, f, result)
    elif failure == 'missing_receipt':
        (tmp_path / p['diagnostic_predecessor']['path']).unlink()
    elif failure == 'opened_development':
        write(tmp_path, previous['arms']['two']['output_root'] + '/development.json', {})
    else:
        p['data']['development'] = write(tmp_path, 'different_data.json', {'rows': ['different']})
    with pytest.raises((ValueError, FileNotFoundError)):
        validate_predecessor(tmp_path, p)


@pytest.mark.parametrize('file', ['modeling_spark.py', 'configuration_spark.py'])
@pytest.mark.parametrize('failure', ['changed', 'missing', 'substituted'])
def test_executable_source_refused_before_loading(tmp_path, file, failure):
    packet, _, _ = synthetic(tmp_path)
    validate_bindings(tmp_path, packet)
    f = tmp_path / 'source' / file
    if failure == 'changed': f.write_text(f.read_text() + '\nchanged=True\n')
    elif failure == 'missing': f.unlink()
    else:
        other = tmp_path / 'substitute.py'; other.write_bytes(f.read_bytes())
        f.unlink(); f.symlink_to(other)
    loaded = []
    with pytest.raises(ValueError):
        validate_bindings(tmp_path, packet)
        loaded.append('model load')
    assert not loaded


def test_rate_has_one_optimizer_owner_and_survives_reload(tmp_path):
    from replica_lab.contextual_spark_fixture import build_contextual_adapter
    from replica_lab.architecture.intrinsic_memory_layout_v4 import spark_layout
    from replica_lab.contextual_memory_v5 import ContextualMemoryV5
    from replica_lab.intrinsic_memory_diagnostic_v5 import DiagnosticEngine
    from test_intrinsic_diagnostic_v5 import spec, ledger, FixtureBackend, row, source
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    def make():
        adapter, *_ = build_contextual_adapter(seed=17, layers=36, spine_sites=(3, 7), spine_slots=8, memory_size=256)
        path = ContextualMemoryV5(adapter, spark_layout('two', hidden_size=8),
                                  source_identity='v9_CPU_fixture', objective_identity=spec().identity(),
                                  lr=TRAINING['lr'])
        adapter.extension.spine = None
        return path
    p = make()
    try:
        e = DiagnosticEngine(p, FixtureBackend(), spec(), ledger(tmp_path), max_new_tokens=3, output=tmp_path/'out')
        e.stage = 'test'
        batch, scope, store = source(e, tmp_path, row())
        try:
            first = e.update(row(), batch, scope)
            second = e.update(row(), batch, scope)
            assert first['changed_tensors'] and second['gradients']['producer']['norm'] > 0
            assert all(g['lr'] == 0.0001 for g in p.optimizer.param_groups)
            p.assert_optimizer_ownership()
            p.save(tmp_path/'checkpoint')
            restored = make()
            try:
                restored.load(tmp_path/'checkpoint')
                restored.assert_optimizer_ownership()
                assert all(g['lr'] == 0.0001 for g in restored.optimizer.param_groups)
                assert all(torch.equal(v, restored._tensors()[k]) for k, v in p._tensors().items())
                assert all(not x.requires_grad for x in restored.adapter.base.parameters())
            finally: restored.close()
        finally: store.close()
    finally:
        p.close()
        torch.set_num_threads(previous_threads)

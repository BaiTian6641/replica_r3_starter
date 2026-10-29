import copy
import json
from pathlib import Path

import pytest

from replica_lab.intrinsic_memory_alignment_v10 import FORMAT, PACKET, AUTHORIZATION, TEACHER, plan, totals, validate_protocol
from replica_lab.intrinsic_memory_execution_v10 import validate_bindings, validate_predecessor, validate_authorization
from replica_lab.intrinsic_memory_execution_v5 import hash_file

ROOT = Path(__file__).resolve().parents[1]


def test_v10_matches_completed_direct_CE_reference_and_accounts_all_teacher_work():
    p = json.loads((ROOT/'configs/spark_intrinsic_alignment_v10.json').read_text())
    old = json.loads((ROOT/'configs/spark_intrinsic_smaller_step_v9.json').read_text())
    validate_protocol(p)
    validate_predecessor(ROOT, p)
    assert p['data'] == old['data'] and p['training'] == old['training']
    for arm, spec in p['arms'].items():
        counts = spec['total_reservations']
        assert counts['forwards'] == 2469 and counts['generated_tokens'] == 2072
        assert counts['input_tokens'] == 632064 and counts['writes'] == 112
        assert counts['updates'] == counts['backwards'] == 129
        assert counts['source_encodes'] == 245 and counts['retrievals'] == 51
        assert counts['model_loads'] == 2 and counts['wall_seconds'] == 7080
        assert counts['artifact_bytes'] == old['arms'][arm]['total_reservations']['artifact_bytes']
        assert spec['wall_seconds'] == 7200


@pytest.mark.parametrize('damage', ['top_k', 'CE_mix', 'teacher_gate', 'utility', 'student_updates', 'forwards', 'cache_bytes', 'wall'])
def test_objective_gates_and_new_work_budget_are_fixed(damage):
    p = json.loads((ROOT/'configs/spark_intrinsic_alignment_v10.json').read_text())
    if damage == 'top_k': p['teacher']['top_k'] = 16
    elif damage == 'CE_mix': p['teacher']['ce_weight'] = 1
    elif damage == 'teacher_gate': p['teacher']['qualification']['correct_min'] = 7
    elif damage == 'utility': p['utility']['contextual_correct_min'] = 4
    elif damage == 'student_updates': p['training']['updates'] = 256
    elif damage == 'forwards': p['arms']['two']['budget']['teacher']['forwards'] = 64
    elif damage == 'cache_bytes': p['arms']['two']['budget']['teacher']['artifact_bytes'] = 0
    else: p['arms']['two']['wall_seconds'] = 7400
    with pytest.raises(ValueError): validate_protocol(p)


def synthetic(tmp_path):
    from test_intrinsic_smaller_step_v9 import synthetic as previous_fixture
    from test_intrinsic_execution_v5 import write
    packet, previous, _ = previous_fixture(tmp_path)
    previous_pin = write(tmp_path, 'direct_CE_reference.json', previous)
    result = {'protocol_sha256': previous_pin['sha256'], 'source_kind': 'official_bf16',
              'arm': 'two', 'native_competence': True, 'useful_recall': False,
              'main': {'fit_memory_dependence_passed': False, 'fit_updates': 128, 'original_weights_unchanged': True},
              'restart': {'fresh_process_logits_exact': True, 'actual_continuation_updates': 1}}
    result_pin = write(tmp_path, previous['arms']['two']['output_root']+'/result.json', result)
    p = copy.deepcopy(previous)
    p.update(format=FORMAT, teacher=copy.deepcopy(TEACHER), diagnostic_predecessor=result_pin,
             experiment_variant={'predecessor': previous_pin})
    for arm, spec in p['arms'].items():
        spec.update(output_root='output_v10/'+arm, budget=plan(arm), total_reservations=totals(plan(arm)))
    pp = write(tmp_path, 'protocol.json', p)
    write(tmp_path, 'packet.json', {'format': PACKET, 'protocol': pp, 'arm': 'two',
                                   'attempts': 1, 'retries': 0, 'output_root': 'output_v10/two'})
    return packet, p, previous


@pytest.mark.parametrize('file', ['modeling_spark.py', 'configuration_spark.py'])
@pytest.mark.parametrize('damage', ['changed', 'missing', 'substituted'])
def test_actual_executable_source_refused_before_model_loading(tmp_path, file, damage):
    packet, p, previous = synthetic(tmp_path)
    validate_bindings(tmp_path, packet)
    f = tmp_path/'source'/file
    if damage == 'changed': f.write_text(f.read_text()+'\nchanged=True\n')
    elif damage == 'missing': f.unlink()
    else:
        substitute = tmp_path/'substitute.py';substitute.write_bytes(f.read_bytes())
        f.unlink();f.symlink_to(substitute)
    loaded = []
    with pytest.raises(ValueError):
        validate_bindings(tmp_path, packet)
        loaded.append('model_loaded')
    assert not loaded


@pytest.mark.parametrize('damage', ['opened_development', 'changed_data', 'changed_rate', 'unverified_continuation', 'missing_reference'])
def test_unopened_matched_data_and_verified_CE_reference_are_required(tmp_path, damage):
    from test_intrinsic_execution_v5 import write
    packet, p, previous = synthetic(tmp_path)
    validate_bindings(tmp_path, packet)
    if damage == 'opened_development': write(tmp_path, previous['arms']['two']['output_root']+'/development.json', {})
    elif damage == 'changed_data': p['data']['development'] = write(tmp_path, 'changed.json', {})
    elif damage == 'changed_rate': p['training']['lr'] = 0.001
    elif damage == 'unverified_continuation':
        name = p['diagnostic_predecessor']['path'];r = json.loads((tmp_path/name).read_text())
        r['restart']['actual_continuation_updates'] = 0
        p['diagnostic_predecessor'] = write(tmp_path, name, r)
    else: (tmp_path/p['diagnostic_predecessor']['path']).unlink()
    with pytest.raises((ValueError, FileNotFoundError)): validate_predecessor(tmp_path, p)


def authorization(tmp_path, packet, p):
    from test_intrinsic_execution_v5 import write
    obj = json.loads(packet.read_text())
    auth = {'format': AUTHORIZATION, 'decision': 'authorized', 'packet_sha256': hash_file(packet),
            'protocol_sha256': obj['protocol']['sha256'], 'arm': obj['arm'], 'source_kind': 'official_bf16',
            'output_root': obj['output_root'], 'attempts': 1, 'retries': 0,
            'budget': p['arms'][obj['arm']]['total_reservations'],
            'implementation_sha256': p['implementation']['sha256'], 'environment_sha256': p['environment']['sha256'],
            'human_approval_reference': 'synthetic unit fixture only', 'human_approval_text': 'synthetic unit fixture only'}
    write(tmp_path, 'auth.json', auth)
    return tmp_path/'auth.json'


def test_consumed_output_and_absent_density_prerequisite_cannot_launch(tmp_path):
    from test_intrinsic_execution_v5 import write
    packet, p, _ = synthetic(tmp_path)
    auth = authorization(tmp_path, packet, p)
    validate_authorization(tmp_path, packet, auth)
    (tmp_path/'output_v10/two').mkdir(parents=True)
    with pytest.raises(ValueError, match='consumed'): validate_authorization(tmp_path, packet, auth)
    obj = json.loads(packet.read_text());obj.update(arm='all', output_root='output_v10/all')
    write(tmp_path, 'packet.json', obj)
    auth = authorization(tmp_path, packet, p)
    with pytest.raises(ValueError, match='useful two-site'): validate_authorization(tmp_path, packet, auth)


@pytest.mark.parametrize('failed_gate', ['baseline', 'teacher'])
def test_native_failed_gate_never_enters_fitting_or_opens_development(tmp_path, monkeypatch, failed_gate):
    import torch
    from replica_lab import intrinsic_memory_native_v10 as native
    from replica_lab.intrinsic_memory_budget_v5 import WorkLedger
    from test_intrinsic_teacher_v10 import make, rows
    from test_tiered_memory import open_store
    import time
    e = make(tmp_path/'fixture')
    p = json.loads((ROOT/'configs/spark_intrinsic_alignment_v10.json').read_text())
    opened = []
    def data(root, pin):
        role = next(k for k, value in p['data'].items() if value == pin)
        assert role != 'development'
        opened.append(role)
        return {'rows': rows()}
    monkeypatch.setattr(native, 'bound_json', data)
    import replica_lab.intrinsic_memory_native_v5 as shared
    monkeypatch.setattr(shared, 'store_at', lambda output, world: open_store(output/'stores', name=world))
    def teacher(self, *args):
        assert failed_gate == 'teacher', 'teacher opened after failed baseline'
        return False
    monkeypatch.setattr(native.TeacherEngine, 'prepare_teacher', teacher)
    monkeypatch.setattr(native.TeacherEngine, 'update', lambda *args: pytest.fail('fitting opened after failed teacher'))
    def calibration(self, row, *args, **kwargs):
        return {'raw_answer': row['answer'], 'correct': failed_gate != 'baseline', 'false_recall': False,
                'truncated': False, 'first_logits': None}
    monkeypatch.setattr(native.TeacherEngine, 'generate', calibration)
    output = tmp_path/'native_controller';output.mkdir()
    ledger = WorkLedger.create(output/'ledger.json', identity='CPU_teacher_failure',
        plan=p['arms']['two']['budget'], deadline=time.monotonic()+120)
    result = native.run_main(tmp_path, p, 'two', ledger, output, e.path, e.backend, e.spec, e.base_identity)
    assert result['competence_passed'] == (failed_gate != 'baseline')
    assert not result['teacher_competence_passed']
    assert result['status'] == failed_gate+'_failed_dependent_stages_closed'
    assert not result['restart_requested']
    assert opened == (['calibration', 'fit'] if failed_gate == 'teacher' else ['calibration'])
    assert ledger.snapshot()['reserved']['updates'] == 0
    assert not (output/'development.json').exists()

import copy
import json
from pathlib import Path
import time

import pytest
import torch

from replica_lab.intrinsic_memory_exposure_v7 import FORMAT, PREFIXES, TRAINING, plan, totals, v5_plan, validate_protocol
from replica_lab.intrinsic_memory_budget_v5 import WorkLedger

ROOT = Path(__file__).resolve().parents[1]


def protocol():
    p = json.loads((ROOT / 'configs/spark_intrinsic_explicit_evidence_v6.json').read_text())
    p.update(format=FORMAT, training=copy.deepcopy(TRAINING), fit_probe_prefixes=list(PREFIXES),
             fit_curve_selection='all_128_updates_fixed_final_prefix_only_opens_development')
    for arm, spec in p['arms'].items():
        spec['budget'] = plan(arm)
        spec['total_reservations'] = totals(spec['budget'])
    return p


def test_complete_curve_budget_and_shared_contracts_match_sealed_control():
    old = json.loads((ROOT / 'configs/spark_intrinsic_diagnostic_v5.json').read_text())
    p = protocol()
    validate_protocol(p)
    for arm, spec in p['arms'].items():
        assert v5_plan(arm) == old['arms'][arm]['budget']
        total = spec['total_reservations']
        assert total['forwards'] == 2397 and total['updates'] == total['backwards'] == 129
        assert total['generated_tokens'] == 2008 and total['input_tokens'] == 613632
        assert total['source_encodes'] == 245 and total['retrievals'] == 51 and total['writes'] == 103
        assert total['model_loads'] == 2 and total['wall_seconds'] == 7080


@pytest.mark.parametrize('mutation', ['criterion', 'source', 'prefix', 'updates', 'budget', 'early_stop'])
def test_exposure_change_does_not_relax_other_gates(mutation):
    p = protocol()
    if mutation == 'criterion':
        p['competence']['correct_min'] = 6
    elif mutation == 'source':
        p['source_kind'] = 'dequantized_fp8'
    elif mutation == 'prefix':
        p['fit_probe_prefixes'].append(160)
    elif mutation == 'updates':
        p['training']['updates'] = 256
    elif mutation == 'budget':
        p['arms']['two']['budget']['fit_probes_064']['forwards'] = 0
    else:
        p['fit_curve_selection'] = 'best_observed_prefix'
    with pytest.raises(ValueError):
        validate_protocol(p)


def test_complete_cpu_curve_rebuilds_after_learning_without_opening_development(tmp_path, monkeypatch):
    from replica_lab import intrinsic_memory_native_v7 as native
    from replica_lab import intrinsic_memory_native_v5 as shared
    from replica_lab.intrinsic_memory_diagnostic_v5 import DiagnosticEngine
    from test_intrinsic_diagnostic_v5 import make, row, spec, FixtureBackend
    from test_tiered_memory import open_store
    torch.set_num_threads(1)
    p = protocol()
    p['generation']['max_new_tokens'] = 3
    p['data'] = {k: {'stage': k} for k in ('calibration', 'fit', 'development')}
    rows = [row('red' if i % 2 == 0 else 'blue', i) for i in range(8)]
    opened = []
    def data(root, pin):
        opened.append(pin['stage'])
        assert pin['stage'] != 'development', 'dependent labels opened'
        return {'rows': rows}
    monkeypatch.setattr(native, 'bound_json', data)
    monkeypatch.setattr(shared, 'store_at', lambda output, world: open_store(output / 'stores', name=world))
    original_generate = DiagnosticEngine.generate
    def generate(self, r, *args, **kwargs):
        if self.stage == 'baseline':
            return {'raw_answer': r['answer'], 'correct': True, 'false_recall': False,
                    'truncated': False, 'first_logits': None}
        return original_generate(self, r, *args, **kwargs)
    monkeypatch.setattr(DiagnosticEngine, 'generate', generate)
    original_probes = DiagnosticEngine.fitting_probes
    calls = []
    def probes(self, *args):
        observed = original_probes(self, *args)
        calls.append(self.stage)
        # Gate fixture only. Keep native semantic claims separate while testing
        # four real producer-release rebuilds and all continuation update paths.
        observed['fit_memory_dependence_passed'] = False
        return observed
    monkeypatch.setattr(DiagnosticEngine, 'fitting_probes', probes)
    ledger = WorkLedger.create(tmp_path / 'ledger.json', identity='CPU_v7_curve',
                               plan=p['arms']['two']['budget'], deadline=time.monotonic() + 180)
    path = make(objective=spec().identity())
    out = tmp_path / 'output'
    out.mkdir()
    result = native.run_main(tmp_path, p, 'two', ledger, out, path, FixtureBackend(), spec(),
                             shared.base_hash(path.adapter.base))
    assert result['fit_updates'] == 128 and result['restart_requested']
    assert result['original_weights_unchanged'] and not result['useful_recall_passed']
    assert opened == ['calibration', 'fit']
    assert calls == [f'fit_probes_{end:03d}' for end in PREFIXES]
    assert ledger.snapshot()['reserved']['updates'] == 128
    assert json.loads((out / 'fitting_closed.json').read_text())['updates'] == 128
    assert not (out / 'development.json').exists()
    assert all((out / f'fit_probes_{end:03d}.json').is_file() for end in PREFIXES)


def synthetic_v7(tmp_path):
    from test_intrinsic_execution_v5 import synthetic, write
    from replica_lab.intrinsic_memory_exposure_v7 import PACKET
    packet_path = synthetic(tmp_path)
    p = protocol()
    fixture = json.loads((tmp_path / 'protocol.json').read_text())
    for key in ('source', 'source_execution', 'implementation', 'environment', 'data'):
        p[key] = fixture[key]
    pin = write(tmp_path, 'protocol.json', p)
    packet = {'format': PACKET, 'protocol': pin, 'arm': 'two', 'attempts': 1,
              'retries': 0, 'output_root': p['arms']['two']['output_root']}
    write(tmp_path, 'packet.json', packet)
    return packet_path


@pytest.mark.parametrize('file', ['modeling_spark.py', 'configuration_spark.py'])
@pytest.mark.parametrize('failure', ['changed', 'missing', 'substituted'])
def test_v7_keeps_executable_source_refusal_before_model_loading(tmp_path, file, failure):
    from replica_lab.intrinsic_memory_execution_v7 import validate_bindings
    packet = synthetic_v7(tmp_path)
    validate_bindings(tmp_path, packet)
    path = tmp_path / 'source' / file
    if failure == 'changed':
        path.write_text(path.read_text() + '\nchanged=True\n')
    elif failure == 'missing':
        path.unlink()
    else:
        other = tmp_path / 'substitute.py'
        other.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(other)
    loaded = []
    with pytest.raises(ValueError):
        validate_bindings(tmp_path, packet)
        loaded.append('model load')
    assert not loaded


def test_v7_authorization_binds_added_work_and_refuses_consumed_output(tmp_path):
    from replica_lab.intrinsic_memory_execution_v7 import validate_bindings, validate_authorization, hash_file
    from replica_lab.intrinsic_memory_exposure_v7 import AUTHORIZATION
    from test_intrinsic_execution_v5 import write
    packet = synthetic_v7(tmp_path)
    item, p, _ = validate_bindings(tmp_path, packet)
    receipt = {'format': AUTHORIZATION, 'decision': 'authorized', 'packet_sha256': hash_file(packet),
               'protocol_sha256': item['protocol']['sha256'], 'arm': 'two', 'source_kind': 'official_bf16',
               'output_root': item['output_root'], 'attempts': 1, 'retries': 0,
               'budget': p['arms']['two']['total_reservations'],
               'implementation_sha256': p['implementation']['sha256'],
               'environment_sha256': p['environment']['sha256'],
               'human_approval_reference': 'synthetic_fixture_only', 'human_approval_text': 'not a real authorization'}
    write(tmp_path, 'auth.json', receipt)
    validate_authorization(tmp_path, packet, tmp_path / 'auth.json')
    wrong = {**receipt, 'budget': totals(v5_plan('two'))}
    write(tmp_path, 'old_budget.json', wrong)
    with pytest.raises(ValueError):
        validate_authorization(tmp_path, packet, tmp_path / 'old_budget.json')
    (tmp_path / item['output_root']).mkdir(parents=True)
    with pytest.raises(ValueError, match='consumed'):
        validate_authorization(tmp_path, packet, tmp_path / 'auth.json')


def test_failed_v7_competence_keeps_every_fitting_prefix_closed(tmp_path, monkeypatch):
    from replica_lab import intrinsic_memory_native_v7 as native
    from test_intrinsic_diagnostic_v5 import make, row, spec, FixtureBackend
    torch.set_num_threads(1)
    path = make(objective=spec().identity())
    p = protocol()
    p['generation']['max_new_tokens'] = 3
    p['data'] = {k: {'stage': k} for k in ('calibration', 'fit', 'development')}
    opened = []
    def data(root, pin):
        opened.append(pin['stage'])
        assert pin['stage'] == 'calibration'
        return {'rows': [{**row(i=i), 'answer': 'never_emitted'} for i in range(8)]}
    monkeypatch.setattr(native, 'bound_json', data)
    ledger = WorkLedger.create(tmp_path / 'ledger.json', identity='CPU_failed_baseline',
                               plan=p['arms']['two']['budget'], deadline=time.monotonic() + 60)
    out = tmp_path / 'output'
    out.mkdir()
    result = native.run_main(tmp_path, p, 'two', ledger, out, path, FixtureBackend(), spec(),
                             native.base_hash(path.adapter.base))
    assert not result['competence_passed'] and opened == ['calibration']
    assert ledger.snapshot()['reserved']['updates'] == 0
    assert not (out / 'fit_curve.json').exists()

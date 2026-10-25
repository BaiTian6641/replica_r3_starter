"""Versioned one-fact / one-write storage discriminator.

This module isolates the smallest storage question left open by the multi-fact
probe: with a single observed fact written once, does the exact observed key
recover the exact written value vector, and if not, is the failure in the
writer/gate path or in the shared-depth convex-mean reduction?

It compares four read paths against source-derived value vectors:

* ``per_depth``  — layer L's own key against layer L's individual proposal state;
* ``shared_mean`` — the production shared-depth reduction (convex mean of every
  layer's proposal), read with layer L's key;
* ``cross_depth`` — layer L's key against the *other* layer's proposal state
  (report-only control that exposes key divergence between depth writers);
* ``zero_state`` — layer L's key against the never-updated state (must be zero).

Everything runs frozen: no optimizer is constructed, no parameter may change.
Rows are preserved before and after an image save/load, and ``score_one_write``
recomputes every cosine and gate from raw readouts and codebooks rather than
trusting producer flags.
"""
from __future__ import annotations

import json
from pathlib import Path
from dataclasses import replace
import time

import torch
import torch.nn.functional as F

from .architecture.contracts import FastMemorySpec
from .memory_system import MemoryRecord, RecordLedger
from .runtime.exact_memory import ExactMemoryReference
from .runtime.fact_memory import fact_source_text
from .runtime.fast_memory import FastMemoryScope
from .runtime.memory_representation import FACT_REPRESENTATION
from .runtime.persistent_fast_memory import PersistentFastMemory
from .memory_diagnosis import (
    DEFAULT_LIMITS, MODE, PROJECT_ROOT, PURPOSE, SUBJECT_ID,
    _cosine_heads, _display_path, _extension_pins, _fresh_token,
    _implementation_sha256, _mps_memory, _opaque_token, _parameter_versions,
    _rss_bytes, _scope, _sha_file, _sha_text, _source_pins, _utc_now,
    _write_json, _write_jsonl, observed_binding,
)

ONE_WRITE_FORMAT = 'replica.memory_one_write_probe.protocol.v1'
ONE_WRITE_WORLD_FORMAT = 'replica.memory_one_write_probe.worlds.v1'
ONE_WRITE_PREPARED_FORMAT = 'replica.memory_one_write_probe.prepared.v1'
ONE_WRITE_ADMISSION_FORMAT = 'replica.memory_one_write_probe.admission.v1'
ONE_WRITE_RUN_FORMAT = 'replica.memory_one_write_probe.run.v1'
ONE_WRITE_THRESHOLDS = {
    'one_write_recovery_floor': 0.6,
    'one_write_reduction_floor': 0.6,
    'one_write_zero_max_abs_cosine': 0.1,
    'one_write_parity_atol': 1e-6,
}
ONE_WRITE_READ_PATHS = ('per_depth', 'shared_mean', 'cross_depth', 'zero_state')


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _freeze(model):
    for parameter in model.parameters():
        parameter.requires_grad_(False)


def generate_one_write_worlds(*, seed=20261025, world_count=8, facts_per_world=1):
    """Deterministic worlds with globally disjoint opaque keys and values."""
    _require(type(seed) is int and seed >= 0, 'Seed must be a nonnegative integer')
    _require(type(world_count) is int
             and 1 <= world_count <= DEFAULT_LIMITS['max_worlds'],
             'World count exceeds the diagnostic limit')
    _require(type(facts_per_world) is int and facts_per_world >= 1,
             'Each world requires at least one observed fact')
    used_keys, used_values, worlds = set(), set(), []
    for index in range(world_count):
        material = f'{seed}:onewrite:{index}'
        world_id = f'onewrite-{_opaque_token("w", material)}'
        facts = []
        for fact_index in range(facts_per_world):
            fact_material = f'{material}:fact:{fact_index}'
            entity = _fresh_token('ent', f'{fact_material}:entity', used_keys)
            attribute = _fresh_token('attr', f'{fact_material}:attribute', used_keys)
            value = _fresh_token('val', f'{fact_material}:value', used_values)
            facts.append({'fact_id': f'{world_id}:fact:{fact_index}', 'entity': entity,
                          'attribute': attribute, 'value': value,
                          'source_world_id': world_id})
        worlds.append({
            'world_id': world_id, 'family_id': f'onewrite-family-{index}',
            'facts': facts, 'distractors': [], 'correction': None,
            'revocation_fact_id': None,
        })
    return worlds


def validate_one_write_worlds(worlds):
    """Reject extra labels, duplicate addresses/values and malformed worlds."""
    _require(isinstance(worlds, list) and bool(worlds), 'Worlds must be a nonempty list')
    world_ids, fact_ids, keys, values = set(), set(), set(), set()
    for world in worlds:
        _require(isinstance(world, dict) and set(world) == {
            'world_id', 'family_id', 'facts', 'distractors', 'correction',
            'revocation_fact_id'}, 'One-write world fields differ')
        world_id = world['world_id']
        _require(world_id not in world_ids, 'Duplicate one-write world')
        world_ids.add(world_id)
        _require(world['distractors'] == [] and world['correction'] is None
                 and world['revocation_fact_id'] is None,
                 'One-write worlds carry no distractors/correction/revocation')
        facts = world['facts']
        _require(isinstance(facts, list) and facts,
                 'Each world requires at least one observed fact')
        for fact in facts:
            _require(isinstance(fact, dict) and set(fact) == {
                'fact_id', 'entity', 'attribute', 'value', 'source_world_id'},
                'One-write fact fields differ')
            _require(fact['fact_id'] not in fact_ids, 'Duplicate one-write fact')
            fact_ids.add(fact['fact_id'])
            key = (fact['entity'], fact['attribute'])
            _require(key not in keys, 'One-write addresses must be disjoint')
            keys.add(key)
            _require(fact['value'] not in values, 'One-write values must be disjoint')
            values.add(fact['value'])
    return True


def one_write_records(world):
    """Render the world's immutable source records in declared order."""
    records = []
    for index, fact in enumerate(world['facts']):
        records.append(MemoryRecord(
            record_id=fact['fact_id'], subject_id=SUBJECT_ID,
            branch_id=world['world_id'],
            text=fact_source_text({name: fact[name]
                                   for name in ('entity', 'attribute', 'value')}),
            observed_at=1, event_at=index + 1, purpose=PURPOSE, kind='source_fact',
            source_id=f"{world['world_id']}:source:{index}",
            authenticity='verified_source'))
    return records


def one_write_workload(*, world_count, layers, facts_per_world=1):
    _require(len(set(layers)) == len(layers) and len(layers) >= 2,
             'One-write diagnosis needs at least two distinct depth layers')
    rows_per_world = len(ONE_WRITE_READ_PATHS) * len(layers) * 2 * facts_per_world
    return {
        'world_count': world_count,
        'layers': list(layers),
        'facts_per_world': facts_per_world,
        'evaluations_per_state': 2,
        'read_paths': list(ONE_WRITE_READ_PATHS),
        'rows_per_world': rows_per_world,
        'planned_rows': rows_per_world * world_count,
        'planned_cases': rows_per_world * world_count // 2,
    }


def _read_weights(weights, query):
    normalized = F.normalize(query.to(device=weights.device, dtype=weights.dtype),
                             dim=-1, eps=1e-6)
    return torch.einsum('bthk,bhkv->bthv', normalized, weights)[0, 0]


def _row(world_id, layer, against, phase, readout, entry, codebook_key):
    return {
        'world_id': world_id, 'layer': layer, 'against': against, 'phase': phase,
        'status': 'ok', 'readout': readout.detach().cpu().tolist(),
        'readout_norm': float(torch.linalg.vector_norm(readout.detach().float())),
        'readout_sha256': _sha_text(json.dumps(readout.detach().cpu().tolist(),
                                               allow_nan=False)),
        'key_sha256': entry['key_sha256'], 'value_sha256': entry['value_sha256'],
        'value_norm': entry['value_norm'],
        'gate_mean': entry['gate_mean'], 'erase_mean': entry['erase_mean'],
        'decay_mean': entry['decay_mean'], 'key_norm': entry['key_norm'],
        'codebook_key': codebook_key,
    }


def run_one_write_diagnostic(model, tokenizer, protocol, *, output_dir,
                             ledger_dir=None, worlds=None, device='cpu',
                             writer_extension_receipt=None):
    """Frozen one-fact/one-write probe: per-depth vs shared reduction, zero/ideal controls."""
    _require(protocol.get('format') == ONE_WRITE_FORMAT,
             'One-write diagnostic requires its versioned protocol format')
    supplied = worlds if worlds is not None else protocol.get('worlds')
    facts_per_world = protocol.get('facts_per_world', 1)
    worlds = supplied if supplied is not None else generate_one_write_worlds(
        seed=protocol['seed'], world_count=protocol['world_count'],
        facts_per_world=facts_per_world)
    validate_one_write_worlds(worlds)
    _require(len(worlds) == protocol['world_count'],
             'Protocol world count differs from supplied one-write worlds')
    layers = list(protocol['layers'])
    facts_per_world = protocol.get('facts_per_world', 1)
    _require(all(len(world['facts']) == facts_per_world for world in worlds),
             'Supplied worlds do not match the declared fact count')
    available = tuple(model.ports.injection_layers)
    _require(all(layer in available for layer in layers),
             f'Declared layers {layers} are not injection layers {available}')
    planned = one_write_workload(world_count=len(worlds), layers=layers,
                                 facts_per_world=facts_per_world)
    _require(planned['planned_rows'] <= protocol['limits']['max_queries'],
             'Scheduled one-write workload exceeds max_queries')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ledger_dir = Path(ledger_dir) if ledger_dir is not None else output_dir / 'ledger'
    ledger_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / 'images'
    images_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / 'readouts.jsonl').exists() or (output_dir / 'codebooks.json').exists():
        raise FileExistsError('Diagnostic raw artifacts already exist')

    model.eval()
    requires_grad_before = {name: parameter.requires_grad
                            for name, parameter in model.named_parameters()}
    _freeze(model)
    versions_before = _parameter_versions(model)
    started = time.perf_counter()
    rss_start = _rss_bytes()
    rows, codebooks = [], {}
    other_layer = {layers[index]: layers[(index + 1) % len(layers)]
                   for index in range(len(layers))}

    with torch.no_grad():
        for world in worlds:
            world_id = world['world_id']
            records = one_write_records(world)
            ledger = RecordLedger(ledger_dir / f'{world_id}.sqlite')
            reference = ExactMemoryReference(ledger)
            try:
                _require(reference.admit(records) == len(records),
                         'One-write source admission must accept every record')
                fast_scope = FastMemoryScope(SUBJECT_ID, world_id, PURPOSE, MODE, 0)
                memory = PersistentFastMemory(
                    model, tokenizer, fast_scope, training=False,
                    representation=FACT_REPRESENTATION)
                for record in records:
                    memory.append_record(record)
                scope = _scope(world, 1)
                binding = memory.bind_source(reference, scope=scope)
                image = images_dir / world_id
                memory.save(image, require_source_binding=True)
                loaded = PersistentFastMemory.load(
                    image, model, tokenizer, fast_scope,
                    representation=FACT_REPRESENTATION,
                    expected_view_digest=binding['view_digest'],
                    require_source_binding=True, source_reference=reference,
                    source_scope=scope)
                primitive = memory.primitive
                zero = primitive.initial_state(1, device=memory.device)
                proposals = {layer: zero for layer in layers}
                shared = zero
                fact_writes, fact_entries, fact_cosines = {}, {}, {}
                for fact in world['facts']:
                    fact_id = fact['fact_id']
                    writes = {layer: observed_binding(model, tokenizer, fact,
                                                      layer=layer)
                              for layer in layers}
                    fact_writes[fact_id] = writes
                    for layer in layers:
                        proposals[layer] = primitive.update_shared_chunk(
                            proposals[layer], (replace(
                                writes[layer],
                                chunk_index=proposals[layer].chunk_index),))
                    shared = primitive.reduce_shared_chunk(
                        shared, {layer: (replace(
                            writes[layer], chunk_index=shared.chunk_index),)
                            for layer in layers})
                    entries = {}
                    for layer, write in writes.items():
                        value_vector = write.values[0, 0].detach().cpu()
                        key_tensor = write.keys[0, 0].detach().cpu()
                        entries[str(layer)] = {
                            'value_vector': value_vector.tolist(),
                            'value_sha256': _sha_text(json.dumps(
                                value_vector.tolist(), allow_nan=False)),
                            'value_norm': float(torch.linalg.vector_norm(
                                value_vector.float())),
                            'key_sha256': _sha_text(json.dumps(
                                key_tensor.tolist(), allow_nan=False)),
                            'key_norm': float(torch.linalg.vector_norm(key_tensor.float())),
                            'gate_mean': float(write.write_gate.float().mean()),
                            'erase_mean': float(write.erase.float().mean()),
                            'decay_mean': float(write.decay.float().mean()),
                        }
                    fact_entries[fact_id] = entries
                    key_cosines, value_cosines = [], []
                    for left_index, left in enumerate(layers):
                        for right in layers[left_index + 1:]:
                            key_cosines.append(_cosine_heads(
                                writes[left].keys[0, 0].cpu(),
                                writes[right].keys[0, 0].cpu()))
                            value_cosines.append(_cosine_heads(
                                writes[left].values[0, 0].cpu(),
                                writes[right].values[0, 0].cpu()))
                    fact_cosines[fact_id] = (key_cosines, value_cosines)
                for fact in world['facts']:
                    fact_id = fact['fact_id']
                    writes = fact_writes[fact_id]
                    entries = fact_entries[fact_id]
                    queries = {layer: writes[layer].keys for layer in layers}
                    shared_before = {layer: _read_weights(shared.weights, queries[layer])
                                     for layer in layers}
                    shared_after = {layer: _read_weights(loaded.read().weights,
                                                         queries[layer])
                                    for layer in layers}
                    depth_states = {layer: _read_weights(proposals[layer].weights,
                                                         queries[layer])
                                    for layer in layers}
                    cross_states = {layer: _read_weights(
                        proposals[other_layer[layer]].weights, queries[layer])
                        for layer in layers}
                    zero_states = {layer: _read_weights(zero.weights, queries[layer])
                                   for layer in layers}
                    key_cosines, value_cosines = fact_cosines[fact_id]
                    for phase, shared_reads in (('before_save', shared_before),
                                                ('after_save', shared_after)):
                        for layer in layers:
                            codebook_key = f'{world_id}/{fact_id}/layer-{layer}'
                            entry = entries[str(layer)]
                            for against, readout in (
                                    ('per_depth', depth_states[layer]),
                                    ('shared_mean', shared_reads[layer]),
                                    ('cross_depth', cross_states[layer]),
                                    ('zero_state', zero_states[layer])):
                                rows.append(_row(world_id, layer, against, phase,
                                                 readout, entry, codebook_key))
                    for layer in layers:
                        codebooks[f'{world_id}/{fact_id}/layer-{layer}'] = {
                            'format': 'replica.memory_one_write_probe.codebook.v1',
                            'world_id': world_id, 'fact_id': fact_id,
                            'layers': {str(layer): entries[str(layer)]},
                            'key_cross_cosine': (sum(key_cosines) / len(key_cosines)
                                                 if key_cosines else None),
                            'value_cross_cosine': (sum(value_cosines) / len(value_cosines)
                                                   if value_cosines else None),
                            'source_binding': binding,
                            'image': _display_path(image),
                            'state_sha256': _sha_text(json.dumps(
                                shared.weights.detach().cpu().numpy().tobytes().hex())),
                        }
                _require(torch.equal(memory.read().weights, shared.weights),
                         'Production shared reduction differs from the declared mean')
                _require(torch.equal(memory.read().weights, loaded.read().weights),
                         'One-write save/load changed the production state')
            finally:
                reference.close()

    _write_jsonl(output_dir / 'readouts.jsonl', rows)
    _write_json(output_dir / 'codebooks.json',
                {'format': 'replica.memory_one_write_probe.codebooks.v1',
                 'codebooks': codebooks})
    mechanical = {
        'frozen': {
            'optimizer_updates': 0, 'parameters_updated': [],
            'parameter_versions_equal': _parameter_versions(model) == versions_before,
            'requires_grad_all_false': not any(
                parameter.requires_grad for parameter in model.parameters()),
            'requires_grad_before_all_false': not any(requires_grad_before.values()),
        },
    }
    score = score_one_write(rows, codebooks,
                            thresholds=protocol.get('thresholds'), mechanical=mechanical)
    elapsed = time.perf_counter() - started
    return {
        'format': ONE_WRITE_RUN_FORMAT,
        'mode': protocol.get('mode', 'onewrite'),
        'device': device,
        'seed': protocol['seed'],
        'layer': None,
        'layers': layers,
        'representation': FACT_REPRESENTATION,
        'counts': {
            'worlds': len(worlds), 'facts': sum(len(world['facts']) for world in worlds),
            'evaluated_rows': len(rows), 'unique_cases': len(rows) // 2,
            'before_save_rows': sum(1 for row in rows if row['phase'] == 'before_save'),
            'after_save_rows': sum(1 for row in rows if row['phase'] == 'after_save'),
        },
        'workload': planned,
        'readouts': rows,
        'readouts_sha256': _sha_file(output_dir / 'readouts.jsonl'),
        'codebooks': codebooks,
        'codebooks_sha256': _sha_file(output_dir / 'codebooks.json'),
        'optimizer_updates': 0,
        'parameters_updated': [],
        'frozen': mechanical['frozen'],
        'promote_v5_memory': False,
        'v5_status': 'rejected_by_prior_evidence',
        'neural_promotion': False,
        'pointer_copy_comparator': ('not_constructed_one_write_gate_failed'
                                    if score['decision'] != 'one_write_storage_ok'
                                    else 'not_admitted_bounded_two_fact_followup_first'),
        'deployment': 'diagnostic_only_not_published',
        'resource': {
            'wall_seconds': elapsed, 'rss_start_bytes': rss_start,
            'max_rss_bytes': _rss_bytes(),
            'mps': _mps_memory() if device == 'mps' else None,
            'ledger_bytes': sum(path.stat().st_size for path in ledger_dir.glob('*.sqlite')),
            'image_bytes': sum(path.stat().st_size for path in images_dir.rglob('*')
                               if path.is_file()),
        },
        **score,
    }


def score_one_write(rows, codebooks, *, thresholds=None, mechanical):
    """Independently recompute one-write cosines and gates from raw rows."""
    settings = dict(ONE_WRITE_THRESHOLDS)
    settings.update(thresholds or {})
    prepared = []
    for row in rows:
        if row.get('status') != 'ok':
            continue
        book = codebooks.get(row.get('codebook_key'))
        if book is None or row['world_id'] != book['world_id']:
            raise ValueError('One-write readout references a foreign codebook')
        entry = book['layers'].get(str(row['layer']))
        if entry is None or row['key_sha256'] != entry['key_sha256']:
            raise ValueError('One-write readout key differs from its source writer')
        value = torch.tensor(entry['value_vector'], dtype=torch.float32)
        cosine = _cosine_heads(torch.tensor(row['readout'], dtype=torch.float32), value)
        prepared.append((row, cosine))

    def metric(against):
        values = [(row, cosine) for row, cosine in prepared
                  if row['against'] == against and row['phase'] == 'after_save']
        if not values:
            return {'n': 0, 'mean_cosine': None, 'min_cosine': None,
                    'max_abs_cosine': None, 'mean_readout_norm': None}
        cosines = [cosine for _, cosine in values]
        return {
            'n': len(values), 'mean_cosine': sum(cosines) / len(cosines),
            'min_cosine': min(cosines),
            'max_abs_cosine': max(abs(cosine) for cosine in cosines),
            'mean_readout_norm': sum(row['readout_norm'] for row, _ in values) / len(values),
        }

    per_depth, shared = metric('per_depth'), metric('shared_mean')
    zero, cross = metric('zero_state'), metric('cross_depth')
    pairing = {}
    for against in ONE_WRITE_READ_PATHS:
        before = {(row['world_id'], row['layer']): row for row in rows
                  if row['against'] == against and row['phase'] == 'before_save'}
        after = {(row['world_id'], row['layer']): row for row in rows
                 if row['against'] == against and row['phase'] == 'after_save'}
        if set(before) != set(after) or not before:
            pairing[against] = {'n': 0, 'max_abs_difference': None, 'equal': False}
            continue
        difference = max(
            float((torch.tensor(before[key]['readout'])
                   - torch.tensor(after[key]['readout'])).abs().max())
            for key in before)
        pairing[against] = {'n': len(before), 'max_abs_difference': difference,
                            'equal': difference <= settings['one_write_parity_atol']}
    frozen = (mechanical or {}).get('frozen') or {}
    frozen_ok = (frozen.get('optimizer_updates') == 0
                 and frozen.get('parameters_updated') == []
                 and frozen.get('parameter_versions_equal') is True
                 and frozen.get('requires_grad_all_false') is True)
    gates = {
        'per_depth_recovery': {
            'status': 'pass' if (per_depth['n'] and per_depth['mean_cosine']
                                 >= settings['one_write_recovery_floor']) else 'fail',
            'metrics': per_depth,
            'thresholds': {'one_write_recovery_floor':
                           settings['one_write_recovery_floor']}},
        'shared_reduction': {
            'status': 'pass' if (shared['n'] and shared['mean_cosine']
                                 >= settings['one_write_reduction_floor']) else 'fail',
            'metrics': shared,
            'thresholds': {'one_write_reduction_floor':
                           settings['one_write_reduction_floor']}},
        'zero_state': {
            'status': 'pass' if (zero['n'] and zero['max_abs_cosine']
                                 <= settings['one_write_zero_max_abs_cosine']) else 'fail',
            'metrics': zero,
            'thresholds': {'one_write_zero_max_abs_cosine':
                           settings['one_write_zero_max_abs_cosine']}},
        'cross_depth_readout': {'status': 'report_only', 'metrics': cross,
                                'thresholds': {}},
        'save_reload_parity': {
            'status': ('pass' if all(item['equal'] for item in pairing.values())
                       else 'fail'),
            'metrics': pairing,
            'thresholds': {'one_write_parity_atol':
                           settings['one_write_parity_atol']}},
        'frozen_weights': {
            'status': 'pass' if frozen_ok else 'fail',
            'metrics': frozen,
            'thresholds': {'optimizer_updates': 0, 'parameters_updated': []}},
    }
    failed = [name for name, gate in gates.items() if gate['status'] == 'fail']
    if 'frozen_weights' in failed:
        decision = 'fail_frozen_weights'
    elif 'zero_state' in failed:
        decision = 'fail_read_path_leakage'
    elif 'per_depth_recovery' in failed:
        decision = 'fail_writer_or_gate_normalization'
    elif 'shared_reduction' in failed:
        decision = 'fail_depth_reduction_interference'
    elif 'save_reload_parity' in failed:
        decision = 'fail_save_reload_parity'
    else:
        decision = 'one_write_storage_ok'
    next_step = {
        'fail_frozen_weights': 'Stop; the probe changed or enabled model parameters.',
        'fail_read_path_leakage': ('Inspect the SGDN-2 read einsum and state '
                                   'initialization; a zero state must return zero.'),
        'fail_writer_or_gate_normalization': ('Compare gate/erase/value norms by layer; '
                                              'repair the writer gate or normalization '
                                              'before any reader/pointer training.'),
        'fail_depth_reduction_interference': ('Per-depth reads recover while the convex '
                                              'mean does not; test a single-depth state or '
                                              'a depth-selection read path instead of a '
                                              'new reader.'),
        'fail_save_reload_parity': ('Inspect image save/load determinism before drawing '
                                    'storage conclusions.'),
        'one_write_storage_ok': ('One write recovers under the observed key; add a '
                                 'two-fact interference case before addressing work.'),
    }[decision]
    return {
        'summary': {
            'per_depth_mean_cosine': per_depth['mean_cosine'],
            'shared_mean_cosine': shared['mean_cosine'],
            'zero_max_abs_cosine': zero['max_abs_cosine'],
            'cross_depth_mean_cosine': cross['mean_cosine'],
            'save_reload_max_abs_difference': max(
                item['max_abs_difference'] or 0.0 for item in pairing.values()),
        },
        'localisation': {
            'key_cross_cosine_mean': _mean(book['key_cross_cosine']
                                           for book in codebooks.values()),
            'value_cross_cosine_mean': _mean(book['value_cross_cosine']
                                             for book in codebooks.values()),
            'gate_mean': _layer_mean(codebooks, 'gate_mean'),
            'erase_mean': _layer_mean(codebooks, 'erase_mean'),
            'decay_mean': _layer_mean(codebooks, 'decay_mean'),
            'key_norm_mean': _layer_mean(codebooks, 'key_norm'),
            'value_norm_mean': _layer_mean(codebooks, 'value_norm'),
        },
        'gates': gates, 'failed_gates': failed, 'decision': decision,
        'smallest_next_repair_discriminator': {'action': next_step,
                                               'failed_gates': failed},
        'quality_scope': ('Observed-key storage recovery only; no addressing, '
                          'native language readout or useful-memory claim'),
    }


def _mean(values):
    values = [value for value in values if value is not None]
    return (sum(values) / len(values)) if values else None


def _layer_mean(codebooks, field):
    return _mean(entry[field] for book in codebooks.values()
                 for entry in book['layers'].values())


def prepare_one_write(output, *, seed=20261025, world_count=8, layers=(13, 27),
                      facts_per_world=1,
                      model_path=PROJECT_ROOT / 'checkpoints/minicpm5_source',
                      model_lock=PROJECT_ROOT / 'configs/minicpm5_sealed_lock.local.json',
                      writer_extension=None):
    """Write seeded one-write worlds plus exact source/data/protocol pins."""
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Prepare output must be fresh or empty')
    layers = list(layers)
    _require(all(type(layer) is int and layer >= 0 for layer in layers),
             'Layers must be nonnegative integers')
    worlds = generate_one_write_worlds(seed=seed, world_count=world_count,
                                       facts_per_world=facts_per_world)
    planned = one_write_workload(world_count=world_count, layers=layers,
                                 facts_per_world=facts_per_world)
    _require(planned['planned_rows'] <= DEFAULT_LIMITS['max_queries'],
             'Scheduled one-write workload exceeds max_queries')
    output.mkdir(parents=True, exist_ok=True)
    data_path = output / 'data.json'
    _write_json(data_path, {'format': ONE_WRITE_WORLD_FORMAT, 'seed': seed,
                            'world_count': world_count, 'worlds': worlds})
    source = _source_pins(model_path, model_lock)
    extension = _extension_pins(writer_extension) if writer_extension is not None else None
    implementation = _implementation_sha256()
    protocol = {
        'format': ONE_WRITE_FORMAT,
        'mode': 'onewrite' if facts_per_world == 1 else 'interference',
        'seed': seed, 'world_count': world_count, 'layers': layers,
        'facts_per_world': facts_per_world,
        'layer': None, 'representation': FACT_REPRESENTATION,
        'native_reader': False,
        'thresholds': dict(ONE_WRITE_THRESHOLDS),
        'engine': {'heads': 4, 'key_size': 64, 'value_size': 64,
                   'update_rule': 'sgdn2', 'depth_mode': 'shared',
                   'depth_writers': True},
        'limits': dict(DEFAULT_LIMITS),
        'planned_workload': planned,
        'source': source, 'writer_extension': extension,
        'data': {'path': 'data.json', 'sha256': _sha_file(data_path),
                 'format': ONE_WRITE_WORLD_FORMAT, 'seed': seed,
                 'world_count': world_count, 'facts_per_world': facts_per_world},
        'implementation_sha256': implementation,
    }
    protocol_path = output / 'protocol.json'
    _write_json(protocol_path, protocol)
    manifest = {
        'format': ONE_WRITE_PREPARED_FORMAT, 'created_at_utc': _utc_now(),
        'native_computation': False, 'seed': seed, 'world_count': world_count,
        'layers': layers, 'facts_per_world': facts_per_world,
        'data_sha256': _sha_file(data_path),
        'protocol_sha256': _sha_file(protocol_path),
        'planned_workload': planned, 'source': source,
        'writer_extension': extension, 'implementation_sha256': implementation,
    }
    _write_json(output / 'manifest.json', manifest)
    return manifest


def verify_one_write_prepared(prepared):
    """Re-verify every one-write pin; a changed input fails clearly."""
    prepared = Path(prepared)
    manifest_path, data_path = prepared / 'manifest.json', prepared / 'data.json'
    protocol_path = prepared / 'protocol.json'
    manifest = json.loads(manifest_path.read_text())
    protocol = json.loads(protocol_path.read_text())
    data = json.loads(data_path.read_text())
    _require(manifest.get('format') == ONE_WRITE_PREPARED_FORMAT,
             'One-write prepared manifest format differs')
    _require(manifest.get('native_computation') is False,
             'Preparation must not perform native computation')
    _require(manifest.get('protocol_sha256') == _sha_file(protocol_path),
             'Protocol bytes differ from the prepared manifest')
    _require(manifest.get('data_sha256') == _sha_file(data_path),
             'Data bytes differ from the prepared manifest')
    _require(protocol.get('format') == ONE_WRITE_FORMAT
             and data.get('format') == ONE_WRITE_WORLD_FORMAT,
             'One-write protocol or data format differs')
    _require(protocol.get('data', {}).get('path') == 'data.json'
             and protocol['data'].get('sha256') == _sha_file(data_path),
             'One-write protocol data pin differs')
    worlds = data.get('worlds')
    validate_one_write_worlds(worlds)
    _require(worlds == generate_one_write_worlds(
        seed=protocol['seed'], world_count=protocol['world_count'],
        facts_per_world=protocol['facts_per_world']),
        'Prepared one-write worlds differ from the seeded generator')
    planned = one_write_workload(world_count=protocol['world_count'],
                                 layers=protocol['layers'],
                                 facts_per_world=protocol['facts_per_world'])
    _require(protocol.get('planned_workload') == planned
             and manifest.get('planned_workload') == planned,
             'Prepared one-write workload differs from the exact plan')
    source = protocol.get('source')
    _require(isinstance(source, dict) and source == _source_pins(
        source.get('model_path'), source.get('model_lock')),
        'One-write source pins changed')
    _require(manifest.get('source') == source,
             'Manifest/protocol one-write source pins differ')
    current_impl = _implementation_sha256()
    _require(protocol.get('implementation_sha256') == current_impl
             and manifest.get('implementation_sha256') == current_impl,
             'One-write implementation pins changed')
    extension = protocol.get('writer_extension')
    if extension is not None:
        _require(_extension_pins(extension['path']) == extension,
                 'One-write writer extension pins changed')
        _require(manifest.get('writer_extension') == extension,
                 'Manifest/protocol writer extension pins differ')
    elif manifest.get('writer_extension') is not None:
        raise ValueError('Manifest pins a writer extension that protocol does not')
    return {
        'format': ONE_WRITE_ADMISSION_FORMAT, 'status': 'admitted',
        'created_at_utc': _utc_now(),
        'prepared_manifest_sha256': _sha_file(manifest_path),
        'protocol_sha256': _sha_file(protocol_path),
        'data_sha256': _sha_file(data_path),
        'source': source, 'source_lock_sha256': source['model_lock_sha256'],
        'writer_extension': extension, 'implementation_sha256': current_impl,
        'seed': protocol['seed'], 'world_count': protocol['world_count'],
        'layers': protocol['layers'], 'planned_workload': planned,
        'checks': {
            'manifest_format': True, 'protocol_bytes': True, 'data_bytes': True,
            'seeded_world_regeneration': True, 'workload_bound': True,
            'source_pins': True, 'implementation_pins': True,
            'writer_extension_pins': True, 'no_native_computation': True,
        },
    }


def admit_one_write_probe(prepared, *, write_receipt=True):
    """Verify and admit a prepared one-write probe; rejections are preserved."""
    prepared = Path(prepared)
    try:
        receipt = verify_one_write_prepared(prepared)
    except Exception as error:
        if write_receipt and prepared.is_dir():
            _write_json(prepared / 'admission.json', {
                'format': ONE_WRITE_ADMISSION_FORMAT, 'status': 'rejected',
                'created_at_utc': _utc_now(),
                'error_type': type(error).__name__, 'error': str(error)})
        raise
    if write_receipt:
        _write_json(prepared / 'admission.json', receipt)
    return receipt

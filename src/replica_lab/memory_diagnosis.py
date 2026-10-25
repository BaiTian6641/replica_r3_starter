"""No-weight-update storage, addressing and readout diagnosis for SGDN-2 state.

The diagnostic writes fresh seeded opaque facts through the production
fact-binding representation, keeps the model frozen, and measures three
different read paths against a codebook built only from observed source values:

* the exact observed write key (a storage upper bound, not an addressing claim),
* independently rendered canonical queries encoded by the same writer, and
* the native reader query projection, when the adapter exposes it.

No answer labels, generated text, optimizer, pointer head or copy head enter a
memory write or a score. Raw readouts, per-entry cosines and all controls are
reported; gates are recomputed here from those raw values rather than trusted
from the producer.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

import torch
import torch.nn.functional as F

from .architecture.contracts import fingerprint
from .memory_system import EligibilityScope, MemoryRecord, RecordLedger
from .runtime.exact_memory import ExactMemoryReference
from .runtime.fact_memory import encode_fact_binding, fact_source_text, validate_fact
from .runtime.fast_memory import FastMemoryScope
from .runtime.memory_representation import (FACT_REPRESENTATION, parse_fact_source,
                                            representation_configuration)
from .runtime.persistent_fast_memory import PersistentFastMemory

DIAGNOSTIC_FORMAT = 'replica.memory_storage_diagnosis.v1'
WORLD_FORMAT = 'replica.memory_storage_probe.worlds.v1'
PROTOCOL_FORMAT = 'replica.memory_storage_probe.protocol.v1'
PREPARED_FORMAT = 'replica.memory_storage_probe.prepared.v1'
ADMISSION_FORMAT = 'replica.memory_storage_probe.admission.v1'
SUBJECT_ID = 'memory_storage_probe'
PURPOSE = 'storage_diagnosis'
MODE = 'parallel_replica'
SEALED_REVISION = 'f97400052a43d642bbc6e9975e2397e3ae6a6b52'
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_SEED = 20261025
DEFAULT_WORLD_COUNT = 2
DEFAULT_FACTS_PER_WORLD = 2
DEFAULT_DISTRACTORS_PER_WORLD = 1
DEFAULT_CORRECTION_INDEX = 0
DEFAULT_REVOCATION_INDEX = 1
DEFAULT_LAYER = 13
NATIVE_ENGINE = {'heads': 4, 'key_size': 64, 'value_size': 64,
                 'update_rule': 'sgdn2', 'depth_mode': 'shared',
                 'depth_writers': True}
DEFAULT_LIMITS = {'max_worlds': 64, 'max_facts_per_world': 32,
                  'max_distractors_per_world': 64, 'max_queries': 20000}
DEFAULT_THRESHOLDS = {
    'match_cosine_floor': 0.5,
    'match_margin_floor': 0.05,
    'storage_top1_floor': 0.6,
    'wrong_key_target_leak_max': 0.1,
    'zero_false_accept_max': 0.0,
    'unknown_false_accept_max': 0.2,
    'addressing_top1_floor': 0.4,
    'cross_world_false_accept_max': 0.2,
    'native_reader_top1_floor': 0.4,
    'correction_recovery_floor': 0.6,
    'superseded_selection_max': 0.1,
    'unaffected_retention_floor': 0.6,
    'revocation_false_accept_max': 0.2,
}
_IMPLEMENTATION_PATHS = (
    'src/replica_lab/memory_diagnosis.py',
    'src/replica_lab/runtime/persistent_fast_memory.py',
    'src/replica_lab/runtime/fact_memory.py',
    'src/replica_lab/runtime/fast_memory.py',
    'src/replica_lab/runtime/memory_representation.py',
    'src/replica_lab/runtime/exact_memory.py',
    'src/replica_lab/memory_system/ledger.py',
    'src/replica_lab/neural.py',
    'src/replica_lab/architecture/minicpm5.py',
    'src/replica_lab/architecture/ports.py',
    'src/replica_lab/architecture/loading.py',
    'src/replica_lab/architecture/contracts.py',
    'scripts/run_memory_storage_probe.py',
    'scripts/run_capacity_preflight.py',
    'src/replica_lab/capacity_guard.py',
    'src/replica_lab/profiling.py',
    'src/replica_lab/contracts.py',
    'src/replica_lab/memory_system/index.py',
    'src/replica_lab/memory_one_write.py',
)


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _sha_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


def _sha_text(text):
    return _sha_bytes(text.encode('utf-8'))


def _sha_file(path):
    return _sha_bytes(Path(path).read_bytes())


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True,
                                     allow_nan=False) + '\n')


def _write_jsonl(path, rows):
    Path(path).write_text(''.join(json.dumps(row, sort_keys=True, allow_nan=False) + '\n'
                                  for row in rows))


def _resolve_path(value):
    path = Path(value)
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def _display_path(value):
    path = Path(value).resolve()
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _resolution_thresholds(thresholds=None):
    result = dict(DEFAULT_THRESHOLDS)
    if thresholds is None:
        return result
    if not isinstance(thresholds, dict):
        raise ValueError('Thresholds must be an object')
    unknown = set(thresholds) - set(DEFAULT_THRESHOLDS)
    if unknown:
        raise ValueError(f'Unknown diagnostic thresholds: {sorted(unknown)}')
    for name, value in thresholds.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f'Threshold {name} must be numeric')
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f'Threshold {name} must be in [0, 1]')
        result[name] = float(value)
    return result


def _protocol_defaults():
    return {
        'format': PROTOCOL_FORMAT,
        'seed': DEFAULT_SEED,
        'world_count': DEFAULT_WORLD_COUNT,
        'facts_per_world': DEFAULT_FACTS_PER_WORLD,
        'distractors_per_world': DEFAULT_DISTRACTORS_PER_WORLD,
        'correction_index': DEFAULT_CORRECTION_INDEX,
        'revocation_index': DEFAULT_REVOCATION_INDEX,
        'layer': DEFAULT_LAYER,
        'representation': FACT_REPRESENTATION,
        'native_reader': True,
        'thresholds': _resolution_thresholds(None),
        'engine': dict(NATIVE_ENGINE),
        'limits': dict(DEFAULT_LIMITS),
    }


def _planned_workload_counts(*, world_count, facts_per_world, distractors_per_world,
                             native_reader, correction_index, revocation_index):
    """Exact scheduled row bound, including before/after-reload evaluations."""
    live_initial = facts_per_world + distractors_per_world
    live_corrected = facts_per_world + distractors_per_world
    live_current = facts_per_world + distractors_per_world - 1
    if live_current < 1:
        raise ValueError('A current view with a revocation must retain one entry')
    stages = (('initial', live_initial, 0),
              ('corrected', live_corrected, 0),
              ('current_view', live_current, 1))
    target_rows = control_rows = unsupported_rows = 0
    per_stage = {}
    for name, live_entries, revoked_rows in stages:
        target = 3 * live_entries + (live_entries if native_reader else 0)
        control = 4 + revoked_rows
        unsupported = 0 if native_reader else 1
        per_stage[name] = {
            'live_entries': live_entries,
            'target_rows': target,
            'control_rows': control,
            'unsupported_rows': unsupported,
            'rows_per_evaluation': target + control + unsupported,
        }
        target_rows += target
        control_rows += control
        unsupported_rows += unsupported
    per_world_per_evaluation = target_rows + control_rows + unsupported_rows
    per_world = per_world_per_evaluation * 2
    return {
        'world_count': world_count,
        'facts_per_world': facts_per_world,
        'distractors_per_world': distractors_per_world,
        'correction_index': correction_index,
        'revocation_index': revocation_index,
        'native_reader': native_reader,
        'evaluations_per_state': 2,
        'states': 3 * world_count,
        'per_stage': per_stage,
        'planned_target_rows': target_rows * 2 * world_count,
        'planned_control_rows': control_rows * 2 * world_count,
        'planned_unsupported_rows': unsupported_rows * 2 * world_count,
        'planned_target_cases': target_rows * world_count,
        'planned_control_cases': control_rows * world_count,
        'planned_unsupported_cases': unsupported_rows * world_count,
        'planned_diagnostic_cases': (target_rows + control_rows + unsupported_rows) * world_count,
        'rows_per_world': per_world,
        'total_planned_rows': per_world * world_count,
    }


def _merge_protocol(protocol):
    if not isinstance(protocol, dict):
        raise TypeError('Protocol must be an object')
    result = _protocol_defaults()
    for name, value in protocol.items():
        if name == 'thresholds':
            result[name] = _resolution_thresholds(value)
        elif name == 'worlds':
            continue
        else:
            result[name] = value
    if result['format'] not in (PROTOCOL_FORMAT,):
        raise ValueError('Storage-probe protocol format differs')
    if type(result['seed']) is not int or result['seed'] < 0:
        raise ValueError('Seed must be a nonnegative integer')
    for name in ('world_count', 'facts_per_world'):
        if type(result[name]) is not int or result[name] < 1:
            raise ValueError(f'{name} must be a positive integer')
    if type(result['distractors_per_world']) is not int or result['distractors_per_world'] < 0:
        raise ValueError('distractors_per_world must be a nonnegative integer')
    for name in ('correction_index', 'revocation_index'):
        if type(result[name]) is not int or not 0 <= result[name] < result['facts_per_world']:
            raise ValueError(f'{name} must index a fact in each world')
    if result['correction_index'] == result['revocation_index']:
        raise ValueError('Correction and revocation must target different facts')
    if result['facts_per_world'] < 2:
        raise ValueError('At least two facts per world are required for correction/revocation')
    if result['distractors_per_world'] and result['world_count'] < 2:
        raise ValueError('Independent-world distractors require at least two worlds')
    if result['layer'] is not None and (type(result['layer']) is not int or result['layer'] < 0):
        raise ValueError('Layer must be a nonnegative integer or null')
    if result['representation'] != FACT_REPRESENTATION:
        raise ValueError('Storage diagnosis requires fact-binding representation')
    if type(result['native_reader']) is not bool:
        raise ValueError('native_reader must be boolean')
    limits = result.get('limits')
    if not isinstance(limits, dict) or set(limits) != set(DEFAULT_LIMITS):
        raise ValueError('Diagnostic limits differ')
    for name, cap in DEFAULT_LIMITS.items():
        value = limits[name]
        if type(value) is not int or value < 1:
            raise ValueError(f'Limit {name} must be a positive integer')
        if name == 'max_worlds' and result['world_count'] > value:
            raise ValueError('World count exceeds the diagnostic limit')
        if name == 'max_facts_per_world' and result['facts_per_world'] > value:
            raise ValueError('Fact count exceeds the diagnostic limit')
        if name == 'max_distractors_per_world' and result['distractors_per_world'] > value:
            raise ValueError('Distractor count exceeds the diagnostic limit')
    planned = _planned_workload_counts(
        world_count=result['world_count'],
        facts_per_world=result['facts_per_world'],
        distractors_per_world=result['distractors_per_world'],
        native_reader=result['native_reader'],
        correction_index=result['correction_index'],
        revocation_index=result['revocation_index'])
    if planned['total_planned_rows'] > limits['max_queries']:
        raise ValueError(
            f"Scheduled workload {planned['total_planned_rows']} rows exceeds "
            f"max_queries={limits['max_queries']}")
    if 'planned_workload' in protocol and protocol['planned_workload'] != planned:
        raise ValueError('Declared planned workload differs from the exact plan')
    result['thresholds'] = _resolution_thresholds(result.get('thresholds'))
    return result


def planned_workload(protocol):
    """Public exact workload bound; raises before any model allocation."""
    normalized = _merge_protocol(protocol)
    return _planned_workload_counts(
        world_count=normalized['world_count'],
        facts_per_world=normalized['facts_per_world'],
        distractors_per_world=normalized['distractors_per_world'],
        native_reader=normalized['native_reader'],
        correction_index=normalized['correction_index'],
        revocation_index=normalized['revocation_index'])


def _opaque_token(tag, material):
    return f'{tag}-{hashlib.sha256(material.encode("utf-8")).hexdigest()[:10]}'


def _fresh_token(tag, material, used):
    for attempt in range(4096):
        value = _opaque_token(tag, f'{material}:{attempt}')
        if value not in used:
            used.add(value)
            return value
    raise ValueError(f'Could not allocate a fresh opaque {tag}')


def generate_worlds(*, seed=DEFAULT_SEED, world_count=DEFAULT_WORLD_COUNT,
                    facts_per_world=DEFAULT_FACTS_PER_WORLD,
                    distractors_per_world=DEFAULT_DISTRACTORS_PER_WORLD,
                    correction_index=DEFAULT_CORRECTION_INDEX,
                    revocation_index=DEFAULT_REVOCATION_INDEX):
    """Build deterministic opaque worlds with disjoint keys and values."""
    if type(seed) is not int or seed < 0:
        raise ValueError('Seed must be a nonnegative integer')
    if type(world_count) is not int or world_count < 1:
        raise ValueError('world_count must be positive')
    if type(facts_per_world) is not int or facts_per_world < 2:
        raise ValueError('facts_per_world must be at least two')
    if type(distractors_per_world) is not int or distractors_per_world < 0:
        raise ValueError('distractors_per_world must be nonnegative')
    if distractors_per_world and world_count < 2:
        raise ValueError('Independent-world distractors require two worlds')
    for name, value in (('correction_index', correction_index),
                        ('revocation_index', revocation_index)):
        if type(value) is not int or not 0 <= value < facts_per_world:
            raise ValueError(f'{name} must index a fact')
    if correction_index == revocation_index:
        raise ValueError('Correction and revocation must target different facts')

    used_values = set()
    used_keys = set()
    definitions = []
    for world_index in range(world_count):
        world_material = f'{seed}:world:{world_index}'
        world_id = _opaque_token('world', world_material)
        facts = []
        for fact_index in range(facts_per_world):
            fact_material = f'{world_material}:fact:{fact_index}'
            entity = _fresh_token('ent', f'{fact_material}:entity', used_keys)
            attribute = _fresh_token('attr', f'{fact_material}:attribute', used_keys)
            value = _fresh_token('val', f'{fact_material}:value', used_values)
            facts.append({
                'fact_id': f'{world_id}:fact:{fact_index}',
                'entity': entity,
                'attribute': attribute,
                'value': value,
                'source_world_id': world_id,
            })
        correction_target = facts[correction_index]
        correction_value = _fresh_token(
            'val', f'{world_material}:correction:value', used_values)
        correction = {
            'record_id': f'{world_id}:correction',
            'fact_id': correction_target['fact_id'],
            'entity': correction_target['entity'],
            'attribute': correction_target['attribute'],
            'value': correction_value,
            'supersedes': correction_target['fact_id'],
        }
        definitions.append({'world_id': world_id,
                            'family_id': f'family-{world_index:02d}',
                            'facts': facts, 'correction': correction})

    worlds = []
    for world_index, definition in enumerate(definitions):
        world_id = definition['world_id']
        facts = definition['facts']
        state_keys = {(fact['entity'], fact['attribute']) for fact in facts}
        state_values = {fact['value'] for fact in facts}
        candidates = [fact for other_index in range(world_count)
                      if other_index != world_index
                      for fact in definitions[other_index]['facts']]
        distractors = []
        for fact in candidates:
            if len(distractors) >= distractors_per_world:
                break
            key = (fact['entity'], fact['attribute'])
            if key in state_keys or fact['value'] in state_values:
                continue
            state_keys.add(key)
            state_values.add(fact['value'])
            distractors.append({
                'fact_id': f'{world_id}:distractor:{len(distractors)}',
                'entity': fact['entity'],
                'attribute': fact['attribute'],
                'value': fact['value'],
                'origin_world_id': fact['source_world_id'],
            })
        if len(distractors) != distractors_per_world:
            raise ValueError('Not enough distinct independent-world distractor facts')
        worlds.append({
            'world_id': world_id,
            'family_id': definition['family_id'],
            'facts': facts,
            'distractors': distractors,
            'correction': definition['correction'],
            'revocation_fact_id': facts[revocation_index]['fact_id'],
        })
    validate_worlds(worlds)
    return worlds


def validate_worlds(worlds):
    """Reject extra labels, duplicate addresses/values and malformed worlds."""
    if not isinstance(worlds, list) or not worlds:
        raise ValueError('Worlds must be a nonempty list')
    world_ids, fact_ids, source_values, source_keys = set(), set(), set(), set()
    correction_values = set()
    known_world_ids = {world.get('world_id') for world in worlds
                       if isinstance(world, dict)}
    for world in worlds:
        if not isinstance(world, dict) or set(world) != {
                'world_id', 'family_id', 'facts', 'distractors', 'correction',
                'revocation_fact_id'}:
            raise ValueError('World record has unexpected fields')
        if not isinstance(world['world_id'], str) or world['world_id'] in world_ids:
            raise ValueError('World IDs must be unique nonempty strings')
        world_ids.add(world['world_id'])
        if not isinstance(world['family_id'], str) or not world['family_id']:
            raise ValueError('World family ID must be nonempty')
        if not isinstance(world['facts'], list) or len(world['facts']) < 2:
            raise ValueError('Each world requires at least two facts')
        state_keys, state_values = set(), set()
        for fact in world['facts']:
            if not isinstance(fact, dict) or set(fact) != {
                    'fact_id', 'entity', 'attribute', 'value', 'source_world_id'}:
                raise ValueError('Fact has unexpected fields or labels')
            validate_fact({name: fact[name] for name in ('entity', 'attribute', 'value')})
            if fact['source_world_id'] != world['world_id']:
                raise ValueError('Source fact belongs to another world')
            if fact['fact_id'] in fact_ids:
                raise ValueError('Fact IDs must be unique')
            fact_ids.add(fact['fact_id'])
            key = (fact['entity'], fact['attribute'])
            if key in source_keys:
                raise ValueError('Observed fact addresses must be unique')
            source_keys.add(key)
            state_keys.add(key)
            if fact['value'] in source_values:
                raise ValueError('Observed source values must be unique')
            source_values.add(fact['value'])
            state_values.add(fact['value'])
        for distractor in world['distractors']:
            if not isinstance(distractor, dict) or set(distractor) != {
                    'fact_id', 'entity', 'attribute', 'value', 'origin_world_id'}:
                raise ValueError('Distractor has unexpected fields or labels')
            validate_fact({name: distractor[name] for name in ('entity', 'attribute', 'value')})
            if distractor['origin_world_id'] == world['world_id']:
                raise ValueError('Distractors must come from an independent world')
            if distractor['origin_world_id'] not in known_world_ids:
                raise ValueError('Distractor origin world is not in the prepared lineage')
            if distractor['fact_id'] in fact_ids:
                raise ValueError('Distractor IDs must be unique')
            fact_ids.add(distractor['fact_id'])
            key = (distractor['entity'], distractor['attribute'])
            if key in state_keys:
                raise ValueError('Distractor address collides with a source fact')
            state_keys.add(key)
            if distractor['value'] in state_values:
                raise ValueError('Distractor value collides with this state')
            state_values.add(distractor['value'])
        correction = world['correction']
        if not isinstance(correction, dict) or set(correction) != {
                'record_id', 'fact_id', 'entity', 'attribute', 'value', 'supersedes'}:
            raise ValueError('Correction has unexpected fields')
        target = next((fact for fact in world['facts']
                       if fact['fact_id'] == correction['fact_id']), None)
        if target is None or correction['supersedes'] != target['fact_id']:
            raise ValueError('Correction must supersede an existing source fact')
        if (correction['entity'], correction['attribute']) != (target['entity'], target['attribute']):
            raise ValueError('Correction address differs from its source fact')
        validate_fact({name: correction[name] for name in ('entity', 'attribute', 'value')})
        if correction['value'] in state_values or correction['value'] in correction_values:
            raise ValueError('Correction requires a fresh opaque value')
        correction_values.add(correction['value'])
        state_values.add(correction['value'])
        if world['revocation_fact_id'] not in {fact['fact_id'] for fact in world['facts']}:
            raise ValueError('Revocation target is not an observed source fact')
        if world['revocation_fact_id'] == correction['fact_id']:
            raise ValueError('Revocation cannot target the corrected fact')
    if correction_values & source_values:
        raise ValueError('Correction values must not reuse an observed source value')
    return True


def world_records(world):
    """Render immutable source records without accepting evaluation labels."""
    world_id = world['world_id']
    initial, origin_by_record = [], {}
    for fact in world['facts']:
        record = MemoryRecord(
            record_id=fact['fact_id'], subject_id=SUBJECT_ID, branch_id=world_id,
            text=fact_source_text({name: fact[name]
                                   for name in ('entity', 'attribute', 'value')}),
            observed_at=1, event_at=len(initial) + 1, purpose=PURPOSE,
            kind='source_fact', source_id=f"{world_id}:source:{len(initial)}",
            authenticity='verified_source')
        initial.append(record)
        origin_by_record[record.record_id] = world_id
    for distractor in world['distractors']:
        record = MemoryRecord(
            record_id=distractor['fact_id'], subject_id=SUBJECT_ID, branch_id=world_id,
            text=fact_source_text({name: distractor[name]
                                   for name in ('entity', 'attribute', 'value')}),
            observed_at=1, event_at=len(initial) + 1, purpose=PURPOSE,
            kind='source_fact', source_id=f"{world_id}:distractor-source:{len(initial)}",
            authenticity='verified_source')
        initial.append(record)
        origin_by_record[record.record_id] = distractor['origin_world_id']
    correction = world['correction']
    correction_record = MemoryRecord(
        record_id=correction['record_id'], subject_id=SUBJECT_ID, branch_id=world_id,
        text=fact_source_text({name: correction[name]
                               for name in ('entity', 'attribute', 'value')},
                              correction=True),
        observed_at=2, event_at=len(initial) + 1, purpose=PURPOSE,
        kind='source_correction', supersedes=correction['supersedes'],
        source_id=f'{world_id}:correction-source', authenticity='verified_source')
    origin_by_record[correction_record.record_id] = world_id
    return tuple(initial), correction_record, origin_by_record


def _scope(world, as_of):
    return EligibilityScope(SUBJECT_ID, world['world_id'], PURPOSE, int(as_of), MODE, None)


def _fast_scope(world, epoch):
    return FastMemoryScope(SUBJECT_ID, world['world_id'], PURPOSE, MODE, int(epoch))


def _resolve_layer(model, protocol):
    layers = tuple(model.ports.injection_layers)
    declared = protocol.get('layer')
    if declared is None:
        declared = layers[0]
    if type(declared) is not int or declared not in layers:
        raise ValueError(f'Declared layer {declared} is not an injection layer {layers}')
    return declared


def _pooled(value, mask):
    if value.ndim != 4 or value.shape[0] != 1 or value.shape[1] < 1:
        raise ValueError('Writer output must be [1, tokens, heads, width]')
    if mask is None:
        return value.mean(dim=1, keepdim=True)
    weights = mask.to(device=value.device, dtype=value.dtype).unsqueeze(-1).unsqueeze(-1)
    if tuple(weights.shape[:2]) != tuple(value.shape[:2]) or not bool(weights.sum() > 0):
        raise ValueError('Query mask carries no valid token')
    return (value * weights).sum(dim=1, keepdim=True) / weights.sum(dim=1, keepdim=True).clamp_min(1)


def _token_ids(tokenizer, text, *, maximum=256):
    if not isinstance(text, str) or not text.strip():
        raise ValueError('Query text must be nonempty')
    ids = tokenizer(text, add_special_tokens=False)['input_ids']
    if not ids or len(ids) > maximum:
        raise ValueError('Query does not fit without silent truncation')
    return ids


def query_key(model, tokenizer, text, *, layer):
    """Encode an independently rendered canonical query through the writer key path."""
    _token_ids(tokenizer, text)
    write = model.encode_fast_observation(text, tokenizer, layer=layer)
    return _pooled(write.keys, write.mask)


def _fact_subset(fact):
    return {name: fact[name] for name in ('entity', 'attribute', 'value')}


def observed_binding(model, tokenizer, fact, *, layer):
    return encode_fact_binding(model, tokenizer, _fact_subset(fact), layer=layer)


def canonical_query_text(fact, variant):
    entity, attribute = fact['entity'], fact['attribute']
    if variant == 'address':
        return f'entity {entity} attribute {attribute}'
    if variant == 'paraphrase':
        return f'which value belongs to {attribute} of project {entity}'
    raise ValueError('Unknown canonical query variant')


def native_query_text(fact):
    return (f"Question: which value is recorded for {fact['attribute']} "
            f"of project {fact['entity']}?")


def native_reader_query(model, tokenizer, text, *, layer):
    """Return the native reader's per-head query at the last prompt token.

    Only the adapter's existing declaration is used. If the adapter has no
    declared fast reader, the caller receives an explicit prerequisite instead
    of a synthetic query.
    """
    ports = getattr(model, 'fast_ports', None)
    if ports is None:
        return None, 'native fast reader unavailable: adapter declares no fast_ports'
    key = str(layer)
    if key not in ports:
        return None, f'native fast reader unavailable: no declared port at layer {layer}'
    if not hasattr(model, '_native'):
        return None, 'native fast reader unavailable: adapter exposes no native forward'
    if not hasattr(tokenizer, 'apply_chat_template'):
        return None, 'native fast reader unavailable: tokenizer has no chat template'
    prompt = tokenizer.apply_chat_template(
        [{'role': 'user', 'content': text}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False)
    ids = tokenizer(prompt, add_special_tokens=False)['input_ids']
    maximum = int(getattr(getattr(model, 'base', None), 'config', type('C', (), {
        'max_position_embeddings': 4096})()).max_position_embeddings)
    if not ids or len(ids) > maximum:
        return None, 'native reader query exceeds the declared native position budget'
    device = next(model.parameters()).device
    ids = torch.tensor([ids], dtype=torch.long, device=device)
    mask = torch.ones_like(ids, dtype=torch.bool)
    hidden, _ = model._native(ids, mask, stop_layer=int(layer))
    port = ports[key]
    query = port.query_projection(port.query_norm(hidden))
    query = query.reshape(1, hidden.shape[1], port.heads, port.key_size)
    query = F.normalize(query.float(), dim=-1, eps=1e-6)[:, -1:]
    return query, None


def _read_heads(memory, query):
    weights = memory.read().weights
    if (query.ndim != 4 or query.shape[0] != 1 or query.shape[1] != 1
            or query.shape[2] != weights.shape[1] or query.shape[3] != weights.shape[2]):
        raise ValueError('Query does not match the committed SGDN-2 state')
    normalized = F.normalize(query.to(device=weights.device, dtype=weights.dtype),
                             dim=-1, eps=1e-6)
    return torch.einsum('bthk,bhkv->bthv', normalized, weights)[0, 0]


def _cosine_heads(readout, vector):
    readout = torch.as_tensor(readout, dtype=torch.float32)
    vector = torch.as_tensor(vector, dtype=torch.float32)
    if readout.shape != vector.shape:
        raise ValueError('Readout and codebook vector shapes differ')
    dot = (readout * vector).sum(dim=-1)
    read_norm = torch.linalg.vector_norm(readout, dim=-1)
    value_norm = torch.linalg.vector_norm(vector, dim=-1)
    valid = (read_norm > 1e-12) & (value_norm > 1e-12)
    if not bool(valid.any()):
        return 0.0
    return float((dot[valid] / (read_norm[valid] * value_norm[valid])).mean())


def _parameter_versions(model):
    return tuple((name, id(parameter), parameter._version, tuple(parameter.shape),
                  str(parameter.dtype), str(parameter.device), parameter.data_ptr())
                 for name, parameter in model.named_parameters())


def _freeze(model):
    for parameter in model.parameters():
        parameter.requires_grad_(False)


def _state_signature(memory):
    return {
        'state_digest': memory.read().state_digest,
        'chunk_index': int(memory.state.chunk_index),
        'lineage': [list(item) for item in memory.lineage],
        'events': memory.events,
        'producer': memory.producer,
        'semantics': memory.semantics,
    }


def _readout_identity(row):
    return (row['world_id'], row['stage'], row['codebook_key'],
            row['query_kind'], row['query_label'], row['target_entry_id'],
            row.get('control_owner_entry_id'), row.get('superseded_entry_id'),
            row.get('revoked_entry_id'), row['query_text_sha256'], row['layer'],
            bool(row.get('negative_used')))


def _readout_inventory(rows):
    items = [(_readout_identity(row), row['status'],
              json.dumps(row['readout'], allow_nan=False))
             for row in rows]
    items.sort(key=lambda item: json.dumps(item[0], ensure_ascii=False,
                                           allow_nan=False))
    return items


def _restart_readout_parity(before_rows, after_rows):
    """Compare the two independently evaluated raw readout sets."""
    before = _readout_inventory(before_rows)
    after = _readout_inventory(after_rows)
    equal = before == after
    mismatch = (len(before) + len(after) if len(before) != len(after)
                else sum(1 for left, right in zip(before, after) if left != right))
    return {
        'equal': equal,
        'before_rows': len(before),
        'after_rows': len(after),
        'paired_rows': min(len(before), len(after)),
        'mismatch_count': mismatch,
        'before_inventory_sha256': _sha_text(json.dumps(before, allow_nan=False)),
        'after_inventory_sha256': _sha_text(json.dumps(after, allow_nan=False)),
    }


def _build_codebook(model, tokenizer, records, *, world_id, stage, layer,
                    origin_by_record, correction_record_id, negative_record=None):
    entries, key_index, negative_entries = {}, {}, {}
    for record in records:
        fact, correction = parse_fact_source(record.text)
        binding = observed_binding(model, tokenizer, fact, layer=layer)
        value_vector = binding.values[0, 0].detach().float().cpu()
        write_key = binding.keys[0, 0].detach().float().cpu()
        if not bool(torch.isfinite(value_vector).all()) or not bool(torch.isfinite(write_key).all()):
            raise ValueError('Writer produced a non-finite key/value vector')
        origin = origin_by_record.get(record.record_id, 'unknown')
        if record.record_id == correction_record_id:
            role, source_family = 'correction_target', 'source'
        elif origin != world_id:
            role, source_family = 'distractor', 'independent_world'
        elif stage == 'current_view':
            role, source_family = 'retained', 'source'
        else:
            role, source_family = 'source', 'source'
        entry = {
            'entry_id': record.record_id,
            'entity': fact['entity'],
            'attribute': fact['attribute'],
            'source_hash': record.source_hash,
            'record_hash': record.record_hash,
            'value_sha256': _sha_text(fact['value']),
            'correction': bool(correction),
            'role': role,
            'source_family': source_family,
            'value_vector': value_vector.tolist(),
            '_value': fact['value'],
            '_write_key': write_key.tolist(),
        }
        if role == 'correction_target':
            entry['superseded_entry_id'] = None
        entries[record.record_id] = entry
        key_index[f"{fact['entity']}\x00{fact['attribute']}"] = record.record_id
    if negative_record is not None:
        if negative_record.record_id in entries:
            raise ValueError('Superseded negative cannot also be a live codebook entry')
        fact, correction = parse_fact_source(negative_record.text)
        binding = observed_binding(model, tokenizer, fact, layer=layer)
        value_vector = binding.values[0, 0].detach().float().cpu()
        write_key = binding.keys[0, 0].detach().float().cpu()
        if not bool(torch.isfinite(value_vector).all()) or not bool(torch.isfinite(write_key).all()):
            raise ValueError('Negative writer produced a non-finite key/value vector')
        target = entries.get(correction_record_id)
        if target is None:
            raise ValueError('Negative entry requires the live correction target')
        if (target['entity'], target['attribute']) != (fact['entity'], fact['attribute']):
            raise ValueError('Superseded negative address differs from the correction target')
        negative_entries[negative_record.record_id] = {
            'entry_id': negative_record.record_id,
            'entity': fact['entity'],
            'attribute': fact['attribute'],
            'source_hash': negative_record.source_hash,
            'record_hash': negative_record.record_hash,
            'value_sha256': _sha_text(fact['value']),
            'correction': bool(correction),
            'role': 'superseded_negative',
            'source_family': ('source' if origin_by_record.get(
                negative_record.record_id) == world_id else 'independent_world'),
            'negative': True,
            'live': False,
            'value_vector': value_vector.tolist(),
            '_value': fact['value'],
            '_write_key': write_key.tolist(),
        }
    if not entries:
        raise ValueError('Codebook cannot be empty')
    return {'key': f'{world_id}/{stage}/layer-{layer}', 'world_id': world_id,
            'stage': stage, 'layer': layer, 'entries': entries,
            'negative_entries': negative_entries, 'key_index': key_index}


def _public_codebook(codebook):
    def public(items):
        return {entry_id: {name: value for name, value in entry.items()
                           if not name.startswith('_')}
                for entry_id, entry in items.items()}
    return {'key': codebook['key'], 'world_id': codebook['world_id'],
            'stage': codebook['stage'], 'layer': codebook['layer'],
            'entries': public(codebook['entries']),
            'negative_entries': public(codebook.get('negative_entries', {})),
            'key_index': dict(codebook['key_index'])}


def _make_row(memory, codebook, *, query, query_kind, query_text_sha256,
              target_entry_id, expected, role, source_family, layer, stage,
              phase=None, control_owner_entry_id=None, superseded_entry_id=None,
              revoked_entry_id=None, thresholds=None, query_label=None,
              status='ok', unavailable_reason=None, negative_entries=None,
              include_negatives=False):
    thresholds = thresholds or DEFAULT_THRESHOLDS
    negatives = (negative_entries or {}) if include_negatives else {}
    row = {
        'world_id': codebook['world_id'],
        'stage': stage,
        'phase': phase,
        'codebook_key': codebook['key'],
        'layer': layer,
        'query_kind': query_kind,
        'query_label': query_label,
        'query_text_sha256': query_text_sha256,
        'target_entry_id': target_entry_id,
        'expected': expected,
        'role': role,
        'source_family': source_family,
        'control_owner_entry_id': control_owner_entry_id,
        'superseded_entry_id': superseded_entry_id,
        'revoked_entry_id': revoked_entry_id,
        'status': status,
        'unavailable_reason': unavailable_reason,
        'negative_used': bool(negatives),
        'negative_entry_ids': sorted(negatives),
    }
    if status != 'ok':
        row.update({'readout': None, 'readout_sha256': None, 'readout_norm': None,
                    'scores': {}, 'top_entry_id': None, 'top_cosine': None,
                    'second_cosine': None, 'margin': None, 'accepted': False,
                    'target_match': False, 'key_owner_match': False,
                    'target_cosine': None, 'superseded_selected': False,
                    'revoked_selected': False})
        return row
    readout = _read_heads(memory, query).detach().float().cpu()
    if not bool(torch.isfinite(readout).all()):
        raise ValueError('Readout is not finite')
    readout_list = readout.tolist()
    readout_norm = float(torch.linalg.vector_norm(readout))
    scores = {entry_id: _cosine_heads(readout, entry['value_vector'])
              for entry_id, entry in codebook['entries'].items()}
    scores.update({entry_id: _cosine_heads(readout, entry['value_vector'])
                   for entry_id, entry in negatives.items()})
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    top_entry_id, top_cosine = (ordered[0] if ordered else (None, None))
    second_cosine = ordered[1][1] if len(ordered) > 1 else None
    margin = (top_cosine - second_cosine
              if top_cosine is not None and second_cosine is not None else None)
    accepted = bool(
        top_entry_id is not None and readout_norm > 1e-12
        and top_cosine >= thresholds['match_cosine_floor']
        and margin is not None and margin >= thresholds['match_margin_floor'])
    target_match = bool(accepted and target_entry_id is not None
                        and top_entry_id == target_entry_id)
    row.update({
        'readout': readout_list,
        'readout_sha256': _sha_text(json.dumps(readout_list, allow_nan=False)),
        'readout_norm': readout_norm,
        'scores': scores,
        'top_entry_id': top_entry_id,
        'top_cosine': top_cosine,
        'second_cosine': second_cosine,
        'margin': margin,
        'accepted': accepted,
        'target_match': target_match,
        'key_owner_match': bool(accepted and control_owner_entry_id is not None
                                and top_entry_id == control_owner_entry_id),
        'target_cosine': (scores.get(target_entry_id)
                          if target_entry_id is not None else None),
        'superseded_selected': bool(accepted and superseded_entry_id is not None
                                    and top_entry_id == superseded_entry_id),
        'revoked_selected': bool(accepted and revoked_entry_id is not None
                                 and top_entry_id == revoked_entry_id),
    })
    return row


def unknown_fact(world, *, seed):
    material = f'{seed}:{world["world_id"]}:unknown'
    return {'entity': _opaque_token('ent', material + ':entity'),
            'attribute': _opaque_token('attr', material + ':attribute'),
            'value': _opaque_token('val', material + ':value')}


def foreign_fact(worlds, world):
    observed_values = {fact['value'] for fact in world['facts']}
    observed_values.update(distractor['value'] for distractor in world['distractors'])
    observed_keys = {(fact['entity'], fact['attribute']) for fact in world['facts']}
    observed_keys.update((distractor['entity'], distractor['attribute'])
                         for distractor in world['distractors'])
    for other in worlds:
        if other['world_id'] == world['world_id']:
            continue
        for fact in other['facts']:
            if fact['value'] in observed_values:
                continue
            if (fact['entity'], fact['attribute']) in observed_keys:
                continue
            return dict(fact)
    return None


def _evaluate_state(memory, model, tokenizer, codebook, *, stage, world, layer,
                    native_reader, unknown, foreign, revoked_fact, thresholds,
                    phase):
    rows = []
    entries = list(codebook['entries'].items())
    negatives = codebook.get('negative_entries', {})

    def add(query, query_kind, digest, **kwargs):
        rows.append(_make_row(memory, codebook, query=query,
                              query_kind=query_kind, query_text_sha256=digest,
                              layer=layer, stage=stage, phase=phase,
                              thresholds=thresholds, **kwargs))

    for entry_id, entry in entries:
        fact = {'entity': entry['entity'], 'attribute': entry['attribute'],
                'value': entry['_value']}
        write_key = torch.tensor(entry['_write_key'], dtype=torch.float32).view(1, 1, *torch.tensor(
            entry['_write_key']).shape)
        source_digest = _sha_text(fact_source_text(fact))
        superseded = entry.get('superseded_entry_id')
        if entry['role'] == 'correction_target':
            superseded = world['correction']['supersedes']
        use_negatives = bool(negatives) and entry['role'] == 'correction_target'
        negative_kwargs = ({'negative_entries': negatives, 'include_negatives': True}
                           if use_negatives else {})
        add(write_key, 'observed_write_key', source_digest,
            target_entry_id=entry_id, expected='target', role=entry['role'],
            source_family=entry['source_family'], superseded_entry_id=superseded,
            query_label='exact observed write key', **negative_kwargs)
        for variant in ('address', 'paraphrase'):
            text = canonical_query_text(fact, variant)
            add(query_key(model, tokenizer, text, layer=layer),
                f'canonical_{variant}', _sha_text(text),
                target_entry_id=entry_id, expected='target', role=entry['role'],
                source_family=entry['source_family'], superseded_entry_id=superseded,
                query_label=variant, **negative_kwargs)

    if native_reader:
        unavailable = None
        for entry_id, entry in entries:
            fact = {'entity': entry['entity'], 'attribute': entry['attribute'],
                    'value': entry['_value']}
            text = native_query_text(fact)
            query, reason = native_reader_query(model, tokenizer, text, layer=layer)
            if query is None:
                unavailable = reason
                break
            use_negatives = bool(negatives) and entry['role'] == 'correction_target'
            add(query, 'native_reader', _sha_text(text),
                target_entry_id=entry_id, expected='target', role=entry['role'],
                source_family=entry['source_family'],
                superseded_entry_id=world['correction']['supersedes']
                if entry['role'] == 'correction_target' else entry.get('superseded_entry_id'),
                query_label='native fast-port query, final prompt token',
                negative_entries=negatives if use_negatives else None,
                include_negatives=use_negatives)
        if unavailable is not None:
            add(None, 'native_reader', _sha_text('unavailable'), target_entry_id=None,
                expected='unsupported', role='native_reader', source_family='none',
                status='unavailable', unavailable_reason=unavailable,
                query_label='native fast-port query, final prompt token')
    else:
        add(None, 'native_reader', _sha_text('disabled'), target_entry_id=None,
            expected='unsupported', role='native_reader', source_family='none',
            status='unavailable', unavailable_reason='native_reader_disabled_by_protocol',
            query_label='native fast-port query disabled by protocol')

    if len(entries) >= 2:
        nominal_id, nominal = entries[0]
        owner_id, owner = entries[1]
        owner_key = torch.tensor(owner['_write_key'], dtype=torch.float32)
        owner_key = owner_key.view(1, 1, *owner_key.shape)
        add(owner_key, 'wrong_key', _sha_text('wrong-key:' + owner_id),
            target_entry_id=nominal_id, expected='key_owner', role='control',
            source_family='control', control_owner_entry_id=owner_id,
            query_label='write key from a different observed fact')
    else:
        add(None, 'wrong_key', _sha_text('wrong-key-unavailable'), target_entry_id=None,
            expected='key_owner', role='control', source_family='control',
            status='unavailable',
            unavailable_reason='at least two codebook entries are required')

    if entries:
        reference_key = torch.tensor(entries[0][1]['_write_key'], dtype=torch.float32)
        zero = torch.zeros(1, 1, *reference_key.shape)
        add(zero, 'zero', _sha_text('zero-query'), target_entry_id=None,
            expected='no_match', role='control', source_family='control',
            query_label='zero query')

    unknown_text = canonical_query_text(unknown, 'address')
    add(query_key(model, tokenizer, unknown_text, layer=layer), 'unknown',
        _sha_text(unknown_text), target_entry_id=None, expected='no_match',
        role='control', source_family='control', query_label='unobserved address')

    if foreign is not None:
        foreign_binding = observed_binding(model, tokenizer, foreign, layer=layer)
        add(foreign_binding.keys, 'cross_world', _sha_text(
            'cross-world:' + fact_source_text(_fact_subset(foreign))), target_entry_id=None,
            expected='no_match', role='control', source_family='control',
            query_label='observed write key from an independent world')
    else:
        add(None, 'cross_world', _sha_text('cross-world-unavailable'),
            target_entry_id=None, expected='no_match', role='control',
            source_family='control', status='unavailable',
            unavailable_reason='independent world is unavailable')

    if stage == 'current_view' and revoked_fact is not None:
        revoked_binding = observed_binding(model, tokenizer, revoked_fact, layer=layer)
        add(revoked_binding.keys, 'revoked_query', _sha_text(
            'revoked:' + fact_source_text(_fact_subset(revoked_fact))), target_entry_id=None,
            expected='no_match', role='revoked_query', source_family='control',
            revoked_entry_id=world['revocation_fact_id'],
            query_label='write key of a revoked source fact')
    elif stage == 'current_view':
        add(None, 'revoked_query', _sha_text('revoked-unavailable'),
            target_entry_id=None, expected='no_match', role='revoked_query',
            source_family='control', status='unavailable',
            unavailable_reason='revoked source fact is unavailable')
    rows.sort(key=lambda row: (row['phase'] or '', row['stage'], row['query_kind'],
                               row['target_entry_id'] or '', row['query_label'] or ''))
    return rows


def _expect_stale_load(image, *, model, tokenizer, fast_scope, binding, reference, scope):
    try:
        PersistentFastMemory.load(
            image, model, tokenizer, fast_scope,
            representation=FACT_REPRESENTATION,
            expected_view_digest=binding['view_digest'],
            require_source_binding=True, source_reference=reference,
            source_scope=scope)
    except ValueError as error:
        return 'stale' in str(error).lower()
    return False


def _rss_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == 'darwin' else value * 1024)


def _mps_memory():
    result = {'available': bool(torch.backends.mps.is_available())}
    for name in ('current_allocated_memory', 'driver_allocated_memory'):
        function = getattr(torch.mps, name, None)
        if function is None:
            result[name + '_bytes'] = None
            continue
        try:
            result[name + '_bytes'] = int(function())
        except Exception as error:  # The raw platform failure is evidence.
            result[name + '_error'] = f'{type(error).__name__}: {error}'
    return result


def _score_rows(rows, codebooks, thresholds):
    """Recompute every acceptance decision from raw readouts and codebook vectors."""
    computed = {}
    for index, row in enumerate(rows):
        if row.get('status') != 'ok' or row.get('readout') is None:
            computed[index] = {'accepted': False, 'top_entry_id': None,
                               'top_cosine': None, 'margin': None, 'scores': {}}
            continue
        codebook = codebooks.get(row['codebook_key'])
        if codebook is None:
            raise ValueError('Readout references an unknown codebook')
        if (row.get('world_id') != codebook['world_id']
                or row.get('stage') != codebook['stage']):
            raise ValueError('Readout world/stage differs from its source codebook')
        target = codebook['entries'].get(row.get('target_entry_id'))
        if row.get('query_kind') in {
                'observed_write_key', 'canonical_address', 'canonical_paraphrase', 'native_reader'}:
            if (target is None or row.get('role') != target['role']
                    or row.get('source_family') != target['source_family']
                    or row.get('expected') != 'target'):
                raise ValueError('Readout target role/source differs from source-derived metadata')
        readout = torch.tensor(row['readout'], dtype=torch.float32)
        negative_entries = codebook.get('negative_entries', {})
        include_negatives = bool(target is not None
                                 and target['role'] == 'correction_target' and negative_entries)
        scores = {entry_id: _cosine_heads(readout, entry['value_vector'])
                  for entry_id, entry in codebook['entries'].items()}
        if include_negatives:
            scores.update({entry_id: _cosine_heads(readout, entry['value_vector'])
                           for entry_id, entry in negative_entries.items()})
        ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        top_entry_id, top_cosine = (ordered[0] if ordered else (None, None))
        second = ordered[1][1] if len(ordered) > 1 else None
        margin = top_cosine - second if top_cosine is not None and second is not None else None
        norm = float(torch.linalg.vector_norm(readout))
        accepted = bool(top_entry_id is not None and norm > 1e-12
                        and top_cosine >= thresholds['match_cosine_floor']
                        and margin is not None
                        and margin >= thresholds['match_margin_floor'])
        computed[index] = {'accepted': accepted, 'top_entry_id': top_entry_id,
                           'top_cosine': top_cosine, 'margin': margin,
                           'scores': scores}
    return computed


def _metric(rows, computed, *, kinds, stage=None, expected=None, role=None,
            source_family=None):
    selected = []
    for index, row in enumerate(rows):
        if row.get('query_kind') not in kinds:
            continue
        if stage is not None and row.get('stage') != stage:
            continue
        if expected is not None and row.get('expected') != expected:
            continue
        if role is not None and row.get('role') != role:
            continue
        if source_family is not None and row.get('source_family') != source_family:
            continue
        selected.append((row, computed[index]))
    ok = [(row, value) for row, value in selected if row.get('status') == 'ok']
    accepted = [value for _, value in ok if value['accepted']]
    target_rows = [(row, value) for row, value in ok
                   if row.get('target_entry_id') is not None]
    top1 = sum(1 for row, value in target_rows
               if value['top_entry_id'] == row['target_entry_id'])
    recovered = sum(1 for row, value in target_rows
                    if value['accepted'] and value['top_entry_id'] == row['target_entry_id'])
    target_cosines = [value['scores'][row['target_entry_id']]
                      for row, value in target_rows
                      if row['target_entry_id'] in value['scores']]
    return {
        'n': len(selected),
        'ok': len(ok),
        'unavailable': len(selected) - len(ok),
        'accepted': len(accepted),
        'accept_rate': (len(accepted) / len(ok)) if ok else None,
        'target_n': len(target_rows),
        'top1': (top1 / len(target_rows)) if target_rows else None,
        'recovered_rate': (recovered / len(target_rows)) if target_rows else None,
        'mean_target_cosine': (sum(target_cosines) / len(target_cosines)
                               if target_cosines else None),
        'min_target_cosine': min(target_cosines) if target_cosines else None,
        'max_target_cosine': max(target_cosines) if target_cosines else None,
    }


def _wrong_key_metrics(rows, computed):
    selected = [(row, computed[index]) for index, row in enumerate(rows)
                if row.get('query_kind') == 'wrong_key' and row.get('status') == 'ok']
    if not selected:
        return {'n': 0, 'owner_rate': None, 'target_leak_rate': None}
    owner = sum(1 for row, value in selected
                if value['accepted'] and value['top_entry_id'] == row.get('control_owner_entry_id'))
    leak = sum(1 for row, value in selected
               if value['accepted'] and value['top_entry_id'] == row.get('target_entry_id'))
    return {'n': len(selected), 'owner_rate': owner / len(selected),
            'target_leak_rate': leak / len(selected)}


def _superseded_rate(rows, computed):
    selected = [(row, computed[index]) for index, row in enumerate(rows)
                if row.get('superseded_entry_id') is not None
                and row.get('status') == 'ok']
    if not selected:
        return {'n': 0, 'rate': None}
    hits = sum(1 for row, value in selected
               if value['accepted'] and value['top_entry_id'] == row['superseded_entry_id'])
    return {'n': len(selected), 'rate': hits / len(selected)}


def _revoked_metrics(rows, computed):
    selected = [(row, computed[index]) for index, row in enumerate(rows)
                if row.get('query_kind') == 'revoked_query' and row.get('status') == 'ok']
    if not selected:
        return {'n': 0, 'false_accept_rate': None}
    hits = sum(1 for _, value in selected if value['accepted'])
    return {'n': len(selected), 'false_accept_rate': hits / len(selected)}


def _gate(status, metrics, thresholds):
    return {'status': status, 'metrics': metrics, 'thresholds': thresholds}


def score_diagnostic(rows, codebooks, *, thresholds=None, mechanical):
    """Independently score raw rows; producer-supplied pass flags are ignored."""
    thresholds = _resolution_thresholds(thresholds)
    computed = _score_rows(rows, codebooks, thresholds)
    storage = _metric(rows, computed, kinds={'observed_write_key'}, expected='target')
    canonical = _metric(rows, computed, kinds={'canonical_address', 'canonical_paraphrase'},
                        expected='target')
    native = _metric(rows, computed, kinds={'native_reader'}, expected='target')
    zero = _metric(rows, computed, kinds={'zero'}, expected='no_match')
    unknown = _metric(rows, computed, kinds={'unknown'}, expected='no_match')
    cross_world = _metric(rows, computed, kinds={'cross_world'}, expected='no_match')
    wrong = _wrong_key_metrics(rows, computed)
    correction = _metric(rows, computed, kinds={'observed_write_key', 'canonical_address',
                                                'canonical_paraphrase'},
                         stage='corrected', expected='target', role='correction_target')
    unaffected = _metric(rows, computed, kinds={'observed_write_key', 'canonical_address',
                                                'canonical_paraphrase'},
                         stage='corrected', expected='target', role='source')
    superseded = _superseded_rate(rows, computed)
    revoked = _revoked_metrics(rows, computed)

    native_rows = [row for row in rows if row.get('query_kind') == 'native_reader']
    native_available = any(row.get('status') == 'ok' for row in native_rows)
    native_reason = next((row.get('unavailable_reason') for row in native_rows
                          if row.get('status') != 'ok'), None)

    frozen = mechanical.get('frozen', {})
    frozen_ok = (frozen.get('optimizer_updates') == 0
                 and frozen.get('parameter_versions_equal') is True
                 and frozen.get('parameters_updated') == []
                 and frozen.get('requires_grad_all_false') is True)
    storage_ok = bool(
        storage['target_n'] > 0 and storage['top1'] is not None
        and storage['top1'] >= thresholds['storage_top1_floor']
        and zero['n'] > 0 and zero['accept_rate'] == 0.0
        and wrong['n'] > 0 and wrong['target_leak_rate'] is not None
        and wrong['target_leak_rate'] <= thresholds['wrong_key_target_leak_max'])
    addressing_ok = bool(
        canonical['target_n'] > 0 and canonical['top1'] is not None
        and canonical['top1'] >= thresholds['addressing_top1_floor']
        and zero['accept_rate'] == 0.0
        and unknown['n'] > 0 and unknown['accept_rate'] is not None
        and unknown['accept_rate'] <= thresholds['unknown_false_accept_max']
        and cross_world['n'] > 0 and cross_world['accept_rate'] is not None
        and cross_world['accept_rate'] <= thresholds['cross_world_false_accept_max'])
    native_ok = bool(native_available and native['target_n'] > 0
                     and native['top1'] is not None
                     and native['top1'] >= thresholds['native_reader_top1_floor'])
    correction_ok = bool(
        correction['target_n'] > 0 and correction['recovered_rate'] is not None
        and correction['recovered_rate'] >= thresholds['correction_recovery_floor']
        and superseded['n'] > 0 and superseded['rate'] is not None
        and superseded['rate'] <= thresholds['superseded_selection_max']
        and unaffected['target_n'] > 0 and unaffected['recovered_rate'] is not None
        and unaffected['recovered_rate'] >= thresholds['unaffected_retention_floor']
        and mechanical.get('correction', {}).get('superseded_entry_absent') is True
        and mechanical.get('correction', {}).get('superseded_negative_present') is True)
    revocation_ok = bool(
        mechanical.get('revocation', {}).get('revoked_entry_absent') is True
        and revoked['n'] > 0 and revoked['false_accept_rate'] is not None
        and revoked['false_accept_rate'] <= thresholds['revocation_false_accept_max'])
    source = mechanical.get('source_binding', {})
    source_ok = bool(source.get('all_bound') is True
                     and source.get('stale_rejected') is True
                     and source.get('require_source_binding') is True)
    restart = mechanical.get('restart', {})
    restart_ok = bool(restart.get('all_state_equal') is True
                      and restart.get('all_readout_equal') is True)

    gates = {
        'frozen_weights': _gate('pass' if frozen_ok else 'fail', frozen, {
            'optimizer_updates': 0, 'parameters_updated': []}),
        'storage_recovery': _gate('pass' if storage_ok else 'fail', {
            'storage': storage, 'zero': zero, 'wrong_key': wrong}, {
            'storage_top1_floor': thresholds['storage_top1_floor'],
            'zero_false_accept_max': thresholds['zero_false_accept_max'],
            'wrong_key_target_leak_max': thresholds['wrong_key_target_leak_max']}),
        'addressing': _gate('pass' if addressing_ok else 'fail', {
            'canonical': canonical, 'zero': zero, 'unknown': unknown,
            'cross_world': cross_world}, {
            'addressing_top1_floor': thresholds['addressing_top1_floor'],
            'unknown_false_accept_max': thresholds['unknown_false_accept_max'],
            'cross_world_false_accept_max': thresholds['cross_world_false_accept_max']}),
        'native_readout': _gate(
            'pass' if native_ok else ('not_available' if not native_available else 'fail'),
            {'native': native, 'prerequisite': native_reason}, {
                'native_reader_top1_floor': thresholds['native_reader_top1_floor']}),
        'correction': _gate('pass' if correction_ok else 'fail', {
            'correction': correction, 'unaffected': unaffected,
            'superseded': superseded,
            'superseded_entry_absent': mechanical.get('correction', {}).get('superseded_entry_absent'),
            'superseded_negative_present': mechanical.get(
                'correction', {}).get('superseded_negative_present')},
            {'correction_recovery_floor': thresholds['correction_recovery_floor'],
             'unaffected_retention_floor': thresholds['unaffected_retention_floor'],
             'superseded_selection_max': thresholds['superseded_selection_max']}),
        'revocation': _gate('pass' if revocation_ok else 'fail', {
            'revoked': revoked,
            'revoked_entry_absent': mechanical.get('revocation', {}).get('revoked_entry_absent')},
            {'revocation_false_accept_max': thresholds['revocation_false_accept_max']}),
        'source_binding': _gate('pass' if source_ok else 'fail', source, {
            'require_source_binding': True, 'stale_rejected': True}),
        'restart': _gate('pass' if restart_ok else 'fail', restart, {
            'all_state_equal': True, 'all_readout_equal': True}),
    }

    def passed(name):
        return gates[name]['status'] == 'pass'

    if not passed('frozen_weights'):
        decision = 'invalid_diagnostic_weights_changed'
    elif not passed('storage_recovery'):
        decision = 'defer_useful_memory_storage_recovery_failed'
    elif not passed('addressing'):
        decision = 'defer_useful_memory_addressing_failed'
    elif not passed('correction'):
        decision = 'defer_useful_memory_correction_or_retention_failed'
    elif not passed('revocation'):
        decision = 'defer_useful_memory_revocation_failed'
    elif not (passed('source_binding') and passed('restart')):
        decision = 'defer_mechanical_source_or_restart_failed'
    elif not native_available:
        decision = 'defer_native_readout_unavailable'
    elif not passed('native_readout'):
        decision = 'defer_native_readout_failed'
    else:
        decision = 'diagnostic_supports_targeted_readout_experiment_only'

    failed = [name for name in ('frozen_weights', 'storage_recovery', 'addressing',
                                'correction', 'revocation', 'source_binding', 'restart')
              if not passed(name)]
    if not native_available or not passed('native_readout'):
        failed.append('native_readout')
    next_probe = _next_probe(failed, gates, thresholds, native_available)
    return {
        'thresholds': thresholds,
        'gates': gates,
        'decision': decision,
        'failed_gates': failed,
        'promote_v5_memory': False,
        'v5_status': 'rejected_by_prior_evidence',
        'neural_promotion': False,
        'deployment': 'diagnostic_only_not_published',
        'pointer_copy_comparator': (
            'not_constructed_storage_recovery_gate_blocked'
            if not passed('storage_recovery')
            else 'not_constructed_no_learned_readout_gate'),
        'targeted_training_justified': decision == 'diagnostic_supports_targeted_readout_experiment_only',
        'smallest_next_repair_discriminator': next_probe,
        'unrun_scope': [
            'full_blind_training_arm',
            'pointer_copy_comparator',
            'source_retrieval_answer_arm',
            'memory_only_reader_training',
            'native_generation_quality',
            'v5_memory_promotion',
            'neural_weight_update',
        ],
        'summary': {
            'storage_top1': storage['top1'],
            'storage_recovered_rate': storage['recovered_rate'],
            'canonical_top1': canonical['top1'],
            'native_top1': native['top1'],
            'unknown_false_accept_rate': unknown['accept_rate'],
            'wrong_key_target_leak_rate': wrong['target_leak_rate'],
            'correction_recovered_rate': correction['recovered_rate'],
            'unaffected_retention_rate': unaffected['recovered_rate'],
            'revocation_false_accept_rate': revoked['false_accept_rate'],
        },
    }


def _next_probe(failed, gates, thresholds, native_available):
    if not failed:
        return {'gate': None, 'observed': None, 'threshold': None,
                'action': 'No repair discriminator is implied; retain this as diagnostic evidence only.'}
    gate = failed[0]
    if gate == 'frozen_weights':
        return {'gate': gate, 'observed': gates[gate]['metrics'],
                'threshold': gates[gate]['thresholds'],
                'action': 'Stop: parameter versions or optimizer state changed; a weight update invalidates this diagnostic.'}
    if gate == 'storage_recovery':
        return {'gate': gate, 'observed': gates[gate]['metrics']['storage'],
                'threshold': gates[gate]['thresholds']['storage_top1_floor'],
                'action': 'Measure a one-fact, one-write state read with its exact observed key. If that fails, repair the writer erase/gate path; if it passes but multi-fact fails, measure decay/overwrite interference with two to three facts. Do not train a reader first.'}
    if gate == 'addressing':
        return {'gate': gate, 'observed': gates[gate]['metrics']['canonical'],
                'threshold': gates[gate]['thresholds']['addressing_top1_floor'],
                'action': 'Measure cosine between the exact observed write key and each canonical query key before touching state. If alignment is the failure, the next repair target is a query-side canonical projection, not another memory write.'}
    if gate == 'correction':
        return {'gate': gate, 'observed': gates[gate]['metrics'],
                'threshold': gates[gate]['thresholds'],
                'action': 'Compare the corrected entry read with its exact key against the superseded entry read. If the new value is not recovered, test whether one correction write overwrites the old key without erasing unaffected keys.'}
    if gate == 'revocation':
        return {'gate': gate, 'observed': gates[gate]['metrics'],
                'threshold': gates[gate]['thresholds'],
                'action': 'Re-encode the authoritative current view and verify the revoked key is absent before reading. If it is absent but still accepted, the reader is matching the wrong codebook entry.'}
    if gate == 'source_binding':
        return {'gate': gate, 'observed': gates[gate]['metrics'],
                'threshold': gates[gate]['thresholds'],
                'action': 'Re-run bind/save/load with the caller-owned EligibilityScope and a fresh ExactMemoryReference view; do not relax source authority.'}
    if gate == 'restart':
        return {'gate': gate, 'observed': gates[gate]['metrics'],
                'threshold': gates[gate]['thresholds'],
                'action': 'Compare saved/reloaded state digest and exact readout bytes before any new query; a difference is a persistence defect, not a memory-capability result.'}
    if gate == 'native_readout':
        if not native_available:
            return {'gate': gate, 'observed': None,
                    'threshold': gates[gate]['thresholds']['native_reader_top1_floor'],
                    'action': 'Provide a manifest with a declared fast port at the protocol layer; no native query was fabricated.'}
        return {'gate': gate, 'observed': gates[gate]['metrics']['native'],
                'threshold': gates[gate]['thresholds']['native_reader_top1_floor'],
                'action': 'Compare the raw native fast-port readout with the exact-key readout at the same layer. If exact-key passes and native fails, inspect the FastMemoryAccessPort query projection and its input hidden state; do not rewrite storage first.'}
    return {'gate': gate, 'observed': None, 'threshold': None,
            'action': 'Inspect the named failed gate before any new training arm.'}


def run_diagnostic(model, tokenizer, protocol, *, output_dir, ledger_dir=None,
                   worlds=None, device='cpu', writer_extension_receipt=None):
    """Run frozen storage/addressing/readout diagnosis and write raw artifacts."""
    supplied_worlds = worlds if worlds is not None else protocol.get('worlds')
    protocol = _merge_protocol(protocol)
    if supplied_worlds is None:
        worlds = generate_worlds(
            seed=protocol['seed'], world_count=protocol['world_count'],
            facts_per_world=protocol['facts_per_world'],
            distractors_per_world=protocol['distractors_per_world'],
            correction_index=protocol['correction_index'],
            revocation_index=protocol['revocation_index'])
    else:
        worlds = supplied_worlds
    validate_worlds(worlds)
    if len(worlds) != protocol['world_count']:
        raise ValueError('Protocol world count differs from supplied worlds')
    planned = _planned_workload_counts(
        world_count=protocol['world_count'],
        facts_per_world=protocol['facts_per_world'],
        distractors_per_world=protocol['distractors_per_world'],
        native_reader=protocol['native_reader'],
        correction_index=protocol['correction_index'],
        revocation_index=protocol['revocation_index'])
    layer = _resolve_layer(model, protocol)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ledger_dir = Path(ledger_dir) if ledger_dir is not None else output_dir / 'ledger'
    ledger_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / 'images'
    images_dir.mkdir(parents=True, exist_ok=True)
    if ((output_dir / 'readouts.jsonl').exists()
            or (output_dir / 'codebooks.json').exists()):
        raise FileExistsError('Diagnostic raw artifacts already exist')

    model.eval()
    requires_grad_before = {name: parameter.requires_grad
                            for name, parameter in model.named_parameters()}
    _freeze(model)
    versions_before = _parameter_versions(model)
    started = time.perf_counter()
    rss_start = _rss_bytes()
    rows, codebooks, public_codebooks = [], {}, {}
    summaries, transitions = [], []
    engine_config = producer = model_producer = dependencies = None
    world_family = {world['world_id']: world['family_id'] for world in worlds}

    with torch.no_grad():
        for world in worlds:
            world_started = time.perf_counter()
            ledger = RecordLedger(ledger_dir / f"{world['world_id']}.sqlite")
            reference = ExactMemoryReference(ledger)
            world_transition = {'world_id': world['world_id']}
            try:
                initial_records, correction_record, origin_by_record = world_records(world)
                superseded_record = next(
                    (record for record in initial_records
                     if record.record_id == world['correction']['supersedes']), None)
                if superseded_record is None:
                    raise ValueError('Correction target is absent from the observed source')
                admitted = reference.admit(initial_records)
                if admitted != len(initial_records):
                    raise ValueError('Source admission did not accept every initial record')
                fast_scope = _fast_scope(world, 0)
                memory = PersistentFastMemory(
                    model, tokenizer, fast_scope, training=False,
                    representation=FACT_REPRESENTATION)
                if engine_config is None:
                    engine_config = memory.config.to_dict()
                    producer = memory.producer
                    model_producer = memory.model_producer
                    dependencies = memory.dependencies
                for record in initial_records:
                    memory.append_record(record)

                initial_scope = _scope(world, 1)
                initial_binding = memory.bind_source(reference, scope=initial_scope)
                initial_image = images_dir / world['world_id'] / 'initial'
                initial_metadata = memory.save(initial_image, require_source_binding=True)

                initial_codebook = _build_codebook(
                    model, tokenizer, reference.view(initial_scope).records,
                    world_id=world['world_id'], stage='initial', layer=layer,
                    origin_by_record=origin_by_record,
                    correction_record_id=correction_record.record_id)
                unknown = unknown_fact(world, seed=protocol['seed'])
                foreign = foreign_fact(worlds, world)
                initial_rows_before = _evaluate_state(
                    memory, model, tokenizer, initial_codebook, stage='initial',
                    world=world, layer=layer, native_reader=protocol['native_reader'],
                    unknown=unknown, foreign=foreign, revoked_fact=None,
                    thresholds=protocol['thresholds'], phase='before_reload')

                loaded = PersistentFastMemory.load(
                    initial_image, model, tokenizer, fast_scope,
                    representation=FACT_REPRESENTATION,
                    expected_view_digest=initial_binding['view_digest'],
                    require_source_binding=True, source_reference=reference,
                    source_scope=initial_scope)
                initial_signature = _state_signature(memory)
                loaded_signature = _state_signature(loaded)
                initial_rows = _evaluate_state(
                    loaded, model, tokenizer, initial_codebook, stage='initial',
                    world=world, layer=layer, native_reader=protocol['native_reader'],
                    unknown=unknown, foreign=foreign, revoked_fact=None,
                    thresholds=protocol['thresholds'], phase='after_reload')
                initial_parity = _restart_readout_parity(initial_rows_before, initial_rows)
                initial_restart = {
                    'state_equal': initial_signature == loaded_signature,
                    'readout_equal': initial_parity['equal'],
                    'readout_parity': initial_parity,
                    'image': _display_path(initial_image),
                    'binding': initial_binding,
                }
                codebooks[initial_codebook['key']] = initial_codebook
                public_codebooks[initial_codebook['key']] = _public_codebook(initial_codebook)
                rows.extend(initial_rows_before)
                rows.extend(initial_rows)

                # Correction: admit the source event, prove a stale image is
                # rejected, append the correction, then save/load a fresh state.
                corrected_admission = reference.admit((correction_record,))
                if corrected_admission != 1:
                    raise ValueError('Correction admission did not accept exactly one record')
                stale_initial_rejected = _expect_stale_load(
                    initial_image, model=model, tokenizer=tokenizer, fast_scope=fast_scope,
                    binding=initial_binding, reference=reference, scope=initial_scope)
                memory = loaded
                memory.append_record(correction_record)
                corrected_scope = _scope(world, 2)
                corrected_binding = memory.bind_source(reference, scope=corrected_scope)
                corrected_image = images_dir / world['world_id'] / 'corrected'
                corrected_metadata = memory.save(corrected_image, require_source_binding=True)
                corrected_codebook = _build_codebook(
                    model, tokenizer, reference.view(corrected_scope).records,
                    world_id=world['world_id'], stage='corrected', layer=layer,
                    origin_by_record=origin_by_record,
                    correction_record_id=correction_record.record_id,
                    negative_record=superseded_record)
                superseded_entry_absent = (
                    world['correction']['supersedes'] not in corrected_codebook['entries'])
                superseded_negative_present = (
                    world['correction']['supersedes'] in corrected_codebook['negative_entries'])
                corrected_rows_before = _evaluate_state(
                    memory, model, tokenizer, corrected_codebook, stage='corrected',
                    world=world, layer=layer, native_reader=protocol['native_reader'],
                    unknown=unknown, foreign=foreign, revoked_fact=None,
                    thresholds=protocol['thresholds'], phase='before_reload')
                corrected_loaded = PersistentFastMemory.load(
                    corrected_image, model, tokenizer, fast_scope,
                    representation=FACT_REPRESENTATION,
                    expected_view_digest=corrected_binding['view_digest'],
                    require_source_binding=True, source_reference=reference,
                    source_scope=corrected_scope)
                corrected_rows = _evaluate_state(
                    corrected_loaded, model, tokenizer, corrected_codebook, stage='corrected',
                    world=world, layer=layer, native_reader=protocol['native_reader'],
                    unknown=unknown, foreign=foreign, revoked_fact=None,
                    thresholds=protocol['thresholds'], phase='after_reload')
                corrected_parity = _restart_readout_parity(corrected_rows_before,
                                                           corrected_rows)
                corrected_restart = {
                    'state_equal': _state_signature(memory) == _state_signature(corrected_loaded),
                    'readout_equal': corrected_parity['equal'],
                    'readout_parity': corrected_parity,
                    'image': _display_path(corrected_image),
                    'binding': corrected_binding,
                }
                codebooks[corrected_codebook['key']] = corrected_codebook
                public_codebooks[corrected_codebook['key']] = _public_codebook(corrected_codebook)
                rows.extend(corrected_rows_before)
                rows.extend(corrected_rows)

                # Revocation and an authoritative current-view rebuild.
                revoked_record = next(record for record in initial_records
                                      if record.record_id == world['revocation_fact_id'])
                revoked = reference.revoke(
                    revoked_record.record_id, authorized_subject=SUBJECT_ID,
                    authorized_branch=world['world_id'])
                if not revoked:
                    raise ValueError('Revocation did not change the ledger')
                stale_corrected_rejected = _expect_stale_load(
                    corrected_image, model=model, tokenizer=tokenizer,
                    fast_scope=fast_scope, binding=corrected_binding,
                    reference=reference, scope=corrected_scope)
                current_scope = _scope(world, 3)
                memory = PersistentFastMemory(
                    model, tokenizer, fast_scope, training=False,
                    representation=FACT_REPRESENTATION)
                memory.rebuild(semantics='current_view', reference=reference,
                               source_scope=current_scope)
                current_binding = memory.bind_source(reference, scope=current_scope)
                current_image = images_dir / world['world_id'] / 'current_view'
                current_metadata = memory.save(current_image, require_source_binding=True)
                current_codebook = _build_codebook(
                    model, tokenizer, reference.view(current_scope).records,
                    world_id=world['world_id'], stage='current_view', layer=layer,
                    origin_by_record=origin_by_record,
                    correction_record_id=correction_record.record_id,
                    negative_record=superseded_record)
                revoked_entry_absent = (
                    world['revocation_fact_id'] not in current_codebook['entries'])
                current_superseded_negative_present = (
                    world['correction']['supersedes'] in current_codebook['negative_entries'])
                revoked_fact = next(fact for fact in world['facts']
                                    if fact['fact_id'] == world['revocation_fact_id'])
                current_rows_before = _evaluate_state(
                    memory, model, tokenizer, current_codebook, stage='current_view',
                    world=world, layer=layer, native_reader=protocol['native_reader'],
                    unknown=unknown, foreign=foreign, revoked_fact=revoked_fact,
                    thresholds=protocol['thresholds'], phase='before_reload')
                current_loaded = PersistentFastMemory.load(
                    current_image, model, tokenizer, fast_scope,
                    representation=FACT_REPRESENTATION,
                    expected_view_digest=current_binding['view_digest'],
                    require_source_binding=True, source_reference=reference,
                    source_scope=current_scope)
                current_rows = _evaluate_state(
                    current_loaded, model, tokenizer, current_codebook, stage='current_view',
                    world=world, layer=layer, native_reader=protocol['native_reader'],
                    unknown=unknown, foreign=foreign, revoked_fact=revoked_fact,
                    thresholds=protocol['thresholds'], phase='after_reload')
                current_parity = _restart_readout_parity(current_rows_before,
                                                         current_rows)
                current_restart = {
                    'state_equal': _state_signature(memory) == _state_signature(current_loaded),
                    'readout_equal': current_parity['equal'],
                    'readout_parity': current_parity,
                    'image': _display_path(current_image),
                    'binding': current_binding,
                }
                codebooks[current_codebook['key']] = current_codebook
                public_codebooks[current_codebook['key']] = _public_codebook(current_codebook)
                rows.extend(current_rows_before)
                rows.extend(current_rows)

                world_transition.update({
                    'initial': initial_restart,
                    'corrected': corrected_restart,
                    'current_view': current_restart,
                    'stale_initial_rejected': stale_initial_rejected,
                    'stale_corrected_rejected': stale_corrected_rejected,
                    'superseded_entry_absent': superseded_entry_absent,
                    'superseded_negative_present': (
                        superseded_negative_present and current_superseded_negative_present),
                    'revoked_entry_absent': revoked_entry_absent,
                    'source_records_admitted': admitted,
                    'correction_records_admitted': corrected_admission,
                    'revoked_record_id': revoked_record.record_id,
                    'lineage_initial': len(initial_metadata['lineage']),
                    'lineage_corrected': len(corrected_metadata['lineage']),
                    'lineage_current_view': len(current_metadata['lineage']),
                    'semantics': memory.semantics,
                })
                transitions.append(world_transition)
                summaries.append({
                    'world_id': world['world_id'],
                    'family_id': world['family_id'],
                    'facts': len(world['facts']),
                    'distractors': len(world['distractors']),
                    'independent_world_distractors': len(world['distractors']),
                    'independent_world_ids': sorted(
                        {distractor['origin_world_id'] for distractor in world['distractors']}),
                    'source_family_ids': sorted(
                        {world_family[fact['source_world_id']] for fact in world['facts']}),
                    'source_family_count': len(
                        {world_family[fact['source_world_id']] for fact in world['facts']}),
                    'independent_world_family_ids': sorted(
                        {world_family[distractor['origin_world_id']]
                         for distractor in world['distractors']}),
                    'independent_world_family_count': len(
                        {world_family[distractor['origin_world_id']]
                         for distractor in world['distractors']}),
                    'family_count_total': len(
                        {world_family[fact['source_world_id']] for fact in world['facts']}
                        | {world_family[distractor['origin_world_id']]
                           for distractor in world['distractors']}),
                    'correction_target': world['correction']['fact_id'],
                    'revocation_target': world['revocation_fact_id'],
                    'codebook_entries': {stage: len(cb['entries'])
                                         for stage, cb in (('initial', initial_codebook),
                                                           ('corrected', corrected_codebook),
                                                           ('current_view', current_codebook))},
                    'source_hashes': {
                        stage: {entry_id: entry['source_hash']
                                for entry_id, entry in cb['entries'].items()}
                        for stage, cb in (('initial', initial_codebook),
                                          ('corrected', corrected_codebook),
                                          ('current_view', current_codebook))},
                    'negative_source_hashes': {
                        stage: {entry_id: entry['source_hash']
                                for entry_id, entry in cb.get('negative_entries', {}).items()}
                        for stage, cb in (('initial', initial_codebook),
                                          ('corrected', corrected_codebook),
                                          ('current_view', current_codebook))},
                    'evaluated_queries': {
                        stage: sum(1 for row in rows
                                   if row['world_id'] == world['world_id']
                                   and row['stage'] == stage)
                        for stage in ('initial', 'corrected', 'current_view')},
                    'diagnostic_queries': {
                        stage: sum(1 for row in rows
                                   if row['world_id'] == world['world_id']
                                   and row['stage'] == stage
                                   and row['phase'] == 'after_reload')
                        for stage in ('initial', 'corrected', 'current_view')},
                    'seconds': time.perf_counter() - world_started,
                    'image_bytes': {
                        stage: sum(path.stat().st_size for path in (images_dir / world['world_id'] / stage).rglob('*')
                                   if path.is_file())
                        for stage in ('initial', 'corrected', 'current_view')},
                })
            finally:
                reference.close()

    versions_after = _parameter_versions(model)
    requires_grad_after = {name: parameter.requires_grad
                           for name, parameter in model.named_parameters()}
    changed = [before[0] for before, after in zip(versions_before, versions_after)
               if before != after]
    parameter_versions_equal = versions_before == versions_after
    frozen = {
        'optimizer_updates': 0,
        'parameter_versions_equal': parameter_versions_equal,
        'parameters_updated': changed,
        'requires_grad_all_false': all(not value for value in requires_grad_after.values()),
        'requires_grad_before_all_false': all(not value for value in requires_grad_before.values()),
    }
    correction_mechanical = {
        'superseded_entry_absent': all(item['superseded_entry_absent'] for item in transitions),
        'superseded_negative_present': all(item['superseded_negative_present']
                                           for item in transitions),
    }
    revocation_mechanical = {
        'revoked_entry_absent': all(item['revoked_entry_absent'] for item in transitions),
    }
    source_binding = {
        'require_source_binding': True,
        'all_bound': all(item[stage]['binding'] is not None
                         for item in transitions
                         for stage in ('initial', 'corrected', 'current_view')),
        'stale_rejected': all(item['stale_initial_rejected']
                              and item['stale_corrected_rejected']
                              for item in transitions),
    }
    restart_details = [
        {'world_id': item['world_id'], 'stage': stage,
         **item[stage]['readout_parity']}
        for item in transitions
        for stage in ('initial', 'corrected', 'current_view')
    ]
    restart = {
        'all_state_equal': all(item[stage]['state_equal']
                               for item in transitions
                               for stage in ('initial', 'corrected', 'current_view')),
        'all_readout_equal': all(detail['equal'] for detail in restart_details),
        'details': restart_details,
        'evaluated_pairs': sum(detail['paired_rows'] for detail in restart_details),
    }
    mechanical = {
        'frozen': frozen,
        'correction': correction_mechanical,
        'revocation': revocation_mechanical,
        'source_binding': source_binding,
        'restart': restart,
    }
    diagnostic_rows = [row for row in rows if row['phase'] == 'after_reload']
    phase_counts = {phase: sum(1 for row in rows if row['phase'] == phase)
                    for phase in ('before_reload', 'after_reload')}
    unique_diagnostic_cases = len({_readout_identity(row)
                                   for row in diagnostic_rows})
    scored = score_diagnostic(
        diagnostic_rows, codebooks, thresholds=protocol['thresholds'],
        mechanical=mechanical)

    elapsed = time.perf_counter() - started
    readouts_path = output_dir / 'readouts.jsonl'
    _write_jsonl(readouts_path, rows)
    (output_dir / 'worlds.json').write_text(json.dumps(
        {'format': WORLD_FORMAT, 'seed': protocol['seed'], 'worlds': worlds},
        indent=2, sort_keys=True, allow_nan=False) + '\n')
    _write_json(output_dir / 'codebooks.json', {
        'format': 'replica.memory_storage_probe.codebooks.v1',
        'codebooks': public_codebooks,
    })
    world_families = {world['world_id']: world['family_id'] for world in worlds}
    independent_worlds = {distractor['origin_world_id']
                          for world in worlds for distractor in world['distractors']}
    report = {
        'format': DIAGNOSTIC_FORMAT,
        'status': 'completed',
        'mode': 'native_mps' if device == 'mps' else 'cpu_fixture',
        'device': device,
        'protocol_digest': fingerprint(protocol),
        'protocol_format': protocol.get('format'),
        'data_sha256': protocol.get('data', {}).get('sha256'),
        'seed': protocol['seed'],
        'layer': layer,
        'representation': FACT_REPRESENTATION,
        'representation_config': representation_configuration(FACT_REPRESENTATION),
        'readout_semantics': {
            'observed_write_key': 'normalized writer key per head -> committed SGDN-2 state einsum',
            'canonical_query': 'writer key path from an independently rendered query text -> same einsum',
            'native_reader': ('FastMemoryAccessPort query_norm/query_projection from the '
                              'final prompt hidden -> same einsum; output_projection is '
                              'not applied and no reader training is claimed'),
        },
        'engine_config': engine_config,
        'producer': producer,
        'model_producer': model_producer,
        'dependencies': dependencies,
        'source_lock': protocol.get('source'),
        'writer_extension': writer_extension_receipt,
        'optimizer_updates': 0,
        'parameters_updated': list(changed),
        'frozen': frozen,
        'worlds': summaries,
        'source_families': sorted({world['family_id'] for world in worlds}),
        'independent_world_families': sorted(
            {world_families[world_id] for world_id in independent_worlds}),
        'epochs': {
            item['world_id']: {
                stage: item[stage]['binding']['epoch']
                for stage in ('initial', 'corrected', 'current_view')}
            for item in transitions},
        'source_vs_memory': {
            'source_retrieval_arm': 'not_run',
            'memory_only_arm': 'measured_by_this_diagnostic',
            'exact_source_authority_used_to_build_codebook': True,
            'memory_only_summary': scored['summary'],
        },
        'transitions': transitions,
        'correction': correction_mechanical,
        'revocation': revocation_mechanical,
        'source_binding': source_binding,
        'restart': restart,
        'gates': scored['gates'],
        'decision': scored['decision'],
        'failed_gates': scored['failed_gates'],
        'targeted_training_justified': scored['targeted_training_justified'],
        'smallest_next_repair_discriminator': scored['smallest_next_repair_discriminator'],
        'pointer_copy_comparator': scored['pointer_copy_comparator'],
        'promote_v5_memory': False,
        'v5_status': scored['v5_status'],
        'neural_promotion': False,
        'deployment': scored['deployment'],
        'unrun_scope': scored['unrun_scope'],
        'summary': scored['summary'],
        'codebooks': public_codebooks,
        'readouts': rows,
        'readouts_sha256': _sha_file(readouts_path),
        'codebooks_sha256': _sha_file(output_dir / 'codebooks.json'),
        'workload': dict(
            planned,
            actual_rows=len(rows),
            actual_evaluated_rows=len(rows),
            actual_diagnostic_cases=len(diagnostic_rows),
            actual_before_reload_rows=phase_counts['before_reload'],
            actual_after_reload_rows=phase_counts['after_reload'],
            unique_after_reload_cases=unique_diagnostic_cases),
        'counts': {
            'worlds': len(worlds),
            'states': 3 * len(worlds),
            'rows': len(rows),
            'evaluated_rows': len(rows),
            'diagnostic_cases': len(diagnostic_rows),
            'before_reload_rows': phase_counts['before_reload'],
            'after_reload_rows': phase_counts['after_reload'],
            'unique_diagnostic_cases': unique_diagnostic_cases,
            'evaluated_target_queries': sum(
                1 for row in rows
                if row.get('expected') == 'target'
                and row.get('target_entry_id') is not None),
            'diagnostic_target_queries': sum(
                1 for row in diagnostic_rows
                if row.get('expected') == 'target'
                and row.get('target_entry_id') is not None),
            'evaluated_controls': sum(
                1 for row in rows
                if row.get('role') in {'control', 'revoked_query'}),
            'diagnostic_controls': sum(
                1 for row in diagnostic_rows
                if row.get('role') in {'control', 'revoked_query'}),
            'evaluated_source_family_rows': sum(
                1 for row in rows if row.get('source_family') == 'source'),
            'diagnostic_source_family_rows': sum(
                1 for row in diagnostic_rows if row.get('source_family') == 'source'),
            'evaluated_independent_world_rows': sum(
                1 for row in rows if row.get('source_family') == 'independent_world'),
            'diagnostic_independent_world_rows': sum(
                1 for row in diagnostic_rows
                if row.get('source_family') == 'independent_world'),
        },
        'resource': {
            'wall_seconds': elapsed,
            'max_rss_bytes': _rss_bytes(),
            'rss_start_bytes': rss_start,
            'mps': _mps_memory() if device == 'mps' else None,
            'state_bytes_per_world': None,
            'ledger_bytes': sum(path.stat().st_size for path in ledger_dir.glob('*.sqlite')),
            'image_bytes': sum(path.stat().st_size for path in images_dir.rglob('*')
                               if path.is_file()),
        },
    }
    if engine_config is not None:
        report['resource']['state_bytes_per_world'] = (
            4 * engine_config['heads'] * engine_config['key_size']
            * engine_config['value_size'])
    return report


def _implementation_sha256():
    result = {}
    for relative in _IMPLEMENTATION_PATHS:
        path = PROJECT_ROOT / relative
        if not path.is_file():
            raise ValueError(f'Implementation pin path is missing: {relative}')
        result[relative] = _sha_file(path)
    return result


def _source_pins(model_path, model_lock):
    model_path = _resolve_path(model_path)
    model_lock = _resolve_path(model_lock)
    if not model_path.exists():
        raise ValueError('Pinned model source does not exist')
    if not model_lock.is_file():
        raise ValueError('Pinned model lock does not exist')
    lock = json.loads(model_lock.read_text())
    if lock.get('revision') != SEALED_REVISION:
        raise ValueError('Pinned model revision differs from the sealed revision')
    snapshot = lock.get('snapshot_files_sha256')
    if not isinstance(snapshot, dict) or not snapshot:
        raise ValueError('Model lock has no exact snapshot-file pins')
    return {
        'model_path': _display_path(model_path),
        'model_lock': _display_path(model_lock),
        'model_lock_sha256': _sha_file(model_lock),
        'source_revision': lock['revision'],
        'required_revision': SEALED_REVISION,
        'source_seal_digest': fingerprint(snapshot),
        'config_sha256': lock.get('config_sha256'),
        'snapshot_files_sha256': snapshot,
        'snapshot_file_count': len(snapshot),
    }


def _extension_pins(directory):
    directory = _resolve_path(directory)
    weights = directory / 'extensions.safetensors'
    manifest_path = directory / 'manifest.json'
    if not weights.is_file() or not manifest_path.is_file():
        raise ValueError('Pinned writer extension lacks extensions.safetensors/manifest.json')
    receipt = json.loads(manifest_path.read_text())
    weights_sha256 = _sha_file(weights)
    if receipt.get('weights_sha256') != weights_sha256:
        raise ValueError('Writer extension manifest does not bind its weights')
    return {
        'path': _display_path(directory),
        'weights_sha256': weights_sha256,
        'manifest_sha256': _sha_file(manifest_path),
        'format': receipt.get('format'),
        'manifest_digest': receipt.get('manifest_digest'),
        'components': sorted((receipt.get('components') or {}).keys()),
    }


def prepare_probe(output, *, seed=DEFAULT_SEED, world_count=DEFAULT_WORLD_COUNT,
                  facts_per_world=DEFAULT_FACTS_PER_WORLD,
                  distractors_per_world=DEFAULT_DISTRACTORS_PER_WORLD,
                  correction_index=DEFAULT_CORRECTION_INDEX,
                  revocation_index=DEFAULT_REVOCATION_INDEX, layer=DEFAULT_LAYER,
                  model_path=PROJECT_ROOT / 'checkpoints/minicpm5_source',
                  model_lock=PROJECT_ROOT / 'configs/minicpm5_sealed_lock.local.json',
                  writer_extension=None):
    """Generate exact worlds, source pins and a native protocol; no model load."""
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Prepare output must be fresh or empty')
    if (type(world_count) is not int or world_count < 1
            or world_count > DEFAULT_LIMITS['max_worlds']):
        raise ValueError('World count exceeds the diagnostic limit')
    if (type(facts_per_world) is not int or facts_per_world < 2
            or facts_per_world > DEFAULT_LIMITS['max_facts_per_world']):
        raise ValueError('Fact count exceeds the diagnostic limit')
    if (type(distractors_per_world) is not int or distractors_per_world < 0
            or distractors_per_world > DEFAULT_LIMITS['max_distractors_per_world']):
        raise ValueError('Distractor count exceeds the diagnostic limit')
    planned = _planned_workload_counts(
        world_count=world_count, facts_per_world=facts_per_world,
        distractors_per_world=distractors_per_world, native_reader=True,
        correction_index=correction_index, revocation_index=revocation_index)
    if planned['total_planned_rows'] > DEFAULT_LIMITS['max_queries']:
        raise ValueError(
            f"Scheduled workload {planned['total_planned_rows']} rows exceeds "
            f"max_queries={DEFAULT_LIMITS['max_queries']}")
    output.mkdir(parents=True, exist_ok=True)
    worlds = generate_worlds(
        seed=seed, world_count=world_count, facts_per_world=facts_per_world,
        distractors_per_world=distractors_per_world,
        correction_index=correction_index, revocation_index=revocation_index)
    data = {'format': WORLD_FORMAT, 'seed': seed, 'world_count': world_count,
            'facts_per_world': facts_per_world,
            'distractors_per_world': distractors_per_world,
            'correction_index': correction_index,
            'revocation_index': revocation_index, 'worlds': worlds}
    data_path = output / 'data.json'
    _write_json(data_path, data)
    source = _source_pins(model_path, model_lock)
    extension = _extension_pins(writer_extension) if writer_extension is not None else None
    implementation = _implementation_sha256()
    protocol = {
        'format': PROTOCOL_FORMAT,
        'seed': seed,
        'world_count': world_count,
        'facts_per_world': facts_per_world,
        'distractors_per_world': distractors_per_world,
        'correction_index': correction_index,
        'revocation_index': revocation_index,
        'layer': layer,
        'representation': FACT_REPRESENTATION,
        'native_reader': True,
        'thresholds': _resolution_thresholds(None),
        'engine': dict(NATIVE_ENGINE),
        'limits': dict(DEFAULT_LIMITS),
        'planned_workload': planned,
        'source': source,
        'writer_extension': extension,
        'data': {'path': 'data.json', 'sha256': _sha_file(data_path),
                 'format': WORLD_FORMAT, 'seed': seed,
                 'world_count': world_count},
        'implementation_sha256': implementation,
    }
    protocol_path = output / 'protocol.json'
    _write_json(protocol_path, protocol)
    manifest = {
        'format': PREPARED_FORMAT,
        'created_at_utc': _utc_now(),
        'native_computation': False,
        'seed': seed,
        'world_count': world_count,
        'data_sha256': _sha_file(data_path),
        'protocol_sha256': _sha_file(protocol_path),
        'planned_workload': planned,
        'source': source,
        'writer_extension': extension,
        'implementation_sha256': implementation,
    }
    _write_json(output / 'manifest.json', manifest)
    return manifest


def _verify_prepared(prepared):
    prepared = Path(prepared)
    required = ('data.json', 'protocol.json', 'manifest.json')
    missing = [name for name in required if not (prepared / name).is_file()]
    if missing:
        raise ValueError(f'Prepared probe is missing {missing}')
    manifest_path = prepared / 'manifest.json'
    data_path = prepared / 'data.json'
    protocol_path = prepared / 'protocol.json'
    manifest = json.loads(manifest_path.read_text())
    protocol = json.loads(protocol_path.read_text())
    if protocol.get('format') == 'replica.memory_one_write_probe.protocol.v1':
        from .memory_one_write import verify_one_write_prepared
        return verify_one_write_prepared(prepared)
    data = json.loads(data_path.read_text())
    if manifest.get('format') != PREPARED_FORMAT:
        raise ValueError('Prepared manifest format differs')
    if manifest.get('native_computation') is not False:
        raise ValueError('Preparation must not perform native computation')
    if manifest.get('protocol_sha256') != _sha_file(protocol_path):
        raise ValueError('Protocol bytes differ from the prepared manifest')
    if manifest.get('data_sha256') != _sha_file(data_path):
        raise ValueError('Data bytes differ from the prepared manifest')
    if protocol.get('format') != PROTOCOL_FORMAT or data.get('format') != WORLD_FORMAT:
        raise ValueError('Protocol or data format differs')
    if protocol.get('data', {}).get('path') != 'data.json':
        raise ValueError('Protocol data path must be exactly data.json')
    if protocol['data'].get('sha256') != _sha_file(data_path):
        raise ValueError('Protocol data pin differs')
    normalized = _merge_protocol(protocol)
    for name in ('seed', 'world_count', 'facts_per_world', 'distractors_per_world',
                 'correction_index', 'revocation_index'):
        if data.get(name) != normalized[name]:
            raise ValueError(f'Data/protocol {name} differs')
    if data.get('seed') != normalized['seed']:
        raise ValueError('Data seed differs')
    planned = _planned_workload_counts(
        world_count=normalized['world_count'],
        facts_per_world=normalized['facts_per_world'],
        distractors_per_world=normalized['distractors_per_world'],
        native_reader=normalized['native_reader'],
        correction_index=normalized['correction_index'],
        revocation_index=normalized['revocation_index'])
    if protocol.get('planned_workload') != planned:
        raise ValueError('Prepared workload differs from the exact plan')
    if manifest.get('planned_workload') != planned:
        raise ValueError('Manifest workload differs from the exact plan')
    worlds = data.get('worlds')
    validate_worlds(worlds)
    regenerated = generate_worlds(
        seed=normalized['seed'], world_count=normalized['world_count'],
        facts_per_world=normalized['facts_per_world'],
        distractors_per_world=normalized['distractors_per_world'],
        correction_index=normalized['correction_index'],
        revocation_index=normalized['revocation_index'])
    if regenerated != worlds:
        raise ValueError('Prepared worlds differ from the seeded generator')
    if type(normalized['layer']) is not int:
        raise ValueError('Native protocol must pin an integer injection layer')
    source = normalized.get('source')
    if not isinstance(source, dict) or set(source) != {
            'model_path', 'model_lock', 'model_lock_sha256', 'source_revision',
            'required_revision', 'source_seal_digest', 'config_sha256',
            'snapshot_files_sha256', 'snapshot_file_count'}:
        raise ValueError('Protocol source pins differ')
    current_source = _source_pins(source['model_path'], source['model_lock'])
    if current_source != source:
        raise ValueError('Source pins changed')
    if source['model_lock_sha256'] != manifest['source']['model_lock_sha256']:
        raise ValueError('Manifest/protocol source pins differ')
    current_impl = _implementation_sha256()
    if normalized.get('implementation_sha256') != current_impl:
        raise ValueError('Runtime implementation changed since preparation')
    if manifest.get('implementation_sha256') != current_impl:
        raise ValueError('Manifest implementation pins changed')
    extension = normalized.get('writer_extension')
    if extension is not None:
        current_extension = _extension_pins(extension['path'])
        if current_extension != extension:
            raise ValueError('Writer extension pins changed')
        if manifest.get('writer_extension') != extension:
            raise ValueError('Manifest/protocol writer extension pins differ')
    elif manifest.get('writer_extension') is not None:
        raise ValueError('Manifest pins a writer extension that protocol does not')
    receipt = {
        'format': ADMISSION_FORMAT,
        'status': 'admitted',
        'created_at_utc': _utc_now(),
        'prepared_manifest_sha256': _sha_file(manifest_path),
        'protocol_sha256': _sha_file(protocol_path),
        'data_sha256': _sha_file(data_path),
        'source': source,
        'source_lock_sha256': source['model_lock_sha256'],
        'writer_extension': extension,
        'implementation_sha256': current_impl,
        'seed': normalized['seed'],
        'world_count': normalized['world_count'],
        'facts_per_world': normalized['facts_per_world'],
        'distractors_per_world': normalized['distractors_per_world'],
        'planned_workload': planned,
        'checks': {
            'manifest_format': True,
            'protocol_bytes': True,
            'data_bytes': True,
            'seeded_world_regeneration': True,
            'workload_bound': True,
            'source_pins': True,
            'implementation_pins': True,
            'writer_extension_pins': True,
            'no_native_computation': True,
        },
    }
    return receipt


def verify_prepared(prepared):
    """Public pin verification used by the guarded native runner."""
    return _verify_prepared(prepared)


def admit_probe(prepared, *, write_receipt=True):
    """Verify and admit a prepared probe; rejected evidence is preserved."""
    prepared = Path(prepared)
    try:
        receipt = _verify_prepared(prepared)
    except Exception as error:
        if write_receipt and prepared.is_dir():
            _write_json(prepared / 'admission.json', {
                'format': ADMISSION_FORMAT,
                'status': 'rejected',
                'created_at_utc': _utc_now(),
                'error_type': type(error).__name__,
                'error': str(error),
            })
        raise
    if write_receipt:
        _write_json(prepared / 'admission.json', receipt)
    return receipt

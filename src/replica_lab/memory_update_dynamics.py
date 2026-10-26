"""Frozen update-dynamics discriminator for exact-key last-write dominance.

The matched-episode diagnostic established that exact observed-key value
recovery fails first while query-derived addressing passes, and that the final
write is almost always the top candidate.  This module holds the episodes,
candidate identities, observation order (except the declared reversed control)
and scoring definitions fixed while varying only how the frozen writer's
per-depth proposals become a state:

* ``shared_mean`` — the production convex mean of every injection depth;
* ``per_depth`` — each depth's proposal updates only its own state matrix.

Write-position ablations (target only, prefix through the target, target plus
later records, reversed within observation-day groups) and recency / erase /
decay controls are evaluator-side interventions on observed records.  They do
not train, update or replace the rejected writer.  Raw readouts, state
inventories, codebooks and rankings are preserved before any interpretation.

Exact observed-key recovery is a storage upper bound, not addressing.  The
query-derived key path is scored separately.  No optimizer is constructed.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from itertools import groupby
import time

import torch
import torch.nn.functional as F

from .memory_diagnosis import (
    MODE,
    PROJECT_ROOT,
    PURPOSE,
    SUBJECT_ID,
    _cosine_heads,
    _display_path,
    _extension_pins,
    _freeze,
    _mps_memory,
    _parameter_versions,
    _resolve_path,
    _rss_bytes,
    _sha_file,
    _sha_text,
    _source_pins,
    _utc_now,
    _write_json,
    _write_jsonl,
    observed_binding,
    query_key,
)
from .memory_matched import (
    _fact_of,
    _record_views,
    derive_matched_labels,
    episode_states,
)
from .neural import SGDN2Memory
from .runtime.fast_memory import FastMemoryScope
from .runtime.memory_representation import FACT_REPRESENTATION
from .runtime.persistent_fast_memory import PersistentFastMemory

PROTOCOL_FORMAT = 'replica.memory_update_dynamics.protocol.v1'
CONFIG_FORMAT = 'replica.memory_update_dynamics.config.v1'
PREPARED_FORMAT = 'replica.memory_update_dynamics.prepared.v1'
ADMISSION_FORMAT = 'replica.memory_update_dynamics.admission.v1'
RUN_FORMAT = 'replica.memory_update_dynamics.run.v1'
READOUT_FORMAT = 'replica.memory_update_dynamics.readout.v1'
STATE_FORMAT = 'replica.memory_update_dynamics.state.v1'
CODEBOOK_FORMAT = 'replica.memory_update_dynamics.codebook.v1'
RANKING_FORMAT = 'replica.memory_update_dynamics.ranking.v1'
INVENTORY_FORMAT = 'replica.memory_update_dynamics.inventory.v1'
MECHANICAL_FORMAT = 'replica.memory_update_dynamics.mechanical.v1'
FAILURES_FORMAT = 'replica.memory_update_dynamics.failures.v1'

REDUCERS = ('shared_mean', 'per_depth')
WRITE_ORDERS = ('full', 'reversed', 'target_only', 'prefix_through_target',
                'target_and_later')
GATE_CONTROLS = ('forced_full_erase', 'forced_full_decay',
                  'forced_full_gate', 'forced_full_forget')
READ_ARMS = ('exact_key_value', 'query_key_value', 'query_key_address',
             'context_removed')
POSITION_LABELS = ('target_only', 'prefix_through_target', 'target_and_later',
                   'reversed')

DEFAULT_LAYERS = (13, 27)
DEFAULT_ENGINE = {'heads': 4, 'key_size': 64, 'value_size': 64,
                  'update_rule': 'sgdn2', 'depth_mode': 'shared',
                  'depth_writers': True}
DEFAULT_THRESHOLDS = {
    'match_cosine_floor': 0.5,
    'match_margin_floor': 0.05,
    'address_cosine_floor': 0.5,
    'address_margin_floor': 0.05,
    'exact_key_top1_floor': 0.85,
    'exact_key_mean_rank_max': 1.2,
    'exact_key_margin_floor': 0.02,
    'exact_key_false_recall_max': 0.05,
    'address_top1_floor': 0.6,
    'address_mean_rank_max': 1.5,
    'address_margin_mean_floor': 0.01,
    'address_false_recall_max': 0.3,
    'correction_top1_floor': 0.6,
    'unaffected_top1_floor': 0.7,
    'unknown_false_recall_max': 0.1,
    'mechanism_top1_delta': 0.25,
    'mechanism_rank_delta': 0.5,
    'mechanism_margin_delta': 0.02,
    'production_parity_atol': 1e-6,
}
DEFAULT_LIMITS = {
    'max_rows': 24000,
    'max_episodes': 64,
    'max_states': 160,
    'max_state_builds': 4096,
    'max_queries': 24000,
    'max_layers': 2,
    'max_preserved_failures': 1000,
    'native_jobs': 1,
    'driver_gib': 50,
    'minimum_free_percent': 15,
}

_IMPLEMENTATION_PATHS = (
    'src/replica_lab/memory_update_dynamics.py',
    'src/replica_lab/memory_matched.py',
    'src/replica_lab/memory_diagnosis.py',
    'src/replica_lab/memory_reliability.py',
    'src/replica_lab/runtime/persistent_fast_memory.py',
    'src/replica_lab/runtime/fast_memory.py',
    'src/replica_lab/runtime/memory_representation.py',
    'src/replica_lab/runtime/fact_memory.py',
    'src/replica_lab/neural.py',
    'src/replica_lab/architecture/contracts.py',
    'src/replica_lab/architecture/minicpm5.py',
    'src/replica_lab/architecture/ports.py',
    'src/replica_lab/architecture/loading.py',
    'src/replica_lab/model_adapter.py',
    'src/replica_lab/profiling.py',
    'src/replica_lab/contracts.py',
    'src/replica_lab/capacity_guard.py',
    'scripts/prepare_memory_update_dynamics.py',
    'scripts/run_memory_update_dynamics.py',
    'configs/memory_update_dynamics_v1.json',
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _sha_json(value):
    return _sha_text(json.dumps(value, sort_keys=True, allow_nan=False))


def _mean(values):
    values = [value for value in values if value is not None]
    return (sum(values) / len(values)) if values else None


def _rate(count, total):
    return (count / total) if total else None


# ---------------------------------------------------------------------------
# Variant plan: only update dynamics vary across variants
# ---------------------------------------------------------------------------

def _variant(reducer, label, *, order='full', gate_control=None,
             refresh_target=False, target_dependent=False, description=''):
    _require(reducer in REDUCERS, 'Unknown reducer')
    _require(order in WRITE_ORDERS, 'Unknown write order')
    _require(gate_control is None or gate_control in GATE_CONTROLS,
             'Unknown gate control')
    return {
        'name': f'{reducer}/{label}',
        'reducer': reducer,
        'label': label,
        'order': order,
        'gate_control': gate_control,
        'refresh_target': bool(refresh_target),
        'target_dependent': bool(target_dependent),
        'description': description,
    }


def variant_plan():
    """Return the deterministic, ordered update-dynamics variant inventory."""
    variants = []
    for reducer in REDUCERS:
        variants.append(_variant(
            reducer, 'full',
            description=('Production order, every eligible observed record '
                         'written once; only the reduction differs.')))
    for reducer in REDUCERS:
        variants.append(_variant(
            reducer, 'reversed', order='reversed',
            description=('Same record multiset as full, reversed inside each '
                         'observation-day group so corrections remain last.')))
    for reducer in REDUCERS:
        for order in ('target_only', 'prefix_through_target',
                      'target_and_later'):
            variants.append(_variant(
                reducer, order, order=order, target_dependent=True,
                description={
                    'target_only': 'Only the target observation is written.',
                    'prefix_through_target': ('Every record through the target; '
                                              'no later writes.'),
                    'target_and_later': ('The target plus every later record; '
                                         'earlier writes removed.'),
                }[order]))
    for reducer in REDUCERS:
        for gate in GATE_CONTROLS:
            variants.append(_variant(
                reducer, gate, gate_control=gate,
                description={
                    'forced_full_erase': ('Production order with the erase '
                                          'factor forced to one (full erase, '
                                          'not literal erasure removal).'),
                    'forced_full_decay': ('Production order with decay forced '
                                          'to one (no per-token decay, not '
                                          'literal decay removal).'),
                    'forced_full_gate': ('Production order with the write gate '
                                         'forced to one (maximum write, not '
                                         'literal gate removal).'),
                    'forced_full_forget': ('Production order with erase and '
                                           'decay both forced to one '
                                           '(full erase/no decay, not literal '
                                           'forgetting removal).'),
                }[gate]))
    for reducer in REDUCERS:
        variants.append(_variant(
            reducer, 'target_refresh_last', refresh_target=True,
            target_dependent=True,
            description=('Full production history, then the target observation '
                         'is written once more as the final write.')))
    names = [variant['name'] for variant in variants]
    _require(len(names) == len(set(names)), 'Duplicate update-dynamics variant')
    return variants


def variant_names():
    return [variant['name'] for variant in variant_plan()]


def variant_by_name(name):
    for variant in variant_plan():
        if variant['name'] == name:
            return variant
    raise ValueError(f'Unknown update-dynamics variant: {name}')


def resolve_variants(names=None):
    plan = variant_plan()
    if names is None:
        return plan
    known = {variant['name']: variant for variant in plan}
    _require(len(set(names)) == len(names), 'Duplicate requested variant')
    missing = [name for name in names if name not in known]
    _require(not missing, f'Unknown requested variants: {missing[:3]}')
    return [known[name] for name in names]


def variant_applies(variant, kind):
    return (kind == 'fact') if variant['target_dependent'] else True


def arm_applies(arm, kind):
    return (kind == 'fact') if arm == 'exact_key_value' else True


# ---------------------------------------------------------------------------
# Deterministic inventory / workload (no model)
# ---------------------------------------------------------------------------

def _iter_state_labels(episodes, labels_by_episode):
    for episode in episodes:
        states = labels_by_episode.get(episode['episode_id'])
        _require(isinstance(states, dict), 'Episode lacks verifier labels')
        for state in episode_states(episode):
            _require(state['name'] in states,
                     'Episode state lacks verifier labels')
            yield episode, state, states[state['name']]


def planned_inventory(episodes, labels_by_episode, *, variants=None,
                      layers=None):
    """Exact expected row identities, independent of any model allocation."""
    variants = variants or variant_plan()
    layers = tuple(layers or DEFAULT_LAYERS)
    rows = []
    for episode, state, labels in _iter_state_labels(episodes, labels_by_episode):
        for query in episode['queries']:
            query_id = query['query_id']
            _require(query_id in labels['targets'], 'Query lacks derived target')
            kind = labels['targets'][query_id]['kind']
            for variant in variants:
                if not variant_applies(variant, kind):
                    continue
                for layer in layers:
                    for arm in READ_ARMS:
                        if not arm_applies(arm, kind):
                            continue
                        rows.append((variant['name'], arm,
                                     episode['episode_id'], state['name'],
                                     query_id, layer))
    _require(len(rows) == len(set(rows)), 'Planned inventory has duplicate rows')
    return sorted(rows)


def _planned_state_builds(episodes, labels_by_episode, variants):
    builds = 0
    unique_records = set()
    for episode, state, labels in _iter_state_labels(episodes, labels_by_episode):
        records, _, _ = _record_views(episode, state)
        unique_records.update((episode['episode_id'], record['record_id'])
                              for record in records)
        for variant in variants:
            if variant['target_dependent']:
                builds += sum(
                    1 for query in episode['queries']
                    if labels['targets'][query['query_id']]['kind'] == 'fact')
            else:
                builds += 1
    return builds, unique_records


def planned_workload(episodes, labels_by_episode, *, variants=None,
                     protocol=None):
    variants = variants or variant_plan()
    limits = _resolve_limits((protocol or {}).get('limits'))
    layers = tuple((protocol or {}).get('layers') or DEFAULT_LAYERS)
    rows = planned_inventory(episodes, labels_by_episode, variants=variants,
                             layers=layers)
    builds, unique_records = _planned_state_builds(
        episodes, labels_by_episode, variants)
    workload = {
        'episodes': len(episodes),
        'states': sum(len(episode_states(episode)) for episode in episodes),
        'queries': sum(len(episode['queries']) for episode in episodes),
        'variants': len(variants),
        'arms': list(READ_ARMS),
        'layers': list(layers),
        'unique_observations': len(unique_records),
        'state_builds': builds,
        'readout_rows': len(rows),
        'planned_queries': len(rows),
        'planned_write_encodes': len(unique_records) * len(layers),
    }
    _require(len(episodes) <= limits['max_episodes'],
             'Episode count exceeds the declared bound')
    _require(workload['states'] <= limits['max_states'],
             'State count exceeds the declared bound')
    _require(workload['state_builds'] <= limits['max_state_builds'],
             'State-build count exceeds the declared bound')
    _require(len(layers) <= limits['max_layers'],
             'Declared layer count exceeds the bound')
    _require(workload['readout_rows'] <= limits['max_rows'],
             'Planned readout rows exceed max_rows')
    _require(workload['planned_queries'] <= limits['max_queries'],
             'Planned query rows exceed max_queries')
    return workload


def _resolve_thresholds(thresholds=None):
    result = dict(DEFAULT_THRESHOLDS)
    result.update(thresholds or {})
    extra = set(result) - set(DEFAULT_THRESHOLDS)
    _require(not extra, f'Unknown update-dynamics thresholds: {sorted(extra)}')
    return result


def _resolve_limits(limits=None):
    result = dict(DEFAULT_LIMITS)
    result.update(limits or {})
    extra = set(result) - set(DEFAULT_LIMITS)
    _require(not extra, f'Unknown update-dynamics limits: {sorted(extra)}')
    return result


# ---------------------------------------------------------------------------
# State histories and update dynamics
# ---------------------------------------------------------------------------

def _target_record(records, target_id):
    matches = [record for record in records
               if record['record_id'] == target_id]
    _require(len(matches) == 1, 'Target observation is absent or duplicated')
    return matches[0]


def _reverse_groups(records):
    result = []
    for _, group in groupby(records, key=lambda record: record['observed_at']):
        result.extend(reversed(list(group)))
    return result


def variant_history(records, variant, *, target_id=None):
    """Return the exact ordered observed records a variant consumes."""
    records = list(records)
    order = variant['order']
    if order == 'full':
        history = records
    elif order == 'reversed':
        history = _reverse_groups(records)
    else:
        _require(variant['target_dependent'],
                 'Position ablation must be declared target-dependent')
        _require(target_id is not None, 'Position ablation requires a target')
        target = _target_record(records, target_id)
        index = records.index(target)
        if order == 'target_only':
            history = [target]
        elif order == 'prefix_through_target':
            history = records[:index + 1]
        elif order == 'target_and_later':
            history = [target] + records[index + 1:]
        else:  # pragma: no cover - variant validation excludes this
            raise ValueError(f'Unsupported write order: {order}')
    if variant['refresh_target']:
        _require(target_id is not None, 'Target refresh requires a target')
        history = list(history) + [_target_record(records, target_id)]
    return history


def _gate_control_write(write, control, chunk_index):
    fields = {'chunk_index': chunk_index}
    if control in ('forced_full_erase', 'forced_full_forget'):
        fields['erase'] = torch.ones_like(write.erase)
    if control in ('forced_full_decay', 'forced_full_forget'):
        fields['decay'] = torch.ones_like(write.decay)
    if control == 'forced_full_gate':
        fields['write_gate'] = torch.ones_like(write.write_gate)
    return replace(write, **fields)


def _build_state(primitive, layers, history_writes, *, reducer, gate_control,
                 device):
    _require(history_writes, 'A state requires at least one write')
    if reducer == 'shared_mean':
        state = primitive.initial_state(1, device=device)
        for writes in history_writes:
            proposals = {
                layer: (_gate_control_write(writes[layer], gate_control,
                                            state.chunk_index),)
                for layer in layers
            }
            state = primitive.reduce_shared_chunk(state, proposals)
        return {'shared': state}
    states = {layer: primitive.initial_state(1, device=device)
              for layer in layers}
    for writes in history_writes:
        for layer in layers:
            states[layer] = primitive.update_shared_chunk(
                states[layer],
                (_gate_control_write(writes[layer], gate_control,
                                     states[layer].chunk_index),))
    return states


def _state_weights(states, reducer, layer):
    return (states['shared'].weights if reducer == 'shared_mean'
            else states[layer].weights)


def _read_weights(weights, query):
    _require(query.ndim == 4 and query.shape[0] == 1 and query.shape[1] == 1,
             'Query must be [1, 1, heads, key]')
    _require(tuple(query.shape[2:]) == tuple(weights.shape[1:3]),
             'Query heads/width differ from the state')
    normalized = F.normalize(query.to(device=weights.device,
                                      dtype=weights.dtype), dim=-1, eps=1e-6)
    return torch.einsum('bthk,bhkv->bthv', normalized, weights)[0, 0]


def _state_sha(weights):
    return hashlib.sha256(weights.detach().float().cpu().contiguous()
                          .numpy().tobytes()).hexdigest()


def _observation_sha(episode, state):
    return _sha_json([
        {'page_id': page['page_id'], 'observed_at': page['observed_at'],
         'date': page['date'], 'lines': list(page['lines'])}
        for page in state['observations']])


def _memory_record(episode, record):
    return {
        'record_id': record['record_id'], 'entity': record['entity'],
        'attribute': record['attribute'], 'value': record['value'],
        'observed_at': record['observed_at'], 'kind': record['kind'],
        'supersedes': record['supersedes'],
    }


def _production_memory(model, tokenizer, episode, state, records):
    scope = FastMemoryScope(SUBJECT_ID, episode['world_id'], PURPOSE, MODE, 0)
    memory = PersistentFastMemory(model, tokenizer, scope, training=False,
                                  representation=FACT_REPRESENTATION)
    for record in records:
        memory.append_fact(record['record_id'], _fact_of(record),
                           correction=record['kind'] == 'source_correction')
    return memory


def _state_context(model, tokenizer, episode, state, layers):
    records, live, negative = _record_views(episode, state)
    writes, entry_vectors = {}, {}
    for record in records:
        fact = _fact_of(record)
        for layer in layers:
            write = observed_binding(model, tokenizer, fact, layer=layer)
            key = write.keys[0, 0].detach().float().cpu()
            value = write.values[0, 0].detach().float().cpu()
            _require(bool(torch.isfinite(key).all())
                     and bool(torch.isfinite(value).all()),
                     'Writer produced a non-finite key/value vector')
            writes[(record['record_id'], layer)] = write
            entry_vectors[(record['record_id'], layer)] = {
                'write_key': key.tolist(),
                'value_vector': value.tolist(),
                'write_key_sha256': _sha_json(key.tolist()),
                'value_sha256': _sha_json(value.tolist()),
                'key_norm': float(torch.linalg.vector_norm(key)),
                'value_norm': float(torch.linalg.vector_norm(value)),
            }
    query_vectors = {}
    for query in episode['queries']:
        for layer in layers:
            query_vectors[(query['query_id'], layer)] = query_key(
                model, tokenizer, query['text'], layer=layer)
    return {
        'episode': episode, 'state': state,
        'records': list(records),
        'live_ids': [record['record_id'] for record in live],
        'negative_ids': [record['record_id'] for record in negative],
        'writes': writes,
        'entry_vectors': entry_vectors,
        'query_vectors': query_vectors,
        'observation_sha256': _observation_sha(episode, state),
    }


def _codebook(context, layers):
    entries = {}
    for layer in layers:
        entries[str(layer)] = {
            record_id: dict(vector)
            for (record_id, entry_layer), vector in context['entry_vectors'].items()
            if entry_layer == layer
        }
    return {
        'format': CODEBOOK_FORMAT,
        'episode_id': context['episode']['episode_id'],
        'world_id': context['episode']['world_id'],
        'state': context['state']['name'],
        'cutoff': context['state']['cutoff'],
        'observation_sha256': context['observation_sha256'],
        'layers': entries,
        'live_entry_ids': sorted(context['live_ids']),
        'negative_entry_ids': sorted(context['negative_ids']),
    }


# ---------------------------------------------------------------------------
# Raw rows
# ---------------------------------------------------------------------------

def _readout_row(*, variant, arm, context, query, layer, readout,
                 readout_space, final_write_id, state_sha256):
    episode, state = context['episode'], context['state']
    readout = torch.as_tensor(readout, dtype=torch.float32)
    _require(readout.ndim == 2, 'Readout must be [heads, width]')
    _require(bool(torch.isfinite(readout).all()), 'Readout is not finite')
    return {
        'format': READOUT_FORMAT,
        'variant': variant['name'], 'reducer': variant['reducer'],
        'arm': arm, 'readout_space': readout_space,
        'episode_id': episode['episode_id'], 'world_id': episode['world_id'],
        'state': state['name'], 'cutoff': state['cutoff'],
        'case_id': query['query_id'], 'query_id': query['query_id'],
        'query_text_sha256': _sha_json(query['text']),
        'codebook_key': f"{episode['episode_id']}/{state['name']}",
        'layer': layer,
        'observation_sha256': context['observation_sha256'],
        'state_sha256': state_sha256,
        'final_write_id': final_write_id,
        'status': 'ok',
        'readout': readout.tolist(),
        'readout_sha256': _sha_json(readout.tolist()),
        'readout_norm': float(torch.linalg.vector_norm(readout)),
    }


def _state_summary(variant, context, history, states, layers):
    episode, state = context['episode'], context['state']
    return {
        'format': STATE_FORMAT,
        'variant': variant['name'], 'reducer': variant['reducer'],
        'episode_id': episode['episode_id'], 'state': state['name'],
        'cutoff': state['cutoff'],
        'case_id': None,
        'written_record_ids': [record['record_id'] for record in history],
        'final_write_id': history[-1]['record_id'],
        'state_sha256': {
            ('shared' if variant['reducer'] == 'shared_mean' else str(layer)):
                _state_sha(_state_weights(states, variant['reducer'], layer))
            for layer in layers
        },
    }


def _evaluate_context_query(*, variant, context, states, history, labels,
                            query, layers, primitive, device, state_shas,
                            zero_sha):
    query_id = query['query_id']
    target_info = labels['targets'][query_id]
    kind = target_info['kind']
    target_id = target_info['target_entry_id']
    rows = []
    final_write_id = history[-1]['record_id']
    zero_weights = primitive.initial_state(1, device=device).weights
    for layer in layers:
        weights = _state_weights(states, variant['reducer'], layer)
        state_sha = state_shas[layer]
        vector = context['query_vectors'][(query_id, layer)]
        if arm_applies('exact_key_value', kind):
            _require(target_id is not None, 'Fact target is absent')
            write = context['writes'][(target_id, layer)]
            exact = write.keys[0, 0].detach().float().cpu()
            rows.append(_readout_row(
                variant=variant, arm='exact_key_value', context=context,
                query=query, layer=layer,
                readout=_read_weights(weights, exact.view(1, 1, *exact.shape)),
                readout_space='value', final_write_id=final_write_id,
                state_sha256=state_sha))
        rows.append(_readout_row(
            variant=variant, arm='query_key_value', context=context,
            query=query, layer=layer,
            readout=_read_weights(weights, vector),
            readout_space='value', final_write_id=final_write_id,
            state_sha256=state_sha))
        rows.append(_readout_row(
            variant=variant, arm='query_key_address', context=context,
            query=query, layer=layer,
            readout=vector[0, 0].detach().float().cpu(),
            readout_space='key', final_write_id=final_write_id,
            state_sha256=state_sha))
        rows.append(_readout_row(
            variant=variant, arm='context_removed', context=context,
            query=query, layer=layer,
            readout=_read_weights(zero_weights, vector),
            readout_space='value', final_write_id=final_write_id,
            state_sha256=zero_sha))
    return rows


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def _candidate_ids(codebook, layer, arm):
    entries = codebook['layers'][str(layer)]
    if arm == 'query_key_address':
        return sorted(codebook['live_entry_ids'])
    _require(set(codebook['live_entry_ids']) <= set(entries),
             'Codebook live entry is absent')
    return sorted(entries)


def _entry_meta(labels):
    meta = {}
    for entry in labels['live_entries']:
        meta[entry['record_id']] = {'role': entry['role'],
                                    'source_family': entry['source_family']}
    for entry in labels['negative_entries']:
        meta[entry['record_id']] = {'role': 'superseded_negative',
                                    'source_family': entry['source_family']}
    return meta


def _recompute_ranking(row, codebook, labels, thresholds):
    _require(row.get('status') == 'ok', 'Only successful rows are ranked')
    _require(row.get('state') == labels.get('state'),
             'Readout state differs from its verifier labels')
    _require(codebook.get('episode_id') == row['episode_id']
             and codebook.get('world_id') == row['world_id']
             and codebook.get('state') == row['state']
             and codebook.get('observation_sha256')
             == row.get('observation_sha256'),
             'Readout row references a foreign or changed codebook')
    layer = row['layer']
    _require(str(layer) in codebook['layers'],
             'Readout layer is absent from the codebook')
    entries = codebook['layers'][str(layer)]
    arm = row['arm']
    _require(arm in READ_ARMS, 'Unknown readout arm')
    case_id = row['case_id']
    _require(case_id in labels['targets'], 'Readout case is not a derived query')
    target_info = labels['targets'][case_id]
    kind = target_info['kind']
    target = target_info['target_entry_id']
    if arm == 'exact_key_value':
        _require(kind == 'fact' and target is not None,
                 'Exact-key arm requires a fact target')
        _require(row['readout_space'] == 'value',
                 'Exact-key arm must read values')
    if arm == 'query_key_address':
        _require(row['readout_space'] == 'key',
                 'Addressing arm must read keys')
    else:
        _require(row['readout_space'] == 'value',
                 'Value arm must read values')
    candidate_ids = _candidate_ids(codebook, layer, arm)
    _require(candidate_ids, 'Candidate set cannot be empty')
    field = 'write_key' if arm == 'query_key_address' else 'value_vector'
    readout = torch.tensor(row['readout'], dtype=torch.float32)
    _require(bool(torch.isfinite(readout).all()), 'Readout is not finite')
    scores = {}
    for entry_id in candidate_ids:
        vector = torch.tensor(entries[entry_id][field], dtype=torch.float32)
        _require(tuple(vector.shape) == tuple(readout.shape),
                 'Readout/candidate shapes differ')
        scores[entry_id] = _cosine_heads(readout, vector)
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    top_entry_id, top_cosine = ordered[0]
    second_cosine = ordered[1][1] if len(ordered) > 1 else None
    non_target = [score for entry_id, score in ordered if entry_id != target]
    strongest_id = next((entry_id for entry_id, _ in ordered
                         if entry_id != target), None)
    strongest_cosine = max(non_target) if non_target else 0.0
    margin = (scores[target] - strongest_cosine
              if target is not None else None)
    target_rank = (1 + sum(1 for _, score in ordered if score > scores[target])
                   if target is not None else None)
    if arm == 'query_key_address':
        floor = thresholds['address_cosine_floor']
        margin_floor = thresholds['address_margin_floor']
    else:
        floor = thresholds['match_cosine_floor']
        margin_floor = thresholds['match_margin_floor']
    accepted = bool(
        top_cosine >= floor
        and ((top_cosine - (second_cosine if second_cosine is not None else 0.0))
             >= margin_floor))
    top1 = bool(target is not None and top_entry_id == target)
    false_recall = bool(accepted and not top1)
    meta = _entry_meta(labels)
    top_role = meta.get(top_entry_id, {}).get('role', 'unknown')
    if accepted and not top1:
        if target is None:
            false_recall_kind = 'unknown_address'
        elif top_entry_id in codebook['negative_entry_ids']:
            false_recall_kind = 'superseded'
        elif top_role == 'distractor':
            false_recall_kind = 'independent_world'
        else:
            false_recall_kind = 'other_source'
    else:
        false_recall_kind = None
    negative_scores = [scores[entry_id] for entry_id
                       in codebook['negative_entry_ids']
                       if entry_id in scores]
    return {
        'format': RANKING_FORMAT,
        'variant': row['variant'], 'reducer': row['reducer'], 'arm': arm,
        'episode_id': row['episode_id'], 'world_id': row['world_id'],
        'state': row['state'], 'cutoff': row['cutoff'],
        'case_id': case_id, 'query_id': row['query_id'],
        'kind': kind, 'split': labels['split'],
        'complexity': labels['complexity'],
        'target_entry_id': target,
        'target_role': target_info['target_role'],
        'target_source_family': target_info['target_source_family'],
        'layer': layer, 'candidate_count': len(candidate_ids),
        'scores': scores, 'ranking': [[entry_id, score] for entry_id, score in ordered],
        'target_rank': target_rank, 'top_entry_id': top_entry_id,
        'top_cosine': top_cosine, 'second_cosine': second_cosine,
        'strongest_distractor_id': strongest_id,
        'strongest_distractor_cosine': strongest_cosine,
        'margin_over_strongest_distractor': margin,
        'accepted': accepted, 'top1': top1,
        'false_recall': false_recall,
        'false_recall_kind': false_recall_kind,
        'superseded_margin': (scores[target] - max(negative_scores)
                              if target is not None and negative_scores else None),
        'superseded_selected': bool(
            accepted and top_entry_id in codebook['negative_entry_ids']),
        'cross_world_selected': bool(
            accepted and meta.get(top_entry_id, {}).get('source_family')
            == 'independent_world'),
        'final_write_id': row['final_write_id'],
        'top_is_final_write': bool(top_entry_id == row['final_write_id']),
        'target_is_final_write': bool(target == row['final_write_id']),
        'readout_norm': float(torch.linalg.vector_norm(readout)),
        'readout_sha256': row['readout_sha256'],
        'state_sha256': row['state_sha256'],
        'observation_sha256': row['observation_sha256'],
    }


def run_update_dynamics(model, tokenizer, protocol, *, output_dir,
                        device='cpu', episodes=None,
                        labels_by_episode=None,
                        writer_extension_receipt=None):
    """Run every declared update-dynamics variant on one fixed episode set."""
    _require(protocol.get('format') == PROTOCOL_FORMAT,
             'Update-dynamics diagnostic requires its versioned protocol')
    thresholds = _resolve_thresholds(protocol.get('thresholds'))
    limits = _resolve_limits(protocol.get('limits'))
    budget = protocol.get('budget') or {}
    _require(isinstance(budget, dict), 'Protocol budget must be an object')
    max_wall = budget.get('max_wall_seconds')
    _require(max_wall is None or (type(max_wall) is int and max_wall > 0),
             'Protocol wall-clock budget must be a positive integer')
    _require(episodes is not None, 'Update-dynamics run requires episodes')
    layers = tuple(protocol.get('layers') or DEFAULT_LAYERS)
    _require(layers and len(set(layers)) == len(layers),
             'Declared layers must be distinct')
    available = tuple(model.ports.injection_layers)
    _require(all(layer in available for layer in layers),
             f'Declared layers {list(layers)} are not available {available}')
    _require(set(layers) == set(available),
             'Update-dynamics must compare every declared injection depth')
    variants = resolve_variants(
        [variant['name'] for variant in protocol.get('variants', [])]
        if protocol.get('variants') else None)
    if labels_by_episode is None:
        labels_by_episode = derive_matched_labels(episodes)
    _require(isinstance(labels_by_episode, dict),
             'Verifier labels must be a mapping')
    workload = planned_workload(episodes, labels_by_episode,
                                variants=variants, protocol=protocol)
    _require(workload['readout_rows'] <= limits['max_rows'],
             'Scheduled workload exceeds max_rows')
    inventory = planned_inventory(episodes, labels_by_episode,
                                  variants=variants, layers=layers)
    if protocol.get('planned_workload') is not None:
        _require(protocol['planned_workload'] == workload,
                 'Protocol planned workload differs from the exact plan')
    if protocol.get('planned_inventory_sha256') is not None:
        _require(protocol['planned_inventory_sha256'] == _sha_json(inventory),
                 'Protocol planned inventory differs from the exact plan')

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ('readouts.jsonl', 'states.jsonl', 'codebooks.json',
                 'mechanical.json', 'rankings.jsonl', 'failures.json'):
        _require(not (output_dir / name).exists(),
                 f'Update-dynamics raw artifact already exists: {name}')

    model.eval()
    requires_grad_before = {name: parameter.requires_grad
                            for name, parameter in model.named_parameters()}
    _freeze(model)
    versions_before = _parameter_versions(model)
    spec = model.manifest.fast_memory
    declared_engine = protocol.get('engine')
    if declared_engine:
        for key, actual in (('heads', spec.heads), ('key_size', spec.key_size),
                            ('value_size', spec.value_size),
                            ('update_rule', spec.update_rule)):
            _require(declared_engine.get(key, actual) == actual,
                     f'Protocol engine {key} differs from the loaded model')
        _require(declared_engine.get('depth_mode', spec.depth_mode)
                 == spec.depth_mode
                 and declared_engine.get('depth_writers', spec.depth_writers)
                 == spec.depth_writers,
                 'Protocol depth mode differs from the loaded model')
    primitive = SGDN2Memory(spec.heads, spec.key_size, spec.value_size)
    started = _utc_now()
    started_monotonic = time.monotonic()

    readouts, state_rows, codebooks, parity_rows = [], [], {}, []
    with torch.no_grad():
        for episode in episodes:
            for state in episode_states(episode):
                labels = labels_by_episode[episode['episode_id']][state['name']]
                context = _state_context(model, tokenizer, episode, state, layers)
                codebook = _codebook(context, layers)
                codebook_key = f"{episode['episode_id']}/{state['name']}"
                _require(codebook_key not in codebooks, 'Duplicate codebook key')
                codebooks[codebook_key] = codebook
                shared_full_state = None
                for variant in variants:
                    if variant['target_dependent']:
                        targets = [query for query in episode['queries']
                                   if labels['targets'][query['query_id']]['kind']
                                   == 'fact']
                        for query in targets:
                            target_id = labels['targets'][
                                query['query_id']]['target_entry_id']
                            history = variant_history(
                                context['records'], variant, target_id=target_id)
                            history_writes = [
                                {layer: context['writes'][(record['record_id'], layer)]
                                 for layer in layers}
                                for record in history]
                            states = _build_state(
                                primitive, layers, history_writes,
                                reducer=variant['reducer'],
                                gate_control=variant['gate_control'],
                                device=device)
                            state_shas = {
                                layer: _state_sha(_state_weights(
                                    states, variant['reducer'], layer))
                                for layer in layers}
                            zero_sha = _state_sha(
                                primitive.initial_state(1, device=device).weights)
                            summary = _state_summary(
                                variant, context, history, states, layers)
                            summary['case_id'] = query['query_id']
                            state_rows.append(summary)
                            readouts.extend(_evaluate_context_query(
                                variant=variant, context=context, states=states,
                                history=history, labels=labels, query=query,
                                layers=layers, primitive=primitive,
                                device=device, state_shas=state_shas,
                                zero_sha=zero_sha))
                    else:
                        history = variant_history(context['records'], variant)
                        history_writes = [
                            {layer: context['writes'][(record['record_id'], layer)]
                             for layer in layers}
                            for record in history]
                        states = _build_state(
                            primitive, layers, history_writes,
                            reducer=variant['reducer'],
                            gate_control=variant['gate_control'],
                            device=device)
                        state_shas = {
                            layer: _state_sha(_state_weights(
                                states, variant['reducer'], layer))
                            for layer in layers}
                        zero_sha = _state_sha(
                            primitive.initial_state(1, device=device).weights)
                        state_rows.append(_state_summary(
                            variant, context, history, states, layers))
                        if (variant['name'] == 'shared_mean/full'
                                and variant['gate_control'] is None
                                and not variant['refresh_target']):
                            shared_full_state = states['shared']
                        for query in episode['queries']:
                            readouts.extend(_evaluate_context_query(
                                variant=variant, context=context, states=states,
                                history=history, labels=labels, query=query,
                                layers=layers, primitive=primitive,
                                device=device, state_shas=state_shas,
                                zero_sha=zero_sha))
                _require(shared_full_state is not None,
                         'Shared full baseline state was not built')
                production = _production_memory(
                    model, tokenizer, episode, state, context['records'])
                difference = float((production.read().weights
                                    - shared_full_state.weights).abs().max())
                parity_rows.append({
                    'episode_id': episode['episode_id'],
                    'state': state['name'],
                    'max_abs_difference': difference,
                    'equal': difference <= thresholds['production_parity_atol'],
                    'shared_state_sha256': _state_sha(shared_full_state.weights),
                    'production_state_sha256': _state_sha(
                        production.read().weights),
                })
                del production
                if max_wall is not None:
                    elapsed = time.monotonic() - started_monotonic
                    _require(elapsed <= max_wall,
                             'Update-dynamics diagnostic exceeded its deadline')

    readouts.sort(key=lambda row: (row['variant'], row['episode_id'],
                                   row['state'], row['case_id'], row['arm'],
                                   row['layer']))
    state_rows.sort(key=lambda row: (row['variant'], row['episode_id'],
                                     row['state'], row['case_id'] or ''))
    for row in readouts:
        _require(not any(marker in row for marker in
                         ('expected', 'expected_answer', 'gold', 'label',
                          'audit_label', 'target_role', 'kind')),
                 'Raw update-dynamics rows must not carry evaluator labels')
    _write_jsonl(output_dir / 'readouts.jsonl', readouts)
    _write_jsonl(output_dir / 'states.jsonl', state_rows)
    _write_json(output_dir / 'codebooks.json', {
        'format': CODEBOOK_FORMAT + '.collection.v1',
        'codebooks': codebooks})
    frozen = {
        'optimizer_updates': 0, 'parameters_updated': [],
        'parameter_versions_equal': _parameter_versions(model) == versions_before,
        'requires_grad_all_false': not any(
            parameter.requires_grad for parameter in model.parameters()),
        'requires_grad_before_all_false': not any(
            requires_grad_before.values()),
    }
    _require(frozen['parameter_versions_equal'],
             'The diagnostic changed model parameters')
    mechanical = {
        'format': MECHANICAL_FORMAT,
        'frozen': frozen,
        'production_parity': {
            'rows': parity_rows,
            'all_equal': all(row['equal'] for row in parity_rows),
            'max_abs_difference': max(
                (row['max_abs_difference'] for row in parity_rows), default=None),
        },
        'full_fact_parity': _full_fact_parity(
            state_rows, labels_by_episode, variants, episodes),
        'limits': limits,
    }
    _write_json(output_dir / 'mechanical.json', mechanical)
    score = score_update_dynamics(
        readouts, codebooks, labels_by_episode,
        thresholds=thresholds, mechanical=mechanical, inventory=inventory,
        failure_limit=limits['max_preserved_failures'], variants=variants)
    _write_jsonl(output_dir / 'rankings.jsonl', score['rankings'])
    _write_json(output_dir / 'failures.json', {
        'format': FAILURES_FORMAT,
        'counts': score['failure_counts'],
        'failed_variants': score['failed_variants'],
        'failures': score['failures'],
        'decision': score['decision'],
        'first_failing_mechanism': score['first_failing_mechanism'],
    })

    def _hash(name):
        return _sha_file(output_dir / name)

    elapsed_seconds = time.monotonic() - started_monotonic
    return {
        'format': RUN_FORMAT, 'status': 'completed',
        'native': device == 'mps', 'device': device,
        'fallback_disabled': True,
        'guard_required': 'REPLICA_CAPACITY_GUARD=1',
        'writer_extension_receipt': writer_extension_receipt,
        'layers': list(layers), 'representation': FACT_REPRESENTATION,
        'counts': {
            'episodes': len(episodes),
            'states': workload['states'],
            'variants': len(variants),
            'state_builds': len(state_rows),
            'readout_rows': len(readouts),
            'unique_cases': len(inventory),
        },
        'workload': workload,
        'inventory': [list(item) for item in inventory],
        'readouts_sha256': _hash('readouts.jsonl'),
        'states_sha256': _hash('states.jsonl'),
        'codebooks_sha256': _hash('codebooks.json'),
        'mechanical_sha256': _hash('mechanical.json'),
        'rankings_sha256': _hash('rankings.jsonl'),
        'failures_sha256': _hash('failures.json'),
        'mechanical': mechanical,
        **score,
        'frozen': frozen,
        'optimizer_updates': 0, 'parameters_updated': [],
        'promote_v5_memory': False, 'neural_promotion': False,
        'v5_status': 'rejected_by_prior_evidence',
        'deployment': 'diagnostic_only_not_published',
        'quality_scope': ('Frozen update-dynamics comparison only. Exact '
                          'observed-key recovery is a storage upper bound, not '
                          'addressing. No training or useful-memory claim.'),
        'resource': {
            'started_at_utc': started, 'finished_at_utc': _utc_now(),
            'elapsed_seconds': elapsed_seconds,
            'deadline_seconds': max_wall,
            'deadline_ok': (True if max_wall is None
                            else elapsed_seconds <= max_wall),
            'rss_start_bytes': _rss_bytes(), 'max_rss_bytes': _rss_bytes(),
            'mps': _mps_memory() if device == 'mps' else None,
        },
    }


def _full_fact_parity(state_rows, labels_by_episode, variants, episodes):
    """Verify exact ordered history identity for every declared variant."""
    by_name = {variant['name']: variant for variant in variants}
    episodes_by_id = {episode['episode_id']: episode for episode in episodes}
    failures = []
    rows = 0
    for summary in state_rows:
        variant = by_name[summary['variant']]
        labels = labels_by_episode[summary['episode_id']][summary['state']]
        episode = episodes_by_id[summary['episode_id']]
        records, _, _ = _record_views(
            episode, {'cutoff': labels['cutoff']})
        full_ids = [record['record_id'] for record in records]
        observed = list(labels['observed_record_ids'])
        written = list(summary['written_record_ids'])
        target = None
        if variant['target_dependent']:
            target_info = labels['targets'].get(summary['case_id'])
            target = (target_info or {}).get('target_entry_id')
        expected = variant_history(records, variant, target_id=target)
        expected_ids = [record['record_id'] for record in expected]
        rows += 1
        valid = written == expected_ids
        if variant['order'] == 'full' and not variant['refresh_target']:
            valid = valid and expected_ids == observed == full_ids
        elif variant['order'] == 'reversed':
            reversed_ids = [record['record_id'] for record in _reverse_groups(records)]
            valid = valid and expected_ids == reversed_ids and sorted(
                expected_ids) == sorted(full_ids)
        elif variant['gate_control']:
            valid = valid and expected_ids == observed == full_ids
        if not valid:
            failures.append({
                'variant': summary['variant'],
                'episode_id': summary['episode_id'],
                'state': summary['state'],
                'case_id': summary['case_id'],
                'written_record_ids': written[:12],
                'expected_record_ids': expected_ids[:12],
                'observed_record_ids': observed[:12],
            })
    return {'rows': rows, 'all_equal': not failures,
            'ordered_identity': True, 'failures': failures[:20]}


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _metric_block(items, *, include_rank=True):
    ranks = [row['target_rank'] for row in items
             if row['target_rank'] is not None]
    margins = [row['margin_over_strongest_distractor'] for row in items
               if row['margin_over_strongest_distractor'] is not None]
    facts = [row for row in items if row['kind'] == 'fact']
    top1 = sum(1 for row in facts if row['top1'])
    accepted = sum(1 for row in items if row['accepted'])
    false = sum(1 for row in items if row['false_recall'])
    return {
        'n': len(items),
        'fact_n': len(facts),
        'top1_rate': _rate(top1, len(facts)),
        'mean_rank': _mean(ranks) if include_rank else None,
        'median_rank': (sorted(ranks)[len(ranks) // 2] if ranks else None),
        'mean_margin': _mean(margins),
        'accepted_rate': _rate(accepted, len(items)),
        'false_recall_rate': _rate(false, len(items)),
        'mean_readout_norm': _mean([row['readout_norm'] for row in items]),
    }


def _variant_metrics(rankings, variants):
    result = {}
    for variant in variants:
        per_arm = {}
        selected = [row for row in rankings
                    if row['variant'] == variant['name']]
        for arm in READ_ARMS:
            items = [row for row in selected if row['arm'] == arm]
            if not items:
                continue
            facts = [row for row in items if row['kind'] == 'fact']
            unknown = [row for row in items if row['kind'] == 'unknown']
            correction = [row for row in facts
                          if row['target_role'] == 'correction_target']
            unaffected = [row for row in facts
                          if row['target_role'] == 'retained']
            final_targets = [row for row in facts if row['target_is_final_write']]
            nonfinal_targets = [row for row in facts
                                if not row['target_is_final_write']]
            block = _metric_block(items)
            block.update({
                'fact': _metric_block(facts),
                'unknown': _metric_block(unknown),
                'correction': _metric_block(correction),
                'unaffected': _metric_block(unaffected),
                'final_write_target': _metric_block(final_targets),
                'nonfinal_write_target': _metric_block(nonfinal_targets),
                'final_write_top_rate': _rate(
                    sum(1 for row in facts if row['top_is_final_write']),
                    len(facts)),
                'superseded_selection_rate': _rate(
                    sum(1 for row in facts if row['superseded_selected']),
                    len(facts)),
                'cross_world_selection_rate': _rate(
                    sum(1 for row in items if row['cross_world_selected']),
                    len(items)),
            })
            per_arm[arm] = block
        result[variant['name']] = per_arm
    return result


def _candidate_parity(rankings):
    groups = {}
    for row in rankings:
        key = (row['episode_id'], row['state'], row['case_id'],
               row['layer'], row['arm'])
        signature = (row['candidate_count'], tuple(sorted(row['scores'])))
        groups.setdefault(key, set()).add(signature)
    failures = [key for key, signatures in groups.items()
                if len(signatures) != 1]
    return {'groups': len(groups), 'all_equal': not failures,
            'failures': [list(item) for item in failures[:20]]}


def _role_quality(metrics, settings):
    """Explicitly report-only role thresholds; no mechanism consumer uses them."""
    result = {}
    for arm in ('exact_key_value', 'query_key_value'):
        metric = _metric(metrics, 'shared_mean/full', arm) or {}
        result[arm] = {
            'status': 'report_only',
            'enforced': False,
            'correction_top1': {
                'value': (metric.get('correction') or {}).get('top1_rate'),
                'threshold': settings['correction_top1_floor']},
            'unaffected_top1': {
                'value': (metric.get('unaffected') or {}).get('top1_rate'),
                'threshold': settings['unaffected_top1_floor']},
            'unknown_false_recall': {
                'value': (metric.get('unknown') or {}).get('false_recall_rate'),
                'threshold': settings['unknown_false_recall_max']},
        }
    return result


def _metric(metrics, variant, arm):
    return (metrics.get(variant) or {}).get(arm)


def _passes_exact(metric, settings):
    block = (metric or {}).get('fact') or {}
    return bool(
        block.get('n')
        and block.get('top1_rate') is not None
        and block['top1_rate'] >= settings['exact_key_top1_floor']
        and block.get('mean_rank') is not None
        and block['mean_rank'] <= settings['exact_key_mean_rank_max']
        and block.get('mean_margin') is not None
        and block['mean_margin'] >= settings['exact_key_margin_floor']
        and (metric.get('false_recall_rate') or 0.0)
        <= settings['exact_key_false_recall_max'])


def _passes_address(metric, settings):
    block = (metric or {}).get('fact') or {}
    return bool(
        block.get('n')
        and block.get('top1_rate') is not None
        and block['top1_rate'] >= settings['address_top1_floor']
        and block.get('mean_rank') is not None
        and block['mean_rank'] <= settings['address_mean_rank_max']
        and (block.get('mean_margin') or 0.0)
        >= settings['address_margin_mean_floor']
        and (metric.get('false_recall_rate') or 0.0)
        <= settings['address_false_recall_max'])


def _improved(control, baseline, settings):
    control_fact = (control or {}).get('fact') or {}
    baseline_fact = (baseline or {}).get('fact') or {}
    if not control_fact.get('n') or not baseline_fact.get('n'):
        return False
    top1_delta = ((control_fact.get('top1_rate') or 0.0)
                  - (baseline_fact.get('top1_rate') or 0.0))
    rank_delta = ((baseline_fact.get('mean_rank') or float('inf'))
                  - (control_fact.get('mean_rank') or float('inf')))
    margin_delta = ((control_fact.get('mean_margin') or 0.0)
                    - (baseline_fact.get('mean_margin') or 0.0))
    return (top1_delta >= settings['mechanism_top1_delta']
            or rank_delta >= settings['mechanism_rank_delta']
            or margin_delta >= settings['mechanism_margin_delta'])


def _comparison(metrics, variant, arm, baseline, settings):
    metric = _metric(metrics, variant, arm) or {}
    base = _metric(metrics, baseline, arm) or {}
    fact = metric.get('fact') or {}
    base_fact = base.get('fact') or {}
    return {
        'variant': variant, 'arm': arm,
        'top1_rate': fact.get('top1_rate'),
        'baseline_top1_rate': base_fact.get('top1_rate'),
        'top1_delta': ((fact.get('top1_rate') or 0.0)
                       - (base_fact.get('top1_rate') or 0.0)),
        'mean_rank': fact.get('mean_rank'),
        'baseline_mean_rank': base_fact.get('mean_rank'),
        'mean_margin': fact.get('mean_margin'),
        'baseline_mean_margin': base_fact.get('mean_margin'),
        'false_recall_rate': metric.get('false_recall_rate'),
        'passes_exact_gate': _passes_exact(metric, settings),
        'improved': _improved(metric, base, settings),
    }


def select_first_failing_mechanism(metrics, *, mechanical_valid=True,
                                   settings=None):
    """Select the first mechanism whose isolated control recovers the upper bound.

    Order is fixed by design: mechanical validity, shared reduction, write
    position, recency (target refresh), forced-full erase/decay/gate, then
    downstream value readout.  Query-key value recovery must fail before a
    downstream readout cause is named; if it passes while the exact observed
    key fails, the result is exact-key-specific.  A later mechanism cannot be
    named when an earlier isolated control passes.
    """
    settings = _resolve_thresholds(settings)
    baseline = 'shared_mean/full'
    comparisons = {}
    if not mechanical_valid:
        return {
            'decision': 'invalid_mechanical_evidence',
            'first_failing_mechanism': 'mechanical',
            'comparisons': comparisons,
            'smallest_next_change': ('Stop; zero-update, production-parity, '
                                     'fact-parity or inventory evidence failed.'),
            'address_passes': None,
        }
    baseline_exact = _metric(metrics, baseline, 'exact_key_value')
    baseline_address = _metric(metrics, baseline, 'query_key_address')
    baseline_value = _metric(metrics, baseline, 'query_key_value')
    _require(baseline_exact is not None and baseline_address is not None
             and baseline_value is not None,
             'Selection requires the shared full exact, address and value baselines')
    if _passes_exact(baseline_exact, settings):
        return {
            'decision': 'exact_key_recovery_ok',
            'first_failing_mechanism': None,
            'comparisons': comparisons,
            'smallest_next_change': ('No exact-key storage failure is isolated; '
                                     'do not change the writer on this evidence.'),
            'address_passes': _passes_address(baseline_address, settings),
        }
    address_passes = _passes_address(baseline_address, settings)

    per_depth = 'per_depth/full'
    comparisons['per_depth'] = _comparison(
        metrics, per_depth, 'exact_key_value', baseline, settings)
    if (comparisons['per_depth']['passes_exact_gate']
            and comparisons['per_depth']['improved']):
        return {
            'decision': 'shared_reduction_interference',
            'first_failing_mechanism': 'shared_depth_convex_mean',
            'comparisons': comparisons,
            'smallest_next_change': ('Test a per-depth or depth-selected read '
                                     'path against the same frozen writer; do '
                                     'not train a reader or writer.'),
            'address_passes': address_passes,
        }

    position_names = [f'shared_mean/{label}' for label in POSITION_LABELS]
    position_names += [f'per_depth/{label}' for label in POSITION_LABELS]
    position_hits = []
    for name in position_names:
        if name not in metrics:
            continue
        item = _comparison(metrics, name, 'exact_key_value', baseline, settings)
        comparisons[f'position:{name}'] = item
        if item['passes_exact_gate'] and item['improved']:
            position_hits.append(item)
    if position_hits:
        def rank_key(item):
            return (-(item['top1_rate'] or 0.0),
                    item['mean_rank'] or float('inf'),
                    item['variant'])
        best = sorted(position_hits, key=rank_key)[0]
        return {
            'decision': 'write_position_or_recency_interference',
            'first_failing_mechanism': 'write_position',
            'comparisons': comparisons,
            'isolated_position_variant': best['variant'],
            'smallest_next_change': (
                f'{best["variant"]} recovers exact-key recovery. Test the '
                'matching order-aware/refresh update only in this diagnostic; '
                'do not train a writer.'),
            'address_passes': address_passes,
        }

    refresh_names = ['shared_mean/target_refresh_last',
                     'per_depth/target_refresh_last']
    refresh_hits = []
    for name in refresh_names:
        if name not in metrics:
            continue
        item = _comparison(metrics, name, 'exact_key_value', baseline, settings)
        comparisons[f'refresh:{name}'] = item
        if item['passes_exact_gate'] and item['improved']:
            refresh_hits.append(item)
    if refresh_hits:
        best = sorted(refresh_hits, key=lambda item: (
            -(item['top1_rate'] or 0.0),
            item['mean_rank'] or float('inf'), item['variant']))[0]
        return {
            'decision': 'last_write_recency_interference',
            'first_failing_mechanism': 'write_recency',
            'comparisons': comparisons,
            'isolated_refresh_variant': best['variant'],
            'smallest_next_change': (
                f'{best["variant"]} recovers exact-key recovery. Test a '
                'last-write-preserving update or re-observation policy in the '
                'diagnostic only; do not train.'),
            'address_passes': address_passes,
        }

    gate_names = [f'shared_mean/{gate}' for gate in GATE_CONTROLS]
    gate_names += [f'per_depth/{gate}' for gate in GATE_CONTROLS]
    gate_hits = []
    for name in gate_names:
        if name not in metrics:
            continue
        item = _comparison(metrics, name, 'exact_key_value', baseline, settings)
        comparisons[f'gate:{name}'] = item
        if item['passes_exact_gate'] and item['improved']:
            gate_hits.append(item)
    if gate_hits:
        best = sorted(gate_hits, key=lambda item: (
            -(item['top1_rate'] or 0.0),
            item['mean_rank'] or float('inf'), item['variant']))[0]
        return {
            'decision': 'forced_full_gate_dynamics_interference',
            'first_failing_mechanism': 'forced_full_erase_decay_gate',
            'comparisons': comparisons,
            'isolated_gate_variant': best['variant'],
            'smallest_next_change': (
                f'{best["variant"]} recovers exact-key recovery. Inspect the '
                'per-head erase/decay/gate overwrite before proposing any '
                'writer change; forced-full factors are diagnostics, not '
                'literal removal of erasure/forgetting. Do not train.'),
            'address_passes': address_passes,
        }

    value_passes = _passes_exact(baseline_value, settings)
    comparisons['query_value_recovery'] = _comparison(
        metrics, baseline, 'query_key_value', baseline, settings)
    comparisons['query_value_recovery']['passes_value_gate'] = value_passes
    if value_passes:
        return {
            'decision': 'exact_key_specific_failure',
            'first_failing_mechanism': 'exact_observed_key_read_path',
            'comparisons': comparisons,
            'smallest_next_change': ('Query-key value recovery passes while '
                                     'exact observed-key recovery fails. '
                                     'Inspect per-head write-key/value '
                                     'alignment and exact-key normalization; '
                                     'do not change candidates or train.'),
            'address_passes': address_passes,
        }

    if address_passes:
        return {
            'decision': 'value_readout_downstream',
            'first_failing_mechanism': 'value_write_or_readout',
            'comparisons': comparisons,
            'smallest_next_change': ('Query-derived addressing passes while '
                                     'query-key and exact-key value recovery '
                                     'fail. Inspect per-head key/value write '
                                     'alignment and candidate scoring; do not '
                                     'train the writer.'),
            'address_passes': True,
        }
    return {
        'decision': 'no_isolated_update_dynamics_cause',
        'first_failing_mechanism': 'unresolved',
        'comparisons': comparisons,
        'smallest_next_change': ('Add per-head overwrite traces or one-write-'
                                 'at-a-time state snapshots. None of the '
                                 'declared reduction, position or gate controls '
                                 'isolates the failure; do not train.'),
        'address_passes': address_passes,
    }


def score_update_dynamics(readouts, codebooks, labels_by_episode, *,
                          thresholds=None, mechanical=None, inventory=None,
                          failure_limit=1000, variants=None):
    settings = _resolve_thresholds(thresholds)
    variants = variants or variant_plan()
    limits = _resolve_limits((mechanical or {}).get('limits'))
    failure_limit = int(failure_limit or limits['max_preserved_failures'])
    _require(isinstance(labels_by_episode, dict), 'Verifier labels are required')
    _require(isinstance(codebooks, dict) and codebooks, 'Codebooks are required')
    known = {variant['name'] for variant in variants}
    seen = set()
    for row in readouts:
        identity = (row['variant'], row['arm'], row['episode_id'],
                    row['state'], row['case_id'], row['layer'])
        _require(identity not in seen, f'Duplicate readout row: {identity}')
        seen.add(identity)
        _require(row.get('format') == READOUT_FORMAT,
                 'Readout row format differs')
        _require(row['variant'] in known, 'Readout names an undeclared variant')
        labels = (labels_by_episode.get(row['episode_id']) or {}).get(row['state'])
        _require(labels is not None, 'Readout references underived labels')
        codebook = codebooks.get(row['codebook_key'])
        _require(codebook is not None, 'Readout references a missing codebook')
        _require(codebook.get('observation_sha256')
                 == row.get('observation_sha256'),
                 'Readout observation differs from its codebook state')
        target = labels['targets'].get(row['query_id'])
        _require(target is not None
                 and target['query_text_sha256'] == row.get('query_text_sha256'),
                 'Readout query differs from its verifier-derived query')
        _require(row.get('readout_sha256') == _sha_json(row.get('readout')),
                 'Readout bytes differ from their stored digest')
    if inventory is not None:
        expected = {tuple(item) for item in inventory}
        observed = {(row['variant'], row['arm'], row['episode_id'],
                     row['state'], row['case_id'], row['layer'])
                    for row in readouts}
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        _require(not missing, f'Missing update-dynamics rows: {missing[:5]}')
        _require(not extra, f'Unexpected update-dynamics rows: {extra[:5]}')
        _require(len(observed) == len(expected),
                 'Update-dynamics row inventory count differs')

    rankings = []
    for row in readouts:
        labels = labels_by_episode[row['episode_id']][row['state']]
        rankings.append(_recompute_ranking(
            row, codebooks[row['codebook_key']], labels, settings))
    metrics = _variant_metrics(rankings, variants)
    candidate_parity = _candidate_parity(rankings)

    frozen = (mechanical or {}).get('frozen') or {}
    frozen_ok = (
        frozen.get('optimizer_updates') == 0
        and frozen.get('parameters_updated') == []
        and frozen.get('parameter_versions_equal') is True
        and frozen.get('requires_grad_all_false') is True)
    parity = (mechanical or {}).get('production_parity') or {}
    parity_ok = bool(parity.get('rows')) and all(
        row.get('equal') is True for row in parity.get('rows') or [])
    parity_failures = [row for row in parity.get('rows') or []
                       if row.get('equal') is not True]
    fact_parity = (mechanical or {}).get('full_fact_parity') or {}
    fact_parity_ok = bool(fact_parity.get('rows')) and fact_parity.get(
        'all_equal') is True
    inventory_ok = inventory is not None
    mechanical_gates = {
        'zero_updates': {'status': 'pass' if frozen_ok else 'fail',
                         'metrics': frozen},
        'production_parity': {
            'status': 'pass' if parity_ok else 'fail',
            'metrics': {'rows': len(parity.get('rows') or []),
                        'failures': parity_failures[:10],
                        'max_abs_difference': parity.get('max_abs_difference')},
            'thresholds': {'production_parity_atol':
                           settings['production_parity_atol']}},
        'full_fact_parity': {
            'status': 'pass' if fact_parity_ok else 'fail',
            'metrics': {key: fact_parity.get(key)
                        for key in ('rows', 'all_equal', 'ordered_identity',
                                    'failures')}},
        'candidate_inventory_parity': {
            'status': 'pass' if candidate_parity['all_equal'] else 'fail',
            'metrics': candidate_parity},
        'row_inventory': {
            'status': 'pass' if inventory_ok else 'fail',
            'metrics': {'expected': len(inventory or []),
                        'observed': len(readouts)}},
    }
    mechanical_valid = all(gate['status'] == 'pass'
                           for gate in mechanical_gates.values())
    selection = select_first_failing_mechanism(
        metrics, mechanical_valid=mechanical_valid, settings=settings)
    decision = selection['decision']
    first = selection['first_failing_mechanism']
    role_quality = _role_quality(metrics, settings)

    failures = {}
    counts = {}
    failed_variants = []
    for variant in variants:
        name = variant['name']
        items = [row for row in rankings
                 if row['variant'] == name
                 and row['arm'] != 'context_removed'
                 and (row['false_recall']
                      or (row['kind'] == 'fact' and not row['top1']))]
        if items:
            failed_variants.append(name)
        counts[name] = len(items)
        failures[name] = [{
            'arm': row['arm'], 'case_id': row['case_id'],
            'episode_id': row['episode_id'], 'state': row['state'],
            'target_entry_id': row['target_entry_id'],
            'target_role': row['target_role'],
            'top_entry_id': row['top_entry_id'],
            'target_rank': row['target_rank'],
            'margin_over_strongest_distractor':
                row['margin_over_strongest_distractor'],
            'false_recall_kind': row['false_recall_kind'],
            'final_write_id': row['final_write_id'],
        } for row in items[:failure_limit]]

    return {
        'rankings': rankings,
        'metrics': metrics,
        'comparisons': selection['comparisons'],
        'mechanical_gates': mechanical_gates,
        'mechanical_valid': mechanical_valid,
        'decision': decision,
        'first_failing_mechanism': first,
        'address_passes': selection.get('address_passes'),
        'isolated_position_variant': selection.get('isolated_position_variant'),
        'isolated_gate_variant': selection.get('isolated_gate_variant'),
        'isolated_refresh_variant': selection.get('isolated_refresh_variant'),
        'smallest_next_change': selection['smallest_next_change'],
        'thresholds': settings,
        'role_quality': role_quality,
        'failure_counts': counts,
        'failure_total': sum(counts.values()),
        'failed_variants': failed_variants,
        'failures': failures,
        'summary': {
            'shared_exact_top1': (
                metrics.get('shared_mean/full', {})
                .get('exact_key_value', {}).get('fact', {}).get('top1_rate')),
            'per_depth_exact_top1': (
                metrics.get('per_depth/full', {})
                .get('exact_key_value', {}).get('fact', {}).get('top1_rate')),
            'shared_address_top1': (
                metrics.get('shared_mean/full', {})
                .get('query_key_address', {}).get('fact', {}).get('top1_rate')),
            'per_depth_address_top1': (
                metrics.get('per_depth/full', {})
                .get('query_key_address', {}).get('fact', {}).get('top1_rate')),
            'shared_final_write_top_rate': (
                metrics.get('shared_mean/full', {})
                .get('exact_key_value', {}).get('final_write_top_rate')),
            'per_depth_final_write_top_rate': (
                metrics.get('per_depth/full', {})
                .get('exact_key_value', {}).get('final_write_top_rate')),
            'readout_rows': len(readouts),
            'variants': len(variants),
        },
    }


# ---------------------------------------------------------------------------
# Protocol / preparation / admission
# ---------------------------------------------------------------------------

def build_protocol(episodes, labels_by_episode, *, layers=DEFAULT_LAYERS,
                   variants=None, thresholds=None, limits=None,
                   engine=None, source=None, writer_extension=None,
                   budget=None, inventory=None):
    variants = variants or variant_plan()
    thresholds = _resolve_thresholds(thresholds)
    limits = _resolve_limits(limits)
    protocol = {
        'format': PROTOCOL_FORMAT,
        'mode': 'frozen_update_dynamics',
        'native_reader': False,
        'generation': False,
        'layers': list(layers),
        'representation': FACT_REPRESENTATION,
        'engine': (dict(engine) if engine is not None else None),
        'variants': variants,
        'arms': list(READ_ARMS),
        'thresholds': thresholds,
        'limits': limits,
        'source': source,
        'writer_extension': writer_extension,
        'budget': dict(budget or {
            'optimizer_updates': 0, 'gradient_examples': 0,
            'native_training_jobs': 0, 'native_diagnostic_jobs': 1,
            'checkpoints_saved': 0, 'model_loads_per_job': 1,
            'extension_reloads_per_job': 1, 'max_wall_seconds': 900}),
    }
    protocol['planned_workload'] = planned_workload(
        episodes, labels_by_episode, variants=variants, protocol=protocol)
    protocol['planned_inventory_sha256'] = _sha_json(
        planned_inventory(episodes, labels_by_episode, variants=variants,
                          layers=layers))
    if inventory is not None:
        protocol['planned_inventory_sha256'] = _sha_json(inventory)
    return protocol


def _implementation_sha256():
    result = {}
    for relative in _IMPLEMENTATION_PATHS:
        path = PROJECT_ROOT / relative
        _require(path.is_file(), f'Implementation pin path is missing: {relative}')
        result[relative] = _sha_file(path)
    return result


def _read_source_prepared(source_prepared):
    source = Path(source_prepared)
    _require(source.is_dir(), 'Source prepared directory is missing')
    episodes_path = source / 'episodes.json'
    labels_path = source / 'labels.json'
    _require(episodes_path.is_file() and labels_path.is_file(),
             'Source preparation requires episodes.json and labels.json')
    episodes_data = json.loads(episodes_path.read_text())
    labels_data = json.loads(labels_path.read_text())
    episodes = episodes_data.get('episodes')
    _require(isinstance(episodes, list) and episodes,
             'Source episodes are missing')
    _require(isinstance(labels_data.get('episodes'), dict),
             'Source verifier labels are missing')
    derived = derive_matched_labels(episodes)
    _require(derived == labels_data['episodes'],
             'Source verifier labels differ from re-derived labels')
    admission_path = source / 'admission.json'
    admission = None
    if admission_path.is_file():
        admission = json.loads(admission_path.read_text())
        _require(admission.get('status') == 'admitted',
                 'Source preparation admission is not admitted')
        _require(admission.get('episodes_sha256') == _sha_file(episodes_path)
                 and admission.get('labels_sha256') == _sha_file(labels_path),
                 'Source admission pins differ from its episode/label bytes')
    return {
        'path': _display_path(source),
        'episodes_sha256': _sha_file(episodes_path),
        'labels_sha256': _sha_file(labels_path),
        'admission_sha256': (_sha_file(admission_path)
                             if admission_path.is_file() else None),
        'episodes': episodes,
        'labels_by_episode': labels_data['episodes'],
        'episodes_format': episodes_data.get('format'),
        'labels_format': labels_data.get('format'),
    }


def prepare_update_dynamics(output, *, config_path=None, config=None,
                            source_prepared=None):
    """Copy exact prepared02 inputs and write deterministic update-dynamics pins.

    No tokenizer, model weight, neural forward or native computation is used.
    """
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Prepare output must be fresh or empty')
    config_path = Path(config_path or
                       PROJECT_ROOT / 'configs/memory_update_dynamics_v1.json')
    if config is None:
        config = json.loads(config_path.read_text())
    _require(config.get('format') == CONFIG_FORMAT,
             'Update-dynamics configuration format differs')
    _require(config.get('variant_names') == variant_names(),
             'Configured variants differ from the declared variant plan')
    _require(list(config.get('arms') or READ_ARMS) == list(READ_ARMS),
             'Configured read arms differ from the declared arms')
    source_path = source_prepared or config['source_prepared']['path']
    source = _read_source_prepared(_resolve_path(source_path))
    episodes = source['episodes']
    labels_by_episode = source['labels_by_episode']
    _require(source['episodes_format'] == 'replica.memory_matched.episodes.v1',
             'Source episodes are not prepared02 matched episodes')
    _require(source['labels_format'] == 'replica.memory_matched.labels.v1',
             'Source labels are not prepared02 verifier labels')
    expected = config.get('source_prepared') or {}
    if expected.get('episodes_sha256'):
        _require(source['episodes_sha256'] == expected['episodes_sha256'],
                 'Source episode bytes differ from the configuration pin')
    if expected.get('labels_sha256'):
        _require(source['labels_sha256'] == expected['labels_sha256'],
                 'Source label bytes differ from the configuration pin')

    output.mkdir(parents=True, exist_ok=True)
    episodes_path = output / 'episodes.json'
    labels_path = output / 'labels.json'
    episodes_path.write_bytes((_resolve_path(source_path) / 'episodes.json')
                              .read_bytes())
    labels_path.write_bytes((_resolve_path(source_path) / 'labels.json')
                            .read_bytes())
    layers = tuple(config.get('layers') or DEFAULT_LAYERS)
    engine = dict(config.get('engine') or DEFAULT_ENGINE)
    thresholds = _resolve_thresholds(config.get('thresholds'))
    limits = _resolve_limits(config.get('limits'))
    variants = resolve_variants(config.get('variant_names'))
    inventory = planned_inventory(episodes, labels_by_episode,
                                  variants=variants, layers=layers)
    inventory_path = output / 'inventory.json'
    _write_json(inventory_path, {
        'format': INVENTORY_FORMAT,
        'rows': [list(item) for item in inventory]})
    source_pins = _source_pins(PROJECT_ROOT / config['source']['model_path'],
                               PROJECT_ROOT / config['source']['model_lock'])
    extension = _extension_pins(
        PROJECT_ROOT / config['writer_extension']['path'])
    _require(extension['weights_sha256']
             == config['writer_extension']['weights_sha256'],
             'Frozen writer extension bytes differ from the configuration pin')
    implementation = _implementation_sha256()
    protocol = build_protocol(
        episodes, labels_by_episode, layers=layers, variants=variants,
        thresholds=thresholds, limits=limits, engine=engine,
        source=source_pins, writer_extension=extension,
        budget=config.get('budget'), inventory=inventory)
    protocol.update({
        'config_format': CONFIG_FORMAT,
        'config_path': _display_path(config_path),
        'config_sha256': _sha_file(config_path),
        'source_prepared': {
            key: source[key] for key in
            ('path', 'episodes_sha256', 'labels_sha256', 'admission_sha256')},
        'episodes': {
            'path': 'episodes.json', 'format': 'replica.memory_matched.episodes.v1',
            'sha256': _sha_file(episodes_path)},
        'labels': {
            'path': 'labels.json', 'format': 'replica.memory_matched.labels.v1',
            'sha256': _sha_file(labels_path)},
        'inventory': {
            'path': 'inventory.json', 'format': INVENTORY_FORMAT,
            'sha256': _sha_file(inventory_path),
            'rows': len(inventory)},
        'implementation_sha256': implementation,
        'declared_stage_order': ['exact_key_storage_upper_bound',
                                 'query_derived_addressing',
                                 'value_readout_controls'],
        'decision_rule': config.get('decision_rule'),
    })
    protocol_path = output / 'protocol.json'
    _write_json(protocol_path, protocol)
    manifest = {
        'format': PREPARED_FORMAT, 'created_at_utc': _utc_now(),
        'native_computation': False,
        'tokenizer_loaded': False, 'model_loaded': False,
        'episodes_sha256': _sha_file(episodes_path),
        'labels_sha256': _sha_file(labels_path),
        'inventory_sha256': _sha_file(inventory_path),
        'protocol_sha256': _sha_file(protocol_path),
        'config_sha256': _sha_file(config_path),
        'source_prepared': protocol['source_prepared'],
        'layers': list(layers), 'engine': engine,
        'variants': [variant['name'] for variant in variants],
        'planned_workload': protocol['planned_workload'],
        'source': source_pins, 'writer_extension': extension,
        'implementation_sha256': implementation,
        'budget': protocol['budget'],
    }
    _write_json(output / 'manifest.json', manifest)
    return manifest


def verify_update_dynamics_prepared(prepared):
    """Re-verify copied prepared02 bytes, plan regeneration and every pin."""
    prepared = Path(prepared)
    manifest_path = prepared / 'manifest.json'
    protocol_path = prepared / 'protocol.json'
    episodes_path = prepared / 'episodes.json'
    labels_path = prepared / 'labels.json'
    inventory_path = prepared / 'inventory.json'
    for path in (manifest_path, protocol_path, episodes_path, labels_path,
                 inventory_path):
        _require(path.is_file(), f'Prepared artifact is missing: {path.name}')
    manifest = json.loads(manifest_path.read_text())
    protocol = json.loads(protocol_path.read_text())
    data = json.loads(episodes_path.read_text())
    label_doc = json.loads(labels_path.read_text())
    stored_inventory = json.loads(inventory_path.read_text())
    _require(manifest.get('format') == PREPARED_FORMAT,
             'Prepared manifest format differs')
    _require(manifest.get('native_computation') is False
             and manifest.get('tokenizer_loaded') is False
             and manifest.get('model_loaded') is False,
             'Preparation must not perform native computation')
    _require(protocol.get('format') == PROTOCOL_FORMAT
             and data.get('format') == 'replica.memory_matched.episodes.v1'
             and label_doc.get('format') == 'replica.memory_matched.labels.v1',
             'Prepared formats differ')
    _require(manifest.get('episodes_sha256') == _sha_file(episodes_path)
             and manifest.get('labels_sha256') == _sha_file(labels_path)
             and manifest.get('inventory_sha256') == _sha_file(inventory_path)
             and manifest.get('protocol_sha256') == _sha_file(protocol_path),
             'Prepared bytes differ from their manifest pins')
    _require(protocol['episodes']['sha256'] == _sha_file(episodes_path)
             and protocol['labels']['sha256'] == _sha_file(labels_path)
             and protocol['inventory']['sha256'] == _sha_file(inventory_path),
             'Protocol pins differ from prepared bytes')
    episodes = data.get('episodes')
    labels_by_episode = label_doc.get('episodes')
    _require(isinstance(episodes, list) and isinstance(labels_by_episode, dict),
             'Prepared episodes or labels are malformed')
    _require(derive_matched_labels(episodes) == labels_by_episode,
             'Prepared verifier labels differ from the derived labels')
    layers = tuple(protocol.get('layers') or DEFAULT_LAYERS)
    variants = resolve_variants(protocol.get('variants') and
                                [item['name'] for item in protocol['variants']])
    _require(protocol['variants'] == variant_plan(),
             'Prepared variant plan differs from the declared plan')
    _require(manifest.get('variants') == [item['name'] for item in variants],
             'Manifest variant names differ from the protocol plan')
    inventory = planned_inventory(episodes, labels_by_episode,
                                  variants=variants, layers=layers)
    _require(stored_inventory.get('format') == INVENTORY_FORMAT
             and stored_inventory.get('rows') == [list(item) for item in inventory],
             'Prepared inventory differs from the regenerated plan')
    workload = planned_workload(episodes, labels_by_episode,
                                variants=variants, protocol=protocol)
    _require(protocol.get('planned_workload') == workload
             and manifest.get('planned_workload') == workload,
             'Prepared workload differs from the exact plan')
    config_path = Path(protocol.get('config_path') or
                       'configs/memory_update_dynamics_v1.json')
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    _require(config_path.is_file()
             and _sha_file(config_path) == protocol['config_sha256'],
             'Configuration bytes changed after preparation')
    current = _implementation_sha256()
    _require(protocol.get('implementation_sha256') == current
             and manifest.get('implementation_sha256') == current,
             'Update-dynamics implementation pins changed')
    source = protocol.get('source')
    _require(isinstance(source, dict) and source == _source_pins(
        source.get('model_path'), source.get('model_lock')),
        'Update-dynamics source pins changed')
    _require(manifest.get('source') == source,
             'Manifest/protocol source pins differ')
    extension = protocol.get('writer_extension')
    _require(isinstance(extension, dict)
             and _extension_pins(extension['path']) == extension,
             'Frozen writer extension pins changed')
    _require(manifest.get('writer_extension') == extension,
             'Manifest/protocol writer extension pins differ')
    source_prepared = protocol.get('source_prepared') or {}
    resolved_source = _resolve_path(source_prepared.get('path'))
    _require(resolved_source.is_dir(), 'Source prepared directory is missing')
    _require(_sha_file(resolved_source / 'episodes.json')
             == source_prepared.get('episodes_sha256')
             and _sha_file(resolved_source / 'labels.json')
             == source_prepared.get('labels_sha256'),
             'Source prepared02 bytes changed after preparation')
    if source_prepared.get('admission_sha256') is not None:
        _require(_sha_file(resolved_source / 'admission.json')
                 == source_prepared['admission_sha256'],
                 'Source prepared02 admission changed after preparation')
    return {
        'format': ADMISSION_FORMAT, 'status': 'admitted',
        'created_at_utc': _utc_now(),
        'prepared_manifest_sha256': _sha_file(manifest_path),
        'protocol_sha256': _sha_file(protocol_path),
        'episodes_sha256': _sha_file(episodes_path),
        'labels_sha256': _sha_file(labels_path),
        'inventory_sha256': _sha_file(inventory_path),
        'config_sha256': protocol['config_sha256'],
        'source_prepared': protocol['source_prepared'],
        'source': source, 'source_lock_sha256': source['model_lock_sha256'],
        'writer_extension': extension,
        'implementation_sha256': current,
        'layers': protocol['layers'], 'variants': [
            item['name'] for item in variants],
        'planned_workload': workload,
        'checks': {
            'manifest_format': True, 'protocol_bytes': True,
            'episode_bytes': True, 'label_bytes': True,
            'inventory_bytes': True, 'verifier_labels_regenerated': True,
            'variant_plan_regenerated': True, 'inventory_regenerated': True,
            'workload_bound': True, 'config_pin': True, 'source_pins': True,
            'implementation_pins': True, 'writer_extension_pins': True,
            'no_native_computation': True, 'tokenizer_not_loaded': True,
            'model_not_loaded': True,
        },
    }


def admit_update_dynamics_prepared(prepared, *, write_receipt=True):
    prepared = Path(prepared)
    try:
        receipt = verify_update_dynamics_prepared(prepared)
    except Exception as error:
        if write_receipt and prepared.is_dir():
            _write_json(prepared / 'admission.json', {
                'format': ADMISSION_FORMAT, 'status': 'rejected',
                'created_at_utc': _utc_now(),
                'error_type': type(error).__name__, 'error': str(error)})
        raise
    if write_receipt:
        _write_json(prepared / 'admission.json', receipt)
    return receipt

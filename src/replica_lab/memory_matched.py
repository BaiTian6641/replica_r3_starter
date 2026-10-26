"""Matched-episode localisation of the remaining learned-memory failure.

The rejected v5 writer is a frozen diagnostic artifact here: no optimizer is
constructed, no parameter may change and no training arm is run.  Episodes are
deterministic and matched: one episode inventory drives every stage and every
control, so a stage difference cannot come from different data.

Stages, evaluated in order against the *same* episodes and the *same* rank /
margin / false-recall definitions:

1. ``exact observed write key`` — storage upper bound.  The query vector is the
   exact key the writer stored for the target observation.  This is not an
   addressing claim.
2. ``independently rendered query-derived addressing`` — the query text is
   rendered from entity+attribute and encoded through the writer key path, then
   ranked against the stored write keys.
3. ``candidate/value discrimination`` — memory is read with the query-derived
   key and the readout is ranked against every live candidate value plus
   superseded negatives.  Measures strongest-distractor margins, false recalls,
   corrected-fact selection and unaffected retention.
4. ``native-language answer generation`` — the frozen production memory read is
   consumed by the existing ``MemorySession`` path.  Arms: fast state, exact
   source, full observed context and context removal.  Exact-source and
   full-context arms see only observed source text, never evaluator labels.

Gold answers and evaluator expectations never enter a writer, query encoder,
sense input or prompt: raw readout/answer rows are label-free and every label is
re-derived by a verifier from the observed records, then compared by the
independent scorer.  Mechanical save/reload parity is reported separately from
useful-memory metrics.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import statistics
import tempfile
import time

import torch

from .architecture.contracts import FastMemorySpec
from .memory_diagnosis import (
    MODE,
    PROJECT_ROOT,
    PURPOSE,
    SEALED_REVISION,
    SUBJECT_ID,
    _cosine_heads,
    _display_path,
    _extension_pins,
    _fresh_token,
    _freeze,
    _mps_memory,
    _opaque_token,
    _parameter_versions,
    _read_heads,
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
from .memory_reliability import render_fact_query
from .memory_system import EligibilityScope, MemoryRecord, RecordLedger
from .runtime.exact_memory import ExactMemoryReference
from .runtime.fact_memory import fact_source_text, validate_fact
from .runtime.fast_memory import FastMemoryScope
from .runtime.memory_representation import FACT_REPRESENTATION
from .runtime.persistent_fast_memory import PersistentFastMemory

PROTOCOL_FORMAT = 'replica.memory_matched.protocol.v1'
CONFIG_FORMAT = 'replica.memory_matched.config.v1'
EPISODES_FORMAT = 'replica.memory_matched.episodes.v1'
LABELS_FORMAT = 'replica.memory_matched.labels.v1'
PREPARED_FORMAT = 'replica.memory_matched.prepared.v1'
ADMISSION_FORMAT = 'replica.memory_matched.admission.v1'
RUN_FORMAT = 'replica.memory_matched.run.v1'
READOUT_FORMAT = 'replica.memory_matched.readout.v1'
ANSWER_FORMAT = 'replica.memory_matched.answer.v1'
CODEBOOK_FORMAT = 'replica.memory_matched.codebook.v1'
PROMPT_AUDIT_FORMAT = 'replica.memory_matched.prompt_audit.v1'
GENERATION_AUDIT_FORMAT = 'replica.memory_matched.generation_audit.v1'
MIGRATION_FORMAT = 'replica.memory_matched.implementation_migrations.v1'
MIGRATION_MODE = 'explicit_read_only_migration'
MIGRATION_FILENAME = 'implementation_migrations.json'

SPLITS = ('train', 'dev', 'heldout')
COMPLEXITIES = ('single', 'pair', 'triple', 'correction', 'interference')
COMPLEXITY_PLAN = {
    'single': {'facts': 1, 'distractors': 0, 'correction': False, 'near_misses': False},
    'pair': {'facts': 2, 'distractors': 1, 'correction': False, 'near_misses': False},
    'triple': {'facts': 3, 'distractors': 2, 'correction': False, 'near_misses': False},
    'correction': {'facts': 3, 'distractors': 1, 'correction': True, 'near_misses': False},
    'interference': {'facts': 4, 'distractors': 2, 'correction': True, 'near_misses': True},
}
INITIAL_DAY = 20260301
CORRECTION_DAY = 20260401
QUERY_FORMS = 4
DEFAULT_SEED = 20261025
DEFAULT_LAYER = 13
DEFAULT_FAMILIES_PER_SPLIT = {'train': 1, 'dev': 1, 'heldout': 1}
DEFAULT_VARIANTS = 2
DEFAULT_ENGINE = {'heads': 4, 'key_size': 64, 'value_size': 64,
                  'update_rule': 'sgdn2', 'depth_mode': 'shared',
                  'depth_writers': True}
STAGE_ORDER = (1, 2, 3, 4)
STAGE_NAMES = {
    1: 'exact_observed_key_storage',
    2: 'query_derived_addressing',
    3: 'candidate_value_discrimination',
    4: 'native_language_generation',
}
ANSWER_ARMS = ('memory', 'exact_source', 'full_context', 'context_removed')

DEFAULT_LIMITS = {
    'max_episodes': 64,
    'max_states': 160,
    'max_facts_per_episode': 8,
    'max_distractors_per_episode': 8,
    'max_queries': 20000,
    'max_generated_answers': 2400,
    'max_generated_tokens': 14400,
    'max_prompt_tokens': 256,
    'max_new_tokens': 6,
    'max_layers': 2,
    'max_preserved_failures': 500,
    'driver_gib': 50,
    'driver_bytes': 53687091200,
    'swap_growth_mib': 512,
    'minimum_free_percent': 15,
    'native_jobs': 1,
}

DEFAULT_THRESHOLDS = {
    # Shared value-space acceptance rule (stages 1 and 3).
    'match_cosine_floor': 0.5,
    'match_margin_floor': 0.05,
    # Stage 1: exact observed key storage upper bound.
    'storage_top1_floor': 0.85,
    'storage_mean_rank_max': 1.20,
    'storage_margin_floor': 0.02,
    'storage_false_recall_max': 0.05,
    'storage_cross_world_false_recall_max': 0.05,
    # Stage 2: query-derived addressing in key space.
    'addressing_cosine_floor': 0.5,
    'addressing_margin_floor': 0.05,
    'addressing_top1_floor': 0.60,
    'addressing_mean_rank_max': 1.50,
    'addressing_margin_mean_floor': 0.01,
    'addressing_false_recall_max': 0.30,
    'addressing_unknown_false_recall_max': 0.30,
    # Stage 3: candidate/value discrimination in value space.
    'candidate_top1_floor': 0.70,
    'candidate_mean_rank_max': 1.40,
    'candidate_margin_floor': 0.02,
    'candidate_false_recall_max': 0.10,
    'correction_top1_floor': 0.60,
    'unaffected_top1_floor': 0.70,
    'unknown_false_recall_max': 0.10,
    'superseded_selection_max': 0.10,
    'cross_world_selection_max': 0.05,
    'zero_false_recall_max': 0.0,
    # Stage 4: native-language generation.
    'answer_accuracy_floor': 0.60,
    'correction_answer_floor': 0.60,
    'unaffected_answer_floor': 0.60,
    'unknown_answer_false_max': 0.20,
    'memory_minus_removed_floor': 0.20,
    'exact_source_answer_floor': 0.60,
    'full_context_answer_floor': 0.60,
}

MATCHED_TEMPLATE = (
    'Use only the stored project facts supplied here. If the requested fact is '
    'absent, reply unknown. Reply with only the value.\n\n'
    'Query: {query}\n\nStored context:\n{observation}')

FORBIDDEN_INPUT_MARKERS = (
    'expected', 'expected_answer', 'gold', 'label', 'audit_label', 'split',
    'kind', 'role', 'complexity', 'variant', 'target_role',
)

_EPISODE_FIELDS = frozenset((
    'episode_id', 'world_id', 'family_id', 'split', 'complexity', 'variant',
    'facts', 'distractors', 'corrections', 'observations', 'queries'))
_FACT_FIELDS = frozenset(('record_id', 'entity', 'attribute', 'value',
                          'observed_at', 'source_world_id'))
_DISTRACTOR_FIELDS = frozenset(('record_id', 'entity', 'attribute', 'value',
                                'observed_at', 'origin_world_id',
                                'origin_record_id'))
_CORRECTION_FIELDS = frozenset(('record_id', 'fact_id', 'entity', 'attribute',
                                'value', 'observed_at', 'supersedes'))
_OBSERVATION_FIELDS = frozenset(('page_id', 'observed_at', 'date', 'lines'))
_QUERY_FIELDS = frozenset(('query_id', 'entity', 'attribute', 'form', 'text'))

WRITER_INPUT_FIELDS = {
    'episode': ('episode_id', 'world_id', 'facts', 'distractors', 'corrections',
                'observations', 'queries'),
    'fact': tuple(sorted(_FACT_FIELDS)),
    'distractor': tuple(sorted(_DISTRACTOR_FIELDS)),
    'correction': tuple(sorted(_CORRECTION_FIELDS)),
    'observation': ('page_id', 'observed_at', 'date', 'lines'),
    'query': ('query_id', 'entity', 'attribute', 'form', 'text'),
}

_IMPLEMENTATION_PATHS = (
    'src/replica_lab/memory_matched.py',
    'src/replica_lab/memory_reliability.py',
    'src/replica_lab/memory_diagnosis.py',
    'src/replica_lab/runtime/persistent_fast_memory.py',
    'src/replica_lab/runtime/fact_memory.py',
    'src/replica_lab/runtime/fast_memory.py',
    'src/replica_lab/runtime/memory_representation.py',
    'src/replica_lab/runtime/exact_memory.py',
    'src/replica_lab/memory_system/ledger.py',
    'src/replica_lab/memory_system/index.py',
    'src/replica_lab/memory_system/session.py',
    'src/replica_lab/architecture/contracts.py',
    'src/replica_lab/architecture/minicpm5.py',
    'src/replica_lab/architecture/ports.py',
    'src/replica_lab/architecture/loading.py',
    'src/replica_lab/neural.py',
    'src/replica_lab/model_adapter.py',
    'src/replica_lab/profiling.py',
    'src/replica_lab/contracts.py',
    'src/replica_lab/capacity_guard.py',
    'scripts/prepare_memory_matched.py',
    'scripts/run_memory_matched.py',
    'configs/memory_matched_v1.json',
)


# ---------------------------------------------------------------------------
# Small deterministic helpers
# ---------------------------------------------------------------------------

def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _sha_json(value):
    return _sha_text(json.dumps(value, sort_keys=True, allow_nan=False))


def _iso_date(day):
    _require(type(day) is int and 10000101 <= day <= 99991231,
             'Observation day must be an integer YYYYMMDD date')
    return f'{day // 10000:04d}-{(day // 100) % 100:02d}-{day % 100:02d}'


def _normalize_answer(text):
    if not isinstance(text, str):
        raise ValueError('Answer must be text')
    value = text.strip().strip('"\'`').strip()
    while value.endswith('.') and not value.endswith('..'):
        value = value[:-1].rstrip()
    return value.casefold()


def enforce_elapsed_deadline(started_monotonic, max_wall_seconds, *,
                             now=None, label='matched diagnostic'):
    """Enforce one declared monotonic wall-clock budget.

    ``now`` is an explicit test seam; production callers omit it.  A negative
    elapsed time is a clock/accounting error, not a pass.
    """
    _require(type(max_wall_seconds) is int and max_wall_seconds > 0,
             'Deadline must be a positive integer number of seconds')
    current = time.monotonic() if now is None else float(now)
    elapsed = current - float(started_monotonic)
    _require(elapsed >= 0, 'Monotonic elapsed time cannot be negative')
    if elapsed > max_wall_seconds:
        raise TimeoutError(
            f'{label} exceeded its declared {max_wall_seconds}s budget '
            f'(elapsed {elapsed:.3f}s)')
    return elapsed


def finalize_completed_report(result, *, started_monotonic, max_wall_seconds,
                              stop_profiler, write_report, clock=None,
                              label='native matched diagnostic'):
    """Finalize profiling/reporting under the same monotonic deadline.

    A completed status is written only after the profiler has stopped.  The
    deadline is then enforced again after report serialization, immediately
    before return.  Any crossing overwrites the artifact with a failed report so
    ``status='completed'`` and ``deadline_ok=False`` can never be persisted
    together.
    """
    _require(isinstance(result, dict), 'Finalization requires a report object')
    _require(type(max_wall_seconds) is int and max_wall_seconds > 0,
             'Deadline must be a positive integer number of seconds')
    _require(callable(stop_profiler) and callable(write_report),
             'Finalization requires profiler and report-writer callables')
    current = time.monotonic if clock is None else clock
    _require(callable(current), 'Finalization clock must be callable')

    def failed(error):
        elapsed = max(0.0, float(current()) - float(started_monotonic))
        result.update({
            'status': 'failed', 'deadline_seconds': max_wall_seconds,
            'elapsed_seconds': elapsed, 'deadline_ok': False,
            'finished_at_utc': _utc_now(),
            'error_type': type(error).__name__, 'error': str(error),
        })
        write_report(result)

    profile = stop_profiler()
    result['profile'] = profile
    profile_elapsed = float(profile.get('elapsed_seconds', 0.0))
    if profile_elapsed > max_wall_seconds:
        error = TimeoutError(
            f'{label} profiler elapsed {profile_elapsed:.3f}s exceeds '
            f'the declared {max_wall_seconds}s budget')
        failed(error)
        raise error
    try:
        elapsed = enforce_elapsed_deadline(
            started_monotonic, max_wall_seconds, now=current(),
            label=f'{label} profiler stop')
        result.update({
            'status': 'completed', 'deadline_seconds': max_wall_seconds,
            'elapsed_seconds': max(elapsed, profile_elapsed),
            'deadline_ok': True, 'finished_at_utc': _utc_now(),
        })
        write_report(result)
        final_elapsed = enforce_elapsed_deadline(
            started_monotonic, max_wall_seconds, now=current(),
            label=f'{label} report finalization')
        result.update({
            'final_elapsed_seconds': final_elapsed,
            'elapsed_seconds': max(final_elapsed, profile_elapsed),
            'deadline_ok': True, 'finished_at_utc': _utc_now(),
        })
        write_report(result)
        enforce_elapsed_deadline(
            started_monotonic, max_wall_seconds, now=current(),
            label=f'{label} return')
    except TimeoutError as error:
        failed(error)
        raise
    return result


def _mean(values):
    values = [value for value in values if value is not None]
    return (sum(values) / len(values)) if values else None


def _median(values):
    values = [value for value in values if value is not None]
    return statistics.median(values) if values else None


def _rate(count, total):
    return (count / total) if total else None


def _resolve_thresholds(thresholds=None):
    result = dict(DEFAULT_THRESHOLDS)
    if thresholds is None:
        return result
    _require(isinstance(thresholds, dict), 'Thresholds must be an object')
    unknown = set(thresholds) - set(DEFAULT_THRESHOLDS)
    _require(not unknown, f'Unknown matched-memory thresholds: {sorted(unknown)}')
    rank_bounds = {'storage_mean_rank_max', 'addressing_mean_rank_max',
                   'candidate_mean_rank_max'}
    for name, value in thresholds.items():
        _require(not isinstance(value, bool) and isinstance(value, (int, float)),
                 f'Threshold {name} must be numeric')
        lower, upper = ((1.0, 16.0) if name in rank_bounds else (0.0, 1.0))
        _require(lower <= float(value) <= upper,
                 f'Threshold {name} must be in [{lower:g}, {upper:g}]')
        result[name] = float(value)
    return result


def _resolve_limits(limits=None):
    result = dict(DEFAULT_LIMITS)
    if limits is None:
        return result
    _require(isinstance(limits, dict), 'Limits must be an object')
    unknown = set(limits) - set(DEFAULT_LIMITS)
    _require(not unknown, f'Unknown matched-memory limits: {sorted(unknown)}')
    for name, value in limits.items():
        _require(type(value) is int and value >= 0,
                 f'Limit {name} must be a nonnegative integer')
        result[name] = value
    return result


def _resolve_families(families_per_split=None):
    value = dict(DEFAULT_FAMILIES_PER_SPLIT if families_per_split is None
                 else families_per_split)
    _require(set(value) == set(SPLITS), 'Family plan must name every split exactly')
    for split, count in value.items():
        _require(type(count) is int and 1 <= count <= 16,
                 f'Family count for {split} must be in 1..16')
    return value


def _implementation_sha256():
    result = {}
    for relative in _IMPLEMENTATION_PATHS:
        path = PROJECT_ROOT / relative
        _require(path.is_file(), f'Implementation pin path is missing: {relative}')
        result[relative] = _sha_file(path)
    return result


def _is_sha256(value):
    return (isinstance(value, str) and len(value) == 64
            and all(character in '0123456789abcdef' for character in value))


def _require_sha256(value, label):
    _require(_is_sha256(value),
             f'{label} is not a lowercase SHA-256 digest')


def _implementation_drift(prepared_pins, current_pins):
    """Validate both pin maps and return the set of files that moved."""
    _require(isinstance(prepared_pins, dict) and bool(prepared_pins)
             and isinstance(current_pins, dict) and bool(current_pins),
             'Matched implementation pins are missing')
    _require(set(prepared_pins) == set(current_pins),
             'Current implementation map does not mirror the prepared pins')
    for relative in prepared_pins:
        _require_sha256(prepared_pins[relative], f'Prepared pin {relative}')
        _require_sha256(current_pins[relative], f'Current pin {relative}')
    return {relative for relative in prepared_pins
            if prepared_pins[relative] != current_pins[relative]}


def validate_implementation_migrations(sidecar, *, prepared_pins, current_pins,
                                       protocol_sha256, manifest_sha256):
    """Validate one explicit read-only implementation migration declaration.

    Pure: every input is already-loaded data, nothing is read from disk.  The
    sidecar pins the immutable prepared protocol and manifest bytes and must
    name exactly the implementation files whose current bytes differ from the
    prepared pins.  Missing, extra, stale or fabricated entries are rejected.
    Returns the concise migration report used by the admission receipt.
    """
    _require(isinstance(sidecar, dict)
             and set(sidecar) == {'format', 'mode', 'reason', 'prepared',
                                  'migrations'},
             'Implementation migration sidecar schema differs')
    _require(sidecar['format'] == MIGRATION_FORMAT,
             'Implementation migration sidecar format differs')
    _require(sidecar['mode'] == MIGRATION_MODE,
             'Implementation migrations must be explicit and read-only')
    _require(isinstance(sidecar['reason'], str) and sidecar['reason'].strip(),
             'Implementation migration sidecar lacks a reason')
    pinned = sidecar['prepared']
    _require(isinstance(pinned, dict)
             and set(pinned) == {'protocol_sha256', 'manifest_sha256'},
             'Implementation migration parent pins are malformed')
    _require_sha256(pinned['protocol_sha256'], 'Migration protocol_sha256')
    _require_sha256(pinned['manifest_sha256'], 'Migration manifest_sha256')
    _require(pinned['protocol_sha256'] == protocol_sha256,
             'Implementation migrations pin different prepared protocol bytes')
    _require(pinned['manifest_sha256'] == manifest_sha256,
             'Implementation migrations pin different prepared manifest bytes')
    drifted = _implementation_drift(prepared_pins, current_pins)
    migrations = sidecar['migrations']
    _require(isinstance(migrations, dict),
             'Implementation migrations are malformed')
    _require(set(migrations) == drifted,
             'Implementation migrations do not cover exactly the changed '
             'files')
    report = {}
    for relative in sorted(drifted):
        migration = migrations[relative]
        _require(isinstance(migration, dict)
                 and set(migration) == {'parent_sha256', 'current_sha256',
                                        'reason'},
                 f'Implementation migration is malformed: {relative}')
        _require_sha256(migration['parent_sha256'],
                        f'Migration parent {relative}')
        _require_sha256(migration['current_sha256'],
                        f'Migration current {relative}')
        _require(migration['parent_sha256'] == prepared_pins[relative],
                 f'Implementation migration parent is stale: {relative}')
        _require(migration['current_sha256'] == current_pins[relative],
                 f'Implementation migration current pin is stale: {relative}')
        _require(isinstance(migration['reason'], str)
                 and bool(migration['reason'].strip()),
                 f'Implementation migration lacks a reason: {relative}')
        report[relative] = dict(migration)
    return {'format': MIGRATION_FORMAT, 'mode': MIGRATION_MODE,
            'protocol_sha256': protocol_sha256,
            'manifest_sha256': manifest_sha256,
            'migrated': sorted(drifted), 'migrations': report}


def _verify_implementation_migrations(prepared, *, prepared_pins, current_pins,
                                      protocol_sha256, manifest_sha256):
    """Load and validate the sidecar; drift without one fails closed."""
    path = prepared / MIGRATION_FILENAME
    if not path.is_file():
        _require(not _implementation_drift(prepared_pins, current_pins),
                 'Matched implementation pins changed and no explicit '
                 f'migration sidecar exists: {MIGRATION_FILENAME}')
        return None
    report = validate_implementation_migrations(
        json.loads(path.read_text()), prepared_pins=prepared_pins,
        current_pins=current_pins, protocol_sha256=protocol_sha256,
        manifest_sha256=manifest_sha256)
    report['path'] = MIGRATION_FILENAME
    report['sha256'] = _sha_file(path)
    return report


# ---------------------------------------------------------------------------
# Episode construction
# ---------------------------------------------------------------------------

def _world_ids(seed, split, complexity, family_index, variant):
    material = f'{seed}:{split}:{complexity}:{family_index}:{variant}'
    world_id = 'world-' + _opaque_token('w', material)
    episode_id = 'epi-' + _opaque_token('e', material)
    family_id = 'family-' + _opaque_token('f', f'{seed}:{split}:{complexity}:{family_index}')
    return world_id, episode_id, family_id


def _pool(seed, split, prefix, count=16):
    return [_opaque_token(prefix, f'{seed}:{split}:{prefix}:{index}')
            for index in range(count)]


def _assign_split_key_blocks(seed, split, families, complexities, *, attempts=512):
    """Deterministically allocate split keys with guaranteed near-miss support.

    Interference families must have both a same-attribute and a same-entity
    independent-family fact available for their corrected target.  A shuffle
    that cannot provide both is rejected and the split is regenerated with the
    next deterministic attempt; there is no silent neutral fallback.
    """
    entities = _pool(seed, split, 'ent')
    attributes = _pool(seed, split, 'attr')
    base = [(entity, attribute) for entity in entities
            for attribute in attributes]
    required = sum(COMPLEXITY_PLAN[complexity]['facts']
                   for complexity in complexities) * families
    _require(required <= len(base),
             'Split key pool is too small for the declared family plan')
    for attempt in range(attempts):
        shuffle = random.Random(f'{seed}:{split}:keys:{attempt}')
        keys = list(base)
        shuffle.shuffle(keys)
        cursor = 0
        blocks = {}
        for complexity in complexities:
            count = COMPLEXITY_PLAN[complexity]['facts']
            for family_index in range(families):
                blocks[(complexity, family_index)] = list(
                    keys[cursor:cursor + count])
                cursor += count
        plan = {}
        failed = False
        for (complexity, family_index), block in blocks.items():
            target_index = family_index % len(block)
            if COMPLEXITY_PLAN[complexity]['near_misses']:
                chosen = None
                for candidate_index, target in enumerate(block):
                    support_attribute = any(
                        key[1] == target[1] and key[0] != target[0]
                        for (other_complexity, other_family), other in blocks.items()
                        if (other_complexity, other_family) != (complexity, family_index)
                        for key in other)
                    support_entity = any(
                        key[0] == target[0] and key[1] != target[1]
                        for (other_complexity, other_family), other in blocks.items()
                        if (other_complexity, other_family) != (complexity, family_index)
                        for key in other)
                    if support_attribute and support_entity:
                        chosen = candidate_index
                        break
                if chosen is None:
                    failed = True
                    break
                ordered = block[chosen:] + block[:chosen]
                if target_index:
                    ordered = ordered[-target_index:] + ordered[:-target_index]
                plan[(complexity, family_index)] = (ordered, target_index)
            else:
                plan[(complexity, family_index)] = (list(block), target_index)
        if not failed:
            return plan
    raise ValueError('Could not allocate split keys with required interference '
                     'near-miss support')


def generate_matched_episodes(*, seed=DEFAULT_SEED, families_per_split=None,
                             complexities=COMPLEXITIES, variants=DEFAULT_VARIANTS):
    """Build deterministic, split-separated, counterfactual matched episodes.

    The returned structures contain only observed inputs: opaque addresses,
    observed values, dated observation pages and rendered queries.  Expected
    answers, roles and split/kind labels are *not* stored; ``derive_state_labels``
    derives them from the observed records for evaluation only.
    """
    _require(type(seed) is int and seed >= 0, 'Seed must be a nonnegative integer')
    _require(tuple(complexities) and len(set(complexities)) == len(complexities),
             'Complexities must be a nonempty unique sequence')
    _require(all(item in COMPLEXITY_PLAN for item in complexities),
             'Unknown matched-memory complexity')
    _require(type(variants) is int and variants >= 2,
             'Matched episodes require at least two counterfactual variants')
    plan_families = _resolve_families(families_per_split)
    rng = random.Random(f'{seed}:matched')

    used_values = set()
    used_record_ids = set()
    worlds = []

    for split in SPLITS:
        attributes = _pool(seed, split, 'attr')
        key_plan = _assign_split_key_blocks(
            seed, split, plan_families[split], complexities)
        for complexity in complexities:
            plan = COMPLEXITY_PLAN[complexity]
            facts_count = plan['facts']
            for family_index in range(plan_families[split]):
                family_keys, correction_index = key_plan[(complexity, family_index)]
                unknown_entity = _opaque_token(
                    'ent', f'{seed}:{split}:{complexity}:{family_index}:unknown:entity')
                unknown_attribute = attributes[(family_index + facts_count) % len(attributes)]
                unknown_form = (family_index + facts_count) % QUERY_FORMS
                for variant in range(variants):
                    world_id, episode_id, family_id = _world_ids(
                        seed, split, complexity, family_index, variant)
                    worlds.append({
                        'split': split, 'complexity': complexity,
                        'family_index': family_index, 'variant': variant,
                        'world_id': world_id, 'episode_id': episode_id,
                        'family_id': family_id,
                        'family_keys': list(family_keys),
                        'correction_index': correction_index,
                        'unknown': {'entity': unknown_entity,
                                    'attribute': unknown_attribute,
                                    'form': unknown_form},
                    })

    # Facts and corrections, then independent-world distractors from another
    # family in the same split.
    for world in worlds:
        values = [
            _fresh_token('val', f"{world['world_id']}:fact:{index}:value", used_values)
            for index in range(len(world['family_keys']))]
        facts = []
        for index, ((entity, attribute), value) in enumerate(
                zip(world['family_keys'], values)):
            facts.append({
                'record_id': _fresh_token(
                    'rec', f"{world['world_id']}:fact:{index}", used_record_ids),
                'entity': entity, 'attribute': attribute, 'value': value,
                'observed_at': INITIAL_DAY, 'source_world_id': world['world_id'],
            })
        world['facts'] = facts
        if COMPLEXITY_PLAN[world['complexity']]['correction']:
            target = facts[world['correction_index']]
            world['corrections'] = [{
                'record_id': _fresh_token(
                    'rec', f"{world['world_id']}:correction", used_record_ids),
                'fact_id': target['record_id'],
                'entity': target['entity'], 'attribute': target['attribute'],
                'value': _fresh_token(
                    'val', f"{world['world_id']}:correction:value", used_values),
                'observed_at': CORRECTION_DAY, 'supersedes': target['record_id'],
            }]
        else:
            world['corrections'] = []

    by_split = {split: [world for world in worlds if world['split'] == split]
                for split in SPLITS}
    for world in worlds:
        plan = COMPLEXITY_PLAN[world['complexity']]
        target_keys = {(fact['entity'], fact['attribute']) for fact in world['facts']}
        target_values = {fact['value'] for fact in world['facts']}
        candidates = []
        for other in by_split[world['split']]:
            if other['family_id'] == world['family_id']:
                continue
            for fact in other['facts']:
                candidates.append({
                    'origin_world_id': other['world_id'],
                    'origin_record_id': fact['record_id'],
                    'entity': fact['entity'], 'attribute': fact['attribute'],
                    'value': fact['value'],
                })
        candidates.sort(key=lambda item: (item['origin_world_id'],
                                          item['origin_record_id']))
        if plan['near_misses']:
            correction_target = world['corrections'][0]
            criteria = [
                ('near_attribute', correction_target['attribute'],
                 correction_target['entity']),
                ('near_entity', correction_target['attribute'],
                 correction_target['entity']),
            ]
        else:
            criteria = [('neutral', None, None)] * plan['distractors']
        target_entities = {item[0] for item in target_keys}
        target_attributes = {item[1] for item in target_keys}
        chosen, chosen_origins = [], set()

        def _options(kind, attribute, entity):
            options = []
            for candidate in candidates:
                if candidate['origin_world_id'] in chosen_origins:
                    continue
                key = (candidate['entity'], candidate['attribute'])
                if key in target_keys or candidate['value'] in target_values:
                    continue
                if any((item['entity'], item['attribute']) == key for item in chosen):
                    continue
                if kind == 'near_attribute':
                    if (candidate['attribute'] != attribute
                            or candidate['entity'] == entity):
                        continue
                elif kind == 'near_entity':
                    if (candidate['entity'] != entity
                            or candidate['attribute'] == attribute):
                        continue
                elif kind == 'neutral':
                    if (candidate['entity'] in target_entities
                            or candidate['attribute'] in target_attributes):
                        continue
                options.append(candidate)
            return options

        for kind, attribute, entity in criteria:
            options = _options(kind, attribute, entity)
            if not options:
                if plan['near_misses'] and kind.startswith('near_'):
                    raise ValueError(
                        'Interference worlds require both same-attribute and '
                        'same-entity independent-world near misses')
                options = _options('any', None, None)
            if not options:
                raise ValueError('Insufficient independent-world distractor facts')
            choice = rng.choice(options)
            chosen_origins.add(choice['origin_world_id'])
            chosen.append(choice)
        if plan['near_misses']:
            target = world['corrections'][0]
            same_attribute = [item for item in chosen
                              if item['attribute'] == target['attribute']
                              and item['entity'] != target['entity']]
            same_entity = [item for item in chosen
                           if item['entity'] == target['entity']
                           and item['attribute'] != target['attribute']]
            _require(same_attribute and same_entity,
                     'Interference distractors must cover both near-miss types')
            _require(same_attribute[0]['origin_world_id']
                     != same_entity[0]['origin_world_id'],
                     'Near-miss distractors must come from distinct worlds')
        distractors = []
        for index, choice in enumerate(chosen):
            distractors.append({
                'record_id': _fresh_token(
                    'rec', f"{world['world_id']}:distractor:{index}", used_record_ids),
                'entity': choice['entity'], 'attribute': choice['attribute'],
                'value': choice['value'], 'observed_at': INITIAL_DAY,
                'origin_world_id': choice['origin_world_id'],
                'origin_record_id': choice['origin_record_id'],
            })
        world['distractors'] = distractors

    episodes = []
    for world in worlds:
        fact_lines = [fact_source_text({name: fact[name]
                                        for name in ('entity', 'attribute', 'value')})
                      for fact in world['facts']]
        distractor_lines = [fact_source_text({name: item[name]
                                              for name in ('entity', 'attribute', 'value')})
                            for item in world['distractors']]
        pages = [{
            'page_id': 'page-' + _opaque_token('p', f"{world['world_id']}:initial"),
            'observed_at': INITIAL_DAY, 'date': _iso_date(INITIAL_DAY),
            'lines': fact_lines + distractor_lines,
        }]
        if world['corrections']:
            correction = world['corrections'][0]
            pages.append({
                'page_id': 'page-' + _opaque_token('p', f"{world['world_id']}:correction"),
                'observed_at': CORRECTION_DAY, 'date': _iso_date(CORRECTION_DAY),
                'lines': [fact_source_text(
                    {name: correction[name]
                     for name in ('entity', 'attribute', 'value')}, correction=True)],
            })
        queries = []
        for index, fact in enumerate(world['facts']):
            form = (world['family_index'] + index) % QUERY_FORMS
            queries.append({
                'query_id': f'q{index}', 'entity': fact['entity'],
                'attribute': fact['attribute'], 'form': form,
                'text': render_fact_query(fact['entity'], fact['attribute'], form),
            })
        unknown = world['unknown']
        queries.append({
            'query_id': 'unknown', 'entity': unknown['entity'],
            'attribute': unknown['attribute'], 'form': unknown['form'],
            'text': render_fact_query(unknown['entity'], unknown['attribute'],
                                      unknown['form']),
        })
        episodes.append({
            'episode_id': world['episode_id'], 'world_id': world['world_id'],
            'family_id': world['family_id'], 'split': world['split'],
            'complexity': world['complexity'], 'variant': world['variant'],
            'facts': world['facts'], 'distractors': world['distractors'],
            'corrections': world['corrections'], 'observations': pages,
            'queries': queries,
        })
    validate_matched_episodes(episodes, families_per_split=plan_families,
                              complexities=tuple(complexities), variants=variants)
    return episodes


def validate_matched_episodes(episodes, *, families_per_split=None,
                              complexities=COMPLEXITIES, variants=DEFAULT_VARIANTS,
                              require_novel_values=True):
    """Reject duplicate addresses, split leakage, labels and malformed oracles."""
    _require(isinstance(episodes, list) and episodes,
             'Matched episodes must be a nonempty list')
    plan_families = _resolve_families(families_per_split)
    complexities = tuple(complexities)
    _require(complexities and len(set(complexities)) == len(complexities)
             and set(complexities) <= set(COMPLEXITY_PLAN),
             'Validation complexities differ from the declared plan')
    _require(type(variants) is int and variants >= 2,
             'Matched episodes require at least two variants')

    episode_ids, world_ids = set(), set()
    record_ids, values = set(), set()
    key_owners = {}
    families = {}
    query_forms = set()
    counts = {split: {name: 0 for name in complexities} for split in SPLITS}

    for episode in episodes:
        _require(isinstance(episode, dict) and set(episode) == _EPISODE_FIELDS,
                 'Episode has unexpected fields')
        episode_id, world_id = episode['episode_id'], episode['world_id']
        _require(isinstance(episode_id, str) and episode_id
                 and episode_id not in episode_ids, 'Episode IDs must be unique')
        episode_ids.add(episode_id)
        _require(isinstance(world_id, str) and world_id
                 and world_id not in world_ids, 'World IDs must be unique')
        world_ids.add(world_id)
        _require(episode['split'] in SPLITS, 'Episode split differs')
        _require(episode['complexity'] in complexities,
                 'Episode complexity differs')
        _require(type(episode['variant']) is int
                 and 0 <= episode['variant'] < variants, 'Episode variant differs')
        plan = COMPLEXITY_PLAN[episode['complexity']]
        families.setdefault(episode['family_id'], []).append(episode)
        counts[episode['split']][episode['complexity']] += 1

        facts = episode['facts']
        _require(isinstance(facts, list) and len(facts) == plan['facts'],
                 'Fact inventory differs from the declared complexity')
        state_keys, state_values = set(), set()
        for fact in facts:
            _require(isinstance(fact, dict) and set(fact) == _FACT_FIELDS,
                     'Fact has unexpected fields or labels')
            validate_fact({name: fact[name] for name in ('entity', 'attribute', 'value')})
            _require(fact['source_world_id'] == world_id,
                     'Source fact belongs to another world')
            _require(fact['record_id'] not in record_ids, 'Record IDs must be unique')
            record_ids.add(fact['record_id'])
            key = (fact['entity'], fact['attribute'])
            _require(key not in state_keys, 'Observed addresses must be unique in a world')
            state_keys.add(key)
            _require(fact['value'] not in state_values,
                     'Observed values must be unique in a world')
            state_values.add(fact['value'])
            _require(fact['value'] not in values, 'Observed values must be globally disjoint')
            values.add(fact['value'])
            owner = key_owners.setdefault(key, (episode['family_id'], world_id))
            _require(owner[0] == episode['family_id'],
                     'Observed address is shared by two counterfactual-unrelated worlds')

        distractors = episode['distractors']
        _require(isinstance(distractors, list)
                 and len(distractors) == plan['distractors'],
                 'Distractor inventory differs from the declared complexity')
        for item in distractors:
            _require(isinstance(item, dict) and set(item) == _DISTRACTOR_FIELDS,
                     'Distractor has unexpected fields or labels')
            validate_fact({name: item[name] for name in ('entity', 'attribute', 'value')})
            _require(item['origin_world_id'] != world_id,
                     'Distractors must come from an independent world')
            _require(item['record_id'] not in record_ids,
                     'Distractor record IDs must be unique')
            record_ids.add(item['record_id'])
            key = (item['entity'], item['attribute'])
            _require(key not in state_keys,
                     'Distractor address collides with the observed state')
            state_keys.add(key)
            _require(item['value'] not in state_values,
                     'Distractor value collides with the observed state')
            state_values.add(item['value'])

        corrections = episode['corrections']
        _require(isinstance(corrections, list)
                 and len(corrections) == (1 if plan['correction'] else 0),
                 'Correction inventory differs from the declared complexity')
        corrected_ids = set()
        for correction in corrections:
            _require(isinstance(correction, dict) and set(correction) == _CORRECTION_FIELDS,
                     'Correction has unexpected fields or labels')
            target = next((fact for fact in facts
                           if fact['record_id'] == correction['fact_id']), None)
            _require(target is not None, 'Correction must supersede an observed fact')
            _require(correction['supersedes'] == target['record_id'],
                     'Correction supersession differs from its target')
            _require((correction['entity'], correction['attribute'])
                     == (target['entity'], target['attribute']),
                     'Correction address differs from its target')
            _require(correction['observed_at'] > target['observed_at'],
                     'Correction must be observed after its target')
            validate_fact({name: correction[name]
                           for name in ('entity', 'attribute', 'value')})
            _require(correction['value'] not in values,
                     'Correction requires a fresh observed value')
            values.add(correction['value'])
            _require(correction['record_id'] not in record_ids,
                     'Correction record IDs must be unique')
            record_ids.add(correction['record_id'])
            corrected_ids.add(target['record_id'])
        _require(len(facts) - len(corrected_ids) >= 1,
                 'At least one unaffected fact is required')
        if plan['near_misses']:
            _require(corrections, 'Interference requires a dated correction target')
            target = next(fact for fact in facts
                          if fact['record_id'] == corrections[0]['fact_id'])
            same_attribute = [item for item in distractors
                              if item['attribute'] == target['attribute']
                              and item['entity'] != target['entity']]
            same_entity = [item for item in distractors
                           if item['entity'] == target['entity']
                           and item['attribute'] != target['attribute']]
            _require(same_attribute and same_entity,
                     'Interference requires same-attribute and same-entity '
                     'independent-world near misses')
            _require(same_attribute[0]['origin_world_id']
                     != same_entity[0]['origin_world_id'],
                     'Interference near misses must come from distinct worlds')

        pages = episode['observations']
        _require(isinstance(pages, list)
                 and len(pages) == (2 if corrections else 1),
                 'Observation page inventory differs')
        expected_lines = [fact_source_text({name: fact[name]
                                            for name in ('entity', 'attribute', 'value')})
                          for fact in facts + distractors]
        if corrections:
            correction = corrections[0]
            expected_lines += [fact_source_text(
                {name: correction[name]
                 for name in ('entity', 'attribute', 'value')}, correction=True)]
        seeded_lines = []
        for page in pages:
            _require(isinstance(page, dict) and set(page) == _OBSERVATION_FIELDS,
                     'Observation page has unexpected fields')
            _require(page['date'] == _iso_date(page['observed_at']),
                     'Observation date differs from its timestamp')
            _require(isinstance(page['lines'], list) and page['lines'],
                     'Observation page must contain observed lines')
            seeded_lines.extend(page['lines'])
        _require(seeded_lines == expected_lines,
                 'Rendered observations differ from the structured oracle')

        queries = episode['queries']
        _require(isinstance(queries, list) and len(queries) == len(facts) + 1,
                 'Query inventory differs')
        query_ids, query_keys, unknown = set(), set(), []
        for query in queries:
            _require(isinstance(query, dict) and set(query) == _QUERY_FIELDS,
                     'Query has unexpected fields or labels')
            _require(query['query_id'] not in query_ids, 'Query IDs must be unique')
            query_ids.add(query['query_id'])
            _require(type(query['form']) is int and 0 <= query['form'] < QUERY_FORMS,
                     'Query form differs')
            validate_fact({'entity': query['entity'], 'attribute': query['attribute'],
                           'value': 'observed'})
            _require(query['text'] == render_fact_query(
                query['entity'], query['attribute'], query['form']),
                'Query text differs from the rendered observed address')
            query_forms.add(query['form'])
            key = (query['entity'], query['attribute'])
            if key in state_keys:
                query_keys.add(key)
            else:
                unknown.append(query)
        _require(query_keys == {(fact['entity'], fact['attribute']) for fact in facts},
                 'Queries must cover every observed fact')
        _require(len(unknown) == 1,
                 'Exactly one unobserved-address query is required')
        unknown_key = (unknown[0]['entity'], unknown[0]['attribute'])
        _require(unknown_key not in key_owners,
                 'The unobserved query address is observed elsewhere in the dataset')

    for split in SPLITS:
        _require(any(episode['split'] == split for episode in episodes),
                 'Every split requires at least one episode')
    for family_id, members in families.items():
        _require(len({episode['split'] for episode in members}) == 1,
                 'A family must stay inside one split')
        _require(len({episode['complexity'] for episode in members}) == 1,
                 'A family must keep one complexity')
        _require(sorted(episode['variant'] for episode in members)
                 == list(range(variants)), 'Every family requires every variant')
        first = members[0]
        first_keys = [(fact['entity'], fact['attribute']) for fact in first['facts']]
        first_queries = [(query['query_id'], query['entity'], query['attribute'],
                          query['form'], query['text']) for query in first['queries']]
        for other in members[1:]:
            _require([(fact['entity'], fact['attribute']) for fact in other['facts']]
                     == first_keys, 'Counterfactual variants must share addresses')
            _require([(query['query_id'], query['entity'], query['attribute'],
                       query['form'], query['text']) for query in other['queries']]
                     == first_queries,
                     'Counterfactual variants must share identical queries')
            _require({fact['value'] for fact in other['facts']}
                     .isdisjoint({fact['value'] for fact in first['facts']}),
                     'Counterfactual variants must carry different values')

    # Distractor provenance must be an observed fact of a same-split world.
    by_world = {episode['world_id']: episode for episode in episodes}
    for episode in episodes:
        for item in episode['distractors']:
            origin = by_world.get(item['origin_world_id'])
            _require(origin is not None and origin['split'] == episode['split'],
                     'Distractor origin world must exist in the same split')
            source = next((fact for fact in origin['facts']
                           if fact['record_id'] == item['origin_record_id']), None)
            _require(source is not None
                     and (source['entity'], source['attribute'], source['value'])
                     == (item['entity'], item['attribute'], item['value']),
                     'Distractor provenance differs from the origin fact')
            _require(origin['family_id'] != episode['family_id'],
                     'Distractors must come from an independent family')

    if require_novel_values:
        heldout = {fact['value'] for episode in episodes
                   if episode['split'] == 'heldout'
                   for fact in episode['facts'] + episode['distractors']}
        heldout |= {correction['value'] for episode in episodes
                    if episode['split'] == 'heldout'
                    for correction in episode['corrections']}
        prior = {fact['value'] for episode in episodes
                 if episode['split'] != 'heldout'
                 for fact in episode['facts'] + episode['distractors']}
        prior |= {correction['value'] for episode in episodes
                  if episode['split'] != 'heldout'
                  for correction in episode['corrections']}
        _require(heldout.isdisjoint(prior),
                 'Heldout observed values overlap train/development values')

    _require(query_forms == set(range(QUERY_FORMS)),
             'The dataset must exercise every declared query form')
    for split in SPLITS:
        for complexity in complexities:
            declared = plan_families[split]
            observed = {episode['family_id'] for episode in episodes
                        if episode['split'] == split
                        and episode['complexity'] == complexity}
            _require(len(observed) == declared,
                     'Family plan differs from the declared split/complexity plan')

    audit = {
        'episodes': len(episodes),
        'worlds': len(world_ids),
        'families': len(families),
        'splits': {split: sum(1 for episode in episodes if episode['split'] == split)
                   for split in SPLITS},
        'per_split_complexity': counts,
        'counterfactual_variants': variants,
        'query_forms': sorted(query_forms),
        'values_globally_unique': True,
        'heldout_values_absent_from_prior_splits': bool(require_novel_values),
        'states': sum(len(episode_states(episode)) for episode in episodes),
        'queries': sum(len(episode['queries']) for episode in episodes),
        'distractor_origins_same_split': True,
        'label_free_observed_inputs': True,
    }
    _assert_label_free(episodes)
    return audit


def _assert_label_free(episodes):
    forbidden = tuple(marker for marker in FORBIDDEN_INPUT_MARKERS)
    for episode in episodes:
        payload = json.dumps(writer_inputs(episode), sort_keys=True).casefold()
        for marker in forbidden:
            _require(marker not in payload,
                     f'Observed writer input carries a forbidden label marker: {marker}')
        for group in ('facts', 'distractors', 'corrections', 'observations', 'queries'):
            for item in episode[group]:
                _require(not any(key in item for key in
                                 ('expected', 'expected_answer', 'gold', 'label',
                                  'audit_label', 'kind', 'role', 'split')),
                         f'{group} item carries an evaluation label')


def writer_inputs(episode):
    """Project exactly what a writer/sense path may consume from an episode."""
    return {
        'episode_id': episode['episode_id'],
        'world_id': episode['world_id'],
        'facts': [{name: fact[name] for name in WRITER_INPUT_FIELDS['fact']}
                  for fact in episode['facts']],
        'distractors': [{name: item[name]
                         for name in WRITER_INPUT_FIELDS['distractor']}
                        for item in episode['distractors']],
        'corrections': [{name: correction[name]
                         for name in WRITER_INPUT_FIELDS['correction']}
                        for correction in episode['corrections']],
        'observations': [{name: page[name]
                          for name in WRITER_INPUT_FIELDS['observation']}
                         for page in episode['observations']],
        'queries': [{name: query[name] for name in WRITER_INPUT_FIELDS['query']}
                    for query in episode['queries']],
    }


# ---------------------------------------------------------------------------
# Verifier-derived labels and state views
# ---------------------------------------------------------------------------

def episode_states(episode):
    """Return the dated evaluation states in order; labels never enter inputs."""
    states = [{'name': 'initial', 'cutoff': INITIAL_DAY,
               'observations': [page for page in episode['observations']
                                if page['observed_at'] <= INITIAL_DAY]}]
    if episode['corrections']:
        states.append({'name': 'corrected', 'cutoff': CORRECTION_DAY,
                       'observations': [page for page in episode['observations']
                                        if page['observed_at'] <= CORRECTION_DAY]})
    return states


def _record_views(episode, state):
    """Eligible records at a cutoff, with superseded originals separated."""
    cutoff = state['cutoff']
    records = []
    event = 0
    for fact in episode['facts']:
        if fact['observed_at'] <= cutoff:
            records.append({
                'record_id': fact['record_id'], 'entity': fact['entity'],
                'attribute': fact['attribute'], 'value': fact['value'],
                'observed_at': fact['observed_at'], 'event_at': event,
                'kind': 'source_fact', 'supersedes': None,
                'origin_world_id': fact['source_world_id'],
            })
            event += 1
    for item in episode['distractors']:
        if item['observed_at'] <= cutoff:
            records.append({
                'record_id': item['record_id'], 'entity': item['entity'],
                'attribute': item['attribute'], 'value': item['value'],
                'observed_at': item['observed_at'], 'event_at': event,
                'kind': 'source_fact', 'supersedes': None,
                'origin_world_id': item['origin_world_id'],
            })
            event += 1
    for correction in episode['corrections']:
        if correction['observed_at'] <= cutoff:
            records.append({
                'record_id': correction['record_id'],
                'entity': correction['entity'],
                'attribute': correction['attribute'],
                'value': correction['value'],
                'observed_at': correction['observed_at'], 'event_at': event,
                'kind': 'source_correction',
                'supersedes': correction['supersedes'],
                'origin_world_id': episode['world_id'],
            })
            event += 1
    records.sort(key=lambda item: (item['observed_at'], item['event_at']))
    superseded = {record['supersedes'] for record in records
                  if record['kind'] == 'source_correction'}
    live = [record for record in records if record['record_id'] not in superseded]
    negative = [record for record in records if record['record_id'] in superseded]
    return tuple(records), tuple(live), tuple(negative)


def _entry_role(record, state, episode):
    if record['kind'] == 'source_correction':
        return 'correction_target'
    if record['origin_world_id'] != episode['world_id']:
        return 'distractor'
    if state['name'] == 'corrected':
        return 'retained'
    return 'source'


def _entry_public(record, role, negative=False):
    return {
        'record_id': record['record_id'],
        'entity': record['entity'],
        'attribute': record['attribute'],
        'value': record['value'],
        'role': role,
        'source_family': ('independent_world'
                          if role == 'distractor' else
                          'source' if not negative else 'source'),
        'origin_world_id': record['origin_world_id'],
        'observed_at': record['observed_at'],
        'live': not negative,
    }


def derive_state_labels(episode, state):
    """Derive evaluation labels from observed records only.

    The result is evaluation-only.  It is computed from the observed facts,
    distractors, dated corrections and queries; no stored gold value and no
    evaluator expectation participates.
    """
    records, live, negative = _record_views(episode, state)
    live_entries = []
    role_by_id = {}
    for record in live:
        role = _entry_role(record, state, episode)
        role_by_id[record['record_id']] = role
        live_entries.append(_entry_public(record, role))
    negative_entries = []
    for record in negative:
        role_by_id[record['record_id']] = 'superseded_negative'
        negative_entries.append(_entry_public(record, 'superseded_negative',
                                              negative=True))
    live_by_key = {(entry['entity'], entry['attribute']): entry
                   for entry in live_entries}
    targets, corrected_ids, unaffected_ids = {}, [], []
    for query in episode['queries']:
        entry = live_by_key.get((query['entity'], query['attribute']))
        if entry is None:
            _require(query['query_id'] == 'unknown',
                     'Only the declared unobserved query may be absent')
            targets[query['query_id']] = {
                'expected': 'unknown', 'kind': 'unknown',
                'target_entry_id': None, 'target_role': 'unknown',
                'target_source_family': 'none', 'query_form': query['form'],
                'query_text_sha256': _sha_json(query['text']),
            }
            continue
        expected = entry['value']
        _require(isinstance(expected, str) and expected,
                 'Observed target carries no value')
        targets[query['query_id']] = {
            'expected': expected, 'kind': 'fact',
            'target_entry_id': entry['record_id'],
            'target_role': entry['role'],
            'target_source_family': entry['source_family'],
            'query_form': query['form'],
            'query_text_sha256': _sha_json(query['text']),
        }
        if entry['role'] == 'correction_target':
            corrected_ids.append(query['query_id'])
        elif entry['role'] == 'retained':
            unaffected_ids.append(query['query_id'])
    unknown_ids = [query['query_id'] for query in episode['queries']
                   if targets[query['query_id']]['kind'] == 'unknown']
    _require(len(unknown_ids) == 1, 'Exactly one unobserved query is required')
    _require(len(corrected_ids) <= 1, 'At most one correction target is allowed')
    if state['name'] == 'corrected' and episode['corrections']:
        _require(len(corrected_ids) == 1,
                 'The corrected state must expose exactly one correction target')
        _require(len(unaffected_ids) >= 1,
                 'The corrected state must retain at least one unaffected fact')
    return {
        'episode_id': episode['episode_id'], 'world_id': episode['world_id'],
        'family_id': episode['family_id'], 'split': episode['split'],
        'complexity': episode['complexity'], 'variant': episode['variant'],
        'state': state['name'], 'cutoff': state['cutoff'],
        'observed_record_ids': [record['record_id'] for record in records],
        'live_entries': live_entries,
        'negative_entries': negative_entries,
        'targets': targets,
        'corrected_query_ids': corrected_ids,
        'unaffected_query_ids': unaffected_ids,
        'unknown_query_ids': unknown_ids,
    }


def derive_matched_labels(episodes):
    labels = {}
    for episode in episodes:
        per_state = {}
        for state in episode_states(episode):
            per_state[state['name']] = derive_state_labels(episode, state)
        labels[episode['episode_id']] = per_state
    return labels


def state_observation_sha256(episode, state):
    return _sha_json([{'page_id': page['page_id'],
                       'observed_at': page['observed_at'],
                       'date': page['date'], 'lines': list(page['lines'])}
                      for page in state['observations']])


def _state_scope(episode, state):
    return EligibilityScope(SUBJECT_ID, episode['world_id'], PURPOSE,
                            state['cutoff'], MODE, None)


# ---------------------------------------------------------------------------
# Workload and row inventory
# ---------------------------------------------------------------------------

def _live_keys(episode, state):
    _, live, _ = _record_views(episode, state)
    return {(record['entity'], record['attribute']) for record in live}


def matched_inventory(episodes):
    """Exact expected raw-row inventory, independent of any model allocation."""
    inventory = []
    for episode in episodes:
        for state in episode_states(episode):
            _, live, negative = _record_views(episode, state)
            live_keys = {(record['entity'], record['attribute'])
                         for record in live}
            fact_queries = [query for query in episode['queries']
                            if (query['entity'], query['attribute']) in live_keys]
            for query in fact_queries:
                inventory.append((1, episode['episode_id'], state['name'],
                                  query['query_id'], 'exact_key'))
            for control in ('zero', 'wrong_key', 'cross_world'):
                inventory.append((1, episode['episode_id'], state['name'],
                                  f'control:{control}', 'control'))
            for query in episode['queries']:
                inventory.append((2, episode['episode_id'], state['name'],
                                  query['query_id'], 'query_key'))
            for control in ('zero', 'cross_world'):
                inventory.append((2, episode['episode_id'], state['name'],
                                  f'control:{control}', 'control'))
            for query in episode['queries']:
                inventory.append((3, episode['episode_id'], state['name'],
                                  query['query_id'], 'query_value'))
            for control in ('zero', 'cross_world'):
                inventory.append((3, episode['episode_id'], state['name'],
                                  f'control:{control}', 'control'))
            if negative:
                corrected = [
                    query for query in episode['queries']
                    if (query['entity'], query['attribute'])
                    in {(entry['entity'], entry['attribute'])
                        for entry in derive_state_labels(episode, state)['live_entries']
                        if entry['role'] == 'correction_target'}]
                _require(len(corrected) == 1,
                         'The corrected state must expose one correction target')
                inventory.append((3, episode['episode_id'], state['name'],
                                  'control:superseded_probe', 'control'))
            for query in episode['queries']:
                for arm in ANSWER_ARMS:
                    inventory.append((4, episode['episode_id'], state['name'],
                                      query['query_id'], arm))
    _require(len(inventory) == len(set(inventory)),
             'Matched inventory contains duplicate rows')
    return sorted(inventory)


def matched_workload(episodes, protocol=None):
    limits = _resolve_limits((protocol or {}).get('limits'))
    inventory = matched_inventory(episodes)
    readout_rows = sum(1 for item in inventory if item[0] in (1, 2, 3))
    answer_rows = sum(1 for item in inventory if item[0] == 4)
    max_new = int((protocol or {}).get('generation', {}).get(
        'max_new_tokens', DEFAULT_LIMITS['max_new_tokens']))
    workload = {
        'episodes': len(episodes),
        'states': sum(len(episode_states(episode)) for episode in episodes),
        'queries': sum(len(episode['queries']) for episode in episodes),
        'readout_rows': readout_rows,
        'answer_rows': answer_rows,
        'planned_queries': readout_rows + answer_rows,
        'planned_generated_tokens': answer_rows * max_new,
        'per_stage': {
            '1': sum(1 for item in inventory if item[0] == 1),
            '2': sum(1 for item in inventory if item[0] == 2),
            '3': sum(1 for item in inventory if item[0] == 3),
            '4': answer_rows,
        },
    }
    _require(len(episodes) <= limits['max_episodes'],
             'Episode count exceeds the declared bound')
    _require(workload['states'] <= limits['max_states'],
             'State count exceeds the declared bound')
    _require(all(len(episode['facts']) <= limits['max_facts_per_episode']
                 for episode in episodes),
             'A matched episode exceeds max_facts_per_episode')
    _require(all(len(episode['distractors']) <= limits['max_distractors_per_episode']
                 for episode in episodes),
             'A matched episode exceeds max_distractors_per_episode')
    _require(limits['max_layers'] >= 1,
             'At least one layer must be declared available')
    _require(workload['planned_queries'] <= limits['max_queries'],
             'Planned query rows exceed max_queries')
    _require(answer_rows <= limits['max_generated_answers'],
             'Planned generated answers exceed the declared bound')
    _require(workload['planned_generated_tokens'] <= limits['max_generated_tokens'],
             'Planned generated tokens exceed the declared bound')
    return workload


# ---------------------------------------------------------------------------
# Writer state, codebook and raw readouts
# ---------------------------------------------------------------------------

def _fact_of(record):
    return {'entity': record['entity'], 'attribute': record['attribute'],
            'value': record['value']}


def _memory_history(episode, state):
    records, _, _ = _record_views(episode, state)
    return records


def _build_memory(model, tokenizer, episode, state):
    scope = FastMemoryScope(SUBJECT_ID, episode['world_id'], PURPOSE, MODE, 0)
    memory = PersistentFastMemory(model, tokenizer, scope, training=False,
                                  representation=FACT_REPRESENTATION)
    for record in _memory_history(episode, state):
        memory.append_fact(record['record_id'], _fact_of(record),
                           correction=record['kind'] == 'source_correction')
    return memory


def _build_codebook(model, tokenizer, episode, state, memory, *, layer):
    _, live, negative = _record_views(episode, state)
    entries = {}
    for record in tuple(live) + tuple(negative):
        binding = observed_binding(model, tokenizer, _fact_of(record),
                                   layer=layer)
        value_vector = binding.values[0, 0].detach().float().cpu()
        write_key = binding.keys[0, 0].detach().float().cpu()
        _require(bool(torch.isfinite(value_vector).all())
                 and bool(torch.isfinite(write_key).all()),
                 'Writer produced a non-finite key/value vector')
        entries[record['record_id']] = {
            'value_vector': value_vector.tolist(),
            'write_key': write_key.tolist(),
            'value_sha256': _sha_json(value_vector.tolist()),
            'write_key_sha256': _sha_json(write_key.tolist()),
            'value_norm': float(torch.linalg.vector_norm(value_vector)),
            'key_norm': float(torch.linalg.vector_norm(write_key)),
        }
    _require(entries, 'Codebook cannot be empty')
    return {
        'format': CODEBOOK_FORMAT, 'episode_id': episode['episode_id'],
        'world_id': episode['world_id'], 'state': state['name'],
        'cutoff': state['cutoff'], 'layer': layer,
        'observation_sha256': state_observation_sha256(episode, state),
        'memory_history_sha256': _sha_json([
            {'record_id': record['record_id'],
             'source': fact_source_text(
                 _fact_of(record),
                 correction=record['kind'] == 'source_correction')}
            for record in _memory_history(episode, state)]),
        'state_digest': memory.read().state_digest,
        'entries': entries,
        'live_entry_ids': sorted(record['record_id'] for record in live),
        'negative_entry_ids': sorted(record['record_id'] for record in negative),
    }


def _key_tensor(entry):
    value = torch.tensor(entry['write_key'], dtype=torch.float32)
    return value.view(1, 1, *value.shape)


def _value_tensor(entry):
    return torch.tensor(entry['value_vector'], dtype=torch.float32)


def _readout_row(*, stage, arm, episode, state, case_id, query_id,
                 query_text_sha256, codebook, readout, readout_space,
                 status='ok', unavailable_reason=None, owner_entry_id=None,
                 context_source_ids=None):
    row = {
        'format': READOUT_FORMAT, 'stage': stage,
        'episode_id': episode['episode_id'], 'world_id': episode['world_id'],
        'state': state['name'], 'cutoff': state['cutoff'],
        'case_id': case_id, 'query_id': query_id, 'arm': arm,
        'codebook_key': f"{episode['episode_id']}/{state['name']}",
        'layer': codebook['layer'],
        'observation_sha256': codebook['observation_sha256'],
        'query_text_sha256': query_text_sha256,
        'readout_space': readout_space,
        'owner_entry_id': owner_entry_id,
        'status': status, 'unavailable_reason': unavailable_reason,
        'readout': None, 'readout_sha256': None, 'readout_norm': None,
    }
    if context_source_ids is not None:
        row['context_source_ids'] = list(context_source_ids)
    if status == 'ok':
        readout = torch.as_tensor(readout, dtype=torch.float32)
        _require(readout.ndim == 2, 'Readout must be [heads, width]')
        _require(bool(torch.isfinite(readout).all()), 'Readout is not finite')
        row.update({
            'readout': readout.tolist(),
            'readout_sha256': _sha_json(readout.tolist()),
            'readout_norm': float(torch.linalg.vector_norm(readout)),
        })
    return row


def _outside_fact(episodes, episode):
    """Return an observed fact whose key and value are absent from this episode.

    Counterfactual variants share their local addresses, so a sibling variant is
    never a valid no-match control.  The helper rejects rather than silently
    falling back when the split has no independent, non-colliding fact.
    """
    local_keys = {(fact['entity'], fact['attribute'])
                  for fact in episode['facts'] + episode['distractors']}
    local_keys.update((correction['entity'], correction['attribute'])
                      for correction in episode['corrections'])
    local_values = {fact['value']
                    for fact in episode['facts'] + episode['distractors']}
    local_values.update(correction['value']
                        for correction in episode['corrections'])
    for candidate in episodes:
        if candidate['split'] != episode['split']:
            continue
        if candidate['world_id'] == episode['world_id']:
            continue
        if candidate['family_id'] == episode['family_id']:
            continue
        for fact in candidate['facts']:
            key = (fact['entity'], fact['attribute'])
            if key in local_keys or fact['value'] in local_values:
                continue
            return fact
    raise ValueError(
        'No independent foreign fact with a non-matching address/value exists '
        'for the cross-world control')


def _evaluate_readouts(model, tokenizer, episodes, episode, state, memory,
                       codebook, *, layer):
    device = memory.read().weights.device
    _, live, negative = _record_views(episode, state)
    live_keys = {(record['entity'], record['attribute']) for record in live}
    fact_queries = [query for query in episode['queries']
                    if (query['entity'], query['attribute']) in live_keys]
    rows = []
    query_keys = {}

    def query_vector(query):
        if query['query_id'] not in query_keys:
            query_keys[query['query_id']] = query_key(
                model, tokenizer, query['text'], layer=layer)
        return query_keys[query['query_id']]

    entry_ids = sorted(codebook['entries'])
    first_entry = codebook['entries'][entry_ids[0]]
    head_shape = _key_tensor(first_entry).shape[2:]

    def read_with(vector):
        return _read_heads(memory, vector).detach().float().cpu()

    # Stage 1: exact observed write keys (storage upper bound, not addressing).
    for query in fact_queries:
        live_by_key = {(record['entity'], record['attribute']): record
                       for record in live}
        target = live_by_key[(query['entity'], query['attribute'])]
        entry = codebook['entries'][target['record_id']]
        rows.append(_readout_row(
            stage=1, arm='exact_key', episode=episode, state=state,
            case_id=query['query_id'], query_id=query['query_id'],
            query_text_sha256=_sha_json(query['text']), codebook=codebook,
            readout=read_with(_key_tensor(entry)), readout_space='value'))
    rows.append(_readout_row(
        stage=1, arm='control', episode=episode, state=state,
        case_id='control:zero', query_id=None,
        query_text_sha256=_sha_json('zero-query'), codebook=codebook,
        readout=read_with(torch.zeros(1, 1, *head_shape, device=device)),
        readout_space='value'))
    if len(entry_ids) >= 2:
        owner_id = entry_ids[1]
        rows.append(_readout_row(
            stage=1, arm='control', episode=episode, state=state,
            case_id='control:wrong_key', query_id=None,
            query_text_sha256=_sha_json('wrong-key:' + owner_id),
            codebook=codebook,
            readout=read_with(_key_tensor(codebook['entries'][owner_id])),
            readout_space='value', owner_entry_id=owner_id))
    else:
        rows.append(_readout_row(
            stage=1, arm='control', episode=episode, state=state,
            case_id='control:wrong_key', query_id=None,
            query_text_sha256=_sha_json('wrong-key-unavailable'),
            codebook=codebook, readout=None, readout_space='value',
            status='unavailable',
            unavailable_reason='at least two live entries are required'))
    outside = _outside_fact(episodes, episode)
    binding = observed_binding(model, tokenizer, outside, layer=layer)
    rows.append(_readout_row(
        stage=1, arm='control', episode=episode, state=state,
        case_id='control:cross_world', query_id=None,
        query_text_sha256=_sha_json('cross-world:' + outside['record_id']),
        codebook=codebook, readout=read_with(binding.keys),
        readout_space='value', context_source_ids=[outside['record_id']]))

    # Stage 2: rendered query-derived key ranked against stored write keys.
    for query in episode['queries']:
        vector = query_vector(query)
        rows.append(_readout_row(
            stage=2, arm='query_key', episode=episode, state=state,
            case_id=query['query_id'], query_id=query['query_id'],
            query_text_sha256=_sha_json(query['text']), codebook=codebook,
            readout=vector[0, 0].detach().float().cpu(), readout_space='key'))
    rows.append(_readout_row(
        stage=2, arm='control', episode=episode, state=state,
        case_id='control:zero', query_id=None,
        query_text_sha256=_sha_json('zero-query'), codebook=codebook,
        readout=torch.zeros(*head_shape).float(), readout_space='key'))
    rows.append(_readout_row(
        stage=2, arm='control', episode=episode, state=state,
        case_id='control:cross_world', query_id=None,
        query_text_sha256=_sha_json('cross-world:' + outside['record_id']),
        codebook=codebook,
        readout=binding.keys[0, 0].detach().float().cpu(),
        readout_space='key', context_source_ids=[outside['record_id']]))

    # Stage 3: candidate/value discrimination with the stage-2 key.
    for query in episode['queries']:
        vector = query_vector(query)
        rows.append(_readout_row(
            stage=3, arm='query_value', episode=episode, state=state,
            case_id=query['query_id'], query_id=query['query_id'],
            query_text_sha256=_sha_json(query['text']), codebook=codebook,
            readout=read_with(vector), readout_space='value'))
    rows.append(_readout_row(
        stage=3, arm='control', episode=episode, state=state,
        case_id='control:zero', query_id=None,
        query_text_sha256=_sha_json('zero-query'), codebook=codebook,
        readout=read_with(torch.zeros(1, 1, *head_shape, device=device)),
        readout_space='value'))
    rows.append(_readout_row(
        stage=3, arm='control', episode=episode, state=state,
        case_id='control:cross_world', query_id=None,
        query_text_sha256=_sha_json('cross-world:' + outside['record_id']),
        codebook=codebook, readout=read_with(binding.keys),
        readout_space='value', context_source_ids=[outside['record_id']]))
    if negative:
        labels = derive_state_labels(episode, state)
        _require(len(labels['corrected_query_ids']) == 1,
                 'A superseded probe requires one correction target')
        correction_query = next(
            query for query in episode['queries']
            if query['query_id'] == labels['corrected_query_ids'][0])
        vector = query_vector(correction_query)
        rows.append(_readout_row(
            stage=3, arm='control', episode=episode, state=state,
            case_id='control:superseded_probe',
            query_id=correction_query['query_id'],
            query_text_sha256=_sha_json(correction_query['text']),
            codebook=codebook, readout=read_with(vector),
            readout_space='value'))
    rows.sort(key=lambda row: (row['stage'], row['case_id'], row['arm']))
    return rows


def _mechanical_reload(memory, model, tokenizer, episode, state, images_dir,
                       codebook):
    image = images_dir / f"{episode['episode_id']}-{state['name']}"
    metadata = memory.save(image)
    loaded = PersistentFastMemory.load(
        image, model, tokenizer, memory.scope,
        representation=FACT_REPRESENTATION)
    difference = float((memory.read().weights - loaded.read().weights)
                       .abs().max())
    return {
        'episode_id': episode['episode_id'], 'state': state['name'],
        'codebook_key': f"{episode['episode_id']}/{state['name']}",
        'image': _display_path(image),
        'state_digest_before': memory.read().state_digest,
        'state_digest_after': loaded.read().state_digest,
        'lineage_equal': list(memory.lineage) == list(loaded.lineage),
        'max_abs_difference': difference,
        'equal': (difference == 0.0
                  and memory.read().state_digest == loaded.read().state_digest
                  and list(memory.lineage) == list(loaded.lineage)),
        'codebook_observation_sha256': codebook['observation_sha256'],
        'metadata_cursor': metadata.get('cursor'),
        'metadata_chunk_index': metadata.get('chunk_index'),
    }


# ---------------------------------------------------------------------------
# Stage 4: frozen native-language answer generation
# ---------------------------------------------------------------------------

def _default_answerer_factory(model, tokenizer, protocol):
    from .architecture.contracts import TaskGraph
    from .memory_system.session import MemorySession

    max_new_tokens = int(protocol['generation']['max_new_tokens'])
    layer = protocol.get('layer')
    _require(type(layer) is int,
             'Default answerer requires the declared injection layer')

    def factory(ledger):
        session = MemorySession(model, tokenizer, ledger)

        def answer(*, prompt_ids, prompt_mask, query, scope, fast_state):
            selection = session.index.retrieve(query, **asdict(scope), limit=0)
            # The fast value read is numerically inert unless its declared
            # injection port is enabled.  Enable it, and the fast branch, for
            # the memory arm only; exact/full-context/removed controls keep no
            # fast state, no fast port and only the empty memory branch.
            ports = (layer,) if fast_state is not None else ()
            request = session.prepare(
                prompt_ids=prompt_ids, prompt_mask=prompt_mask, query=query,
                scope=scope,
                task=TaskGraph(f"{scope.branch_id}:matched", query),
                core_impl='none', selection=selection,
                fast_state=fast_state, enabled_ports=ports,
                enabled_branches=(('fast',) if fast_state is not None
                                  else ('memory',)))
            try:
                generated = session.generate(request, max_new_tokens=max_new_tokens)
                ids = generated[0, prompt_ids.shape[1]:].tolist()
                text = tokenizer.decode(ids, skip_special_tokens=True).strip()
                return text, ids
            finally:
                request.release()

        return answer

    return factory


def _ledger_for_state(episode, state, ledger_dir):
    path = ledger_dir / f"{episode['episode_id']}-{state['name']}.sqlite"
    if path.exists():
        raise FileExistsError(f'Matched ledger already exists: {path}')
    ledger = RecordLedger(path)
    records, _, _ = _record_views(episode, state)
    for record in records:
        text = fact_source_text(
            _fact_of(record), correction=record['kind'] == 'source_correction')
        item = MemoryRecord(
            record_id=record['record_id'], subject_id=SUBJECT_ID,
            branch_id=episode['world_id'], text=text,
            observed_at=record['observed_at'], event_at=record['event_at'],
            purpose=PURPOSE, kind=record['kind'],
            supersedes=record['supersedes'],
            source_id=f"{episode['world_id']}:source:{record['event_at']}",
            authenticity='verified_source')
        _require(ledger.admit(item, authorized_subject=SUBJECT_ID,
                              authorized_branch=episode['world_id']),
                 'Ledger admission must accept every observed record')
    return ledger


def _context_text(arm, episode, state, reference, scope, query):
    if arm in ('memory', 'context_removed'):
        return '', []
    if arm == 'full_context':
        lines = [line for page in state['observations'] for line in page['lines']]
        return '\n'.join(lines), [record['record_id']
                                  for record in _memory_history(episode, state)]
    context = reference.select(query['text'], scope)
    if context.enabled:
        return context.text, list(context.record_ids)
    return '', []


def _evaluate_answers(episode, state, memory, *, reference, scope,
                      answerer, tokenizer, protocol, device):
    from .architecture.loading import encode_prompt
    max_new_tokens = int(protocol['generation']['max_new_tokens'])
    max_prompt = int(protocol['generation']['max_prompt_tokens'])
    observation_sha = state_observation_sha256(episode, state)
    rows = []
    for query in episode['queries']:
        for arm in ANSWER_ARMS:
            observation, source_ids = _context_text(
                arm, episode, state, reference, scope, query)
            prompt_text = MATCHED_TEMPLATE.replace(
                '{query}', query['text']).replace('{observation}', observation)
            prompt_ids = encode_prompt(tokenizer, prompt_text, device)
            _require(prompt_ids.shape[1] <= max_prompt,
                     'Matched prompt exceeds the protocol bound')
            fast_state = memory.read() if arm == 'memory' else None
            answer, token_ids = answerer(
                prompt_ids=prompt_ids,
                prompt_mask=torch.ones_like(prompt_ids, dtype=torch.bool),
                query=query['text'], scope=scope, fast_state=fast_state)
            _require(isinstance(answer, str), 'Answerer must return text')
            token_ids = [int(token) for token in token_ids]
            _require(len(token_ids) <= max_new_tokens,
                     'Answerer exceeded the declared generation budget')
            rows.append({
                'format': ANSWER_FORMAT, 'stage': 4,
                'episode_id': episode['episode_id'],
                'world_id': episode['world_id'],
                'state': state['name'], 'cutoff': state['cutoff'],
                'query_id': query['query_id'], 'query_form': query['form'],
                'query_text_sha256': _sha_json(query['text']),
                'observation_sha256': observation_sha,
                'arm': arm, 'context_sha256': _sha_json(observation),
                'context_source_ids': source_ids,
                'prompt_sha256': _sha_json(prompt_ids[0].tolist()),
                'prompt_tokens': int(prompt_ids.shape[1]),
                'max_new_tokens': max_new_tokens,
                'token_ids': token_ids, 'generated_tokens': len(token_ids),
                'answer': answer, 'answer_sha256': _sha_text(answer),
                'status': 'ok',
            })
    rows.sort(key=lambda row: (row['query_id'], row['arm']))
    return rows


def _load_sealed_tokenizer(model_path):
    """Load only the sealed tokenizer; no model weights or native forward."""
    from transformers import AutoTokenizer
    path = _resolve_path(model_path)
    _require(path.is_dir(), 'Sealed tokenizer source directory is missing')
    tokenizer = AutoTokenizer.from_pretrained(
        str(path), local_files_only=True, trust_remote_code=False)
    _require(getattr(tokenizer, 'is_fast', False),
             'Sealed source requires its native fast tokenizer')
    return tokenizer


def audit_prompt_bounds(episodes, tokenizer, *, configured_bound):
    """Render every arm prompt with the sealed tokenizer before any native run.

    Full-context observations are complete observed pages; the audit never
    truncates a page or a rendered prompt.  It records each arm's maximum, the
    exact global maximum row, the first row that would exceed the previous
    declared bound, and the bound that covers every row.
    """
    from .architecture.loading import encode_prompt
    from .runtime.memory_representation import tokenizer_identity

    _require(type(configured_bound) is int and configured_bound > 0,
             'Configured prompt bound must be a positive integer')
    _require(isinstance(episodes, list) and episodes,
             'Prompt audit requires episodes')
    per_arm = {arm: {'rows': 0, 'max_tokens': -1, 'max_row': None}
               for arm in ANSWER_ARMS}
    rows = []
    max_row = None
    first_over = None
    rows_over = 0
    with tempfile.TemporaryDirectory(prefix='matched-prompt-audit-') as temporary:
        ledger_root = Path(temporary)
        for episode in episodes:
            for state in episode_states(episode):
                full_lines = [line for page in state['observations']
                              for line in page['lines']]
                full_context = '\n'.join(full_lines)
                ledger = _ledger_for_state(episode, state, ledger_root)
                reference = ExactMemoryReference(ledger)
                try:
                    scope = _state_scope(episode, state)
                    for query in episode['queries']:
                        for arm in ANSWER_ARMS:
                            observation, source_ids = _context_text(
                                arm, episode, state, reference, scope, query)
                            if arm == 'full_context':
                                _require(
                                    observation == full_context,
                                    'Full-context prompt dropped observed page lines')
                                _require(
                                    len(full_lines) == sum(
                                        len(page['lines'])
                                        for page in state['observations']),
                                    'Full-context page inventory differs')
                            prompt = MATCHED_TEMPLATE.replace(
                                '{query}', query['text']).replace(
                                    '{observation}', observation)
                            ids = encode_prompt(tokenizer, prompt, 'cpu')
                            tokens = int(ids.shape[1])
                            row = {
                                'episode_id': episode['episode_id'],
                                'state': state['name'], 'split': episode['split'],
                                'complexity': episode['complexity'],
                                'query_id': query['query_id'],
                                'query_form': query['form'], 'arm': arm,
                                'tokens': tokens,
                                'prompt_sha256': _sha_text(prompt),
                                'observation_sha256':
                                    state_observation_sha256(episode, state),
                                'source_record_ids': list(source_ids),
                            }
                            rows.append(row)
                            arm_metric = per_arm[arm]
                            arm_metric['rows'] += 1
                            if tokens > arm_metric['max_tokens']:
                                arm_metric['max_tokens'] = tokens
                                arm_metric['max_row'] = row
                            if max_row is None or tokens > max_row['tokens']:
                                max_row = row
                            if tokens > configured_bound:
                                rows_over += 1
                                if first_over is None:
                                    first_over = row
                finally:
                    reference.close()
                    ledger.close()
    _require(rows and max_row is not None, 'Prompt audit produced no rows')
    observed = int(max_row['tokens'])
    bound = max(configured_bound, observed)
    model_max = int(getattr(tokenizer, 'model_max_length', 0) or 0)
    recorded_model_max = model_max if 0 < model_max < 10 ** 9 else None
    if recorded_model_max is not None:
        _require(bound <= recorded_model_max,
                 'Measured prompt bound exceeds the sealed tokenizer/model '
                 'position budget; narrow the observed pages explicitly')
    audit = {
        'format': PROMPT_AUDIT_FORMAT,
        'configured_bound': configured_bound,
        'bound': bound,
        'bound_justification': (
            'maximum rendered prompt token count across every '
            'episode/state/query/arm row; complete observed pages, no truncation'),
        'observed_max_tokens': observed,
        'max_row': max_row,
        'per_arm': per_arm,
        'first_row_over_previous_bound': first_over,
        'rows_over_previous_bound': rows_over,
        'previous_bound_would_fail': rows_over > 0,
        'total_rows': len(rows),
        'complete_pages': True,
        'truncation': False,
        'tokenizer_identity': tokenizer_identity(tokenizer),
        'tokenizer_model_max_length': recorded_model_max,
        'rows': rows,
    }
    _require(all(row['tokens'] <= bound for row in rows),
             'Audited prompt does not fit the recommended bound')
    _require(audit['observed_max_tokens'] <= audit['bound'],
             'Observed maximum exceeds the audited bound')
    return audit


def audit_generation_bounds(episodes, labels_by_episode, tokenizer, *,
                            configured_bound):
    """Measure every expected-answer token length with the sealed tokenizer.

    The bound covers the answer tokens plus the tokenizer EOS token, so the
    native generator is neither truncated mid-answer nor denied its stop token.
    Per-kind and per-role maxima, the exact maximum row and the first row over
    the previous bound are recorded for review before any native work.
    """
    from .runtime.memory_representation import tokenizer_identity

    _require(type(configured_bound) is int and configured_bound > 0,
             'Configured generation bound must be a positive integer')
    _require(isinstance(labels_by_episode, dict) and labels_by_episode,
             'Generation audit requires verifier labels')
    eos_token_id = tokenizer.eos_token_id
    eos_required = eos_token_id is not None
    per_kind = {}
    per_role = {}
    rows = []
    max_row = None
    first_over = None
    first_over_answer_only = None
    rows_over = 0
    rows_over_answer_only = 0

    def group(store, name):
        return store.setdefault(name, {
            'n': 0, 'max_required_tokens': -1,
            'max_required_with_eos': -1, 'max_row': None,
            'over_previous_bound_answer_only': 0,
            'over_previous_bound_with_eos': 0,
        })

    for episode in episodes:
        states = labels_by_episode.get(episode['episode_id'])
        _require(isinstance(states, dict), 'Generation audit lacks episode labels')
        for state in episode_states(episode):
            labels = states.get(state['name'])
            _require(isinstance(labels, dict), 'Generation audit lacks state labels')
            for query in episode['queries']:
                target = labels['targets'].get(query['query_id'])
                _require(isinstance(target, dict),
                         'Generation audit lacks query labels')
                expected = target['expected']
                _require(isinstance(expected, str) and expected,
                         'Expected answer must be nonempty text')
                token_ids = tokenizer(
                    expected, add_special_tokens=False)['input_ids']
                required = len(token_ids)
                required_with_eos = required + (1 if eos_required else 0)
                row = {
                    'episode_id': episode['episode_id'],
                    'state': state['name'], 'split': episode['split'],
                    'complexity': episode['complexity'],
                    'query_id': query['query_id'], 'kind': target['kind'],
                    'target_role': target['target_role'],
                    'expected': expected,
                    'expected_sha256': _sha_text(expected),
                    'required_tokens': required,
                    'required_with_eos': required_with_eos,
                }
                rows.append(row)
                for name, store in (('kind', per_kind),
                                    ('role', per_role)):
                    key = target['kind'] if name == 'kind' else target['target_role']
                    metric = group(store, key)
                    metric['n'] += 1
                    if required > metric['max_required_tokens']:
                        metric['max_required_tokens'] = required
                    if required_with_eos > metric['max_required_with_eos']:
                        metric['max_required_with_eos'] = required_with_eos
                        metric['max_row'] = row
                    if required > configured_bound:
                        metric['over_previous_bound_answer_only'] += 1
                        rows_over_answer_only += 1
                        if first_over_answer_only is None:
                            first_over_answer_only = row
                    if required_with_eos > configured_bound:
                        metric['over_previous_bound_with_eos'] += 1
                        rows_over += 1
                        if first_over is None:
                            first_over = row
                if max_row is None or required_with_eos > max_row['required_with_eos']:
                    max_row = row

    _require(rows and max_row is not None,
             'Generation audit produced no expected answers')
    observed_tokens = int(max_row['required_tokens'])
    observed_with_eos = int(max_row['required_with_eos'])
    bound = observed_with_eos
    _require(bound >= 1, 'Generation audit bound must be positive')
    model_max = int(getattr(tokenizer, 'model_max_length', 0) or 0)
    recorded_model_max = model_max if 0 < model_max < 10 ** 9 else None
    if recorded_model_max is not None:
        _require(bound <= recorded_model_max,
                 'Required generation bound exceeds the sealed tokenizer/model '
                 'position budget; narrow observed answers explicitly')
    audit = {
        'format': GENERATION_AUDIT_FORMAT,
        'evaluation_only': True,
        'note': ('Expected answers are tokenized only to bound native generation; '
                 'this file is never consumed by a writer, sense input or prompt.'),
        'configured_bound': configured_bound,
        'bound': bound,
        'bound_justification': (
            'maximum expected-answer tokens plus the tokenizer EOS token across '
            'every episode/state/query target; answer truncation and EOS '
            'suppression are both excluded'),
        'eos_token_id': eos_token_id,
        'eos_required': eos_required,
        'answer_truncation': False,
        'eos_suppression': False,
        'observed_max_required_tokens': observed_tokens,
        'observed_max_required_with_eos': observed_with_eos,
        'max_row': max_row,
        'per_kind': per_kind,
        'per_role': per_role,
        'first_row_over_previous_bound': first_over,
        'first_row_over_previous_bound_answer_only': first_over_answer_only,
        'rows_over_previous_bound': rows_over,
        'rows_over_previous_bound_answer_only': rows_over_answer_only,
        'previous_bound_would_fail': rows_over > 0,
        'previous_bound_answer_only_would_fail': rows_over_answer_only > 0,
        'total_targets': len(rows),
        'tokenizer_identity': tokenizer_identity(tokenizer),
        'tokenizer_model_max_length': recorded_model_max,
        'rows': rows,
    }
    _require(all(row['required_with_eos'] <= bound for row in rows),
             'An expected answer does not fit the audited generation bound')
    _require(audit['observed_max_required_with_eos'] == audit['bound'],
             'Audited generation bound is not the exact covering maximum')
    return audit


# ---------------------------------------------------------------------------
# Independent scorer
# ---------------------------------------------------------------------------

def _entry_meta(labels):
    meta = {}
    for entry in labels['live_entries']:
        meta[entry['record_id']] = {'role': entry['role'],
                                    'source_family': entry['source_family']}
    for entry in labels['negative_entries']:
        meta[entry['record_id']] = {'role': 'superseded_negative',
                                    'source_family': entry['source_family']}
    return meta


def _recompute_row(row, codebook, labels, thresholds):
    stage = row['stage']
    _require(row.get('state') == labels.get('state'),
             'Readout row state differs from its verifier labels')
    if row.get('status') != 'ok':
        _require(isinstance(row.get('unavailable_reason'), str)
                 and row['unavailable_reason'],
                 'An unavailable row must state its reason')
        return None
    entries = codebook['entries']
    negative_ids = set(codebook['negative_entry_ids'])
    _require(codebook['episode_id'] == row['episode_id']
             and codebook['world_id'] == row['world_id']
             and codebook['state'] == row['state'],
             'Readout row references a foreign or mismatched codebook')
    case = row['case_id']
    if case.startswith('control:'):
        control = case.split(':', 1)[1]
        if control == 'wrong_key':
            target = row.get('owner_entry_id')
            _require(target in entries, 'Wrong-key owner is not in the codebook')
            role, source_family, kind = 'control_owner', 'control', 'control'
        elif control == 'superseded_probe':
            query_id = row.get('query_id')
            _require(query_id in labels['targets'],
                     'Superseded probe lacks its derived query')
            target_info = labels['targets'][query_id]
            target = target_info['target_entry_id']
            role = target_info['target_role']
            source_family = target_info['target_source_family']
            kind = target_info['kind']
        elif control in ('zero', 'cross_world'):
            target, role, source_family, kind = None, 'control', 'control', 'control'
        else:
            raise ValueError(f'Unknown matched control: {control}')
    else:
        _require(case in labels['targets'],
                 'Readout case is not a derived query')
        target_info = labels['targets'][case]
        target = target_info['target_entry_id']
        role = target_info['target_role']
        source_family = target_info['target_source_family']
        kind = target_info['kind']

    field = 'write_key' if stage == 2 else 'value_vector'
    if stage == 2:
        floor, margin_floor = (thresholds['addressing_cosine_floor'],
                               thresholds['addressing_margin_floor'])
    else:
        floor, margin_floor = (thresholds['match_cosine_floor'],
                               thresholds['match_margin_floor'])
    readout = row.get('readout')
    _require(isinstance(readout, list) and readout, 'Readout must be a vector')
    readout_tensor = torch.tensor(readout, dtype=torch.float32)
    _require(bool(torch.isfinite(readout_tensor).all()), 'Readout is not finite')
    _require(row.get('readout_space') == ('key' if stage == 2 else 'value'),
             'Readout space differs from its stage')
    # Addressing ranks distinct live addresses.  A superseded negative shares
    # its address with the live correction, so it is a value-level candidate
    # only (stages 1 and 3), never a distinct addressing candidate.
    if stage == 2:
        candidate_ids = sorted(codebook['live_entry_ids'])
        _require(all(entry_id in entries for entry_id in candidate_ids),
                 'Live addressing entry is absent from the codebook')
    else:
        candidate_ids = sorted(entries)
    scores = {}
    for entry_id in candidate_ids:
        vector = torch.tensor(entries[entry_id][field], dtype=torch.float32)
        _require(tuple(vector.shape) == tuple(readout_tensor.shape),
                 'Readout/vector shapes differ')
        scores[entry_id] = _cosine_heads(readout_tensor, vector)
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    top_entry_id, top_cosine = ordered[0]
    second_cosine = ordered[1][1] if len(ordered) > 1 else None
    non_target = [score for entry_id, score in ordered
                  if entry_id != target]
    strongest_id = next((entry_id for entry_id, _ in ordered
                         if entry_id != target), None)
    strongest_cosine = max(non_target) if non_target else 0.0
    margin = (scores[target] - strongest_cosine
              if target is not None else None)
    target_rank = (1 + sum(1 for _, score in ordered if score > scores[target])
                   if target is not None else None)
    accepted = bool(
        top_cosine >= floor
        and ((top_cosine - (second_cosine if second_cosine is not None
                            else 0.0)) >= margin_floor))
    top1 = bool(target is not None and top_entry_id == target)
    false_recall = bool(accepted and not top1)
    meta = _entry_meta(labels)
    top_role = meta.get(top_entry_id, {}).get('role', 'unknown')
    false_recall_kind = None
    if accepted and not top1:
        if target is None:
            false_recall_kind = 'unknown_address' if kind == 'unknown' else 'control'
        elif top_entry_id in negative_ids:
            false_recall_kind = 'superseded'
        elif top_role == 'distractor':
            false_recall_kind = 'independent_world'
        else:
            false_recall_kind = 'other_source'
    negative_scores = [scores[entry_id] for entry_id in negative_ids
                       if entry_id in scores]
    superseded_margin = (scores[target] - max(negative_scores)
                         if target is not None and negative_scores else None)
    result = {
        'stage': stage, 'arm': row['arm'], 'case_id': case,
        'query_id': row.get('query_id'), 'kind': kind,
        'episode_id': row['episode_id'], 'world_id': row['world_id'],
        'state': row['state'], 'cutoff': row['cutoff'],
        'split': labels['split'], 'complexity': labels['complexity'],
        'target_entry_id': target, 'target_role': role,
        'target_source_family': source_family,
        'candidate_count': len(candidate_ids),
        'scores': scores, 'ranking': [[entry_id, score] for entry_id, score in ordered],
        'target_rank': target_rank, 'top_entry_id': top_entry_id,
        'top_cosine': top_cosine, 'second_cosine': second_cosine,
        'strongest_distractor_id': strongest_id,
        'strongest_distractor_cosine': strongest_cosine,
        'margin_over_strongest_distractor': margin,
        'accepted': accepted, 'top1': top1,
        'false_recall': false_recall,
        'false_recall_kind': false_recall_kind,
        'superseded_margin': superseded_margin,
        'superseded_selected': bool(accepted and top_entry_id in negative_ids),
        'cross_world_selected': bool(
            accepted and meta.get(top_entry_id, {}).get('source_family')
            == 'independent_world'),
        'readout_norm': float(torch.linalg.vector_norm(readout_tensor)),
        'readout_sha256': _sha_json(readout),
        'observation_sha256': row['observation_sha256'],
    }
    if result['readout_sha256'] != row.get('readout_sha256'):
        _require(False, 'Readout bytes differ from the stored measurement')
    return result


def _metric_summary(rows, *, stages, arm, control_prefix='control:'):
    selected = [row for row in rows if row['stage'] == stages
                and (row['arm'] == arm or row['arm'] == 'control')]
    queries = [row for row in selected if not row['case_id'].startswith(control_prefix)]
    facts = [row for row in queries if row['kind'] == 'fact']
    unknowns = [row for row in queries if row['kind'] == 'unknown']
    controls = {}
    for row in selected:
        if row['case_id'].startswith(control_prefix):
            name = row['case_id'].split(':', 1)[1]
            controls.setdefault(name, []).append(row)
    correction = [row for row in facts if row['target_role'] == 'correction_target']
    unaffected = [row for row in facts if row['target_role'] == 'retained']

    def block(items):
        ranks = [row['target_rank'] for row in items
                 if row['target_rank'] is not None]
        margins = [row['margin_over_strongest_distractor'] for row in items
                   if row['margin_over_strongest_distractor'] is not None]
        superseded = [row['superseded_margin'] for row in items
                      if row['superseded_margin'] is not None]
        return {
            'n': len(items),
            'top1_rate': _rate(sum(row['top1'] for row in items), len(items)),
            'mean_rank': _mean(ranks), 'median_rank': _median(ranks),
            'mrr': _mean([1.0 / rank for rank in ranks]),
            'mean_margin': _mean(margins), 'min_margin': min(margins) if margins else None,
            'mean_superseded_margin': _mean(superseded),
            'mean_candidate_count': _mean([row['candidate_count'] for row in items]),
        }

    def control_rate(name, field):
        rows = controls.get(name, [])
        return (_rate(sum(int(bool(row[field])) for row in rows), len(rows))
                if rows else None)

    false_kinds = {}
    for row in queries:
        if row['false_recall_kind']:
            false_kinds[row['false_recall_kind']] = (
                false_kinds.get(row['false_recall_kind'], 0) + 1)
    summary = {
        'rows': len(selected), 'query_rows': len(queries),
        'fact': block(facts), 'unknown': {
            'n': len(unknowns),
            'false_recall_rate': _rate(sum(row['false_recall'] for row in unknowns),
                                       len(unknowns)),
            'mean_candidate_count': _mean([row['candidate_count']
                                           for row in unknowns]),
        },
        'correction': block(correction),
        'unaffected': block(unaffected),
        'false_recall_rate': _rate(sum(row['false_recall'] for row in queries),
                                   len(queries)),
        'false_recall_by_kind': false_kinds,
        'superseded_selection_rate': _rate(
            sum(row['superseded_selected'] for row in queries), len(queries)),
        'cross_world_selection_rate': _rate(
            sum(row['cross_world_selected'] for row in queries), len(queries)),
        'zero_false_recall_rate': control_rate('zero', 'false_recall'),
        'cross_world_false_recall_rate': control_rate('cross_world',
                                                       'false_recall'),
        'wrong_key_owner_rate': control_rate('wrong_key', 'top1'),
        'superseded_probe_top1_rate': control_rate('superseded_probe',
                                                    'top1'),
        'superseded_probe_selected_rate': control_rate('superseded_probe',
                                                        'superseded_selected'),
        'control_counts': {name: len(rows) for name, rows in controls.items()},
    }
    return summary


def _gate(name, metrics, passed, thresholds):
    return {'status': 'pass' if passed else 'fail',
            'metrics': metrics, 'thresholds': thresholds}


def _stage1_gates(metrics, thresholds):
    fact = metrics['fact']
    return {
        'storage_top1': _gate(
            'storage_top1', fact['top1_rate'],
            fact['n'] > 0 and fact['top1_rate'] >= thresholds['storage_top1_floor'],
            {'storage_top1_floor': thresholds['storage_top1_floor']}),
        'storage_mean_rank': _gate(
            'storage_mean_rank', fact['mean_rank'],
            fact['mean_rank'] is not None
            and fact['mean_rank'] <= thresholds['storage_mean_rank_max'],
            {'storage_mean_rank_max': thresholds['storage_mean_rank_max']}),
        'storage_margin': _gate(
            'storage_margin', fact['mean_margin'],
            fact['mean_margin'] is not None
            and fact['mean_margin'] >= thresholds['storage_margin_floor'],
            {'storage_margin_floor': thresholds['storage_margin_floor']}),
        'storage_false_recall': _gate(
            'storage_false_recall', metrics['false_recall_rate'],
            metrics['false_recall_rate'] is not None
            and metrics['false_recall_rate'] <= thresholds['storage_false_recall_max'],
            {'storage_false_recall_max': thresholds['storage_false_recall_max']}),
        'storage_cross_world': _gate(
            'storage_cross_world', metrics['cross_world_false_recall_rate'],
            metrics['cross_world_false_recall_rate'] is not None
            and metrics['cross_world_false_recall_rate']
            <= thresholds['storage_cross_world_false_recall_max'],
            {'storage_cross_world_false_recall_max':
             thresholds['storage_cross_world_false_recall_max']}),
        'storage_zero': _gate(
            'storage_zero', metrics['zero_false_recall_rate'],
            metrics['zero_false_recall_rate'] is not None
            and metrics['zero_false_recall_rate']
            <= thresholds['zero_false_recall_max'],
            {'zero_false_recall_max': thresholds['zero_false_recall_max']}),
    }


def _stage2_gates(metrics, thresholds):
    fact = metrics['fact']
    unknown = metrics['unknown']
    return {
        'addressing_top1': _gate(
            'addressing_top1', fact['top1_rate'],
            fact['n'] > 0 and fact['top1_rate'] >= thresholds['addressing_top1_floor'],
            {'addressing_top1_floor': thresholds['addressing_top1_floor']}),
        'addressing_mean_rank': _gate(
            'addressing_mean_rank', fact['mean_rank'],
            fact['mean_rank'] is not None
            and fact['mean_rank'] <= thresholds['addressing_mean_rank_max'],
            {'addressing_mean_rank_max': thresholds['addressing_mean_rank_max']}),
        'addressing_margin': _gate(
            'addressing_margin', fact['mean_margin'],
            fact['mean_margin'] is not None
            and fact['mean_margin'] >= thresholds['addressing_margin_mean_floor'],
            {'addressing_margin_mean_floor':
             thresholds['addressing_margin_mean_floor']}),
        'addressing_false_recall': _gate(
            'addressing_false_recall', metrics['false_recall_rate'],
            metrics['false_recall_rate'] is not None
            and metrics['false_recall_rate'] <= thresholds['addressing_false_recall_max'],
            {'addressing_false_recall_max':
             thresholds['addressing_false_recall_max']}),
        'addressing_unknown': _gate(
            'addressing_unknown', unknown['false_recall_rate'],
            unknown['n'] > 0
            and unknown['false_recall_rate']
            <= thresholds['addressing_unknown_false_recall_max'],
            {'addressing_unknown_false_recall_max':
             thresholds['addressing_unknown_false_recall_max']}),
        'addressing_zero': _gate(
            'addressing_zero', metrics['zero_false_recall_rate'],
            metrics['zero_false_recall_rate'] is not None
            and metrics['zero_false_recall_rate']
            <= thresholds['zero_false_recall_max'],
            {'zero_false_recall_max': thresholds['zero_false_recall_max']}),
        'addressing_cross_world': _gate(
            'addressing_cross_world', metrics['cross_world_false_recall_rate'],
            metrics['cross_world_false_recall_rate'] is not None
            and metrics['cross_world_false_recall_rate']
            <= thresholds['addressing_false_recall_max'],
            {'addressing_false_recall_max':
             thresholds['addressing_false_recall_max']}),
    }


def _stage3_gates(metrics, thresholds):
    fact = metrics['fact']
    correction = metrics['correction']
    unaffected = metrics['unaffected']
    unknown = metrics['unknown']
    return {
        'candidate_top1': _gate(
            'candidate_top1', fact['top1_rate'],
            fact['n'] > 0 and fact['top1_rate'] >= thresholds['candidate_top1_floor'],
            {'candidate_top1_floor': thresholds['candidate_top1_floor']}),
        'candidate_mean_rank': _gate(
            'candidate_mean_rank', fact['mean_rank'],
            fact['mean_rank'] is not None
            and fact['mean_rank'] <= thresholds['candidate_mean_rank_max'],
            {'candidate_mean_rank_max': thresholds['candidate_mean_rank_max']}),
        'candidate_margin': _gate(
            'candidate_margin', fact['mean_margin'],
            fact['mean_margin'] is not None
            and fact['mean_margin'] >= thresholds['candidate_margin_floor'],
            {'candidate_margin_floor': thresholds['candidate_margin_floor']}),
        'candidate_false_recall': _gate(
            'candidate_false_recall', metrics['false_recall_rate'],
            metrics['false_recall_rate'] is not None
            and metrics['false_recall_rate'] <= thresholds['candidate_false_recall_max'],
            {'candidate_false_recall_max':
             thresholds['candidate_false_recall_max']}),
        'correction_top1': _gate(
            'correction_top1', correction['top1_rate'],
            correction['n'] > 0
            and correction['top1_rate'] >= thresholds['correction_top1_floor'],
            {'correction_top1_floor': thresholds['correction_top1_floor']}),
        'unaffected_top1': _gate(
            'unaffected_top1', unaffected['top1_rate'],
            unaffected['n'] > 0
            and unaffected['top1_rate'] >= thresholds['unaffected_top1_floor'],
            {'unaffected_top1_floor': thresholds['unaffected_top1_floor']}),
        'unknown_false_recall': _gate(
            'unknown_false_recall', unknown['false_recall_rate'],
            unknown['n'] > 0
            and unknown['false_recall_rate'] <= thresholds['unknown_false_recall_max'],
            {'unknown_false_recall_max': thresholds['unknown_false_recall_max']}),
        'superseded_selection': _gate(
            'superseded_selection', metrics['superseded_selection_rate'],
            metrics['superseded_selection_rate'] is not None
            and metrics['superseded_selection_rate']
            <= thresholds['superseded_selection_max'],
            {'superseded_selection_max': thresholds['superseded_selection_max']}),
        'cross_world_selection': _gate(
            'cross_world_selection', metrics['cross_world_selection_rate'],
            metrics['cross_world_selection_rate'] is not None
            and metrics['cross_world_selection_rate']
            <= thresholds['cross_world_selection_max'],
            {'cross_world_selection_max': thresholds['cross_world_selection_max']}),
        'zero_false_recall': _gate(
            'zero_false_recall', metrics['zero_false_recall_rate'],
            metrics['zero_false_recall_rate'] is not None
            and metrics['zero_false_recall_rate'] <= thresholds['zero_false_recall_max'],
            {'zero_false_recall_max': thresholds['zero_false_recall_max']}),
    }


def _answer_summary(rows, arm):
    selected = [row for row in rows if row['arm'] == arm]
    facts = [row for row in selected if row['kind'] == 'fact']
    unknowns = [row for row in selected if row['kind'] == 'unknown']
    corrected = [row for row in facts if row['target_role'] == 'correction_target']
    unaffected = [row for row in facts if row['target_role'] == 'retained']

    def block(items):
        return {'n': len(items),
                'accuracy': _rate(sum(row['correct'] for row in items), len(items)),
                'answered_rate': _rate(sum(bool(row['answer'].strip())
                                           for row in items), len(items))}

    return {
        'n': len(selected),
        'accuracy': _rate(sum(row['correct'] for row in selected), len(selected)),
        'answered_rate': _rate(sum(bool(row['answer'].strip())
                                   for row in selected), len(selected)),
        'fact': block(facts),
        'unknown': {
            'n': len(unknowns),
            'accuracy': _rate(sum(row['correct'] for row in unknowns),
                              len(unknowns)),
            'false_answer_rate': _rate(
                sum(row['normalized_answer'] != 'unknown'
                    for row in unknowns), len(unknowns)),
        },
        'correction': block(corrected),
        'unaffected': block(unaffected),
        'mean_generated_tokens': _mean([row['generated_tokens'] for row in selected]),
    }


def _stage4_gates(metrics, thresholds):
    memory = metrics['memory']
    exact = metrics['exact_source']
    full = metrics['full_context']
    removed = metrics['context_removed']
    delta = (memory['accuracy'] - removed['accuracy']
             if memory['accuracy'] is not None
             and removed['accuracy'] is not None else None)
    return {
        'memory_answer_accuracy': _gate(
            'memory_answer_accuracy', memory['accuracy'],
            memory['n'] > 0
            and memory['accuracy'] >= thresholds['answer_accuracy_floor'],
            {'answer_accuracy_floor': thresholds['answer_accuracy_floor']}),
        'memory_correction': _gate(
            'memory_correction', memory['correction']['accuracy'],
            memory['correction']['n'] > 0
            and memory['correction']['accuracy']
            >= thresholds['correction_answer_floor'],
            {'correction_answer_floor': thresholds['correction_answer_floor']}),
        'memory_unaffected': _gate(
            'memory_unaffected', memory['unaffected']['accuracy'],
            memory['unaffected']['n'] > 0
            and memory['unaffected']['accuracy']
            >= thresholds['unaffected_answer_floor'],
            {'unaffected_answer_floor': thresholds['unaffected_answer_floor']}),
        'memory_unknown': _gate(
            'memory_unknown', memory['unknown']['false_answer_rate'],
            memory['unknown']['n'] > 0
            and memory['unknown']['false_answer_rate']
            <= thresholds['unknown_answer_false_max'],
            {'unknown_answer_false_max': thresholds['unknown_answer_false_max']}),
        'exact_source_control': _gate(
            'exact_source_control', exact['accuracy'],
            exact['n'] > 0
            and exact['accuracy'] >= thresholds['exact_source_answer_floor'],
            {'exact_source_answer_floor': thresholds['exact_source_answer_floor']}),
        'full_context_control': _gate(
            'full_context_control', full['accuracy'],
            full['n'] > 0
            and full['accuracy'] >= thresholds['full_context_answer_floor'],
            {'full_context_answer_floor': thresholds['full_context_answer_floor']}),
        'memory_vs_removed': _gate(
            'memory_vs_removed',
            {'memory_accuracy': memory['accuracy'],
             'context_removed_accuracy': removed['accuracy'], 'delta': delta},
            delta is not None
            and delta >= thresholds['memory_minus_removed_floor'],
            {'memory_minus_removed_floor':
             thresholds['memory_minus_removed_floor']}),
    }


def _validate_answer_row(row, labels, max_new_tokens):
    _require(row.get('stage') == 4, 'Answer row has the wrong stage')
    _require(row.get('state') == labels.get('state'),
             'Answer row state differs from its verifier labels')
    _require(row.get('query_id') in labels['targets'],
             'Answer row references an underived query')
    _require(labels['targets'][row['query_id']]['query_text_sha256']
             == row.get('query_text_sha256'),
             'Answer query differs from its verifier-derived query')
    _require(row.get('arm') in ANSWER_ARMS, 'Answer row has an unknown arm')
    _require(row.get('status') == 'ok', 'Answer row is not a completed generation')
    _require(isinstance(row.get('answer'), str), 'Answer must be text')
    _require(row['answer_sha256'] == _sha_text(row['answer']),
             'Answer bytes differ from the stored digest')
    token_ids = row.get('token_ids')
    _require(isinstance(token_ids, list)
             and all(type(token) is int for token in token_ids),
             'Generated token IDs are invalid')
    _require(row.get('generated_tokens') == len(token_ids),
             'Generated token count differs')
    _require(len(token_ids) <= max_new_tokens,
             'Generated tokens exceed the declared bound')
    target = labels['targets'][row['query_id']]
    expected = target['expected']
    observed = _normalize_answer(row['answer'])
    correct = observed == _normalize_answer(expected)
    return {
        'stage': 4, 'arm': row['arm'], 'query_id': row['query_id'],
        'episode_id': row['episode_id'], 'world_id': row['world_id'],
        'state': row['state'], 'split': labels['split'],
        'complexity': labels['complexity'],
        'kind': target['kind'], 'target_role': target['target_role'],
        'expected': expected, 'answer': row['answer'],
        'normalized_answer': observed, 'normalized_expected': _normalize_answer(expected),
        'correct': correct, 'generated_tokens': len(token_ids),
        'answer_sha256': row['answer_sha256'],
        'observation_sha256': row['observation_sha256'],
        'context_sha256': row['context_sha256'],
    }


def select_first_failing_stage(stage_results, *, mechanical_valid):
    """Return the first failing stage in order; 0 means every stage passed."""
    if not mechanical_valid:
        return None
    for stage in STAGE_ORDER:
        if stage_results[stage]['status'] == 'fail':
            return stage
    return 0


def _decision_name(stage):
    if stage is None:
        return 'mechanical_state_invalid'
    return {
        0: 'all_stages_pass_native_verdict_unchanged',
        1: 'stage1_storage_upper_bound_failed',
        2: 'stage2_addressing_failed',
        3: 'stage3_value_discrimination_failed',
        4: 'stage4_native_generation_failed',
    }[stage]


def _next_discriminator(stage, stage4_components=None):
    if stage is None:
        return ('Stop: the mechanical zero-update or persistence state is invalid; '
                'no memory conclusion is interpretable.')
    if stage == 1:
        return ('Inspect per-head writer key/value norms, gate/erase values and '
                'exact-key rankings. Storage failure comes before addressing or '
                'reader work; do not propose a new reader.')
    if stage == 2:
        return ('Compare observed-key and rendered-query key-space rankings. A '
                'storage-passing state with failed addressing localises the gap to '
                'the query/writer key path, not to value storage.')
    if stage == 3:
        return ('Compare corrected-fact, unaffected-fact and independent-world '
                'candidate scores. Separate value collisions from correction '
                'interference before proposing any reader or training change.')
    if stage == 4:
        if stage4_components and stage4_components.get('exact_source_control') == 'fail':
            return ('Exact-source and full-context controls are the upper bound: '
                    'if they also fail, the frozen generator/readout is the first '
                    'bottleneck; do not attribute it to fast memory.')
        return ('The frozen generator can answer from source controls, so inspect '
                'consumption of the fast state (route, frame, read digest) before '
                'any writer/reader change.')
    return ('Every declared stage passed on the frozen artifact. Replicate on the '
            'native MPS run before considering any promotion; the prior v5 verdict '
            'is not rewritten.')


def score_matched(readouts, answers, codebooks, labels_by_episode, *,
                  thresholds=None, mechanical=None, inventory=None,
                  failure_limit=None):
    """Recompute every ranked decision from raw readouts and derived labels."""
    settings = _resolve_thresholds(thresholds)
    limits = _resolve_limits((mechanical or {}).get('limits'))
    failure_limit = int(failure_limit or limits['max_preserved_failures'])
    _require(isinstance(codebooks, dict) and codebooks,
             'Matched scoring requires codebooks')
    _require(isinstance(labels_by_episode, dict),
             'Matched scoring requires verifier labels')

    seen = set()
    for row in readouts:
        key = (row['stage'], row['episode_id'], row['state'],
               row['case_id'], row['arm'])
        _require(key not in seen, f'Duplicate matched readout row: {key}')
        seen.add(key)
        _require(row['stage'] in (1, 2, 3), 'Readout row has an unknown stage')
        labels = labels_by_episode.get(row['episode_id'], {}).get(row['state'])
        _require(labels is not None, 'Readout row references underived labels')
        codebook = codebooks.get(row['codebook_key'])
        _require(codebook is not None, 'Readout row references a missing codebook')
        _require(codebook.get('observation_sha256') == row.get('observation_sha256'),
                 'Readout observation differs from its codebook state')
        if row.get('query_id') is not None:
            target = labels['targets'].get(row['query_id'])
            _require(target is not None
                     and target['query_text_sha256'] == row.get('query_text_sha256'),
                     'Readout query differs from its verifier-derived query')
        if (row['stage'] == 1 and not row['case_id'].startswith('control:')):
            _require(row['arm'] == 'exact_key', 'Stage 1 row has the wrong arm')
        elif row['stage'] == 2 and not row['case_id'].startswith('control:'):
            _require(row['arm'] == 'query_key', 'Stage 2 row has the wrong arm')
        elif row['stage'] == 3 and not row['case_id'].startswith('control:'):
            _require(row['arm'] == 'query_value', 'Stage 3 row has the wrong arm')

    ranking_rows = []
    for row in readouts:
        if row.get('status') != 'ok':
            _require(isinstance(row.get('unavailable_reason'), str)
                     and row['unavailable_reason'],
                     'Unavailable row must state its reason')
            continue
        labels = labels_by_episode[row['episode_id']][row['state']]
        ranking_rows.append(_recompute_row(
            row, codebooks[row['codebook_key']], labels, settings))

    answer_seen = set()
    validated_answers = []
    for row in answers:
        key = (row['episode_id'], row['state'], row['query_id'], row['arm'])
        _require(key not in answer_seen, f'Duplicate matched answer row: {key}')
        answer_seen.add(key)
        labels = labels_by_episode.get(row['episode_id'], {}).get(row['state'])
        _require(labels is not None, 'Answer row references underived labels')
        codebook = codebooks.get(f"{row['episode_id']}/{row['state']}")
        _require(codebook is not None
                 and codebook.get('observation_sha256')
                 == row.get('observation_sha256'),
                 'Answer observation differs from its codebook state')
        validated_answers.append(_validate_answer_row(
            row, labels, limits['max_new_tokens']))

    if inventory is not None:
        expected = {tuple(item) for item in inventory}
        observed = {(row['stage'], row['episode_id'], row['state'],
                     row['case_id'], row['arm']) for row in readouts}
        observed |= {(4, row['episode_id'], row['state'], row['query_id'],
                      row['arm']) for row in answers}
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        _require(not missing, f'Missing matched rows: {missing[:5]}')
        _require(not extra, f'Unexpected matched rows: {extra[:5]}')
        _require(len(observed) == len(expected),
                 'Matched row inventory count differs')

    metrics1 = _metric_summary(ranking_rows, stages=1, arm='exact_key')
    metrics2 = _metric_summary(ranking_rows, stages=2, arm='query_key')
    metrics3 = _metric_summary(ranking_rows, stages=3, arm='query_value')
    stage1 = {'name': STAGE_NAMES[1], 'gates': _stage1_gates(metrics1, settings),
              'metrics': metrics1}
    stage2 = {'name': STAGE_NAMES[2], 'gates': _stage2_gates(metrics2, settings),
              'metrics': metrics2}
    stage3 = {'name': STAGE_NAMES[3], 'gates': _stage3_gates(metrics3, settings),
              'metrics': metrics3}
    answer_metrics = {arm: _answer_summary(validated_answers, arm)
                      for arm in ANSWER_ARMS}
    stage4 = {'name': STAGE_NAMES[4], 'gates': {}, 'metrics': answer_metrics}
    stage4['gates'] = _stage4_gates(answer_metrics, settings)
    for stage in (stage1, stage2, stage3, stage4):
        stage['failed_gates'] = sorted(
            name for name, gate in stage['gates'].items()
            if gate['status'] == 'fail')
        stage['status'] = 'fail' if stage['failed_gates'] else 'pass'

    mechanical = mechanical or {}
    frozen = mechanical.get('frozen') or {}
    zero_updates = bool(
        frozen.get('optimizer_updates') == 0
        and frozen.get('parameters_updated') == []
        and frozen.get('parameter_versions_equal') is True
        and frozen.get('requires_grad_all_false') is True)
    parity_rows = mechanical.get('rows') or []
    parity_ok = bool(parity_rows) and all(row.get('equal') is True
                                          for row in parity_rows)
    inventory_ok = inventory is not None
    mechanical_gates = {
        'zero_updates': {'status': 'pass' if zero_updates else 'fail',
                         'metrics': frozen},
        'save_reload_parity': {'status': 'pass' if parity_ok else 'fail',
                               'metrics': {'rows': len(parity_rows),
                                           'equal': sum(bool(row.get('equal'))
                                                        for row in parity_rows)}},
        'row_inventory': {'status': 'pass' if inventory_ok else 'fail',
                          'metrics': {'expected': (len(inventory)
                                                   if inventory is not None else 0),
                                      'observed_readouts': len(readouts),
                                      'observed_answers': len(answers)}},
    }
    mechanical_valid = all(gate['status'] == 'pass'
                           for gate in mechanical_gates.values())
    stage_results = {1: stage1, 2: stage2, 3: stage3, 4: stage4}
    first = select_first_failing_stage(stage_results,
                                       mechanical_valid=mechanical_valid)

    failures = {'stage1': [], 'stage2': [], 'stage3': [], 'stage4': []}
    for row in ranking_rows:
        if row['case_id'].startswith('control:'):
            continue
        if not row['top1'] or not row['accepted'] or row['false_recall']:
            bucket = failures[f"stage{row['stage']}"]
            if len(bucket) < failure_limit:
                bucket.append({
                    'case_id': row['case_id'], 'episode_id': row['episode_id'],
                    'state': row['state'], 'target_entry_id': row['target_entry_id'],
                    'target_role': row['target_role'], 'top_entry_id': row['top_entry_id'],
                    'target_rank': row['target_rank'],
                    'margin_over_strongest_distractor':
                        row['margin_over_strongest_distractor'],
                    'false_recall_kind': row['false_recall_kind'],
                    'scores': row['scores'],
                })
    for row in validated_answers:
        if row['arm'] == 'memory' and not row['correct']:
            bucket = failures['stage4']
            if len(bucket) < failure_limit:
                bucket.append({'query_id': row['query_id'],
                               'episode_id': row['episode_id'],
                               'state': row['state'],
                               'kind': row['kind'],
                               'target_role': row['target_role'],
                               'expected': row['expected'],
                               'answer': row['answer'],
                               'context_sha256': row['context_sha256']})
    failure_counts = {name: len(items) for name, items in failures.items()}

    stage4_components = {
        name: gate['status'] for name, gate in stage4['gates'].items()
    }
    decision = _decision_name(first)
    return {
        'stages': {stage: stage_results[stage] for stage in STAGE_ORDER},
        'stage_verdicts': [
            {'stage': stage, 'name': stage_results[stage]['name'],
             'status': stage_results[stage]['status'],
             'failed_gates': stage_results[stage]['failed_gates']}
            for stage in STAGE_ORDER],
        'failed_gates': {str(stage): stage_results[stage]['failed_gates']
                         for stage in STAGE_ORDER},
        'mechanical_gates': mechanical_gates,
        'mechanical_valid': mechanical_valid,
        'first_failing_stage': first,
        'decision': decision,
        'smallest_next_discriminator': _next_discriminator(
            first, stage4_components),
        'stage4_components': stage4_components,
        'thresholds': settings,
        'rankings': ranking_rows,
        'answers': validated_answers,
        'failures': failures,
        'failure_counts': failure_counts,
        'summary': {
            'stage1_fact_top1': stage1['metrics']['fact']['top1_rate'],
            'stage1_fact_margin': stage1['metrics']['fact']['mean_margin'],
            'stage2_fact_top1': stage2['metrics']['fact']['top1_rate'],
            'stage2_fact_margin': stage2['metrics']['fact']['mean_margin'],
            'stage3_fact_top1': stage3['metrics']['fact']['top1_rate'],
            'stage3_fact_margin': stage3['metrics']['fact']['mean_margin'],
            'stage3_correction_top1': stage3['metrics']['correction']['top1_rate'],
            'stage3_unaffected_top1': stage3['metrics']['unaffected']['top1_rate'],
            'stage3_unknown_false_recall':
                stage3['metrics']['unknown']['false_recall_rate'],
            'stage4_memory_accuracy': answer_metrics['memory']['accuracy'],
            'stage4_exact_source_accuracy':
                answer_metrics['exact_source']['accuracy'],
            'stage4_full_context_accuracy':
                answer_metrics['full_context']['accuracy'],
            'stage4_context_removed_accuracy':
                answer_metrics['context_removed']['accuracy'],
            'readout_rows': len(readouts), 'answer_rows': len(answers),
        },
    }


# ---------------------------------------------------------------------------
# Full diagnostic run
# ---------------------------------------------------------------------------

def run_matched_diagnostic(model, tokenizer, protocol, *, output_dir,
                           ledger_dir=None, device='cpu', episodes=None,
                           answerer_factory=None,
                           writer_extension_receipt=None):
    """Run all four stages against one identical episode inventory.

    ``answerer_factory`` is an explicit test seam: it receives a state ledger and
    returns a callable with the frozen answerer's signature.  Production uses
    :func:`_default_answerer_factory`, i.e. the existing ``MemorySession`` path.
    """
    _require(protocol.get('format') == PROTOCOL_FORMAT,
             'Matched diagnostic requires its versioned protocol format')
    thresholds = _resolve_thresholds(protocol.get('thresholds'))
    limits = _resolve_limits(protocol.get('limits'))
    budget = protocol.get('budget') or {}
    _require(isinstance(budget, dict), 'Protocol budget must be an object')
    max_wall_seconds = budget.get('max_wall_seconds')
    _require(max_wall_seconds is None
             or (type(max_wall_seconds) is int and max_wall_seconds > 0),
             'Protocol wall-clock budget must be a positive integer')
    layer = protocol.get('layer')
    available = tuple(model.ports.injection_layers)
    _require(type(layer) is int and layer in available,
             f'Declared layer {layer} is not an injection layer {available}')
    supplied = episodes if episodes is not None else protocol.get('episodes')
    episodes = supplied if supplied is not None else protocol.get('worlds')
    _require(episodes is not None, 'Matched diagnostic requires episodes')
    validate_matched_episodes(
        episodes, families_per_split=protocol.get('families_per_split'),
        complexities=tuple(protocol.get('complexities') or COMPLEXITIES),
        variants=int(protocol.get('variants', DEFAULT_VARIANTS)))
    workload = matched_workload(episodes, protocol)
    inventory = matched_inventory(episodes)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for name in ('readouts.jsonl', 'answers.jsonl', 'codebooks.json',
                 'mechanical.json', 'rankings.jsonl', 'failures.json'):
        _require(not (output / name).exists(),
                 f'Matched raw artifact already exists: {name}')
    ledger_dir = Path(ledger_dir) if ledger_dir is not None else output / 'ledger'
    ledger_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output / 'images'
    images_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    requires_grad_before = {name: parameter.requires_grad
                            for name, parameter in model.named_parameters()}
    _freeze(model)
    versions_before = _parameter_versions(model)
    started = _utc_now()
    started_monotonic = time.monotonic()
    readouts, answers, codebooks, mechanical_rows = [], [], {}, []
    if answerer_factory is None:
        answerer_factory = _default_answerer_factory(model, tokenizer, protocol)

    with torch.no_grad():
        for episode in episodes:
            for state in episode_states(episode):
                memory = _build_memory(model, tokenizer, episode, state)
                codebook = _build_codebook(
                    model, tokenizer, episode, state, memory, layer=layer)
                codebook_key = f"{episode['episode_id']}/{state['name']}"
                _require(codebook_key not in codebooks,
                         'Duplicate matched codebook key')
                codebooks[codebook_key] = codebook
                mechanical_rows.append(_mechanical_reload(
                    memory, model, tokenizer, episode, state, images_dir,
                    codebook))
                readouts.extend(_evaluate_readouts(
                    model, tokenizer, episodes, episode, state, memory,
                    codebook, layer=layer))
                ledger = _ledger_for_state(episode, state, ledger_dir)
                reference = ExactMemoryReference(ledger)
                try:
                    scope = _state_scope(episode, state)
                    answerer = answerer_factory(ledger)
                    answers.extend(_evaluate_answers(
                        episode, state, memory, reference=reference,
                        scope=scope, answerer=answerer, tokenizer=tokenizer,
                        protocol={**protocol,
                                  'generation': protocol.get('generation', {
                                      'max_new_tokens': limits['max_new_tokens'],
                                      'max_prompt_tokens': limits['max_prompt_tokens']})},
                        device=device))
                finally:
                    reference.close()
                    ledger.close()
                memory = None
                if max_wall_seconds is not None:
                    enforce_elapsed_deadline(
                        started_monotonic, max_wall_seconds,
                        label='matched diagnostic')

    readouts.sort(key=lambda row: (row['stage'], row['episode_id'],
                                   row['state'], row['case_id'], row['arm']))
    answers.sort(key=lambda row: (row['episode_id'], row['state'],
                                  row['query_id'], row['arm']))
    for row in readouts + answers:
        _require(not any(marker in row for marker in
                         ('expected', 'expected_answer', 'gold', 'label',
                          'audit_label')),
                 'Raw matched rows must not carry evaluator labels')
    _write_jsonl(output / 'readouts.jsonl', readouts)
    _write_jsonl(output / 'answers.jsonl', answers)
    _write_json(output / 'codebooks.json',
                {'format': CODEBOOK_FORMAT + '.collection.v1',
                 'codebooks': codebooks})
    _write_json(output / 'mechanical.json',
                {'format': RUN_FORMAT + '.mechanical.v1',
                 'rows': mechanical_rows})

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
        'frozen': frozen, 'rows': mechanical_rows,
        'limits': limits,
    }
    labels = derive_matched_labels(episodes)
    score = score_matched(
        readouts, answers, codebooks, labels, thresholds=thresholds,
        mechanical=mechanical, inventory=inventory,
        failure_limit=limits['max_preserved_failures'])
    _write_jsonl(output / 'rankings.jsonl', score['rankings'])
    _write_json(output / 'failures.json', {
        'format': RUN_FORMAT + '.failures.v1',
        'counts': score['failure_counts'], 'failures': score['failures'],
        'first_failing_stage': score['first_failing_stage'],
        'decision': score['decision'],
    })

    def _hash(path):
        return _sha_file(output / path)

    elapsed_seconds = time.monotonic() - started_monotonic
    if max_wall_seconds is not None:
        enforce_elapsed_deadline(started_monotonic, max_wall_seconds,
                                 label='matched diagnostic')
    resource = {
        'started_at_utc': started,
        'finished_at_utc': _utc_now(),
        'elapsed_seconds': elapsed_seconds,
        'deadline_seconds': max_wall_seconds,
        'deadline_ok': (True if max_wall_seconds is None
                        else elapsed_seconds <= max_wall_seconds),
        'rss_start_bytes': _rss_bytes(),
        'max_rss_bytes': _rss_bytes(),
        'mps': _mps_memory() if device == 'mps' else None,
        'ledger_bytes': sum(path.stat().st_size for path in ledger_dir.glob('*.sqlite')),
        'image_bytes': sum(path.stat().st_size for path in images_dir.rglob('*')
                           if path.is_file()),
    }
    return {
        'format': RUN_FORMAT, 'status': 'completed', 'native': device == 'mps',
        'device': device, 'fallback_disabled': True,
        'guard_required': 'REPLICA_CAPACITY_GUARD=1',
        'seed': protocol.get('seed'),
        'layer': layer, 'representation': FACT_REPRESENTATION,
        'writer_extension_receipt': writer_extension_receipt,
        'counts': {
            'episodes': len(episodes), 'states': workload['states'],
            'readout_rows': len(readouts), 'answer_rows': len(answers),
            'generated_tokens': sum(row['generated_tokens'] for row in answers),
            'unique_cases': len(inventory),
        },
        'workload': workload, 'inventory': list(inventory),
        'readouts_sha256': _hash('readouts.jsonl'),
        'answers_sha256': _hash('answers.jsonl'),
        'codebooks_sha256': _hash('codebooks.json'),
        'mechanical_sha256': _hash('mechanical.json'),
        'rankings_sha256': _hash('rankings.jsonl'),
        'failures_sha256': _hash('failures.json'),
        'mechanical': mechanical,
        **score,
        'optimizer_updates': 0, 'parameters_updated': [],
        'promote_v5_memory': False,
        'v5_status': 'rejected_by_prior_evidence',
        'neural_promotion': False,
        'deployment': 'diagnostic_only_not_published',
        'quality_scope': ('Matched-episode localisation only. Stage 1 is a storage '
                          'upper bound, not addressing. No training, no useful-memory '
                          'or general-language claim.'),
        'resource': resource,
    }


# ---------------------------------------------------------------------------
# Preparation / admission
# ---------------------------------------------------------------------------

def _protocol_from_config(config, *, episodes_sha256, labels_sha256,
                          config_sha256, config_path):
    _require(config.get('format') == CONFIG_FORMAT,
             'Matched configuration format differs')
    _require(config.get('representation', FACT_REPRESENTATION)
             == FACT_REPRESENTATION,
             'Matched config requires the fact-binding representation')
    thresholds = _resolve_thresholds(config.get('thresholds'))
    limits = _resolve_limits(config.get('limits'))
    families = _resolve_families(config.get('families_per_split'))
    variants = int(config.get('variants', DEFAULT_VARIANTS))
    layer = int(config.get('layer', DEFAULT_LAYER))
    generation = config.get('generation') or {}
    _require(generation.get('max_new_tokens', limits['max_new_tokens'])
             == limits['max_new_tokens'],
             'Generation token budget differs from the limit declaration')
    _require(generation.get('max_prompt_tokens', limits['max_prompt_tokens'])
             == limits['max_prompt_tokens'],
             'Prompt budget differs from the limit declaration')
    source = config.get('source')
    _require(isinstance(source, dict), 'Configuration carries no source pins')
    extension = config.get('writer_extension')
    _require(isinstance(extension, dict)
             and extension.get('weights_sha256')
             and extension.get('path'),
             'Configuration must pin the frozen rejected writer extension')
    budget = dict(config.get('budget') or {})
    max_wall = budget.get('max_wall_seconds', 900)
    _require(type(max_wall) is int and 1 <= max_wall <= 86400,
             'Declared wall-clock budget must be 1..86400 seconds')
    _require(int(budget.get('optimizer_updates', 0)) == 0,
             'Matched diagnostic must declare zero optimizer updates')
    _require(int(budget.get('native_training_jobs', 0)) == 0,
             'Matched diagnostic must declare zero native training jobs')
    budget['max_wall_seconds'] = max_wall
    return {
        'format': PROTOCOL_FORMAT, 'config_format': CONFIG_FORMAT,
        'config_path': config_path, 'config_sha256': config_sha256,
        'seed': int(config['seed']), 'families_per_split': families,
        'complexities': list(config.get('complexities') or COMPLEXITIES),
        'variants': variants,
        'layer': layer, 'representation': FACT_REPRESENTATION,
        'engine': dict(config.get('engine') or DEFAULT_ENGINE),
        'thresholds': thresholds, 'limits': limits, 'budget': budget,
        'generation': {'max_new_tokens': limits['max_new_tokens'],
                       'max_prompt_tokens': limits['max_prompt_tokens'],
                       'greedy': True},
        'answer_arms': list(ANSWER_ARMS),
        'source': source, 'writer_extension': extension,
        'episodes': {'path': 'episodes.json', 'sha256': episodes_sha256,
                     'format': EPISODES_FORMAT},
        'labels': {'path': 'labels.json', 'sha256': labels_sha256,
                   'format': LABELS_FORMAT},
        'zero_updates': True, 'optimizer_updates': 0,
        'native_jobs': 0,
        'declared_stage_order': list(STAGE_ORDER),
        'stop_conditions': [
            'capacity or host-pressure guard',
            'nonfinite state or readout',
            'hash/provenance/scope/source-view mismatch',
            'row inventory, token or artifact bound',
            'declared monotonic wall-clock deadline',
            'frozen parameter or save/reload parity failure',
            'missing or stale MPS telemetry',
        ],
        'mechanical_separation': ('Zero updates and save/reload parity are '
                                  'reported separately from useful-memory stages.'),
    }


def prepare_matched(output, *, config_path=None, config=None, tokenizer=None):
    """Write deterministic episodes, verifier labels and exact pins; no model.

    The sealed tokenizer is loaded for a prompt-bound audit only.  No model
    weights are loaded, no neural forward runs and no native computation occurs.
    """
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Prepare output must be fresh or empty')
    config_path = Path(config_path or PROJECT_ROOT / 'configs/memory_matched_v1.json')
    if config is None:
        config = json.loads(config_path.read_text())
    _require(config.get('format') == CONFIG_FORMAT,
             'Matched configuration format differs')
    seed = int(config['seed'])
    families = _resolve_families(config.get('families_per_split'))
    variants = int(config.get('variants', DEFAULT_VARIANTS))
    complexities = tuple(config.get('complexities') or COMPLEXITIES)
    episodes = generate_matched_episodes(
        seed=seed, families_per_split=families, complexities=complexities,
        variants=variants)
    audit = validate_matched_episodes(
        episodes, families_per_split=families, complexities=complexities,
        variants=variants)
    labels = derive_matched_labels(episodes)
    output.mkdir(parents=True, exist_ok=True)
    episodes_path = output / 'episodes.json'
    labels_path = output / 'labels.json'
    _write_json(episodes_path, {
        'format': EPISODES_FORMAT, 'seed': seed,
        'families_per_split': families, 'complexities': list(complexities),
        'variants': variants, 'audit': audit, 'episodes': episodes})
    _write_json(labels_path, {
        'format': LABELS_FORMAT, 'evaluation_only': True,
        'note': ('Derived by the verifier from observed records; never consumed '
                 'by a writer, query encoder, sense input or prompt.'),
        'episodes': labels})
    source = _source_pins(PROJECT_ROOT / config['source']['model_path'],
                          PROJECT_ROOT / config['source']['model_lock'])
    extension = _extension_pins(PROJECT_ROOT / config['writer_extension']['path'])
    _require(extension['weights_sha256']
             == config['writer_extension']['weights_sha256'],
             'Frozen writer extension bytes differ from the configuration pin')
    configured_bound = _resolve_limits(config.get('limits'))['max_prompt_tokens']
    if tokenizer is None:
        tokenizer = _load_sealed_tokenizer(config['source']['model_path'])
    prompt_audit = audit_prompt_bounds(
        episodes, tokenizer, configured_bound=configured_bound)
    prompt_audit_path = output / 'prompt_audit.json'
    _write_json(prompt_audit_path, prompt_audit)
    configured_generation_bound = _resolve_limits(
        config.get('limits'))['max_new_tokens']
    generation_audit = audit_generation_bounds(
        episodes, labels, tokenizer,
        configured_bound=configured_generation_bound)
    generation_audit_path = output / 'generation_audit.json'
    _write_json(generation_audit_path, generation_audit)
    if prompt_audit['tokenizer_model_max_length'] is not None:
        _require(prompt_audit['bound'] + generation_audit['bound']
                 <= prompt_audit['tokenizer_model_max_length'],
                 'Prompt plus generation budget exceeds the sealed position budget')
    implementation = _implementation_sha256()
    protocol = _protocol_from_config(
        config, episodes_sha256=_sha_file(episodes_path),
        labels_sha256=_sha_file(labels_path),
        config_sha256=_sha_file(config_path),
        config_path=_display_path(config_path))
    protocol['source'] = source
    protocol['writer_extension'] = extension
    protocol['implementation_sha256'] = implementation
    protocol['limits']['max_prompt_tokens'] = prompt_audit['bound']
    protocol['generation']['max_prompt_tokens'] = prompt_audit['bound']
    protocol['prompt_audit'] = {
        'format': prompt_audit['format'],
        'path': 'prompt_audit.json',
        'sha256': _sha_file(prompt_audit_path),
        'configured_bound': prompt_audit['configured_bound'],
        'bound': prompt_audit['bound'],
        'observed_max_tokens': prompt_audit['observed_max_tokens'],
        'max_row': prompt_audit['max_row'],
        'first_row_over_previous_bound':
            prompt_audit['first_row_over_previous_bound'],
        'rows_over_previous_bound': prompt_audit['rows_over_previous_bound'],
        'previous_bound_would_fail':
            prompt_audit['previous_bound_would_fail'],
        'per_arm_max_tokens': {
            arm: prompt_audit['per_arm'][arm]['max_tokens']
            for arm in ANSWER_ARMS},
        'total_rows': prompt_audit['total_rows'],
        'complete_pages': prompt_audit['complete_pages'],
        'truncation': prompt_audit['truncation'],
        'tokenizer_identity': prompt_audit['tokenizer_identity'],
        'tokenizer_model_max_length':
            prompt_audit['tokenizer_model_max_length'],
        'bound_justification': prompt_audit['bound_justification'],
    }
    protocol['limits']['max_new_tokens'] = generation_audit['bound']
    protocol['generation']['max_new_tokens'] = generation_audit['bound']
    protocol['generation_audit'] = {
        'format': generation_audit['format'],
        'path': 'generation_audit.json',
        'sha256': _sha_file(generation_audit_path),
        'configured_bound': generation_audit['configured_bound'],
        'bound': generation_audit['bound'],
        'observed_max_required_tokens':
            generation_audit['observed_max_required_tokens'],
        'observed_max_required_with_eos':
            generation_audit['observed_max_required_with_eos'],
        'max_row': generation_audit['max_row'],
        'first_row_over_previous_bound':
            generation_audit['first_row_over_previous_bound'],
        'first_row_over_previous_bound_answer_only':
            generation_audit['first_row_over_previous_bound_answer_only'],
        'rows_over_previous_bound': generation_audit['rows_over_previous_bound'],
        'rows_over_previous_bound_answer_only':
            generation_audit['rows_over_previous_bound_answer_only'],
        'previous_bound_would_fail':
            generation_audit['previous_bound_would_fail'],
        'previous_bound_answer_only_would_fail':
            generation_audit['previous_bound_answer_only_would_fail'],
        'per_kind': generation_audit['per_kind'],
        'per_role': generation_audit['per_role'],
        'total_targets': generation_audit['total_targets'],
        'eos_token_id': generation_audit['eos_token_id'],
        'eos_required': generation_audit['eos_required'],
        'answer_truncation': generation_audit['answer_truncation'],
        'eos_suppression': generation_audit['eos_suppression'],
        'tokenizer_identity': generation_audit['tokenizer_identity'],
        'tokenizer_model_max_length':
            generation_audit['tokenizer_model_max_length'],
        'bound_justification': generation_audit['bound_justification'],
    }
    protocol['planned_workload'] = matched_workload(episodes, protocol)
    protocol_path = output / 'protocol.json'
    _write_json(protocol_path, protocol)
    manifest = {
        'format': PREPARED_FORMAT, 'created_at_utc': _utc_now(),
        'native_computation': False,
        'audit_tokenizer_only': True,
        'seed': seed, 'variants': variants,
        'families_per_split': families,
        'episodes_sha256': _sha_file(episodes_path),
        'labels_sha256': _sha_file(labels_path),
        'prompt_audit_sha256': _sha_file(prompt_audit_path),
        'prompt_bound': prompt_audit['bound'],
        'observed_max_prompt_tokens': prompt_audit['observed_max_tokens'],
        'previous_bound_would_fail': prompt_audit['previous_bound_would_fail'],
        'per_arm_max_prompt_tokens': {
            arm: prompt_audit['per_arm'][arm]['max_tokens']
            for arm in ANSWER_ARMS},
        'generation_audit_sha256': _sha_file(generation_audit_path),
        'max_new_tokens': generation_audit['bound'],
        'observed_max_required_tokens':
            generation_audit['observed_max_required_tokens'],
        'observed_max_required_with_eos':
            generation_audit['observed_max_required_with_eos'],
        'generation_previous_bound_would_fail':
            generation_audit['previous_bound_would_fail'],
        'generation_previous_bound_answer_only_would_fail':
            generation_audit['previous_bound_answer_only_would_fail'],
        'per_kind_max_required_tokens': {
            kind: metric['max_required_tokens']
            for kind, metric in generation_audit['per_kind'].items()},
        'per_kind_max_required_with_eos': {
            kind: metric['max_required_with_eos']
            for kind, metric in generation_audit['per_kind'].items()},
        'per_role_max_required_tokens': {
            role: metric['max_required_tokens']
            for role, metric in generation_audit['per_role'].items()},
        'per_role_max_required_with_eos': {
            role: metric['max_required_with_eos']
            for role, metric in generation_audit['per_role'].items()},
        'eos_token_id': generation_audit['eos_token_id'],
        'answer_truncation': generation_audit['answer_truncation'],
        'eos_suppression': generation_audit['eos_suppression'],
        'protocol_sha256': _sha_file(protocol_path),
        'config_sha256': protocol['config_sha256'],
        'audit': audit,
        'planned_workload': protocol['planned_workload'],
        'budget': protocol['budget'],
        'source': source, 'writer_extension': extension,
        'implementation_sha256': implementation,
    }
    _write_json(output / 'manifest.json', manifest)
    return manifest


def verify_matched_prepared(prepared):
    """Re-verify data, labels, pins and regenerative determinism."""
    prepared = Path(prepared)
    manifest_path = prepared / 'manifest.json'
    episodes_path = prepared / 'episodes.json'
    labels_path = prepared / 'labels.json'
    protocol_path = prepared / 'protocol.json'
    prompt_audit_path = prepared / 'prompt_audit.json'
    generation_audit_path = prepared / 'generation_audit.json'
    for path in (manifest_path, episodes_path, labels_path, protocol_path,
                 prompt_audit_path, generation_audit_path):
        _require(path.is_file(), f'Prepared matched artifact is missing: {path.name}')
    manifest = json.loads(manifest_path.read_text())
    protocol = json.loads(protocol_path.read_text())
    data = json.loads(episodes_path.read_text())
    labels = json.loads(labels_path.read_text())
    prompt_audit = json.loads(prompt_audit_path.read_text())
    generation_audit = json.loads(generation_audit_path.read_text())
    _require(manifest.get('format') == PREPARED_FORMAT,
             'Prepared matched manifest format differs')
    _require(manifest.get('native_computation') is False,
             'Preparation must not perform native computation')
    _require(protocol.get('format') == PROTOCOL_FORMAT
             and data.get('format') == EPISODES_FORMAT
             and labels.get('format') == LABELS_FORMAT,
             'Prepared matched format differs')
    _require(manifest.get('episodes_sha256') == _sha_file(episodes_path)
             and manifest.get('labels_sha256') == _sha_file(labels_path)
             and manifest.get('prompt_audit_sha256')
             == _sha_file(prompt_audit_path)
             and manifest.get('generation_audit_sha256')
             == _sha_file(generation_audit_path)
             and manifest.get('protocol_sha256') == _sha_file(protocol_path),
             'Prepared matched bytes differ from their manifest pins')
    _require(protocol['episodes']['sha256'] == _sha_file(episodes_path)
             and protocol['labels']['sha256'] == _sha_file(labels_path)
             and protocol['prompt_audit']['sha256']
             == _sha_file(prompt_audit_path)
             and protocol['generation_audit']['sha256']
             == _sha_file(generation_audit_path),
             'Protocol episode/label/audit pins differ')
    episodes = data.get('episodes')
    audit = validate_matched_episodes(
        episodes, families_per_split=data['families_per_split'],
        complexities=tuple(data['complexities']), variants=data['variants'])
    _require(audit == data['audit'] == manifest.get('audit'),
             'Prepared episode audit changed')
    regenerated = generate_matched_episodes(
        seed=data['seed'], families_per_split=data['families_per_split'],
        complexities=tuple(data['complexities']), variants=data['variants'])
    _require(episodes == regenerated,
             'Prepared episodes differ from the seeded generator')
    _require(labels['episodes'] == derive_matched_labels(episodes),
             'Prepared verifier labels differ from the derived labels')
    _require(protocol.get('planned_workload') == matched_workload(episodes, protocol)
             and manifest.get('planned_workload') == protocol['planned_workload'],
             'Prepared matched workload differs from the exact plan')
    _require(protocol.get('budget') == manifest.get('budget')
             and isinstance(protocol.get('budget'), dict)
             and type(protocol['budget'].get('max_wall_seconds')) is int
             and protocol['budget']['max_wall_seconds'] > 0,
             'Prepared wall-clock budget differs or is invalid')
    _require(protocol['config_sha256'] == manifest['config_sha256'],
             'Configuration pin differs between protocol and manifest')
    config_path = Path(protocol.get('config_path') or
                       'configs/memory_matched_v1.json')
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    _require(config_path.is_file()
             and _sha_file(config_path) == protocol['config_sha256'],
             'Configuration bytes changed after preparation')
    current = _implementation_sha256()
    prepared_pins = protocol.get('implementation_sha256')
    _require(manifest.get('implementation_sha256') == prepared_pins,
             'Matched implementation pins differ between protocol and manifest')
    implementation_migrations = _verify_implementation_migrations(
        prepared, prepared_pins=prepared_pins, current_pins=current,
        protocol_sha256=_sha_file(protocol_path),
        manifest_sha256=_sha_file(manifest_path))
    source = protocol.get('source')
    _require(isinstance(source, dict) and source == _source_pins(
        source.get('model_path'), source.get('model_lock')),
        'Matched source pins changed')
    _require(manifest.get('source') == source,
             'Matched source pins differ between protocol and manifest')
    tokenizer = _load_sealed_tokenizer(source['model_path'])
    recomputed_prompt_audit = audit_prompt_bounds(
        episodes, tokenizer,
        configured_bound=prompt_audit['configured_bound'])
    _require(recomputed_prompt_audit == prompt_audit,
             'Prepared prompt audit differs from the sealed-tokenizer audit')
    recomputed_generation_audit = audit_generation_bounds(
        episodes, labels['episodes'], tokenizer,
        configured_bound=generation_audit['configured_bound'])
    _require(recomputed_generation_audit == generation_audit,
             'Prepared generation audit differs from the sealed-tokenizer audit')
    prompt_bound = prompt_audit['bound']
    _require(protocol['limits']['max_prompt_tokens'] == prompt_bound
             and protocol['generation']['max_prompt_tokens'] == prompt_bound,
             'Protocol prompt bound differs from the audited bound')
    _require(prompt_audit['observed_max_tokens'] <= prompt_bound
             and all(row['tokens'] <= prompt_bound
                     for row in prompt_audit['rows']),
             'Audited prompt rows exceed the protocol prompt bound')
    _require(prompt_audit['complete_pages'] is True
             and prompt_audit['truncation'] is False,
             'Prompt audit must keep complete pages without truncation')
    _require(manifest.get('prompt_bound') == prompt_bound
             and manifest.get('observed_max_prompt_tokens')
             == prompt_audit['observed_max_tokens']
             and manifest.get('per_arm_max_prompt_tokens')
             == {arm: prompt_audit['per_arm'][arm]['max_tokens']
                 for arm in ANSWER_ARMS}
             and manifest.get('previous_bound_would_fail')
             == prompt_audit['previous_bound_would_fail'],
             'Prompt-audit manifest summary differs from the audit')
    _require(protocol['prompt_audit']['bound'] == prompt_bound
             and protocol['prompt_audit']['observed_max_tokens']
             == prompt_audit['observed_max_tokens']
             and protocol['prompt_audit']['per_arm_max_tokens']
             == {arm: prompt_audit['per_arm'][arm]['max_tokens']
                 for arm in ANSWER_ARMS},
             'Prompt-audit protocol summary differs from the audit')
    generation_bound = generation_audit['bound']
    _require(protocol['limits']['max_new_tokens'] == generation_bound
             and protocol['generation']['max_new_tokens'] == generation_bound,
             'Protocol generation bound differs from the audited bound')
    _require(generation_audit['observed_max_required_with_eos']
             == generation_bound
             and all(row['required_with_eos'] <= generation_bound
                     for row in generation_audit['rows']),
             'Audited expected answers exceed the generation bound')
    _require(generation_audit['answer_truncation'] is False
             and generation_audit['eos_suppression'] is False,
             'Generation audit must exclude truncation and EOS suppression')
    if prompt_audit['tokenizer_model_max_length'] is not None:
        _require(prompt_bound + generation_bound
                 <= prompt_audit['tokenizer_model_max_length'],
                 'Prompt plus generation budget exceeds the sealed position budget')
    _require(manifest.get('max_new_tokens') == generation_bound
             and manifest.get('observed_max_required_tokens')
             == generation_audit['observed_max_required_tokens']
             and manifest.get('observed_max_required_with_eos')
             == generation_audit['observed_max_required_with_eos']
             and manifest.get('generation_previous_bound_would_fail')
             == generation_audit['previous_bound_would_fail']
             and manifest.get('generation_previous_bound_answer_only_would_fail')
             == generation_audit['previous_bound_answer_only_would_fail']
             and manifest.get('per_kind_max_required_tokens')
             == {kind: metric['max_required_tokens']
                 for kind, metric in generation_audit['per_kind'].items()}
             and manifest.get('per_kind_max_required_with_eos')
             == {kind: metric['max_required_with_eos']
                 for kind, metric in generation_audit['per_kind'].items()}
             and manifest.get('per_role_max_required_tokens')
             == {role: metric['max_required_tokens']
                 for role, metric in generation_audit['per_role'].items()}
             and manifest.get('per_role_max_required_with_eos')
             == {role: metric['max_required_with_eos']
                 for role, metric in generation_audit['per_role'].items()}
             and manifest.get('eos_token_id')
             == generation_audit['eos_token_id']
             and manifest.get('answer_truncation') is False
             and manifest.get('eos_suppression') is False,
             'Generation-audit manifest summary differs from the audit')
    _require(protocol['generation_audit']['bound'] == generation_bound
             and protocol['generation_audit']['observed_max_required_tokens']
             == generation_audit['observed_max_required_tokens']
             and protocol['generation_audit']['observed_max_required_with_eos']
             == generation_audit['observed_max_required_with_eos']
             and protocol['generation_audit']['previous_bound_would_fail']
             == generation_audit['previous_bound_would_fail']
             and protocol['generation_audit'][
                 'previous_bound_answer_only_would_fail']
             == generation_audit['previous_bound_answer_only_would_fail']
             and protocol['generation_audit']['per_kind']
             == generation_audit['per_kind']
             and protocol['generation_audit']['per_role']
             == generation_audit['per_role']
             and protocol['generation_audit']['eos_token_id']
             == generation_audit['eos_token_id']
             and protocol['generation_audit']['answer_truncation'] is False
             and protocol['generation_audit']['eos_suppression'] is False,
             'Generation-audit protocol summary differs from the audit')
    extension = protocol.get('writer_extension')
    _require(isinstance(extension, dict)
             and _extension_pins(extension['path']) == extension,
             'Frozen writer extension pins changed')
    _require(manifest.get('writer_extension') == extension,
             'Writer extension pins differ between protocol and manifest')
    return {
        'format': ADMISSION_FORMAT, 'status': 'admitted',
        'created_at_utc': _utc_now(),
        'prepared_manifest_sha256': _sha_file(manifest_path),
        'episodes_sha256': _sha_file(episodes_path),
        'labels_sha256': _sha_file(labels_path),
        'prompt_audit_sha256': _sha_file(prompt_audit_path),
        'prompt_bound': prompt_bound,
        'observed_max_prompt_tokens': prompt_audit['observed_max_tokens'],
        'per_arm_max_prompt_tokens': {
            arm: prompt_audit['per_arm'][arm]['max_tokens']
            for arm in ANSWER_ARMS},
        'previous_bound_would_fail':
            prompt_audit['previous_bound_would_fail'],
        'generation_audit_sha256': _sha_file(generation_audit_path),
        'max_new_tokens': generation_bound,
        'observed_max_required_tokens':
            generation_audit['observed_max_required_tokens'],
        'observed_max_required_with_eos':
            generation_audit['observed_max_required_with_eos'],
        'generation_previous_bound_would_fail':
            generation_audit['previous_bound_would_fail'],
        'per_kind_max_required_tokens': {
            kind: metric['max_required_tokens']
            for kind, metric in generation_audit['per_kind'].items()},
        'per_role_max_required_with_eos': {
            role: metric['max_required_with_eos']
            for role, metric in generation_audit['per_role'].items()},
        'eos_token_id': generation_audit['eos_token_id'],
        'protocol_sha256': _sha_file(protocol_path),
        'config_sha256': protocol['config_sha256'],
        'source': source, 'source_lock_sha256': source['model_lock_sha256'],
        'writer_extension': extension,
        'implementation_sha256': current,
        'implementation_migrations': implementation_migrations,
        'seed': data['seed'], 'variants': data['variants'],
        'families_per_split': data['families_per_split'],
        'audit': audit, 'planned_workload': protocol['planned_workload'],
        'checks': {
            'manifest_format': True, 'episode_bytes': True,
            'label_bytes': True, 'protocol_bytes': True,
            'seeded_episode_regeneration': True,
            'verifier_labels_regenerated': True,
            'prompt_audit_regenerated': True,
            'prompt_bound_covers_rows': True,
            'complete_pages_no_truncation': True,
            'generation_audit_regenerated': True,
            'generation_bound_covers_targets': True,
            'no_answer_truncation_or_eos_suppression': True,
            'workload_bound': True, 'config_pin': True,
            'source_pins': True, 'implementation_pins': True,
            'writer_extension_pins': True, 'no_native_computation': True,
            'label_free_writer_inputs': True,
        },
    }


def admit_matched_prepared(prepared, *, write_receipt=True):
    prepared = Path(prepared)
    try:
        receipt = verify_matched_prepared(prepared)
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

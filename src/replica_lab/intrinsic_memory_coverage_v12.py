"""Fixed v12 three-site/two-site source-feature comparison, separately versioned.

This module owns the scientific constants and the *actual* v12 work envelope.
The sealed v11 protocol remains the matched comparison: reader sites, masked
four-ordered-slot layout, teacher objective, optimizer, learning rate, exposure,
template, scorer and generation rules are inherited unchanged. Only the
mechanism (frozen canonical-source hidden states followed by a live trainable
compressor) and the work vocabulary are new.

Two slot-policy fields are declared on purpose:

``slot_policy`` keeps the inherited sealed value ``fixed_eight_masked_padding_v5``
because the older admitted validators in this lineage require that exact string
for the shared eight-slot masked-padding layout; ``slot_policy_v12`` is the
runtime string used by ``contextual_memory_v12``. Neither is allowed to drift
from its own consumer. The inherited field describes the layout contract; it
does not assert that v12 executes v5 code.

No stage total here is inherited silently: every stage is recomputed from the
v12 mechanism below and checked by :func:`validate_protocol`.
"""
from copy import deepcopy
import hashlib
import json

from .intrinsic_memory_budget_v12 import COUNTERS, validate_plan
from .intrinsic_memory_coverage_v11 import (
    FORMAT as PREVIOUS_FORMAT, TRAINING as PREVIOUS_TRAINING,
    TEACHER as PREVIOUS_TEACHER, COVERAGE as PREVIOUS_COVERAGE,
    BUDGET_DERIVATION as PREVIOUS_DERIVATION, PREFIXES as PREVIOUS_PREFIXES,
    plan as previous_plan, totals as totals,
)

FORMAT = 'replica.intrinsic_memory.coverage.v12'
PACKET = 'replica.intrinsic_memory.launch_packet.v12'
AUTHORIZATION = 'replica.intrinsic_memory.external_authorization.v12'
TEACHER_CACHE = 'replica.intrinsic_memory.teacher_cache.v12'
FEATURE_CACHE_FORMAT = 'replica.canonical_source_features.v12'
REPRESENTATION_VALUE = 'four_ordered_frozen_source_contextual_256_v12'
CODEC_VALUE = 'contextual_source_features_fp32_slots_v12'
SLOT_POLICY_V12 = 'fixed_eight_masked_padding_v12'

FIT_CASES = 32
DEVELOPMENT_CASES = 16
GROUP_SIZE = 8
PREFIXES = (128, 256, 384, 512)
OUTER_WALL_SECONDS = 7200

# One canonical span of admitted source text. Splits are bounded to 256 encoded
# bytes, so 256 is a sound upper bound for both its token count and its
# tokenizer input reservation.
SOURCE_MAX_BYTES = 256
SOURCE_MAX_TOKENS = 256
# Per dispatched canonical span: tokenizer reservation (bytes) + frozen forward
# reservation (tokens) can both be charged.
SOURCE_INPUT_TOKENS_PER_SPAN = SOURCE_MAX_BYTES + SOURCE_MAX_TOKENS
FEATURE_CACHE_BYTES = 32 * 1024**2
FEATURE_CACHE_ENTRIES = 64
TEACHER_CACHE_BYTES = 2 * 1024**2

TRAINING = {**PREVIOUS_TRAINING, 'updates': 512,
            'schedule': 'fixed_round_robin_32_worlds_16_passes'}
TEACHER = {**deepcopy(PREVIOUS_TEACHER),
           'cache_format': TEACHER_CACHE,
           'cache_bytes_max': TEACHER_CACHE_BYTES,
           'target_cache': '32_detached_targets_created_once_reused_512_plus_1_updates',
           'feature_contract_bound_in_cache': True}
COVERAGE = deepcopy(PREVIOUS_COVERAGE)

# ---------------------------------------------------------------------------
# X0: a genuinely independent v12 development split.
#
# The v11 development split is retired for this successor: the re-review scout
# read `data/intrinsic_memory_coverage_v11_20261029/development.json`, so that
# split can no longer support a fresh utility claim even though its historical
# receipt still says unopened. Calibration and fitting keep the exposed v11 rows
# for a matched comparison; development is new before any observation.
#
# Independence is structural, not cosmetic. The retired split used a fixed case
# table (pair order 0,0,1,1,2,2,3,3), fixed unknown positions (4 and 7), one
# fixed extra query name and fixed rotations (2, 3). Here the case order, the
# name-to-slot assignment, the unknown sides, the extra query name and the
# palette rotation are all derived from `DEVELOPMENT_SEED` through an explicit
# sha256 byte stream, so the same positional case no longer carries the same
# label. The two declared rules, both group sizes, the counterfactual pairs, the
# two-unknown coverage per group and the scorer are preserved.
#
# Only the sealed oracle, renderer and question builder are reused (the
# scientific content). Row labels are produced deterministically from the
# structured oracle; preparation never inspects or selects on them, and the
# public plan below contains no colour assignment, so it cannot leak a label.
DEVELOPMENT_SEED = 2026102912
DEVELOPMENT_ENTITIES = (
    'Thalric', 'Ulmara', 'Veskin', 'Wendral', 'Yorveth', 'Zaleth',
    'Brannoc', 'Celdra', 'Draveth', 'Feldric', 'Gorveth', 'Halvra',
)
DEVELOPMENT_ROOTS = tuple(f'v12_development_decision_{i:02d}' for i in range(8))
DEVELOPMENT_WORLD_PREFIX = 'v12_development_'
DEVELOPMENT_PALETTE = ('red', 'blue', 'green', 'gold')


def _seed_stream(seed, label, count):
    """Deterministic byte stream from a declared seed; no library RNG state."""
    import hashlib as _hashlib
    blocks, produced, counter = [], 0, 0
    while produced < count:
        block = _hashlib.sha256(f'{seed}|{label}|{counter}'.encode()).digest()
        blocks.append(block); produced += len(block); counter += 1
    return b''.join(blocks)[:count]


def _permute(seed, label, values):
    """Fisher-Yates permutation driven by the seed stream."""
    items = list(values)
    stream = _seed_stream(seed, label, 8*len(items)+16)
    cursor = 0
    for index in range(len(items)-1, 0, -1):
        cursor = cursor % len(stream)
        swap = stream[cursor] % (index+1); cursor += 1
        items[index], items[swap] = items[swap], items[index]
    return items


def v12_development_plan():
    """Seed-driven structural plan; contains no colour assignment.

    Per group: which entity is asked in each pair slot, the order in which the
    four pairs are presented, which side of which pair is the unknown query, the
    two extra entities that back those unknown cases, and the palette rotation.
    None of these fields is a label, so the plan can be published for admission.
    """
    plan = []
    per_group = len(DEVELOPMENT_ENTITIES) // 2
    for group in (0, 1):
        pool = list(DEVELOPMENT_ENTITIES[group*per_group:(group+1)*per_group])
        extras = _permute(DEVELOPMENT_SEED, f'extras|{group}', pool)[:2]
        pair_names = [name for name in pool if name not in extras]
        slots = _permute(DEVELOPMENT_SEED, f'slots|{group}', pair_names)
        order = _permute(DEVELOPMENT_SEED, f'order|{group}', [0, 1, 2, 3])
        unknown_slots = _permute(DEVELOPMENT_SEED, f'unknown|{group}', [0, 1, 2, 3])[:2]
        unknown_side = {slot: _seed_stream(DEVELOPMENT_SEED, f'side|{group}|{slot}', 1)[0] % 2
                        for slot in unknown_slots}
        extra_for_slot = dict(zip(unknown_slots, extras))
        rotation = _seed_stream(DEVELOPMENT_SEED, f'rotation|{group}', 1)[0] % 4
        plan.append({'group': group, 'extras': extras, 'slots': slots, 'order': order,
                     'unknown_side': unknown_side, 'extra_for_slot': extra_for_slot,
                     'rotation': rotation})
    return plan


def v12_development_case_positions(plan):
    """Public positional structure: slot and unknown flag per presented case."""
    positions = []
    for entry in plan:
        for slot_position, slot in enumerate(entry['order']):
            for side in (0, 1):
                positions.append({'group': entry['group'], 'slot': slot,
                                  'side': side,
                                  'unknown': entry['unknown_side'].get(slot) == side})
    return positions


def v12_development_worlds():
    """Deterministic fresh development corpus; labels come from the sealed oracle."""
    from .intrinsic_memory_worlds_v11 import FAMILIES, oracle, question, render
    rows = []
    for entry in v12_development_plan():
        group = entry['group']
        family = FAMILIES[group]
        rotation = entry['rotation']
        palette = (DEVELOPMENT_PALETTE[rotation:]
                   + DEVELOPMENT_PALETTE[:rotation])
        stream = _seed_stream(DEVELOPMENT_SEED, f'colours|{group}', 4*4+8)
        cursor = 0
        for slot_position, slot in enumerate(entry['order']):
            query = entry['slots'][slot]
            root = DEVELOPMENT_ROOTS[group*4+slot]
            # The two sides of a pair must carry different answers, so their new
            # colours are drawn without replacement from the rotated palette.
            first = palette[stream[cursor] % 4]; cursor += 1
            second = [colour for colour in palette if colour != first][stream[cursor] % 3]
            cursor += 1
            for side, new_colour in enumerate((first, second)):
                missing = entry['unknown_side'].get(slot) == side
                entity = entry['extra_for_slot'][slot] if missing else query
                offset = (stream[cursor] % 3) + 1; cursor += 1
                old_colour = palette[(palette.index(new_colour)+offset) % 4]
                facts = ([{'entity': entity, 'color': old_colour, 'revision': 1},
                          {'entity': entity, 'color': new_colour, 'revision': 2}]
                         if family == FAMILIES[1]
                         else [{'entity': entity, 'color': new_colour, 'revision': 1}])
                index = group*8 + slot_position*2 + side
                rows.append({
                    'world_id': f'{DEVELOPMENT_WORLD_PREFIX}{index:02d}',
                    'family': root,
                    'criterion_group': group, 'rule_family': family,
                    'facts': facts, 'query_entity': query,
                    'text': render(facts, family), 'question': question(query, family),
                    'answer': oracle(facts, query),
                    'chronology': max(f['revision'] for f in facts),
                    'source_kind': 'scripted_complete_world_snapshot',
                    'verification': 'structured_revision_oracle_not_generated_explanation'})
    return rows


def validate_v12_development(rows):
    """Structural validation of the fresh split; it never re-reads the answers."""
    if not isinstance(rows, list) or len(rows) != DEVELOPMENT_CASES:
        raise ValueError('fresh development split has the wrong case count')
    if len({r['world_id'] for r in rows}) != DEVELOPMENT_CASES:
        raise ValueError('fresh development world ids are not unique')
    if any(not r['world_id'].startswith(DEVELOPMENT_WORLD_PREFIX) for r in rows):
        raise ValueError('fresh development world id prefix differs')
    if [r['criterion_group'] for r in rows] != [i // GROUP_SIZE for i in range(DEVELOPMENT_CASES)]:
        raise ValueError('fresh development group order differs')
    for index, row in enumerate(rows):
        if not row['text'] or len(row['text'].encode()) > 256:
            raise ValueError('fresh development span is empty or unbounded')
        if row['family'] not in DEVELOPMENT_ROOTS:
            raise ValueError('fresh development decision root differs')
        if not row['text'].startswith(('The recorded color', 'Earlier, the recorded color')):
            raise ValueError('fresh development source text differs')
        if row['answer'] not in ('red', 'blue', 'green', 'gold', 'unknown'):
            raise ValueError('fresh development answer is outside the declared palette')
        if index and row['criterion_group'] == rows[index-1]['criterion_group']:
            if (index - 1) % 2 == 0:
                if row['question'] != rows[index-1]['question'] or row['answer'] == rows[index-1]['answer']:
                    raise ValueError('fresh development counterfactual pair differs')
    for start in range(0, DEVELOPMENT_CASES, GROUP_SIZE):
        group = rows[start:start+GROUP_SIZE]
        if sum(r['answer'] == 'unknown' for r in group) != 2:
            raise ValueError('fresh development group must contain exactly two unknowns')
        if len({r['family'] for r in group}) != 4:
            raise ValueError('fresh development group must contain four decision roots')
        if len({r['rule_family'] for r in group}) != 1:
            raise ValueError('fresh development group must use one rule family')
    # Independence: no entity, root, world id or source text from any v11 split.
    from .intrinsic_memory_worlds_v11 import ENTITIES as V11_ENTITIES
    v11_entities = {name for names in V11_ENTITIES.values() for name in names}
    if set(DEVELOPMENT_ENTITIES) & v11_entities:
        raise ValueError('fresh development reuses a v11 entity')
    if {r['query_entity'] for r in rows} & v11_entities:
        raise ValueError('fresh development query reuses a v11 entity')
    return True


def v12_development_public_identity():
    """Role-bound public metadata for admission; it contains no label or text."""
    validate_v12_development(v12_development_worlds())
    return {'split': 'development', 'role': 'fresh_v12_development',
            'cases': DEVELOPMENT_CASES, 'entities': list(DEVELOPMENT_ENTITIES),
            'decision_roots': list(DEVELOPMENT_ROOTS), 'seed': DEVELOPMENT_SEED,
            'generator': 'seeded_structural_plan_v12',
            'plan_sha256': identity_digest(v12_development_plan()),
            'label_content_included': False}


def public_split_metadata(split):
    """Public metadata of the reused sealed v11 splits; never their labels."""
    from .intrinsic_memory_worlds_v11 import public_split_metadata as sealed
    return sealed(split)

# Every difference from the sealed v11 comparison, declared once. The admission
# validator requires the prepared protocol to carry exactly these lists, so a
# silent change cannot be introduced by editing only the configuration.
DIFFERENCES_FROM_V11 = (
    'value_producer: frozen canonical-source decoder hidden states followed by a live '
    'trainable compressor instead of a trainable encoder over template-framed text',
    'source_input: canonical span text only, no chat template and no special tokens, '
    'explicit zero-based positions, extensions disabled and audited',
    'feature_cache: bounded 32 MiB / 64 entry in-process C that is never serialized',
    'slots: four ordered 256-wide slots per admitted span, learned order added exactly '
    'once by the v12 snapshot path',
    'work_ledger: append-only replica.intrinsic_memory.reservations.v12 journal with '
    'source-feature, canonical-source tokenizer, prompt tokenizer, cache and restart '
    'vocabulary, and dual monotonic/wall deadlines',
    'stage_budget: re-derived per-stage allocation (500 s per fit segment, 900 s per probe, '
    'plus two declared terminal-receipt stages and two capacity-guard stages) instead of the '
    'inherited 7080 s; the authoritative total is stage_wall_value, computed from the caps',
    'checkpoint: v12 complete extension plus optimizer with bound feature, adapter, '
    'tokenizer and layout contracts, the complete representation release, an explicit '
    'release alias and the compressor-only release; no v11 checkpoint is migrated',
    'teacher_cache: v12 format bound to the source-feature contract, adapter contract and '
    'tokenizer contract',
    'slot_policy_declaration: slot_policy_v12 added next to the inherited '
    'fixed_eight_masked_padding_v5 field required by the sealed validators',
    'release_corpus: replica.contextual_producer_release.v12 namespace, now publishing the '
    'complete representation release (producer, order, reader and feature contract), so any '
    'contextual-path update invalidates previously durable values',
    'restart: source features are recomputed from a cleared cache; the recomputed feature '
    'contract and adapter content binding are compared with the checkpoint before mutation',
    'adapter_contract: content-bound through the already-verified full original parameter '
    'digest, so an adapter with identical geometry but different content is refused',
    'data_manifest: new v12 manifest with an explicit exposure contract over the same rows',
)
UNCHANGED_FROM_V11 = (
    'official_BF16_source_and_revision',
    'reader_sites_3_and_7_block_input_query_post_block_residual',
    'eight_masked_ordered_slots_two_span_budget',
    'teacher_objective_top64_kl_temperature1_no_ce_mix',
    'NativeFusedAdamW_learning_rate_0.0001_seed_17',
    '512_fitting_updates_plus_one_restart_update_fixed_round_robin',
    'four_fixed_probe_prefixes_no_best_prefix_selection',
    'per_group_fitting_and_utility_criteria_eight_world_gates',
    'calibration_8_fitting_32_development_16_same_rows',
    'generation_template_scorer_context_and_token_limits',
    'persistent_P_updates_zero_and_no_promotion',
    'physical_resource_limits_and_7200_second_outer_ceiling',
)

SOURCE_FEATURES = {
    'producer': 'contextual_source_features_v12',
    'representation': REPRESENTATION_VALUE,
    'codec': CODEC_VALUE,
    'slot_policy_v12': SLOT_POLICY_V12,
    'feature_contract_format': FEATURE_CACHE_FORMAT,
    'input_policy': 'canonical_span_only_no_template_no_special_tokens',
    'output_policy': 'last_decoder_hidden_after_final_norm',
    'source_positions': 'explicit_zero_to_token_count_minus_one',
    'extensions_during_source_pass': 'disabled_by_arguments_and_audited',
    'source_backbone': 'frozen_eval_bfloat16_no_grad',
    'feature_dtype': 'float32',
    'detach_boundary': 'frozen_original_hidden_before_trainable_compressor',
    'max_tokens': SOURCE_MAX_TOKENS,
    'cache_bytes': FEATURE_CACHE_BYTES,
    'cache_entries': FEATURE_CACHE_ENTRIES,
    'cache_class': 'C_in_process_not_serialized',
    'cache_clear_boundaries': ['correction', 'revocation', 'source_free_transition',
                               'release_publication', 'checkpoint_restore'],
    'original_updates': 0,
    'accounting': 'reserve_before_tokenizer_and_before_frozen_forward',
    'adapter_contract_policy': ('runtime digest over the verified source snapshot, geometry, '
                                'attention implementation, regime and the exact base content '
                                'digest; the content digest is supplied as the already-verified '
                                'full original parameter digest so the backbone is not rehashed, '
                                'and the resulting contract is bound in the v12 checkpoint and '
                                'teacher cache and recomputed on restart'),
}

# Dense arms are declared for envelope comparison only. Their packets are
# deliberately not prepared before a qualified two-site result exists.
ARMS = {'two': [3, 7], 'full_attention_nine': list(range(3, 36, 4)),
        'all': list(range(36))}
GATED_ARMS = ('full_attention_nine', 'all')
CHECKPOINT_BYTES = {'two': 96 * 1024**2, 'full_attention_nine': 384 * 1024**2,
                    'all': 1024 * 1024**2}
ADMISSION_ENVELOPE_BYTES = 96 * 1024**2

# The canonical-source tokenizer contract is derived from the verified source
# receipt, so preparation can declare it without loading the tokenizer. The
# native loader recomputes it with the producer's own helper and refuses any
# difference; the checkpoint and teacher cache bind the recomputed digest.
TOKENIZER_IDENTITY_FILES = ('chat_template.jinja', 'merges.txt', 'special_tokens_map.json',
                            'tokenizer.json', 'tokenizer_config.json', 'vocab.json')
SOURCE_VOCAB_SIZE = 131072
SOURCE_HIDDEN_SIZE = 2560
SOURCE_FEATURE_BINDING_KEYS = ('tokenizer_identity', 'tokenizer_sha256',
                               'original_parameters_sha256', 'source_files_sha256')
# Verified size of the official BF16 local source, from its own receipt. Every
# mandatory hashing pass and weight load reads these bytes; the admission
# validator re-derives the total from the bound file list.
SOURCE_FILE_BYTES = 8238996247

# Restart restores 32 teacher targets and each `expected(row)` performs a student
# encoding and an exact-text encoding. The canary then adds one encoding, and the
# continuation update adds a student encoding plus the exact-text encoding that
# `alignment_loss` re-binds. That is 64 + 3 = 67 prompt-tokenizer reservations;
# the cap below is derived from those call sites, not estimated.
RESTART_TARGET_REVALIDATION_ENCODINGS = FIT_CASES * 2
RESTART_CANARY_ENCODINGS = 1
RESTART_UPDATE_ENCODINGS = 2
RESTART_PROMPT_PASSES = (RESTART_TARGET_REVALIDATION_ENCODINGS
                         + RESTART_CANARY_ENCODINGS + RESTART_UPDATE_ENCODINGS)
RESTART_WALL_SECONDS = 150

# Executable dependency closure of the qualified environment: interpreter, the
# declared module origins, the loaded native extensions, every file in torch/lib,
# every wheel RECORD and the sealed receipt's critical files. The preparation
# script rebuilds the manifest and refuses if this total differs, so the read
# reservation below is tied to the actual closure rather than to an estimate.
ENVIRONMENT_CLOSURE_FILES = 67
ENVIRONMENT_CLOSURE_BYTES = 450061586

# Every durable read may touch the release manifest (bounded to 4 MiB by the
# producer), a 4 KiB slot payload per record, a bounded canonical text span and
# the pointer record. The engine reserves this bound before the read and records
# the observed bytes afterwards, so release source/authority/payload reads are
# charged to native artifact_read_bytes instead of only to local cost fields.
RELEASE_READ_BYTES_BOUND = 4 * 1024**2 + 64 * 1024

# Direct reads the release corpus performs itself (pointer, manifest authority,
# canonical source text, payload values, and the pointer re-read on each batch
# validation). Two authority reads of up to 4 MiB plus payload/text headroom per
# published world; the corpus reserves each of them before reading.
RELEASE_DIRECT_READ_BOUND = 9 * 1024**2

# A durable read persists three descriptors (pending, dispatch reservation,
# terminal). Up to four explicit caller resumes may follow, each persisting at
# most two more. The retrieval reservation therefore bounds the whole read at
# 3 + 2*4 = 11 descriptor writes, and the resume events carry timing only so a
# read is never charged twice for the same descriptor.
MAX_CALLER_RESUMES = 4
READ_BASE_DESCRIPTOR_WRITES = 3
RESUME_DESCRIPTOR_WRITES = 2
READ_WRITE_BOUND = READ_BASE_DESCRIPTOR_WRITES + RESUME_DESCRIPTOR_WRITES*MAX_CALLER_RESUMES
# Per-world probe receipts: release rebuild, one durable read with resume
# headroom, and (on the final prefix only) the fitting-closed receipt.
PROBE_RECEIPTS = 2


def identity_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True,
                                     separators=(',', ':')).encode()).hexdigest()


def declared_tokenizer_identity(files, *, vocab_size=SOURCE_VOCAB_SIZE):
    """Declared immutable tokenizer identity from the verified source file list."""
    by_name = {}
    for item in files:
        name = str(item['path']).rsplit('/', 1)[-1]
        by_name[name] = item['sha256']
    missing = [name for name in TOKENIZER_IDENTITY_FILES if name not in by_name]
    if missing:
        raise ValueError('source receipt lacks declared tokenizer files: '+','.join(missing))
    if type(vocab_size) is not int or vocab_size < 1:
        raise ValueError('declared tokenizer vocabulary size is invalid')
    return {'files': {name: by_name[name] for name in TOKENIZER_IDENTITY_FILES},
            'vocab_size': vocab_size, 'eos_token_id': 1, 'pad_token_id': 2,
            'is_fast': True, 'add_special_tokens_for_source': False,
            'truncation_for_source': False, 'local_files_only': True}


def tokenizer_identity_digest(identity):
    """The producer's exact digest formula for the declared tokenizer mapping."""
    return identity_digest({'format': FEATURE_CACHE_FORMAT+'.tokenizer',
                            'identity': identity})


def stage(wall, **counts):
    return {**{k: 0 for k in COUNTERS}, 'wall_seconds': wall, **counts}


def plan(arm):
    """Every v12 stage, derived from the declarations in BUDGET_DERIVATION.

    Counts are worst-case reservations, not estimates of typical work. A
    reservation is charged before dispatch and is never refunded, so a
    pessimistic bound is the safe direction and an optimistic one is a defect.
    Two per-stage facts drive the arithmetic:

    * a cache-missing canonical span costs one tokenizer pass (charged
      ``source_encodes``/``tokenizer_calls``/``source_token_bytes`` and
      ``input_tokens = 256``) and one frozen source forward (charged
      ``forwards``/``source_feature_forwards``/``cache_misses`` and
      ``input_tokens = 256``);
    * a backbone decode step is charged one ``forwards`` and
      ``input_tokens = 256`` (the fixed right-padded context width).
    """
    if arm not in ARMS: raise ValueError('undeclared density arm')
    checkpoint = CHECKPOINT_BYTES[arm]

    def span(tokenizer_passes, source_forwards, prompt_passes=0,
             backbone_forwards=0, backbone_tokens=0, extra_input_tokens=0):
        return {
            'forwards': backbone_forwards + source_forwards,
            'input_tokens': (backbone_forwards*backbone_tokens
                             + tokenizer_passes*SOURCE_MAX_BYTES
                             + source_forwards*SOURCE_MAX_TOKENS + extra_input_tokens),
            'source_encodes': tokenizer_passes,
            'tokenizer_calls': tokenizer_passes,
            'prompt_tokenizer_calls': prompt_passes,
            'source_token_bytes': tokenizer_passes*SOURCE_MAX_BYTES,
            'source_feature_forwards': source_forwards,
            'cache_misses': source_forwards,
            'cache_evictions': source_forwards,
            'cache_recomputations': source_forwards,
        }

    result = {
        # The admission stage charges its own receipt plus the audit/store/log
        # envelope for the whole owned output tree. Terminal receipts are charged
        # in their own declared stages below, never outside a stage.
        'admission': stage(30, artifact_bytes=ADMISSION_ENVELOPE_BYTES,
                           artifact_read_bytes=SOURCE_FILE_BYTES + ENVIRONMENT_CLOSURE_BYTES,
                           writes=2),
        # One hashing pass over the source, one environment-closure pass, one
        # weight load and the capacity-guard receipt written after that worker.
        'load': stage(180, model_loads=1,
                      artifact_read_bytes=(2*SOURCE_FILE_BYTES + ENVIRONMENT_CLOSURE_BYTES),
                      writes=1),
        # The supervising parent owns distinct per-window stages so each capacity
        # guard has its own path and its own pre-reserved write, and the terminal
        # result never depends on a stage another receipt already consumed.
        'capacity_main': stage(15, writes=1),
        'capacity_restart': stage(15, writes=1),
        'finalize_parent': stage(20, writes=2),
        # The child writes its terminal result inside a declared stage, so no
        # receipt is ever written with an inactive stage.
        'finalize_worker': stage(30, writes=2),
        # 8 calibration worlds x 8 reserved answer tokens, extensions disabled.
        'baseline': stage(45, **span(0, 0, prompt_passes=16, backbone_forwards=64,
                                     backbone_tokens=256),
                          generated_tokens=64, writes=8),
        'numerical': stage(30, **span(1, 1, prompt_passes=2, backbone_forwards=2,
                                      backbone_tokens=256), writes=1),
        # 32 admissions (one declared store write each) and 32 durable reads.
        # A read reserves three descriptor writes (the exact enumerated maximum),
        # and one bounded resume (two more) is budgeted as headroom.
        'prepare': stage(60, **span(32, 32), retrievals=32,
                         artifact_read_bytes=32*RELEASE_READ_BYTES_BOUND,
                         writes=32*(1 + READ_WRITE_BOUND)),
        # 32 complete frozen teacher answers (8 tokens each) + 32 target forwards.
        'teacher': stage(90, **span(0, 0, prompt_passes=128, backbone_forwards=288,
                                    backbone_tokens=256),
                         generated_tokens=256, writes=33, artifact_bytes=TEACHER_CACHE_BYTES),
        # Four checkpoint files plus the fitting-closed and restart-request receipts.
        'checkpoint': stage(60, **span(1, 1, prompt_passes=2, backbone_forwards=1,
                                       backbone_tokens=256),
                            artifact_bytes=checkpoint, writes=6),
        # 16 fresh worlds x 6 controls x 8 reserved tokens; 16 admissions/reads
        # plus the development receipt.
        'development': stage(540, **span(16, 16, prompt_passes=14*16, backbone_forwards=768,
                                        backbone_tokens=256),
                             artifact_read_bytes=16*RELEASE_READ_BYTES_BOUND,
                             generated_tokens=768, retrievals=16,
                             writes=16*(1 + READ_WRITE_BOUND) + 1),
        # Two tokenizer passes: the corrected span, then its re-read after the
        # correction evicts that record's cached tokens. One source forward.
        # Writes: correction record, re-read (3 + 2 headroom), revocation record,
        # refused post-revocation read (3 + 2 headroom), and the receipt.
        'correction': stage(60, **span(2, 1, prompt_passes=6, backbone_forwards=24,
                                       backbone_tokens=256),
                            artifact_read_bytes=2*RELEASE_READ_BYTES_BOUND,
                            generated_tokens=24, retrievals=2,
                            writes=1 + READ_WRITE_BOUND + 1 + READ_WRITE_BOUND + 1),
        # A fresh process reloads (one hashing pass + one weight load), restores
        # the extension and the teacher cache, recomputes every source span and
        # writes its own result. Prompt tokenization is derived from the real
        # call sites (64 target re-validations + canary + 2 update encodings).
        'restart': stage(RESTART_WALL_SECONDS,
                         **span(2, 2, prompt_passes=RESTART_PROMPT_PASSES,
                                backbone_forwards=2, backbone_tokens=256),
                         backwards=1, updates=1, model_loads=1,
                         artifact_read_bytes=(2*SOURCE_FILE_BYTES + ENVIRONMENT_CLOSURE_BYTES
                                              + checkpoint + TEACHER_CACHE_BYTES
                                              + RELEASE_READ_BYTES_BOUND
                                              + RELEASE_DIRECT_READ_BOUND),
                         restart_restores=2, retrievals=1,
                         writes=READ_WRITE_BOUND + 1),
    }
    for end in PREFIXES:
        updates = 128
        result[f'fit_{end:03d}'] = stage(
            500, **span(updates, updates, prompt_passes=2*updates,
                        backbone_forwards=updates, backbone_tokens=256),
            backwards=updates, updates=updates, writes=updates)
        # Per world: release rebuild (1 tokenizer + 1 forward, 4 store writes),
        # durable read (1 tokenizer for the provenance re-check after publication
        # cleared C, 3 descriptor writes + 2 resume headroom), live re-encode
        # (1 tokenizer + 1 forward) and wrong-world re-encode (1 tokenizer +
        # 1 forward); then 2 parity + 6 control + 6*8 generation forwards. Plus
        # the probe and curve receipts.
        worlds, controls, tokens = FIT_CASES, 6, 8
        backbone = worlds*(2 + controls + controls*tokens)
        result[f'fit_probes_{end:03d}'] = stage(
            900, **span(worlds*4, worlds*3, prompt_passes=20*worlds,
                        backbone_forwards=backbone, backbone_tokens=256),
            artifact_read_bytes=worlds*(RELEASE_READ_BYTES_BOUND
                                        + RELEASE_DIRECT_READ_BOUND),
            generated_tokens=worlds*controls*tokens, retrievals=worlds,
            writes=worlds*(4 + READ_WRITE_BOUND) + PROBE_RECEIPTS
                   + (1 if end == PREFIXES[-1] else 0))
    # The supervised parent writes the admission, capacity-guard and result
    # receipts, and every stage closed through close_stage() also writes one
    # observed source-cost receipt. Those writes are declared, not hidden.
    result['admission']['writes'] = 2
    for name in result:
        if name.startswith(('fit_', 'fit_probes_')) or name in (
                'baseline', 'numerical', 'prepare', 'teacher', 'checkpoint',
                'development', 'correction', 'restart'):
            result[name]['writes'] += 1
    return result


# The declared totals are computed from the same stage caps the validator
# checks, term by term, so the prose cannot drift from the envelope.
_TWO_SITE_CAPS = plan('two')
_TWO_SITE_TOTALS = totals(_TWO_SITE_CAPS)
_TWO_SITE_TOTAL_WRITES = _TWO_SITE_TOTALS['writes']
_WRITE_STAGE_TERMS = ' + '.join(f"{name}={caps['writes']}"
                                for name, caps in _TWO_SITE_CAPS.items())
_WALL_STAGE_TERMS = ' + '.join(f"{name}={caps['wall_seconds']}"
                               for name, caps in _TWO_SITE_CAPS.items())

BUDGET_DERIVATION = {
    'basis': ('every stage is re-derived from the v12 mechanism, not projected from the v11 '
              'totals; all counts are worst-case reservations and none is refunded'),
    'source_span_unit': (
        'one cache-missing canonical span reserves tokenizer_calls=1, source_encodes=1, '
        'source_token_bytes=256 and input_tokens=256 for the tokenizer, then '
        'forwards=1, source_feature_forwards=1, cache_misses/evictions/recomputations=1 '
        'and input_tokens=256 for the frozen source forward'),
    'backbone_token_unit': 'each decode/parity forward reserves forwards=1 and input_tokens=256 '
                           '(fixed right-padded context width)',
    'baseline_forwards': '8 worlds * 8 reserved tokens = 64; no source features',
    'numerical_forwards': '2 disabled/enabled forwards + 1 canonical span = 3',
    'prepare_forwards': ('32 admissions * 1 canonical span = 32; 32 retrievals; per world one '
                         'declared admission write, three descriptor writes for the read, and '
                         'two of resume headroom'),
    'teacher_forwards': '32 answers * 8 reserved tokens + 32 target forwards = 288; no source features',
    'fitting_updates': ('32 worlds * 16 passes = 512; each update reserves 1 backbone forward, '
                        '1 canonical span, 1 backward, 1 update and 1 receipt write'),
    'fit_probe_forwards': ('per prefix, 32 worlds * (2 parity + 6 control + 6*8 generated) '
                           '= 1792 backbone forwards, plus 96 source forwards; per world 4 '
                           'tokenizer passes (release rebuild, provenance re-check after '
                           'publication cleared C, live, wrong-world), 4 release writes, 3 '
                           'descriptor writes and 2 of resume headroom; plus 2 stage receipts'),
    'development_forwards': '16 worlds * 6 controls * 8 reserved tokens = 768 + 16 admissions',
    'correction_forwards': ('24 generated + 1 source forward; 2 tokenizer passes because the '
                            'correction evicts the corrected record before the re-read'),
    'restart_forwards': ('2 canary/update backbone forwards + 2 source spans recomputed from a '
                         'cleared feature cache; 1 backward, 1 update, 1 model load, 2 restores '
                         '(checkpoint and teacher cache) and their read bytes'),
    'total_forwards': ('64 + 3 + 32 + 288 + 4*256 + 4*1888 + 2 + 784 + 25 + 4 = 9778; a source '
                       'forward appears twice on purpose: once as a frozen backbone forward '
                       '(forwards) and once as source_feature_forwards, because v12 dispatches both'),
    'total_backwards': '512 fitting + 1 restart = 513',
    'total_updates': '512 fitting + 1 fresh-process continuation = 513',
    'total_tokenizer_passes': '0 + 1 + 32 + 0 + 512 + 512 + 1 + 16 + 2 + 2 = 1078',
    'restart_prompt_tokenization': (
        f'restart reserves {RESTART_PROMPT_PASSES} prompt-tokenizer passes = '
        f'{RESTART_TARGET_REVALIDATION_ENCODINGS} (2 encodings for each of {FIT_CASES} '
        f'restored teacher targets: student and exact text) + {RESTART_CANARY_ENCODINGS} '
        f'(canary encoding) + {RESTART_UPDATE_ENCODINGS} (continuation student encoding '
        'and the exact-text encoding that the alignment binding re-derives). Nothing '
        'else in the restart path tokenizes a prompt'),
    'total_prompt_tokenizer_passes': (
        f'all-stage canonical chat-template passes = {_TWO_SITE_TOTALS["prompt_tokenizer_calls"]}'),
    'total_prompt_tokenizer_passes_value': _TWO_SITE_TOTALS['prompt_tokenizer_calls'],
    'total_source_forwards': '0 + 1 + 32 + 0 + 512 + 384 + 1 + 16 + 1 + 2 = 949',
    'total_input_tokens': '16384 + 1024 + 16384 + 73728 + 393216 + 2064384 + 768 + 204800 '
                          '+ 6912 + 1536 = 2779136',
    'total_generated_tokens': '64 + 256 + 4*1536 + 768 + 24 = 7256',
    'total_retrievals': '32 + 4*32 + 16 + 2 + 1 = 179',
    'read_descriptor_writes': ('a successful wait-based read persists three pending descriptors '
                               '(initial pending, dispatch reservation, terminal); a waiting read '
                               'that times out persists two; a read whose own dispatch fails '
                               'persists three. Three is therefore the exact unconditional bound '
                               'and every stage reserves 3 per read. Each explicit bounded caller '
                               'resume persists at most two (one when it attaches to an '
                               'in-flight ticket, none when refused before dispatch), so 2 of '
                               'resume headroom is reserved per read. The read\'s own internal '
                               'resume is inside that 3 and is not charged again. There is no '
                               'automatic retry: at most four caller waits, then a typed refusal'),
    'total_writes': (f'sum of all {len(_TWO_SITE_CAPS)} two-site stage caps: '
                     f'{_WRITE_STAGE_TERMS} = {_TWO_SITE_TOTAL_WRITES} declared writes, '
                     'including descriptor persists, one observed source-cost receipt per '
                     'closed stage, and the receipts written outside a stage'),
    'total_writes_value': _TWO_SITE_TOTAL_WRITES,
    'stage_wall': (f'sum of all {len(_TWO_SITE_CAPS)} two-site stage caps: {_WALL_STAGE_TERMS} '
                   f'= {_TWO_SITE_TOTALS["wall_seconds"]} seconds inside the 7200-second '
                   'whole-process-group envelope; the remaining '
                   f'{7200-_TWO_SITE_TOTALS["wall_seconds"]} seconds are outer headroom, never '
                   'a stage allowance'),
    'stage_wall_value': _TWO_SITE_TOTALS['wall_seconds'],
    'artifact_accounting': ('194 MiB reserved = 96 MiB audit/store/receipt/log envelope + 2 MiB '
                            'teacher cache + 96 MiB two-site extension/optimizer checkpoint. '
                            'Checkpoint arithmetic: 4.22M trainable float32 parameters '
                            '(2.90M reader + 1.32M compressor) plus two AdamW moment buffers; '
                            'this is a bound, not a fit measurement'),
    'artifact_read_accounting': (
        f'{_TWO_SITE_TOTALS["artifact_read_bytes"]} declared file-read bytes: '
        f'{SOURCE_FILE_BYTES} of source hashing per process admission/worker and the same '
        'again for each weight load (2x in load and 2x in restart), '
        f'{ENVIRONMENT_CLOSURE_BYTES} of environment-closure hashing in admission, load and '
        'restart, and the checkpoint plus teacher cache at restart. Physical page-cache '
        'reuse is not measured and is not subtracted'),
    'artifact_read_bytes_value': _TWO_SITE_TOTALS['artifact_read_bytes'],
    'source_file_bytes': SOURCE_FILE_BYTES,
    'environment_closure_files': ENVIRONMENT_CLOSURE_FILES,
    'environment_closure_bytes': ENVIRONMENT_CLOSURE_BYTES,
    'audit_io': ('append-only journal: logical appended bytes equal appended bytes; physical SSD '
                 'amplification and flush cost are not measured and are not claimed to be lower '
                 'than v11. Lower audit cost is not better learning'),
    'failed_or_interrupted_work': 'reserved_before_dispatch_no_refunds',
    'deadline': ('monotonic and wall-clock outer deadline both persisted; the stricter remaining '
                 'time wins, so a monotonic-clock restart cannot extend the budget'),
    'I/O_costs': ('logical/serialized bytes, synchronized wall and dispatch wall are recorded '
                  'separately; physical I/O is unmeasured'),
}


def validate_source_feature_binding(binding):
    if not isinstance(binding, dict) or set(binding) != set(SOURCE_FEATURE_BINDING_KEYS):
        raise ValueError('declared source-feature binding is incomplete')
    identity = binding['tokenizer_identity']
    if identity != declared_tokenizer_identity(
            [{'path': name, 'sha256': digest}
             for name, digest in identity.get('files', {}).items()]):
        raise ValueError('tokenizer identity differs from the canonical file-derived mapping')
    for name in ('tokenizer_sha256', 'original_parameters_sha256', 'source_files_sha256'):
        value = binding[name]
        if (not isinstance(value, str) or len(value) != 64
                or any(character not in '0123456789abcdef' for character in value)):
            raise ValueError(f'incomplete {name} digest in the source-feature binding')
    if binding['tokenizer_sha256'] != tokenizer_identity_digest(identity):
        raise ValueError('declared tokenizer digest does not match its identity mapping')
    return binding


def validate_protocol(protocol):
    if (protocol.get('format') != FORMAT
            or protocol.get('source_kind') != 'official_bf16'
            or protocol.get('source_repo') != 'XHToken/Spark-X2.6-4B'
            or protocol.get('source_revision') != '66e3ae7bee54e0a523d692cb8ef841f35409807b'
            or protocol.get('base_dtype') != 'bfloat16'
            or protocol.get('extension_dtype') != 'float32'
            or protocol.get('optimizer') != 'NativeFusedAdamW'
            or protocol.get('placement') != 'block_input_query_to_post_block_residual'
            or protocol.get('slots') != 8
            or protocol.get('slot_policy') != 'fixed_eight_masked_padding_v5'
            or protocol.get('slot_policy_v12') != 'fixed_eight_masked_padding_v12'
            or protocol.get('persistent_P_updates') != 0
            or protocol.get('automatic_promotion') is not False
            or protocol.get('source_features') != SOURCE_FEATURES
            or protocol.get('training') != TRAINING
            or protocol.get('teacher') != TEACHER
            or protocol.get('coverage') != COVERAGE
            or protocol.get('budget_derivation') != BUDGET_DERIVATION
            or protocol.get('fit_probe_prefixes') != list(PREFIXES)
            or protocol.get('fit_curve_selection')
            != 'all_512_updates_fixed_final_prefix_all_groups_only_opens_development'):
        raise ValueError('fixed v12 source-feature, coverage or training contract differs')
    validate_source_feature_binding(protocol.get('source_feature_binding'))
    # Derived totals must equal the summed stage caps, not a hand-written list.
    derivation = protocol['budget_derivation']
    if derivation.get('total_writes_value') != totals(plan('two'))['writes']:
        raise ValueError('declared total write envelope differs from the stage caps')
    if derivation.get('total_prompt_tokenizer_passes_value') != \
            totals(plan('two'))['prompt_tokenizer_calls']:
        raise ValueError('declared total prompt-tokenization envelope differs from the caps')
    if derivation.get('stage_wall_value') != totals(plan('two'))['wall_seconds']:
        raise ValueError('declared stage wall differs from the stage caps')
    if derivation.get('artifact_read_bytes_value') != totals(plan('two'))['artifact_read_bytes']:
        raise ValueError('declared read-byte envelope differs from the stage caps')
    if (derivation.get('environment_closure_bytes') != ENVIRONMENT_CLOSURE_BYTES
            or derivation.get('environment_closure_files') != ENVIRONMENT_CLOSURE_FILES):
        raise ValueError('declared environment closure size differs from the qualified closure')
    # Restart must reserve every prompt tokenization its own call sites perform.
    restart = protocol['arms']['two']['budget']['restart']
    required = (2*FIT_CASES + 3)
    if restart['prompt_tokenizer_calls'] < required:
        raise ValueError('restart prompt-tokenization cap is below its required passes')
    if derivation.get('source_file_bytes') != SOURCE_FILE_BYTES:
        raise ValueError('declared source byte size differs from the reviewed constant')
    variant = protocol.get('experiment_variant') or {}
    if (variant.get('changed') != list(DIFFERENCES_FROM_V11)
            or variant.get('unchanged') != list(UNCHANGED_FROM_V11)):
        raise ValueError('declared v11 differences differ from the reviewed list')
    if set(protocol.get('arms', {})) != set(ARMS):
        raise ValueError('density arm inventory differs')
    for arm, spec in protocol['arms'].items():
        if spec['sites'] != ARMS[arm]: raise ValueError('reader sites differ')
        if spec['wall_seconds'] != OUTER_WALL_SECONDS:
            raise ValueError('outer deadline differs from the declared 7200 seconds')
        validate_plan(spec['budget'])
        if spec['budget'] != plan(arm): raise ValueError('v12 stage budget differs from derivation')
        allocated = spec['total_reservations']
        if allocated != totals(plan(arm)):
            raise ValueError('v12 reservation totals differ')
        if allocated['wall_seconds'] > spec['wall_seconds']-11:
            raise ValueError('stage allocation exceeds the outer envelope')
        if arm == 'two' and allocated['updates'] != 513:
            raise ValueError('the two-site packet must reserve exactly 512+1 updates')
    # Every historical source, numerical, ownership, competence, eight-world
    # fit/utility and resource requirement is still checked by the sealed v11
    # validator. Only its own inherited schedule, teacher count, prefixes,
    # derivation and per-arm envelopes are projected; no other field is replaced.
    common = deepcopy(protocol)
    common.update(format=PREVIOUS_FORMAT, training=deepcopy(PREVIOUS_TRAINING),
                  teacher=deepcopy(PREVIOUS_TEACHER), coverage=deepcopy(PREVIOUS_COVERAGE),
                  budget_derivation=deepcopy(PREVIOUS_DERIVATION),
                  fit_probe_prefixes=list(PREVIOUS_PREFIXES),
                  fit_curve_selection='all_512_updates_fixed_final_prefix_all_groups_only_opens_development')
    common.pop('source_features', None)
    common.pop('source_feature_binding', None)
    common.pop('slot_policy_v12', None)
    for arm, spec in common['arms'].items():
        spec['budget'] = previous_plan(arm)
        spec['total_reservations'] = totals(spec['budget'])
    from .intrinsic_memory_coverage_v11 import validate_protocol as validate_previous
    validate_previous(common)

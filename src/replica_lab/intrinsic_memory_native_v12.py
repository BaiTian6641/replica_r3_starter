"""Native v12 worker: frozen-source features, live compressor, fixed group gates.

Called only after external admission by the launcher. Nothing here is an
authorization, a scientific gate or a launch path. The original Spark weights
are loaded exactly as in the sealed v5/v11 lineage; what changes is the value
producer (frozen canonical-source hidden states followed by a live trainable
compressor) and the complete v12 work vocabulary.

Identity order matters and is deliberate:

1. executable code, environment, prepared data and *source files* are verified
   before any weight is touched (the launcher does this);
2. original weights are loaded, the full original parameter digest is compared
   with the declared reference, and only then is the feature contract built;
3. the feature contract binds the actual adapter structure, the tokenizer and
   the original parameter digest, and is written into the v12 checkpoint and
   teacher cache, so a restart recomputes and compares it.

No original parameter is trainable, exported or serialized. No P update, no
promotion, no 9/36 arm and no development opening can occur here unless the
sealed protocol's gates allow it.
"""
import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path

import torch

from .architecture.spark2_5 import (
    load_spark2_5_config, official_bf16_precision, load_tokenizer_identity,
    resolve_thinking_mode)
from .architecture.spark_adapter import Spark2_5Adapter
from .architecture.intrinsic_memory_layout_v4 import spark_layout
from .contextual_completion_objective_v1 import CompletionObjectiveSpec, stop_policy_identity
from .contextual_teacher_objective_v1 import TeacherObjective
from .contextual_memory import canonical_spans
from .contextual_memory_v12 import SLOT_POLICY as SLOT_POLICY_RUNTIME, ContextualMemoryV12
from .contextual_release_v12 import (ProducerReleaseCorpus,
    WorkLedgerReleaseReadAccounting)
from .contextual_source_features_v12 import (
    CODEC, FEATURE_FORMAT, FrozenSourceFeatures, PRODUCER, REPRESENTATION,
    SourceFeatureSpec, adapter_content_identity, adapter_contract_digest,
    tokenizer_contract_digest)
from .intrinsic_memory_budget_v12 import atomic, sha_json
from .intrinsic_memory_coverage_v12 import (
    DEVELOPMENT_CASES, FEATURE_CACHE_BYTES, FEATURE_CACHE_ENTRIES, FIT_CASES,
    LEDGER_METADATA_READ_BYTES, SOURCE_FILE_BYTES,
    SOURCE_HIDDEN_SIZE, SOURCE_MAX_TOKENS, TOKENIZER_IDENTITY_FILES)
from .intrinsic_memory_diagnostic_v12 import (
    PROMPT, StageRef, SourceWorkAccounting, logits_digest, right_padded_prefix)
from .intrinsic_memory_execution_v5 import bound_json, hash_file
from .intrinsic_memory_native_v5 import (
    base_hash, store_at, world_scope, verified_source_classes)
from .intrinsic_memory_teacher_v12 import TeacherEngineV12
from .intrinsic_memory_acceptance_v15 import chronological_correction, expect_invalidation
from .intrinsic_memory_correction_v18 import (
    correction_evaluation_view, publish_correction, fresh_revocation_check)
from .spark_device_placement_v2 import place_frozen_spark_base


class NativeBackend:
    def __init__(self, tokenizer, template, mode):
        self.tokenizer, self.template, self.mode = tokenizer, template, mode

    def encode(self, content):
        text = self.tokenizer.apply_chat_template(
            [{'role': 'user', 'content': content}], chat_template=self.template,
            tokenize=False, add_generation_prompt=True, **self.mode)
        return self.tokenizer(text, add_special_tokens=False)['input_ids']

    def answer_ids(self, text):
        return self.tokenizer(text, add_special_tokens=False)['input_ids']

    def decode(self, ids):
        return self.tokenizer.decode(ids, skip_special_tokens=False,
                                     clean_up_tokenization_spaces=False)


class BoundSourceTokenizer:
    """Immutable identity wrapper derived from the *loaded* tokenizer.

    The frozen-source provider refuses a tokenizer without an explicit
    ``identity()``. That identity is now computed from the tokenizer actually
    returned by ``from_pretrained`` and from the bytes of the files in the
    verified source directory: the six declared tokenizer files are re-hashed,
    the vocabulary size and special-token ids are read from the live object, and
    the wrapping directory is compared with the tokenizer's own ``name_or_path``.
    It is then compared with the identity declared in the reviewed protocol, so
    a substituted tokenizer is refused *before* any source work rather than
    being compared with its own declaration.

    The independent full-source file and parameter hashes in admission remain
    the primary authority check; this is the runtime binding of the object that
    is actually used.
    """

    def __init__(self, tokenizer, directory, declared):
        if not callable(getattr(tokenizer, '__call__', None)):
            raise ValueError('callable tokenizer required')
        if not isinstance(declared, dict) or not declared:
            raise ValueError('declared tokenizer identity required')
        observed = observed_tokenizer_identity(tokenizer, directory)
        if observed != declared:
            raise ValueError('declared tokenizer identity differs from the loaded tokenizer')
        self._tokenizer, self._identity = tokenizer, observed

    def identity(self):
        return json.loads(json.dumps(self._identity))

    def __call__(self, *args, **kwargs):
        return self._tokenizer(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._tokenizer, name)


def observed_tokenizer_identity(tokenizer, directory):
    """Identity of the loaded tokenizer, derived from bytes and live properties."""
    directory = Path(directory)
    files = {}
    for name in TOKENIZER_IDENTITY_FILES:
        path = directory/name
        if path.is_symlink() or not path.is_file():
            raise ValueError('tokenizer file missing or substituted: '+name)
        files[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    origin = getattr(tokenizer, 'name_or_path', None)
    if not isinstance(origin, str) or Path(origin).resolve() != directory.resolve():
        raise ValueError('loaded tokenizer did not come from the verified source directory')
    vocab_size = getattr(tokenizer, 'vocab_size', None)
    if type(vocab_size) is not int or vocab_size < 1:
        raise ValueError('loaded tokenizer exposes no vocabulary size')
    vocabulary = tokenizer.get_vocab()
    if not isinstance(vocabulary, dict) or len(vocabulary) < vocab_size:
        raise ValueError('loaded tokenizer vocabulary is smaller than its declared size')
    identity = {'files': files, 'vocab_size': vocab_size,
                'eos_token_id': tokenizer.eos_token_id,
                'pad_token_id': tokenizer.pad_token_id,
                'is_fast': bool(tokenizer.is_fast),
                'add_special_tokens_for_source': False,
                'truncation_for_source': False, 'local_files_only': True}
    if (type(identity['eos_token_id']) is not int
            or type(identity['pad_token_id']) is not int):
        raise ValueError('loaded tokenizer special-token ids are not integers')
    return identity


METADATA_FILES = ('config.json', 'chat_template.jinja', 'tokenizer.json',
                  'tokenizer_config.json', 'special_tokens_map.json', 'vocab.json',
                  'merges.txt', 'generation_config.json', 'model.safetensors.index.json',
                  'configuration_spark.py', 'modeling_spark.py')


def reserved_read(ledger, stage, operation, path, *, bound=None):
    """Reserve before reading, then record the bytes actually observed.

    Every metadata and pin read goes through here so a read can never happen
    outside a reservation, and the charged amount is the length of the data that
    was returned rather than a stat size that may differ from what was read.
    """
    size = path.stat().st_size if bound is None else bound
    with ledger.work(stage, operation, artifact_read_bytes=max(1, size)) as event:
        data = path.read_bytes()
        event['actual'] = {'artifact_read_bytes': len(data)}
    return data


def load_native(root, protocol, arm, ledger, stage):
    """Load the verified official BF16 source, then build the v12 producer."""
    from transformers import AutoTokenizer
    if not torch.backends.mps.is_available(): raise ValueError('native MPS unavailable; no fallback')
    # Pin reads are reserved and observed like any other read.
    source = json.loads(reserved_read(ledger, stage, 'source_binding_read',
                                      root / protocol['source']['path']))
    directory = root/source['source_root']
    # Configuration, template and tokenizer reads happen before the model load;
    # reserve each file, then record the bytes actually read from it.
    blobs = {name: reserved_read(ledger, stage, 'source_metadata_reads',
                                 directory / name)
             for name in METADATA_FILES if (directory / name).is_file()}
    raw = json.loads(blobs['config.json']); config = load_spark2_5_config(raw)
    template = blobs['chat_template.jinja'].decode()
    tokenizer_identity = load_tokenizer_identity(
        config, json.loads(blobs['tokenizer_config.json']), template,
        tokenizer_json=blobs['tokenizer.json'].decode())
    execution = json.loads(reserved_read(ledger, stage, 'source_execution_read',
                                         root / protocol['source_execution']['path']))
    classes = verified_source_classes(directory, raw, execution)
    tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True,
                                             trust_remote_code=True)
    if not tokenizer.is_fast or (tokenizer.eos_token_id, tokenizer.pad_token_id) != (1, 2):
        raise ValueError('bound fast tokenizer/EOS/PAD differs')
    declared = protocol['source_feature_binding']['tokenizer_identity']
    bound_tokenizer = BoundSourceTokenizer(tokenizer, directory, declared)
    tokenizer_digest = tokenizer_contract_digest(bound_tokenizer)
    if tokenizer_digest != protocol['source_feature_binding']['tokenizer_sha256']:
        raise ValueError('declared tokenizer contract differs from the loaded tokenizer')
    # The declared mechanism identity must equal the code that is about to run.
    features = protocol['source_features']
    if (protocol['slot_policy_v12'] != SLOT_POLICY_RUNTIME
            or features['producer'] != PRODUCER
            or features['representation'] != REPRESENTATION
            or features['codec'] != CODEC
            or features['feature_contract_format'] != FEATURE_FORMAT
            or config.geometry.hidden_size != SOURCE_HIDDEN_SIZE):
        raise ValueError('declared mechanism identity differs from the loaded implementation')
    gen = protocol['generation']
    tokenizer_hash = hashlib.sha256(reserved_read(
        ledger, stage, 'source_tokenizer_hash', directory/'tokenizer.json')).hexdigest()
    template_hash = hashlib.sha256(reserved_read(
        ledger, stage, 'source_template_hash', directory/'chat_template.jinja')).hexdigest()
    spec = CompletionObjectiveSpec(config.geometry.vocab_size, gen['context_tokens'], 1, 2, (1,),
        stop_policy_identity((1,)), tokenizer_hash,
        sha_json({'chat_template_sha256': template_hash,
                  'user_prompt': PROMPT,
                  'thinking_mode': resolve_thinking_mode(tokenizer_identity, 'none'),
                  'add_generation_prompt': True}), 'none')
    stage_ref = StageRef()
    weight_bytes = sum(item['bytes'] for item in source['files']
                       if item['path'].endswith('.safetensors'))
    with ledger.work(stage, 'official_bf16_model_load', model_loads=1,
                     artifact_read_bytes=SOURCE_FILE_BYTES) as event:
        base = classes['AutoModelForCausalLM'].from_pretrained(directory,
            config=classes['AutoConfig'].from_pretrained(directory, local_files_only=True),
            local_files_only=True, torch_dtype=torch.bfloat16, attn_implementation='eager',
            use_safetensors=True)
        if any(p.dtype != torch.bfloat16 for p in base.parameters()):
            raise ValueError('original BF16 parameter dtype differs')
        if sum(p.numel() for p in base.parameters()) != 4112079360:
            raise ValueError('official parameter inventory differs')
        placement = place_frozen_spark_base(base, device='mps'); base.eval()
        precision = official_bf16_precision(config, artifact_sha256=sha_json(source['files']))
        adapter = Spark2_5Adapter(base, config, precision, device='mps',
            require_official_geometry=True, projector_in_dim=config.geometry.hidden_size,
            zero_init_output=True, memory_size=256, heads=4, spine_sites=(3, 7), spine_slots=8)
        adapter.eval(); adapter.seal_original_identity()
        # Full original identity is established before any source-feature work.
        initial_hash = base_hash(base)
        declared_parameters = protocol['source_feature_binding']['original_parameters_sha256']
        if initial_hash != declared_parameters:
            raise ValueError('original parameter digest differs from the declared source')
        # The repaired v12 source contract is content-bound: the adapter must
        # carry the exact base content digest before the feature contract is
        # built. Supplying the digest we just verified is both stricter and
        # cheaper than letting the provider hash 4.1B parameters again, and it
        # is the value the checkpoint, teacher cache and restart will bind.
        adapter.v12_adapter_content_sha256 = initial_hash
        feature_spec = SourceFeatureSpec(
            original_parameters_sha256=initial_hash,
            source_execution_sha256=adapter_contract_digest(adapter),
            tokenizer_sha256=tokenizer_digest,
            hidden_size=config.geometry.hidden_size,
            max_tokens=SOURCE_MAX_TOKENS, cache_bytes=FEATURE_CACHE_BYTES,
            cache_entries=FEATURE_CACHE_ENTRIES)
        provider = FrozenSourceFeatures(adapter, bound_tokenizer, feature_spec,
            accounting=SourceWorkAccounting(ledger, stage_ref))
        path = ContextualMemoryV12(adapter, spark_layout(
            arm, placement=protocol.get('intrinsic_placement',
                                        'block_input_query_to_post_block_residual')),
            source_identity=sha_json(source['files']),
            feature_provider=provider, lr=protocol['training']['lr'], initialization_seed=17,
            key_mode=protocol.get('intrinsic_reader_key_mode', 'ordered'),
            objective_identity=spec.identity(),
            execution_identity=sha_json({'protocol': protocol, 'arm': arm}))
        path.objective_identity = TeacherObjective().identity(spec)
        adapter.extension.spine = None
        event.update(placement=placement, precision=adapter.base_snapshot(),
                     frozen_parameter_sha256=initial_hash, original_dtype='bfloat16',
                     extension_dtype='float32', reader_key_mode=path.key_mode,
                     feature_contract=provider.spec.identity(),
                     adapter_contract=provider.adapter_contract,
                     adapter_content_sha256=adapter_content_identity(adapter),
                     tokenizer_contract=provider.tokenizer_contract,
                     feature_contract_sha256=sha_json(provider.spec.identity()))
        # Actual, not refunded: one load and the weight bytes actually read.
        event['actual'] = {'model_loads': 1, 'artifact_read_bytes': weight_bytes}
    backend = NativeBackend(bound_tokenizer, template,
                            resolve_thinking_mode(tokenizer_identity, 'none'))
    return path, backend, spec, initial_hash, stage_ref


def begin_stage(engine, name):
    engine.ledger.enter(name); engine.stage = name


def close_stage(engine, name, outcome):
    engine.write_source_costs(name)
    engine.ledger.leave(name, status=outcome)


def write_receipt(output, name, payload):
    atomic(output/(name+'.json'), payload)


def current_feature_binding(path):
    """The immutable v12 feature identity as recomputed in this process."""
    provider = path.feature_provider
    return {'feature_contract_sha256': sha_json(provider.spec.identity()),
            'adapter_contract': provider.adapter_contract,
            'adapter_content_sha256': adapter_content_identity(provider.adapter),
            'tokenizer_contract': provider.tokenizer_contract,
            'tokenizer_sha256': provider.spec.tokenizer_sha256,
            'original_parameters_sha256': provider.spec.original_parameters_sha256}


def admit_rows(engine, rows, output):
    """One canonical span per world: reserve, admit, then read back durably."""
    batches, scopes, stores = [], [], []
    for row in rows:
        store = store_at(output, row['world_id']); scope = world_scope(row)
        span = canonical_spans(row['text'], record_id=row['world_id'], source_id=row['world_id'],
                               source_epoch=1, available_at=1)
        if len(span) != 1:
            raise ValueError('one bounded canonical span required by this fixed experiment')
        with engine.measured('canonical_admission', writes=1):
            result = engine.path.admit(store, span[0], scope=scope)
            if not result.outcome.ok: raise ValueError('canonical admission failed')
        batch = engine.read(store, scope, row['question'])
        batches.append(batch); scopes.append(scope); stores.append(store)
    return batches, scopes, stores


def finalize_receipt(engine, name, payload):
    """Write a terminal receipt inside a declared stage, never outside one.

    A still-active stage is closed as failed first; then the declared
    ``finalize_worker`` stage charges the receipt. If the ledger refuses even
    that (for example an interrupted reservation that only operator recovery may
    close), the receipt is still written so evidence is not lost, and the receipt
    records that it was uncharged rather than pretending otherwise.
    """
    try:
        for stage, state in engine.ledger.snapshot()['stages'].items():
            if state['status'] == 'active':
                engine.ledger.leave(stage, status='failed')
        if 'finalize_worker' not in engine.ledger.snapshot()['stages']:
            begin_stage(engine, 'finalize_worker')
            engine.write_receipt(name, payload)
            close_stage(engine, 'finalize_worker', 'completed')
            return True
        reason = 'finalize_worker_already_consumed'
    except Exception as error:      # never swallow KeyboardInterrupt/SystemExit
        reason = type(error).__name__
    payload['final_receipt_charging'] = {'charged': False, 'reason': reason}
    try:
        write_receipt(engine.output, name, payload)
    except Exception:
        pass        # a broken fallback must not mask the error in flight
    return False


def run_correction_probe(engine, row, replacement, store, scope, old):
    """Run the controller correction stage with real generation and source checks.

    The chronological row is retained in immutable history; both controls use
    one explicitly bound current view. Revocation is checked cached and fresh.
    """
    corrected_row = chronological_correction(row, replacement)
    span, publication = publish_correction(engine, store, old, corrected_row, scope)
    stale = expect_invalidation(
        'stale_snapshot', lambda: engine.path.validate(old, scope))
    new = engine.read(store, scope, row['question'])
    evaluation_view = correction_evaluation_view(corrected_row, span, publication)
    if [item.record_id for item in new.spans] != [span.record_id]:
        raise ValueError('correction evaluation selected a different published record')
    corrected = engine.generate(evaluation_view, new, scope); corrected.pop('first_logits')
    # The exact-text path disables memory in _forward, but validates the same
    # authenticated batch and canonical text as the contextual control.
    exact = engine.generate(evaluation_view, new, scope, control='exact_text')
    exact.pop('first_logits')
    with engine.measured('revocation', writes=1):
        store.revoke_record(span.record_id, reason='predeclared_probe', scope=scope)
    revoked_read = expect_invalidation(
        'stale_snapshot', lambda: engine.path.validate(new, scope))
    with engine.measured('fresh_revocation_retrieval', retrievals=1,
                         artifact_read_bytes=1024 * 1024) as event:
        fresh_revoked = fresh_revocation_check(store, row['question'], scope, [span.record_id])
        event['actual'] = {'artifact_read_bytes': fresh_revoked['io_bytes']}
    revoked = engine.generate({**row, 'text': '', 'answer': 'unknown'}, None, scope,
                              control='absent')
    revoked.pop('first_logits')
    return {
        'stale_snapshot_rejected': True,
        'stale_snapshot_evidence': vars(stale),
        'revoked_retrieval_rejected': fresh_revoked['revoked_ids_excluded'],
        'fresh_revocation_retrieval': fresh_revoked,
        'correction_publication': publication,
        'evaluation_view': evaluation_view,
        'revoked_read_evidence': vars(revoked_read),
        'corrected_contextual': corrected, 'corrected_exact': exact,
        'correction_oracle': corrected_row['correction'],
        'revision_history': corrected_row.get('revision_history'),
        'after_revocation': revoked, 'learned_unlearning_claim': False}


def run_main(root, protocol, arm, ledger, output, path, backend, spec, initial_hash, stage_ref):
    engine = TeacherEngineV12(path, backend, spec, ledger,
                              max_new_tokens=protocol['generation']['max_new_tokens'],
                              output=output, stage_ref=stage_ref,
                              base_identity=initial_hash,
                              fit_identity=protocol['data']['fit']['sha256'])
    result = {'format': 'replica.intrinsic_memory.native_result.v12',
              'source_kind': 'official_bf16', 'arm': arm,
              'mechanism': 'frozen_canonical_source_features_live_contextual_compression_v12',
              'reader_key_mode': path.key_mode,
              'competence_passed': False, 'teacher_competence_passed': False,
              'fit_memory_dependence_passed': False, 'useful_recall_passed': False,
              'persistent_P_updates': 0, 'promotion': False, 'restart_requested': False,
              'fit_cases': FIT_CASES, 'development_cases': DEVELOPMENT_CASES,
              'group_criteria': 'unchanged_eight_world_gates_required_for_every_fixed_group'}
    stores = []
    try:
        begin_stage(engine, 'baseline')
        calibration = json.loads(reserved_read(
            ledger, 'baseline', 'calibration_split_read',
            root / protocol['data']['calibration']['path']))['rows']
        answers = []
        if len(calibration) != 8: raise ValueError('incomplete calibration')
        for row in calibration:
            answer = engine.generate(row, None, world_scope(row), control='exact_text')
            answer.pop('first_logits')
            answers.append({'world_id': row['world_id'], **answer})
            engine.write_receipt('baseline', {'rows': answers,
                                              'complete': len(answers) == 8})
        passed = len(answers) == 8 and all(
            r['correct'] and not r['false_recall'] and r['terminated'] and not r['truncated']
            for r in answers)
        result['competence_passed'] = passed
        close_stage(engine, 'baseline', 'passed' if passed else 'failed_competence')
        if not passed:
            result['status'] = 'baseline_failed_dependent_stages_closed'; return result
        # Fitting labels are parsed only after the frozen pre-fit baseline gate.
        begin_stage(engine, 'numerical')
        rows = json.loads(reserved_read(
            ledger, 'numerical', 'fit_split_read',
            root / protocol['data']['fit']['path']))['rows']
        if len(rows) != FIT_CASES or any(
                rows[i]['question'] != rows[i ^ 1]['question']
                or rows[i]['answer'] == rows[i ^ 1]['answer'] for i in range(FIT_CASES)):
            raise ValueError('counterfactual fitting corpus differs')
        ids, mask = right_padded_prefix(engine.tokens(rows[0]), spec, engine.device)
        spans = canonical_spans(rows[0]['text'], record_id='numerical', source_id='calibration',
                                source_epoch=1, available_at=1)
        b = path.training_values(spans, scope=world_scope(rows[0]))
        with torch.no_grad():
            disabled = engine.forward(ids, mask, None, world_scope(rows[0]))
            enabled = engine.forward(ids, mask, b, world_scope(rows[0]))
        if not torch.equal(disabled, enabled):
            raise ValueError('zero-init disabled-extension parity failed')
        del disabled, enabled, b
        engine.write_receipt('numerical', {
            'disabled_zero_init_exact': True,
            'single_optimizer_owner': path.assert_optimizer_ownership(),
            'feature_contract': path.feature_provider.spec.identity()})
        close_stage(engine, 'numerical', 'passed')
        begin_stage(engine, 'prepare')
        sources, scopes, stores = admit_rows(engine, rows, output)
        close_stage(engine, 'prepare', 'completed')
        begin_stage(engine, 'teacher')
        result['teacher_competence_passed'] = engine.prepare_teacher(rows, scopes)
        if not result['teacher_competence_passed']:
            close_stage(engine, 'teacher', 'failed_teacher_competence')
            result['status'] = 'teacher_failed_dependent_stages_closed'
            return result
        teacher_cache = engine.save_teacher_cache(output/'teacher_targets.json')
        result['teacher_cache'] = teacher_cache
        result['teacher_parameter_updates'] = 0
        result['teacher_feature_contract'] = engine.feature_identity()
        close_stage(engine, 'teacher', 'passed')
        updates = []; curve = []
        for end in protocol['fit_probe_prefixes']:
            fit_stage = f'fit_{end:03d}'; probe_stage = f'fit_probes_{end:03d}'
            begin_stage(engine, fit_stage)
            for index in range(end-128, end):
                i = index % FIT_CASES
                updates.append(engine.update(rows[i], sources[i], scopes[i]))
                engine.write_receipt('fit', {'updates': updates, 'planned_updates': 512})
                if index == 1 and (updates[-1]['gradients']['producer']['norm'] <= 0
                        or any(v['output_norm'] <= 0
                               for v in updates[-1]['gradients']['sites'].values())):
                    raise ValueError('producer/early reader gradient path failed after zero-init update')
            close_stage(engine, fit_stage, 'completed'); result['fit_updates'] = end
            begin_stage(engine, probe_stage)
            probes = engine.fitting_probes(rows, sources, scopes, stores)
            engine.write_receipt(probe_stage, probes)
            curve.append({'updates': end, 'correct': probes['correct'],
                          'memory_dependence_passed': probes['fit_memory_dependence_passed'],
                          'counterfactual_pairs_both_correct': probes['counterfactual_pairs_both_correct'],
                          'groups': probes['groups']})
            engine.write_receipt('fit_curve', {'prefixes': curve,
                          'selection': 'fixed_final_512_all_groups_only',
                          'early_stopping': False, 'development_opened': False})
            result['fit_memory_dependence_passed'] = probes['fit_memory_dependence_passed']
            if end == protocol['fit_probe_prefixes'][-1]:
                # The final probe stage stays active through the freeze and the
                # fitting-closed receipt. Writing that receipt after the stage had
                # closed made every real launch fail at the ledger (review X14).
                engine.freeze()
                engine.write_receipt('fitting_closed', {
                    'closed': True, 'updates': 512, 'objective': path.objective_identity,
                    'feature_contract': engine.feature_identity(),
                    'parameter_sha256': {k: logits_digest(v)
                                         for k, v in path._tensors().items()}})
                result['fitting_closed_within_stage'] = probe_stage
            close_stage(engine, probe_stage,
                        'passed' if result['fit_memory_dependence_passed']
                        else 'failed_memory_dependence')
        result['fit_curve'] = curve
        begin_stage(engine, 'checkpoint')
        with torch.no_grad():
            live = engine.live(sources[0], scopes[0]); enc = engine.encoding(rows[0])
            canary = engine.forward(enc.input_ids, enc.attention_mask, live, scopes[0])
        expected = logits_digest(canary)
        cap = protocol['arms'][arm]['budget']['checkpoint']['artifact_bytes']
        with engine.measured('extension_optimizer_checkpoint', artifact_bytes=cap,
                             writes=4) as event:
            manifest = path.save(output/'checkpoint')
            actual = sum(p.stat().st_size for p in (output/'checkpoint').rglob('*') if p.is_file())
            event['actual'] = {'artifact_bytes': actual}
            event['contains_original_weights'] = False
            event['feature_contract_bound'] = True
        engine.write_receipt('restart_request', {
            'canary_logits_sha256': expected, 'world_id': rows[0]['world_id'],
            'parent_model_pid': os.getpid(), 'objective': path.objective_identity,
            'checkpoint_manifest_sha256': hash_file(output/'checkpoint/manifest.json'),
            'original_parameters_sha256': initial_hash, 'teacher_cache': teacher_cache,
            'feature_contract': path.feature_provider.spec.identity(),
            'feature_contract_sha256': sha_json(path.feature_provider.spec.identity()),
            'feature_binding': current_feature_binding(path),
            'updates_allowed': 1, 'uninterrupted_equivalence_claim': False,
            'restart_source_recomputation_required': True})
        result['restart_requested'] = True
        close_stage(engine, 'checkpoint', 'completed')
        if result['fit_memory_dependence_passed']:
            begin_stage(engine, 'development')
            development = json.loads(reserved_read(
                ledger, 'development', 'development_split_read',
                root / protocol['data']['development']['path']))['rows']
            batches, dscopes, dstores = admit_rows(engine, development, output)
            stores.extend(dstores)
            evaluation = engine.evaluate(development, batches, dscopes)
            engine.write_receipt('development', evaluation)
            result['useful_recall_passed'] = evaluation['useful_recall_passed']
            close_stage(engine, 'development',
                        'passed' if result['useful_recall_passed'] else 'failed_utility')
            begin_stage(engine, 'correction')
            row = development[0]; store = dstores[0]; scope = dscopes[0]; old = batches[0]
            alternative = development[1]
            replacement = (alternative['facts']['answer']
                           if isinstance(alternative.get('facts'), dict)
                           else alternative['answer'])
            engine.write_receipt('correction', run_correction_probe(
                engine, row, replacement, store, scope, old))
            close_stage(engine, 'correction', 'completed')
        result['status'] = 'completed_frozen_fit_diagnostic'
        return result
    finally:
        result['path_costs'] = dict(path.cost)
        result['source_feature_costs'] = path.feature_provider.cost
        result['feature_contract'] = path.feature_provider.spec.identity()
        result['teacher_evicted'] = {'targets': 0 if engine.fitting_closed else len(engine.targets)}
        result['final_original_parameters_sha256'] = base_hash(path.adapter.base)
        result['original_weights_unchanged'] = (
            result['final_original_parameters_sha256'] == initial_hash)
        result['persistent_P_updates'] = 0
        finalize_receipt(engine, 'main_result', result)
        for store in stores: store.close()
        path.close()
        if not result['original_weights_unchanged']:
            raise ValueError('original parameter bytes changed')


def run_restart(root, protocol, arm, ledger, output, path, backend, spec, initial_hash, stage_ref):
    with ledger.work('restart', 'restart_identity_reads',
                     artifact_read_bytes=LEDGER_METADATA_READ_BYTES) as identity:
        request = json.loads((output/'restart_request.json').read_text())
        manifest_sha256 = hash_file(output/'checkpoint/manifest.json')
        identity['actual'] = {'artifact_read_bytes': (
            (output/'restart_request.json').stat().st_size
            + (output/'checkpoint/manifest.json').stat().st_size)}
    if request.get('parent_model_pid') == os.getpid() or type(request.get('parent_model_pid')) is not int:
        raise ValueError('restart must use a different model process')
    if (request['objective'] != path.objective_identity
            or manifest_sha256 != request['checkpoint_manifest_sha256']):
        raise ValueError('restart objective/checkpoint differs')
    if initial_hash != request['original_parameters_sha256']:
        raise ValueError('restart original base differs')
    # The recomputed feature contract must equal the one the checkpoint bound.
    current = current_feature_binding(path)
    if (current['feature_contract_sha256']
            != request['feature_binding']['feature_contract_sha256']
            or current != request['feature_binding']
            or sha_json(path.feature_provider.spec.identity())
            != request['feature_contract_sha256']):
        raise ValueError('fresh-process feature contract differs')
    with ledger.work('restart', 'extension_optimizer_restore', restart_restores=1,
                     artifact_read_bytes=protocol['arms'][arm]['budget']['checkpoint'][
                         'artifact_bytes']) as event:
        event['artifact_file_bytes'] = sum(
            p.stat().st_size for p in (output/'checkpoint').rglob('*') if p.is_file())
        path.load(output/'checkpoint'); path.assert_optimizer_ownership()
        event['checksum_and_deserialization_io_included_in_wall'] = True
        event['feature_contract_bound'] = True
    engine = TeacherEngineV12(path, backend, spec, ledger, max_new_tokens=8, output=output,
                              stage_ref=stage_ref, base_identity=initial_hash,
                              fit_identity=protocol['data']['fit']['sha256'])
    engine.stage = 'restart'
    fit_rows = bound_json(root, protocol['data']['fit'])['rows']
    row = fit_rows[0]; scope = world_scope(row); store = store_at(output, row['world_id'])
    try:
        engine.load_teacher_cache(output/'teacher_targets.json', request['teacher_cache'],
                                  fit_rows)
        path.release_corpus = ProducerReleaseCorpus(
            store, scope,
            read_accounting=WorkLedgerReleaseReadAccounting(ledger, 'restart'))
        source = engine.read(store, scope, row['question']); enc = engine.encoding(row)
        with torch.no_grad():
            canary = engine.forward(enc.input_ids, enc.attention_mask,
                                    engine.live(source, scope), scope)
        equal = logits_digest(canary) == request['canary_logits_sha256']
        if not equal: raise ValueError('fresh-process logits differ')
        path.resume_training(); update = engine.update(row, source, scope)
        unchanged = base_hash(path.adapter.base) == initial_hash
        if not unchanged: raise ValueError('restart modified original weights')
        result = {'fresh_process_logits_exact': equal, 'actual_continuation_updates': 1,
                  'update': update, 'parent_model_pid': request['parent_model_pid'],
                  'continuation_model_pid': os.getpid(),
                  'original_weights_unchanged': unchanged,
                  'optimizer_ownership': path.assert_optimizer_ownership(),
                  'uninterrupted_update_equivalence_measured': False,
                  'path_costs': path.cost, 'corpus_costs': path.release_corpus.cost,
                  'source_feature_costs': path.feature_provider.cost,
                  'feature_contract': path.feature_provider.spec.identity(),
                  'feature_contract_recomputed_matches_request': True,
                  'teacher_cache_restored': True, 'teacher_parameter_updates': 0,
                  'persistent_P_updates': 0, 'objective': path.objective_identity}
        engine.write_receipt('restart_result', result)
        engine.write_source_costs('restart')
        ledger.leave('restart', status='completed')
        return result
    except BaseException as error:
        # Any exit records a typed terminal receipt and then re-raises, so
        # KeyboardInterrupt/SystemExit propagate and the original error wins.
        # The restart stage is still declared and active here, so the failure
        # receipt is charged before it is written.
        payload = {'type': type(error).__name__, 'message': str(error),
                   'reserved_work_not_refunded': True,
                   'parent_model_pid': request['parent_model_pid'],
                   'continuation_model_pid': os.getpid()}
        try:
            engine.write_receipt('restart_failure', payload)
        except Exception as ledger_error:   # evidence is never lost, original wins
            payload['receipt_charging'] = {'charged': False,
                                           'error': type(ledger_error).__name__}
            write_receipt(output, 'restart_failure', payload)
        raise
    finally:
        store.close(); path.close()


def load_and_run(root, protocol, arm, ledger, output, *, stage, mode):
    """Load once, then run the main or restart workflow in this process."""
    if mode not in ('main', 'restart'): raise ValueError('undeclared worker mode')
    if stage != ('load' if mode == 'main' else 'restart'):
        raise ValueError('worker mode and ledger stage differ')
    path, backend, spec, initial_hash, stage_ref = load_native(root, protocol, arm, ledger, stage)
    if mode == 'main':
        # The parent owns the 'load' stage; the child closes it only after the
        # original identity and the feature contract are established.
        ledger.leave('load', status='completed')
        return run_main(root, protocol, arm, ledger, output, path, backend, spec,
                        initial_hash, stage_ref)
    return run_restart(root, protocol, arm, ledger, output, path, backend, spec,
                       initial_hash, stage_ref)

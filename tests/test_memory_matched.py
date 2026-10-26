"""Matched-episode localisation contracts on tiny CPU adapters.

These tests are self-contained: a random tiny native adapter exercises the real
writer, shared-state read, save/reload and frozen-answer plumbing on CPU.  They
establish episode identity, rank/margin/unknown/correction scoring, first
failing-stage selection, row-inventory rejection and the information boundary.
They do not measure pretrained MiniCPM quality and never run native MPS.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest
import torch

from architecture_helpers import tiny_adapter
from conftest import require_local_artifacts
from replica_lab.memory_matched import (
    CODEBOOK_FORMAT,
    COMPLEXITIES,
    DEFAULT_LIMITS,
    FORBIDDEN_INPUT_MARKERS,
    LABELS_FORMAT,
    PROTOCOL_FORMAT,
    admit_matched_prepared,
    derive_matched_labels,
    derive_state_labels,
    enforce_elapsed_deadline,
    episode_states,
    finalize_completed_report,
    generate_matched_episodes,
    matched_inventory,
    matched_workload,
    prepare_matched,
    run_matched_diagnostic,
    score_matched,
    select_first_failing_stage,
    state_observation_sha256,
    validate_matched_episodes,
    verify_matched_prepared,
    writer_inputs,
)
from replica_lab.runtime.fact_memory import fact_source_text
from test_memory_diagnosis import SPEC, Tokenizer

ROOT = Path(__file__).resolve().parents[1]
SMALL = ('correction', 'interference')
FAMILY_PLAN = {'train': 1, 'dev': 1, 'heldout': 1}


def _protocol(**overrides):
    value = {
        'format': PROTOCOL_FORMAT, 'seed': 20261026, 'layer': 0,
        'families_per_split': FAMILY_PLAN, 'complexities': list(SMALL),
        'variants': 2, 'thresholds': {}, 'limits': {},
        'generation': {'max_new_tokens': 6, 'max_prompt_tokens': 256},
        'budget': {'max_wall_seconds': 900, 'optimizer_updates': 0,
                   'native_training_jobs': 0},
        'zero_updates': True,
    }
    value.update(overrides)
    return value


def _stub_answerer(ledger):
    def answer(*, prompt_ids, prompt_mask, query, scope, fast_state):
        return 'unknown', []
    return answer


def _tiny_run(tmp_path, *, complexities=SMALL, answerer_factory=_stub_answerer):
    model = tiny_adapter(fast_memory=SPEC).eval()
    episodes = generate_matched_episodes(
        families_per_split=FAMILY_PLAN, complexities=complexities)
    protocol = _protocol(complexities=list(complexities))
    report = run_matched_diagnostic(
        model, Tokenizer(), protocol, output_dir=tmp_path / 'run',
        episodes=episodes, answerer_factory=answerer_factory)
    return model, episodes, protocol, report, tmp_path / 'run'


def _rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def _sha_json(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True).encode('utf-8')).hexdigest()


def test_episodes_are_deterministic_split_separated_and_label_free():
    first = generate_matched_episodes()
    assert first == generate_matched_episodes()
    assert first != generate_matched_episodes(seed=20261027)
    audit = validate_matched_episodes(first)
    assert audit['episodes'] == 30 and audit['families'] == 15
    assert audit['states'] == 42 and audit['queries'] == 108
    assert audit['query_forms'] == [0, 1, 2, 3]
    assert audit['heldout_values_absent_from_prior_splits'] is True

    for episode in first:
        payload = writer_inputs(episode)
        text = json.dumps(payload, sort_keys=True).casefold()
        for marker in FORBIDDEN_INPUT_MARKERS:
            assert marker not in text
        assert set(payload) == {'episode_id', 'world_id', 'facts',
                                'distractors', 'corrections', 'observations',
                                'queries'}
        labels = derive_state_labels(episode, episode_states(episode)[0])
        assert 'expected' in labels['targets']['q0']
        # The writer input is unchanged by deriving evaluation labels.
        assert json.dumps(writer_inputs(episode), sort_keys=True) == \
            json.dumps(payload, sort_keys=True)
        for item in episode['distractors']:
            assert item['origin_world_id'] != episode['world_id']
            origin = next(other for other in first
                          if other['world_id'] == item['origin_world_id'])
            assert origin['split'] == episode['split']

    # Query text is rendered from the observed address only, never from a value.
    changed = deepcopy(first[0])
    changed['facts'][0]['value'] = 'val-never-observed-elsewhere'
    assert [query['text'] for query in changed['queries']] == \
        [query['text'] for query in first[0]['queries']]

    with pytest.raises(ValueError):
        fact_source_text({'entity': 'e', 'attribute': 'a', 'value': 'v',
                          'expected_answer': 'leak'})


def test_episode_corruptions_are_rejected():
    episodes = generate_matched_episodes(complexities=SMALL)
    with pytest.raises(ValueError, match='unique|duplicate'):
        validate_matched_episodes(episodes + [deepcopy(episodes[0])],
                                  families_per_split=FAMILY_PLAN,
                                  complexities=SMALL)

    duplicate = deepcopy(episodes)
    duplicate[2]['facts'][0]['entity'] = duplicate[0]['facts'][0]['entity']
    duplicate[2]['facts'][0]['attribute'] = duplicate[0]['facts'][0]['attribute']
    with pytest.raises(ValueError, match='address'):
        validate_matched_episodes(duplicate, families_per_split=FAMILY_PLAN,
                                  complexities=SMALL)

    dated = deepcopy(episodes)
    dated[0]['corrections'][0]['observed_at'] = dated[0]['facts'][0]['observed_at']
    with pytest.raises(ValueError, match='observed after|generator'):
        validate_matched_episodes(dated, families_per_split=FAMILY_PLAN,
                                  complexities=SMALL)

    observations = deepcopy(episodes)
    observations[0]['observations'][0]['lines'].pop()
    with pytest.raises(ValueError, match='observation'):
        validate_matched_episodes(observations, families_per_split=FAMILY_PLAN,
                                  complexities=SMALL)

    labelled = deepcopy(episodes)
    labelled[0]['facts'][0]['expected_answer'] = 'gold'
    with pytest.raises(ValueError, match='unexpected fields|label'):
        validate_matched_episodes(labelled, families_per_split=FAMILY_PLAN,
                                  complexities=SMALL)


def test_verifier_labels_flip_correction_and_keep_unaffected(tmp_path):
    episodes = generate_matched_episodes(complexities=SMALL)
    labels = derive_matched_labels(episodes)
    episode = episodes[0]
    initial = labels[episode['episode_id']]['initial']
    corrected = labels[episode['episode_id']]['corrected']
    corrected_query = corrected['corrected_query_ids'][0]
    old_expected = initial['targets'][corrected_query]['expected']
    new_expected = corrected['targets'][corrected_query]['expected']
    assert old_expected != new_expected
    assert initial['targets'][corrected_query]['target_role'] == 'source'
    assert corrected['targets'][corrected_query]['target_role'] == 'correction_target'
    for query_id in corrected['unaffected_query_ids']:
        assert corrected['targets'][query_id]['expected'] == \
            initial['targets'][query_id]['expected']
        assert corrected['targets'][query_id]['target_role'] == 'retained'
    unknown = corrected['unknown_query_ids'][0]
    assert corrected['targets'][unknown]['expected'] == 'unknown'
    assert corrected['targets'][unknown]['target_entry_id'] is None
    assert corrected['negative_entries'], 'A correction must expose a superseded negative'


def test_tiny_run_preserves_episode_identity_and_mechanics(tmp_path):
    model, episodes, protocol, report, output = _tiny_run(tmp_path)
    assert report['status'] == 'completed'
    assert report['optimizer_updates'] == 0
    assert report['parameters_updated'] == []
    assert report['mechanical']['frozen']['parameter_versions_equal'] is True
    assert report['mechanical']['frozen']['requires_grad_all_false'] is True
    assert report['mechanical_gates']['zero_updates']['status'] == 'pass'
    assert report['mechanical_gates']['save_reload_parity']['status'] == 'pass'
    assert report['mechanical_gates']['row_inventory']['status'] == 'pass'
    assert report['mechanical_valid'] is True
    assert report['counts']['readout_rows'] == len(_rows(output / 'readouts.jsonl'))
    assert report['counts']['answer_rows'] == len(_rows(output / 'answers.jsonl'))
    assert report['readouts_sha256'] == hashlib.sha256(
        (output / 'readouts.jsonl').read_bytes()).hexdigest()
    for name in ('readouts.jsonl', 'answers.jsonl', 'codebooks.json',
                 'mechanical.json', 'rankings.jsonl', 'failures.json'):
        assert (output / name).is_file()
    assert all(row['equal'] is True for row in report['mechanical']['rows'])

    readouts = _rows(output / 'readouts.jsonl')
    answers = _rows(output / 'answers.jsonl')
    codebooks = json.loads((output / 'codebooks.json').read_text())['codebooks']
    for row in readouts + answers:
        assert not any(marker in row for marker in
                       ('expected', 'expected_answer', 'gold', 'label',
                        'audit_label', 'role', 'kind', 'split'))
    # One codebook per episode/state serves every readout stage.
    for key, book in codebooks.items():
        assert key == f"{book['episode_id']}/{book['state']}"
        assert book['observation_sha256'] == _sha_json([
            {'page_id': page['page_id'], 'observed_at': page['observed_at'],
             'date': page['date'], 'lines': list(page['lines'])}
            for page in next(state['observations'] for episode in episodes
                             for state in episode_states(episode)
                             if episode['episode_id'] == book['episode_id']
                             and state['name'] == book['state'])])
    # Every stage reads the identical observation bytes for an episode/state.
    by_state = {}
    for row in readouts:
        by_state.setdefault((row['episode_id'], row['state']), set()).add(
            row['observation_sha256'])
    assert all(len(values) == 1 for values in by_state.values())
    # Stage 2 and 3 share the independently rendered query digest.
    query_digests = {}
    for row in readouts:
        if row['stage'] in (1, 2, 3) and not row['case_id'].startswith('control:'):
            query_digests.setdefault(
                (row['episode_id'], row['state'], row['query_id']), set()).add(
                    row['query_text_sha256'])
    assert all(len(values) == 1 for values in query_digests.values())
    for row in answers:
        assert row['observation_sha256'] == next(
            value for (episode_id, state), values in by_state.items()
            if episode_id == row['episode_id'] and state == row['state']
            for value in values)
        assert row['query_text_sha256'] == next(
            value for (episode_id, state, query_id), values in query_digests.items()
            if episode_id == row['episode_id'] and state == row['state']
            and query_id == row['query_id'] for value in values)
    # Context removal changes only the fast state: the prompt bytes are identical.
    memory_prompts = {(row['episode_id'], row['state'], row['query_id']):
                      row['prompt_sha256'] for row in answers
                      if row['arm'] == 'memory'}
    removed_prompts = {(row['episode_id'], row['state'], row['query_id']):
                       row['prompt_sha256'] for row in answers
                       if row['arm'] == 'context_removed'}
    assert memory_prompts == removed_prompts
    assert {row['arm'] for row in answers} == {
        'memory', 'exact_source', 'full_context', 'context_removed'}
    # Cross-world controls for every readout stage use a foreign, non-local fact.
    episodes_by_id = {episode['episode_id']: episode for episode in episodes}
    fact_index = {fact['record_id']: (episode, fact)
                  for episode in episodes for fact in episode['facts']}
    cross_world_stages = set()
    for row in readouts:
        if row['case_id'] != 'control:cross_world':
            continue
        cross_world_stages.add(row['stage'])
        assert row['status'] == 'ok' and row['context_source_ids']
        source = fact_index[row['context_source_ids'][0]]
        owner = episodes_by_id[row['episode_id']]
        assert source[0]['family_id'] != owner['family_id']
        local_keys = {(fact['entity'], fact['attribute'])
                      for fact in owner['facts'] + owner['distractors']}
        local_keys.update((correction['entity'], correction['attribute'])
                          for correction in owner['corrections'])
        assert (source[1]['entity'], source[1]['attribute']) not in local_keys
        assert row['context_source_ids'][0] not in codebooks[
            row['codebook_key']]['entries']
    assert cross_world_stages == {1, 2, 3}
    assert report['resource']['deadline_seconds'] == 900
    assert report['resource']['deadline_ok'] is True
    assert report['decision'] in {
        'stage1_storage_upper_bound_failed', 'stage2_addressing_failed',
        'stage3_value_discrimination_failed',
        'stage4_native_generation_failed'}
    assert report['first_failing_stage'] in (1, 2, 3, 4)
    assert report['promote_v5_memory'] is False
    assert report['neural_promotion'] is False


def _synthetic_codebook():
    return {
        'a': {'value_vector': [[1.0, 0.0]], 'write_key': [[1.0, 0.0]],
              'value_sha256': 'a', 'write_key_sha256': 'a',
              'value_norm': 1.0, 'key_norm': 1.0},
        'b': {'value_vector': [[0.0, 1.0]], 'write_key': [[0.0, 1.0]],
              'value_sha256': 'b', 'write_key_sha256': 'b',
              'value_norm': 1.0, 'key_norm': 1.0},
        'c': {'value_vector': [[0.8, 0.2]], 'write_key': [[0.8, 0.2]],
              'value_sha256': 'c', 'write_key_sha256': 'c',
              'value_norm': 0.824621125, 'key_norm': 0.824621125},
        'old': {'value_vector': [[0.9, 0.1]], 'write_key': [[0.9, 0.1]],
                'value_sha256': 'old', 'write_key_sha256': 'old',
                'value_norm': 0.905538514, 'key_norm': 0.905538514},
    }


def _synthetic_labels():
    live = [
        {'record_id': 'a', 'entity': 'e1', 'attribute': 'a1', 'value': 'v1',
         'role': 'correction_target', 'source_family': 'source',
         'origin_world_id': 'w1', 'observed_at': 1, 'live': True},
        {'record_id': 'b', 'entity': 'e2', 'attribute': 'a2', 'value': 'v2',
         'role': 'retained', 'source_family': 'source',
         'origin_world_id': 'w1', 'observed_at': 1, 'live': True},
        {'record_id': 'c', 'entity': 'e3', 'attribute': 'a3', 'value': 'v3',
         'role': 'distractor', 'source_family': 'independent_world',
         'origin_world_id': 'w2', 'observed_at': 1, 'live': True},
    ]
    negative = [
        {'record_id': 'old', 'entity': 'e1', 'attribute': 'a1', 'value': 'v0',
         'role': 'superseded_negative', 'source_family': 'source',
         'origin_world_id': 'w1', 'observed_at': 1, 'live': False},
    ]
    return {
        'episode_id': 'e1', 'world_id': 'w1', 'family_id': 'f1',
        'split': 'dev', 'complexity': 'correction', 'variant': 0,
        'state': 'corrected', 'cutoff': 2,
        'observed_record_ids': ['a', 'b', 'c', 'old'],
        'live_entries': live, 'negative_entries': negative,
        'targets': {
            'qf': {'expected': 'v1', 'kind': 'fact',
                   'target_entry_id': 'a', 'target_role': 'correction_target',
                   'target_source_family': 'source', 'query_form': 0,
                   'query_text_sha256': 'query'},
            'qg': {'expected': 'v2', 'kind': 'fact',
                   'target_entry_id': 'b', 'target_role': 'retained',
                   'target_source_family': 'source', 'query_form': 1,
                   'query_text_sha256': 'query'},
            'qu': {'expected': 'unknown', 'kind': 'unknown',
                   'target_entry_id': None, 'target_role': 'unknown',
                   'target_source_family': 'none', 'query_form': 2,
                   'query_text_sha256': 'query'},
        },
        'corrected_query_ids': ['qf'], 'unaffected_query_ids': ['qg'],
        'unknown_query_ids': ['qu'],
    }


def _synthetic_codebook_record():
    return {
        'format': CODEBOOK_FORMAT, 'episode_id': 'e1', 'world_id': 'w1',
        'state': 'corrected', 'cutoff': 2, 'layer': 0,
        'observation_sha256': 'obs', 'memory_history_sha256': 'hist',
        'state_digest': 'digest',
        'entries': _synthetic_codebook(),
        'live_entry_ids': ['a', 'b', 'c'],
        'negative_entry_ids': ['old'],
    }


def _synthetic_row(case_id, readout, *, query_id=None, stage=3, arm=None,
                   episode_id='e1', owner_entry_id=None):
    arm = arm or ('query_value' if not case_id.startswith('control:')
                  else 'control')
    return {
        'format': 'replica.memory_matched.readout.v1', 'stage': stage,
        'episode_id': episode_id, 'world_id': 'w1', 'state': 'corrected',
        'cutoff': 2, 'case_id': case_id, 'query_id': query_id,
        'arm': arm, 'codebook_key': f'{episode_id}/corrected', 'layer': 0,
        'observation_sha256': 'obs', 'query_text_sha256': 'query',
        'readout_space': 'value', 'owner_entry_id': owner_entry_id,
        'status': 'ok',
        'unavailable_reason': None,
        'readout': readout, 'readout_sha256': _sha_json(readout),
        'readout_norm': (sum(value * value for head in readout for value in head)
                         ** 0.5),
    }


def _synthetic_mechanical():
    limits = dict(DEFAULT_LIMITS)
    return {
        'frozen': {'optimizer_updates': 0, 'parameters_updated': [],
                   'parameter_versions_equal': True,
                   'requires_grad_all_false': True},
        'rows': [{'equal': True}], 'limits': limits,
    }


def test_rank_margin_unknown_and_correction_scoring():
    labels = {'e1': {'corrected': _synthetic_labels()}}
    codebooks = {'e1/corrected': _synthetic_codebook_record()}
    readouts = [
        _synthetic_row('qf', [[1.0, 0.0]], query_id='qf'),
        _synthetic_row('qg', [[0.0, 1.0]], query_id='qg'),
        _synthetic_row('qu', [[0.0, 1.0]], query_id='qu'),
        _synthetic_row('control:zero', [[0.0, 0.0]]),
    ]
    inventory = [(3, 'e1', 'corrected', row['case_id'], row['arm'])
                 for row in readouts]
    score = score_matched(
        readouts, [], codebooks, labels,
        mechanical=_synthetic_mechanical(), inventory=inventory)
    rankings = {row['case_id']: row for row in score['rankings']}
    qf = rankings['qf']
    assert qf['target_rank'] == 1 and qf['top1'] is True
    assert qf['margin_over_strongest_distractor'] > 0
    assert qf['superseded_margin'] > 0
    qg = rankings['qg']
    assert qg['target_rank'] == 1 and qg['top1'] is True
    qu = rankings['qu']
    assert qu['target_entry_id'] is None
    assert qu['false_recall'] is True
    assert qu['false_recall_kind'] == 'unknown_address'
    zero = rankings['control:zero']
    assert zero['accepted'] is False and zero['false_recall'] is False
    stage3 = score['stages'][3]['metrics']
    assert stage3['correction']['top1_rate'] == 1.0
    assert stage3['unaffected']['top1_rate'] == 1.0
    assert stage3['unknown']['false_recall_rate'] == 1.0
    assert stage3['false_recall_by_kind']['unknown_address'] == 1

    # A target ranked behind a stronger distractor has rank > 1 and a negative margin.
    weaker = _synthetic_row('qf', [[0.2, 0.8]], query_id='qf')
    weaker_inventory = [(3, 'e1', 'corrected', 'qf', 'query_value')]
    ranked = score_matched(
        [weaker], [], codebooks, labels,
        mechanical=_synthetic_mechanical(), inventory=weaker_inventory)
    row = ranked['rankings'][0]
    assert row['target_rank'] > 1
    assert row['top1'] is False
    assert row['margin_over_strongest_distractor'] < 0


def test_scoring_rejects_duplicate_missing_and_foreign_rows():
    labels = {'e1': {'corrected': _synthetic_labels()}}
    codebooks = {'e1/corrected': _synthetic_codebook_record()}
    rows = [
        _synthetic_row('qf', [[1.0, 0.0]], query_id='qf'),
        _synthetic_row('qu', [[0.0, 1.0]], query_id='qu'),
    ]
    inventory = [(3, 'e1', 'corrected', row['case_id'], row['arm'])
                 for row in rows]
    with pytest.raises(ValueError, match='Duplicate'):
        score_matched(rows + [deepcopy(rows[0])], [], codebooks, labels,
                      mechanical=_synthetic_mechanical(), inventory=inventory)
    missing = [(3, 'e1', 'corrected', 'absent', 'query_value')]
    with pytest.raises(ValueError, match='Missing'):
        score_matched(rows, [], codebooks, labels,
                      mechanical=_synthetic_mechanical(),
                      inventory=missing + inventory[:-1])
    foreign = deepcopy(rows)
    foreign[0]['codebook_key'] = 'other/corrected'
    with pytest.raises(ValueError, match='missing codebook'):
        score_matched(foreign, [], codebooks, labels,
                      mechanical=_synthetic_mechanical(), inventory=inventory)
    forged = deepcopy(rows)
    forged[0]['readout_sha256'] = '0' * 64
    with pytest.raises(ValueError, match='Readout bytes'):
        score_matched(forged, [], codebooks, labels,
                      mechanical=_synthetic_mechanical(), inventory=inventory)
    duplicated_answers = [{
        'format': 'replica.memory_matched.answer.v1', 'stage': 4,
        'episode_id': 'e1', 'world_id': 'w1', 'state': 'corrected', 'cutoff': 2,
        'query_id': 'qf', 'query_form': 0, 'query_text_sha256': 'query',
        'observation_sha256': 'obs', 'arm': 'memory', 'context_sha256': 'ctx',
        'context_source_ids': [], 'prompt_sha256': 'p', 'prompt_tokens': 1,
        'max_new_tokens': 6, 'token_ids': [1], 'generated_tokens': 1,
        'answer': 'v1', 'answer_sha256': hashlib.sha256(b'v1').hexdigest(),
        'status': 'ok'}]
    with pytest.raises(ValueError, match='Duplicate matched answer'):
        score_matched(
            rows, duplicated_answers + [deepcopy(duplicated_answers[0])],
            codebooks, labels, mechanical=_synthetic_mechanical(),
            inventory=inventory + [(4, 'e1', 'corrected', 'qf', 'memory')])


def test_answer_scoring_uses_verifier_labels_and_context_removal():
    labels = {'e1': {'corrected': _synthetic_labels()}}
    codebooks = {'e1/corrected': _synthetic_codebook_record()}
    answers = []
    plan = {
        'memory': {'qf': 'v1', 'qg': 'v2', 'qu': 'unknown'},
        'context_removed': {'qf': 'v2', 'qg': 'v1', 'qu': 'v2'},
        'exact_source': {'qf': 'v1', 'qg': 'v2', 'qu': 'unknown'},
        'full_context': {'qf': 'v1', 'qg': 'v2', 'qu': 'unknown'},
    }
    for arm, mapping in plan.items():
        for query_id, answer in mapping.items():
            answers.append({
                'format': 'replica.memory_matched.answer.v1', 'stage': 4,
                'episode_id': 'e1', 'world_id': 'w1', 'state': 'corrected',
                'cutoff': 2, 'query_id': query_id, 'query_form': 0,
                'query_text_sha256': 'query', 'observation_sha256': 'obs',
                'arm': arm, 'context_sha256': 'ctx', 'context_source_ids': [],
                'prompt_sha256': 'p', 'prompt_tokens': 1, 'max_new_tokens': 6,
                'token_ids': [1], 'generated_tokens': 1, 'answer': answer,
                'answer_sha256': hashlib.sha256(answer.encode()).hexdigest(),
                'status': 'ok',
            })
    inventory = [(4, 'e1', 'corrected', row['query_id'], row['arm'])
                 for row in answers]
    score = score_matched([], answers, codebooks, labels,
                          mechanical=_synthetic_mechanical(),
                          inventory=inventory)
    metrics = score['stages'][4]['metrics']
    assert metrics['memory']['accuracy'] == 1.0
    assert metrics['memory']['unknown']['false_answer_rate'] == 0.0
    assert metrics['context_removed']['accuracy'] == 0.0
    assert metrics['exact_source']['accuracy'] == 1.0
    assert metrics['full_context']['accuracy'] == 1.0


def test_first_failing_stage_selection():
    def stages(fail):
        return {stage: {'status': ('fail' if stage == fail else 'pass')}
                for stage in (1, 2, 3, 4)}
    assert select_first_failing_stage(stages(1), mechanical_valid=True) == 1
    assert select_first_failing_stage(stages(2), mechanical_valid=True) == 2
    assert select_first_failing_stage(stages(3), mechanical_valid=True) == 3
    assert select_first_failing_stage(stages(4), mechanical_valid=True) == 4
    assert select_first_failing_stage(stages(0), mechanical_valid=True) == 0
    assert select_first_failing_stage(stages(3), mechanical_valid=False) is None


def test_inventory_and_workload_are_model_free_and_exact():
    episodes = generate_matched_episodes(complexities=SMALL)
    inventory = matched_inventory(episodes)
    assert len(inventory) == len(set(inventory))
    workload = matched_workload(episodes, {'limits': {}})
    assert workload['readout_rows'] + workload['answer_rows'] == len(inventory)
    assert workload['per_stage']['4'] == workload['answer_rows']
    stages = {item[0] for item in inventory}
    assert stages == {1, 2, 3, 4}
    assert any(item[3] == 'control:superseded_probe' for item in inventory)
    assert any(item[3] == 'unknown' for item in inventory)


@require_local_artifacts('checkpoints/minicpm5_source',
                         'configs/minicpm5_sealed_lock.local.json',
                         'runs/memory_reliability_v5_attempt01/candidate_extension')
def test_prepare_verify_and_admit_pins_are_regenerative(tmp_path):
    output = tmp_path / 'prepared'
    manifest = prepare_matched(output)
    assert manifest['native_computation'] is False
    assert manifest['audit']['episodes'] == 30
    assert (output / 'episodes.json').is_file()
    assert (output / 'labels.json').is_file()
    assert (output / 'protocol.json').is_file()
    assert (output / 'prompt_audit.json').is_file()
    assert (output / 'generation_audit.json').is_file()
    labels = json.loads((output / 'labels.json').read_text())
    protocol = json.loads((output / 'protocol.json').read_text())
    prompt_audit = json.loads((output / 'prompt_audit.json').read_text())
    generation_audit = json.loads((output / 'generation_audit.json').read_text())
    assert labels['format'] == LABELS_FORMAT
    assert labels['evaluation_only'] is True
    assert protocol['budget']['max_wall_seconds'] == 900
    assert protocol['budget'] == manifest['budget']
    assert 'declared monotonic wall-clock deadline' in protocol['stop_conditions']
    assert protocol['generation']['max_prompt_tokens'] == prompt_audit['bound']
    assert protocol['limits']['max_prompt_tokens'] == prompt_audit['bound']
    assert manifest['prompt_audit_sha256'] == hashlib.sha256(
        (output / 'prompt_audit.json').read_bytes()).hexdigest()
    assert manifest['prompt_bound'] >= manifest['observed_max_prompt_tokens']
    assert all(row['tokens'] <= prompt_audit['bound']
               for row in prompt_audit['rows'])
    assert generation_audit['previous_bound_would_fail'] is True
    assert generation_audit['previous_bound_answer_only_would_fail'] is True
    assert generation_audit['per_role']['correction_target'][
        'over_previous_bound_answer_only'] == 12
    assert protocol['generation']['max_new_tokens'] == generation_audit['bound']
    assert protocol['limits']['max_new_tokens'] == generation_audit['bound']
    assert manifest['generation_audit_sha256'] == hashlib.sha256(
        (output / 'generation_audit.json').read_bytes()).hexdigest()
    assert manifest['max_new_tokens'] == generation_audit['bound']
    assert generation_audit['answer_truncation'] is False
    assert generation_audit['eos_suppression'] is False
    assert all(row['required_with_eos'] <= generation_audit['bound']
               for row in generation_audit['rows'])
    receipt = admit_matched_prepared(output)
    assert receipt['status'] == 'admitted'
    assert receipt['checks']['seeded_episode_regeneration'] is True
    assert receipt['checks']['verifier_labels_regenerated'] is True
    assert receipt['checks']['prompt_audit_regenerated'] is True
    assert receipt['checks']['prompt_bound_covers_rows'] is True
    assert receipt['checks']['complete_pages_no_truncation'] is True
    assert receipt['checks']['generation_audit_regenerated'] is True
    assert receipt['checks']['generation_bound_covers_targets'] is True
    assert receipt['checks']['no_answer_truncation_or_eos_suppression'] is True
    assert receipt['checks']['label_free_writer_inputs'] is True
    assert verify_matched_prepared(output)['status'] == 'admitted'

    labels_path = output / 'labels.json'
    original = labels_path.read_text()
    payload = json.loads(original)
    payload['episodes'][next(iter(payload['episodes']))]['initial']['targets'][
        'q0']['expected'] = 'tampered'
    labels_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    with pytest.raises(ValueError):
        verify_matched_prepared(output)
    labels_path.write_text(original)
    assert verify_matched_prepared(output)['status'] == 'admitted'


def test_default_frozen_answerer_enables_fast_port_for_memory_only(tmp_path):
    from unittest.mock import patch as mock_patch

    from replica_lab.architecture.loading import encode_prompt
    from replica_lab.memory_matched import (
        MATCHED_TEMPLATE, _build_memory, _default_answerer_factory,
        _ledger_for_state, _state_scope,
    )
    from replica_lab.memory_system.session import MemorySession
    from replica_lab.runtime.exact_memory import ExactMemoryReference

    episodes = generate_matched_episodes(complexities=SMALL)
    episode = episodes[0]
    state = episode_states(episode)[0]
    layer = 0
    model = tiny_adapter(fast_memory=SPEC).eval()
    before = tuple((name, parameter._version)
                   for name, parameter in model.named_parameters())
    memory = _build_memory(model, Tokenizer(), episode, state)
    ledger = _ledger_for_state(episode, state, tmp_path)
    reference = ExactMemoryReference(ledger)
    captured, port_calls = [], []
    original_prepare = MemorySession.prepare

    def spy(session, **kwargs):
        request = original_prepare(session, **kwargs)
        frame = request.frame
        captured.append({
            'ports': tuple(frame.enabled_ports),
            'branches': tuple(frame.enabled_branches),
            'fast': frame.fast is not None,
            'fast_digest': (frame.fast.state_digest
                            if frame.fast is not None else None),
        })
        return request

    handle = model.fast_ports[str(layer)].register_forward_pre_hook(
        lambda module, args, kwargs: port_calls.append(bool(kwargs.get('enabled'))),
        with_kwargs=True)
    try:
        scope = _state_scope(episode, state)
        answerer = _default_answerer_factory(
            model, Tokenizer(),
            {'layer': layer, 'generation': {'max_new_tokens': 2}})(ledger)
        query = episode['queries'][0]
        prompt = MATCHED_TEMPLATE.replace('{query}', query['text']).replace(
            '{observation}', '')
        ids = encode_prompt(Tokenizer(), prompt, 'cpu')
        with mock_patch.object(MemorySession, 'prepare', new=spy):
            text, tokens = answerer(
                prompt_ids=ids, prompt_mask=torch.ones_like(ids),
                query=query['text'], scope=scope, fast_state=memory.read())
            assert 0 < len(tokens) <= 2
            assert text == Tokenizer().decode(
                tokens, skip_special_tokens=True).strip()
            memory_port_calls = list(port_calls)
            port_calls.clear()
            text, tokens = answerer(
                prompt_ids=ids, prompt_mask=torch.ones_like(ids),
                query=query['text'], scope=scope, fast_state=None)
            assert 0 < len(tokens) <= 2
            assert text == Tokenizer().decode(
                tokens, skip_special_tokens=True).strip()
    finally:
        handle.remove()
        reference.close()
        ledger.close()
    memory_frame, removed_frame = captured
    assert memory_frame['ports'] == (layer,)
    assert memory_frame['branches'] == ('fast',)
    assert memory_frame['fast'] is True and memory_frame['fast_digest']
    assert True in memory_port_calls, (
        'memory fast-state port must be numerically enabled')
    assert removed_frame['ports'] == ()
    assert 'fast' not in removed_frame['branches']
    assert removed_frame['fast'] is False
    assert True not in port_calls
    after = tuple((name, parameter._version)
                  for name, parameter in model.named_parameters())
    assert before == after


def test_outside_fact_excludes_counterfactual_and_local_matches():
    from replica_lab.memory_matched import _outside_fact

    episodes = generate_matched_episodes()
    sibling_pair = [episodes[0], episodes[1]]
    assert sibling_pair[0]['family_id'] == sibling_pair[1]['family_id']
    with pytest.raises(ValueError, match='No independent foreign fact'):
        _outside_fact(sibling_pair, sibling_pair[0])

    for episode in episodes:
        outside = _outside_fact(episodes, episode)
        owner = next(item for item in episodes
                     if item['world_id'] == outside['source_world_id'])
        assert owner['family_id'] != episode['family_id']
        local_keys = {(fact['entity'], fact['attribute'])
                      for fact in episode['facts'] + episode['distractors']}
        local_keys.update((correction['entity'], correction['attribute'])
                          for correction in episode['corrections'])
        local_values = {fact['value']
                        for fact in episode['facts'] + episode['distractors']}
        local_values.update(correction['value']
                            for correction in episode['corrections'])
        assert (outside['entity'], outside['attribute']) not in local_keys
        assert outside['value'] not in local_values


def test_interference_requires_both_near_miss_types():
    episodes = generate_matched_episodes()
    validate_matched_episodes(episodes)
    interference = [episode for episode in episodes
                    if episode['complexity'] == 'interference']
    assert interference
    for episode in interference:
        target = next(fact for fact in episode['facts']
                      if fact['record_id'] == episode['corrections'][0]['fact_id'])
        same_attribute = [item for item in episode['distractors']
                          if item['attribute'] == target['attribute']
                          and item['entity'] != target['entity']]
        same_entity = [item for item in episode['distractors']
                       if item['entity'] == target['entity']
                       and item['attribute'] != target['attribute']]
        assert same_attribute and same_entity
        assert (same_attribute[0]['origin_world_id']
                != same_entity[0]['origin_world_id'])

    corrupted = deepcopy(interference[0])
    target = next(fact for fact in corrupted['facts']
                  if fact['record_id'] == corrupted['corrections'][0]['fact_id'])
    near_entity = next(item for item in corrupted['distractors']
                       if item['entity'] == target['entity']
                       and item['attribute'] != target['attribute'])
    near_entity['entity'] = 'ent-unobserved-corruption'
    with pytest.raises(ValueError, match='same-attribute and same-entity'):
        validate_matched_episodes([corrupted] + [
            episode for episode in episodes
            if episode['episode_id'] != corrupted['episode_id']])


def test_control_rates_aggregate_every_row_and_are_order_independent():
    episodes = ['e1', 'e2', 'e3']
    labels = {episode_id: {'corrected': dict(
        _synthetic_labels(), episode_id=episode_id)} for episode_id in episodes}
    codebooks = {}
    for episode_id in episodes:
        book = _synthetic_codebook_record()
        book['episode_id'] = episode_id
        book['world_id'] = 'w1'
        codebooks[f'{episode_id}/corrected'] = book

    def row(episode_id, case_id, readout, **kwargs):
        return _synthetic_row(case_id, readout, episode_id=episode_id, **kwargs)

    rows = [
        row('e1', 'control:zero', [[0.0, 0.0]]),
        row('e1', 'control:wrong_key', [[1.0, 0.0]], owner_entry_id='a'),
        row('e1', 'control:superseded_probe', [[1.0, 0.0]], query_id='qf'),
        row('e2', 'control:zero', [[0.0, 1.0]]),
        row('e2', 'control:wrong_key', [[0.0, 1.0]], owner_entry_id='a'),
        row('e2', 'control:superseded_probe', [[0.0, 1.0]], query_id='qf'),
        row('e3', 'control:zero', [[0.0, 0.0]]),
        row('e3', 'control:wrong_key', [[1.0, 0.0]], owner_entry_id='a'),
        row('e3', 'control:superseded_probe', [[1.0, 0.0]], query_id='qf'),
    ]

    def inventory(items):
        return [(3, item['episode_id'], 'corrected', item['case_id'], item['arm'])
                for item in items]

    forward = score_matched(
        rows, [], codebooks, labels, mechanical=_synthetic_mechanical(),
        inventory=inventory(rows))
    reversed_score = score_matched(
        list(reversed(rows)), [], codebooks, labels,
        mechanical=_synthetic_mechanical(), inventory=inventory(rows))
    metrics = forward['stages'][3]['metrics']
    assert metrics['zero_false_recall_rate'] == pytest.approx(1 / 3)
    assert metrics['wrong_key_owner_rate'] == pytest.approx(2 / 3)
    assert metrics['superseded_probe_top1_rate'] == pytest.approx(2 / 3)
    assert metrics['control_counts'] == {
        'zero': 3, 'wrong_key': 3, 'superseded_probe': 3}
    for key in ('zero_false_recall_rate', 'wrong_key_owner_rate',
                'superseded_probe_top1_rate', 'control_counts'):
        assert reversed_score['stages'][3]['metrics'][key] == metrics[key]


def test_unknown_answer_rate_uses_the_same_normalized_answer():
    labels = _synthetic_labels()
    labels['targets'].pop('qg')
    labels['unaffected_query_ids'] = []
    labels['unknown_query_ids'] = ['qu1', 'qu2', 'qu3', 'qu4']
    for index, query_id in enumerate(('qu1', 'qu2', 'qu3', 'qu4'), start=2):
        labels['targets'][query_id] = {
            'expected': 'unknown', 'kind': 'unknown', 'target_entry_id': None,
            'target_role': 'unknown', 'target_source_family': 'none',
            'query_form': index, 'query_text_sha256': 'query'}
    labels_by_episode = {'e1': {'corrected': labels}}
    codebooks = {'e1/corrected': _synthetic_codebook_record()}
    answers = []
    mapping = {'qf': 'v1', 'qu1': 'Unknown.', 'qu2': '"unknown"',
               'qu3': 'unknown.', 'qu4': 'unknown value'}
    for query_id, answer in mapping.items():
        answers.append({
            'format': 'replica.memory_matched.answer.v1', 'stage': 4,
            'episode_id': 'e1', 'world_id': 'w1', 'state': 'corrected',
            'cutoff': 2, 'query_id': query_id, 'query_form': 0,
            'query_text_sha256': 'query', 'observation_sha256': 'obs',
            'arm': 'memory', 'context_sha256': 'ctx', 'context_source_ids': [],
            'prompt_sha256': 'p', 'prompt_tokens': 1, 'max_new_tokens': 6,
            'token_ids': [1], 'generated_tokens': 1, 'answer': answer,
            'answer_sha256': hashlib.sha256(answer.encode()).hexdigest(),
            'status': 'ok',
        })
    score = score_matched(
        [], answers, codebooks, labels_by_episode,
        mechanical=_synthetic_mechanical(),
        inventory=[(4, 'e1', 'corrected', row['query_id'], 'memory')
                   for row in answers])
    unknown = score['stages'][4]['metrics']['memory']['unknown']
    assert unknown['n'] == 4
    assert unknown['accuracy'] == pytest.approx(0.75)
    assert unknown['false_answer_rate'] == pytest.approx(0.25)


def test_declared_deadline_is_enforced():
    assert enforce_elapsed_deadline(100.0, 900, now=999.999) == pytest.approx(
        899.999)
    assert enforce_elapsed_deadline(100.0, 900, now=1000.0) == pytest.approx(
        900.0)
    with pytest.raises(TimeoutError):
        enforce_elapsed_deadline(100.0, 900, now=1000.001)
    with pytest.raises(ValueError, match='negative'):
        enforce_elapsed_deadline(100.0, 900, now=99.999)
    with pytest.raises(ValueError, match='positive integer'):
        enforce_elapsed_deadline(100.0, 0, now=101.0)


@require_local_artifacts('checkpoints/minicpm5_source',
                         'configs/minicpm5_sealed_lock.local.json',
                         'runs/memory_reliability_v5_attempt01/candidate_extension')
def test_prepared02_audits_interference_and_declared_budget():
    prepared = ROOT / 'data/memory_matched_20261025/prepared02'
    if not (prepared / 'episodes.json').is_file():
        pytest.skip('prepared02 is not provisioned; run scripts/prepare_memory_matched.py --admit')
    data = json.loads((prepared / 'episodes.json').read_text())
    protocol = json.loads((prepared / 'protocol.json').read_text())
    manifest = json.loads((prepared / 'manifest.json').read_text())
    audit = validate_matched_episodes(
        data['episodes'], families_per_split=data['families_per_split'],
        complexities=tuple(data['complexities']), variants=data['variants'])
    assert audit == data['audit'] == manifest['audit']
    assert protocol['budget']['max_wall_seconds'] == 900
    assert protocol['budget'] == manifest['budget']
    assert 'declared monotonic wall-clock deadline' in protocol['stop_conditions']
    prompt_audit = json.loads((prepared / 'prompt_audit.json').read_text())
    assert prompt_audit['previous_bound_would_fail'] is True
    assert prompt_audit['observed_max_tokens'] > prompt_audit['configured_bound']
    assert protocol['generation']['max_prompt_tokens'] == prompt_audit['bound']
    assert protocol['limits']['max_prompt_tokens'] == prompt_audit['bound']
    assert protocol['prompt_audit']['sha256'] == hashlib.sha256(
        (prepared / 'prompt_audit.json').read_bytes()).hexdigest()
    assert all(row['tokens'] <= prompt_audit['bound']
               for row in prompt_audit['rows'])
    generation_audit = json.loads((prepared / 'generation_audit.json').read_text())
    assert generation_audit['previous_bound_would_fail'] is True
    assert generation_audit['previous_bound_answer_only_would_fail'] is True
    assert generation_audit['per_kind']['fact'][
        'over_previous_bound_answer_only'] == 96
    assert generation_audit['per_role']['correction_target'][
        'over_previous_bound_answer_only'] == 12
    assert protocol['generation']['max_new_tokens'] == generation_audit['bound']
    assert protocol['limits']['max_new_tokens'] == generation_audit['bound']
    assert manifest['generation_audit_sha256'] == hashlib.sha256(
        (prepared / 'generation_audit.json').read_bytes()).hexdigest()
    assert generation_audit['answer_truncation'] is False
    assert generation_audit['eos_suppression'] is False
    assert all(row['required_with_eos'] <= generation_audit['bound']
               for row in generation_audit['rows'])
    receipt = verify_matched_prepared(prepared)
    assert receipt['status'] == 'admitted'
    assert receipt['checks']['verifier_labels_regenerated'] is True
    assert receipt['checks']['prompt_audit_regenerated'] is True
    assert receipt['checks']['prompt_bound_covers_rows'] is True
    assert receipt['checks']['generation_audit_regenerated'] is True
    assert receipt['checks']['generation_bound_covers_targets'] is True
    interference = [episode for episode in data['episodes']
                    if episode['complexity'] == 'interference']
    assert interference
    for episode in interference:
        target = next(fact for fact in episode['facts']
                      if fact['record_id'] == episode['corrections'][0]['fact_id'])
        assert any(item['attribute'] == target['attribute']
                   and item['entity'] != target['entity']
                   for item in episode['distractors'])
        assert any(item['entity'] == target['entity']
                   and item['attribute'] != target['attribute']
                   for item in episode['distractors'])


def test_post_profiler_and_post_report_deadline_crossings_fail_closed():
    # Profiler stop itself crosses the deadline: no completed report is written.
    now = [0.0]
    writes = []

    def clock():
        return now[0]

    def stop_crossing():
        now[0] = 901.0
        return {'elapsed_seconds': 900.0}

    result = {'format': 'test'}
    with pytest.raises(TimeoutError):
        finalize_completed_report(
            result, started_monotonic=0.0, max_wall_seconds=900,
            stop_profiler=stop_crossing, write_report=writes.append,
            clock=clock)
    assert writes and writes[-1]['status'] == 'failed'
    assert writes[-1]['deadline_ok'] is False
    assert all(item['status'] != 'completed' for item in writes)

    # Report serialization itself crosses after a completed write: the artifact
    # is overwritten with a failed report and never left completed.
    now[0] = 0.0
    writes.clear()

    def stop_within_budget():
        now[0] = 100.0
        return {'elapsed_seconds': 99.0}

    def write_then_cross(value):
        writes.append(deepcopy(value))
        if len(writes) == 1:
            now[0] = 901.0

    result = {'format': 'test'}
    with pytest.raises(TimeoutError):
        finalize_completed_report(
            result, started_monotonic=0.0, max_wall_seconds=900,
            stop_profiler=stop_within_budget,
            write_report=write_then_cross, clock=clock)
    assert writes[0]['status'] == 'completed'
    assert writes[-1]['status'] == 'failed'
    assert writes[-1]['deadline_ok'] is False
    assert not any(item['status'] == 'completed'
                   and item['deadline_ok'] is False for item in writes)
    assert result['status'] == 'failed' and result['deadline_ok'] is False

    # A profiler summary over budget is rejected even if the wall clock agrees.
    writes.clear()
    result = {'format': 'test'}
    with pytest.raises(TimeoutError):
        finalize_completed_report(
            result, started_monotonic=0.0, max_wall_seconds=900,
            stop_profiler=lambda: {'elapsed_seconds': 901.0},
            write_report=writes.append, clock=lambda: 100.0)
    assert writes[-1]['status'] == 'failed'
    assert writes[-1]['deadline_ok'] is False

    # Within-budget finalization writes only completed reports.
    now[0] = 0.0
    writes.clear()

    def stop_ok():
        now[0] = 50.0
        return {'elapsed_seconds': 49.0}

    def write_ok(value):
        writes.append(deepcopy(value))
        now[0] += 10.0

    result = {'format': 'test'}
    finalized = finalize_completed_report(
        result, started_monotonic=0.0, max_wall_seconds=900,
        stop_profiler=stop_ok, write_report=write_ok, clock=clock)
    assert finalized['status'] == 'completed'
    assert finalized['deadline_ok'] is True
    assert finalized['elapsed_seconds'] >= finalized['profile']['elapsed_seconds']
    assert all(item['status'] == 'completed' for item in writes)
    assert all(item['deadline_ok'] is True for item in writes)


@require_local_artifacts('checkpoints/minicpm5_source',
                         'configs/minicpm5_sealed_lock.local.json')
def test_prompt_bound_audit_detects_previous_256_failure_before_native():
    from replica_lab.memory_matched import _load_sealed_tokenizer, audit_prompt_bounds

    prepared = ROOT / 'data/memory_matched_20261025/prepared02'
    if not (prepared / 'episodes.json').is_file():
        pytest.skip('prepared02 is not provisioned; run scripts/prepare_memory_matched.py --admit')
    data = json.loads((prepared / 'episodes.json').read_text())
    protocol = json.loads((prepared / 'protocol.json').read_text())
    stored = json.loads((prepared / 'prompt_audit.json').read_text())
    tokenizer = _load_sealed_tokenizer(ROOT / 'checkpoints/minicpm5_source')
    audit = audit_prompt_bounds(
        data['episodes'], tokenizer, configured_bound=256)
    assert audit == stored
    assert audit['configured_bound'] == 256
    assert audit['previous_bound_would_fail'] is True
    assert audit['observed_max_tokens'] > 256
    assert audit['max_row']['tokens'] == audit['observed_max_tokens']
    assert audit['first_row_over_previous_bound']['tokens'] > 256
    assert audit['bound'] >= audit['observed_max_tokens']
    assert audit['bound'] > 256
    assert all(row['tokens'] <= audit['bound'] for row in audit['rows'])
    assert audit['complete_pages'] is True
    assert audit['truncation'] is False
    assert audit['per_arm']['full_context']['max_tokens'] <= audit['bound']
    assert protocol['generation']['max_prompt_tokens'] == audit['bound']
    assert protocol['limits']['max_prompt_tokens'] == audit['bound']
    assert protocol['prompt_audit']['previous_bound_would_fail'] is True
    assert protocol['prompt_audit']['max_row']['tokens'] == \
        audit['observed_max_tokens']


@require_local_artifacts('checkpoints/minicpm5_source',
                         'configs/minicpm5_sealed_lock.local.json')
def test_generation_budget_audit_detects_6_token_confound_before_native():
    from replica_lab.memory_matched import (
        _load_sealed_tokenizer, audit_generation_bounds,
    )

    prepared = ROOT / 'data/memory_matched_20261025/prepared02'
    if not (prepared / 'episodes.json').is_file():
        pytest.skip('prepared02 is not provisioned; run scripts/prepare_memory_matched.py --admit')
    data = json.loads((prepared / 'episodes.json').read_text())
    labels = json.loads((prepared / 'labels.json').read_text())
    protocol = json.loads((prepared / 'protocol.json').read_text())
    manifest = json.loads((prepared / 'manifest.json').read_text())
    stored = json.loads((prepared / 'generation_audit.json').read_text())
    tokenizer = _load_sealed_tokenizer(ROOT / 'checkpoints/minicpm5_source')
    audit = audit_generation_bounds(
        data['episodes'], labels['episodes'], tokenizer, configured_bound=6)
    assert audit == stored
    assert audit['configured_bound'] == 6
    assert audit['previous_bound_would_fail'] is True
    assert audit['previous_bound_answer_only_would_fail'] is True
    assert audit['per_kind']['fact']['n'] == 120
    assert audit['per_kind']['fact']['over_previous_bound_answer_only'] == 96
    assert audit['per_role']['correction_target']['n'] == 12
    assert audit['per_role']['correction_target'][
        'over_previous_bound_answer_only'] == 12
    assert audit['observed_max_required_tokens'] == \
        audit['max_row']['required_tokens']
    assert audit['bound'] == audit['observed_max_required_with_eos'] == \
        audit['max_row']['required_with_eos']
    assert audit['bound'] > 6
    assert audit['bound'] == max(row['required_with_eos']
                                 for row in audit['rows'])
    assert all(row['required_with_eos'] <= audit['bound']
               for row in audit['rows'])
    assert audit['eos_required'] is True
    assert audit['eos_token_id'] == tokenizer.eos_token_id
    assert audit['answer_truncation'] is False
    assert audit['eos_suppression'] is False
    assert protocol['generation']['max_new_tokens'] == audit['bound']
    assert protocol['limits']['max_new_tokens'] == audit['bound']
    assert protocol['generation_audit']['bound'] == audit['bound']
    assert protocol['generation_audit']['previous_bound_would_fail'] is True
    assert manifest['max_new_tokens'] == audit['bound']
    assert manifest['generation_previous_bound_would_fail'] is True
    assert manifest['per_kind_max_required_tokens'][
        'fact'] == audit['per_kind']['fact']['max_required_tokens']

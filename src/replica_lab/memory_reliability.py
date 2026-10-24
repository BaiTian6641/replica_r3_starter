"""Counterfactual persistent-memory episodes and dataset-derived answer contracts."""
from __future__ import annotations

import random
from collections import Counter, defaultdict

TEMPLATE = ('Use relevant stored project facts when needed. Otherwise use general knowledge. '
            'If a requested project fact is absent, reply unknown. Reply with only the value.\n\n'
            'Query: {query}\n\nAvailable source context:\n{observation}')
VALUES = ('cobalt', 'saffron', 'teal', 'violet', 'amber', 'copper')
GENERAL = {'train': [('What is 2 plus 3?', '5'), ('What is 8 minus 2?', '6')],
           'dev': [('What is 3 plus 4?', '7'), ('What is 9 minus 5?', '4')],
           'heldout': [('What is 7 plus 8?', '15'), ('What is 9 minus 3?', '6')]}


def render_fact_query(entity, attribute, form=0):
    templates = (
        'What is the {attribute} for project {entity}?',
        'Tell me the {attribute} of project {entity}.',
        'Project {entity}: what is its {attribute}?',
        'Return the {attribute} assigned to project {entity}.')
    if type(form) is not int or not 0 <= form < len(templates):
        raise ValueError('Unknown query form')
    return templates[form].format(entity=entity, attribute=attribute)


def build_episodes(seed=20261024, family_plan=None, values_by_split=None, query_forms=False):
    rng = random.Random(seed)
    episodes = []
    if family_plan is None:
        family_plan = [('train', range(4)), ('dev', range(100, 101)), ('heldout', range(200, 201))]
    if values_by_split is None:
        values_by_split = {'train': VALUES, 'dev': VALUES,
                           'heldout': ('indigo', 'crimson', 'teal', 'violet', 'amber', 'copper')}
    for split, families in family_plan:
        for family in families:
            names = (f'Cedar{family}', f'Birch{family}')
            values = list(values_by_split[split])
            rng.shuffle(values)
            keys = [(name, attr) for name in names for attr in ('launch code', 'deploy region')]
            for variant in range(2):
                eid = f'reliability_{split}_{family}_{variant}'
                facts = [{'entity': name, 'attribute': attr, 'value': values[(i + variant * 2) % 4]}
                         for i, (name, attr) in enumerate(keys)]
                distractors = [{'entity': f'Other{family}', 'attribute': 'review code', 'value': 'silver'},
                               {'entity': f'Other{family}', 'attribute': 'backup window', 'value': 'bronze'}]
                displayed = facts + distractors
                rng.shuffle(displayed)
                pages = ['\n'.join(f"The {f['attribute']} for project {f['entity']} is {f['value']}." for f in displayed[i:i+3])
                         for i in (0, 3)]
                changed = dict(facts[family % 4], value=values[4 + variant])
                questions = []
                for i, (name, attr) in enumerate(keys):
                    question = {'query_id': f'q{i}', 'query': render_fact_query(name, attr, (family + i) % 4 if query_forms else 0),
                                'entity': name, 'attribute': attr, 'kind': 'fact'}
                    if query_forms:
                        question['query_form'] = (family + i) % 4
                    questions.append(question)
                unknown_form = family % 4 if query_forms else 0
                unknown = {'query_id': 'unknown', 'query': render_fact_query(f'Missing{family}', 'launch code', unknown_form),
                           'entity': f'Missing{family}', 'attribute': 'launch code', 'kind': 'unknown'}
                if query_forms:
                    unknown['query_form'] = unknown_form
                questions.append(unknown)
                general = [{'query_id': f'g{i}', 'query': text, 'expected_answer': answer, 'kind': 'general'}
                           for i, (text, answer) in enumerate(GENERAL[split])]
                episodes.append({'episode_id': eid, 'family_id': f'family_{family}', 'split': split,
                                 'variant': variant, 'facts': facts, 'distractors': distractors,
                                 'observations': pages, 'correction': changed,
                                 'correction_observation': f"Update: the {changed['attribute']} for project {changed['entity']} is {changed['value']}.",
                                 'questions': questions, 'general_questions': general})
    return episodes


def cases(episode, stage):
    if stage not in ('initial', 'corrected'):
        raise ValueError('Unknown memory stage')
    facts = {(f['entity'], f['attribute']): f['value'] for f in episode['facts']}
    correction = episode['correction']
    if stage == 'corrected':
        facts[(correction['entity'], correction['attribute'])] = correction['value']
    result = []
    for q in episode['questions'] + (episode['general_questions'] if stage == 'corrected' else []):
        expected = (q['expected_answer'] if q['kind'] == 'general'
                    else facts.get((q['entity'], q['attribute']), 'unknown'))
        kind = q['kind']
        if stage == 'corrected' and kind == 'fact':
            kind = ('changed' if (q['entity'], q['attribute']) == (correction['entity'], correction['attribute']) else 'unaffected')
        result.append({'case_id': f"{episode['episode_id']}:{stage}:{q['query_id']}",
                       'query': q['query'], 'expected_answer': expected, 'kind': kind,
                       'episode_id': episode['episode_id'], 'family_id': episode['family_id'],
                       'stage': stage, 'split': episode['split'],
                       'subject_id': 'memory_reliability', 'purpose': 'reliability',
                       'branch_id': episode['episode_id'], 'as_of': 3 if stage == 'corrected' else 2,
                       'observations': episode['observations'] + ([episode['correction_observation']] if stage == 'corrected' else [])})
    return result


def audit_dataset(episodes, require_novel_values=False):
    if len({e['episode_id'] for e in episodes}) != len(episodes):
        raise ValueError('Duplicate episode')
    families, answers, variants = defaultdict(set), defaultdict(list), defaultdict(list)
    query_texts = defaultdict(set)
    last_correct = first_correct = total = 0
    if not episodes or set(e['split'] for e in episodes) != {'train', 'dev', 'heldout'}:
        raise ValueError('Require nonempty training, development and heldout splits')
    for e in episodes:
        families[e['family_id']].add(e['split'])
        variants[e['family_id']].append(e['variant'])
        if len(e['facts']) != 4 or len(e['distractors']) != 2 or len(e['observations']) != 2:
            raise ValueError('Episode fact/page inventory differs')
        if len({(f['entity'], f['attribute']) for f in e['facts'] + e['distractors']}) != 6:
            raise ValueError('Ambiguous or duplicate fact key')
        if len(set(f['value'] for f in e['facts'])) != 4:
            raise ValueError('Multi-query facts must have distinct values')
        rendered = [f"The {f['attribute']} for project {f['entity']} is {f['value']}."
                    for f in e['facts'] + e['distractors']]
        if (any(len(page.splitlines()) != 3 for page in e['observations'])
                or Counter(line for page in e['observations'] for line in page.splitlines()) != Counter(rendered)):
            raise ValueError('Rendered observations differ from structured oracle')
        fact_map = {(f['entity'], f['attribute']): f['value'] for f in e['facts']}
        changed = e['correction']
        key = changed['entity'], changed['attribute']
        if key not in fact_map or changed['value'] in fact_map.values():
            raise ValueError('Correction must change exactly one known fact to a new value')
        if e['correction_observation'] != f"Update: the {changed['attribute']} for project {changed['entity']} is {changed['value']}.":
            raise ValueError('Correction rendering differs from structured oracle')
        if (len({q['query_id'] for q in e['questions'] + e['general_questions']}) != 7
                or len(e['questions']) != 5 or len(e['general_questions']) != 2):
            raise ValueError('Question inventory differs')
        queried = {(q['entity'], q['attribute']) for q in e['questions'] if q['kind'] == 'fact'}
        if queried != set(fact_map) or sum(q['kind'] == 'unknown' for q in e['questions']) != 1:
            raise ValueError('Queries must cover every fact and one absent key')
        for q in e['questions']:
            if q['query'] != render_fact_query(q['entity'], q['attribute'], q.get('query_form', 0)) or \
                    (q['kind'] == 'unknown' and (q['entity'], q['attribute']) in fact_map):
                raise ValueError('Query text or absent-fact oracle differs')
        for q in e['questions'] + e['general_questions']:
            query_texts[(e['family_id'], q['query_id'])].add(q['query'])
        if [(q['query'], q['expected_answer']) for q in e['general_questions']] != GENERAL[e['split']]:
            raise ValueError('General-answer oracle differs')
        for q in cases(e, 'initial'):
            if q['kind'] != 'fact':
                continue
            answers[(e['split'], q['query'])].append(q['expected_answer'])
            last = e['observations'][-1].splitlines()[-1].rsplit(' is ', 1)[-1].removesuffix('.')
            first = e['observations'][0].splitlines()[0].rsplit(' is ', 1)[-1].removesuffix('.')
            last_correct += last == q['expected_answer']; first_correct += first == q['expected_answer']; total += 1
    if any(len(values) != 1 for values in query_texts.values()):
        raise ValueError('Counterfactual variants must have identical queries')
    if any(len(x) != 1 for x in families.values()):
        raise ValueError('Family crosses splits')
    if any(sorted(x) != [0, 1] for x in variants.values()):
        raise ValueError('Every family requires both counterfactual variants')
    if any(len(v) != 2 or len(set(v)) != 2 for v in answers.values()):
        raise ValueError('Missing counterfactual pair with changed answer')
    query_only = sum(max(Counter(v).values()) for v in answers.values()) / total
    if max(last_correct / total, first_correct / total) > .25 or query_only > .5:
        raise ValueError('Shortcut bound exceeded')
    if require_novel_values:
        heldout_values = {f['value'] for e in episodes if e['split'] == 'heldout'
                          for f in e['facts'] + [e['correction']]}
        prior_values = {f['value'] for e in episodes if e['split'] != 'heldout' for f in e['facts']}
        if heldout_values & prior_values:
            raise ValueError('Heldout values overlap training or development values')
    return {'episodes': len(episodes), 'families': len(families), 'fact_queries': total,
            'first_line_accuracy': first_correct / total, 'last_line_accuracy': last_correct / total,
            'query_only_best_accuracy': query_only, 'counterfactual_pairs': len(answers),
            'family_splits_disjoint': True, 'rendered_oracle_verified': True,
            'split_episodes': dict(Counter(e['split'] for e in episodes)),
            'split_families': {split: len({e['family_id'] for e in episodes if e['split'] == split})
                               for split in ('train', 'dev', 'heldout')},
            'query_forms': dict(Counter(q.get('query_form', 0) for e in episodes for q in e['questions'])),
            'heldout_values_absent_from_train_and_dev': sorted(
                {f['value'] for e in episodes if e['split'] == 'heldout'
                 for f in e['facts'] + [e['correction']]}
                - {f['value'] for e in episodes if e['split'] != 'heldout' for f in e['facts']}),
            'heldout_values_absent_from_training': sorted(
                {f['value'] for e in episodes if e['split'] == 'heldout' for f in e['facts']}
                - {f['value'] for e in episodes if e['split'] == 'train' for f in e['facts']})}

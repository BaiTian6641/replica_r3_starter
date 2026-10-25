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

# Prospective retention pool: arithmetic, factual, abbreviation and relational
# questions. It is opt-in so historical datasets and verdicts stay unchanged.
GENERAL_V2 = {
    'train': [('What is 12 plus 7?', '19'), ('What is 15 minus 4?', '11'),
              ('What color is cobalt?', 'blue'), ('What is the capital of France?', 'Paris'),
              ('What does HTTP stand for?', 'Hypertext Transfer Protocol'),
              ('What is 6 times 3?', '18'), ('What is the opposite of hot?', 'cold'),
              ('How many days are in a week?', '7')],
    'dev': [('What is 11 plus 6?', '17'), ('What is 14 minus 5?', '9'),
            ('What color is saffron?', 'yellow'), ('What is the capital of Japan?', 'Tokyo'),
            ('What does CPU stand for?', 'central processing unit'),
            ('What is 5 times 4?', '20'), ('What is the opposite of cold?', 'hot'),
            ('How many hours are in a day?', '24')],
    'heldout': [('What is 13 plus 8?', '21'), ('What is 16 minus 7?', '9'),
                ('What color is violet?', 'purple'), ('What is the capital of Canada?', 'Ottawa'),
                ('What does RAM stand for?', 'random access memory'),
                ('What is 7 times 2?', '14'), ('What is the opposite of light?', 'dark'),
                ('How many minutes are in an hour?', '60')],
}


def _general_selection(split, family, oracle=None):
    custom = oracle is not None
    if isinstance(oracle, str):
        if oracle != 'general_v2':
            raise ValueError('Unknown general-question oracle')
        oracle = GENERAL_V2
    pools = GENERAL if oracle is None else oracle
    if split not in pools:
        raise ValueError('Missing general-question split')
    pool = [tuple(item) for item in pools[split]]
    if len(pool) < 2 or any(len(item) != 2 or not all(isinstance(x, str) and x.strip() for x in item)
                            for item in pool):
        raise ValueError('General-question pool must contain nonempty query/answer pairs')
    if len({item[0] for item in pool}) != len(pool):
        raise ValueError('Duplicate general-question text')
    if not custom:
        return pool
    try:
        index = int(family)
    except (TypeError, ValueError) as exc:
        raise ValueError('Custom general pools require numeric family indices') from exc
    start = (index * 2) % len(pool)
    return [pool[start], pool[(start + 1) % len(pool)]]


def render_fact_query(entity, attribute, form=0):
    templates = (
        'What is the {attribute} for project {entity}?',
        'Tell me the {attribute} of project {entity}.',
        'Project {entity}: what is its {attribute}?',
        'Return the {attribute} assigned to project {entity}.')
    if type(form) is not int or not 0 <= form < len(templates):
        raise ValueError('Unknown query form')
    return templates[form].format(entity=entity, attribute=attribute)


def build_episodes(seed=20261024, family_plan=None, values_by_split=None, query_forms=False,
                   distractor_values_by_split=None, general_by_split=None):
    rng = random.Random(seed)
    episodes = []
    if family_plan is None:
        family_plan = [('train', range(4)), ('dev', range(100, 101)), ('heldout', range(200, 201))]
    if values_by_split is None:
        values_by_split = {'train': VALUES, 'dev': VALUES,
                           'heldout': ('indigo', 'crimson', 'teal', 'violet', 'amber', 'copper')}
    for split, families in family_plan:
        for family in families:
            distractor_values = (distractor_values_by_split or {}).get(split, ('silver', 'bronze'))
            names = (f'Cedar{family}', f'Birch{family}')
            values = list(values_by_split[split])
            rng.shuffle(values)
            keys = [(name, attr) for name in names for attr in ('launch code', 'deploy region')]
            for variant in range(2):
                eid = f'reliability_{split}_{family}_{variant}'
                facts = [{'entity': name, 'attribute': attr, 'value': values[(i + variant * 2) % 4]}
                         for i, (name, attr) in enumerate(keys)]
                distractors = [{'entity': f'Other{family}', 'attribute': 'review code', 'value': distractor_values[0]},
                               {'entity': f'Other{family}', 'attribute': 'backup window', 'value': distractor_values[1]}]
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
                           for i, (text, answer) in enumerate(_general_selection(split, family, general_by_split))]
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


TRAINING_SCHEDULE_VERSION = 'round_robin_seeded_shuffle.v1'
TRAINING_REPORT_FORMAT = 'replica.memory_reliability.training.v2'
HISTORICAL_TRAINING_PROTOCOLS = frozenset(
    f'replica.memory_reliability.protocol.v{version}' for version in range(1, 6))


def training_schedule_contract(protocol):
    """Resolve only explicitly supported schedules, including recorded v1–v5."""
    config = protocol['training']
    historical = protocol.get('format') in HISTORICAL_TRAINING_PROTOCOLS
    version = config.get('schedule_version')
    repeat = config.get('repeat_policy')
    if historical:
        version = TRAINING_SCHEDULE_VERSION if version is None else version
        repeat = 'once' if repeat is None else repeat
    if version != TRAINING_SCHEDULE_VERSION or repeat != 'once':
        raise ValueError('Unsupported or undeclared training schedule/repeat policy')
    for name in ('updates', 'examples_per_update'):
        if type(config.get(name)) is not int or config[name] < 1:
            raise ValueError(f'Training {name} must be a positive integer')
    if type(config.get('seed')) is not int:
        raise ValueError('Training seed must be an integer, excluding booleans')
    return version, repeat


def training_schedule(episodes, protocol):
    """Reconstruct the declared one-visit round-robin, then seeded shuffle."""
    training_schedule_contract(protocol)
    worlds = [(episode, cases(episode, 'initial') + cases(episode, 'corrected'))
              for episode in episodes if episode['split'] == 'train']
    if not worlds or any(len(questions) != 12 for _, questions in worlds):
        raise ValueError('Declared memory schedule requires twelve cases per training world')
    schedule = [(episode, questions[index]) for index in range(12)
                for episode, questions in worlds]
    if len({case['case_id'] for _, case in schedule}) != len(schedule):
        raise ValueError('Training schedule has duplicate case IDs')
    config = protocol['training']
    if config['updates'] * config['examples_per_update'] != len(schedule):
        raise ValueError('Declared once-only schedule and update budget differ')
    random.Random(config['seed']).shuffle(schedule)
    return tuple(schedule)


def audit_dataset(episodes, require_novel_values=False, general_oracle=None, min_unique_general=0):
    if len({e['episode_id'] for e in episodes}) != len(episodes):
        raise ValueError('Duplicate episode')
    if type(min_unique_general) is not int or min_unique_general < 0:
        raise ValueError('Minimum unique general-question count must be nonnegative')
    general_queries = defaultdict(set)
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
        if any(q.get('kind') != 'general' for q in e['general_questions']):
            raise ValueError('General-question kind differs')
        expected_general = _general_selection(
            e['split'], e['family_id'].rsplit('_', 1)[-1], general_oracle)
        if [(q['query'], q['expected_answer']) for q in e['general_questions']] != expected_general:
            raise ValueError('General-answer oracle differs')
        general_queries[e['split']].update(q['query'] for q in e['general_questions'])
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
                          for f in e['facts'] + e['distractors'] + [e['correction']]}
        prior_values = {f['value'] for e in episodes if e['split'] != 'heldout'
                        for f in e['facts'] + e['distractors'] + [e['correction']]}
        if heldout_values & prior_values:
            raise ValueError('Heldout values overlap training or development values')
    unique_general_by_split = {split: len(general_queries[split]) for split in ('train', 'dev', 'heldout')}
    unique_general = len(general_queries['dev'] | general_queries['heldout'])
    if unique_general < min_unique_general:
        raise ValueError('Insufficient unique evaluation general questions')
    return {'episodes': len(episodes), 'families': len(families), 'fact_queries': total,
            'first_line_accuracy': first_correct / total, 'last_line_accuracy': last_correct / total,
            'query_only_best_accuracy': query_only, 'counterfactual_pairs': len(answers),
            'family_splits_disjoint': True, 'rendered_oracle_verified': True,
            'unique_general_questions': unique_general,
            'unique_general_questions_by_split': unique_general_by_split,
            'split_episodes': dict(Counter(e['split'] for e in episodes)),
            'split_families': {split: len({e['family_id'] for e in episodes if e['split'] == split})
                               for split in ('train', 'dev', 'heldout')},
            'query_forms': dict(Counter(q.get('query_form', 0) for e in episodes for q in e['questions'])),
            'heldout_values_absent_from_train_and_dev': sorted(
                {f['value'] for e in episodes if e['split'] == 'heldout'
                 for f in e['facts'] + e['distractors'] + [e['correction']]}
                - {f['value'] for e in episodes if e['split'] != 'heldout'
                 for f in e['facts'] + e['distractors'] + [e['correction']]}),
            'heldout_values_absent_from_training': sorted(
                {f['value'] for e in episodes if e['split'] == 'heldout'
                 for f in e['facts'] + e['distractors'] + [e['correction']]}
                - {f['value'] for e in episodes if e['split'] == 'train'
                 for f in e['facts'] + e['distractors'] + [e['correction']]})}

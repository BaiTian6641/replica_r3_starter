"""Deterministic source-backed synthetic worlds, with disjoint decision groups.

Labels come from the structured world oracle, never a generated explanation.
Creating a split is preparation. Native admission hashes prepared bytes only;
the runner parses calibration, fitting and development behind separate gates.
"""
from collections import Counter

FORMAT = 'replica.intrinsic_memory.coverage_worlds.v11'
SEED = 2026102911
COLORS = ('red', 'blue', 'green', 'gold')
FAMILIES = ('explicit_entity_color_lookup', 'latest_recorded_color')
# Names/roots are split independently from source variants. Every counterfactual
# and question variant of a decision group stays in exactly one split.
ENTITIES = {
    'calibration': ('Arven', 'Belis', 'Cevra', 'Dunel', 'Eshra', 'Fovik'),
    'fit': ('Gavren', 'Hesil', 'Ivora', 'Jumet', 'Kovren', 'Leris', 'Mavik', 'Nesra',
            'Ovrin', 'Pelis', 'Quvara', 'Romet', 'Suvren', 'Teris', 'Uvik', 'Vesra',
            'Woren', 'Xelis', 'Yavik', 'Zesra'),
    'development': ('Amrel', 'Bovin', 'Ceril', 'Dovra', 'Emret', 'Ferin', 'Govel', 'Hirra',
                    'Imrel', 'Jovin', 'Keril', 'Lovra'),
}


def oracle(facts, query):
    matches = [f for f in facts if f['entity'] == query]
    if not matches:
        return 'unknown'
    revisions = [f['revision'] for f in matches]
    if len(revisions) != len(set(revisions)):
        raise ValueError('ambiguous same-revision authority')
    return max(matches, key=lambda fact: fact['revision'])['color']


def render(facts, family):
    if family == FAMILIES[0]:
        if len(facts) != 1 or facts[0]['revision'] != 1:
            raise ValueError('one initial fact required')
        text = f"The recorded color of {facts[0]['entity']} is {facts[0]['color']}. "
    elif family == FAMILIES[1]:
        if (len(facts) != 2 or [f['revision'] for f in facts] != [1, 2]
                or facts[0]['entity'] != facts[1]['entity']):
            raise ValueError('two ordered revisions of one entity required')
        text = (f"Earlier, the recorded color of {facts[0]['entity']} was {facts[0]['color']}. "
                f"The latest record replaces it: the color of {facts[1]['entity']} is {facts[1]['color']}. ")
    else:
        raise ValueError('undeclared rule family')
    return text + 'No color is recorded for any other entity.'


def question(entity, family):
    adjective = 'latest recorded' if family == FAMILIES[1] else 'recorded'
    return f'What is the {adjective} color of {entity}?'


def group_rows(split, group, family, rotation, names):
    result = []
    palette = COLORS[rotation:] + COLORS[:rotation]
    cases = [(0, palette[0], False), (0, palette[1], False),
             (1, palette[2], False), (1, palette[3], False),
             (2, palette[1], False), (2, palette[0], True),
             (3, palette[3], False), (3, palette[2], True)]
    for offset, (pair, color, missing) in enumerate(cases):
        query = names[pair]
        entity = names[4] if missing else query
        # Earlier facts cannot equal either supported answer in the first pair;
        # the final direct-family correction case therefore has a simple target.
        old = palette[(pair + 2) % 4]
        facts = ([{'entity': entity, 'color': old, 'revision': 1},
                  {'entity': entity, 'color': color, 'revision': 2}]
                 if family == FAMILIES[1]
                 else [{'entity': entity, 'color': color, 'revision': 1}])
        result.append({'world_id': f'v11_{split}_{group*8+offset:02d}',
                       'family': f'v11_{split}_decision_{group*4+pair:02d}',
                       'criterion_group': group, 'rule_family': family,
                       'facts': facts, 'query_entity': query,
                       'text': render(facts, family), 'question': question(query, family),
                       'answer': oracle(facts, query), 'chronology': max(f['revision'] for f in facts),
                       'source_kind': 'scripted_complete_world_snapshot',
                       'verification': 'structured_revision_oracle_not_generated_explanation'})
    return result


def worlds(split):
    if split not in ENTITIES:
        raise ValueError('undeclared split')
    names = ENTITIES[split]
    if split == 'fit':
        rows = []
        for g in range(4):
            # Two complete eight-case groups per rule; rotations balance all
            # four supported colors (three each) plus four unknowns per rule.
            group_names = names[g*4:g*4+4] + (names[16+g],)
            rows.extend(group_rows(split, g, FAMILIES[g//2], g%2, group_names))
    elif split == 'development':
        rows = []
        for g in range(2):
            rows.extend(group_rows(split, g, FAMILIES[g], 2+g,
                                    names[g*4:g*4+4] + (names[8+g],)))
    else:
        direct = group_rows(split, 0, FAMILIES[0], 0, names[:4]+(names[4],))
        latest = group_rows(split, 0, FAMILIES[1], 1, names[:4]+(names[5],))
        rows = direct[:4] + latest[4:]
        for i, row in enumerate(rows):
            row['world_id'] = f'v11_calibration_{i:02d}'
    return {'format': FORMAT, 'split': split, 'seed': SEED,
            'provenance': 'deterministic_predeclared_revision_worlds_v11',
            'prior_development_data_used': False, 'sealed_final': False,
            'independent_seed_replication': False, 'heldout_rule_family_claim': False,
            'rows': rows}


def validate_corpus(document, expected_split):
    if document != worlds(expected_split):
        raise ValueError('source, oracle, order, group or split assignment differs')
    rows = document['rows']
    expected_n = {'calibration': 8, 'fit': 32, 'development': 16}[expected_split]
    if len(rows) != expected_n or len({r['world_id'] for r in rows}) != expected_n:
        raise ValueError('incomplete unique worlds')
    if any(not r['text'] or len(r['text'].encode()) > 256 for r in rows):
        raise ValueError('one bounded source span required')
    for i in range(0, len(rows), 2):
        a, b = rows[i:i+2]
        if (a['question'] != b['question'] or a['answer'] == b['answer']
                or a['family'] != b['family']):
            raise ValueError('same-question counterfactual decision group differs')
    if expected_split == 'fit':
        for family in FAMILIES:
            counts = Counter(r['answer'] for r in rows if r['rule_family'] == family)
            if counts != dict(red=3, blue=3, green=3, gold=3, unknown=4):
                raise ValueError('balanced answer coverage differs')
    return document


def public_split_metadata(split):
    # No answer or canonical text is returned. This metadata may be read at
    # admission; it neither opens labels nor replaces the prepared file digest.
    group_count = {'calibration': 1, 'fit': 4, 'development': 2}[split]
    return {'split': split, 'cases': group_count*8, 'entities': list(ENTITIES[split]),
            'decision_roots': [f'v11_{split}_decision_{i:02d}' for i in range(group_count*4)],
            'rule_families': list(FAMILIES), 'seed': SEED}

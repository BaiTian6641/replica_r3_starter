"""Answer verification from immutable cases and decoded tokens, not cached scores."""
from __future__ import annotations


def audit_rows(rows, cases, decode):
    """Require exactly one output per case and independently verify every answer.

    ``cases`` is a trusted dataset mapping; generated rows cannot supply gold.
    ``decode`` must use the sealed tokenizer (a small deterministic fake in tests).
    """
    ids = [row.get('case_id') for row in rows]
    if len(ids) != len(set(ids)) or set(ids) != set(cases):
        raise ValueError('Missing, duplicate or unexpected answer cases')
    checked = []
    for row in rows:
        case = cases[row['case_id']]
        tokens = row.get('token_ids')
        if not isinstance(tokens, list) or any(type(x) is not int or x < 0 for x in tokens):
            raise ValueError('Invalid generated token IDs')
        answer = decode(tokens).strip()
        expected = case['expected_answer']
        correct = answer.casefold() == expected.casefold()
        if (row.get('answer') != answer or row.get('expected') != expected
                or type(row.get('correct')) is not bool or row['correct'] != correct
                or row.get('generated_tokens') != len(tokens)):
            raise ValueError('Decoded answer, gold, correctness or token count differs')
        for key in ('split', 'case_role', 'episode_id', 'stage', 'kind'):
            if key in row and key in case and row[key] != case[key]:
                raise ValueError(f'Answer {key} differs from dataset')
        checked.append(dict(row, correct=correct))
    return checked


def accuracy(rows):
    if not rows:
        raise ValueError('An empty evaluation cannot pass')
    return sum(row['correct'] for row in rows) / len(rows)


def audit_m4_answers(report, queries, decode):
    """Validate M4 memberships, tokens and aggregates before evaluating its gates."""
    by_id = {q['case_id']: q for q in queries}
    if len(by_id) != len(queries):
        raise ValueError('Duplicate dataset cases')
    main = {k: q for k, q in by_id.items() if q['split'] in {'dev', 'heldout', 'regression'}}
    groups = [
        (report['candidate'], main, report['metrics'], True),
        (report['parent']['main'], main, report['parent_metrics'], True),
        (report['general'], {k: q for k, q in by_id.items() if q['split'] == 'retention'}, report['general_metrics'], False),
        (report['general_dev'], {k: q for k, q in by_id.items() if q['split'] == 'retention_dev'}, report['general_dev_metrics'], False),
        (report['parent']['retention'], {k: q for k, q in by_id.items() if q['split'] == 'retention'}, report['parent_general_metrics'], False),
        (report['parent']['retention_dev'], {k: q for k, q in by_id.items() if q['split'] == 'retention_dev'}, report['parent_general_dev_metrics'], False),
    ]
    count = 0
    required_arms = [
        {'fast', 'pattern', 'both', 'router', 'linear', 'shuffled', 'zero', 'disabled',
         'full_context', 'read_13_only', 'read_27_only', 'write_13_only', 'write_27_only'},
        {'fast', 'disabled'},
        {'pattern', 'router', 'fast', 'zero', 'disabled', 'shuffled', 'full_context', 'adversarial'},
        {'pattern', 'router', 'fast', 'zero', 'disabled', 'shuffled', 'full_context', 'adversarial'},
        {'fast', 'disabled', 'adversarial'}, {'fast', 'disabled', 'adversarial'},
    ]
    for (arms, cases, cached, split_metrics), required in zip(groups, required_arms, strict=True):
        if set(arms) != required or set(cached) != required:
            raise ValueError('Missing or unexpected M4 arm')
        for arm, rows in arms.items():
            checked = audit_rows(rows, cases, decode)
            measured = ({split: accuracy([x for x in checked if cases[x['case_id']]['split'] == split])
                         for split in ('dev', 'heldout', 'regression')}
                        if split_metrics else accuracy(checked))
            if cached[arm] != measured:
                raise ValueError(f'Cached metrics differ from decoded answers: {arm}')
            count += len(checked)
    train = {k: q for k, q in by_id.items() if q.get('case_role') == 'delayed_training'}
    checked = audit_rows(report['train_results'], train, decode)
    if accuracy(checked) != report['train_correct']:
        raise ValueError('Cached training accuracy differs')
    count += len(checked)
    corrections = report['correction']
    if len(corrections) != len(main) or {x['case_id'] for x in corrections} != set(main):
        raise ValueError('Correction case inventory differs')
    for item in corrections:
        cid = item['case_id']; case = main[cid]
        for key, gold in [('original_answer', case['expected_answer']), ('corrected_answer', case['corrected_answer'])]:
            audit_rows([{'case_id': cid, **item[key]}], {cid: dict(case, expected_answer=gold)}, decode)
            count += 1
    first = next(q for q in queries if q['split'] == 'heldout')
    for key in ('correct_answer', 'reloaded_answer', 'pattern_correct_answer', 'pattern_reloaded_answer'):
        # Legacy nested reload rows omit IDs; their case is fixed by the runner.
        audit_rows([{'case_id': first['case_id'], **report['restart'][key]}], {first['case_id']: first}, decode)
        count += 1
    return {'verified_answer_rows_including_external_controls': count,
            'cached_metrics_verified': True, 'gold_source': 'hashed_dataset',
            'answer_source': 'decoded_token_ids'}

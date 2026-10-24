"""Independent replay of reliability answers and acceptance gates."""
from .answer_audit import audit_rows, accuracy
from .memory_reliability import cases


def _expected(episodes):
    return {q['case_id']: q for e in episodes if e['split'] != 'train'
            for stage in ('initial', 'corrected') for q in cases(e, stage)}


def _case_scope(case):
    from .memory_system import EligibilityScope
    return EligibilityScope(case['subject_id'], case['branch_id'], case['purpose'],
                            case['as_of'], 'parallel_replica', None)


def _gate_contexts(episodes):
    from .memory_system import RecordLedger
    from .runtime.exact_memory import ExactMemoryReference
    reference = ExactMemoryReference(RecordLedger(':memory:'))
    try:
        for episode in episodes:
            reference.admit_episode(episode)
            reference.correct_episode(episode)
        return {case['case_id']: reference.select(case['query'], _case_scope(case))
                for case in _expected(episodes).values()}
    finally:
        reference.close()


def regression_inventory(report, episodes, decode, primary='fast'):
    """List each parent-disabled-correct to candidate-primary-wrong case."""
    expected = _expected(episodes)
    parent = audit_rows([r for r in report['parent'] if r['arm'] == 'disabled'], expected, decode)
    candidate = audit_rows([r for r in report['candidate'] if r['arm'] == primary], expected, decode)
    parent_by_id = {r['case_id']: r for r in parent}
    candidate_by_id = {r['case_id']: r for r in candidate}
    result = {}
    for split in ('dev', 'heldout'):
        result[split] = {}
        for kind in ('fact', 'changed', 'unaffected', 'unknown', 'general'):
            result[split][kind] = sorted(
                cid for cid, case in expected.items()
                if case['split'] == split and case['kind'] == kind
                and parent_by_id[cid]['correct'] and not candidate_by_id[cid]['correct'])
    return result


def family_metrics(report, episodes, decode):
    """Return per-family metrics and counts without treating duplicates as independent."""
    expected = _expected(episodes)
    result = {}
    for phase in ('parent', 'candidate'):
        result[phase] = {}
        for arm in sorted({row['arm'] for row in report[phase]}):
            rows = audit_rows([r for r in report[phase] if r['arm'] == arm], expected, decode)
            result[phase][arm] = {}
            for split in ('dev', 'heldout'):
                families = sorted({case.get('family_id', case['episode_id']) for case in expected.values()
                                   if case['split'] == split})
                result[phase][arm][split] = {}
                for family in families:
                    family_rows = [row for row in rows
                                   if expected[row['case_id']].get('family_id', expected[row['case_id']]['episode_id']) == family]
                    result[phase][arm][split][family] = {
                        kind: accuracy([row for row in family_rows if row['kind'] == kind])
                        for kind in ('fact', 'changed', 'unaffected', 'unknown', 'general')}
    counts = {split: len({case.get('family_id', case['episode_id']) for case in expected.values()
                          if case['split'] == split}) for split in ('dev', 'heldout')}
    return {'by_family': result, 'independent_family_counts': counts}


def metrics_and_checks(report, episodes, decode, protocol=None):
    expected = _expected(episodes)
    protocol = protocol or {}
    configured = protocol.get('evaluation_arms', {})
    parent_arms = tuple(configured.get('parent', ('fast', 'disabled')))
    candidate_arms = tuple(configured.get('candidate', ('fast', 'disabled', 'shuffled', 'last_chunk', 'full_context')))
    primary = protocol.get('primary_fast_arm', 'fast')
    if primary not in candidate_arms:
        raise ValueError('Primary fast arm is not declared')
    if report.get('arm_contract') is not None and report['arm_contract'] != {
            'parent': list(parent_arms), 'candidate': list(candidate_arms)}:
        raise ValueError('Run arm contract differs from protocol')
    metrics, verified = {}, {}
    for phase, arms in [('parent', parent_arms), ('candidate', candidate_arms)]:
        rows = report[phase]
        if set(row['arm'] for row in rows) != set(arms):
            raise ValueError('Missing or unexpected arm')
        verified[phase] = {}
        metrics[phase] = {}
        for arm in arms:
            checked = audit_rows([r for r in rows if r['arm'] == arm], expected, decode)
            verified[phase][arm] = {r['case_id']: r for r in checked}
            metrics[phase][arm] = {split: {kind: accuracy([r for r in checked if expected[r['case_id']]['split'] == split and expected[r['case_id']]['kind'] == kind])
                                         for kind in ('fact', 'changed', 'unaffected', 'unknown', 'general')}
                                  for split in ('dev', 'heldout')}
    regressions = regression_inventory(report, episodes, decode, primary)
    if 'regressions' in report and report['regressions'] != regressions:
        raise ValueError('Cached per-case regressions differ from decoded answers')
    if 'families' in report and report['families'] != family_metrics(report, episodes, decode):
        raise ValueError('Cached family metrics differ from decoded answers')
    gate_contexts = (_gate_contexts(episodes)
                     if protocol.get('exact_reference', {}).get('enabled', False) else {})
    checks = {}
    acceptance = protocol.get('acceptance', {})
    minimum = acceptance.get('fact_correct_changed_unaffected_min_each_split', .75)
    unknown_minimum = acceptance.get('unknown_min_each_split', .75)
    no_regression_kinds = acceptance.get('no_correct_to_wrong_kinds', ('general',))
    for split in ('dev', 'heldout'):
        fast = metrics['candidate'][primary][split]
        for kind in ('fact', 'changed', 'unaffected'):
            checks[f'{split}_{kind}_reliable'] = fast[kind] >= minimum
        checks[f'{split}_unknown'] = fast['unknown'] >= unknown_minimum
        controls = [metrics['candidate'][arm][split]['fact']
                    for arm in ('disabled', 'shuffled', 'last_chunk') if arm in candidate_arms]
        checks[f'{split}_delayed_above_controls'] = fast['fact'] > max(controls)
        checks[f'{split}_retention'] = fast['general'] >= max(
            metrics['candidate']['disabled'][split]['general'], metrics['parent']['fast'][split]['general'])
        for kind in no_regression_kinds:
            check_name = (f'{split}_retention_no_correct_to_wrong' if kind == 'general'
                          else f'{split}_{kind}_no_correct_to_wrong')
            checks[check_name] = not regressions[split][kind]
    checks['disabled_token_parity'] = all(
        verified['parent']['disabled'][cid]['token_ids'] == verified['candidate']['disabled'][cid]['token_ids']
        for cid in expected)
    transitions = report['transitions']
    checks['persistent_correction_lineage'] = (
        len(transitions) == sum(e['split'] != 'train' for e in episodes)
        and all(t['lineage_prefix_preserved'] and t['initial_cursor'] == 2 and t['corrected_cursor'] == 3
                and t['reloaded_before_append'] and not t['source_reencoded_on_reload']
                and t['initial_digest'] != t['corrected_digest'] for t in transitions))
    if protocol.get('exact_reference', {}).get('enabled', False):
        from .runtime.exact_memory import episode_records
        checks['persistent_correction_lineage'] = checks['persistent_correction_lineage'] and all(
            t.get('correction_supersedes') == episode_records(next(
                e for e in episodes if e['episode_id'] == t['episode_id']))[1].supersedes
            and all((t.get(key) or {}).get('view_digest') for key in
                    ('initial_source_binding', 'corrected_source_binding'))
            for t in transitions)
    if protocol.get('evidence_gate', {}).get('enabled', False):
        gate_arms = [arm for arm in candidate_arms if arm != 'ungated']
        gated = [row for row in report['candidate'] if row['arm'] in gate_arms]
        checks['gate_routes_verified'] = all(
            row.get('gate') == gate_contexts[row['case_id']].status
            and tuple(row.get('gate_record_ids', [])) == gate_contexts[row['case_id']].record_ids
            for row in gated)
        off = [row for row in gated if row['arm'] in {primary, 'adversarial'}
               and gate_contexts[row['case_id']].status != 'relevant']
        checks['gate_off_disabled_parity'] = all(
            row['token_ids'] == verified['candidate']['disabled'][row['case_id']]['token_ids']
            for row in off)
        checks['gate_off_coverage'] = len(off) > 0
    reference_arm = protocol.get('reference_arm')
    if reference_arm is not None:
        if reference_arm not in candidate_arms:
            raise ValueError('Exact-reference arm is not declared')
        reference_rows = verified['candidate'][reference_arm]
        full_rows = verified['candidate']['full_context']
        reference_regressions = [cid for cid in expected
                                 if full_rows[cid]['correct'] and not reference_rows[cid]['correct']]
        checks['reference_all_cases_correct'] = all(row['correct'] for row in reference_rows.values())
        checks['reference_full_context_parity'] = all(
            reference_rows[cid]['correct'] == full_rows[cid]['correct'] for cid in expected)
        checks['reference_no_correct_to_wrong'] = not reference_regressions
        checks['reference_off_disabled_parity'] = all(
            row['token_ids'] == verified['candidate']['disabled'][cid]['token_ids']
            for cid, row in reference_rows.items() if gate_contexts[cid].status != 'relevant')
    if 'metrics' in report and report['metrics'] != metrics:
        raise ValueError('Cached metrics differ from independently decoded answers')
    return metrics, checks

"""Independent replay of reliability answers and acceptance gates."""
from .answer_audit import audit_rows, accuracy
from .memory_reliability import (HISTORICAL_TRAINING_PROTOCOLS, TRAINING_REPORT_FORMAT,
                                 cases, training_schedule, training_schedule_contract)


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

def _training_gate_contexts(episodes):
    from .memory_system import RecordLedger
    from .runtime.exact_memory import ExactMemoryReference
    reference = ExactMemoryReference(RecordLedger(':memory:'))
    try:
        for episode in episodes:
            if episode['split'] == 'train':
                reference.admit_episode(episode)
                reference.correct_episode(episode)
        return {case['case_id']: reference.select(case['query'], _case_scope(case))
                for episode in episodes if episode['split'] == 'train'
                for stage in ('initial', 'corrected') for case in cases(episode, stage)}
    finally:
        reference.close()


def _validate_gradient_audit(audit, config):
    if not isinstance(audit, dict) or set(audit) != {'original', 'new'}:
        raise ValueError('Malformed accumulated gradient audit')
    for group in ('original', 'new'):
        item = audit[group]
        if not isinstance(item, dict) or type(item.get('eligible')) is not int or item['eligible'] < 0:
            raise ValueError('Invalid accumulated gradient eligibility')
        for field in ('present', 'nonzero', 'nonfinite'):
            names = item.get(field)
            if (not isinstance(names, list) or any(not isinstance(name, str) for name in names)
                    or len(set(names)) != len(names)):
                raise ValueError('Malformed accumulated gradient tensor inventory')
        if (not set(item['nonzero']) <= set(item['present'])
                or not set(item['nonfinite']) <= set(item['present']) or item['nonfinite']):
            raise ValueError('Inconsistent or nonfinite accumulated gradients')
    if config.get('regime', 'modules_only') == 'modules_only':
        if audit['original']['eligible'] or audit['original']['present']:
            raise ValueError('Original gradients violate modules-only ownership')
    expected = set(config.get('expected_trainable_tensors', ()))
    prefixes = tuple(config.get('trainable_prefixes', ('fast_depth_write_encoders.', 'fast_ports.')))
    if expected:
        if audit['new']['eligible'] != len(expected) or set(audit['new']['present']) - expected:
            raise ValueError('Accumulated gradients differ from declared trainable tensors')
    elif any(not name.startswith(prefixes) for name in audit['new']['present']):
        raise ValueError('Accumulated gradients differ from declared trainable prefixes')


def validate_training_routes(report, episodes, protocol):
    """Verify the complete declared order, grouping, routes and accumulated audits."""
    config = protocol['training']
    version, repeat = training_schedule_contract(protocol)
    schedule = training_schedule(episodes, protocol)
    training = report['training']
    historical = (training.get('format') is None
                  and protocol.get('format') in HISTORICAL_TRAINING_PROTOCOLS)
    if not historical and training.get('format') != TRAINING_REPORT_FORMAT:
        raise ValueError('Unsupported training report format')
    if not historical and (training.get('schedule_version') != version
                           or training.get('repeat_policy') != repeat):
        raise ValueError('Training report schedule differs from protocol')
    rows, updates = training.get('cases'), training.get('updates')
    if not isinstance(rows, list) or not isinstance(updates, list):
        raise ValueError('Per-example rows and ordered update summaries are required')
    expected_ids = [case['case_id'] for _, case in schedule]
    if len(rows) != len(schedule) or training.get('case_ids') != expected_ids:
        raise ValueError('Training case order differs from declared seeded schedule')
    if (len(updates) != config['updates']
            or any(not isinstance(update, dict) or type(update.get('update')) is not int
                   or update['update'] != position
                   for position, update in enumerate(updates, 1))):
        raise ValueError('Training update indices must be exact, unique and ordered')
    probes = config.get('gradient_probe_updates', [1, 2])
    if (not isinstance(probes, (list, tuple))
            or any(type(index) is not int or index < 1 for index in probes)
            or list(probes) != sorted(set(probes))):
        raise ValueError('Invalid gradient probe schedule')
    gated = (protocol.get('evidence_gate', {}).get('enabled', False)
             and protocol.get('exact_reference', {}).get('enabled', False))
    contexts = _training_gate_contexts(episodes) if gated else {}
    totals = {'fast': 0, 'disabled': 0}
    count = config['examples_per_update']
    for position, (row, (_, case)) in enumerate(zip(rows, schedule)):
        expected_update = position // count + 1
        if (not isinstance(row, dict) or type(row.get('update')) is not int
                or row['update'] != expected_update or row.get('case_id') != case['case_id']):
            raise ValueError('Training row order/grouping differs from declared seeded schedule')
        if (row.get('route') not in {'fast', 'disabled'}
                or row.get('state_use') not in {'fast', 'none'}
                or type(row.get('memory_active')) is not bool
                or type(row.get('backward_called')) is not bool):
            raise ValueError('Malformed training route record')
        route = 'fast' if not gated or contexts[case['case_id']].enabled else 'disabled'
        if (row['route'] != route or row['state_use'] != ('fast' if route == 'fast' else 'none')
                or row['memory_active'] != (route == 'fast')):
            raise ValueError('Training route/state differs from independently reconstructed gate')
        if route == 'fast' and not row['backward_called']:
            raise ValueError('Fast-route training example omitted its differentiable backward')
        totals[route] += 1
    for index, update in enumerate(updates, 1):
        scheduled = rows[(index - 1) * count:index * count]
        counts = {route: sum(row['route'] == route for row in scheduled)
                  for route in ('fast', 'disabled')}
        counts = {route: value for route, value in counts.items() if value}
        expected_summary = {
            'scheduled_examples': count,
            'memory_active_examples': counts.get('fast', 0),
            'backward_calls': sum(row['backward_called'] for row in scheduled),
            'gate_off_examples': counts.get('disabled', 0),
            'gate_on_examples': counts.get('fast', 0),
        }
        if any(type(update.get(field)) is not int or update[field] != value
               for field, value in expected_summary.items()):
            raise ValueError('Per-update counters differ from scheduled per-example records')
        if (update.get('route_counts') != counts
                or any(type(value) is not int for value in update['route_counts'].values())):
            raise ValueError('Per-update route inventory differs from scheduled records')
        gradient = update.get('gradients')
        participation = update.get('gradient_participation')
        if index in probes and gradient is None:
            raise ValueError('Declared accumulated gradient probe is missing')
        if gradient is not None:
            _validate_gradient_audit(gradient, config)
            if participation != {'scope': 'accumulated_update', 'audit': gradient}:
                raise ValueError('Accumulated gradient participation differs from the update audit')
        elif participation is not None:
            raise ValueError('Gradient participation has no accumulated audit')
    expected_counts = config.get('expected_route_counts')
    if expected_counts is not None and (totals != expected_counts
            or any(type(value) is not int for value in expected_counts.values())):
        raise ValueError('Training route totals differ from protocol declaration')
    return {'examples': len(rows), 'route_counts': totals,
            'memory_active_examples': totals['fast'],
            'backward_calls': sum(row['backward_called'] for row in rows),
            'disabled_backward_calls': sum(row['backward_called'] and row['route'] == 'disabled'
                                           for row in rows),
            'updates_checked': len(updates), 'schedule_version': version, 'repeat_policy': repeat,
            'historical_protocol': protocol.get('format') if historical else None,
            'gradient_attribution': 'update_level_accumulated_only'}


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
    lineage = protocol.get('state_lineage', {})
    initial_cursor = lineage.get('initial_cursor', 2)
    corrected_cursor = lineage.get('corrected_cursor', 3)
    checks['persistent_correction_lineage'] = (
        len(transitions) == sum(e['split'] != 'train' for e in episodes)
        and all(t['lineage_prefix_preserved'] and t['initial_cursor'] == initial_cursor
                and t['corrected_cursor'] == corrected_cursor
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
        reference_mode = protocol.get('reference_mode', 'parity')
        if reference_mode not in {'parity', 'non_regression'}:
            raise ValueError('Unknown reference acceptance mode')
        if reference_mode == 'parity':
            checks['reference_full_context_parity'] = all(
                reference_rows[cid]['correct'] == full_rows[cid]['correct'] for cid in expected)
        else:
            checks['reference_full_context_non_regression'] = not reference_regressions
        checks['reference_no_correct_to_wrong'] = not reference_regressions
        checks['reference_off_disabled_parity'] = all(
            row['token_ids'] == verified['candidate']['disabled'][cid]['token_ids']
            for cid, row in reference_rows.items() if gate_contexts[cid].status != 'relevant')
    if 'metrics' in report and report['metrics'] != metrics:
        raise ValueError('Cached metrics differ from independently decoded answers')
    return metrics, checks

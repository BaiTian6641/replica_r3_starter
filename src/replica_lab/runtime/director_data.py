"""C13 checked-proposal registry and data-sufficiency gate.

No synthesized labels and no learned weights are created here. The gate only
reports whether observed candidate comparisons are sufficient for a later
director candidate.
"""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

OUTCOMES = {'accepted', 'accepted_opt_in', 'deferred', 'rejected'}
TARGETS = {'record', 'procedure', 'neural', 'policy'}


def validate_registry(payload: dict) -> dict:
    if payload.get('format') != 'replica.c13.checked_proposals.v1':
        raise ValueError('unknown checked-proposal registry format')
    minimums = payload.get('minimums')
    proposals = payload.get('proposals')
    if not isinstance(minimums, dict) or not isinstance(proposals, list) or not proposals:
        raise ValueError('minimums and proposals are required')
    ids, groups = set(), set()
    outcomes, targets, rows = Counter(), Counter(), []
    for item in proposals:
        for key in ('proposal_id', 'target_kind', 'source_group', 'outcome', 'source_report'):
            if not isinstance(item.get(key), str) or not item[key]:
                raise ValueError(f'missing {key}')
        if item['proposal_id'] in ids:
            raise ValueError('duplicate proposal ID')
        ids.add(item['proposal_id']); groups.add(item['source_group'])
        if item['outcome'] not in OUTCOMES:
            raise ValueError('unknown outcome')
        if item['target_kind'] not in TARGETS:
            raise ValueError('unknown target kind')
        outcomes[item['outcome']] += 1; targets[item['target_kind']] += 1
        rows.append({**item, 'accepted': item['outcome'].startswith('accepted')})
    accepted = sum(1 for row in rows if row['accepted'])
    rejected = len(rows) - accepted
    eligible = (len(groups) >= int(minimums['independent_source_groups'])
                and accepted >= int(minimums['accepted'])
                and rejected >= int(minimums['rejected_or_deferred']))
    return {'proposals': rows, 'groups': len(groups), 'accepted': accepted,
            'rejected_or_deferred': rejected,
            'target_counts': dict(targets), 'outcome_counts': dict(outcomes),
            'training_eligible': eligible,
            'decision': 'eligible' if eligible else 'defer_insufficient_checked_proposals',
            'authority': 'data-sufficiency only; no selection or promotion authority'}


def load_registry(path):
    return validate_registry(json.loads(Path(path).read_text()))

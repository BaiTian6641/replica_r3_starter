"""Reusable host-enumerated action candidates and model likelihood reranking.

The host constructs a bounded legal menu from the exact task state and actual
observations. The model scores each candidate JSON continuation; the host
executes the model-selected argmin. This is an opt-in narrower action interface,
not a free-form policy and not a claim that the menu contains no information.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Sequence

MAX_CANDIDATES = 8
MAX_PROMPT_TOKENS = 2048
MAX_TARGET_TOKENS = 96


def canonical_action(value: dict) -> str:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False)


def build_action_menu(tools, state) -> tuple[dict, ...]:
    """Build a bounded legal menu; only observed failure text/constants are used."""
    candidates = [
        {'kind': 'read', 'path': tools.workspace_entry},
        {'kind': 'invoke_tool', 'tool': 'run_test', 'arguments': {}},
        {'kind': 'stop'},
    ]
    failure_text = ''
    for outcome in state.outcomes:
        result = dict(outcome.result) if isinstance(outcome.result, dict) else {}
        receipt = dict(outcome.receipt or {})
        failure_text = ('\n'.join((
            receipt.get('stderr_excerpt') or '', receipt.get('stdout_excerpt') or '',
            result.get('stderr_excerpt') or '', result.get('stdout_excerpt') or ''))).strip()
        if failure_text:
            break
    if failure_text:
        try:
            from .procedures import parse_failure_bound
            bound = parse_failure_bound(failure_text)
            name, expected, actual = (bound['name'], float(bound['expected']),
                                      float(bound['actual']))
            for value in (expected, actual, expected + max(1.0, abs(expected) * 0.1)):
                candidates.append({'kind': 'invoke_tool', 'tool': 'set_constant',
                                   'arguments': {'name': name, 'value': value}})
        except ValueError:
            pass
    try:
        for item in tools.constant_candidates():
            if item.get('name') and isinstance(item.get('value'), (int, float)):
                candidates.append({'kind': 'invoke_tool', 'tool': 'set_constant',
                                   'arguments': {'name': item['name'],
                                                 'value': float(item['value'])}})
    except Exception:
        pass
    result, seen = [], set()
    for candidate in candidates:
        raw = canonical_action(candidate)
        if raw not in seen:
            seen.add(raw); result.append(candidate)
        if len(result) >= MAX_CANDIDATES:
            break
    return tuple(result)


@dataclass(frozen=True)
class CandidateSelection:
    candidates: tuple[dict, ...]
    scores: tuple[float, ...]
    chosen_index: int

    def __post_init__(self):
        if not self.candidates or len(self.candidates) != len(self.scores):
            raise ValueError('candidate/scores length mismatch')
        if type(self.chosen_index) is not int or not 0 <= self.chosen_index < len(self.candidates):
            raise ValueError('chosen index outside candidate menu')
        if any(not isinstance(x, (int, float)) or x != x for x in self.scores):
            raise ValueError('candidate scores must be finite numbers')

    @property
    def chosen(self):
        return self.candidates[self.chosen_index]


class CandidateReranker:
    """Score candidate JSON continuations with one prepared request."""
    def __init__(self, session, tokenizer):
        self.session, self.tokenizer = session, tokenizer

    def select(self, *, prompt_ids, prompt_mask, candidates: Sequence[dict], query,
               scope, task, task_features, task_digest: str, revision: int,
               selection) -> CandidateSelection:
        import torch
        candidates = tuple(candidates)
        if not candidates or len(candidates) > MAX_CANDIDATES:
            raise ValueError('candidate menu size outside bounds')
        if prompt_ids.ndim != 2 or prompt_ids.shape[0] != 1:
            raise ValueError('prompt_ids must be [1, tokens]')
        if prompt_ids.shape[1] > MAX_PROMPT_TOKENS:
            raise ValueError('candidate prompt exceeds bounded size')
        request = self.session.prepare(
            prompt_ids=prompt_ids, prompt_mask=prompt_mask, query=query, scope=scope,
            task=task, selection=selection, task_features=task_features,
            task_digest=task_digest, revision=revision, previous_core=None,
            core_impl='recurrent', enabled_branches=('memory', 'workspace'))
        scores = []
        try:
            with torch.no_grad():
                for candidate in candidates:
                    target = self.tokenizer(canonical_action(candidate),
                                            add_special_tokens=False)['input_ids']
                    target = list(target) + [self.tokenizer.eos_token_id]
                    if len(target) > MAX_TARGET_TOKENS:
                        raise ValueError('candidate target exceeds bounded size')
                    target_ids = torch.tensor([target], dtype=torch.long,
                                              device=prompt_ids.device)
                    all_ids = torch.cat((prompt_ids, target_ids), dim=1)
                    mask = torch.ones_like(all_ids, dtype=torch.bool)
                    labels = all_ids.clone()
                    labels[:, :prompt_ids.shape[1]] = -100
                    output = self.session.forward(request, all_ids, labels=labels,
                                                  attention_mask=mask)
                    score = float(output.loss.detach())
                    if score != score:
                        raise ValueError('nonfinite candidate score')
                    scores.append(score)
        finally:
            request.release()
        chosen = min(range(len(candidates)), key=lambda index: (scores[index], index))
        return CandidateSelection(candidates, tuple(scores), chosen)

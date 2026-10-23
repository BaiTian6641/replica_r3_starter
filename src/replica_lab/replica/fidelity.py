"""C16 fidelity scoring harness; fictional contracts only.

This module never claims personal fidelity. It scores explicitly marked fictional
cases and keeps factual provenance, conditional choice, clarification, unknown
handling, technical correctness, style and branch divergence separate. Authentic
source review remains external.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

BASELINES = ('generic', 'approved_summary', 'archive_retrieval')


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{name} must be nonempty text')
    return value


@dataclass(frozen=True)
class FidelityCase:
    case_id: str
    family: str
    prompt: str
    required_evidence_ids: tuple[str, ...]
    expected_choice: str | None = None
    expected_unknown: bool = False
    requires_clarification: bool = False
    technical_expected: bool | None = None
    style_tag: str | None = None
    provenance: str = 'fictional'

    def __post_init__(self):
        for name in ('case_id', 'family', 'prompt'):
            _text(getattr(self, name), name)
        if self.provenance != 'fictional':
            raise ValueError('this harness accepts fictional cases only')
        if not isinstance(self.required_evidence_ids, tuple):
            raise ValueError('evidence IDs must be a tuple')
        if self.expected_choice is not None:
            _text(self.expected_choice, 'expected_choice')
        if self.requires_clarification and self.expected_unknown:
            raise ValueError('clarification and unknown expectations are distinct')


@dataclass(frozen=True)
class FidelityOutput:
    case_id: str
    response: str
    evidence_ids: tuple[str, ...]
    choice: str | None = None
    clarification: str | None = None
    technical_passed: bool | None = None
    style_tags: tuple[str, ...] = ()
    branch_event_ids: tuple[str, ...] = ()

    def __post_init__(self):
        _text(self.case_id, 'case_id')
        if not isinstance(self.response, str):
            raise ValueError('response must be text')
        for name in ('evidence_ids', 'style_tags', 'branch_event_ids'):
            value = getattr(self, name)
            if not isinstance(value, tuple) or any(not isinstance(x, str) or not x for x in value):
                raise ValueError(f'{name} must be nonempty text')
        if self.choice is not None:
            _text(self.choice, 'choice')
        if self.clarification is not None:
            _text(self.clarification, 'clarification')


def score_case(case: FidelityCase, output: FidelityOutput) -> dict:
    if case.case_id != output.case_id:
        raise ValueError('case/output identity mismatch')
    cited = set(output.evidence_ids)
    required = set(case.required_evidence_ids)
    provenance = len(required & cited) / len(required) if required else 1.0
    choice = None if case.expected_choice is None else output.choice == case.expected_choice
    clarified = bool(output.clarification)
    unknown = ('unknown' in output.response.casefold() or 'do not know' in output.response.casefold())
    technical = None if case.technical_expected is None else output.technical_passed == case.technical_expected
    style = None if case.style_tag is None else case.style_tag in output.style_tags
    return {
        'case_id': case.case_id, 'family': case.family,
        'provenance_coverage': provenance,
        'missing_evidence_ids': sorted(required - cited),
        'choice_correct': choice,
        'clarification_present': clarified,
        'clarification_required': case.requires_clarification,
        'unknown_labeled': unknown,
        'unknown_expected': case.expected_unknown,
        'technical_correct': technical,
        'style_match': style,
        'branch_event_count': len(set(output.branch_event_ids)),
        'response_chars': len(output.response),
        'authority': 'score_only; never declares task success or personal fidelity',
    }


def evaluate_baselines(cases: Iterable[FidelityCase], predictions: dict[str, dict[str, FidelityOutput]]) -> dict:
    cases = tuple(cases)
    unknown = set(predictions) - set(BASELINES)
    missing = set(BASELINES) - set(predictions)
    if unknown or missing:
        raise ValueError(f'baseline set differs; unknown={sorted(unknown)} missing={sorted(missing)}')
    result = {}
    for baseline in BASELINES:
        outputs = predictions[baseline]
        if set(outputs) != {c.case_id for c in cases}:
            raise ValueError(f'{baseline} cases differ')
        scored = [score_case(case, outputs[case.case_id]) for case in cases]
        result[baseline] = scored
    return {
        'format': 'replica.fidelity.baseline_report.v1',
        'baselines': result,
        'case_count': len(cases),
        'authority': 'executable scoring only; authentic human source review is external',
        'personal_fidelity_claim': False,
    }

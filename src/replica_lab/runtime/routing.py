"""Deterministic, typed routing over exact, adaptive, pattern and procedure paths."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RouteChoice:
    kind: str
    reason: str
    known_hashes: int
    total_hashes: int


def choose_route(*, case_role: str, query_hashes: frozenset,
                 trained_hashes: frozenset, exact_available: bool = False,
                 fast_available: bool = True, pattern_available: bool = True,
                 procedure_available: bool = False, correction: bool = False) -> RouteChoice:
    """Route without inspecting gold answers, confidence or generated quality text."""
    known = len(query_hashes & trained_hashes)
    total = len(query_hashes)
    if exact_available:
        return RouteChoice('record', 'eligible exact evidence', known, total)
    if correction:
        return RouteChoice('fast', 'correction requires editable state', known, total)
    if procedure_available:
        return RouteChoice('procedure', 'verified matching procedure', known, total)
    if case_role.startswith('delayed_'):
        if total and known == total and pattern_available:
            return RouteChoice('pattern', 'all delayed query hashes covered', known, total)
        if fast_available:
            return RouteChoice('fast', 'delayed query has uncovered hashes', known, total)
        if pattern_available:
            return RouteChoice('pattern', 'no adaptive state available', known, total)
    else:
        if pattern_available:
            return RouteChoice('pattern', 'general query uses static pattern path', known, total)
        if fast_available:
            return RouteChoice('fast', 'no pattern table available', known, total)
    return RouteChoice('none', 'no eligible memory or procedure', known, total)

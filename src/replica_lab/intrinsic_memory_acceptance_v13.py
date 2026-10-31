"""Successor acceptance contracts; sealed v12 scoring and receipts are unchanged.

These pure host-side contracts separate answer content from response completion,
make expected invalidation typed, and construct chronological corrections from
structured facts with an independent source-render replay.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import re

from .baseline_competence import score
from .contextual_memory import ContextualMemoryError
from .intrinsic_memory_worlds_v11 import oracle as revision_oracle

CONTROLS = frozenset(("contextual", "exact_text", "wrong", "shuffled", "zero", "absent"))


@dataclass(frozen=True)
class ResponseAssessment:
    control: str
    content_correct: bool
    complete: bool
    terminated: bool
    truncated: bool
    false_recall: bool

    @property
    def complete_correct(self) -> bool:
        return self.content_correct and self.complete

    def as_dict(self) -> dict:
        return {**asdict(self), "complete_correct": self.complete_correct}


def assess_response(expected: str, raw_answer: str, *, control: str,
                    terminated: bool, truncated: bool) -> ResponseAssessment:
    """Score content and completion independently for either generation control."""
    if control not in CONTROLS:
        raise ValueError("undeclared response control")
    if type(terminated) is not bool or type(truncated) is not bool:
        raise ValueError("termination fields must be booleans")
    if terminated and truncated:
        raise ValueError("a response cannot both terminate and be truncated")
    content_correct = score(expected, raw_answer)
    return ResponseAssessment(
        control=control, content_correct=content_correct,
        complete=terminated and not truncated,
        terminated=terminated, truncated=truncated,
        false_recall=expected == "unknown" and not content_correct)


class InvalidationContractError(ValueError):
    pass


@dataclass(frozen=True)
class ExpectedInvalidation:
    kind: str
    error_type: str
    message: str


def expect_invalidation(kind: str, operation) -> ExpectedInvalidation:
    """Return only a typed memory refusal; operational failures propagate.

    BudgetError, OSError, RuntimeError and implementation errors are deliberately
    not converted into successful stale/revoked-state evidence.
    """
    try:
        operation()
    except ContextualMemoryError as error:
        return ExpectedInvalidation(kind, type(error).__name__, str(error))
    raise InvalidationContractError(f"{kind} did not invalidate the dependent state")


_REVISION = re.compile(r"The revision ([1-9][0-9]*) recorded color of (.+?) is ([A-Za-z]+)\.")


def render_chronological(facts: list[dict]) -> str:
    """Render every fact and revision; never silently replace the old source."""
    if not facts:
        raise ValueError("chronological source requires facts")
    rendered = []
    previous = 0
    for fact in facts:
        revision = fact["revision"]
        if type(revision) is not int or revision <= previous:
            raise ValueError("revisions must be strictly increasing")
        previous = revision
        rendered.append(
            f"The revision {revision} recorded color of {fact['entity']} is {fact['color']}.")
    return " ".join(rendered) + " No color is recorded for any other entity."


def parse_chronological(text: str) -> list[dict]:
    """Independent strict reader used to replay the answer from rendered source."""
    suffix = " No color is recorded for any other entity."
    if not text.endswith(suffix):
        raise ValueError("chronological source is incomplete")
    body = text[:-len(suffix)]
    facts = []
    position = 0
    for match in _REVISION.finditer(body):
        if match.start() != position and body[position:match.start()] != " ":
            raise ValueError("unexpected chronological source text")
        facts.append({"revision": int(match.group(1)), "entity": match.group(2),
                      "color": match.group(3)})
        position = match.end()
    if not facts or body[position:] not in ("", " "):
        raise ValueError("unparsed chronological source text")
    previous = 0
    for fact in facts:
        if fact["revision"] <= previous:
            raise ValueError("revisions must be strictly increasing")
        previous = fact["revision"]
    return facts


def chronological_correction(row: dict, new_color: str) -> dict:
    """Create and independently validate one chronological structured correction.

    Original facts remain intact, only the queried entity gains a later fact,
    unrelated entity answers stay fixed, and the expected answer is replayed from
    both the structured oracle and the actual rendered source.
    """
    if not isinstance(new_color, str) or not new_color.isalpha():
        raise ValueError("invalid correction color")
    original = deepcopy(row)
    facts = deepcopy(row.get("facts"))
    query = row.get("query_entity")
    if not isinstance(facts, list) or not facts or not isinstance(query, str):
        raise ValueError("structured query facts required")
    for fact in facts:
        if (set(fact) != {"entity", "color", "revision"}
                or not all(isinstance(fact[key], str) and fact[key]
                           for key in ("entity", "color"))
                or type(fact["revision"]) is not int or fact["revision"] < 1):
            raise ValueError("invalid structured fact")
    if revision_oracle(facts, query) != row.get("answer"):
        raise ValueError("original answer does not follow from its structured oracle")
    if new_color == row["answer"]:
        raise ValueError("correction must change the intended answer")
    unrelated = sorted({fact["entity"] for fact in facts if fact["entity"] != query})
    before_unrelated = {entity: revision_oracle(facts, entity) for entity in unrelated}
    revision = 1 + max(fact["revision"] for fact in facts)
    corrected_facts = [*facts, {"entity": query, "color": new_color, "revision": revision}]
    if corrected_facts[:len(facts)] != facts:
        raise ValueError("correction modified an earlier fact")
    after_unrelated = {entity: revision_oracle(corrected_facts, entity) for entity in unrelated}
    if after_unrelated != before_unrelated:
        raise ValueError("correction changed an unrelated fact")
    answer = revision_oracle(corrected_facts, query)
    if answer != new_color:
        raise ValueError("corrected answer does not follow from the intended fact")
    text = render_chronological(corrected_facts)
    replayed_facts = parse_chronological(text)
    if replayed_facts != corrected_facts or revision_oracle(replayed_facts, query) != answer:
        raise ValueError("rendered source does not reproduce the corrected oracle answer")
    return {**original, "facts": corrected_facts, "text": text, "answer": answer,
            "chronology": revision,
            "correction": {"intended_entity": query, "supersedes_revision": revision - 1,
                           "earlier_facts_preserved": True, "unrelated_facts_unchanged": True,
                           "verification": "structured_and_rendered_source_revision_oracle"}}

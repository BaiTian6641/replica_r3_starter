"""Auditable answer-balanced updates for small training diagnostics.

Only row indices leave the sampler. Gold responses determine sampling strata;
they are never rendered into a prompt or copied into observed evidence.
Balancing answers is an explicit experiment intervention, not an evaluation
sampling policy or a substitute for held-out source-family separation.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterator, Mapping, Sequence
import random
import re
from typing import Any

from .contracts import audit_splits, fingerprint, validate_example


Row = Mapping[str, Any]


def dataset_diagnostics(rows: Sequence[Row]) -> dict[str, Any]:
    """Return exact row/answer/source counts after checking export contracts.

    ``audit_splits`` rejects repeated IDs and source-family split crossings.
    Explicit source IDs are checked as well: renaming a group must not permit
    the same observed source to cross train/dev boundaries.
    """
    if not rows:
        raise ValueError("Nonempty rows required")
    audit = audit_splits(rows)
    sources: dict[str, str] = {}
    for row in rows:
        validate_example(row, require_verified=True)
        for source in row["source_ids"]:
            if source in sources and sources[source] != row["split"]:
                raise ValueError(f"Source crosses splits: {source}")
            sources[source] = row["split"]
    answers = dict(sorted(Counter(row["response"] for row in rows).items()))
    return {
        **audit,
        "answer_histogram": answers,
        "answer_classes": len(answers),
        "answer_counts_equal": len(set(answers.values())) == 1,
        "group_ids": sorted({row["group_id"] for row in rows}),
        "source_ids": sorted(sources),
        "example_ids": [row["id"] for row in rows],
        "rows_fingerprint": fingerprint(list(rows)),
    }


def batch_diagnostics(rows: Sequence[Row], indices: Sequence[int]) -> dict[str, Any]:
    """Record the actual selected IDs and exact response histogram per update."""
    if not indices:
        raise ValueError("Nonempty update indices required")
    if any(type(index) is not int or not 0 <= index < len(rows) for index in indices):
        raise ValueError("Update index out of range")
    selected = [rows[index] for index in indices]
    histogram = dict(sorted(Counter(row["response"] for row in selected).items()))
    all_answers = {row["response"] for row in rows}
    return {
        "example_indices": list(indices),
        "example_ids": [row["id"] for row in selected],
        "answer_histogram": histogram,
        "all_answer_classes_present": set(histogram) == all_answers,
        "answer_counts_equal": len(set(histogram.values())) == 1,
    }


def balanced_batches(rows: Sequence[Row], grad_accum: int, steps: int,
                     seed: int = 17) -> Iterator[list[int]]:
    """Yield indices with exactly equal answer exposure in every update.

    ``grad_accum`` must be a positive multiple of the number of exact response
    strings. For each class, examples are visited without replacement before
    reshuffling that class. Small classes are consequently oversampled relative
    to large ones. A row can repeat within an update only when its class quota
    exceeds that class's size. Sampling uses a private RNG and never modifies
    input rows or global random state. Only verified training rows are accepted.
    """
    diagnostics = dataset_diagnostics(rows)
    if any(row["split"] != "train" for row in rows):
        raise ValueError("Balanced training batches require split=train only")
    classes = diagnostics["answer_classes"]
    if type(grad_accum) is not int or grad_accum < 1 or grad_accum % classes:
        raise ValueError("grad_accum must be a positive multiple of answer class count")
    if type(steps) is not int or steps < 0:
        raise ValueError("steps must be a nonnegative integer")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    rng = random.Random(seed)
    pools: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        pools[row["response"]].append(index)
    answers = sorted(pools)
    cursors = {answer: 0 for answer in answers}
    for answer in answers:
        rng.shuffle(pools[answer])
    per_class = grad_accum // classes
    for _ in range(steps):
        selected = []
        for answer in answers:
            for _ in range(per_class):
                if cursors[answer] == len(pools[answer]):
                    rng.shuffle(pools[answer])
                    cursors[answer] = 0
                selected.append(pools[answer][cursors[answer]])
                cursors[answer] += 1
        rng.shuffle(selected)
        yield selected


def verify_xor_fixture(row: Row) -> dict[str, Any]:
    """Recompute a shipped synthetic XOR fixture and validate its provenance.

    This strict helper intentionally accepts the project's deterministic fixture
    grammar only. A permissive integer search could accidentally bless a label
    or copied answer in the prompt, so every input field is matched in full.
    It does not claim to verify arbitrary natural-language training records.
    """
    validate_example(row, require_verified=True)
    group = re.fullmatch(r"xor-width-([1-9][0-9]*)-mask-([0-9]+)", row["group_id"])
    prompt = re.fullmatch(
        r"Input is ([0-9]+)\. Apply the fixture's bus transformation\. "
        r"Return only the decimal integer\.", row["prompt"])
    if group is None or prompt is None:
        raise ValueError("Not a supported XOR fixture grammar")
    width, mask = map(int, group.groups())
    value = int(prompt.group(1))
    if width > 64 or max(mask, value) >= 2 ** width:
        raise ValueError("XOR fixture exceeds its bounded bus width")
    expected_memory = [f"This synthetic {width}-bit bus transforms x to x XOR {mask}."]
    expected_sources = [f"fixture://xor/width-{width}/mask-{mask}"]
    if row["memory_texts"] != expected_memory or row["source_ids"] != expected_sources:
        raise ValueError("XOR fixture observed evidence/provenance differs")
    if row["id"] != f"{row['split']}-w{width}-m{mask}-x{value}":
        raise ValueError("XOR fixture ID does not match its inputs")
    # Independent per-bit arithmetic, rather than trusting the response or
    # reusing a gold-answer field as input to the solver.
    expected = sum(((((value >> bit) & 1) + ((mask >> bit) & 1)) % 2) << bit
                   for bit in range(width))
    if row["response"] != str(expected):
        raise ValueError("XOR fixture response disagrees with independent solver")
    return {"id": row["id"], "verified": True, "expected": str(expected),
            "solver": "independent_per_bit_xor", "width": width,
            "mask": mask, "input": value}

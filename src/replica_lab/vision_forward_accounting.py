"""Measured native decoder accounting for bounded vision diagnostics.

This module owns the model-free half of forward accounting.  It answers a
narrow question with evidence rather than inference: **how many physical
native decoder passes did a readout actually execute, in which phase, and what
did each pass process?**

The historical v1 curve receipt declared ``3 * 98 = 294`` "scoring forwards"
per readout and ``1,470`` in total.  That number is real, but its unit was
``prepare/sense pass``: the generation loop, explicit prefill logits and the
teacher-forced loss forward were never counted.  The preserved reconstruction
is 7,695 readout passes plus 60 training/probe events (7,755 total).

Design rules enforced here:

* one tap on the adapter's private ``_native`` is the only physical-forward
  source of truth; there are no forward/backward hooks and no counter storage
  inside graph hooks;
* every native pass must occur inside a declared accounting phase, otherwise
  the run fails closed;
* progress is counted at call time against an explicit unit, with separate
  per-readout and cumulative budgets;
* logical scoring cells/operations are counted in their own units and are
  never charged as decoder passes;
* wrappers restore the stored bound methods even when the native call raises;
* wall checks occur inside the wrapper, not only once per row or readout.
"""
from __future__ import annotations

from contextlib import contextmanager
import math
import time
from typing import Any, Callable, Mapping, Sequence


VERSION = "replica.vision_forward_accounting.v2"
ACCOUNTING_FORMAT = "replica.vision_forward_accounting.snapshot.v2"
BUDGET_FORMAT = "replica.vision_forward_accounting.budget.v2"
RECONSTRUCTION_FORMAT = "replica.vision_forward_accounting.reconstruction.v2"
HISTORICAL_FORMAT = "replica.vision_forward_accounting.historical_correction.v2"

DECODER_UNIT = "decoder_forward_call"
PREPARE_UNIT = "prepare_sense_pass"
LOGITS_UNIT = "explicit_logits_forward"
PREFILL_UNIT = "generation_prefill_call"
DECODE_UNIT = "generation_decode_call"
GENERATED_UNIT = "generated_token"
TEXT_POSITION_UNIT = "processed_text_position"
VISUAL_POSITION_UNIT = "processed_visual_position"
VISUAL_OBSERVATION_UNIT = "visual_observation_call"
ENCODER_UNIT = "visual_encoder_call"
CACHE_HIT_UNIT = "visual_encoder_cache_hit"
CACHE_MISS_UNIT = "visual_encoder_cache_miss"
PROJECTOR_UNIT = "projector_pass"
BACKWARD_UNIT = "backward_call"
OPTIMIZER_UNIT = "optimizer_step"
LOGICAL_CELL_UNIT = "logical_scoring_cell"
LOGICAL_OPERATION_UNIT = "logical_scoring_operation"

PHASE_PREPARE = "prepare_sense"
PHASE_LOGITS = "explicit_logits_loss"
PHASE_GENERATION = "generation"
PHASE_TRAINING = "training"
PHASE_PROBE = "probe"
PHASES = (PHASE_PREPARE, PHASE_LOGITS, PHASE_GENERATION, PHASE_TRAINING,
          PHASE_PROBE)
FORWARD_PHASES = PHASES

COUNTER_KEYS = (
    "decoder_forward_calls",
    "prepare_sense_passes",
    "explicit_logits_forwards",
    "generation_prefill_calls",
    "generation_decode_calls",
    "generated_tokens",
    "processed_text_positions",
    "processed_visual_positions",
    "visual_forward_calls",
    "projector_passes",
    "visual_observation_calls",
    "visual_encoder_calls",
    "visual_encoder_cache_hits",
    "visual_encoder_cache_misses",
    "backward_calls",
    "optimizer_steps",
    "logical_scoring_cells",
    "logical_scoring_operations",
)

UNIT_BY_COUNTER = {
    "decoder_forward_calls": DECODER_UNIT,
    "prepare_sense_passes": PREPARE_UNIT,
    "explicit_logits_forwards": LOGITS_UNIT,
    "generation_prefill_calls": PREFILL_UNIT,
    "generation_decode_calls": DECODE_UNIT,
    "generated_tokens": GENERATED_UNIT,
    "processed_text_positions": TEXT_POSITION_UNIT,
    "processed_visual_positions": VISUAL_POSITION_UNIT,
    "visual_forward_calls": "visual_forward_call",
    "projector_passes": PROJECTOR_UNIT,
    "visual_observation_calls": VISUAL_OBSERVATION_UNIT,
    "visual_encoder_calls": ENCODER_UNIT,
    "visual_encoder_cache_hits": CACHE_HIT_UNIT,
    "visual_encoder_cache_misses": CACHE_MISS_UNIT,
    "backward_calls": BACKWARD_UNIT,
    "optimizer_steps": OPTIMIZER_UNIT,
    "logical_scoring_cells": LOGICAL_CELL_UNIT,
    "logical_scoring_operations": LOGICAL_OPERATION_UNIT,
}

# Counters that the native tap increments itself.  Budgets may declare a cap
# for any COUNTER_KEYS entry, but declarations must use these exact units.
NATIVE_COUNTERS = (
    "decoder_forward_calls",
    "prepare_sense_passes",
    "explicit_logits_forwards",
    "generation_prefill_calls",
    "generation_decode_calls",
    "generated_tokens",
    "processed_text_positions",
    "processed_visual_positions",
    "visual_forward_calls",
    "projector_passes",
)

HISTORICAL_PREFIXES = (0, 1, 2, 4, 7)
HISTORICAL_GENERATION_LENGTHS = (1362, 1442, 1396, 721, 324)
HISTORICAL_RECORD_TOKEN_SUMS = (1362, 1442, 1396, 721, 324)
HISTORICAL_AVAILABLE_PER_READOUT = 98
HISTORICAL_PREPARES_PER_CELL = 3
HISTORICAL_PREFILL_PER_CELL = 1
HISTORICAL_TEACHER_FORCED_PER_CELL = 1
HISTORICAL_UPDATE_FORWARDS = 28
HISTORICAL_PROBE_FORWARDS = 2
HISTORICAL_UPDATE_BACKWARDS = 28
HISTORICAL_PROBE_BACKWARDS = 2
HISTORICAL_SCORING_PASSES = (1852, 1932, 1886, 1211, 814)
HISTORICAL_TOTAL_SCORING_PASSES = 7695
HISTORICAL_TRAINING_PROBE_EVENTS = 60
HISTORICAL_GRAND_TOTAL = 7755

_MISSING = object()


class AccountingError(RuntimeError):
    """An accounting declaration, phase or measured invariant was violated."""


def _require(condition: Any, message: str) -> None:
    if not condition:
        raise AccountingError(message)


def _nonnegative_int(value: Any, name: str) -> int:
    _require(type(value) is int and value >= 0,
             f"{name} must be a nonnegative integer")
    return value


def _sum(values: Sequence[int]) -> int:
    return sum(int(value) for value in values)


def _empty_counters() -> dict[str, int]:
    return {key: 0 for key in COUNTER_KEYS}


class ForwardBudget:
    """Per-readout and cumulative caps with call-time enforcement.

    Caps are keyed by measured counter name and each declaration carries an
    explicit unit.  A prospective increment is rejected *before* the physical
    call is made, so the first excess forward raises instead of being noticed
    after a readout or after the whole run.
    """

    def __init__(self, declaration: Mapping[str, Any], *,
                 clock: Callable[[], float] = time.monotonic):
        self.declaration = validate_budget_declaration(declaration)
        self._clock = clock
        self._readout_counters: dict[str, int] = {}
        self._cumulative_total: dict[str, int] = {key: 0
                                                  for key in COUNTER_KEYS}
        self._readout_prefix: int | None = None
        self._started = clock()
        self._calls_since_wall = 0

    # -- lifecycle ---------------------------------------------------------
    def begin_readout(self, prefix: int) -> None:
        _require(type(prefix) is int and prefix >= 0,
                 "Readout prefix must be a nonnegative integer")
        self._readout_prefix = prefix
        self._readout_counters = {key: 0 for key in COUNTER_KEYS}

    def end_readout(self) -> dict[str, int]:
        _require(self._readout_counters,
                 "Physical accounting requires begin_readout(prefix) first")
        return dict(self._readout_counters)

    def finish_readout(self) -> None:
        self._readout_counters = {}

    @property
    def readout_prefix(self) -> int | None:
        return self._readout_prefix

    # -- enforcement -------------------------------------------------------
    def _wall(self) -> None:
        elapsed = self._clock() - self._started
        limit = self.declaration["wall_limit_seconds"]
        if elapsed > float(limit):
            raise AccountingError(
                f"Wall budget exceeded inside accounting phase: "
                f"{elapsed:.3f}s > {limit}s")

    def _maybe_wall(self) -> None:
        self._calls_since_wall += 1
        if self._calls_since_wall >= int(
                self.declaration["wall_check_every_calls"]):
            self._calls_since_wall = 0
            self._wall()

    def check_forward(self, *, phase: str, text_positions: int,
                      visual_tokens: int, generation_index: int | None = None,
                      generation_sequence: int | None = None) -> None:
        """Validate a prospective native pass before it executes."""
        _require(phase in FORWARD_PHASES,
                 f"Native forward outside a declared phase: {phase!r}")
        _nonnegative_int(text_positions, "text_positions")
        _nonnegative_int(visual_tokens, "visual_tokens")
        _require(text_positions > 0, "Native forward must process text tokens")
        _require(visual_tokens in (0, 196),
                 "Visual tokens must be zero or the declared 196 patches")
        if phase == PHASE_GENERATION:
            _require(type(generation_index) is int and generation_index >= 0,
                     "Generation accounting requires a zero-based call index")
            _require(type(generation_sequence) is int
                     and generation_sequence >= 0,
                     "Generation accounting requires a sequence index")
            _require(visual_tokens in (0, 196),
                     "Generation visual tokens are malformed")
        increments: dict[str, int] = {
            "decoder_forward_calls": 1,
            "processed_text_positions": int(text_positions),
            "processed_visual_positions": int(visual_tokens),
            "visual_forward_calls": 1 if visual_tokens else 0,
            "projector_passes": 1 if visual_tokens else 0,
        }
        if phase == PHASE_PREPARE:
            increments["prepare_sense_passes"] = 1
        elif phase == PHASE_LOGITS:
            increments["explicit_logits_forwards"] = 1
        elif phase == PHASE_GENERATION:
            if generation_index == 0:
                increments["generation_prefill_calls"] = 1
            else:
                increments["generation_decode_calls"] = 1
            increments["generated_tokens"] = 1
        for key, delta in increments.items():
            if key in self.declaration["per_readout"]:
                cap = int(self.declaration["per_readout"][key])
                if self._readout_counters:
                    prospective = self._readout_counters[key] + int(delta)
                    _require(prospective <= cap,
                             f"Per-readout budget exceeded at call time: {key} "
                             f"current={self._readout_counters[key]} "
                             f"attempt={prospective} cap={cap} "
                             f"(prefix {self._readout_prefix})")
                cumulative = self._cumulative_total[key] + int(delta)
                total_cap = int(self.declaration["cumulative"].get(key, cap))
                _require(cumulative <= total_cap,
                         f"Cumulative budget exceeded at call time: {key} "
                         f"current={self._cumulative_total[key]} "
                         f"attempt={cumulative} cap={total_cap}")
        self._maybe_wall()

    def commit_forward(self, increments: Mapping[str, int]) -> None:
        """Record the checked increment after the native call returns."""
        for key, delta in increments.items():
            _require(key in COUNTER_KEYS, f"Unknown measured counter: {key}")
            if self._readout_counters:
                self._readout_counters[key] += int(delta)
            self._cumulative_total[key] += int(delta)

    def check_event(self, key: str, amount: int = 1) -> None:
        _require(key in {"backward_calls", "optimizer_steps",
                         "visual_observation_calls", "visual_encoder_calls",
                         "visual_encoder_cache_hits",
                         "visual_encoder_cache_misses",
                         "logical_scoring_cells",
                         "logical_scoring_operations"},
                 f"Unknown event counter: {key}")
        _nonnegative_int(amount, key)
        if key in self.declaration["per_readout"]:
            cap = int(self.declaration["per_readout"][key])
            if self._readout_counters:
                prospective = self._readout_counters[key] + amount
                _require(prospective <= cap,
                         f"Per-readout budget exceeded at call time: {key} "
                         f"current={self._readout_counters[key]} "
                         f"attempt={prospective} cap={cap}")
            total_cap = int(self.declaration["cumulative"].get(key, cap))
            _require(self._cumulative_total[key] + amount <= total_cap,
                     f"Cumulative budget exceeded at call time: {key} "
                     f"current={self._cumulative_total[key]} "
                     f"attempt={self._cumulative_total[key] + amount} "
                     f"cap={total_cap}")
        self._maybe_wall()

    def commit_event(self, key: str, amount: int = 1) -> None:
        _require(key in COUNTER_KEYS, f"Unknown measured counter: {key}")
        if self._readout_counters:
            self._readout_counters[key] += amount
        self._cumulative_total[key] += amount

    def check_wall(self) -> None:
        self._wall()


class ForwardAccounting:
    """Phase-aware tap for real native forwards, backwards and optimizer steps."""

    def __init__(self, *, budget: ForwardBudget | None = None):
        self.budget = budget
        self._phase_stack: list[str] = []
        self._forced_phase: str | None = None
        self._readouts: list[dict[str, Any]] = []
        self._cumulative_total: dict[str, int] = {key: 0
                                                  for key in COUNTER_KEYS}
        self._counters: dict[str, int] = {key: 0 for key in COUNTER_KEYS}
        self._generation_sequence = 0
        self._generation_index: int | None = None
        self._readout_prefix: int | None = None
        self._phase_counts: dict[str, dict[str, int]] = {
            name: {
                "forward_calls": 0,
                "text_positions": 0,
                "visual_positions": 0,
                "backward_calls": 0,
            } for name in PHASES
        }
        self._totals_phase_counts: dict[str, dict[str, int]] = {
            name: {
                "forward_calls": 0,
                "text_positions": 0,
                "visual_positions": 0,
                "backward_calls": 0,
            } for name in PHASES
        }
        self._wrappers_restored = 0
        self._native_calls = 0

    # -- phases ------------------------------------------------------------
    @property
    def phase_name(self) -> str | None:
        if self._forced_phase is not None:
            return self._forced_phase
        return self._phase_stack[-1] if self._phase_stack else None

    @contextmanager
    def phase(self, name: str):
        _require(name in PHASES, f"Unknown accounting phase: {name!r}")
        self._phase_stack.append(name)
        try:
            yield
        finally:
            _require(self._phase_stack and self._phase_stack[-1] == name,
                     "Accounting phase stack corrupted")
            self._phase_stack.pop()

    @contextmanager
    def forced_phase(self, name: str):
        """Override session-level phase labels for a nested operation."""
        _require(name in PHASES, f"Unknown accounting phase: {name!r}")
        previous = self._forced_phase
        self._forced_phase = name
        try:
            yield
        finally:
            self._forced_phase = previous

    @contextmanager
    def generation_scope(self):
        _require(self._generation_index is None,
                 "Generation scopes may not nest")
        self._generation_sequence += 1
        self._generation_index = 0
        try:
            yield
        finally:
            self._generation_index = None

    def begin_readout(self, prefix: int) -> None:
        _require(self._readout_prefix is None,
                 "Finish the previous accounting readout first")
        self._readout_prefix = int(prefix)
        self._counters = {key: 0 for key in COUNTER_KEYS}
        self._phase_counts = {
            name: {
                "forward_calls": 0,
                "text_positions": 0,
                "visual_positions": 0,
                "backward_calls": 0,
            } for name in PHASES
        }
        if self.budget is not None:
            self.budget.begin_readout(int(prefix))

    def end_readout(self) -> dict[str, Any]:
        _require(self._readout_prefix is not None,
                 "No accounting readout is open")
        if self.budget is not None:
            self.budget.end_readout()
        snapshot = self.snapshot()
        snapshot["prefix"] = int(self._readout_prefix)
        self._readouts.append(snapshot)
        self._readout_prefix = None
        self._counters = {key: 0 for key in COUNTER_KEYS}
        self._phase_counts = {
            name: {
                "forward_calls": 0,
                "text_positions": 0,
                "visual_positions": 0,
                "backward_calls": 0,
            } for name in PHASES
        }
        if self.budget is not None:
            self.budget.end_readout()
            self.budget.finish_readout()
        return snapshot

    # -- recording ---------------------------------------------------------
    def _increments(self, *, phase: str, text_positions: int,
                    visual_tokens: int, generation_index: int | None) -> dict:
        increments = {
            "decoder_forward_calls": 1,
            "processed_text_positions": int(text_positions),
            "processed_visual_positions": int(visual_tokens),
            "visual_forward_calls": 1 if visual_tokens else 0,
            "projector_passes": 1 if visual_tokens else 0,
        }
        if phase == PHASE_PREPARE:
            increments["prepare_sense_passes"] = 1
        elif phase == PHASE_LOGITS:
            increments["explicit_logits_forwards"] = 1
        elif phase == PHASE_GENERATION:
            _require(generation_index is not None,
                     "Generation forward requires a generation call index")
            if int(generation_index) == 0:
                increments["generation_prefill_calls"] = 1
            else:
                increments["generation_decode_calls"] = 1
            increments["generated_tokens"] = 1
        return increments

    def record_forward(self, *, ids: Any, visual: Any = None,
                       phase: str | None = None) -> None:
        """Record one actual native forward."""
        selected = self.phase_name if phase is None else phase
        _require(selected in FORWARD_PHASES,
                 "Native forward outside a declared accounting phase")
        shape = getattr(ids, "shape", None)
        _require(shape is not None and len(shape) == 2,
                 "Native forward IDs must be [batch,tokens]")
        text_positions = int(shape[1])
        visual_tokens = 0
        if visual is not None:
            features = getattr(visual, "features", None)
            _require(features is not None and getattr(features, "ndim", 0) == 3,
                     "Visual input has no [1,patches,width] features")
            visual_tokens = int(features.shape[1])
        generation_index = None
        if selected == PHASE_GENERATION:
            _require(self._generation_index is not None,
                     "Generation forward outside generation_scope()")
            generation_index = self._generation_index
        increments = self._increments(
            phase=selected, text_positions=text_positions,
            visual_tokens=visual_tokens, generation_index=generation_index)
        if self.budget is not None:
            self.budget.check_forward(
                phase=selected, text_positions=text_positions,
                visual_tokens=visual_tokens,
                generation_index=generation_index,
                generation_sequence=(self._generation_sequence
                                     if selected == PHASE_GENERATION else None))
            self.budget.commit_forward(increments)
        self._apply(increments)
        phase_counts = self._phase_counts[selected]
        phase_counts["forward_calls"] += 1
        phase_counts["text_positions"] += int(text_positions)
        phase_counts["visual_positions"] += int(visual_tokens)
        global_counts = self._totals_phase_counts[selected]
        global_counts["forward_calls"] += 1
        global_counts["text_positions"] += int(text_positions)
        global_counts["visual_positions"] += int(visual_tokens)
        self._native_calls += 1
        if selected == PHASE_GENERATION:
            self._generation_index += 1

    def _apply(self, increments: Mapping[str, int]) -> None:
        for key, delta in increments.items():
            _require(key in COUNTER_KEYS, f"Unknown measured counter: {key}")
            self._counters[key] += int(delta)
            self._cumulative_total[key] += int(delta)

    def record_backward(self, *, phase: str | None = None) -> None:
        selected = self.phase_name if phase is None else phase
        _require(selected in PHASES,
                 "Backward outside a declared accounting phase")
        self._event("backward_calls")
        if selected in self._phase_counts:
            self._phase_counts[selected]["backward_calls"] += 1
            self._totals_phase_counts[selected]["backward_calls"] += 1

    def record_optimizer_step(self) -> None:
        self._event("optimizer_steps")

    def record_logical(self, *, cells: int = 0, operations: int = 0) -> None:
        _nonnegative_int(cells, "logical cells")
        _nonnegative_int(operations, "logical operations")
        if cells:
            if self.budget is not None:
                self.budget.check_event("logical_scoring_cells", cells)
                self.budget.commit_event("logical_scoring_cells", cells)
            self._apply({"logical_scoring_cells": cells})
        if operations:
            if self.budget is not None:
                self.budget.check_event("logical_scoring_operations",
                                        operations)
                self.budget.commit_event("logical_scoring_operations",
                                         operations)
            self._apply({"logical_scoring_operations": operations})

    def _event(self, key: str) -> None:
        if self.budget is not None:
            self.budget.check_event(key, 1)
            self.budget.commit_event(key, 1)
        self._apply({key: 1})

    # -- taps --------------------------------------------------------------
    @contextmanager
    def wrap_native(self, adapter: Any):
        """Tap ``adapter._native`` exactly once and restore it on any exit."""
        stored = vars(adapter).get("_native", _MISSING)
        original = getattr(adapter, "_native")
        _require(callable(original), "Adapter exposes no callable _native")
        accounting = self

        def wrapper(*args, **kwargs):
            ids = args[0] if args else kwargs.get("ids")
            if ids is None:
                ids = kwargs.get("input_ids")
            visual = kwargs.get("visual")
            frame = kwargs.get("frame")
            if visual is None and frame is not None:
                visual = getattr(getattr(frame, "sense", None), "visual", None)
            accounting.record_forward(ids=ids, visual=visual)
            return original(*args, **kwargs)

        setattr(adapter, "_native", wrapper)
        try:
            yield wrapper
        finally:
            if stored is _MISSING:
                try:
                    delattr(adapter, "_native")
                except AttributeError:
                    pass
            else:
                setattr(adapter, "_native", stored)
            self._wrappers_restored += 1

    @contextmanager
    def wrap_feature_service(self, service: Any):
        """Count observe calls and encoder cache hits/misses per observation."""
        original = service.observe
        audit = getattr(service, "cache_audit", None)
        _require(callable(audit) and callable(original),
                 "Feature service exposes no observe/cache_audit")
        accounting = self

        def wrapper(*args, **kwargs):
            before = audit()
            result = original(*args, **kwargs)
            after = audit()
            hits = int(after.get("hits", 0)) - int(before.get("hits", 0))
            misses = int(after.get("misses", 0)) - int(before.get("misses", 0))
            _require(hits >= 0 and misses >= 0 and hits + misses == 1,
                     "Feature observation must be exactly one cache hit or miss")
            for key, value in (("visual_observation_calls", 1),
                               ("visual_encoder_cache_hits", hits),
                               ("visual_encoder_cache_misses", misses),
                               ("visual_encoder_calls", misses)):
                if accounting.budget is not None:
                    accounting.budget.check_event(key, value)
                    accounting.budget.commit_event(key, value)
                accounting._apply({key: value})
            return result

        setattr(service, "observe", wrapper)
        try:
            yield wrapper
        finally:
            setattr(service, "observe", original)
            self._wrappers_restored += 1

    @contextmanager
    def measured_session(self, session: Any):
        """Label prepare/forward/generate/backward on an existing session.

        ``forced_phase`` around this context overrides the default labels (for
        example all update passes are labelled ``training``).
        """
        stored = {name: vars(session).get(name, _MISSING)
                  for name in ("prepare", "forward", "generate", "backward")}
        originals = {name: getattr(session, name) for name in stored}
        accounting = self

        def prepare(*args, **kwargs):
            if accounting.phase_name is not None:
                return originals["prepare"](*args, **kwargs)
            with accounting.phase(PHASE_PREPARE):
                return originals["prepare"](*args, **kwargs)

        def forward(*args, **kwargs):
            if accounting.phase_name is not None:
                return originals["forward"](*args, **kwargs)
            with accounting.phase(PHASE_LOGITS):
                return originals["forward"](*args, **kwargs)

        def generate(*args, **kwargs):
            with accounting.generation_scope():
                with accounting.phase(PHASE_GENERATION):
                    return originals["generate"](*args, **kwargs)

        def backward(*args, **kwargs):
            result = originals["backward"](*args, **kwargs)
            accounting.record_backward()
            return result

        try:
            setattr(session, "prepare", prepare)
            setattr(session, "forward", forward)
            setattr(session, "generate", generate)
            setattr(session, "backward", backward)
            yield session
        finally:
            for name, previous in stored.items():
                if previous is _MISSING:
                    try:
                        delattr(session, name)
                    except AttributeError:
                        pass
                else:
                    setattr(session, name, previous)
            self._wrappers_restored += 4

    @contextmanager
    def counted_optimizer_step(self, optimizer: Any):
        original = optimizer.step
        accounting = self

        def wrapper(*args, **kwargs):
            result = original(*args, **kwargs)
            accounting.record_optimizer_step()
            return result

        setattr(optimizer, "step", wrapper)
        try:
            yield optimizer
        finally:
            setattr(optimizer, "step", original)
            self._wrappers_restored += 1

    # -- snapshots ---------------------------------------------------------
    def snapshot(self) -> dict[str, Any]:
        counters = dict(self._counters)
        phase_sum = sum(int(values["forward_calls"])
                        for values in self._phase_counts.values())
        _require(int(counters["decoder_forward_calls"]) == phase_sum,
                 "Decoder passes do not equal the measured phase sum")
        _require(int(counters["generated_tokens"])
                 == int(counters["generation_prefill_calls"])
                 + int(counters["generation_decode_calls"]),
                 "Generated tokens do not equal prefill plus decode calls")
        _require(int(counters["visual_encoder_calls"])
                 == int(counters["visual_encoder_cache_misses"]),
                 "Visual encoder calls must equal cache misses")
        _require(int(counters["visual_observation_calls"])
                 == int(counters["visual_encoder_cache_hits"])
                 + int(counters["visual_encoder_cache_misses"]),
                 "Visual observations must equal cache hits plus misses")
        _require(int(counters["processed_visual_positions"])
                 == 196 * int(counters["visual_forward_calls"]),
                 "Visual positions must equal 196 per visual-bearing pass")
        _require(int(counters["projector_passes"])
                 == int(counters["visual_forward_calls"]),
                 "Projector passes must equal visual-bearing passes")
        return {
            "format": ACCOUNTING_FORMAT,
            "version": VERSION,
            "counters": counters,
            "units": {key: UNIT_BY_COUNTER[key] for key in COUNTER_KEYS},
            "phases": {
                name: {
                    **{field: int(value)
                       for field, value in self._phase_counts[name].items()},
                    "phase": name,
                }
                for name in PHASES
            },
            "native_calls_observed": int(self._native_calls),
            "wrappers_restored": int(self._wrappers_restored),
            "logical": {
                "scoring_cells": int(counters["logical_scoring_cells"]),
                "scoring_operations":
                    int(counters["logical_scoring_operations"]),
                "physical_decoder_passes":
                    int(counters["decoder_forward_calls"]),
            },
            "readouts_completed": len(self._readouts),
            "scope": ("measured native taps only; logical cells/operations "
                      "are separate units and never charged as forwards"),
        }

    def historical_totals(self) -> dict[str, Any]:
        return {
            "format": ACCOUNTING_FORMAT,
            "version": VERSION,
            "counters": dict(self._cumulative_total),
            "phase_totals": {
                name: {**{field: int(value)
                          for field, value in self._totals_phase_counts[
                              name].items()},
                       "phase": name}
                for name in PHASES
            },
            "readouts": list(self._readouts),
            "units": {key: UNIT_BY_COUNTER[key] for key in COUNTER_KEYS},
        }


def validate_budget_declaration(declaration: Mapping[str, Any], *,
                                available_per_readout: int | None = None,
                                expected_readouts: int | None = None
                                ) -> dict[str, Any]:
    """Validate an explicit-unit, call-time budget declaration.

    The historical error is rejected structurally: a total decoder cap equal
    to three prepare passes per available cell cannot be declared, because a
    scored cell necessarily includes generation and explicit logits passes in
    addition to its three prepare/sense passes.
    """
    _require(isinstance(declaration, Mapping),
             "Budget declaration must be a mapping")
    _require(declaration.get("format") == BUDGET_FORMAT,
             "Budget format differs")
    _require(set(declaration) == {
        "format", "units", "per_readout", "cumulative",
        "wall_limit_seconds", "wall_check_every_calls",
    }, f"Budget fields differ: {sorted(declaration)}")
    units = declaration["units"]
    _require(isinstance(units, Mapping) and units,
             "Budget units are required")
    for key, unit in units.items():
        _require(key in UNIT_BY_COUNTER,
                 f"Unknown budget counter: {key}")
        _require(unit == UNIT_BY_COUNTER[key],
                 f"Budget unit differs for {key}: {unit!r}")
    for section in ("per_readout", "cumulative"):
        caps = declaration[section]
        _require(isinstance(caps, Mapping),
                 f"Budget {section} caps must be a mapping")
        for key, cap in caps.items():
            _require(key in UNIT_BY_COUNTER,
                     f"Unknown budget counter: {key}")
            _nonnegative_int(cap, f"{section}.{key}")
            _require(cap > 0, f"{section}.{key} cap must be positive")
        _require(set(caps) <= set(units),
                 f"Budget {section} has a counter without a unit")
    for key in declaration["per_readout"]:
        _require(key in declaration["cumulative"],
                 f"Budget lacks a cumulative cap for {key}")
        _require(declaration["cumulative"][key]
                 >= declaration["per_readout"][key],
                 f"Cumulative cap below per-readout cap for {key}")
    wall = declaration["wall_limit_seconds"]
    _require(isinstance(wall, (int, float)) and not isinstance(wall, bool)
             and math.isfinite(wall) and wall > 0,
             "Wall limit must be a positive finite number")
    every = declaration["wall_check_every_calls"]
    _require(type(every) is int and every > 0,
             "wall_check_every_calls must be a positive integer")
    if available_per_readout is not None:
        _nonnegative_int(available_per_readout, "available_per_readout")
        if "decoder_forward_calls" in declaration["per_readout"]:
            cap = int(declaration["per_readout"]["decoder_forward_calls"])
            minimum = 6 * available_per_readout
            _require(cap >= minimum,
                     "The decoder cap cannot be three prepare passes per "
                     "cell; it must cover prepares, explicit logits and "
                     "generation")
            _require(cap != 3 * available_per_readout,
                     "Three prepare calls cannot be declared as total "
                     "scoring forwards")
        if "prepare_sense_passes" in declaration["per_readout"]:
            _require(int(declaration["per_readout"]
                         ["prepare_sense_passes"])
                     == 3 * available_per_readout,
                     "A full-control readout requires exactly three "
                     "prepare/sense passes per available cell")
        if "explicit_logits_forwards" in declaration["per_readout"]:
            _require(int(declaration["per_readout"]
                         ["explicit_logits_forwards"])
                     == 2 * available_per_readout,
                     "A full-control readout requires prefill and "
                     "teacher-forced logits per available cell")
        if "generation_prefill_calls" in declaration["per_readout"]:
            _require(int(declaration["per_readout"]
                         ["generation_prefill_calls"])
                     <= available_per_readout,
                     "Generation prefill calls cannot exceed cells")
    if expected_readouts is not None:
        _nonnegative_int(expected_readouts, "expected_readouts")
        for key, cap in declaration["cumulative"].items():
            if key not in declaration["per_readout"]:
                continue
            per = int(declaration["per_readout"][key])
            _require(int(cap) >= per,
                     f"Cumulative cap below one readout for {key}")
            if key in {"prepare_sense_passes", "explicit_logits_forwards"}:
                _require(int(cap) == per * expected_readouts,
                         f"Deterministic counter {key} must have an exact "
                         f"cumulative cap")
    return dict(declaration)


def reconstruct_scoring_accounting(
        generation_lengths: Sequence[int], *,
        prefixes: Sequence[int] = HISTORICAL_PREFIXES,
        available_per_readout: int = HISTORICAL_AVAILABLE_PER_READOUT,
        prepares_per_cell: int = HISTORICAL_PREPARES_PER_CELL,
        prefill_per_cell: int = HISTORICAL_PREFILL_PER_CELL,
        teacher_forced_per_cell: int = HISTORICAL_TEACHER_FORCED_PER_CELL,
        update_forwards: int = HISTORICAL_UPDATE_FORWARDS,
        probe_forwards: int = HISTORICAL_PROBE_FORWARDS,
        update_backwards: int = HISTORICAL_UPDATE_BACKWARDS,
        probe_backwards: int = HISTORICAL_PROBE_BACKWARDS,
        reconstruction: str = "reconstructed_from_preserved_receipts",
        measured_by_old_run: bool = False,
) -> dict[str, Any]:
    """Reconstruct physical passes from preserved cells and generation lengths."""
    lengths = [int(value) for value in generation_lengths]
    _require(len(lengths) == len(prefixes),
             "Generation lengths must cover every prefix")
    _require(all(value >= 0 for value in lengths),
             "Generation lengths must be nonnegative")
    per_readout: list[dict[str, Any]] = []
    total_generated = 0
    total_prepares = 0
    total_prefill = 0
    total_teacher = 0
    total_scoring = 0
    for prefix, generated in zip(prefixes, lengths):
        prepares = prepares_per_cell * available_per_readout
        prefill = prefill_per_cell * available_per_readout
        teacher = teacher_forced_per_cell * available_per_readout
        # Prepares occur once for generation, once for prefill and once for
        # the teacher-forced cell.  The generation loop contributes exactly
        # one full decoder pass per emitted token (the native path has no KV
        # cache), and the prefill/teacher-forced paths contribute one logits
        # forward each.
        scoring = generated + prepares + prefill + teacher
        total_generated += generated
        total_prepares += prepares
        total_prefill += prefill
        total_teacher += teacher
        total_scoring += scoring
        per_readout.append({
            "prefix": int(prefix),
            "generated_tokens": generated,
            "prepare_sense_passes": prepares,
            "prefill_logits_forwards": prefill,
            "teacher_forced_logits_forwards": teacher,
            "explicit_logits_forwards": prefill + teacher,
            "scoring_decoder_passes": scoring,
        })
    training_probe_events = (
        update_forwards + probe_forwards + update_backwards + probe_backwards)
    return {
        "format": RECONSTRUCTION_FORMAT,
        "classification": reconstruction,
        "measured_by_old_run": bool(measured_by_old_run),
        "units": {
            "scoring_decoder_passes": DECODER_UNIT,
            "prepare_sense_passes": PREPARE_UNIT,
            "explicit_logits_forwards": LOGITS_UNIT,
            "generated_tokens": GENERATED_UNIT,
            "logical_scoring_operations": LOGICAL_OPERATION_UNIT,
        },
        "available_per_readout": int(available_per_readout),
        "per_readout": per_readout,
        "totals": {
            "generated_tokens": total_generated,
            "prepare_sense_passes": total_prepares,
            "prefill_logits_forwards": total_prefill,
            "teacher_forced_logits_forwards": total_teacher,
            "explicit_logits_forwards": total_prefill + total_teacher,
            "scoring_decoder_passes": total_scoring,
            "training_forward_passes": update_forwards + probe_forwards,
            "training_backward_calls": update_backwards + probe_backwards,
            "training_probe_events": training_probe_events,
            "grand_total_decoder_and_backward_events":
                total_scoring + training_probe_events,
        },
        "arithmetic": {
            "scoring_identity":
                "generated_tokens + 3*available + 2*available per readout",
            "logical_operations_per_readout":
                3 * available_per_readout,
            "warning": ("the historical 1,470 count is exactly the "
                        "prepare/sense subset and is not a decoder total"),
        },
    }


def validate_reconstruction(report: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the historical 7,695 + 60 = 7,755 reconciliation."""
    _require(report.get("format") == RECONSTRUCTION_FORMAT,
             "Reconstruction format differs")
    _require(report.get("measured_by_old_run") is False,
             "Historical values must be labelled reconstructed, not measured")
    _require(report.get("classification")
             == "reconstructed_from_preserved_receipts",
             "Historical values must be labelled reconstructed")
    per = report.get("per_readout") or []
    prefixes = [item.get("prefix") for item in per]
    _require(tuple(prefixes) == HISTORICAL_PREFIXES,
             "Historical prefixes differ")
    for item in per:
        _require(item["scoring_decoder_passes"]
                 == item["generated_tokens"]
                 + item["prepare_sense_passes"]
                 + item["explicit_logits_forwards"],
                 f"Scoring arithmetic differs at prefix {item['prefix']}")
        _require(item["prepare_sense_passes"]
                 == HISTORICAL_PREPARES_PER_CELL
                 * HISTORICAL_AVAILABLE_PER_READOUT,
                 "Prepare count differs")
        _require(item["explicit_logits_forwards"]
                 == 2 * HISTORICAL_AVAILABLE_PER_READOUT,
                 "Explicit logits count differs")
    totals = report.get("totals") or {}
    _require(totals.get("generated_tokens") == _sum(
        HISTORICAL_GENERATION_LENGTHS),
        "Historical generated-token total differs")
    _require(totals.get("prepare_sense_passes") == 1470,
             "Historical prepare total differs")
    _require(totals.get("scoring_decoder_passes")
             == HISTORICAL_TOTAL_SCORING_PASSES,
             "Historical scoring total differs")
    _require(tuple(item["scoring_decoder_passes"] for item in per)
             == HISTORICAL_SCORING_PASSES,
             "Historical per-prefix scoring passes differ")
    _require(totals.get("training_probe_events")
             == HISTORICAL_TRAINING_PROBE_EVENTS,
             "Historical training/probe events differ")
    _require(totals.get("grand_total_decoder_and_backward_events")
             == HISTORICAL_GRAND_TOTAL,
             "Historical grand total differs")
    return {
        "ok": True,
        "totals": dict(totals),
        "per_prefix_scoring_passes":
            tuple(item["scoring_decoder_passes"] for item in per),
        "reconstructed_not_measured": True,
    }


def eos_only_ids(generated_ids: Sequence[int],
                 stop_tokens: Sequence[int]) -> bool:
    ids = [int(value) for value in generated_ids]
    return len(ids) == 1 and int(ids[0]) in {
        int(value) for value in stop_tokens}


def whitespace_only_text(text: Any) -> bool:
    return isinstance(text, str) and text != "" and text.strip() == ""

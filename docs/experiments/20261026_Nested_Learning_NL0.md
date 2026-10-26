# Nested learning NL0 software/math qualification

**26 October 2026 · standalone contracts implemented and reviewed; no native
training or learned-capability claim.** The machine-readable receipt is
[qualification.json](../../reports/nested_learning_nl0_20261026/qualification.json).

## Implemented contract

`src/replica_lab/nested_learning.py` now provides the bounded plastic-expert
bank, selected-expert-only `NativeFusedAdamW` updates, an exact differentiable
AdamW comparator, associative fast-state recurrence, one-clock teaching
transactions, sealed observation chunks, causal frozen views, revision caches,
fixed/shuffled mechanical routes and stable personality identity.

The host commits one selected expert per eligible teaching event. Inactive
experts receive no gradient, moment creation or decay. Every publication is
atomic across parameters, moments, counters, clock, dedup and revision. The
observation coordinator aggregates eligible events in canonical event order and
performs at most one transition per sealed chunk, independent of reader count.
Empty and ineligible-only chunks perform zero transitions; nonempty visible
writes still advance the view revision and invalidate dependents.

Backdated state views/seals fail closed rather than combining old observations
with future fast state. Checkpoints require finite FP32 values, validate chunk
accounting against actual observation contents, and restore the complete live
state on every late failure. Public fast-state reads cannot mutate canonical
state. Global selected-gradient clipping occurs before every AdamW step.

## Verification

The final focused suite passes **28 tests in 1.17 s**. It includes direct and
later-read finite differences, exact-versus-first-order adaptation, two-step
nonzero-weight-decay fused parity, later-step clipping, changed-identity
rejection, used-controller/coordinator rollback, causal visibility, canonical
reduction order, empty/ineligible/eligible checkpoint restore and retry, and
personality consistency. Raw JUnit evidence is
[tests_final.xml](../../reports/nested_learning_nl0_20261026/tests_final.xml).

Independent review iterated through causal-view mixing, partial checkpoint
restore, visible-write cache staleness, clipping, mutable state exposure,
dtype/overflow, reduction order, personality mismatch and content-independent
chunk accounting. All findings are resolved; the closeout verdict is **correct**
with 0.99 confidence. The first status-only failed XML is preserved separately;
it records the old `specified_not_executed` assertion after qualification, not
a numerical or state-contract failure.

## Boundaries and next gate

No model was loaded and no native training, optimizer run or evaluation split
was executed. Nothing here establishes new-instance learning, correction,
retention, adaptive-router utility, fast/slow timing, multimodal learning,
authored personality content, recursive self-improvement or promotion.

The next gate is integration into the declared one-clock MiniCPM text baseline
with fixed routing and a separately predeclared source-free evaluation. Any
learned utility claim requires new-instance gain, correction, retention and
measured cost before additional clocks or mechanisms are considered.

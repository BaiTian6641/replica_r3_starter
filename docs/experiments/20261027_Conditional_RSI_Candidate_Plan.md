# Conditional RSI candidate-improvement plan — 27 October 2026

## Decision

Autonomous RSI remains **deferred**. Existing procedure publication, independent
verification, rollback and recovery contracts are implemented, but they are not
neural promotion evidence. The current neural gate remains fail-closed because
correction and retention evidence pins are absent. MiniCPM5 stays the baseline;
NativeFusedAdamW stays the default optimizer. Personality parameters, protected
identity goals, evaluator rules, resource policy and permissions remain fixed.

Current blockers are measured, not administrative:

- Response-CE teaching fits **0/28** exactly; dependent development is therefore
  not admitted by the new fit-receipt gate.
- Earlier one-clock development scored **0/16** new instances, **0/4** corrections
  and **0/8** unaffected retention. These are rejected learned-utility results,
  not candidate-promotion evidence.
- Memory acceptance v4 rejects the candidate at **21/384** accepted-correct with
  **15.22%** accepted precision. Train/dev calibration rejects every declared
  value point and every joint address operating point.
- Vision's historical curve preserves `curve_no_optimization_progress`: large
  loss reductions and rank changes do not establish generated-answer correctness
  or functional image dependence.

## Preconditions for any candidate

A candidate proposal must pin the exact parent/source epoch, data and episode
manifest, protocol/config, scorer and stopping versions, resource policy and
resulting state. It must provide independently checkable evidence for all four
capabilities below before any RSI cycle is considered:

| Gate | Required evidence | Current disposition |
|---|---|---|
| Useful recall | Future answers improve over exact-context and disabled/reset-memory controls on fresh development cases | Not established |
| Correction | A later correction changes the relevant future answer without wrong-world or stale-state recall | Not established |
| Unaffected retention | Unrelated answers remain stable under the same update and matched replay | Not established |
| Visual grounding | Correct/sibling/shuffled/no-image controls show causal image dependence plus generated-answer correctness | Not established |

A fit receipt is necessary before dependent development evaluation. It is not
sufficient for promotion. Held-out or sealed-final rows remain closed until the
declared development gates pass; consumed or exposed rows cannot be reused as
untouched evaluation.

## Frozen comparison policy

The first admissible learning hypothesis must retain the existing
one-step-per-event state machine. It must not revive rejected budget02 by changing
one configuration field or adding a second clock.

- **Model and optimizer:** sealed MiniCPM5; NativeFusedAdamW with its declared
  auxiliary groups; no backbone, personality, writer or reader expansion.
- **Gradient scope:** under the current constant/shared-routing path, only the
  selected expert's `gate`, `up` and `down` tensors receive gradients for teaching
  and correction events. The decoder, read projections, record writer, backbone,
  router, personality and fast associative state receive none. Current unaffected
  development rows receive no teaching gradient; that limitation must be explicit
  rather than described as retention training.
- **Controls:** exact context, selected-record literal readout, disabled memory,
  reset memory, shuffled values and wrong-world state remain separate arms.
  Address correctness, value correctness, abstention and multi-layer aggregation
  are reported separately.
- **Evaluation:** a frozen evaluator and decoding policy score fresh development
  cases only after a valid passing fit receipt. Independent evaluation owns
  success labels. Model-written success claims have no authority.
- **Candidate identity:** the candidate binds its parent, exact P-state, protocol,
  episode manifest, scorer, stopper, case coverage, attempts, teaching run, actual
  teaching cost and resulting canonical state. Stale or cross-state receipts fail
  before native loading or evaluation-role access.

## Bounded candidate cycle

1. Predeclare one hypothesis, one primary parameter change, exact update/data
   limits and physical work budgets. Commit the protocol before native execution.
2. Run one exclusive MPS worker with CPU fallback disabled, offline sources and
   the unchanged 50 GiB driver / 512 MiB sustained swap-growth / 15% host-free
   guard. Use a fresh output path and retain failures.
3. Freeze the resulting candidate and comparison policy. Evaluate independently
   on development only if the fit gate passes. Count actual model forwards,
   sensing, visual encoding, generation prefill/decode, backward calls, optimizer
   steps, tools, latency and artifact bytes; logical scoring operations are not
   model-forward counts.
4. Reject on any missing coverage, generation failure, false unknown/revoked
   correctness, wrong-world leakage, stale receipt, retention regression or
   unexplained cost overrun. Preserve negative trajectories and rejected
   candidates; do not automatically retry.
5. Publish only through the existing authenticated manifest registry. Rollback
   restores an authenticated ancestor and invalidates dependent state. A
   candidate cannot alter its verifier, guard, evidence epoch or protected goals.

## Rejection examples and rollback

The following remain explicit rejection examples, not tuning targets:

- Missing, failed, incomplete, stale or cross-state fit receipts.
- Arbitrary wrong answers, unknown/revoked mismatches, empty or truncated
  generations, contradiction and ambiguity.
- Candidate value readout with low precision even when address lookup is correct;
  accepting one layer while another abstains is not full group correctness.
- Rank-only improvement, EOS collapse, correct/no-image confusion, sibling or
  shuffled-image invariance, or any development/held-out access without admission.
- Unexplained implementation drift, guard-script substitution, changed source
  bytes without a reviewed current receipt, or total-compute accounting that
  omits generation/training work.

`BoundedRsiCycle` already authenticates issued candidates, evaluates disjoint
verification/rejection families, supports ancestor rollback and replays
interrupted publish/rollback transactions. Those mechanics stay available for a
future admitted candidate; no neural candidate is published by this plan.

## Total-compute accounting

Every candidate receipt must separate measured values from estimates. Measured
work includes decoder forwards by phase, prepare/sense passes, visual encoder
cache hits/misses and calls, projector passes, generated tokens, text/visual
positions, backward calls, optimizer steps, wall time, tool/evaluator calls and
artifact storage. Historical reconstruction is labeled as such. Estimated future
GPU-hours or memory floors never qualify an executable configuration.

The historical vision correction illustrates the required unit discipline:
**1,470** recorded prepare/sense passes are not total scoring forwards. Direct
sums of saved generation IDs reconstruct **7,695** scoring decoder passes and
**7,755** including training/probe work. Future protocols enforce caps in those
physical units at call time.

## Next executable gate

Do not start an RSI cycle. First obtain a fresh, one-clock-compatible learning
hypothesis whose fit receipt passes and whose development evidence demonstrates
future-answer utility, correction and unaffected retention under the frozen
controls. Independently, obtain a new visual result with correct generated answers
and causal image dependence. Only then pin correction/retention/visual evidence
for `NeuralGate.assess` and review a separate bounded candidate protocol.

Reference contracts:
`src/replica_lab/runtime/bounded_rsi.py`,
`src/replica_lab/runtime/learning_proposal.py`,
`src/replica_lab/learning.py`, and
`docs/architecture/22_State_Mechanism_and_RSI_Contracts.md`.

# M0a — corrected MiniCPM5 baseline measurements

24 October 2026 UTC. Protocols were committed before native execution:
`80e58fe` (capacity and fresh-menu family), `06c6334` (fixture isolation), and
`d0f2105` (old-case regression). Native execution used MPS with CPU fallback
disabled, one worker, and the implemented **50 GiB** driver policy plus
512 MiB swap-growth and 15% host-free checks.

**Decision: measurement package accepted; no menu promotion and no fresh-family
capability claim.** Corrected inference is detached and the selected 4096-unit
workload passes. Corrected action menus preserve the old-case result but fail on
the fresh family because the model selects unrelated numeric constants.

## 1. Corrected inference and backward

| Workload | Result | Peak driver | Wall | Other |
|---|---|---:|---:|---|
| Small: 1 inference + 1 backward point | 2/2 complete; writer/sense detached; 28 nonzero gradient tensors | **10.1247 GiB** | 8.85 s | 0 swap growth |
| Selected: 4096 units, 6 inference + 6 backward points | 12/12 complete; inference detached; 28 nonzero gradients per backward point | **28.5944 GiB** | 558.27 s | 0 swap growth; 9,135,648-byte run directory |

The old 4096-unit peak was 30.59 GiB and the old 8192 stop was 48.08 GiB. Both
predate the no-grad correction. The new 4096 measurement supersedes the old peak
for the same selected workload; 8192 was **not rerun**, so no corrected upper
limit is claimed. Inference, modules-only backward and optimizer-step capacity
remain distinct; there were **zero optimizer updates and zero base saves**.

Commands:

```bash
.venv/bin/python scripts/run_capacity_preflight.py \
  --entry-point run_e2_capacity.py \
  --report reports/m0a_capacity_small_attempt01/guard.json -- \
  --device mps --output runs/m0a_capacity_small_attempt01 \
  --protocol configs/m0a_e2_capacity_small_v1.json

.venv/bin/python scripts/run_capacity_preflight.py \
  --entry-point run_e2_capacity.py \
  --report reports/m0a_capacity4096_attempt01/guard.json -- \
  --device mps --output runs/m0a_capacity4096_attempt01 \
  --protocol configs/m0a_e2_capacity4096_v1.json
```

Evidence: [small report](../../runs/m0a_capacity_small_attempt01/run_report.json),
[small guard](../../reports/m0a_capacity_small_attempt01/guard.json),
[4096 report](../../runs/m0a_capacity4096_attempt01/run_report.json),
[4096 guard](../../reports/m0a_capacity4096_attempt01/guard.json).

## 2. Corrected action menus

| Arm | Fresh four-case family | Held-out | False success claims | Interpretation |
|---|---:|---:|---:|---|
| Model, canonical | **1/4** | **0/2** | 2 | Selects unrelated constants in three cases |
| Model, shuffled | **0/4** | **0/2** | 3 | Repeated distractor selection and premature stop |
| Random, same menu | **4/4** | **2/2** | 8 | Invalid success baseline: two premature claims per case |
| Deterministic procedure | **3/3** | 2/2 | 0 | Parses the actual bound and repairs correctly |
| Model, old-case regression | **3/3** | 3/3 | 0 | Old result preserved |

The fresh failure is **action selection/generalization**, not numeric parsing.
The menu contains the target and other exact constants. After the actual failed
test, the model often selects another constant instead of the evaluator target;
the deterministic procedure succeeds because it parses the bound. Old-case
regression remains 3/3 with zero false success claims, so the corrected interface
does not itself force failure. Fresh-family results do not support promotion.

Commands used the same `run_capacity_preflight.py` wrapper with
`run_e3_candidate_rerank.py`, protocol `configs/m0a_e3_candidate_menus_v1.json`,
and `--selection model`, `--selection model --shuffle-candidates`, or
`--selection random`. The procedure reference used:

```bash
.venv/bin/python scripts/run_e5_procedure.py \
  --config configs/m0a_procedure_reference_v1.json \
  --output runs/m0a_procedure_reference_attempt02
```

The old-case regression used `configs/m0a_e3_candidate_order_regression_v1.json`.
Raw traces, candidate scores and receipts are in the run directories cited by
[the validation summary](../../reports/validation_m0a_baselines_20261024.json).

## 3. Preserved defects and decisions

- A fixture test initially executed repairs in place and mutated the generated
  fixtures. It was corrected to use temporary copies; the mutated directory was
  moved to `runs/m0a_fixture_mutation_attempt01`. The procedure reference then
  passed. This was a test defect, not model evidence.
- The first procedure reference failed on that mutated fixture and is preserved
  as attempt 1; attempt 2 is the accepted reference.
- The old 8192 boundary remains a guard failure, not an accepted capacity result.
- No 8192 rerun was admitted because the selected workload was the required M0a
  measurement, not an open-ended capacity ladder.

**Next:** M1's real compiled procedure path is complete. The next unresolved
model dependency is M2: one residual adaptive-memory insertion with distinct
head writes and delayed-use controls. No action-menu promotion is carried into
that work.

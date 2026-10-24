# M2 — useful learned delayed memory on MiniCPM5

24 October 2026 UTC. The M2 SGDN-2 path is implemented in
`src/replica_lab/neural.py`, `src/replica_lab/architecture/ports.py`,
`src/replica_lab/architecture/minicpm5.py` and
`src/replica_lab/runtime/fast_memory.py`. Protocols: `85c07b7`, gate fix
`57596d1`, shared-value protocol `ea4491f`.

**Result: accept the shared-value arm.** Six values shared across 24 fresh
training entities produced useful delayed memory after the original context was
removed. This is the first accepted learned memory result, not a general memory
or reasoning claim.

## Results

| Arm | Dev exact | Held-out exact | Interpretation |
|---|---:|---:|---|
| SGDN-2 correct | **4/4** | **4/4** | Useful learned delayed memory |
| Linear update control | 1/4 | 2/4 | Weaker than SGDN-2 |
| Shuffled observation | 0/4 | 0/4 | No accidental association |
| Zero state | 0/4 | 0/4 | No memory fallback |
| Disabled fast branch | 0/4 | 0/4 | No memory fallback |
| Full-context reference | 4/4 | 3/4 | Exact text remains available |

The candidate also completed **24/24 train**, all **8/8** dev+held-out correction
checks, and a fresh image reload with the correct answer. Retention is **2/2**
before and after training. Peak sampled MPS driver was **11.004 GiB**, wall time
**58.10 s**, zero swap growth and zero downloads. No base copy was saved.

The prior unique-value arm (`runs/m2_dynamic_attempt06`) trained to **8/8 train**
but scored **0/4 dev and 0/4 held-out**. Its loss collapsed to ~0.001 and the
model emitted training values instead of requested unseen values. That is a
memorization/generalization failure, not a useful memory result. The first
attempts also exposed an SGDN-2 gate initialization defect (`_softcap_silu(0)=0`),
which was fixed and regression-tested before the accepted arm.

## Mechanism and checks

- Independent per-head keys, values, erase, write and decay gates; no matrix is
  repeated across heads.
- Pure functional chunk-start transition with masked writes and a bounded
  gradient horizon; detached host-owned snapshots at deployment.
- Native residual port after the complete block with zero-initialized final
  projection and a nonzero output gate for gradient flow.
- Correct, linear, shuffled, zero, disabled and full-context controls.
- Correction, restart, source lineage, retention replay and explicit raw answers.

The shared-value family contains 24 train, 4 dev, 4 held-out and 2 retention
cases with fresh entities and six reused values. The held-out entities are new
bindings of seen values, so this measures copying rather than training-value
memorization. It remains synthetic evidence, not broad transfer.

Raw report: `runs/m2_shared_attempt01/run_report.json`; validation:
`reports/validation_m2_dynamic_memory_20261024.json`.

## Next dependency

M3 must compare shared and private depth ownership with the same SGDN-2 update
engine, explicit state/compute accounting, cross-depth gradient checks and
leakage controls. M4 Engram and M5 RSI remain separate later packages.

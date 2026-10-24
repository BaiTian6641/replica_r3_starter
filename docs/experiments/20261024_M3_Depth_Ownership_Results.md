# M3 — shared and private SGDN-2 depth ownership

24 October 2026 UTC. **Decision: `accept_m3_ownership_interfaces`.** The independent
[comparison](../../reports/m3_depth_comparison_attempt02/comparison.json) passed
every declared check. This accepts depth ownership, post-open gradient paths and
fresh-process restart on an exposed synthetic family. It does **not** accept the
observed retention regression, a capacity-matched advantage, uTTT reproduction,
or production promotion.

## 1. Matched outcomes

All arms used the sealed MiniCPM5 source, the E2 parent, 16 modules-only updates,
two independent per-depth writers, and the same exposed M2 cases. Shared averages
both next-state proposals from one chunk-start state; private commits them to
separate states. The reducer is a project SGDN-2 convex mean, not uTTT gradient
aggregation.

| Metric (dev / held-out) | Shared | Private | First-chunk gradient-cut |
|---|---:|---:|---:|
| Correct delayed answer | **4/4 / 4/4** | **4/4 / 4/4** | **4/4 / 4/4** |
| Linear update control | 1/4 / 2/4 | 0/4 / 2/4 | 1/4 / 1/4 |
| Shuffled / zero / disabled | 0/4 / 0/4, all | 0/4 / 0/4, all | 0/4 / 0/4, all |
| Full-context reference | 4/4 / 3/4 | 4/4 / 3/4 | 4/4 / 3/4 |
| Read block 13 only | 0/4 / 0/4 | 0/4 / 0/4 | 0/4 / 0/4 |
| Read block 27 only | 4/4 / 4/4 | 4/4 / 4/4 | 4/4 / 4/4 |
| Write block 13 only | 3/4 / 3/4 | 0/4 / 0/4 | 3/4 / 3/4 |
| Write block 27 only | 3/4 / 3/4 | 4/4 / 4/4 | 3/4 / 2/4 |

Shared and private produced identical answers on all eight paired cases. This is
descriptive evidence, not proof of equivalent internal states or superiority.
Block 27 is sufficient for these trained diagnostics; no general port optimum is
claimed.

## 2. Gradients and restart

- Update 1 retained the expected zero-output-projection path: write-gradient L1
  was zero before the projections opened. Update 2 had finite, nonzero gradients
  for every layer and both observation chunks in shared and private arms.
- Gradient-cut made every first-chunk write detached (`requires_grad=false`, no
  gradient), while later per-depth writes retained nonzero gradients at update 2.
- Both candidates saved and reloaded actual depth images with all tensors,
  layer IDs, lineage, counters, and stable component hashes. No observations were
  re-encoded.
- Fresh native processes reloaded each extension and image and answered the
  held-out case correctly (`amber`). Shared took **4.761 s**, private **4.747 s**.
- The shared attempt01/attempt02 extension bytes are identical. Private attempts
  1/2 failed only the linear-control frame boundary; attempt 3 completed.

## 3. Retention failure — not accepted

Both shared and private candidates scored **0/2** on the fixed retention replay,
while their parents scored **2/2**. Both answered `copper` for the correct
`Paris` and `H2O` replay cases. The preregistered ownership rule did not gate
retention, so the interface verdict remains valid, but this is a real unresolved
competence regression. An inference-only ablation diagnostic precedes any repair
arm or M4 expansion.

## 4. Costs and artifacts

| Arm | Train wall | Peak sampled driver | Generated tokens | Extension | Depth image | Logical state | Final optimizer state |
|---|---:|---:|---:|---:|---:|---:|---:|
| Shared | 72.961 s | 11.0043 GiB | 470 | 48,956,635 B | 66,801 B | 65,536 B | 7,485,456 B |
| Private | 70.394 s | 11.0043 GiB | 435 | 48,956,636 B | 132,444 B | 131,072 B | 7,485,456 B |
| Gradient-cut | 72.542 s | 10.1623 GiB | 477 | 48,956,635 B | 66,801 B | 65,536 B | 7,485,456 B |

Optimizer state comprised 7,340,032 bytes of int8 momentum and 145,424 bytes of
FP32 auxiliary state. Trainable parameters/gradients and AdamW state remained
FP32. Each arm generated 134 answers, within 804 tokens. Accepted training took
215.896 s sequentially; both restarts added 9.508 s. Selected run directories are
about 48 MiB each. No base copy or download occurred.

All native jobs used one MPS worker, CPU fallback disabled, the 50 GiB driver
stop, 512 MiB sustained swap-growth limit and 15% host-free threshold. Peak swap
growth was zero and minimum observed host free was 76%. Memory samples can miss
transients; MPS is not ANE.

## 5. Evidence and commits

Final evidence set:

- Shared: `runs/m3_depth_shared_attempt02/run_report.json` (protocol `9fd1be1`).
- Private: `runs/m3_depth_private_attempt03/run_report.json` (protocol `79e6e4a`).
- Gradient-cut: `runs/m3_depth_gradient_cut_attempt01/run_report.json`.
- Fresh restarts: `runs/m3_depth_shared_restart_attempt01/run_report.json` and
  `runs/m3_depth_private_restart_attempt01/run_report.json` (protocol `82660a3`).
- Verdict: `reports/m3_depth_comparison_attempt02/comparison.json` (`bde63a7`).
- Machine summary: `reports/validation_m3_depth_20261024.json`.

Private attempt01/02 failure reports remain preserved. The first scorer defer
was a path-normalization defect, corrected without native reruns; its output is
also preserved. Implementation commits were `6edd090`, `94e7308`, `2c72564`,
`b276e1f`, `00a19a4`, `ae9d7cb`, and `8eadde8`.

The full integration suite passed **1249 tests and 21 subtests in 133.03 s**.
Subsequent focused regressions covered private linear frames and shared linear
byte equivalence.

**Next:** inference-only retention ablation, then a separately preregistered
repair only if the controls identify a concrete cause. M4 remains pending.

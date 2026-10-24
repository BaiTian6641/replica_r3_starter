# M3 retention diagnostic — fast-residual interference

24 October 2026 UTC. Protocol `a97c2c8` (`configs/m3_retention_diagnostic_v2.json`).
**Decision: fast-residual interference is confirmed; no single-port repair is
justified.** The candidates remain diagnostic-only and no weights were changed.

## Results

Both preserved candidates answered both replay cases incorrectly. All other
arms also failed except the three recovery controls:

| Arm | Shared exact | Private exact | Interpretation |
|---|---:|---:|---|
| Correct trained state | 0/2 (`copper`) | 0/2 (`amber`) | Regression reproduced |
| Linear update | 0/2 | 0/2 | Does not recover |
| Shuffled observation | 0/2 | 0/2 | Does not recover |
| Zero state | **2/2** | **2/2** | Fast residual is required for failure |
| Disabled fast branch | **2/2** | **2/2** | Frozen base remains capable |
| Full context | **2/2** | **2/2** | No base/evaluation anomaly |
| Read block 13 only | 0/2 | 0/2 | No isolated recovery |
| Read block 27 only | 0/2 | 0/2 | Does not reproduce the joint wrong answers |
| Write block 13 only | 0/2 | 0/2 | No isolated recovery |
| Write block 27 only | 0/2 | 0/2 | Does not reproduce the joint wrong answers |

Block 27 alone is therefore not a supported repair target. Removing either
reader or either writer does not restore the correct answers, while removing the
entire fast residual does. The next admissible hypothesis is **fixed retention
replay during training**, not a port ablation. Any repair requires fresh replay,
development and final-audit roles; the two diagnostic cases stay evaluation-only.

## Costs and evidence

- Shared: 20 answers, 61 generated tokens, 10.583 s profiled wall,
  11.0043 GiB sampled peak driver.
- Private: 20 answers, 51 generated tokens, 9.554 s profiled wall,
  11.0043 GiB sampled peak driver.
- No optimizer updates, checkpoints, downloads, base copies or swap growth.
- Raw reports: `runs/m3_retention_shared_attempt02/run_report.json` and
  `runs/m3_retention_private_attempt01/run_report.json`.
- Machine summary: `reports/validation_m3_retention_diagnostic_20261024.json`.

Attempt 1 stopped before model execution because dataset role prose was compared
as data identity. It is preserved at
`runs/m3_retention_shared_attempt01/failure_report.json`; attempt 2 compares only
root and content hashes. No result was inferred from the failed startup.

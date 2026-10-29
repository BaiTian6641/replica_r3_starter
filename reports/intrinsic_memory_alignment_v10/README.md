# Frozen-teacher intrinsic-memory comparison v10

**Completed native attempt: fitting passes; useful development recall fails.**
The [experiment report](../../docs/experiments/20261029_Spark_Memory_Teacher_Alignment_V10.md)
records the implementation, fixed gates, actual outcomes and full costs. See the
[current evidence summary](verification_summary.json), [native result summary](native_attempt01_summary.json)
and [matched direct-CE comparison](direct_ce_comparison.json).

| Stage | Result |
|---|---|
| Official BF16 source/code/environment admission | Passed |
| Frozen complete-context calibration | 8/8, false recall 0, truncation 0 |
| Separate frozen fitting teacher | 8/8, false recall 0, truncation 0 |
| Fixed final fitting after 128 updates | 8/8; four counterfactual pairs pass |
| Fresh development | Contextual 5/8, exact 8/8, wrong 0/8, shuffled 5/8, zero/absent 1/8; utility fails |
| Correction/revocation | Stale/revoked reads refused; corrected contextual answer wrong, exact text correct |
| Fresh process | Exact logits, restored teacher cache and one actual continuation update |
| Original weights | Full digest unchanged; zero P updates and no promotion |
| Native 9/36 | Unrun; useful matching two-site evidence absent |

New clean snapshot `ad58c14b` passes **502/502** tests, zero skips, in 226.228 s.
The 2/9/36 CPU workflow passes in 14.439167 s; its teacher qualification is
synthetic and cannot establish pretrained capability. The immutable
[prelaunch receipt](prelaunch_verification.json) retains its original prelaunch
scope. [Frozen producer inspection](frozen_producer_diagnosis.json) is an additional
CPU-only fitting diagnostic, with no backbone load or update.

Native wall time was 617.853465 s, peak sampled driver
11,261,542,400 bytes, swap growth zero, minimum host free 70%, and
output 49,944,544 bytes. Actual work was 922 forwards, 525 generated tokens,
236,032 padded input positions, 129 backward/updates, two sequential loads,
245 source encodes, 51 retrievals and 112 declared writes. Raw answers, teacher
targets, logs, refusal events and resource telemetry are retained. Physical
memory samples, audit bytes and design reservations have distinct meanings.

## Consumed command and independent density budgets

This exact command has completed and **must not be repeated**:

```sh
PYTHONDONTWRITEBYTECODE=1 .venv-spark-bf16/bin/python scripts/run_spark_intrinsic_alignment_v10.py --packet reports/intrinsic_memory_alignment_v10/native_packet_two_v10.json --authorization reports/intrinsic_memory_alignment_v10/authorization_two_v10.json --launch
```

The [exact receipt](authorization_two_v10.json) records the user’s standing
bounded-testing instruction. No more permission question is required for new
bounded tests, but every new packet still needs its own accurate receipt,
execution identity, unused output and full budget. Consumed outputs have zero
retries. Official BF16 is distinct from FP8-derived BF16 and official FP8.

Per-arm caps remain 2,469 forwards, 2,072 generated tokens, 632,064 padded input
positions, 129 updates/backwards, two loads, 245 source encodes, 51 retrievals,
112 writes, 7,080 s stage allocation and 7,200 s outer wall. Artifact caps are
160/416/1,056 MiB for 2/9/36. One exclusive worker, 50 GiB sampled driver,
512 MiB sustained swap growth and >=15% host free remain mandatory. Nine/all-layer
packets are prepared but their useful two-site prerequisite is not met.

## Next evidence boundary

The [development exposure receipt](development_exposure.json) retires the shared
v8/v9/v10 development split from fresh use. Preserve it as a labeled historical
control; use fresh predeclared worlds for a successor. The next scientific question
is whether a controlled improvement to source coverage or ordered readout transfers
to new entities and relations while beating the same negative controls. This
needs a new protocol, CPU qualification and measured native budget; it is not an
admitted launch. Do not tune repeatedly against this exposed development set.

The v10 implementation is complete. Useful memory, native 9/36 feasibility,
formation, source-free P learning, retention, nested adaptation, Replica fidelity
and RSI remain unqualified. Keep P1, separate E/S/P/phi/C/I/K ownership, approved
identity outside task reward, physical tiers independent of learning clocks, and
the external spine reserved for specialists, sensors and actors.

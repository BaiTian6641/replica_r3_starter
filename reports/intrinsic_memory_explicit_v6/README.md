# Explicit entity/color evidence v6

**Current outcome:** Completed: frozen competence passed 8/8; contextual fitting failed 0/8 after 32 updates. Development stayed unopened. Fresh-process exact logits, one actual continuation update and full frozen-weight integrity passed. See the [measured summary](native_attempt01_summary.json) and [current native report](../../docs/experiments/20261029_Spark_Memory_BF16_Native_Diagnostics.md). The two-site output is consumed with zero retries. Commands below document its historical preparation and must not be relaunched. Native 9/36 and persistent learning remain gated.

This is one data-only successor to the preserved
[official BF16 v5 attempt](../intrinsic_memory_successor_v5/native_attempt01_summary.json).
V5 completed all eight calibration cases at 6/8 correct, one false recall and
zero truncation; no fitting or development opened. Its copper-lamp wording can
describe material, and “missing signal” does not unambiguously identify a distinct
entity. These are possible explanations, not a demonstrated sole cause.

V6 states an explicit recorded-color relation and queries exact entity names.
An unknown query names an entity absent from the complete fact table. A structured
oracle validates the answer independently of the prose rendering. Four fitting
pairs keep the identical question and switch only the evidence color. Entities
and palettes are disjoint across calibration, fitting and development; this is
still one lookup rule family and one feasibility seed, not broad generalization.
No historical development corpus is read by preparation.

The unchanged v5 runner, official BF16 source, chat template, strict scorer,
eight-of-eight competence gate, 32+1 update schedule, fitting/development criteria,
zero P updates and per-arm budgets are reused. A new protocol, executable manifest,
data hashes, output destinations and checkpoint execution identity bind the change.
All v6 choices are fixed before its first model answer. The old v5 packet and
observations remain immutable; its attempt is consumed and will not be retried.

Portable checks: 28 passed in 17.40 seconds, including the new corpus and existing
execution/gate tests. Raw receipts are in
`../intrinsic_memory_successor_v5/explicit_successor_cpu_tests.{log,xml}`.
The user explicitly authorized continuing the testing plan without more approval
questions. Each actual launch still gets a truthful exact-packet receipt; the
scientific gates and resource ceilings are unchanged.

The two-site command is:

```sh
PYTHONDONTWRITEBYTECODE=1 .venv-spark-bf16/bin/python scripts/run_spark_intrinsic_diagnostic_v5.py --packet reports/intrinsic_memory_explicit_v6/native_packet_two_v6.json --authorization reports/intrinsic_memory_explicit_v6/authorization_two_v6.json --launch
```

Each arm has one attempt, zero retries, at most 7,200 seconds, 957 forwards,
33 backwards/updates, 856 generated tokens, 244,992 padded input positions, two
sequential original model loads, and zero teacher/P updates. Artifact limits are
160/416/1,056 MiB for 2/9/36 readers. Keep one exclusive native worker and the
50 GiB sampled driver / 512 MiB sustained swap / 15% minimum host-free guards.
Nine/all remain closed unless this protocol's two-site result qualifies useful
readout and numerical continuation. No automatic promotion is permitted.

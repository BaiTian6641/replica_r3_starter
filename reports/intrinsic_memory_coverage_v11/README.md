# Intrinsic-memory broader-teaching v11

**Current state: attempt01 has ended; its output is consumed, zero retries.**
Useful intrinsic recall remains unqualified. See [terminal summary](native_attempt01_summary.json),
[verification summary](verification_summary.json) and [group analysis](fitting_group_analysis.json).

The [fixed experiment](../../docs/experiments/20261029_Spark_Memory_Coverage_V11.md)
and [protocol](../../configs/spark_intrinsic_coverage_v11.json) declare fresh
8/32/16 worlds, unchanged producer/reader and top-64 teacher alignment, 512 fixed
updates plus one numerical restart update, and unchanged criteria required for
every eight-world group. All choices precede native observations.

[Working CPU evidence](working_cpu_summary.json): 60 passing focused tests and
an actual tiny BF16 2/9/36 workflow, including fresh-process continuation. This is
engineering evidence, with synthetic teacher qualification, not native capability.
Final clean `71f7ab4d` passes 562/562, zero skips, in 239.645 seconds.
The clean CPU workflow ran at `607ba7a8` in 18.968 seconds. See
[immutable prelaunch verification](prelaunch_verification.json). Native results are recorded below.

The exact [two-site packet](native_packet_two_v11.json) uses official BF16 under
its recorded identity. [Authorization](authorization_two_v11.json) records the
user's standing direct instruction to continue bounded testing without another
approval question. The packet is one attempt, zero retries, zero P updates and
no promotion. Never repeat a consumed output.

```sh
PYTHONDONTWRITEBYTECODE=1 .venv-spark-bf16/bin/python scripts/run_spark_intrinsic_coverage_v11.py --packet reports/intrinsic_memory_coverage_v11/native_packet_two_v11.json --environment-only
PYTHONDONTWRITEBYTECODE=1 .venv-spark-bf16/bin/python scripts/run_spark_intrinsic_coverage_v11.py --packet reports/intrinsic_memory_coverage_v11/native_packet_two_v11.json --review
PYTHONDONTWRITEBYTECODE=1 .venv-spark-bf16/bin/python scripts/run_spark_intrinsic_coverage_v11.py --packet reports/intrinsic_memory_coverage_v11/native_packet_two_v11.json --authorization reports/intrinsic_memory_coverage_v11/authorization_two_v11.json --launch
```

Limits: one worker, 50 GiB driver, 512 MiB sustained swap growth, 15% host free,
7,200 seconds, 8,829 forwards, 7,256 generated tokens, 513 updates, two sequential
loads and 256 MiB artifacts. Stage budgets pre-reserve failed work too. The nine
and all-layer packets have separate budgets and require useful matching two-site
evidence; they cannot use density to bypass a failed semantic gate.

Preserve v3–v10 results and unrelated work. The original model is frozen. No
large downloads, base exports, persistent learning, native multimodal or RSI.

## Terminal evidence

Latest completed fitting probe at 512 updates: contextual 26/32, exact_text 32/32, wrong 2/32, shuffled 26/32, zero 8/32, absent 8/32. Development remains unopened because the final fitting gate did not pass.

A fresh process restored exact logits and the teacher cache, then performed one actual continuation update. Original weights remained unchanged in both processes. This is numerical continuation; uninterrupted-update equivalence was not measured.

Native wall 3656.043 s, peak sampled driver 10.488 GiB,
swap growth 0 bytes, minimum host free 69.0%.
P updates and promotions are zero. Native 9/36 remains closed. Review final fitting groups and distinguish source representation from reader alignment before a new two-site mechanism experiment. Keep fixed gates; no best-prefix selection or retry of this output.

The launch command above is preserved for provenance and must not be repeated
against this consumed output. The [unused initial preparation](preparation_revision1/supersession.json)
and its receipt remain archived; stage-time reallocation preceded native answers.

The [budget compliance receipt](budget_compliance.json) passes all recorded
limits. The [post-run producer probe](frozen_producer_diagnosis.json) is a CPU
fitting-only representation diagnostic, with no native load or update. Run it
only against a verified terminal checkpoint; it refuses to overwrite its result.
The [descriptive prelaunch erratum](prelaunch_audit_navigation_errata.json)
preserves old audit hashes while resolving their archived packet paths.

The separate [shared-reader CPU candidate](../intrinsic_memory_matched_reader_v1/README.md)
passes 29 clean tests and keeps exactly two parameter sets across 2/9/36 sites.
Its native integration and the next source-feature producer experiment are not
yet launchable. Keep the useful-readout gate closed and continue independent CPU
implementation under the standing testing authorization.

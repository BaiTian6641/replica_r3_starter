# Intrinsic-memory broader-teaching v11

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
[immutable prelaunch verification](prelaunch_verification.json). Native results are pending.

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

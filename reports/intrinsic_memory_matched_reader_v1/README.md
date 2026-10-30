# Matched reader parameter control v1

The [implementation report](../../docs/experiments/20261029_Spark_Memory_Matched_Readers_V1.md)
and [candidate protocol](../../configs/intrinsic_memory_matched_reader_v1.json)
describe a separate CPU implementation. All 2/9/36 layouts register exactly two
reader groups and preserve the real contextual producer/store path. No running
or historical native packet uses this candidate.

Working-tree verification passes 29 tests, zero skips, including exact fresh
process logits, continuation weights and optimizer moments for every layout.
See [working receipt](working_final/summary.json) for exact source hashes,
parameter counts, snapshot costs and measured CPU resources. The earlier
26-test run is retained as an intermediate scope. Native usefulness, density,
formation, persistent P learning and RSI remain unqualified by these checks.

Reproduce from a clean sparse checkout with the protected module and pretrained
checkpoint weights excluded:

```sh
PYTHONDONTWRITEBYTECODE=1 /Users/flare/replica_r3_starter/.venv/bin/python reports/intrinsic_memory_matched_reader_v1/verify_cpu.py --root . --out /private/tmp/matched-reader-clean-result --require-clean
```

The output path must be new. Verification never loads pretrained weights and
uses synthetic canonical records and a tiny random BF16 36-block backbone.
Its single-forward CPU timings do not establish native physical fit or match
total teaching cost. The protocol separately specifies that later experiment.

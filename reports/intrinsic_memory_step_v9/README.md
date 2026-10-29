# Smaller-step intrinsic-memory control v9

**Current outcome:** Completed: frozen competence passed 8/8. Contextual fitting was 3/8, 6/8, 7/8 and 6/8 at 32/64/96/128 updates. The fixed final gate failed; the earlier 7/8 checkpoint was not selected and development stayed unopened. Fresh-process continuation and full frozen-weight integrity passed. See the [measured summary](native_attempt01_summary.json) and [current native report](../../docs/experiments/20261029_Spark_Memory_BF16_Native_Diagnostics.md). The two-site output is consumed with zero retries. Commands below document its historical preparation and must not be relaunched. Native 9/36 and persistent learning remain gated.

The [v8 result](../intrinsic_memory_supported_v8/native_attempt01_summary.json)
passes 8/8 complete-context competence but scores 0/8 contextual fitting recall
at every fixed 32/64/96/128 prefix. Exact text scores 8/8. Development stays
unopened. Exact fresh-process logits, one actual continuation update, finite
intended gradients and full frozen-backbone hashes pass. The experiment fails
its learning gate; numerical continuation is a separate outcome.

V9 changes **only the ordinary optimizer learning rate from 0.001 to 0.0001**,
plus the required execution/output identities. Earlier measured large reader
residuals and the v8 fitting collapse motivate this control; they do not prove
that learning rate is the cause. The comparison resets the extension to the same
initialization. It does not continue or select the v8 checkpoint.

The unchanged v7 native engine runs all 128 updates and all four fitting probes,
then one separately budgeted fresh-process update. There is no early stopping or
best-prefix choice. Data, source, memory mechanism, template, scorer, numerical
rules, fitting/utility criteria and work/resource budgets are unchanged. Admission
binds the failed-fit v8 result and rejects missing or passing predecessor evidence,
changed comparison contracts or any v8 development receipt. The native engine's
result schema remains v7; the v9 protocol and packet hashes identify this distinct
execution. Historical v5–v8 code, protocols and observations remain unchanged.

Each arm still permits 2,397 forwards, 129 backwards/updates, 613,632 padded input
positions, 2,008 generated tokens, two sequential model loads, 245 source encodes,
51 retrievals and 103 write transactions. Stage wall allocation is 7,080 seconds
inside a 7,200-second outer supervisor. Artifact caps remain 160/416/1,056 MiB for
2/9/36 readers. Keep one exclusive native worker, the 50 GiB sampled driver
ceiling, 512 MiB sustained swap-growth limit and 15% minimum host-free rule.
P/teacher updates are zero. Density, formation, persistent learning and promotion
remain closed until their own useful-recall and later evidence gates pass.

The user's continuing testing authorization applies. The exact receipt records
that direct instruction; consumed earlier approvals are not reused. The one
two-site launch is:

```sh
PYTHONDONTWRITEBYTECODE=1 .venv-spark-bf16/bin/python scripts/run_spark_intrinsic_smaller_step_v9.py --packet reports/intrinsic_memory_step_v9/native_packet_two_v9.json --authorization reports/intrinsic_memory_step_v9/authorization_two_v9.json --launch
```

`--environment-only` checks the qualified environment; `--review` revalidates the
source, execution closure, predecessor, data and output bindings without loading
weights. Portable tests use synthetic executable source trees. Sixteen checks
pass, including failed-predecessor/data-opening rejection, model/configuration
tampering, unchanged budgets/gates and smaller-rate optimizer ownership/reload.
These tests are engineering evidence, not native recall evidence.

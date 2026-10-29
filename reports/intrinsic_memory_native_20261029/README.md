# Official BF16 intrinsic-memory evidence, 29 October 2026

**Current successor:** [v10 teacher-alignment results](../../docs/experiments/20261029_Spark_Memory_Teacher_Alignment_V10.md) record the separately versioned implementation, 502-test clean verification and completed native diagnostic. The v5–v9 evidence below is preserved.

The [current report](../../docs/experiments/20261029_Spark_Memory_BF16_Native_Diagnostics.md)
records five separately bound native attempts. All use the existing official
Spark-X2.6-4B BF16 source; FP8-derived BF16 and official FP8 remain distinct.
The user's continuing testing authorization is recorded in each exact receipt.
All five outputs are consumed and cannot be retried.

| Attempt | Frozen competence | Final fitting | Fresh development |
|---|---:|---:|---|
| v5 | 6/8, failed | Closed | Closed |
| v6 | 8/8 | 0/8, failed | Closed |
| v7 | 8/8 | 8/8 | 0/8 contextual versus 8/8 exact text; one false recall |
| v8 | 8/8 | 0/8, failed | Closed |
| v9 | 8/8 | 6/8, failed | Closed |

Useful recall is not established. V6–v9 pass exact fresh-process logits, one actual
continuation update and full frozen-original-weight hashes. These are numerical
outcomes. There are zero P updates and no promotion. Native 9/36, formation,
persistent learning, nested adaptation and RSI remain gated.

The [evidence summary](verification_summary.json) separates new and historical
checks. Clean snapshot `8ad52904` passes 444/444 tests, zero skips, in 222.114 s;
two earlier portability failures and their raw logs remain preserved. The
2/9/36 CPU integration passes in 13.378906 s with 598,392,832 bytes parent peak RSS,
105,170,044 bytes of arm artifacts, shared-snapshot parity, live gradients and
one actual fresh-process continuation per arm. It loads no native checkpoint.

[Established verification](established_verification.json) at `b42a7d55` passes
1,283/1,283 strict tracked review checks, 540/540 final-product checks, all 216
bundle bindings and the tracked artifact guard. The [changed-file inventory](changed_files.json)
names implementation, protocol, documentation and evidence changes and separates
this final receipt-only commit. Unrelated dirty files and local extension/store
artifacts remain preserved; the protected runtime module was never accessed.

Across these five official BF16 runs, peak sampled driver memory is
11,261,542,400 bytes, swap growth is zero and minimum host free is 70%. Driver and
RSS overlap; sampled maxima can miss transients. Per-attempt wall, token, teaching,
retrieval, write, hashing and audit costs are retained in the summary and raw ledgers.

The next research comparison is a separately implemented and budgeted full-context
teacher/memory-student objective against direct CE. It has no runnable native packet
yet. The standing authorization covers bounded tests once preparation is complete.
Preserve unopened development
labels, the fixed gates, P1 and the separate Replica identity/task/state contracts.

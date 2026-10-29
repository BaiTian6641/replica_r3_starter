# Official BF16 intrinsic-memory diagnostics

**Current successor:** [v10 teacher-alignment results](20261029_Spark_Memory_Teacher_Alignment_V10.md) record the separately versioned implementation, 502-test clean verification and completed native diagnostic. The v5–v9 evidence below is preserved.

29 October 2026. This report follows the v5 preparation at `86ba96e0` and
preserves the separately bound v5–v9 native attempts. The source is the
existing official **XHToken/Spark-X2.6-4B BF16** artifact at revision
`66e3ae7bee54e0a523d692cb8ef841f35409807b`. It is neither the FP8-derived BF16
artifact used by attempt03 nor official FP8 execution. No checkpoint download,
duplicate base checkpoint or full-backbone export occurred.

The user's later instruction authorizes continuing the planned tests without
more approval questions. Each attempt records that actual instruction in an
exact packet-bound receipt. Earlier single-attempt receipts remain consumed;
the instruction does not change old results, permit retries of consumed output
directories or remove scientific and resource gates.

## Separate competence, fitting and generalization

The [v5 native summary](../../reports/intrinsic_memory_successor_v5/native_attempt01_summary.json)
records **6/8** frozen complete-context competence, one false recall and zero
truncations. No fitting, development or restart occurred. The failed prompts
include a copper lamp without an explicit color predicate and a query about a
"missing signal" when a signal is present. Ambiguous oracle wording is a
possible explanation, not a demonstrated causal result. The original scores and
all raw answers remain unchanged.

The separately bound [v6 explicit-evidence corpus](../../scripts/prepare_spark_intrinsic_explicit_evidence_v6.py)
uses explicit entity/color predicates, distinct entity names for unsupported
queries, and a structured oracle. Its three roles have disjoint entity names.
Calibration uses copper/silver, fitting red/blue, and development green/gold.
These are distinct worlds within **one lookup rule family**, not independent
rule families. The unchanged v5 runner, source, optimizer, template, scorer,
complete-response objective, budgets and acceptance criteria execute v6.

The [v6 summary](../../reports/intrinsic_memory_explicit_v6/native_attempt01_summary.json)
records **8/8** competence with zero false recall or truncation, followed by
exactly 32 fitting updates. Contextual/wrong/shuffled/zero memory emitted immediate
EOS on all eight fitting worlds. Selected exact text scored 8/8; absent memory
answered unknown and scored 0/8. The memory-dependent fitting gate failed, so
development stayed unopened. A separately budgeted numerical restart reproduced
exact logits and performed one actual continuation update.

V7 changes only the fixed training exposure and diagnostic schedule. It resets
to the same initialization and learning rate, uses the same v6 source/data,
and executes 128 updates with all fitting probes at 32/64/96/128. Every prefix
runs, with no early stopping, best-prefix selection or interim development
access. Only the fixed final prefix may open development after fitting freezes.
The [v7 runbook](../../reports/intrinsic_memory_exposure_v7/README.md) and
[protocol](../../configs/spark_intrinsic_exposure_curve_v7.json) declare the new
execution identity and work budget. Its
[completed summary](../../reports/intrinsic_memory_exposure_v7/native_attempt01_summary.json)
records the following fixed curve:

| Fitting updates | Contextual | Exact text | Wrong | Shuffled | Zero | Absent | Counterfactual pairs both correct |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 0/8 | 8/8 | 0/8 | 0/8 | 0/8 | 0/8 | 0/4 |
| 64 | 8/8 | 8/8 | 0/8 | 8/8 | 0/8 | 0/8 | 4/4 |
| 96 | 8/8 | 8/8 | 0/8 | 8/8 | 0/8 | 0/8 | 4/4 |
| 128 | 8/8 | 8/8 | 0/8 | 8/8 | 0/8 | 0/8 | 4/4 |

The final fitting gate passed. Development opened only after all 128 updates and
the recorded freeze boundary. **Development contextual/exact/wrong/shuffled/zero/
absent scored 0/8, 8/8, 0/8, 0/8, 0/8 and 1/8**, with respective false-recall
counts 1, 0, 1, 1, 1 and 0. Useful recall failed. The fit-curve file's
`development_opened: false` describes its prefix-probe phase; the later development
receipt and final result record that development did open. This is memory-dependent
fitting, not generalization. The same final model remains insensitive to reversing
slots on the fitting task; the stricter utility gate is unchanged.

The chronological correction rejected the old snapshot, and revocation rejected
the now-ineligible retrieval. The corrected contextual answer was **red**, while
the corrected exact-text answer was **gold** and correct. After revocation, absent
memory answered unknown correctly. These observations establish invalidation
mechanics and show failed semantic correction; they do not establish learned
unlearning, replay correction or retention. The fresh process reproduced exact
logits and performed one actual update. Both main and restart full original-weight
hashes remained unchanged.

The separately bound [supported-value v8 control](../../reports/intrinsic_memory_supported_v8/README.md)
addresses training exposure without changing the mechanism. Fitting now includes
red, blue, green, gold and unsupported queries; development uses fresh entity names
with that supported vocabulary. This changes two data-exposure factors, so it is
not an isolated causal test of vocabulary coverage alone. Its unchanged v7 runner,
128+1 schedule, four fixed probes and all competence/fitting/utility criteria are
bound at `5a232e43`. The nine corpus/protocol tests pass. The
[completed v8 summary](../../reports/intrinsic_memory_supported_v8/native_attempt01_summary.json)
records 8/8 competence, then **0/8 contextual fitting at every 32/64/96/128 prefix**.
Exact text remains 8/8, and absent memory scores 2/8 on the two unsupported-query
fitting worlds. Development remains unopened. The fresh-process continuation and
full frozen-backbone hashes pass. Broader answer exposure did not solve fitting
under this fixed schedule; the experiment does not establish which data factor
caused the difference from v7.

The [v9 smaller-step control](../../reports/intrinsic_memory_step_v9/README.md),
committed at `b3ca64ac`, changes only learning rate from 0.001 to 0.0001, plus
required execution/output identities. It resets to the same initialization and
retains the exact v8 data, unchanged v7 native engine, 128+1 schedule and all gates.
Admission requires the completed failed-fit v8 receipt and rejects a v8 development
receipt or changed comparison contracts. Earlier large residuals and fitting
collapse motivate the hypothesis; they do not prove a learning-rate cause. Sixteen
CPU checks and full preflight pass. The
[completed v9 summary](../../reports/intrinsic_memory_step_v9/native_attempt01_summary.json)
records 8/8 competence and this fixed fitting curve:

| Fitting updates | Contextual | Exact text | Wrong | Shuffled | Zero | Absent | Counterfactual pairs both correct |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 3/8 | 8/8 | 3/8 | 3/8 | 2/8 | 2/8 | 0/4 |
| 64 | 6/8 | 8/8 | 2/8 | 6/8 | 2/8 | 2/8 | 2/4 |
| 96 | 7/8 | 8/8 | 1/8 | 7/8 | 2/8 | 2/8 | 3/4 |
| 128 | 6/8 | 8/8 | 2/8 | 6/8 | 2/8 | 2/8 | 2/4 |

The predeclared final-prefix gate fails at **6/8**. The better 96-update prefix is
not selected, and development remains unopened. At the final prefix, the Kela
red/blue counterfactuals both generate red, and the Domi green/gold counterfactuals
both generate gold. The two supported/unsupported pairs are correct. All eight
contextual responses terminate; the remaining errors are semantic first-token
errors. In the first recorded world, layer-3 hidden/residual RMS is
0.082680/0.027192 and layer-7 is 0.134828/0.047233. Smaller steps improve fitting
over v8 on this fixed seed/data but do not qualify readout. Fresh-process exact
logits, one actual update and full original-weight integrity pass again.

## What the engineering checks establish

All five attempts validate the qualified `.venv-spark-bf16` environment,
official source bytes, configuration/model executable hashes, local dependency
closure, data and output identities before loading weights. V6–v9 use
NativeFusedAdamW with one owner for each intended producer/reader parameter.
Original BF16 parameters are frozen; extension parameters and stored value slots
are FP32. Full original-parameter content hashes are checked, rather than
inferring unchanged weights from optimizer ownership alone.

The v6 probes establish finite producer and both reader gradients after the
zero-initialized output projections start learning, eight exact live/durable
slot-and-logit comparisons, and 48 exact fitting/generation prefix comparisons.
They do not establish useful memory. The first update's nonzero output-projection
gradients and later producer gradients are expected consequences of zero
initialization. The early-layer residual is large relative to its hidden state;
the measured scales are retained for diagnosis rather than silently adjusted.

The implementation retains block-input queries and post-block residuals. It does
not change pretrained attention, positions, gated attention, GELU MLP or the tied
output path. One shared snapshot serves active readers; reading more layers does
not create additional retrievals or writes. Pre-MLP fusion remains a separate
unrun experiment. The external spine remains responsible for specialist models,
sensors and actors.

## Costs and measurement boundaries

Each native attempt has one exclusive worker, an outer 7,200-second process-group
supervisor and a common durable ledger for admission, main work and fresh-process
continuation. Work is reserved before dispatch, including failed work. The
sampled guard remains 50 GiB driver memory, 512 MiB sustained swap growth and at
least 15% host free. Driver and RSS overlap and must not be summed; sampled peaks
can miss transients. No P update, replay teaching, capacity expansion or promotion
occurs.

| Attempt | Outer seconds | Peak driver bytes | Peak RSS bytes | Swap growth | Minimum host free | Output bytes |
|---|---:|---:|---:|---:|---:|---:|
| v5 | 38.548710 | 9,326,706,688 | 8,504,066,048 | 0 | 73% | 80,801 |
| v6 | 146.688949 | 11,261,542,400 | 8,501,231,616 | 0 | 70% | 45,960,171 |
| v7 | 676.494547 | 11,257,364,480 | 8,681,537,536 | 0 | 70% | 49,880,283 |
| v8 | 457.874750 | 11,261,542,400 | 8,629,993,472 | 0 | 70% | 47,584,163 |
| v9 | 508.269728 | 11,261,542,400 | 8,504,541,184 | 0 | 70% | 47,700,126 |

V5 used 20 forwards, 20 generated tokens and one model load, with no backward or
update. V6 used 188 forwards, 33 backwards/updates, 87 generated tokens, 48,128
padded input positions, two sequential loads, 68 source encodes, 17 retrievals
and 35 write transactions. Its audit ledger serialized 138,590,092 bytes through
repeated atomic writes; this is not physical storage-device write amplification.
V5 rehashed 16,477,992,494 source bytes inside the attempt; v6 rehashed
24,716,988,741. Separate review preflights are additional work outside the
native supervisor wall. Raw ledgers retain synchronized operation time, dispatch
time, source/provenance/hash costs, fallback text cost and artifact growth.

V7 actually used 1,058 forwards, 129 backwards/updates, 669 generated tokens,
270,848 padded input positions, two sequential loads, 245 source encodes,
51 retrievals and 103 write transactions. It rehashed 24,716,988,741 source bytes
inside the attempt. Its audit ledger serialized 3,432,658,720 bytes through
repeated rewrites. The growing audit overhead is measured; optimization of that
ledger would require a separate execution version and crash/recovery checks.

V8 used 710 forwards, 129 backwards/updates, 321 generated tokens, 181,760 padded
input positions, two loads, 236 source encodes, 41 retrievals and 83 write
transactions. Its audit ledger serialized 1,705,189,653 bytes and the attempt
rehashed 24,716,988,741 source bytes. Its lower work than v7 reflects failed
fitting with development/correction closed; this is not a reasoning-efficiency
improvement at matched answer quality.

V9 used 796 forwards, 129 backwards/updates, 407 generated tokens, 203,776 padded
input positions, two loads, 236 source encodes, 41 retrievals and 83 write
transactions. Audit serialization was 2,020,429,322 bytes; source-file rehashing
was 24,716,988,741 bytes. These source-file counts exclude the separate in-memory
original-parameter integrity hashes, whose work is included in supervisor wall
time and placement receipts. Neither byte count measures physical bus traffic.

V7 independently budgets 2,397 forwards, 129 backwards/updates, 613,632 padded
input positions, 2,008 generated tokens, two loads, 245 source encodes,
51 retrievals and 103 write transactions. Its 7,080-second stage allocation is
inside the 7,200-second outer wall. Two-site artifacts are capped at 160 MiB;
9/36-site packets have separate 416/1,056 MiB caps and cannot execute without
qualified useful two-site evidence under the same protocol.

## Verification and the next evidence gate

New v6 corpus and existing execution tests passed 28/28 in 17.40 seconds. Initial
v7 curve/corpus checks passed 18/18 in 156.66 seconds, followed by eight additional
source/authorization/failed-baseline checks in 3.40 seconds. The full CPU curve
executes 128 real producer/reader updates and four release rebuilds; only baseline
and final gate decisions are fixtures, so it makes no native competence claim.
V8's additional nine source-oracle, split, counterfactual and unchanged-budget/gate
tests pass in 0.05 seconds. Environment-only checks and complete source preflight
pass before the v8 launch; no native weight is loaded by either check.
V9 adds 16 checks in 3.14 seconds: fixed-rate/limit validation, failed-predecessor
and unopened-label enforcement, missing/changed/substituted executable refusal,
and actual smaller-rate producer/reader updates with optimizer ownership and reload.
The first current clean-checkout run failed during collection because the environment
test imported its launcher through an ambient `scripts/` path. The test now loads
the repository-relative file explicitly; the production launcher is unchanged.
Its 21 standalone tests pass with `PYTHONPATH` removed. The second clean run
passed 443/444 tests; a placement restart test had the same ambient-path assumption
in its child process. Its fixture now selects the repository's script directory
explicitly, and the child runs from an unrelated temporary directory without
`PYTHONPATH`. All 25 standalone placement tests pass. Both failed receipts remain
preserved; these are test portability repairs, with production execution unchanged.

The final clean checkout at `8ad52904` passes **444/444 tests, zero skips**, in
222.114 seconds (222.745361 seconds including wrapper overhead), with child peak
RSS 816,447,488 bytes. It has no checkpoint weights, no ambient `PYTHONPATH` and
no protected runtime module. The
[clean regression receipt](../../reports/intrinsic_memory_native_20261029/clean_regression03_summary.json)
names every test in this scope; it is not a claim to have rerun the entire repository.

The new [integrated CPU workflow](../../reports/intrinsic_memory_native_20261029/clean_cpu_integrated/result.json)
passes for 2/9/36 readers in 13.378906 seconds, with parent peak RSS 598,392,832 bytes
and 105,170,044 bytes across the three arm directories. Every layout has a live
producer gradient and nonzero gradients at all active readers, exact live/durable
logits, one 65,536-byte snapshot copy per forward and no additional retrieval or
write caused by reader count. Each fresh child restores exact logits and one
optimizer owner, then performs one actual update with the original backbone
unchanged. This is a tiny random BF16 fixture with zero native model loads;
uninterrupted-update equivalence and native reader-density quality are not measured.

The earlier clean 393-test pass, 2/9/36 integrated CPU workflow and unchanged
Spark-code tiny random BF16 probe remain recorded engineering evidence in the
[v5 verification summary](../../reports/intrinsic_memory_successor_v5/verification_summary.json).
They are not newly executed native capability tests. The
[current evidence summary](../../reports/intrinsic_memory_native_20261029/verification_summary.json)
separates those historical receipts from this increment's tests and native results.

Useful recall is required before reader-density execution, completed-interaction
formation and source-free P learning. Fitting success alone cannot open those
gates. Preserve the eleven P controls, independent future queries, chronological
corrections, revocations, interference, delayed retention and rollback in later
learning experiments. Keep P1 until a measured capacity bottleneck warrants more.

For Replica, the [formation and Replica successor contract](20261029_Spark_Memory_Formation_and_Replica_Successors.md)
connects useful memory to the existing bounded cognitive core, exact task journal
and approved identity. Preserve E/S/P/phi/C/I/K ownership and keep identity outside
task reward. Hot/warm/cold are storage placements, independent of learning clocks;
active identity/task budgets need measured residency and archival history needs
explicit recall. Neither paging nor reader density is nested learning. Persistent
learning, Replica fidelity, nested adaptation, RSI and native multimodal work
remain unqualified.

The next decisive research step is a separately versioned full-context-teacher /
memory-student alignment comparison with a direct-CE reference. Before native
execution it still needs an implemented differentiable objective, explicit
response-position alignment and detach boundaries, verified teacher outputs,
teacher-forward/token/cache costs, CPU gradient/reload checks and a complete
execution packet. The user has authorized bounded testing; scientific preparation
is the remaining requirement, not another approval question. No such teacher
experiment, useful density comparison or persistent-P learning is claimed here.

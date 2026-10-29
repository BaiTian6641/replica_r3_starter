# Spark intrinsic memory: placement-corrected native result

**29 October 2026 · S0 passes; S1 useful recall fails. No promotion or P learning.**

The exact user-authorized [placement packet](../../reports/intrinsic_memory_v3/native_experiment_s0_s1_placement_v3.json)
(SHA-256 `3f70b564ef6c743760b15a578fbeae7248997f5dacaa3e353190d26544c612e7`)
completed under the existing qualified `.venv-spark-bf16` environment. This is the
existing **FP8-derived BF16 artifact**, with Spark-X2.6-4B-FP8 source lineage;
it is neither official FP8 execution nor official BF16. The first two attempts
were execution failures before any answer. Attempt03 is the first completed native
competence/readout measurement. All three single-attempt approvals are consumed.

The [portable result](../../reports/intrinsic_memory_v3/native_readout03/result.json)
binds the authorization, raw reports, trajectories, profiles and guard output.
An [independent audit](../../reports/intrinsic_memory_v3/native_readout03/independent_audit.json)
rescored all 40 development responses and reconciled work/resource totals from
raw evidence; it did not rerun the model.
The sealed 53-file original execution and 55-file placement execution still match
their declared hashes. Implementation `260dc15e` was admitted at launch HEAD
`c2d5f0cf`; independent CPU commits made during the run were not imported by it.
Environment admission checked 32 package versions and 12 selected runtime-file
hashes, followed by full source, prepared-data, implementation and output checks.
No dependency or sealed Spark source was patched, and no new weights were downloaded.

## Measured S0 and S1

With extensions disabled and original weights frozen, all eight complete-context
calibration worlds passed: **8/8 correct, zero false recall and zero truncation**.
This took 3.742959 seconds and generated 20 tokens.
It establishes competence only on this declared calibration fixture.

The fixed 32-update producer/readout fit completed before development labels opened.
One NativeFusedAdamW owned the intended parameters. Loss fell from
8.515210 to 1.680115; all recorded losses/gradients
were finite. Zero-initialized output projections caused the expected initial zero
producer gradient, followed by nonzero producer gradients on all 31 later updates.
Reader gradients were nonzero on all 32 updates. Lower fit loss did not establish
useful recall on the separate development worlds.

| Development arm | Correct | Protocol false recall | Generated tokens | Generation seconds |
|---|---:|---:|---:|---:|
| absent | 1/8 | 0 | 16 | 38.663 |
| contextual | 0/8 | 1 | 256 | 638.774 |
| selected_exact | 8/8 | 0 | 20 | 48.506 |
| shuffled | 0/8 | 1 | 256 | 634.242 |
| wrong | 0/8 | 1 | 256 | 633.967 |

All five arms used the same eight development worlds and fixed selected evidence.
The contextual arm fails the predeclared requirement of at least 6/8, zero false
recall, within one correct answer of selected exact text, and better than every
negative control. Its long repeated `unknown` responses remain scored as emitted;
no post-hoc normalization, extra fitting or threshold change was applied. The
protocol's false-recall count is an incorrect whole response on an expected
`unknown` case, including malformed repeated abstentions; it is not a count of
all incorrect factual answers. Sealed-final data stayed closed.

Fresh-process extension/optimizer reload reproduced **exact logits** and completed
**one actual continuation update**. All declared original-weight ownership/version
checks passed. This verifies numerical execution and continuation, not learning
utility. There were **33 updates total and zero P updates**. No native
comparison of continued weights against an uninterrupted parent update was made;
that stronger optimizer-continuation comparison is existing CPU v4 evidence.

## Actual scope and costs

The fixed control used readers at layers **3 and 7**, block-input queries and a
post-block residual. Each world realized **one admitted record/span and four ordered
width-256 slots**. Two records/eight slots are the maximum, not actual occupancy;
no padding slots were added. The external spine was absent from intrinsic recall
and retains its separate specialist-model and sensor/actor responsibilities.

The training graph has no explicit query detach. Site 3 begins from frozen original
features; site 7 can receive the earlier memory residual through later frozen
blocks. Durable serialization and inference values detach. The original attention,
positions, gated attention, GELU MLP and tied output path were preserved.

| Resource | Observed |
|---|---:|
| Guard wall time | 2246.087616 s |
| Runner wall, including load and restart | 2236.954109 s |
| Peak sampled MPS driver | 39,990,558,720 bytes (37.244 GiB) |
| Peak sampled MPS allocation | 32,739,055,104 bytes |
| Largest process lifetime RSS | 8,767,913,984 bytes |
| Peak sampled swap growth | 0 bytes |
| Minimum guard host free | 31.0% |
| Native output files after completion | 48,141,905 bytes |
| Model / tokenizer loads | 2 / 2 |
| Forwards / backwards / optimizer steps | 859 / 33 / 33 |
| Generation requests / generated tokens | 48 / 824 |
| Processed text positions | 1,731,672 |

The sampled 50 GiB driver, 512 MiB sustained swap-growth and 15% host-free rules
were respected. Samples cannot exclude unsampled transients. RSS and MPS driver
memory overlap on this unified-memory machine and must not be added; host free and
swap are system-wide. Peaks are whole-attempt measurements, not phase attribution.
Small independent CPU tests ran concurrently, so this is not an isolated performance
benchmark or a density comparison. Only one native worker ran.

The sealed runner/guard lacked an overall deadline covering admission and all
children. An external PID/start-identity watchdog supervised this attempt against
the original 7,200-second deadline; it exited naturally without signaling. The
new monotonic process-group supervisor below is for a future execution identity,
not retroactively part of this sealed run. Future launchers must bind a complete
outer deadline before admission. Raw tokenizer/fork warnings are preserved.

All observed work fits the narrower active-stage packet. However, the runtime
ledger retains broader whole-protocol ceilings for forwards, updates and tokens;
only readout-step/case counters directly use 33/40. The sealed fixed loops constrain
this concrete path, but the launcher does not generically derive or pre-reserve
the packet's narrower 1,571/33/1,536 ceilings. Future admission must bind and enforce
those active-stage limits before work; a post-hoc within-budget audit is not a
claim of that missing enforcement.

The native report records parent producer dispatch time/bytes, retrieval and
fallback costs, and merged parent/restart work. Durable-write bytes/time and restart
producer costs are not separately returned; producer dispatch time is not synchronized
kernel time. There is no separate exact-text prefill timer. S1 left-pads prompts to
2,048 positions and regenerates the entire growing prefix without caching, while
fit right-pads. These costs include padding/repeated prefill. No reasoning-cost
reduction follows from this result.

Placement hashed all 8,224,158,720 unique original tensor bytes before and after
CPU-to-MPS movement with identical content digests. Post-fit/restart integrity uses
frozen ownership, object/version, shape, dtype, device, pointer and configuration
stamps. It did **not** perform another full content hash after training. Saved model
artifacts contain extension weights and optimizer state only, retained under
`runs/intrinsic_memory_v3/native_readout03`; portable reports omit those binaries.

The correction/revocation development cases were statements in admitted source
text, not native chronological store mutations. Dynamic invalidation, cache
rejection and hot/warm/cold validity remain separate CPU engineering evidence.

## Independent CPU work completed

Clean checkout `c014dea2` passes **67 tests in 5.36 seconds**, zero failures/skips.
The [CPU receipt](../../reports/intrinsic_memory_v3/native_readout03/cpu_successor_verification.json)
separates these new checks from earlier results:

- `wall_deadline_v1.py` and `run_wall_bounded_command_v1.py` provide a monotonic
  outer budget, bounded signal escalation/reaping, raw output and a receipt for an
  owned process group. They do not themselves grant or validate native authorization.
  Escaped process groups are outside that scope and cleanup uncertainty fails closed.
- `contextual_completion_objective_v1.py` and its separately versioned protocol add
  exactly one explicit EOS training target, retain ignored prompt/PAD targets and
  right padding, and bind tokenizer/template/thinking/terminal/decoder-stop identities.
  CPU tests distinguish correct answer words from incorrect completion and preserve
  live producer/reader gradients with one NativeFusedAdamW. Decoder stop inventory
  may include PAD, but PAD is never a training target.

The sealed historical objective has no EOS target. That is an objective-coverage
gap, **not a proven explanation of wrong first words or failed recall**. The new
objective is unused by attempt03. Its metadata is ready; future checkpoint-loader
refusal, native runner integration, complete execution bindings and measured native
utility remain open. It does not silently modify v3 or v4 checkpoint identities.

A separate manifest-verified random-source CPU
[padding probe](../../reports/intrinsic_memory_v3/native_readout03/padding_probe.json)
checks 16 aligned comparisons (two seeds, FP32/BF16, left/right, lengths 32/2,048).
All are finite with identical real-token argmaxes. FP32 maximum difference is
1.1921e-7; BF16 left-padding differences reach 0.00390625, while right-padding
comparisons are exact. This 8.778-second probe used no checkpoint weights, tokenizer,
trained reader or development data and peaked at 468,287,488 bytes RSS. It does not
show a padding-mask defect or establish native padding equivalence.

Previously recorded, **not rerun in this increment**: clean `a4461059` passed
268 affected tests in 37.26 seconds. The combined 2/9/36 producer/store/reader CPU
workflow passed in 13.329595 seconds, including shared snapshots, active branch
skipping, early-reader gradients through frozen blocks, original-weight integrity,
correction/revocation, complete checkpoints, exact fresh logits and actual optimizer
continuation. Its 104,937,953 artifact bytes and tiny-fixture timings are not native
all-layer capacity or capability evidence. The earlier full-suite 3,371-pass receipt
retains its own snapshot and scope.

## Next decisive experiment and Replica boundary

The useful-readout gate is closed. Do not launch P learning, learned writers,
formation or automatic promotion from this failed S1. Preserve the one-P1 capacity
and the previous working version. The 2/9/36 feasibility protocol remains prepared;
its dedicated native runner, numerical/resource qualification and exact per-arm
receipts are absent. In addition, its declared eight-slot masked-padding contract
must be implemented/tested or replaced by an explicitly versioned realized-slot
contract before admission. Reader parameter estimates alone do not establish fit.

The next executable independent check is the 67-test CPU command below. The next
decisive **native** experiment needs a new reviewable packet: integrate the completion
objective and its checkpoint identity into a separate two-site runner, bind a full
outer deadline and pre-reserved active-stage budgets, close the write/restart
cost-accounting gaps, and verify training
versus generation parity using fitting-only evidence. Freeze all choices before
fresh development worlds are opened. Re-run the unchanged competent-baseline gate,
then compare contextual values against matched exact text and the three negatives.
Do not reuse the observed development set to choose updates or thresholds. The EOS
candidate alone does not justify expecting semantic recall to improve. No successor
native command or approval is claimed ready here; exact authorization follows a
complete packet, budgets and independent review.

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:tests:scripts .venv/bin/python -m pytest tests/test_wall_deadline_v1.py tests/test_contextual_completion_objective_v1.py -q
```

The [formation and Replica successor specification](20261029_Spark_Memory_Formation_and_Replica_Successors.md)
keeps the semi-cognitive core, TaskRuntime journal and approved identity separate
from task reward. Completed-interaction writes precede within-reasoning chunk
commits, with sequence/position/revision/retry/recomputation identity. Distinct future
queries, source removal, restart, interference, delayed retention, correction and
rollback precede persistent-learning claims. Hot/warm/cold residency is independent
of learning timescales; stale or incomplete recall must be detectable across moves.
Approved identity has a bounded active core and explicit archival recall. Neither
reader density nor paging is nested learning, and RSI remains unestablished.
